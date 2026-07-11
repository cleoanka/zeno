//! Metal GPU backend (f32 state vector in unified memory).
//!
//! Architecture:
//! - The state lives **permanently** in two `StorageModeShared` MTLBuffers
//!   (split re/im, mirroring [`crate::state`]'s SoA layout). Gates are GPU
//!   compute dispatches; reductions (measure, sampling, norm) run on the
//!   CPU over the very same memory — free on Apple Silicon unified memory.
//! - Dispatches are **batched**: encoded eagerly (one compute encoder per
//!   dispatch; default hazard tracking serializes accesses to the state
//!   buffers), committed lazily. `waitUntilCompleted` happens only when a
//!   read needs the data (measure, sample, statevector, norm, reset), and
//!   a command buffer rolls over after 256 dispatches.
//! - Kernels are MSL compiled at runtime from an embedded source string:
//!   one fused k-qubit unitary kernel per k in 1..=6 (generated from a
//!   template so the inner loops unroll) plus one diagonal kernel (k ≤ 12,
//!   the compiler's diagonal-fusion cap).
//! - Fast math is disabled and FP contraction is off; the kernels use the
//!   exact operation order of the CPU f32 kernels, so amplitudes match the
//!   CPU backend bit for bit (modulo the sign of zeros) and sampling with
//!   the same seed yields identical counts.
//!
//! Contract:
//! - `MetalBackend::new(n_qubits) -> Result<Self, Error>`
//! - implements [`crate::exec::Backend`]
//! - f32 only (Metal has no f64); the executor enforces precision.
//! - buffers are `StorageModeShared` so reductions/sampling can run on the
//!   CPU over the same memory without copies (Apple Silicon unified memory).

use crate::exec::Backend;
use crate::ir::C64;
use crate::sample::sample_indices;
use crate::state::StateVec;
use crate::Error;
use metal::{
    Buffer, CommandBuffer, CommandQueue, ComputePipelineState, Device, MTLResourceOptions, MTLSize,
};
use objc::rc::autoreleasepool;
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::sync::OnceLock;

/// Largest fused-unitary width (must match [`crate::state::MAX_FUSED_QUBITS`]).
const MAX_K: usize = crate::state::MAX_FUSED_QUBITS;
/// Largest diagonal-table width the compiler can emit (`diag_max` cap).
const MAX_DIAG_K: usize = 12;
/// Roll a command buffer over after this many encoded dispatches.
const MAX_DISPATCHES_PER_CB: usize = 256;
/// Below this many items the CPU-side reductions run single-threaded
/// (same threshold as `crate::state`).
const PAR_MIN: usize = 1 << 13;

// ---------------------------------------------------------------------------
// Embedded MSL
// ---------------------------------------------------------------------------

/// Common header: params structs + the diagonal kernel.
///
/// The arithmetic is written as separate statements on purpose: together
/// with fast-math off and `FP_CONTRACT OFF` this forbids fma contraction,
/// so every f32 operation rounds exactly like the CPU kernels.
const MSL_HEADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

#pragma STDC FP_CONTRACT OFF

struct UParams {
    ulong offs[64];
    uint  qs[6];
    uint  k;
    uint  pad;
};

struct DParams {
    uint qs[12];
    uint k;
    uint pad;
};

// One thread per amplitude: multiply by the diagonal-table entry selected
// by the amplitude's qubit bits. Table layout: [re[dim] | im[dim]].
kernel void diag_mul(
    device float*       re  [[buffer(0)]],
    device float*       im  [[buffer(1)]],
    device const float* tab [[buffer(2)]],
    constant DParams&   p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    ulong i = ulong(gid);
    uint j = 0;
    for (uint b = 0; b < p.k; ++b) {
        j |= uint((i >> ulong(p.qs[b])) & ulong(1)) << b;
    }
    float wr = tab[j];
    float wi = tab[(1u << p.k) + j];
    float ar = re[i];
    float ai = im[i];
    float t0 = wr * ar;
    float t1 = wi * ai;
    float t2 = wr * ai;
    float t3 = wi * ar;
    re[i] = t0 - t1;
    im[i] = t2 + t3;
}
"#;

/// Fused unitary kernel template; `KVAL`/`DIMVAL` are substituted per k so
/// the loops fully unroll and the gather arrays are exactly sized.
const MSL_UNITARY_TEMPLATE: &str = r#"
// Fused KVAL-qubit unitary: each thread owns one group of DIMVAL
// amplitudes. Matrix layout: [re[dim*dim] | im[dim*dim]], row-major.
kernel void unitary_kKVAL(
    device float*       re  [[buffer(0)]],
    device float*       im  [[buffer(1)]],
    device const float* mat [[buffer(2)]],
    constant UParams&   p   [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint dim = DIMVAL;
    // insert_zeros: place zero bits at the (ascending) qubit positions,
    // mirroring crate::state::insert_zeros.
    ulong x = ulong(gid);
    for (uint b = 0; b < KVAL; ++b) {
        ulong q = ulong(p.qs[b]);
        ulong low = x & ((ulong(1) << q) - ulong(1));
        x = ((x >> q) << (q + ulong(1))) | low;
    }
    const ulong base = x;
    device const float* mre = mat;
    device const float* mim = mat + dim * dim;
    float xr[DIMVAL];
    float xi[DIMVAL];
    for (uint j = 0; j < dim; ++j) {
        ulong idx = base | p.offs[j];
        xr[j] = re[idx];
        xi[j] = im[idx];
    }
    for (uint r = 0; r < dim; ++r) {
        uint row = r * dim;
        float ar = 0.0f;
        float ai = 0.0f;
        // Accumulation order identical to the CPU kernel (contraction off).
        for (uint c = 0; c < dim; ++c) {
            float wr = mre[row + c];
            float wi = mim[row + c];
            float wxr = wr * xr[c];
            float wxi = wi * xi[c];
            ar = ar + wxr;
            ar = ar - wxi;
            float wyr = wr * xi[c];
            float wyi = wi * xr[c];
            ai = ai + wyr;
            ai = ai + wyi;
        }
        ulong idx = base | p.offs[r];
        re[idx] = ar;
        im[idx] = ai;
    }
}
"#;

fn msl_source() -> String {
    let mut s = String::with_capacity(8 * 1024);
    s.push_str(MSL_HEADER);
    for k in 1..=MAX_K {
        s.push_str(
            &MSL_UNITARY_TEMPLATE
                .replace("DIMVAL", &(1usize << k).to_string())
                .replace("KVAL", &k.to_string()),
        );
    }
    s
}

// ---------------------------------------------------------------------------
// Kernel parameter blocks (layouts mirror the MSL structs above)
// ---------------------------------------------------------------------------

/// Read by the GPU only (passed via `setBytes`), hence the dead-code allow.
#[allow(dead_code)]
#[repr(C)]
struct UParams {
    offs: [u64; 64],
    qs: [u32; MAX_K],
    k: u32,
    _pad: u32,
}

/// Read by the GPU only (passed via `setBytes`), hence the dead-code allow.
#[allow(dead_code)]
#[repr(C)]
struct DParams {
    qs: [u32; MAX_DIAG_K],
    k: u32,
    _pad: u32,
}

// ---------------------------------------------------------------------------
// Device + pipeline cache (MSL is compiled once per process)
// ---------------------------------------------------------------------------

struct Pipelines {
    device: Device,
    unitary: [ComputePipelineState; MAX_K],
    diag: ComputePipelineState,
}

fn build_pipelines() -> Result<Pipelines, String> {
    let device = Device::system_default()
        .ok_or_else(|| "no Metal device found (MTLCreateSystemDefaultDevice)".to_string())?;
    let opts = metal::CompileOptions::new();
    opts.set_fast_math_enabled(false);
    let lib = device
        .new_library_with_source(&msl_source(), &opts)
        .map_err(|e| format!("MSL compilation failed: {e}"))?;
    let make = |name: &str| -> Result<ComputePipelineState, String> {
        let f = lib
            .get_function(name, None)
            .map_err(|e| format!("kernel '{name}': {e}"))?;
        device
            .new_compute_pipeline_state_with_function(&f)
            .map_err(|e| format!("pipeline '{name}': {e}"))
    };
    let unitary = [
        make("unitary_k1")?,
        make("unitary_k2")?,
        make("unitary_k3")?,
        make("unitary_k4")?,
        make("unitary_k5")?,
        make("unitary_k6")?,
    ];
    let diag = make("diag_mul")?;
    Ok(Pipelines {
        device,
        unitary,
        diag,
    })
}

fn pipelines() -> Result<&'static Pipelines, Error> {
    static PIPES: OnceLock<Result<Pipelines, String>> = OnceLock::new();
    PIPES
        .get_or_init(build_pipelines)
        .as_ref()
        .map_err(|e| Error::Unsupported(format!("Metal backend unavailable: {e}")))
}

// ---------------------------------------------------------------------------
// CPU-side reductions over the shared buffers
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}
impl<T> SendPtr<T> {
    #[inline(always)]
    fn ptr(&self) -> *mut T {
        self.0
    }
}

/// Split `0..total` into parallel task ranges (same policy as
/// `crate::state`'s private `task_ranges`).
fn ranges(total: usize) -> Vec<(usize, usize)> {
    if total == 0 {
        return vec![];
    }
    let threads = rayon::current_num_threads().max(1);
    let ntasks = if total <= PAR_MIN {
        1
    } else {
        (total / PAR_MIN).min(threads * 8).max(1)
    };
    let per = total.div_ceil(ntasks);
    (0..ntasks)
        .map(|t| (t * per, ((t + 1) * per).min(total)))
        .filter(|(a, b)| a < b)
        .collect()
}

/// Probability that qubit `q` measures 1 (f64 accumulation), same math as
/// [`crate::state::prob_one`] but over raw f32 slices.
fn prob_one_slices(re: &[f32], im: &[f32], q: u32) -> f64 {
    let half = re.len() >> 1;
    ranges(half)
        .into_par_iter()
        .map(|(h0, h1)| {
            let s = 1usize << q;
            let mask = s - 1;
            let mut acc = 0.0f64;
            let mut h = h0;
            while h < h1 {
                let run = (s - (h & mask)).min(h1 - h);
                let a0 = ((h >> q) << (q + 1)) | (h & mask);
                for t in 0..run {
                    let (r, i) = (f64::from(re[a0 + s + t]), f64::from(im[a0 + s + t]));
                    acc += r * r + i * i;
                }
                h += run;
            }
            acc
        })
        .sum()
}

/// Collapse qubit `q` to `outcome` (observed with probability `prob`) and
/// renormalize, same math as [`crate::state::collapse`].
fn collapse_slices(re: &mut [f32], im: &mut [f32], q: u32, outcome: bool, prob: f64) {
    let scale = (1.0 / prob.sqrt()) as f32;
    let half = re.len() >> 1;
    let rp = SendPtr(re.as_mut_ptr());
    let ip = SendPtr(im.as_mut_ptr());
    ranges(half).into_par_iter().for_each(|(h0, h1)| unsafe {
        let re = rp.ptr();
        let im = ip.ptr();
        let s = 1usize << q;
        let mask = s - 1;
        let mut h = h0;
        while h < h1 {
            let run = (s - (h & mask)).min(h1 - h);
            let a0 = ((h >> q) << (q + 1)) | (h & mask);
            let (keep, zero) = if outcome { (a0 + s, a0) } else { (a0, a0 + s) };
            for t in 0..run {
                *re.add(keep + t) = *re.add(keep + t) * scale;
                *im.add(keep + t) = *im.add(keep + t) * scale;
                *re.add(zero + t) = 0.0;
                *im.add(zero + t) = 0.0;
            }
            h += run;
        }
    });
}

// ---------------------------------------------------------------------------
// The backend
// ---------------------------------------------------------------------------

pub struct MetalBackend {
    n_qubits: u32,
    pipes: &'static Pipelines,
    queue: CommandQueue,
    re: Buffer,
    im: Buffer,
    /// Command buffer currently accepting encodes (not yet committed).
    open: Option<CommandBuffer>,
    /// Dispatches encoded into `open` so far.
    open_dispatches: usize,
    /// Committed but not yet waited-on command buffers, oldest first.
    in_flight: Vec<CommandBuffer>,
    /// Matrix/table buffers referenced by not-yet-completed dispatches
    /// (kept alive until the next flush).
    gate_data: Vec<Buffer>,
}

impl MetalBackend {
    pub fn new(n_qubits: u32) -> Result<Self, Error> {
        if n_qubits > 32 {
            return Err(Error::Unsupported(format!(
                "the Metal backend caps at 32 qubits (one GPU thread per \
                 amplitude in a 1-D grid); {n_qubits} requested — use \
                 --backend cpu"
            )));
        }
        let pipes = pipelines()?;
        let len = 1usize << n_qubits;
        let bytes = (len * std::mem::size_of::<f32>()) as u64;
        if bytes > pipes.device.max_buffer_length() {
            return Err(Error::Unsupported(format!(
                "state vector needs {} per component buffer but this GPU \
                 caps buffers at {}",
                crate::human_bytes(u128::from(bytes)),
                crate::human_bytes(u128::from(pipes.device.max_buffer_length())),
            )));
        }
        let queue = pipes.device.new_command_queue();
        let re = pipes
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let im = pipes
            .device
            .new_buffer(bytes, MTLResourceOptions::StorageModeShared);
        let mut be = MetalBackend {
            n_qubits,
            pipes,
            queue,
            re,
            im,
            open: None,
            open_dispatches: 0,
            in_flight: vec![],
            gate_data: vec![],
        };
        be.reset_all(); // zero-init + |0…0⟩ (nothing in flight yet)
        Ok(be)
    }

    /// The shared buffers as f32 slices. Callers must flush first.
    fn slices(&self) -> (&[f32], &[f32]) {
        let len = 1usize << self.n_qubits;
        // SAFETY: both buffers are StorageModeShared and exactly `len` f32
        // long; every caller flushes pending GPU work before reading.
        unsafe {
            (
                std::slice::from_raw_parts(self.re.contents() as *const f32, len),
                std::slice::from_raw_parts(self.im.contents() as *const f32, len),
            )
        }
    }

    /// Mutable view of the shared buffers. Callers must flush first.
    fn slices_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        let len = 1usize << self.n_qubits;
        // SAFETY: as in `slices`, plus `&mut self` guarantees exclusivity
        // on the CPU side and the flush guarantees the GPU is idle.
        unsafe {
            (
                std::slice::from_raw_parts_mut(self.re.contents() as *mut f32, len),
                std::slice::from_raw_parts_mut(self.im.contents() as *mut f32, len),
            )
        }
    }

    /// Encode one dispatch into the open command buffer (creating or
    /// rolling it over as needed). Does not commit unless the cap is hit.
    fn encode<P>(&mut self, pipe: &ComputePipelineState, data: Buffer, params: &P, grid: u64) {
        autoreleasepool(|| {
            if self.open.is_none() {
                self.open = Some(self.queue.new_command_buffer().to_owned());
                self.open_dispatches = 0;
            }
            let cb = self.open.as_ref().expect("open command buffer");
            let enc = cb.new_compute_command_encoder();
            enc.set_compute_pipeline_state(pipe);
            enc.set_buffer(0, Some(&self.re), 0);
            enc.set_buffer(1, Some(&self.im), 0);
            enc.set_buffer(2, Some(&data), 0);
            enc.set_bytes(
                3,
                std::mem::size_of::<P>() as u64,
                (params as *const P).cast(),
            );
            let w = pipe.thread_execution_width().max(1);
            enc.dispatch_threads(MTLSize::new(grid, 1, 1), MTLSize::new(w, 1, 1));
            enc.end_encoding();
        });
        self.gate_data.push(data);
        self.open_dispatches += 1;
        if self.open_dispatches >= MAX_DISPATCHES_PER_CB {
            let cb = self.open.take().expect("open command buffer");
            cb.commit();
            self.in_flight.push(cb);
            self.open_dispatches = 0;
        }
    }

    /// Commit pending work and wait for everything in flight to complete.
    /// After this the shared buffers are safe to read/write from the CPU.
    fn flush(&mut self) {
        if let Some(cb) = self.open.take() {
            cb.commit();
            self.in_flight.push(cb);
            self.open_dispatches = 0;
        }
        if !self.in_flight.is_empty() {
            autoreleasepool(|| {
                for cb in self.in_flight.drain(..) {
                    cb.wait_until_completed();
                }
            });
        }
        self.gate_data.clear();
    }

    fn new_gate_buffer(&self, data: &[f32]) -> Buffer {
        self.pipes.device.new_buffer_with_data(
            data.as_ptr().cast(),
            std::mem::size_of_val(data) as u64,
            MTLResourceOptions::StorageModeShared,
        )
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal-f32"
    }

    fn apply_unitary(&mut self, qs: &[u32], mat: &[C64]) {
        let k = qs.len();
        debug_assert!((1..=MAX_K).contains(&k));
        debug_assert!(k as u32 <= self.n_qubits);
        debug_assert!(qs.windows(2).all(|w| w[0] < w[1]));
        let dim = 1usize << k;
        debug_assert_eq!(mat.len(), dim * dim);
        // SoA f32 matrix: [re | im], same f64→f32 conversion as the CPU.
        let mut data = Vec::with_capacity(2 * dim * dim);
        data.extend(mat.iter().map(|z| z.re as f32));
        data.extend(mat.iter().map(|z| z.im as f32));
        let buf = self.new_gate_buffer(&data);
        let mut p = UParams {
            offs: [0; 64],
            qs: [0; MAX_K],
            k: k as u32,
            _pad: 0,
        };
        for (b, &q) in qs.iter().enumerate() {
            p.qs[b] = q;
        }
        for j in 0..dim {
            let mut o = 0u64;
            for (b, &q) in qs.iter().enumerate() {
                if (j >> b) & 1 == 1 {
                    o |= 1u64 << q;
                }
            }
            p.offs[j] = o;
        }
        let grid = 1u64 << (self.n_qubits - k as u32);
        let pipes = self.pipes;
        self.encode(&pipes.unitary[k - 1], buf, &p, grid);
    }

    fn apply_diagonal(&mut self, qs: &[u32], d: &[C64]) {
        let k = qs.len();
        debug_assert!((1..=MAX_DIAG_K).contains(&k));
        debug_assert!(k as u32 <= self.n_qubits);
        debug_assert!(qs.windows(2).all(|w| w[0] < w[1]));
        let dim = 1usize << k;
        debug_assert_eq!(d.len(), dim);
        let mut data = Vec::with_capacity(2 * dim);
        data.extend(d.iter().map(|z| z.re as f32));
        data.extend(d.iter().map(|z| z.im as f32));
        let buf = self.new_gate_buffer(&data);
        let mut p = DParams {
            qs: [0; MAX_DIAG_K],
            k: k as u32,
            _pad: 0,
        };
        for (b, &q) in qs.iter().enumerate() {
            p.qs[b] = q;
        }
        let grid = 1u64 << self.n_qubits;
        let pipes = self.pipes;
        self.encode(&pipes.diag, buf, &p, grid);
    }

    fn prob_one(&mut self, q: u32) -> f64 {
        self.flush();
        let (re, im) = self.slices();
        prob_one_slices(re, im, q)
    }

    fn measure(&mut self, q: u32, u: f64) -> bool {
        self.flush();
        let (re, im) = self.slices();
        let p1 = prob_one_slices(re, im, q);
        let outcome = u < p1;
        let p = if outcome { p1 } else { 1.0 - p1 };
        let (re, im) = self.slices_mut();
        collapse_slices(re, im, q, outcome, p);
        outcome
    }

    fn reset_all(&mut self) {
        self.flush();
        let (re, im) = self.slices_mut();
        re.par_iter_mut().for_each(|x| *x = 0.0);
        im.par_iter_mut().for_each(|x| *x = 0.0);
        re[0] = 1.0;
    }

    fn sample(&mut self, shots: usize, seed: u64) -> Vec<u64> {
        self.flush();
        let (re, im) = self.slices();
        // One copy into a StateVec so the exact chunked sampler of the CPU
        // backend runs here too: identical amplitudes → identical draws.
        let mut st = StateVec::<f32>::zero_state(self.n_qubits);
        st.re.copy_from_slice(re);
        st.im.copy_from_slice(im);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        sample_indices(&st, shots, &mut rng)
    }

    fn statevector(&mut self) -> Vec<C64> {
        self.flush();
        let (re, im) = self.slices();
        re.iter()
            .zip(im)
            .map(|(&r, &i)| C64::new(f64::from(r), f64::from(i)))
            .collect()
    }

    fn norm_sqr(&mut self) -> f64 {
        self.flush();
        let (re, im) = self.slices();
        re.par_chunks(1 << 16)
            .zip(im.par_chunks(1 << 16))
            .map(|(rs, is)| {
                let mut acc = 0.0f64;
                for (r, i) in rs.iter().zip(is) {
                    let (r, i) = (f64::from(*r), f64::from(*i));
                    acc += r * r + i * i;
                }
                acc
            })
            .sum()
    }

    fn finish(&mut self) {
        self.flush();
    }
}

// ---------------------------------------------------------------------------
// Parity tests against the CPU backend
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "metal"))]
mod tests {
    use super::*;
    use crate::circuit::{random_circuit, Circuit};
    use crate::compiler::{permute_diag_to_sorted, permute_unitary_to_sorted};
    use crate::exec::{CpuBackend, RunOptions};
    use crate::gates::{self, GateMatrix};
    use crate::ir::{GateInstr, Instr, Program, Reg};
    use crate::{run_program, BackendChoice, Precision};
    use rand::Rng;

    /// Apply a named native gate through the [`Backend`] interface,
    /// converting to the sorted-qubit convention the executor uses.
    fn apply_named(be: &mut dyn Backend, name: &str, qubits: &[u32], params: &[f64]) {
        match gates::build(name, params).expect("native gate") {
            GateMatrix::Unitary(m) => {
                let (qs, m) = permute_unitary_to_sorted(qubits, &m);
                be.apply_unitary(&qs, &m);
            }
            GateMatrix::Diagonal(d) => {
                let (qs, d) = permute_diag_to_sorted(qubits, &d);
                be.apply_diagonal(&qs, &d);
            }
        }
    }

    /// Drive both backends into the same (well-entangled) 5-qubit state.
    fn prep_random_5q(be: &mut dyn Backend, seed: u64) {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let tau = std::f64::consts::TAU;
        for _layer in 0..2 {
            for q in 0..5u32 {
                let p: Vec<f64> = (0..3).map(|_| rng.gen::<f64>() * tau).collect();
                apply_named(be, "u3", &[q], &p);
            }
            for q in 0..4u32 {
                apply_named(be, "cx", &[q, q + 1], &[]);
            }
        }
    }

    fn max_delta(a: &[C64], b: &[C64]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).norm())
            .fold(0.0, f64::max)
    }

    /// (a) Every native gate on a random 5-qubit state: Metal vs CPU f32.
    #[test]
    fn parity_all_native_gates_on_random_state() {
        let names: &[&str] = &[
            "id", "u0", "x", "y", "z", "h", "s", "sdg", "t", "tdg", "sx", "sxdg", "rx", "ry", "rz",
            "u1", "p", "u2", "u3", "u", "cx", "cy", "cz", "ch", "swap", "cp", "cu1", "crx", "cry",
            "crz", "rxx", "rzz", "cu3", "ccx", "cswap",
        ];
        for &name in names {
            let def = gates::lookup(name).expect("gate in native set");
            let params: Vec<f64> = (0..def.n_params)
                .map(|i| 0.3 + 0.71 * f64::from(i))
                .collect();
            // Unsorted argument orders on purpose (exercises permutation).
            let qubits: Vec<u32> = match def.arity {
                1 => vec![3],
                2 => vec![4, 1],
                _ => vec![4, 0, 2],
            };
            let mut gpu = MetalBackend::new(5).expect("metal device");
            let mut cpu = CpuBackend::<f32>::new(5);
            prep_random_5q(&mut gpu, 7);
            prep_random_5q(&mut cpu, 7);
            apply_named(&mut gpu, name, &qubits, &params);
            apply_named(&mut cpu, name, &qubits, &params);
            let d = max_delta(&gpu.statevector(), &cpu.statevector());
            assert!(d < 1e-6, "gate {name}: max |delta| = {d:e}");
        }
    }

    /// (b) Full Simulator pipeline on random circuits, fusion 0 and 5:
    /// identical counts with the same seed, statevectors within 1e-5.
    #[test]
    fn parity_random_circuits_full_pipeline() {
        for n in 3..=12u32 {
            for fusion in [0u8, 5] {
                let mut c = random_circuit(n, 6, 100 + u64::from(n));
                c.measure_all();
                let opts = |backend| RunOptions {
                    shots: 512,
                    seed: Some(9000 + u64::from(n)),
                    precision: Some(Precision::F32),
                    backend,
                    fusion_max: Some(fusion),
                    want_statevector: true,
                    ..Default::default()
                };
                let p = c.to_program();
                let g = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
                let h = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
                assert_eq!(g.backend, "metal-f32");
                assert_eq!(h.backend, "cpu-f32");
                let d = max_delta(
                    g.statevector.as_ref().unwrap(),
                    h.statevector.as_ref().unwrap(),
                );
                assert!(d < 1e-5, "n={n} fusion={fusion}: max |delta| = {d:e}");
                assert_eq!(g.counts, h.counts, "n={n} fusion={fusion}");
            }
        }
    }

    /// (b addendum) Cross the 256-dispatch command-buffer rollover: with
    /// fusion 0, random_circuit(11, 30) issues ~480 dispatches between
    /// reads, forcing at least one rollover — ordering across command
    /// buffers must not change the result.
    #[test]
    fn parity_across_command_buffer_rollover() {
        let mut c = random_circuit(11, 30, 4242);
        c.measure_all();
        let opts = |backend| RunOptions {
            shots: 512,
            seed: Some(4242),
            precision: Some(Precision::F32),
            backend,
            fusion_max: Some(0),
            want_statevector: true,
            ..Default::default()
        };
        let p = c.to_program();
        let g = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
        let h = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
        assert!(
            g.stats.output_ops > MAX_DISPATCHES_PER_CB,
            "test must cross the rollover boundary ({} ops)",
            g.stats.output_ops
        );
        let d = max_delta(
            g.statevector.as_ref().unwrap(),
            h.statevector.as_ref().unwrap(),
        );
        assert!(d < 1e-4, "rollover parity: max |delta| = {d:e}");
        assert_eq!(g.counts, h.counts, "rollover parity counts");
    }

    /// (b addendum) A QFT ladder on a rotated product state: diagonal
    /// fusion emits wide (up to 10-qubit) diagonal ops for the GPU.
    #[test]
    fn parity_qft_wide_diagonals() {
        let n = 12u32;
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let tau = std::f64::consts::TAU;
        let mut c = Circuit::new(n);
        for q in 0..n {
            c.u(
                q,
                rng.gen::<f64>() * tau,
                rng.gen::<f64>() * tau,
                rng.gen::<f64>() * tau,
            );
        }
        for i in (0..n).rev() {
            c.h(i);
            for j in (0..i).rev() {
                let angle = std::f64::consts::PI / f64::from(1u32 << (i - j));
                c.cp(j, i, angle);
            }
        }
        c.measure_all();
        let opts = |backend| RunOptions {
            shots: 1024,
            seed: Some(31),
            precision: Some(Precision::F32),
            backend,
            want_statevector: true,
            ..Default::default()
        };
        let p = c.to_program();
        let g = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
        let h = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
        let d = max_delta(
            g.statevector.as_ref().unwrap(),
            h.statevector.as_ref().unwrap(),
        );
        assert!(d < 1e-5, "qft: max |delta| = {d:e}");
        assert_eq!(g.counts, h.counts);
    }

    /// (c) GHZ(16) on Metal: exactly the two all-equal bitstrings.
    #[test]
    fn ghz16_two_keys() {
        let mut c = Circuit::new(16);
        c.h(0);
        for q in 0..15 {
            c.cx(q, q + 1);
        }
        c.measure_all();
        let r = run_program(
            &c.to_program(),
            &RunOptions {
                shots: 2048,
                seed: Some(3),
                backend: BackendChoice::Metal,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(r.backend, "metal-f32");
        assert_eq!(r.precision, Precision::F32);
        assert_eq!(r.counts.len(), 2, "counts: {:?}", r.counts);
        assert_eq!(
            r.counts.get(&"0".repeat(16)) + r.counts.get(&"1".repeat(16)),
            2048
        );
    }

    /// (d) Dynamic teleportation: mid-circuit measurement + classical
    /// control, Metal vs CPU with the same seed → identical counts.
    #[test]
    fn dynamic_teleport_parity() {
        let g = |name: &str, qubits: &[u32]| {
            Instr::Gate(GateInstr {
                name: name.into(),
                params: vec![],
                qubits: qubits.to_vec(),
            })
        };
        let p = Program {
            qregs: vec![Reg {
                name: "q".into(),
                size: 3,
            }],
            cregs: vec![
                Reg {
                    name: "m0".into(),
                    size: 1,
                },
                Reg {
                    name: "m1".into(),
                    size: 1,
                },
                Reg {
                    name: "out".into(),
                    size: 1,
                },
            ],
            instrs: vec![
                g("x", &[0]),
                g("h", &[1]),
                g("cx", &[1, 2]),
                g("cx", &[0, 1]),
                g("h", &[0]),
                Instr::Measure { qubit: 0, clbit: 0 },
                Instr::Measure { qubit: 1, clbit: 1 },
                Instr::If {
                    creg: 1,
                    value: 1,
                    op: Box::new(g("x", &[2])),
                },
                Instr::If {
                    creg: 0,
                    value: 1,
                    op: Box::new(g("z", &[2])),
                },
                Instr::Measure { qubit: 2, clbit: 2 },
            ],
        };
        let opts = |backend| RunOptions {
            shots: 256,
            seed: Some(5),
            precision: Some(Precision::F32),
            backend,
            ..Default::default()
        };
        let a = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
        let b = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
        assert_eq!(a.counts, b.counts);
        assert_eq!(a.counts.total(), 256);
        for (key, _) in a.counts.iter() {
            assert_eq!(&key[0..1], "1", "teleported qubit must read 1: {key}");
        }
    }

    /// (d addendum) Mid-circuit reset + measurement parity.
    #[test]
    fn dynamic_reset_and_midmeasure_parity() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1);
        c.reset(0);
        c.h(0);
        c.measure(0, 0);
        c.x(1);
        c.measure(1, 1);
        let opts = |backend| RunOptions {
            shots: 256,
            seed: Some(11),
            precision: Some(Precision::F32),
            backend,
            ..Default::default()
        };
        let p = c.to_program();
        let a = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
        let b = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
        assert_eq!(a.counts, b.counts);
        assert_eq!(a.counts.total(), 256);
    }

    /// (e) Perf smoke: 22-qubit random depth 10, Metal vs CPU wall time.
    /// The print is informational (run with --nocapture); the only assert
    /// is result parity.
    #[test]
    fn perf_smoke_22q() {
        let mut c = random_circuit(22, 10, 424_242);
        c.measure_all();
        // Pin fusion on BOTH sides: bit-exact counts parity holds only when
        // the two backends execute the identical op stream (the auto
        // default is 1 on cpu / 5 on metal, which changes f32 rounding).
        let opts = |backend| RunOptions {
            shots: 128,
            seed: Some(1),
            precision: Some(Precision::F32),
            backend,
            fusion_max: Some(5),
            ..Default::default()
        };
        let p = c.to_program();
        let g = run_program(&p, &opts(BackendChoice::Metal)).unwrap();
        let h = run_program(&p, &opts(BackendChoice::Cpu)).unwrap();
        println!(
            "perf smoke 22q depth 10 (fusion 5 both — same-config, not best-vs-best): \
             metal {:.1?} vs cpu-f32 {:.1?} ({:.2}x)",
            g.sim_time,
            h.sim_time,
            h.sim_time.as_secs_f64() / g.sim_time.as_secs_f64().max(1e-9),
        );
        assert_eq!(g.counts, h.counts);
    }
}
