//! Execution engine: backends, shot handling, measurement statistics.
//!
//! Non-dynamic circuits (all measurements trailing) are simulated **once**
//! and shots are drawn analytically from |ψ|² — 1 shot and 10⁶ shots cost
//! almost the same. Dynamic circuits (mid-circuit measurement, reset,
//! classical control) are re-executed per shot, in parallel across shots
//! when the per-thread state vectors fit in the memory budget.

use crate::compiler::{COp, Compiled};
use crate::ir::{Reg, C64};
use crate::sample::sample_indices;
use crate::state::{self, Real, StateVec};
use crate::{BackendChoice, Error, Precision};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use rayon::prelude::*;
use std::collections::{BTreeMap, HashMap};
use std::time::{Duration, Instant};

/// Options for a run. `Default` gives 1024 shots, auto precision, CPU
/// backend, fusion 5, 75% of physical RAM as budget.
#[derive(Debug, Clone)]
pub struct RunOptions {
    pub shots: u64,
    pub seed: Option<u64>,
    /// `None` = auto: f64 if it fits the budget, else f32.
    pub precision: Option<Precision>,
    pub backend: BackendChoice,
    /// Maximum *dense* fused-gate width, 0..=6 (0 disables all fusion;
    /// diagonal fusion runs whenever this is ≥ 1). `None` = auto: 1 on the
    /// CPU backend (dense fusion loses to the memory-bound 1q sweeps
    /// there), 5 on Metal (the GPU loves fat fused ops).
    pub fusion_max: Option<u8>,
    /// Fraction of physical RAM usable for the state (ignored if
    /// `mem_limit` is set).
    pub mem_fraction: f64,
    pub mem_limit: Option<u64>,
    /// Capture the final state vector (non-dynamic circuits only).
    pub want_statevector: bool,
    /// Override rayon thread count.
    pub threads: Option<usize>,
}

impl Default for RunOptions {
    fn default() -> Self {
        RunOptions {
            shots: 1024,
            seed: None,
            precision: None,
            backend: BackendChoice::Auto,
            fusion_max: None,
            mem_fraction: 0.75,
            mem_limit: None,
            want_statevector: false,
            threads: None,
        }
    }
}

/// Measurement statistics, keyed by classical bitstring
/// (qiskit convention: cregs joined by spaces, last-declared leftmost;
/// within a creg, bit `k-1` … bit `0` left to right).
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize)]
pub struct Counts(pub BTreeMap<String, u64>);

impl Counts {
    pub fn get(&self, key: &str) -> u64 {
        self.0.get(key).copied().unwrap_or(0)
    }

    pub fn total(&self) -> u64 {
        self.0.values().sum()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&String, &u64)> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// `(bitstring, probability)` sorted by descending probability.
    pub fn probabilities(&self) -> Vec<(String, f64)> {
        let total = self.total().max(1) as f64;
        let mut v: Vec<(String, f64)> = self
            .0
            .iter()
            .map(|(k, &c)| (k.clone(), c as f64 / total))
            .collect();
        v.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        v
    }
}

/// Everything a run produces.
#[derive(Debug, Clone)]
pub struct RunResult {
    pub counts: Counts,
    pub shots: u64,
    pub n_qubits: u32,
    pub precision: Precision,
    pub backend: &'static str,
    /// Master seed actually used (recorded so unseeded runs can be replayed).
    pub seed: u64,
    /// Wall time of state preparation + gates + sampling.
    pub sim_time: Duration,
    pub statevector: Option<Vec<C64>>,
    pub stats: crate::compiler::Stats,
    pub mem_bytes: u64,
    pub notices: Vec<String>,
}

/// A simulation backend: something that owns an n-qubit state and can
/// apply compiled ops to it.
pub trait Backend: Send {
    fn name(&self) -> &'static str;
    fn apply_unitary(&mut self, qs: &[u32], mat: &[C64]);
    fn apply_diagonal(&mut self, qs: &[u32], d: &[C64]);
    /// Controlled-X fast path. The default falls back to the dense
    /// 2-qubit unitary; CPU overrides with a pure swap sweep.
    fn apply_cx(&mut self, control: u32, target: u32) {
        let mat = match crate::gates::build("cx", &[]).expect("cx is native") {
            crate::gates::GateMatrix::Unitary(m) => m,
            _ => unreachable!(),
        };
        let (qs, m) = crate::compiler::permute_unitary_to_sorted(&[control, target], &mat);
        self.apply_unitary(&qs, &m);
    }
    /// Probability that qubit `q` would measure 1, without collapsing.
    /// Used by state-dependent noise channels (amplitude damping).
    fn prob_one(&mut self, q: u32) -> f64;
    /// Measure qubit `q` using the uniform draw `u ∈ [0,1)`; collapses.
    fn measure(&mut self, q: u32, u: f64) -> bool;
    fn reset_all(&mut self);
    fn sample(&mut self, shots: usize, seed: u64) -> Vec<u64>;
    fn statevector(&mut self) -> Vec<C64>;
    fn norm_sqr(&mut self) -> f64;
    /// Wait for all queued work to complete. No-op on synchronous
    /// backends; the Metal backend commits + waits here so reported sim
    /// times always include the actual GPU work, even when nothing is
    /// read back (e.g. measurement-free circuits).
    fn finish(&mut self) {}
}

pub struct CpuBackend<T: Real> {
    st: StateVec<T>,
}

impl<T: Real> CpuBackend<T> {
    pub fn new(n_qubits: u32) -> Self {
        CpuBackend {
            st: StateVec::zero_state(n_qubits),
        }
    }
}

impl<T: Real> Backend for CpuBackend<T> {
    fn name(&self) -> &'static str {
        if std::mem::size_of::<T>() == 4 {
            "cpu-f32"
        } else {
            "cpu-f64"
        }
    }

    fn apply_unitary(&mut self, qs: &[u32], mat: &[C64]) {
        state::apply_kq(&mut self.st, qs, mat);
    }

    fn apply_diagonal(&mut self, qs: &[u32], d: &[C64]) {
        state::apply_diag(&mut self.st, qs, d);
    }

    fn apply_cx(&mut self, control: u32, target: u32) {
        state::apply_cx(&mut self.st, control, target);
    }

    fn prob_one(&mut self, q: u32) -> f64 {
        state::prob_one(&self.st, q)
    }

    fn measure(&mut self, q: u32, u: f64) -> bool {
        let p1 = state::prob_one(&self.st, q);
        let outcome = u < p1;
        let p = if outcome { p1 } else { 1.0 - p1 };
        state::collapse(&mut self.st, q, outcome, p);
        outcome
    }

    fn reset_all(&mut self) {
        self.st.reset_zero();
    }

    fn sample(&mut self, shots: usize, seed: u64) -> Vec<u64> {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        sample_indices(&self.st, shots, &mut rng)
    }

    fn statevector(&mut self) -> Vec<C64> {
        self.st.to_c64()
    }

    fn norm_sqr(&mut self) -> f64 {
        self.st.norm_sqr()
    }
}

fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Format a clbit word as a counts key (see [`Counts`] docs).
pub fn format_key(clbits: u64, cregs: &[Reg]) -> String {
    if cregs.is_empty() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::with_capacity(cregs.len());
    let mut offset = 0u32;
    for reg in cregs {
        let mask = if reg.size >= 64 {
            u64::MAX
        } else {
            (1u64 << reg.size) - 1
        };
        let val = (clbits >> offset) & mask;
        parts.push(format!("{:0width$b}", val, width = reg.size as usize));
        offset += reg.size;
    }
    parts.reverse();
    parts.join(" ")
}

struct Resolved {
    precision: Precision,
    mem_bytes: u64,
    notices: Vec<String>,
}

fn resolve_precision(c: &Compiled, opts: &RunOptions) -> Result<Resolved, Error> {
    let budget = crate::mem::budget_bytes(opts.mem_limit, opts.mem_fraction);
    let mut notices = vec![];
    if matches!(opts.backend, BackendChoice::Metal) {
        // Metal has no f64.
        if opts.precision == Some(Precision::F64) {
            return Err(Error::Unsupported(
                "the Metal backend is f32-only (Metal has no f64); \
                 drop --precision f64 or use --backend cpu"
                    .into(),
            ));
        }
        crate::mem::plan(c.n_qubits, Precision::F32, budget)?;
        return Ok(Resolved {
            precision: Precision::F32,
            mem_bytes: crate::mem::state_bytes(c.n_qubits, Precision::F32) as u64,
            notices,
        });
    }
    let precision = match opts.precision {
        Some(p) => {
            crate::mem::plan(c.n_qubits, p, budget)?;
            p
        }
        None => {
            if crate::mem::plan(c.n_qubits, Precision::F64, budget).is_ok() {
                Precision::F64
            } else {
                crate::mem::plan(c.n_qubits, Precision::F32, budget)?;
                notices.push(format!(
                    "auto-precision: {} qubits exceed the f64 budget, using f32 \
                     (8 B/amplitude)",
                    c.n_qubits
                ));
                Precision::F32
            }
        }
    };
    Ok(Resolved {
        precision,
        mem_bytes: crate::mem::state_bytes(c.n_qubits, precision) as u64,
        notices,
    })
}

fn make_backend(
    n_qubits: u32,
    precision: Precision,
    choice: BackendChoice,
) -> Result<Box<dyn Backend>, Error> {
    match choice {
        BackendChoice::Auto | BackendChoice::Cpu => Ok(match precision {
            Precision::F32 => Box::new(CpuBackend::<f32>::new(n_qubits)),
            Precision::F64 => Box::new(CpuBackend::<f64>::new(n_qubits)),
        }),
        BackendChoice::Metal => {
            #[cfg(feature = "metal")]
            {
                debug_assert_eq!(precision, Precision::F32);
                Ok(Box::new(crate::metal::MetalBackend::new(n_qubits)?))
            }
            #[cfg(not(feature = "metal"))]
            {
                let _ = (n_qubits, precision);
                Err(Error::Unsupported(
                    "this build has no Metal backend (rebuild with `--features metal`)".into(),
                ))
            }
        }
    }
}

/// Run a compiled circuit.
pub fn run_compiled(c: &Compiled, opts: &RunOptions) -> Result<RunResult, Error> {
    match opts.threads {
        Some(t) => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(t)
                .build()
                .map_err(|e| Error::Unsupported(format!("thread pool: {e}")))?;
            pool.install(|| run_inner(c, opts))
        }
        None => run_inner(c, opts),
    }
}

fn run_inner(c: &Compiled, opts: &RunOptions) -> Result<RunResult, Error> {
    let resolved = resolve_precision(c, opts)?;
    let mut notices = resolved.notices.clone();
    let seed = opts.seed.unwrap_or_else(rand::random);

    if c.dynamic && opts.want_statevector {
        return Err(Error::InvalidCircuit(
            "circuit is dynamic (mid-circuit measure/reset/if): it has no \
             single final state vector"
                .into(),
        ));
    }

    let t0 = Instant::now();
    let (counts, statevector, backend_name) = if !c.dynamic {
        run_sampled(c, opts, &resolved, seed, &mut notices)?
    } else {
        run_dynamic(c, opts, &resolved, seed, &mut notices)?
    };
    let sim_time = t0.elapsed();

    Ok(RunResult {
        counts,
        shots: opts.shots,
        n_qubits: c.n_qubits,
        precision: resolved.precision,
        backend: backend_name,
        seed,
        sim_time,
        statevector,
        stats: c.stats,
        mem_bytes: resolved.mem_bytes,
        notices,
    })
}

fn apply_op(be: &mut dyn Backend, op: &COp) {
    match op {
        COp::Unitary { qubits, mat } => be.apply_unitary(qubits, mat),
        COp::Diagonal { qubits, diag } => be.apply_diagonal(qubits, diag),
        COp::Cx { control, target } => be.apply_cx(*control, *target),
        _ => unreachable!("non-unitary op in static body"),
    }
}

fn run_sampled(
    c: &Compiled,
    opts: &RunOptions,
    resolved: &Resolved,
    seed: u64,
    notices: &mut Vec<String>,
) -> Result<(Counts, Option<Vec<C64>>, &'static str), Error> {
    let mut be = make_backend(c.n_qubits, resolved.precision, opts.backend)?;
    for op in &c.ops {
        apply_op(be.as_mut(), op);
    }
    be.finish();

    let statevector = if opts.want_statevector {
        Some(be.statevector())
    } else {
        None
    };

    let mut counts = Counts::default();
    if c.final_measures.is_empty() {
        if opts.shots > 0 {
            notices.push(
                "circuit has no measurements: counts are empty \
                 (use --statevector to inspect the state)"
                    .into(),
            );
        }
    } else {
        let samples = be.sample(opts.shots as usize, splitmix64(seed));
        // Count on raw basis indices first, format unique values once.
        let mut raw: HashMap<u64, u64> = HashMap::new();
        for s in samples {
            *raw.entry(s).or_insert(0) += 1;
        }
        for (basis, n) in raw {
            let mut clbits = 0u64;
            for &(q, cb) in &c.final_measures {
                let bit = (basis >> q) & 1;
                clbits = (clbits & !(1u64 << cb)) | (bit << cb);
            }
            *counts.0.entry(format_key(clbits, &c.cregs)).or_insert(0) += n;
        }
    }
    Ok((counts, statevector, be.name()))
}

fn run_dynamic(
    c: &Compiled,
    opts: &RunOptions,
    resolved: &Resolved,
    seed: u64,
    notices: &mut Vec<String>,
) -> Result<(Counts, Option<Vec<C64>>, &'static str), Error> {
    let shots = opts.shots;
    let threads = rayon::current_num_threads().max(1);
    let budget = crate::mem::budget_bytes(opts.mem_limit, opts.mem_fraction);
    let metal = matches!(opts.backend, BackendChoice::Metal);
    let parallel =
        !metal && shots > 1 && (resolved.mem_bytes as u128) * (threads as u128) <= (budget as u128);

    let backend_name = make_backend(c.n_qubits, resolved.precision, opts.backend)?.name();

    let run_shot = |be: &mut dyn Backend, shot: u64| -> u64 {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(splitmix64(seed ^ shot));
        let mut clbits = 0u64;
        for op in &c.ops {
            exec_dynamic_op(be, op, &mut rng, &mut clbits);
        }
        clbits
    };

    // Aggregate on raw clbit words; format only the unique keys at the end
    // (String formatting per shot costs ~2x on high-shot dynamic runs).
    let mut raw: HashMap<u64, u64> = HashMap::new();
    if parallel {
        raw = (0..shots)
            .into_par_iter()
            .fold(
                || (None::<Box<dyn Backend>>, HashMap::<u64, u64>::new()),
                |(be, mut map), shot| {
                    let mut be = be.unwrap_or_else(|| {
                        make_backend(c.n_qubits, resolved.precision, opts.backend).unwrap()
                    });
                    be.reset_all();
                    let k = run_shot(be.as_mut(), shot);
                    *map.entry(k).or_insert(0) += 1;
                    (Some(be), map)
                },
            )
            .map(|(_, map)| map)
            .reduce(HashMap::new, |mut a, b| {
                for (k, v) in b {
                    *a.entry(k).or_insert(0) += v;
                }
                a
            });
    } else {
        if !parallel && shots > 1 && !metal {
            notices.push(format!(
                "dynamic circuit at {} qubits: shots run sequentially \
                 (per-thread copies exceed the memory budget)",
                c.n_qubits
            ));
        }
        let mut be = make_backend(c.n_qubits, resolved.precision, opts.backend)?;
        for shot in 0..shots {
            be.reset_all();
            let k = run_shot(be.as_mut(), shot);
            *raw.entry(k).or_insert(0) += 1;
        }
    }
    let mut counts = Counts::default();
    for (k, n) in raw {
        *counts.0.entry(format_key(k, &c.cregs)).or_insert(0) += n;
    }
    Ok((counts, None, backend_name))
}

fn exec_dynamic_op(be: &mut dyn Backend, op: &COp, rng: &mut Xoshiro256PlusPlus, clbits: &mut u64) {
    match op {
        COp::Unitary { qubits, mat } => be.apply_unitary(qubits, mat),
        COp::Diagonal { qubits, diag } => be.apply_diagonal(qubits, diag),
        COp::Cx { control, target } => be.apply_cx(*control, *target),
        COp::Measure { qubit, clbit } => {
            let bit = be.measure(*qubit, rng.gen::<f64>()) as u64;
            *clbits = (*clbits & !(1u64 << clbit)) | (bit << clbit);
        }
        COp::Reset { qubit } => {
            if be.measure(*qubit, rng.gen::<f64>()) {
                // flip back to |0⟩
                let x = [
                    C64::default(),
                    C64::new(1.0, 0.0),
                    C64::new(1.0, 0.0),
                    C64::default(),
                ];
                be.apply_unitary(&[*qubit], &x);
            }
        }
        COp::If {
            creg_offset,
            creg_len,
            value,
            inner,
        } => {
            let mask = if *creg_len >= 64 {
                u64::MAX
            } else {
                (1u64 << creg_len) - 1
            };
            if (*clbits >> creg_offset) & mask == *value {
                exec_dynamic_op(be, inner, rng, clbits);
            }
        }
        COp::Barrier => unreachable!("barriers are stripped at compile time"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::Circuit;
    use crate::compiler::{compile, CompileOptions};

    fn run(c: &Circuit, opts: &RunOptions) -> RunResult {
        let compiled = compile(&c.to_program(), &CompileOptions::default()).unwrap();
        run_compiled(&compiled, opts).unwrap()
    }

    #[test]
    fn bell_counts() {
        let mut c = Circuit::new(2);
        c.h(0).cx(0, 1).measure_all();
        let r = run(
            &c,
            &RunOptions {
                shots: 4096,
                seed: Some(1),
                ..Default::default()
            },
        );
        assert_eq!(r.counts.get("00") + r.counts.get("11"), 4096);
        let p = r.counts.get("00") as f64 / 4096.0;
        assert!((p - 0.5).abs() < 0.05, "p={p}");
    }

    #[test]
    fn deterministic_with_seed() {
        let mut c = Circuit::new(3);
        c.h(0).h(1).h(2).measure_all();
        let opts = RunOptions {
            shots: 1000,
            seed: Some(99),
            ..Default::default()
        };
        assert_eq!(run(&c, &opts).counts, run(&c, &opts).counts);
    }

    #[test]
    fn teleportation_dynamic() {
        // Teleport |1⟩ from q0 to q2 using mid-circuit measurement and
        // classically controlled corrections (QASM2-style 1-bit cregs).
        use crate::ir::{GateInstr, Instr, Program, Reg};
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
                g("x", &[0]), // state to teleport: |1⟩
                g("h", &[1]),
                g("cx", &[1, 2]), // EPR pair
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
        let compiled = compile(&p, &CompileOptions::default()).unwrap();
        assert!(compiled.dynamic);
        let r = run_compiled(
            &compiled,
            &RunOptions {
                shots: 512,
                seed: Some(5),
                ..Default::default()
            },
        )
        .unwrap();
        // Key layout: "out m1 m0" — qubit 2 must always land in |1⟩.
        assert_eq!(r.counts.total(), 512);
        for (key, n) in r.counts.iter() {
            assert!(*n > 0);
            assert_eq!(&key[0..1], "1", "out bit must be 1, got key {key}");
        }
    }

    #[test]
    fn f32_matches_f64_for_ghz() {
        let mut c = Circuit::new(10);
        c.h(0);
        for q in 0..9 {
            c.cx(q, q + 1);
        }
        c.measure_all();
        let mk = |prec| RunOptions {
            shots: 2000,
            seed: Some(3),
            precision: Some(prec),
            ..Default::default()
        };
        let a = run(&c, &mk(Precision::F64));
        let b = run(&c, &mk(Precision::F32));
        assert_eq!(a.counts, b.counts);
        let z = "0".repeat(10);
        let o = "1".repeat(10);
        assert_eq!(a.counts.get(&z) + a.counts.get(&o), 2000);
    }

    #[test]
    fn format_key_qiskit_convention() {
        let cregs = vec![
            Reg {
                name: "a".into(),
                size: 2,
            },
            Reg {
                name: "b".into(),
                size: 3,
            },
        ];
        // a = bits 0..2 = 0b10, b = bits 2..5 = 0b011
        let clbits = 0b01110u64;
        assert_eq!(format_key(clbits, &cregs), "011 10");
    }
}
