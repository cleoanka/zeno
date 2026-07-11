//! State vector storage and gate kernels.
//!
//! Amplitudes are stored *split* (structure-of-arrays: one array of reals,
//! one of imaginaries). On Apple Silicon this lets the NEON kernels below
//! stream whole vectors of independent lanes with zero shuffle traffic,
//! and it halves the working set per stream, which matters because these
//! kernels are memory-bound.
//!
//! ## SIMD and the bit-exactness contract
//!
//! On aarch64 the hot kernels (`kern_1q`, both `apply_diag` paths and the
//! fused `apply_kq`) run explicit NEON (f32×4 / f64×2). Each vector lane
//! holds one independent amplitude — or, in `apply_kq`, one independent
//! gate *group* — and is computed with plain lanewise mul/add/sub in
//! exactly the order of the scalar expression tree. Fused multiply-add is
//! never used: it would change rounding. Lanewise NEON arithmetic is
//! IEEE-754 identical to scalar arithmetic, so the NEON paths produce
//! bit-for-bit the same state as the retained `*_scalar` kernels, which
//! stay compiled both as the portable fallback and as the oracle the NEON
//! paths are exhaustively tested against (see the `to_bits` tests below).
//! `prob_one`, `collapse` and `norm_sqr` are deliberately left scalar:
//! their f64 reduction order feeds measurement decisions that the Metal
//! parity suite pins.
//!
//! Parallelism: rayon over contiguous chunks of the "half index" space
//! (or group space for fused gates). Each task owns a disjoint slice of
//! amplitude pairs/groups, so the raw-pointer writes below are data-race
//! free by construction.

use crate::ir::C64;
use rayon::prelude::*;

/// Scalar type of a state vector (f32 or f64).
pub trait Real:
    num_traits::Float + std::ops::AddAssign + Send + Sync + std::fmt::Debug + 'static
{
    fn fr(x: f64) -> Self;
    fn to64(self) -> f64;
}

impl Real for f32 {
    #[inline(always)]
    fn fr(x: f64) -> Self {
        x as f32
    }
    #[inline(always)]
    fn to64(self) -> f64 {
        self as f64
    }
}

impl Real for f64 {
    #[inline(always)]
    fn fr(x: f64) -> Self {
        x
    }
    #[inline(always)]
    fn to64(self) -> f64 {
        self
    }
}

/// Largest fused-gate dimension (2^6 → up to 6-qubit fused unitaries).
pub const MAX_FUSED_QUBITS: usize = 6;
const MAX_DIM: usize = 1 << MAX_FUSED_QUBITS;

/// Below this many items a kernel runs single-threaded (fork overhead
/// dominates for tiny states).
const PAR_MIN: usize = 1 << 13;

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

/// Split-complex state vector of `n_qubits`.
#[derive(Debug, Clone)]
pub struct StateVec<T: Real> {
    pub re: Vec<T>,
    pub im: Vec<T>,
    pub n_qubits: u32,
}

fn task_ranges(total: usize) -> Vec<(usize, usize)> {
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

impl<T: Real> StateVec<T> {
    /// |00…0⟩
    pub fn zero_state(n_qubits: u32) -> Self {
        let len = 1usize << n_qubits;
        let mut re = Vec::with_capacity(len);
        let mut im = Vec::with_capacity(len);
        re.resize(len, T::zero());
        im.resize(len, T::zero());
        re[0] = T::one();
        StateVec { re, im, n_qubits }
    }

    pub fn len(&self) -> usize {
        self.re.len()
    }

    pub fn is_empty(&self) -> bool {
        false
    }

    /// Reset to |00…0⟩ in place.
    pub fn reset_zero(&mut self) {
        self.re.par_iter_mut().for_each(|x| *x = T::zero());
        self.im.par_iter_mut().for_each(|x| *x = T::zero());
        self.re[0] = T::one();
    }

    pub fn amp(&self, i: usize) -> C64 {
        C64::new(self.re[i].to64(), self.im[i].to64())
    }

    pub fn to_c64(&self) -> Vec<C64> {
        (0..self.len()).map(|i| self.amp(i)).collect()
    }

    /// ⟨ψ|ψ⟩, accumulated in f64.
    pub fn norm_sqr(&self) -> f64 {
        self.re
            .par_chunks(1 << 16)
            .zip(self.im.par_chunks(1 << 16))
            .map(|(rs, is)| {
                let mut acc = 0.0f64;
                for (r, i) in rs.iter().zip(is) {
                    let (r, i) = (r.to64(), i.to64());
                    acc += r * r + i * i;
                }
                acc
            })
            .sum()
    }
}

/// Insert zero bits into `g` at the (ascending) positions `qs`, producing
/// the base index of a gate group.
#[inline(always)]
pub fn insert_zeros(g: usize, qs: &[u32]) -> usize {
    let mut x = g;
    for &q in qs {
        let low = x & ((1usize << q) - 1);
        x = ((x >> q) << (q + 1)) | low;
    }
    x
}

/// Apply a 1-qubit gate (row-major 2×2 matrix) to qubit `q`.
pub fn apply_1q<T: Real>(st: &mut StateVec<T>, q: u32, m: &[C64]) {
    debug_assert_eq!(m.len(), 4);
    let half = st.len() >> 1;
    let mm: [(T, T); 4] = [
        (T::fr(m[0].re), T::fr(m[0].im)),
        (T::fr(m[1].re), T::fr(m[1].im)),
        (T::fr(m[2].re), T::fr(m[2].im)),
        (T::fr(m[3].re), T::fr(m[3].im)),
    ];
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(half)
        .into_par_iter()
        .for_each(|(h0, h1)| unsafe {
            kern_1q::<T>(re.ptr(), im.ptr(), q, h0, h1, &mm);
        });
}

/// NEON on aarch64 for f32/f64, scalar otherwise. Bit-identical either way
/// (see the module docs).
///
/// # Safety
/// Same contract as [`kern_1q_scalar`].
unsafe fn kern_1q<T: Real>(re: *mut T, im: *mut T, q: u32, h0: usize, h1: usize, m: &[(T, T); 4]) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32 (checked above); identical layout, so the
            // pointer and matrix casts are the identity at runtime.
            return neon::kern_1q_f32(
                re.cast(),
                im.cast(),
                q,
                h0,
                h1,
                &*(m as *const [(T, T); 4]).cast(),
            );
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T == f64; identity casts as above.
            return neon::kern_1q_f64(
                re.cast(),
                im.cast(),
                q,
                h0,
                h1,
                &*(m as *const [(T, T); 4]).cast(),
            );
        }
    }
    kern_1q_scalar(re, im, q, h0, h1, m)
}

/// Scalar 1-qubit kernel: the portable fallback and the bit-exactness
/// oracle for the NEON path.
///
/// # Safety
/// `[h0, h1)` ranges across calls must partition `0..len/2`; each half
/// index maps to a unique disjoint amplitude pair.
unsafe fn kern_1q_scalar<T: Real>(
    re: *mut T,
    im: *mut T,
    q: u32,
    h0: usize,
    h1: usize,
    m: &[(T, T); 4],
) {
    let s = 1usize << q;
    let mask = s - 1;
    let [(m00r, m00i), (m01r, m01i), (m10r, m10i), (m11r, m11i)] = *m;
    let mut h = h0;
    while h < h1 {
        let run = (s - (h & mask)).min(h1 - h);
        let a0 = ((h >> q) << (q + 1)) | (h & mask);
        let ra = std::slice::from_raw_parts_mut(re.add(a0), run);
        let ia = std::slice::from_raw_parts_mut(im.add(a0), run);
        let rb = std::slice::from_raw_parts_mut(re.add(a0 + s), run);
        let ib = std::slice::from_raw_parts_mut(im.add(a0 + s), run);
        for t in 0..run {
            let (ar, ai, br, bi) = (ra[t], ia[t], rb[t], ib[t]);
            ra[t] = m00r * ar - m00i * ai + m01r * br - m01i * bi;
            ia[t] = m00r * ai + m00i * ar + m01r * bi + m01i * br;
            rb[t] = m10r * ar - m10i * ai + m11r * br - m11i * bi;
            ib[t] = m10r * ai + m10i * ar + m11r * bi + m11i * br;
        }
        h += run;
    }
}

/// Read-only tables shared by every task of one `apply_kq` call.
struct KqTables<'a, T> {
    /// Sorted target qubits.
    qs: &'a [u32],
    /// `offs[j]` = local matrix index `j` scattered to state-index bits.
    offs: &'a [usize],
    /// Row-major `2^k × 2^k` matrix, split re/im.
    mre: &'a [T],
    mim: &'a [T],
    /// `2^k`.
    dim: usize,
}

/// Apply a `k`-qubit unitary (row-major `2^k × 2^k`) to sorted, distinct
/// qubits `qs`. Bit `b` of a local matrix index corresponds to `qs[b]`.
pub fn apply_kq<T: Real>(st: &mut StateVec<T>, qs: &[u32], m: &[C64]) {
    let k = qs.len();
    debug_assert!((1..=MAX_FUSED_QUBITS).contains(&k));
    debug_assert!(qs.windows(2).all(|w| w[0] < w[1]));
    debug_assert_eq!(m.len(), (1 << k) * (1 << k));
    if k == 1 {
        return apply_1q(st, qs[0], m);
    }
    let dim = 1usize << k;
    let mre: Vec<T> = m.iter().map(|z| T::fr(z.re)).collect();
    let mim: Vec<T> = m.iter().map(|z| T::fr(z.im)).collect();
    let offs: Vec<usize> = (0..dim)
        .map(|j| {
            let mut o = 0usize;
            for (b, &q) in qs.iter().enumerate() {
                if (j >> b) & 1 == 1 {
                    o |= 1usize << q;
                }
            }
            o
        })
        .collect();
    let groups = st.len() >> k;
    let tbl = KqTables {
        qs,
        offs: &offs,
        mre: &mre,
        mim: &mim,
        dim,
    };
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(groups)
        .into_par_iter()
        .for_each(|(g0, g1)| unsafe {
            kern_kq::<T>(re.ptr(), im.ptr(), &tbl, g0, g1);
        });
}

/// NEON on aarch64 for f32/f64, scalar otherwise. Bit-identical either way.
///
/// # Safety
/// Same contract as [`kern_kq_scalar`].
unsafe fn kern_kq<T: Real>(re: *mut T, im: *mut T, tbl: &KqTables<'_, T>, g0: usize, g1: usize) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32; `KqTables<'_, f32>` is the same type, so
            // the reference cast is the identity.
            return neon::kern_kq_f32(
                re.cast(),
                im.cast(),
                &*(tbl as *const KqTables<'_, T>).cast::<KqTables<'_, f32>>(),
                g0,
                g1,
            );
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T == f64; identity cast as above.
            return neon::kern_kq_f64(
                re.cast(),
                im.cast(),
                &*(tbl as *const KqTables<'_, T>).cast::<KqTables<'_, f64>>(),
                g0,
                g1,
            );
        }
    }
    kern_kq_scalar(re, im, tbl, g0, g1)
}

/// Scalar fused-gate kernel: the portable fallback and the bit-exactness
/// oracle for the NEON path.
///
/// # Safety
/// `[g0, g1)` ranges across calls must partition the group space; each
/// group owns a disjoint set of `2^k` amplitudes.
unsafe fn kern_kq_scalar<T: Real>(
    re: *mut T,
    im: *mut T,
    tbl: &KqTables<'_, T>,
    g0: usize,
    g1: usize,
) {
    let dim = tbl.dim;
    let mut xr = [T::zero(); MAX_DIM];
    let mut xi = [T::zero(); MAX_DIM];
    let mut yr = [T::zero(); MAX_DIM];
    let mut yi = [T::zero(); MAX_DIM];
    for g in g0..g1 {
        let base = insert_zeros(g, tbl.qs);
        for j in 0..dim {
            let p = base | tbl.offs[j];
            xr[j] = *re.add(p);
            xi[j] = *im.add(p);
        }
        for r in 0..dim {
            let row = r * dim;
            let mut ar = T::zero();
            let mut ai = T::zero();
            for c in 0..dim {
                let (wr, wi) = (tbl.mre[row + c], tbl.mim[row + c]);
                ar = ar + wr * xr[c] - wi * xi[c];
                ai = ai + wr * xi[c] + wi * xr[c];
            }
            yr[r] = ar;
            yi[r] = ai;
        }
        for j in 0..dim {
            let p = base | tbl.offs[j];
            *re.add(p) = yr[j];
            *im.add(p) = yi[j];
        }
    }
}

/// Controlled-X as a pure permutation: swap the target pair inside the
/// control=1 half. Zero arithmetic, touches half the state — ~4x cheaper
/// than the dense 2-qubit path.
pub fn apply_cx<T: Real>(st: &mut StateVec<T>, control: u32, target: u32) {
    debug_assert_ne!(control, target);
    let qs = [control.min(target), control.max(target)];
    let coff = 1usize << control;
    let toff = 1usize << target;
    let groups = st.len() >> 2;
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(groups)
        .into_par_iter()
        .for_each(|(g0, g1)| unsafe {
            let re = re.ptr();
            let im = im.ptr();
            for g in g0..g1 {
                let a = insert_zeros(g, &qs) | coff;
                let b = a | toff;
                std::ptr::swap(re.add(a), re.add(b));
                std::ptr::swap(im.add(a), im.add(b));
            }
        });
}

/// Apply a diagonal unitary given its diagonal (length `2^k`) on sorted
/// distinct qubits `qs`. Single O(2^n) sweep.
pub fn apply_diag<T: Real>(st: &mut StateVec<T>, qs: &[u32], d: &[C64]) {
    let k = qs.len();
    debug_assert_eq!(d.len(), 1 << k);
    debug_assert!(qs.windows(2).all(|w| w[0] < w[1]));
    if k == 1 {
        let half = st.len() >> 1;
        let q = qs[0];
        let d0 = (T::fr(d[0].re), T::fr(d[0].im));
        let d1 = (T::fr(d[1].re), T::fr(d[1].im));
        let re = SendPtr(st.re.as_mut_ptr());
        let im = SendPtr(st.im.as_mut_ptr());
        task_ranges(half)
            .into_par_iter()
            .for_each(|(h0, h1)| unsafe {
                kern_diag1::<T>(re.ptr(), im.ptr(), q, h0, h1, d0, d1);
            });
        return;
    }
    let dre: Vec<T> = d.iter().map(|z| T::fr(z.re)).collect();
    let dim_: Vec<T> = d.iter().map(|z| T::fr(z.im)).collect();
    let qs: Vec<u32> = qs.to_vec();
    let n = st.len();
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(n).into_par_iter().for_each(|(i0, i1)| unsafe {
        kern_diagk::<T>(re.ptr(), im.ptr(), &qs, &dre, &dim_, i0, i1);
    });
}

/// NEON on aarch64 for f32/f64, scalar otherwise. Bit-identical either way.
///
/// # Safety
/// Same contract as [`kern_diag1_scalar`].
unsafe fn kern_diag1<T: Real>(
    re: *mut T,
    im: *mut T,
    q: u32,
    h0: usize,
    h1: usize,
    d0: (T, T),
    d1: (T, T),
) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32; identity casts (checked by TypeId).
            return neon::kern_diag1_f32(
                re.cast(),
                im.cast(),
                q,
                h0,
                h1,
                std::mem::transmute_copy::<(T, T), (f32, f32)>(&d0),
                std::mem::transmute_copy::<(T, T), (f32, f32)>(&d1),
            );
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T == f64; identity casts as above.
            return neon::kern_diag1_f64(
                re.cast(),
                im.cast(),
                q,
                h0,
                h1,
                std::mem::transmute_copy::<(T, T), (f64, f64)>(&d0),
                std::mem::transmute_copy::<(T, T), (f64, f64)>(&d1),
            );
        }
    }
    kern_diag1_scalar(re, im, q, h0, h1, d0, d1)
}

/// Scalar 1-qubit diagonal kernel: portable fallback and NEON oracle.
///
/// # Safety
/// `[h0, h1)` ranges across calls must partition `0..len/2`.
unsafe fn kern_diag1_scalar<T: Real>(
    re: *mut T,
    im: *mut T,
    q: u32,
    h0: usize,
    h1: usize,
    d0: (T, T),
    d1: (T, T),
) {
    let s = 1usize << q;
    let mask = s - 1;
    let mut h = h0;
    while h < h1 {
        let run = (s - (h & mask)).min(h1 - h);
        let a0 = ((h >> q) << (q + 1)) | (h & mask);
        let ra = std::slice::from_raw_parts_mut(re.add(a0), run);
        let ia = std::slice::from_raw_parts_mut(im.add(a0), run);
        let rb = std::slice::from_raw_parts_mut(re.add(a0 + s), run);
        let ib = std::slice::from_raw_parts_mut(im.add(a0 + s), run);
        for t in 0..run {
            let (ar, ai) = (ra[t], ia[t]);
            ra[t] = d0.0 * ar - d0.1 * ai;
            ia[t] = d0.0 * ai + d0.1 * ar;
            let (br, bi) = (rb[t], ib[t]);
            rb[t] = d1.0 * br - d1.1 * bi;
            ib[t] = d1.0 * bi + d1.1 * br;
        }
        h += run;
    }
}

/// NEON on aarch64 for f32/f64, scalar otherwise. Bit-identical either way.
///
/// # Safety
/// Same contract as [`kern_diagk_scalar`].
unsafe fn kern_diagk<T: Real>(
    re: *mut T,
    im: *mut T,
    qs: &[u32],
    dre: &[T],
    dim_: &[T],
    i0: usize,
    i1: usize,
) {
    #[cfg(target_arch = "aarch64")]
    {
        use std::any::TypeId;
        if TypeId::of::<T>() == TypeId::of::<f32>() {
            // SAFETY: T == f32; identity casts (checked by TypeId).
            return neon::kern_diagk_f32(
                re.cast(),
                im.cast(),
                qs,
                std::slice::from_raw_parts(dre.as_ptr().cast::<f32>(), dre.len()),
                std::slice::from_raw_parts(dim_.as_ptr().cast::<f32>(), dim_.len()),
                i0,
                i1,
            );
        }
        if TypeId::of::<T>() == TypeId::of::<f64>() {
            // SAFETY: T == f64; identity casts as above.
            return neon::kern_diagk_f64(
                re.cast(),
                im.cast(),
                qs,
                std::slice::from_raw_parts(dre.as_ptr().cast::<f64>(), dre.len()),
                std::slice::from_raw_parts(dim_.as_ptr().cast::<f64>(), dim_.len()),
                i0,
                i1,
            );
        }
    }
    kern_diagk_scalar(re, im, qs, dre, dim_, i0, i1)
}

/// Scalar k-qubit diagonal kernel: portable fallback and NEON oracle.
///
/// # Safety
/// `[i0, i1)` ranges across calls must partition `0..len`.
unsafe fn kern_diagk_scalar<T: Real>(
    re: *mut T,
    im: *mut T,
    qs: &[u32],
    dre: &[T],
    dim_: &[T],
    i0: usize,
    i1: usize,
) {
    for i in i0..i1 {
        let mut j = 0usize;
        for (b, &q) in qs.iter().enumerate() {
            j |= ((i >> q) & 1) << b;
        }
        let (wr, wi) = (dre[j], dim_[j]);
        let (ar, ai) = (*re.add(i), *im.add(i));
        *re.add(i) = wr * ar - wi * ai;
        *im.add(i) = wr * ai + wi * ar;
    }
}

/// Probability that qubit `q` measures 1. Accumulated in f64.
pub fn prob_one<T: Real>(st: &StateVec<T>, q: u32) -> f64 {
    let half = st.len() >> 1;
    let re = st.re.as_ptr() as usize;
    let im = st.im.as_ptr() as usize;
    task_ranges(half)
        .into_par_iter()
        .map(|(h0, h1)| unsafe {
            let re = re as *const T;
            let im = im as *const T;
            let s = 1usize << q;
            let mask = s - 1;
            let mut acc = 0.0f64;
            let mut h = h0;
            while h < h1 {
                let run = (s - (h & mask)).min(h1 - h);
                let a0 = ((h >> q) << (q + 1)) | (h & mask);
                let rb = std::slice::from_raw_parts(re.add(a0 + s), run);
                let ib = std::slice::from_raw_parts(im.add(a0 + s), run);
                for t in 0..run {
                    let (r, i) = (rb[t].to64(), ib[t].to64());
                    acc += r * r + i * i;
                }
                h += run;
            }
            acc
        })
        .sum()
}

/// Collapse qubit `q` to `outcome` (which was observed with probability
/// `prob`) and renormalize.
pub fn collapse<T: Real>(st: &mut StateVec<T>, q: u32, outcome: bool, prob: f64) {
    let scale = T::fr(1.0 / prob.sqrt());
    let half = st.len() >> 1;
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(half)
        .into_par_iter()
        .for_each(|(h0, h1)| unsafe {
            let re = re.ptr();
            let im = im.ptr();
            let s = 1usize << q;
            let mask = s - 1;
            let mut h = h0;
            while h < h1 {
                let run = (s - (h & mask)).min(h1 - h);
                let a0 = ((h >> q) << (q + 1)) | (h & mask);
                // "a" side = qubit 0, "b" side = qubit 1
                let (keep, zero) = if outcome { (a0 + s, a0) } else { (a0, a0 + s) };
                let rk = std::slice::from_raw_parts_mut(re.add(keep), run);
                let ik = std::slice::from_raw_parts_mut(im.add(keep), run);
                let rz = std::slice::from_raw_parts_mut(re.add(zero), run);
                let iz = std::slice::from_raw_parts_mut(im.add(zero), run);
                for t in 0..run {
                    rk[t] = rk[t] * scale;
                    ik[t] = ik[t] * scale;
                    rz[t] = T::zero();
                    iz[t] = T::zero();
                }
                h += run;
            }
        });
}

/// Explicit NEON kernels (aarch64 only).
///
/// Bit-exactness: every lane holds one independent element and is computed
/// with the *same* mul/add/sub expression tree, in the same order, as the
/// scalar kernels — FMA is never emitted (NEON intrinsics lower to plain
/// `fmul`/`fadd`/`fsub`, and rustc does not contract them). Deinterleaving
/// loads (`vld2q`, `vuzp`/`vzip`) only move bits; they never round. The
/// scalar kernels in the parent module remain the oracle these paths are
/// tested against with `to_bits` equality.
#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{KqTables, MAX_DIM};
    use std::arch::aarch64::*;

    /// f32 `kern_1q` for `s == 2` (qubit 1): runs are two lanes wide, so
    /// four half-indices span the pattern `[a a b b | a a b b]`. Viewed as
    /// f64 pairs that is `[A B A' B']`, which `vuzp1q/vuzp2q_f64`
    /// deinterleave into an all-a and an all-b vector (pure bit moves).
    ///
    /// # Safety
    /// Same contract as [`super::kern_1q_scalar`]; `q` must be 1.
    unsafe fn kern_1q_f32_s2(
        re: *mut f32,
        im: *mut f32,
        q: u32,
        h0: usize,
        h1: usize,
        m: &[(f32, f32); 4],
    ) {
        debug_assert_eq!(q, 1);
        let [(m00r, m00i), (m01r, m01i), (m10r, m10i), (m11r, m11i)] = *m;
        // Scalar pre-roll to an even half-index (start of an a-run).
        let hp = (h0 + (h0 & 1)).min(h1);
        if h0 < hp {
            super::kern_1q_scalar(re, im, q, h0, hp, m);
        }
        let w00r = vdupq_n_f32(m00r);
        let w00i = vdupq_n_f32(m00i);
        let w01r = vdupq_n_f32(m01r);
        let w01i = vdupq_n_f32(m01i);
        let w10r = vdupq_n_f32(m10r);
        let w10i = vdupq_n_f32(m10i);
        let w11r = vdupq_n_f32(m11r);
        let w11i = vdupq_n_f32(m11i);
        let mut h = hp;
        while h + 4 <= h1 {
            let pr = re.add(2 * h);
            let pi = im.add(2 * h);
            let r0 = vreinterpretq_f64_f32(vld1q_f32(pr));
            let r1 = vreinterpretq_f64_f32(vld1q_f32(pr.add(4)));
            let i0 = vreinterpretq_f64_f32(vld1q_f32(pi));
            let i1 = vreinterpretq_f64_f32(vld1q_f32(pi.add(4)));
            let ar = vreinterpretq_f32_f64(vuzp1q_f64(r0, r1));
            let br = vreinterpretq_f32_f64(vuzp2q_f64(r0, r1));
            let ai = vreinterpretq_f32_f64(vuzp1q_f64(i0, i1));
            let bi = vreinterpretq_f32_f64(vuzp2q_f64(i0, i1));
            let nra = vsubq_f32(
                vaddq_f32(
                    vsubq_f32(vmulq_f32(w00r, ar), vmulq_f32(w00i, ai)),
                    vmulq_f32(w01r, br),
                ),
                vmulq_f32(w01i, bi),
            );
            let nia = vaddq_f32(
                vaddq_f32(
                    vaddq_f32(vmulq_f32(w00r, ai), vmulq_f32(w00i, ar)),
                    vmulq_f32(w01r, bi),
                ),
                vmulq_f32(w01i, br),
            );
            let nrb = vsubq_f32(
                vaddq_f32(
                    vsubq_f32(vmulq_f32(w10r, ar), vmulq_f32(w10i, ai)),
                    vmulq_f32(w11r, br),
                ),
                vmulq_f32(w11i, bi),
            );
            let nib = vaddq_f32(
                vaddq_f32(
                    vaddq_f32(vmulq_f32(w10r, ai), vmulq_f32(w10i, ar)),
                    vmulq_f32(w11r, bi),
                ),
                vmulq_f32(w11i, br),
            );
            let nrad = vreinterpretq_f64_f32(nra);
            let nrbd = vreinterpretq_f64_f32(nrb);
            let niad = vreinterpretq_f64_f32(nia);
            let nibd = vreinterpretq_f64_f32(nib);
            vst1q_f32(pr, vreinterpretq_f32_f64(vzip1q_f64(nrad, nrbd)));
            vst1q_f32(pr.add(4), vreinterpretq_f32_f64(vzip2q_f64(nrad, nrbd)));
            vst1q_f32(pi, vreinterpretq_f32_f64(vzip1q_f64(niad, nibd)));
            vst1q_f32(pi.add(4), vreinterpretq_f32_f64(vzip2q_f64(niad, nibd)));
            h += 4;
        }
        if h < h1 {
            super::kern_1q_scalar(re, im, q, h, h1, m);
        }
    }

    /// f32 `kern_diag1` for `s == 2` (qubit 1); same pair trick as
    /// [`kern_1q_f32_s2`].
    ///
    /// # Safety
    /// Same contract as [`super::kern_diag1_scalar`]; `q` must be 1.
    unsafe fn kern_diag1_f32_s2(
        re: *mut f32,
        im: *mut f32,
        q: u32,
        h0: usize,
        h1: usize,
        d0: (f32, f32),
        d1: (f32, f32),
    ) {
        debug_assert_eq!(q, 1);
        let hp = (h0 + (h0 & 1)).min(h1);
        if h0 < hp {
            super::kern_diag1_scalar(re, im, q, h0, hp, d0, d1);
        }
        let w0r = vdupq_n_f32(d0.0);
        let w0i = vdupq_n_f32(d0.1);
        let w1r = vdupq_n_f32(d1.0);
        let w1i = vdupq_n_f32(d1.1);
        let mut h = hp;
        while h + 4 <= h1 {
            let pr = re.add(2 * h);
            let pi = im.add(2 * h);
            let r0 = vreinterpretq_f64_f32(vld1q_f32(pr));
            let r1 = vreinterpretq_f64_f32(vld1q_f32(pr.add(4)));
            let i0 = vreinterpretq_f64_f32(vld1q_f32(pi));
            let i1 = vreinterpretq_f64_f32(vld1q_f32(pi.add(4)));
            let ar = vreinterpretq_f32_f64(vuzp1q_f64(r0, r1));
            let br = vreinterpretq_f32_f64(vuzp2q_f64(r0, r1));
            let ai = vreinterpretq_f32_f64(vuzp1q_f64(i0, i1));
            let bi = vreinterpretq_f32_f64(vuzp2q_f64(i0, i1));
            let nra = vsubq_f32(vmulq_f32(w0r, ar), vmulq_f32(w0i, ai));
            let nia = vaddq_f32(vmulq_f32(w0r, ai), vmulq_f32(w0i, ar));
            let nrb = vsubq_f32(vmulq_f32(w1r, br), vmulq_f32(w1i, bi));
            let nib = vaddq_f32(vmulq_f32(w1r, bi), vmulq_f32(w1i, br));
            let nrad = vreinterpretq_f64_f32(nra);
            let nrbd = vreinterpretq_f64_f32(nrb);
            let niad = vreinterpretq_f64_f32(nia);
            let nibd = vreinterpretq_f64_f32(nib);
            vst1q_f32(pr, vreinterpretq_f32_f64(vzip1q_f64(nrad, nrbd)));
            vst1q_f32(pr.add(4), vreinterpretq_f32_f64(vzip2q_f64(nrad, nrbd)));
            vst1q_f32(pi, vreinterpretq_f32_f64(vzip1q_f64(niad, nibd)));
            vst1q_f32(pi.add(4), vreinterpretq_f32_f64(vzip2q_f64(niad, nibd)));
            h += 4;
        }
        if h < h1 {
            super::kern_diag1_scalar(re, im, q, h, h1, d0, d1);
        }
    }

    macro_rules! neon_kernels {
        (
            $ty:ty,
            $lanes:literal,
            $vld1:ident,
            $vst1:ident,
            $vld2:ident,
            $vst2:ident,
            $vx2:ident,
            $vdup:ident,
            $vmul:ident,
            $vadd:ident,
            $vsub:ident,
            $small_1q:path,
            $small_diag1:path,
            $kern_1q:ident,
            $kern_diag1:ident,
            $kern_diagk:ident,
            $kern_kq:ident
        ) => {
            /// Vector `kern_1q`: lanes stream the contiguous a/b runs the
            /// run-walk produces; `q == 0` deinterleaves adjacent pairs
            /// with `vld2q`; sub-vector runs fall back to the scalar (or
            /// pair-trick) kernel. Tails inside a run use the identical
            /// scalar expressions.
            ///
            /// # Safety
            /// Same contract as [`super::kern_1q_scalar`].
            pub unsafe fn $kern_1q(
                re: *mut $ty,
                im: *mut $ty,
                q: u32,
                h0: usize,
                h1: usize,
                m: &[($ty, $ty); 4],
            ) {
                const L: usize = $lanes;
                let s = 1usize << q;
                if s < L && s != 1 {
                    return $small_1q(re, im, q, h0, h1, m);
                }
                let mask = s - 1;
                let [(m00r, m00i), (m01r, m01i), (m10r, m10i), (m11r, m11i)] = *m;
                let w00r = $vdup(m00r);
                let w00i = $vdup(m00i);
                let w01r = $vdup(m01r);
                let w01i = $vdup(m01i);
                let w10r = $vdup(m10r);
                let w10i = $vdup(m10i);
                let w11r = $vdup(m11r);
                let w11i = $vdup(m11i);
                if s >= L {
                    let mut h = h0;
                    while h < h1 {
                        let run = (s - (h & mask)).min(h1 - h);
                        let a0 = ((h >> q) << (q + 1)) | (h & mask);
                        let ra = re.add(a0);
                        let ia = im.add(a0);
                        let rb = re.add(a0 + s);
                        let ib = im.add(a0 + s);
                        let mut t = 0usize;
                        while t + L <= run {
                            let ar = $vld1(ra.add(t));
                            let ai = $vld1(ia.add(t));
                            let br = $vld1(rb.add(t));
                            let bi = $vld1(ib.add(t));
                            let nra = $vsub(
                                $vadd($vsub($vmul(w00r, ar), $vmul(w00i, ai)), $vmul(w01r, br)),
                                $vmul(w01i, bi),
                            );
                            let nia = $vadd(
                                $vadd($vadd($vmul(w00r, ai), $vmul(w00i, ar)), $vmul(w01r, bi)),
                                $vmul(w01i, br),
                            );
                            let nrb = $vsub(
                                $vadd($vsub($vmul(w10r, ar), $vmul(w10i, ai)), $vmul(w11r, br)),
                                $vmul(w11i, bi),
                            );
                            let nib = $vadd(
                                $vadd($vadd($vmul(w10r, ai), $vmul(w10i, ar)), $vmul(w11r, bi)),
                                $vmul(w11i, br),
                            );
                            $vst1(ra.add(t), nra);
                            $vst1(ia.add(t), nia);
                            $vst1(rb.add(t), nrb);
                            $vst1(ib.add(t), nib);
                            t += L;
                        }
                        while t < run {
                            let (ar, ai) = (*ra.add(t), *ia.add(t));
                            let (br, bi) = (*rb.add(t), *ib.add(t));
                            *ra.add(t) = m00r * ar - m00i * ai + m01r * br - m01i * bi;
                            *ia.add(t) = m00r * ai + m00i * ar + m01r * bi + m01i * br;
                            *rb.add(t) = m10r * ar - m10i * ai + m11r * br - m11i * bi;
                            *ib.add(t) = m10r * ai + m10i * ar + m11r * bi + m11i * br;
                            t += 1;
                        }
                        h += run;
                    }
                } else {
                    // s == 1 (qubit 0): pairs are adjacent, [a b a b …].
                    let mut h = h0;
                    while h + L <= h1 {
                        let pr = re.add(2 * h);
                        let pi = im.add(2 * h);
                        let r2 = $vld2(pr);
                        let i2 = $vld2(pi);
                        let (ar, br) = (r2.0, r2.1);
                        let (ai, bi) = (i2.0, i2.1);
                        let nra = $vsub(
                            $vadd($vsub($vmul(w00r, ar), $vmul(w00i, ai)), $vmul(w01r, br)),
                            $vmul(w01i, bi),
                        );
                        let nia = $vadd(
                            $vadd($vadd($vmul(w00r, ai), $vmul(w00i, ar)), $vmul(w01r, bi)),
                            $vmul(w01i, br),
                        );
                        let nrb = $vsub(
                            $vadd($vsub($vmul(w10r, ar), $vmul(w10i, ai)), $vmul(w11r, br)),
                            $vmul(w11i, bi),
                        );
                        let nib = $vadd(
                            $vadd($vadd($vmul(w10r, ai), $vmul(w10i, ar)), $vmul(w11r, bi)),
                            $vmul(w11i, br),
                        );
                        $vst2(pr, $vx2(nra, nrb));
                        $vst2(pi, $vx2(nia, nib));
                        h += L;
                    }
                    if h < h1 {
                        super::kern_1q_scalar(re, im, q, h, h1, m);
                    }
                }
            }

            /// Vector 1-qubit diagonal kernel; structure mirrors the
            /// 1-qubit dense kernel above.
            ///
            /// # Safety
            /// Same contract as [`super::kern_diag1_scalar`].
            pub unsafe fn $kern_diag1(
                re: *mut $ty,
                im: *mut $ty,
                q: u32,
                h0: usize,
                h1: usize,
                d0: ($ty, $ty),
                d1: ($ty, $ty),
            ) {
                const L: usize = $lanes;
                let s = 1usize << q;
                if s < L && s != 1 {
                    return $small_diag1(re, im, q, h0, h1, d0, d1);
                }
                let mask = s - 1;
                let w0r = $vdup(d0.0);
                let w0i = $vdup(d0.1);
                let w1r = $vdup(d1.0);
                let w1i = $vdup(d1.1);
                if s >= L {
                    let mut h = h0;
                    while h < h1 {
                        let run = (s - (h & mask)).min(h1 - h);
                        let a0 = ((h >> q) << (q + 1)) | (h & mask);
                        let ra = re.add(a0);
                        let ia = im.add(a0);
                        let rb = re.add(a0 + s);
                        let ib = im.add(a0 + s);
                        let mut t = 0usize;
                        while t + L <= run {
                            let ar = $vld1(ra.add(t));
                            let ai = $vld1(ia.add(t));
                            $vst1(ra.add(t), $vsub($vmul(w0r, ar), $vmul(w0i, ai)));
                            $vst1(ia.add(t), $vadd($vmul(w0r, ai), $vmul(w0i, ar)));
                            let br = $vld1(rb.add(t));
                            let bi = $vld1(ib.add(t));
                            $vst1(rb.add(t), $vsub($vmul(w1r, br), $vmul(w1i, bi)));
                            $vst1(ib.add(t), $vadd($vmul(w1r, bi), $vmul(w1i, br)));
                            t += L;
                        }
                        while t < run {
                            let (ar, ai) = (*ra.add(t), *ia.add(t));
                            *ra.add(t) = d0.0 * ar - d0.1 * ai;
                            *ia.add(t) = d0.0 * ai + d0.1 * ar;
                            let (br, bi) = (*rb.add(t), *ib.add(t));
                            *rb.add(t) = d1.0 * br - d1.1 * bi;
                            *ib.add(t) = d1.0 * bi + d1.1 * br;
                            t += 1;
                        }
                        h += run;
                    }
                } else {
                    // s == 1 (qubit 0): pairs are adjacent, [a b a b …].
                    let mut h = h0;
                    while h + L <= h1 {
                        let pr = re.add(2 * h);
                        let pi = im.add(2 * h);
                        let r2 = $vld2(pr);
                        let i2 = $vld2(pi);
                        let (ar, br) = (r2.0, r2.1);
                        let (ai, bi) = (i2.0, i2.1);
                        let nra = $vsub($vmul(w0r, ar), $vmul(w0i, ai));
                        let nia = $vadd($vmul(w0r, ai), $vmul(w0i, ar));
                        let nrb = $vsub($vmul(w1r, br), $vmul(w1i, bi));
                        let nib = $vadd($vmul(w1r, bi), $vmul(w1i, br));
                        $vst2(pr, $vx2(nra, nrb));
                        $vst2(pi, $vx2(nia, nib));
                        h += L;
                    }
                    if h < h1 {
                        super::kern_diag1_scalar(re, im, q, h, h1, d0, d1);
                    }
                }
            }

            /// Vector k-qubit diagonal kernel. Key observation: between two
            /// multiples of `2^qs[0]` no support bit changes, so the table
            /// index `j` — and the diagonal entry — is *constant* across
            /// each such run. The kernel walks runs, splats the entry once
            /// and streams the amplitudes; runs shorter than a vector
            /// (`qs[0] < log2(lanes)`) fall back to the scalar kernel.
            ///
            /// # Safety
            /// Same contract as [`super::kern_diagk_scalar`].
            pub unsafe fn $kern_diagk(
                re: *mut $ty,
                im: *mut $ty,
                qs: &[u32],
                dre: &[$ty],
                dim_: &[$ty],
                i0: usize,
                i1: usize,
            ) {
                const L: usize = $lanes;
                let s0 = 1usize << qs[0];
                if s0 < L {
                    return super::kern_diagk_scalar(re, im, qs, dre, dim_, i0, i1);
                }
                let m0 = s0 - 1;
                let mut i = i0;
                while i < i1 {
                    let run = (s0 - (i & m0)).min(i1 - i);
                    let mut j = 0usize;
                    for (b, &q) in qs.iter().enumerate() {
                        j |= ((i >> q) & 1) << b;
                    }
                    let (wr, wi) = (dre[j], dim_[j]);
                    let vwr = $vdup(wr);
                    let vwi = $vdup(wi);
                    let mut t = 0usize;
                    while t + L <= run {
                        let ar = $vld1(re.add(i + t));
                        let ai = $vld1(im.add(i + t));
                        $vst1(re.add(i + t), $vsub($vmul(vwr, ar), $vmul(vwi, ai)));
                        $vst1(im.add(i + t), $vadd($vmul(vwr, ai), $vmul(vwi, ar)));
                        t += L;
                    }
                    while t < run {
                        let (ar, ai) = (*re.add(i + t), *im.add(i + t));
                        *re.add(i + t) = wr * ar - wi * ai;
                        *im.add(i + t) = wr * ai + wi * ar;
                        t += 1;
                    }
                    i += run;
                }
            }

            /// Vector fused-gate kernel, vectorized ACROSS groups: lane
            /// `l` holds group `g + l`, gathers/scatters are scalar per
            /// lane, and the whole `2^k × 2^k` matvec runs on vectors with
            /// per-lane arithmetic order identical to the scalar kernel.
            ///
            /// # Safety
            /// Same contract as [`super::kern_kq_scalar`].
            pub unsafe fn $kern_kq(
                re: *mut $ty,
                im: *mut $ty,
                tbl: &KqTables<'_, $ty>,
                g0: usize,
                g1: usize,
            ) {
                const L: usize = $lanes;
                let dim = tbl.dim;
                let zero = $vdup(0.0);
                let mut xr = [zero; MAX_DIM];
                let mut xi = [zero; MAX_DIM];
                let mut yr = [zero; MAX_DIM];
                let mut yi = [zero; MAX_DIM];
                let mut bases = [0usize; L];
                let mut g = g0;
                while g + L <= g1 {
                    for (l, base) in bases.iter_mut().enumerate() {
                        *base = super::insert_zeros(g + l, tbl.qs);
                    }
                    for j in 0..dim {
                        let o = tbl.offs[j];
                        let mut tr: [$ty; L] = [0.0; L];
                        let mut ti: [$ty; L] = [0.0; L];
                        for l in 0..L {
                            tr[l] = *re.add(bases[l] | o);
                            ti[l] = *im.add(bases[l] | o);
                        }
                        xr[j] = $vld1(tr.as_ptr());
                        xi[j] = $vld1(ti.as_ptr());
                    }
                    for r in 0..dim {
                        let row = r * dim;
                        let mut ar = zero;
                        let mut ai = zero;
                        for c in 0..dim {
                            let wr = $vdup(tbl.mre[row + c]);
                            let wi = $vdup(tbl.mim[row + c]);
                            ar = $vsub($vadd(ar, $vmul(wr, xr[c])), $vmul(wi, xi[c]));
                            ai = $vadd($vadd(ai, $vmul(wr, xi[c])), $vmul(wi, xr[c]));
                        }
                        yr[r] = ar;
                        yi[r] = ai;
                    }
                    for j in 0..dim {
                        let o = tbl.offs[j];
                        let mut tr: [$ty; L] = [0.0; L];
                        let mut ti: [$ty; L] = [0.0; L];
                        $vst1(tr.as_mut_ptr(), yr[j]);
                        $vst1(ti.as_mut_ptr(), yi[j]);
                        for l in 0..L {
                            *re.add(bases[l] | o) = tr[l];
                            *im.add(bases[l] | o) = ti[l];
                        }
                    }
                    g += L;
                }
                if g < g1 {
                    super::kern_kq_scalar(re, im, tbl, g, g1);
                }
            }
        };
    }

    neon_kernels!(
        f32,
        4,
        vld1q_f32,
        vst1q_f32,
        vld2q_f32,
        vst2q_f32,
        float32x4x2_t,
        vdupq_n_f32,
        vmulq_f32,
        vaddq_f32,
        vsubq_f32,
        kern_1q_f32_s2,
        kern_diag1_f32_s2,
        kern_1q_f32,
        kern_diag1_f32,
        kern_diagk_f32,
        kern_kq_f32
    );

    neon_kernels!(
        f64,
        2,
        vld1q_f64,
        vst1q_f64,
        vld2q_f64,
        vst2q_f64,
        float64x2x2_t,
        vdupq_n_f64,
        vmulq_f64,
        vaddq_f64,
        vsubq_f64,
        super::kern_1q_scalar,    // unreachable: every s < 2 is s == 1
        super::kern_diag1_scalar, // unreachable, as above
        kern_1q_f64,
        kern_diag1_f64,
        kern_diagk_f64,
        kern_kq_f64
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{build, GateMatrix};

    fn mat(name: &str) -> Vec<C64> {
        match build(name, &[]).unwrap() {
            GateMatrix::Unitary(m) => m,
            GateMatrix::Diagonal(d) => {
                let dim = d.len();
                let mut m = vec![C64::default(); dim * dim];
                for (j, v) in d.iter().enumerate() {
                    m[j * dim + j] = *v;
                }
                m
            }
        }
    }

    #[test]
    fn insert_zeros_places_free_bits() {
        // qs = {1, 3}: free positions are {0, 2}
        assert_eq!(insert_zeros(0b00, &[1, 3]), 0b0000);
        assert_eq!(insert_zeros(0b01, &[1, 3]), 0b0001);
        assert_eq!(insert_zeros(0b10, &[1, 3]), 0b0100);
        assert_eq!(insert_zeros(0b11, &[1, 3]), 0b0101);
    }

    #[test]
    fn hadamard_then_measure_probabilities() {
        let mut st = StateVec::<f64>::zero_state(3);
        apply_1q(&mut st, 1, &mat("h"));
        assert!((prob_one(&st, 1) - 0.5).abs() < 1e-12);
        assert!(prob_one(&st, 0) < 1e-12);
        assert!((st.norm_sqr() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bell_state_via_kq_cx() {
        let mut st = StateVec::<f64>::zero_state(2);
        apply_1q(&mut st, 0, &mat("h"));
        // cx control=0, target=1: qubits sorted [0,1], matrix bit0=q0=control
        apply_kq(&mut st, &[0, 1], &mat("cx"));
        let a = st.to_c64();
        let s = std::f64::consts::FRAC_1_SQRT_2;
        assert!((a[0] - C64::new(s, 0.0)).norm() < 1e-12);
        assert!((a[3] - C64::new(s, 0.0)).norm() < 1e-12);
        assert!(a[1].norm() < 1e-12 && a[2].norm() < 1e-12);
    }

    #[test]
    fn diag_matches_dense() {
        // CZ as diagonal vs as dense unitary on random-ish state
        let mut a = StateVec::<f64>::zero_state(3);
        let mut b = a.clone();
        for q in 0..3 {
            apply_1q(&mut a, q, &mat("h"));
            apply_1q(&mut b, q, &mat("h"));
        }
        apply_1q(&mut a, 2, &mat("t"));
        apply_1q(&mut b, 2, &mat("t"));
        let cz_diag = match build("cz", &[]).unwrap() {
            GateMatrix::Diagonal(d) => d,
            _ => unreachable!(),
        };
        apply_diag(&mut a, &[0, 2], &cz_diag);
        apply_kq(&mut b, &[0, 2], &mat("cz"));
        for i in 0..8 {
            assert!((a.amp(i) - b.amp(i)).norm() < 1e-12);
        }
    }

    #[test]
    fn apply_cx_matches_dense() {
        for (c, t) in [(0u32, 2u32), (2, 0), (1, 2), (2, 1)] {
            let mut a = StateVec::<f64>::zero_state(3);
            for q in 0..3 {
                apply_1q(&mut a, q, &mat("h"));
            }
            apply_1q(&mut a, 1, &mat("t"));
            let mut b = a.clone();
            apply_cx(&mut a, c, t);
            let (qs, m) = crate::compiler::permute_unitary_to_sorted(&[c, t], &mat("cx"));
            apply_kq(&mut b, &qs, &m);
            for i in 0..8 {
                assert!((a.amp(i) - b.amp(i)).norm() < 1e-12, "cx({c},{t}) idx {i}");
            }
        }
    }

    #[test]
    fn collapse_renormalizes() {
        let mut st = StateVec::<f64>::zero_state(2);
        apply_1q(&mut st, 0, &mat("h"));
        apply_kq(&mut st, &[0, 1], &mat("cx"));
        let p1 = prob_one(&st, 0);
        collapse(&mut st, 0, true, p1);
        assert!((st.norm_sqr() - 1.0).abs() < 1e-12);
        // After collapsing qubit 0 to 1, the Bell state is |11⟩.
        assert!((st.amp(3).norm() - 1.0).abs() < 1e-12);
    }

    // --------------------------------------------------------------------
    // SIMD bit-exactness: the dispatched (NEON on aarch64) kernels must
    // reproduce the retained scalar kernels bit-for-bit (`to_bits`, not
    // tolerance) — random states, every target qubit / sorted support,
    // both precisions, and odd task boundaries that force vector tails.
    // --------------------------------------------------------------------

    /// splitmix64 — deterministic, dependency-free test RNG.
    struct Sm64(u64);

    impl Sm64 {
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }

        /// Uniform in [-1, 1).
        fn sym(&mut self) -> f64 {
            ((self.next_u64() >> 11) as f64 / (1u64 << 52) as f64) - 1.0
        }

        fn c64(&mut self) -> C64 {
            C64::new(self.sym(), self.sym())
        }
    }

    fn rand_state<T: Real>(n: u32, rng: &mut Sm64) -> StateVec<T> {
        let len = 1usize << n;
        StateVec {
            re: (0..len).map(|_| T::fr(rng.sym())).collect(),
            im: (0..len).map(|_| T::fr(rng.sym())).collect(),
            n_qubits: n,
        }
    }

    #[track_caller]
    fn assert_bits<T: Real>(got: &StateVec<T>, want: &StateVec<T>, ctx: &str) {
        for i in 0..want.len() {
            assert_eq!(
                got.re[i].to64().to_bits(),
                want.re[i].to64().to_bits(),
                "re[{i}] differs: {ctx}"
            );
            assert_eq!(
                got.im[i].to64().to_bits(),
                want.im[i].to64().to_bits(),
                "im[{i}] differs: {ctx}"
            );
        }
    }

    /// Odd split points: forces vector kernels to start/stop mid-run and
    /// exercise every scalar tail (`total` up to 12 qubits of halves).
    fn odd_ranges(total: usize) -> Vec<(usize, usize)> {
        let mut cuts = vec![0usize];
        let mut x = 1usize;
        while x < total {
            cuts.push(x);
            x = x * 3 + 1; // 1, 4, 13, 40, 121, 364, 1093, …
        }
        cuts.push(total);
        cuts.dedup();
        cuts.windows(2)
            .map(|w| (w[0], w[1]))
            .filter(|(a, b)| a < b)
            .collect()
    }

    /// All ascending k-subsets of 0..n.
    fn supports(n: u32, k: usize) -> Vec<Vec<u32>> {
        (0u32..1 << n)
            .filter(|m| m.count_ones() as usize == k)
            .map(|m| (0..n).filter(|&b| (m >> b) & 1 == 1).collect())
            .collect()
    }

    fn conv2x2<T: Real>(m: &[C64]) -> [(T, T); 4] {
        [
            (T::fr(m[0].re), T::fr(m[0].im)),
            (T::fr(m[1].re), T::fr(m[1].im)),
            (T::fr(m[2].re), T::fr(m[2].im)),
            (T::fr(m[3].re), T::fr(m[3].im)),
        ]
    }

    fn check_1q_bits<T: Real>() {
        let mut rng = Sm64(0x1A01);
        for n in 1..=12u32 {
            let base = rand_state::<T>(n, &mut rng);
            let half = base.len() >> 1;
            for q in 0..n {
                let m: Vec<C64> = (0..4).map(|_| rng.c64()).collect();
                let mm = conv2x2::<T>(&m);
                let mut want = base.clone();
                unsafe {
                    kern_1q_scalar(want.re.as_mut_ptr(), want.im.as_mut_ptr(), q, 0, half, &mm);
                }
                // Dispatched path over the full range (what apply_1q runs).
                let mut got = base.clone();
                apply_1q(&mut got, q, &m);
                assert_bits(&got, &want, &format!("apply_1q n={n} q={q}"));
                // Dispatched path across odd task boundaries.
                let mut got = base.clone();
                for (h0, h1) in odd_ranges(half) {
                    unsafe {
                        kern_1q(got.re.as_mut_ptr(), got.im.as_mut_ptr(), q, h0, h1, &mm);
                    }
                }
                assert_bits(&got, &want, &format!("kern_1q odd ranges n={n} q={q}"));
            }
        }
    }

    #[test]
    fn simd_1q_bit_exact_f32() {
        check_1q_bits::<f32>();
    }

    #[test]
    fn simd_1q_bit_exact_f64() {
        check_1q_bits::<f64>();
    }

    fn check_diag_bits<T: Real>() {
        let mut rng = Sm64(0xD1A6);
        for n in 1..=12u32 {
            let base = rand_state::<T>(n, &mut rng);
            let len = base.len();
            // k == 1 run-walk path, every qubit.
            for q in 0..n {
                let d: Vec<C64> = (0..2).map(|_| rng.c64()).collect();
                let d0 = (T::fr(d[0].re), T::fr(d[0].im));
                let d1 = (T::fr(d[1].re), T::fr(d[1].im));
                let mut want = base.clone();
                unsafe {
                    kern_diag1_scalar(
                        want.re.as_mut_ptr(),
                        want.im.as_mut_ptr(),
                        q,
                        0,
                        len >> 1,
                        d0,
                        d1,
                    );
                }
                let mut got = base.clone();
                apply_diag(&mut got, &[q], &d);
                assert_bits(&got, &want, &format!("apply_diag k=1 n={n} q={q}"));
                let mut got = base.clone();
                for (h0, h1) in odd_ranges(len >> 1) {
                    unsafe {
                        kern_diag1(got.re.as_mut_ptr(), got.im.as_mut_ptr(), q, h0, h1, d0, d1);
                    }
                }
                assert_bits(&got, &want, &format!("kern_diag1 odd ranges n={n} q={q}"));
            }
            // General path, every sorted support of every k (incl. k == n).
            for k in 2..=MAX_FUSED_QUBITS.min(n as usize) {
                let d: Vec<C64> = (0..1usize << k).map(|_| rng.c64()).collect();
                let dre: Vec<T> = d.iter().map(|z| T::fr(z.re)).collect();
                let dim_: Vec<T> = d.iter().map(|z| T::fr(z.im)).collect();
                for qs in supports(n, k) {
                    let mut want = base.clone();
                    unsafe {
                        kern_diagk_scalar(
                            want.re.as_mut_ptr(),
                            want.im.as_mut_ptr(),
                            &qs,
                            &dre,
                            &dim_,
                            0,
                            len,
                        );
                    }
                    let mut got = base.clone();
                    apply_diag(&mut got, &qs, &d);
                    assert_bits(&got, &want, &format!("apply_diag n={n} qs={qs:?}"));
                    let mut got = base.clone();
                    for (i0, i1) in odd_ranges(len) {
                        unsafe {
                            kern_diagk(
                                got.re.as_mut_ptr(),
                                got.im.as_mut_ptr(),
                                &qs,
                                &dre,
                                &dim_,
                                i0,
                                i1,
                            );
                        }
                    }
                    assert_bits(
                        &got,
                        &want,
                        &format!("kern_diagk odd ranges n={n} qs={qs:?}"),
                    );
                }
            }
        }
    }

    #[test]
    fn simd_diag_bit_exact_f32() {
        check_diag_bits::<f32>();
    }

    #[test]
    fn simd_diag_bit_exact_f64() {
        check_diag_bits::<f64>();
    }

    fn check_kq_bits<T: Real>() {
        let mut rng = Sm64(0x4B71);
        for n in 2..=12u32 {
            let base = rand_state::<T>(n, &mut rng);
            for k in 2..=MAX_FUSED_QUBITS.min(n as usize) {
                let dim = 1usize << k;
                let m: Vec<C64> = (0..dim * dim).map(|_| rng.c64()).collect();
                let mre: Vec<T> = m.iter().map(|z| T::fr(z.re)).collect();
                let mim: Vec<T> = m.iter().map(|z| T::fr(z.im)).collect();
                let all = supports(n, k);
                let last = all.len() - 1;
                for (si, qs) in all.iter().enumerate() {
                    let offs: Vec<usize> = (0..dim)
                        .map(|j| {
                            let mut o = 0usize;
                            for (b, &q) in qs.iter().enumerate() {
                                if (j >> b) & 1 == 1 {
                                    o |= 1usize << q;
                                }
                            }
                            o
                        })
                        .collect();
                    let tbl = KqTables {
                        qs,
                        offs: &offs,
                        mre: &mre,
                        mim: &mim,
                        dim,
                    };
                    let groups = base.len() >> k;
                    let mut want = base.clone();
                    unsafe {
                        kern_kq_scalar(want.re.as_mut_ptr(), want.im.as_mut_ptr(), &tbl, 0, groups);
                    }
                    let mut got = base.clone();
                    apply_kq(&mut got, qs, &m);
                    assert_bits(&got, &want, &format!("apply_kq n={n} qs={qs:?}"));
                    // Odd group boundaries (vector tails): sampled to keep
                    // the exhaustive sweep fast, always incl. first/last.
                    if si % 5 == 0 || si == last {
                        let mut got = base.clone();
                        for (g0, g1) in odd_ranges(groups) {
                            unsafe {
                                kern_kq(got.re.as_mut_ptr(), got.im.as_mut_ptr(), &tbl, g0, g1);
                            }
                        }
                        assert_bits(&got, &want, &format!("kern_kq odd ranges n={n} qs={qs:?}"));
                    }
                }
            }
        }
    }

    #[test]
    fn simd_kq_bit_exact_f32() {
        check_kq_bits::<f32>();
    }

    #[test]
    fn simd_kq_bit_exact_f64() {
        check_kq_bits::<f64>();
    }
}
