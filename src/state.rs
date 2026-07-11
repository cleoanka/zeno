//! State vector storage and gate kernels.
//!
//! Amplitudes are stored *split* (structure-of-arrays: one array of reals,
//! one of imaginaries). On Apple Silicon this lets LLVM emit clean NEON
//! FMA loops without shuffle traffic, and it halves the working set per
//! stream, which matters because these kernels are memory-bound.
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

/// # Safety
/// `[h0, h1)` ranges across calls must partition `0..len/2`; each half
/// index maps to a unique disjoint amplitude pair.
unsafe fn kern_1q<T: Real>(re: *mut T, im: *mut T, q: u32, h0: usize, h1: usize, m: &[(T, T); 4]) {
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
    let re = SendPtr(st.re.as_mut_ptr());
    let im = SendPtr(st.im.as_mut_ptr());
    task_ranges(groups)
        .into_par_iter()
        .for_each(|(g0, g1)| unsafe {
            let re = re.ptr();
            let im = im.ptr();
            let mut xr = [T::zero(); MAX_DIM];
            let mut xi = [T::zero(); MAX_DIM];
            let mut yr = [T::zero(); MAX_DIM];
            let mut yi = [T::zero(); MAX_DIM];
            for g in g0..g1 {
                let base = insert_zeros(g, qs);
                for j in 0..dim {
                    let p = base | offs[j];
                    xr[j] = *re.add(p);
                    xi[j] = *im.add(p);
                }
                for r in 0..dim {
                    let row = r * dim;
                    let mut ar = T::zero();
                    let mut ai = T::zero();
                    for c in 0..dim {
                        let (wr, wi) = (mre[row + c], mim[row + c]);
                        ar = ar + wr * xr[c] - wi * xi[c];
                        ai = ai + wr * xi[c] + wi * xr[c];
                    }
                    yr[r] = ar;
                    yi[r] = ai;
                }
                for j in 0..dim {
                    let p = base | offs[j];
                    *re.add(p) = yr[j];
                    *im.add(p) = yi[j];
                }
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
                let s = 1usize << q;
                let mask = s - 1;
                let mut h = h0;
                while h < h1 {
                    let run = (s - (h & mask)).min(h1 - h);
                    let a0 = ((h >> q) << (q + 1)) | (h & mask);
                    let ra = std::slice::from_raw_parts_mut(re.ptr().add(a0), run);
                    let ia = std::slice::from_raw_parts_mut(im.ptr().add(a0), run);
                    let rb = std::slice::from_raw_parts_mut(re.ptr().add(a0 + s), run);
                    let ib = std::slice::from_raw_parts_mut(im.ptr().add(a0 + s), run);
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
        let re = re.ptr();
        let im = im.ptr();
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
    });
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
}
