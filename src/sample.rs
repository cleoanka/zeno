//! Shot sampling from a final state vector.
//!
//! Strategy: one parallel pass computes per-chunk probability masses; the
//! sorted uniform draws are then routed to their chunks and resolved by a
//! second parallel pass that scans each chunk once. Total work is
//! O(2^n + shots·log shots) regardless of shot count — no per-shot
//! re-simulation and no full prefix-sum materialization.

use crate::state::{Real, StateVec};
use rand::Rng;
use rayon::prelude::*;

const CHUNK: usize = 1 << 16;

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

/// Draw `shots` basis-state indices from |ψ|². Deterministic given `rng`
/// state (the parallel phase writes to disjoint, pre-computed slots).
/// Returned indices are in ascending order (fine for counting).
pub fn sample_indices<T: Real>(st: &StateVec<T>, shots: usize, rng: &mut impl Rng) -> Vec<u64> {
    let n = st.len();
    if shots == 0 {
        return vec![];
    }
    let chunk = n.min(CHUNK);
    let nchunks = n.div_ceil(chunk);

    // Pass 1: probability mass per chunk (f64 accumulation).
    let sums: Vec<f64> = (0..nchunks)
        .into_par_iter()
        .map(|c| {
            let lo = c * chunk;
            let hi = (lo + chunk).min(n);
            let mut acc = 0.0f64;
            for i in lo..hi {
                let (r, im) = (st.re[i].to64(), st.im[i].to64());
                acc += r * r + im * im;
            }
            acc
        })
        .collect();
    let mut cum_before = vec![0.0f64; nchunks + 1];
    for c in 0..nchunks {
        cum_before[c + 1] = cum_before[c] + sums[c];
    }
    let total = cum_before[nchunks];

    // Sorted draws, scaled by the actual total mass so rounding in the
    // chunk sums can never push a draw past the last chunk.
    let mut us: Vec<f64> = (0..shots).map(|_| rng.gen::<f64>() * total).collect();
    us.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    // Route draws to chunks.
    let starts: Vec<usize> = (0..=nchunks)
        .map(|c| us.partition_point(|u| *u < cum_before[c]))
        .collect();

    let mut out = vec![0u64; shots];
    let out_ptr = SendPtr(out.as_mut_ptr());
    (0..nchunks)
        .into_par_iter()
        .filter(|c| starts[*c] < starts[*c + 1])
        .for_each(|c| unsafe {
            let lo = c * chunk;
            let hi = (lo + chunk).min(n);
            let (s0, s1) = (starts[c], starts[c + 1]);
            let mut acc = cum_before[c];
            let mut ptr = s0;
            'scan: for i in lo..hi {
                let (r, im) = (st.re[i].to64(), st.im[i].to64());
                acc += r * r + im * im;
                while ptr < s1 && us[ptr] < acc {
                    *out_ptr.ptr().add(ptr) = i as u64;
                    ptr += 1;
                }
                if ptr == s1 {
                    break 'scan;
                }
            }
            // Floating-point edge: any unresolved draws land on the last
            // index of the chunk.
            while ptr < s1 {
                *out_ptr.ptr().add(ptr) = (hi - 1) as u64;
                ptr += 1;
            }
        });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gates::{build, GateMatrix};
    use crate::state::{apply_1q, apply_kq};
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn h() -> Vec<crate::ir::C64> {
        match build("h", &[]).unwrap() {
            GateMatrix::Unitary(m) => m,
            _ => unreachable!(),
        }
    }

    #[test]
    fn bell_sampling_is_balanced_and_correlated() {
        let mut st = StateVec::<f64>::zero_state(2);
        apply_1q(&mut st, 0, &h());
        let cx = match build("cx", &[]).unwrap() {
            GateMatrix::Unitary(m) => m,
            _ => unreachable!(),
        };
        apply_kq(&mut st, &[0, 1], &cx);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let samples = sample_indices(&st, 100_000, &mut rng);
        let ones = samples.iter().filter(|&&s| s == 3).count();
        let zeros = samples.iter().filter(|&&s| s == 0).count();
        assert_eq!(ones + zeros, 100_000, "only |00> and |11> may appear");
        let frac = ones as f64 / 100_000.0;
        assert!((frac - 0.5).abs() < 0.01, "frac={frac}");
    }

    #[test]
    fn deterministic_given_seed() {
        let mut st = StateVec::<f64>::zero_state(4);
        for q in 0..4 {
            apply_1q(&mut st, q, &h());
        }
        let mut r1 = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(42);
        assert_eq!(
            sample_indices(&st, 1000, &mut r1),
            sample_indices(&st, 1000, &mut r2)
        );
    }
}
