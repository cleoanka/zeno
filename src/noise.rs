//! Trajectory-sampled noise channels.
//!
//! Method: **stochastic quantum trajectories** on the state vector (what
//! qiskit-aer calls per-shot noise sampling). Each modeled channel is
//! unraveled exactly into stochastic pure-state updates, so averaging over
//! shots reproduces the channel's density-matrix action *exactly* — no
//! density matrices are ever materialized and no channel is truncated;
//! statistical (shot) error is the only error.
//!
//! Channels attach to **compiled-as-written** gates, so noisy runs force
//! gate fusion off (the executor compiles with `fusion_max = 0` and pushes
//! a notice). After each gate op the executor calls [`apply_gate_noise`];
//! at every measurement it passes the collapsed bit through
//! [`readout_flip`]. All randomness comes from the per-shot Xoshiro stream
//! (`splitmix64(seed ^ shot)`), so noisy runs are seed-reproducible.
//!
//! See `docs/NOISE.md` for the full semantics: channel order, the
//! amplitude-damping trajectory math, the frozen RNG stream layout, and
//! the analytic values pinned by `tests/noise.rs`.

use crate::exec::Backend;
use crate::ir::C64;
use crate::Error;
use rand::Rng;

/// Per-gate and readout error probabilities, all in `[0, 1]`.
///
/// Gate channels fire after every executed gate; readout flips apply to
/// every recorded measurement bit (mid-circuit and final). A default
/// (all-zero) model is *trivial*: the executor treats it as no noise at
/// all and keeps the ideal fused/sampled fast path.
///
/// `depolarizing_1q` uses the standard "with probability `p` apply a
/// uniform non-identity Pauli" convention, so `p = 3/4` is already the
/// fully depolarizing channel; values up to 1 are allowed and remain
/// well-defined under the same convention.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NoiseModel {
    pub depolarizing_1q: f64,   // after each 1-qubit gate
    pub depolarizing_2q: f64,   // after each >=2-qubit gate (see semantics)
    pub bit_flip: f64,          // X error, per touched qubit, after each gate
    pub phase_flip: f64,        // Z error, per touched qubit, after each gate
    pub amplitude_damping: f64, // gamma, per touched qubit, after each gate
    pub readout_flip_0to1: f64, // P(record 1 | true 0) at each measurement
    pub readout_flip_1to0: f64, // P(record 0 | true 1)
}

impl NoiseModel {
    fn fields(&self) -> [(&'static str, f64); 7] {
        [
            ("depolarizing_1q", self.depolarizing_1q),
            ("depolarizing_2q", self.depolarizing_2q),
            ("bit_flip", self.bit_flip),
            ("phase_flip", self.phase_flip),
            ("amplitude_damping", self.amplitude_damping),
            ("readout_flip_0to1", self.readout_flip_0to1),
            ("readout_flip_1to0", self.readout_flip_1to0),
        ]
    }

    /// Parse and validate a model from JSON. Every field is optional and
    /// defaults to 0; unknown fields are rejected (typos must not silently
    /// produce an ideal simulation).
    pub fn from_json(s: &str) -> Result<Self, Error> {
        let m: NoiseModel = serde_json::from_str(s).map_err(|e| Error::Noise(e.to_string()))?;
        m.validate()?;
        Ok(m)
    }

    /// True when every probability is exactly 0 — the model is a no-op and
    /// the executor uses the ideal (fused, analytically sampled) path.
    pub fn is_trivial(&self) -> bool {
        self.fields().iter().all(|&(_, v)| v == 0.0)
    }

    /// Every field must be a probability in `[0, 1]` (NaN is rejected).
    pub fn validate(&self) -> Result<(), Error> {
        for (name, v) in self.fields() {
            if !(0.0..=1.0).contains(&v) {
                return Err(Error::Noise(format!(
                    "{name} = {v} is not a probability in [0, 1]"
                )));
            }
        }
        Ok(())
    }
}

/// Apply a single Pauli by 2-bit code: 0 = I, 1 = X, 2 = Y, 3 = Z.
fn apply_pauli(be: &mut dyn Backend, q: u32, code: u64) {
    let one = C64::new(1.0, 0.0);
    let zero = C64::default();
    match code {
        1 => be.apply_unitary(&[q], &[zero, one, one, zero]),
        2 => be.apply_unitary(&[q], &[zero, C64::new(0.0, -1.0), C64::new(0.0, 1.0), zero]),
        3 => be.apply_diagonal(&[q], &[one, C64::new(-1.0, 0.0)]),
        _ => {}
    }
}

/// One trajectory step of the amplitude-damping channel with parameter
/// `gamma` on qubit `q`, driven by the uniform draw `u ∈ [0, 1)`.
///
/// Kraus operators: K0 = diag(1, √(1−γ)), K1 = √γ·|0⟩⟨1|.
/// - P(jump) = ⟨ψ|K1†K1|ψ⟩ = γ·P(1), with P(1) = `backend.prob_one(q)`.
/// - Jump (`u < P(jump)`): new state K1|ψ⟩/‖K1|ψ⟩‖, implemented as
///   "collapse to outcome 1, then apply X" (the √γ scalar normalizes
///   away). `measure(q, 0.0)` always collapses to 1 here because
///   `u < γ·P(1)` with `u ≥ 0` implies `P(1) > 0`.
/// - No-jump: new state K0|ψ⟩/‖K0|ψ⟩‖ with ‖K0|ψ⟩‖² = 1 − γ·P(1)
///   = 1 − P(jump); the renormalization is folded into the diagonal so a
///   single sweep applies diag(1, √(1−γ))/√(1−P(jump)) and ‖ψ‖ stays 1.
fn amplitude_damp(be: &mut dyn Backend, q: u32, gamma: f64, u: f64) {
    let p_jump = gamma * be.prob_one(q);
    if u < p_jump {
        let collapsed_to_one = be.measure(q, 0.0);
        debug_assert!(collapsed_to_one, "damping jump needs nonzero |1⟩ mass");
        apply_pauli(be, q, 1);
    } else {
        let s = 1.0 / (1.0 - p_jump).sqrt();
        be.apply_diagonal(
            &[q],
            &[C64::new(s, 0.0), C64::new(s * (1.0 - gamma).sqrt(), 0.0)],
        );
    }
}

/// Apply the per-gate noise channels after a gate that touched `qubits`
/// (the compiled, ascending-sorted support of the op).
///
/// Order (frozen; documented in docs/NOISE.md):
/// 1. Depolarizing — `depolarizing_1q` for 1-qubit gates, `depolarizing_2q`
///    for k ≥ 2 qubits: one uniform draw; on a hit, one uniform integer in
///    `[1, 4^k)` picks a non-identity Pauli string (qubit `qubits[b]` gets
///    the Pauli coded by bits `2b..2b+2`; each of the `4^k − 1` strings has
///    probability `p / (4^k − 1)`).
/// 2. For each touched qubit in ascending order: bit flip (X w.p. `p`),
///    phase flip (Z w.p. `p`), amplitude damping (see [`amplitude_damp`]).
///
/// RNG stream layout (frozen so seeds stay stable across releases):
/// channels with probability 0 consume **no** draws; each active channel
/// consumes exactly one `f64` draw per opportunity, plus, for a
/// depolarizing hit, one `gen_range(1..4^k)` integer draw.
pub fn apply_gate_noise<R: Rng>(
    be: &mut dyn Backend,
    model: &NoiseModel,
    qubits: &[u32],
    rng: &mut R,
) {
    let k = qubits.len() as u32;
    let p_dep = if k == 1 {
        model.depolarizing_1q
    } else {
        model.depolarizing_2q
    };
    if p_dep > 0.0 && rng.gen::<f64>() < p_dep {
        let idx = rng.gen_range(1..1u64 << (2 * k));
        for (b, &q) in qubits.iter().enumerate() {
            apply_pauli(be, q, (idx >> (2 * b)) & 3);
        }
    }
    for &q in qubits {
        if model.bit_flip > 0.0 && rng.gen::<f64>() < model.bit_flip {
            apply_pauli(be, q, 1);
        }
        if model.phase_flip > 0.0 && rng.gen::<f64>() < model.phase_flip {
            apply_pauli(be, q, 3);
        }
        if model.amplitude_damping > 0.0 {
            let u = rng.gen::<f64>();
            amplitude_damp(be, q, model.amplitude_damping, u);
        }
    }
}

/// Classical readout error: flip the RECORDED bit (0→1 w.p.
/// `readout_flip_0to1`, 1→0 w.p. `readout_flip_1to0`). The collapsed
/// quantum state is untouched — this is a purely classical error on the
/// stored clbit, which is also the value classical control (`if`) sees.
pub fn readout_flip<R: Rng>(model: &NoiseModel, bit: bool, rng: &mut R) -> bool {
    let p = if bit {
        model.readout_flip_1to0
    } else {
        model.readout_flip_0to1
    };
    if p > 0.0 && rng.gen::<f64>() < p {
        !bit
    } else {
        bit
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::exec::CpuBackend;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn default_is_trivial_and_valid() {
        let m = NoiseModel::default();
        assert!(m.is_trivial());
        m.validate().unwrap();
    }

    #[test]
    fn validate_bounds() {
        let mk = |v| NoiseModel {
            depolarizing_1q: v,
            ..Default::default()
        };
        mk(0.0).validate().unwrap();
        mk(1.0).validate().unwrap(); // up to 1 allowed (see struct docs)
        for bad in [-0.1, 1.5, f64::NAN] {
            let err = mk(bad).validate().unwrap_err().to_string();
            assert!(
                err.contains("depolarizing_1q") && err.contains("[0, 1]"),
                "value {bad}: got '{err}'"
            );
        }
        let err = NoiseModel {
            readout_flip_1to0: 1.0001,
            ..Default::default()
        }
        .validate()
        .unwrap_err()
        .to_string();
        assert!(err.contains("readout_flip_1to0"), "got '{err}'");
    }

    #[test]
    fn from_json_defaults_unknown_fields_and_range() {
        assert!(NoiseModel::from_json("{}").unwrap().is_trivial());
        let m = NoiseModel::from_json(r#"{"bit_flip": 0.25}"#).unwrap();
        assert_eq!(m.bit_flip, 0.25);
        assert_eq!(m.phase_flip, 0.0);

        let err = NoiseModel::from_json(r#"{"bitflip": 0.25}"#)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("invalid noise model") && err.contains("bitflip"),
            "unknown field must be named: '{err}'"
        );

        let err = NoiseModel::from_json(r#"{"phase_flip": 1.5}"#)
            .unwrap_err()
            .to_string();
        assert!(err.contains("phase_flip"), "range error: '{err}'");
    }

    #[test]
    fn pauli_string_codes_map_to_qubits() {
        // idx = 0b0111: qubit 0 gets code 3 (Z), qubit 1 gets code 1 (X).
        // On |00⟩: Z0 is a no-op, X1 flips qubit 1 → |10⟩ = index 2.
        let mut be = CpuBackend::<f64>::new(2);
        let idx = 0b0111u64;
        for (b, &q) in [0u32, 1u32].iter().enumerate() {
            apply_pauli(&mut be, q, (idx >> (2 * b)) & 3);
        }
        let sv = be.statevector();
        assert!((sv[2].re - 1.0).abs() < 1e-12, "expected |10⟩, got {sv:?}");
    }

    #[test]
    fn damping_jump_and_no_jump_preserve_norm() {
        // |+⟩ exercises both branches; the norm must stay 1 either way.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        let had = [
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(h, 0.0),
            C64::new(-h, 0.0),
        ];
        for seed in 0..32u64 {
            let mut be = CpuBackend::<f64>::new(1);
            be.apply_unitary(&[0], &had);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
            let u = rng.gen::<f64>();
            amplitude_damp(&mut be, 0, 0.6, u);
            let n = be.norm_sqr();
            assert!((n - 1.0).abs() < 1e-12, "seed {seed}: norm² = {n}");
        }
    }

    #[test]
    fn readout_flip_only_touches_the_matching_value() {
        let m = NoiseModel {
            readout_flip_1to0: 1.0,
            ..Default::default()
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        assert!(!readout_flip(&m, true, &mut rng), "1 must flip to 0 at p=1");
        assert!(!readout_flip(&m, false, &mut rng), "0 must stay 0");
    }
}
