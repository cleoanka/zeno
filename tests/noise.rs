//! Acceptance tests for trajectory-sampled noise (src/noise.rs +
//! docs/NOISE.md).
//!
//! Every statistical bound is 6σ of the exact binomial, so a failure is
//! evidence of a real defect, not shot noise. Every test is seeded. Each
//! analytic value is derived in a comment next to its assertion; the
//! composite-model cross-check uses qiskit-aer as an independent oracle
//! (skipped with a message when /tmp/qk-venv is absent).

use std::collections::HashMap;
use std::io::Write;
use std::process::{Command, Stdio};

use zeno::exec::CpuBackend;
use zeno::noise::apply_gate_noise;
use zeno::{Backend, Circuit, Counts, NoiseModel, RunOptions, RunResult, Simulator};

/// Assert `count` is within 6σ of Binomial(shots, p).
fn six_sigma(count: u64, shots: u64, p: f64, what: &str) {
    let mean = shots as f64 * p;
    let sigma = (shots as f64 * p * (1.0 - p)).sqrt();
    let dev = (count as f64 - mean).abs();
    assert!(
        dev <= 6.0 * sigma,
        "{what}: count {count}, mean {mean:.1}, deviation {dev:.1} > 6σ = {:.1}",
        6.0 * sigma
    );
}

fn run_noisy(c: &Circuit, model: NoiseModel, shots: u64, seed: u64) -> RunResult {
    zeno::run_program(
        &c.to_program(),
        &RunOptions {
            shots,
            seed: Some(seed),
            noise: Some(model),
            ..Default::default()
        },
    )
    .unwrap()
}

// ------------------------------------------------------ single channels

#[test]
fn bit_flip_after_x_flips_the_measured_bit() {
    // x |0⟩ = |1⟩; the bit-flip channel then applies X w.p. p → P(0) = p.
    let mut c = Circuit::new(1);
    c.x(0).measure(0, 0);
    let p = 0.1;
    let shots = 40_000;
    let r = run_noisy(
        &c,
        NoiseModel {
            bit_flip: p,
            ..Default::default()
        },
        shots,
        11,
    );
    assert_eq!(r.counts.total(), shots);
    six_sigma(r.counts.get("0"), shots, p, "bit_flip P(0)");
}

#[test]
fn phase_flip_is_visible_in_the_hadamard_basis() {
    // h |0⟩ = |+⟩; a Z error (w.p. p) turns it into |−⟩; the second h maps
    // |+⟩→|0⟩, |−⟩→|1⟩, so P(1) = p. The phase flip fired after the second
    // h acts on a computational basis state and cannot change P(1). The
    // barrier stops the compiler from cancelling the h h pair (noise
    // attaches to compiled-as-written gates — docs/NOISE.md).
    let mut c = Circuit::new(1);
    c.h(0).barrier().h(0).measure(0, 0);
    let p = 0.15;
    let shots = 40_000;
    let r = run_noisy(
        &c,
        NoiseModel {
            phase_flip: p,
            ..Default::default()
        },
        shots,
        12,
    );
    six_sigma(r.counts.get("1"), shots, p, "phase_flip P(1) in X basis");
}

#[test]
fn depolarizing_1q_flips_two_thirds_of_the_time() {
    // z |0⟩ = |0⟩ (kept as a compiled gate). Depolarizing then applies
    // X, Y or Z each w.p. p/3: X and Y flip the measured bit, Z does not,
    // so P(1) = 2p/3.
    let mut c = Circuit::new(1);
    c.z(0).measure(0, 0);
    let p = 0.3;
    let shots = 40_000;
    let r = run_noisy(
        &c,
        NoiseModel {
            depolarizing_1q: p,
            ..Default::default()
        },
        shots,
        13,
    );
    six_sigma(
        r.counts.get("1"),
        shots,
        2.0 * p / 3.0,
        "depolarizing_1q P(1) = 2p/3",
    );
}

#[test]
fn amplitude_damping_survival_is_one_minus_gamma_per_gate() {
    // x |0⟩ = |1⟩, then k−1 rz(0.0) gates (identity ops that are kept and
    // carry noise, and never cancel). After each of the k gates the
    // damping channel jumps |1⟩→|0⟩ w.p. γ (P(jump) = γ·P(1) with
    // P(1) = 1), and a jumped trajectory stays |0⟩ (P(1) = 0 ⇒ no more
    // jumps, and the no-jump renormalized K0 fixes |0⟩). Hence
    // P(1) = (1−γ)^k — this pins the no-jump renormalization chain.
    let gamma = 0.3;
    let shots = 40_000;
    for k in [1u32, 4] {
        let mut c = Circuit::new(1);
        c.x(0);
        for _ in 1..k {
            c.rz(0, 0.0);
        }
        c.measure(0, 0);
        let r = run_noisy(
            &c,
            NoiseModel {
                amplitude_damping: gamma,
                ..Default::default()
            },
            shots,
            14 + u64::from(k),
        );
        let want = (1.0 - gamma).powi(k as i32);
        six_sigma(
            r.counts.get("1"),
            shots,
            want,
            &format!("amplitude damping (1−γ)^{k}"),
        );
    }
}

#[test]
fn amplitude_damping_on_superposition_pins_renormalization() {
    // h |0⟩ = (|0⟩+|1⟩)/√2, P(1) = 1/2, so P(jump) = γ/2.
    //  - jump (γ/2): state → |0⟩, P(1) = 0.
    //  - no jump (1−γ/2): state → (|0⟩ + √(1−γ)|1⟩)/√(1−γ/2), whose
    //    P(1) = (1−γ)/(2−γ).
    // Total P(1) = (1−γ/2)·(1−γ)/(2−γ) = (1−γ)/2 — the exact channel
    // value ⟨1|E(ρ)|1⟩. Without the no-jump renormalization the sampled
    // p would instead be (1−γ/2)(1−γ)/2 (prob_one sees absolute mass of
    // an unnormalized state), which for γ = 0.5 is 0.1875 vs the correct
    // 0.25 — far outside 6σ at these shots.
    let gamma = 0.5;
    let shots = 40_000;
    let mut c = Circuit::new(1);
    c.h(0).measure(0, 0);
    let r = run_noisy(
        &c,
        NoiseModel {
            amplitude_damping: gamma,
            ..Default::default()
        },
        shots,
        15,
    );
    six_sigma(
        r.counts.get("1"),
        shots,
        (1.0 - gamma) / 2.0,
        "damped |+⟩ P(1) = (1−γ)/2",
    );
}

// -------------------------------------------------------------- readout

#[test]
fn readout_flip_applies_to_the_recorded_bit_only() {
    // x |0⟩ = |1⟩ with zero gate noise: the true outcome is always 1, so
    // P(recorded 0) = readout_flip_1to0 exactly, independent of any state
    // effect.
    let shots = 40_000;
    let mut c = Circuit::new(1);
    c.x(0).measure(0, 0);
    let r = run_noisy(
        &c,
        NoiseModel {
            readout_flip_1to0: 0.2,
            ..Default::default()
        },
        shots,
        16,
    );
    six_sigma(r.counts.get("0"), shots, 0.2, "readout 1→0");

    // Measure-only circuit: true outcome always 0 → P(recorded 1) = p01.
    let mut c0 = Circuit::new(1);
    c0.measure(0, 0);
    let r0 = run_noisy(
        &c0,
        NoiseModel {
            readout_flip_0to1: 0.15,
            ..Default::default()
        },
        shots,
        17,
    );
    six_sigma(r0.counts.get("1"), shots, 0.15, "readout 0→1");
}

#[test]
fn readout_errors_do_not_collapse_or_flip_the_state() {
    // Measure the SAME qubit twice: x |0⟩ = |1⟩ stays |1⟩ through both
    // collapses, so the two recorded bits are independent flips:
    //   P(c0, c1) = P(c0)·P(c1) with P(record 1) = 1 − p = 0.7.
    // A (wrong) implementation that flips the state instead of the record
    // would correlate them (e.g. P(c1=1 | c0=0) = 0, not 0.7).
    let p = 0.3;
    let shots = 40_000;
    let mut c = Circuit::new(2);
    c.x(0).measure(0, 0).measure(0, 1);
    let r = run_noisy(
        &c,
        NoiseModel {
            readout_flip_1to0: p,
            ..Default::default()
        },
        shots,
        18,
    );
    // Keys are "c1 c0" as one 2-bit creg: "b1b0".
    six_sigma(r.counts.get("11"), shots, 0.49, "both recorded 1");
    six_sigma(r.counts.get("01"), shots, 0.21, "c0=1, c1=0");
    six_sigma(r.counts.get("10"), shots, 0.21, "c0=0, c1=1");
    six_sigma(r.counts.get("00"), shots, 0.09, "both recorded 0");
}

#[test]
fn classical_control_sees_the_recorded_flipped_bit() {
    // x q0; measure q0 → m; if (m == 1) x q1; measure q1 → out.
    // With readout_flip_1to0 = 0.3 and no other noise:
    //   m records 1 w.p. 0.7 → the correction fires → q1 = |1⟩, whose own
    //   readout records 1 w.p. 0.7 ⇒ P("1 1") = 0.49, P("0 1") = 0.21;
    //   m records 0 w.p. 0.3 → q1 stays |0⟩ and p(0→1) = 0 ⇒
    //   P("0 0") = 0.3 and P("1 0") = 0.
    // This pins that `if` reads the RECORDED (noisy) bit — a real
    // controller acts on its own record, not on the hidden true outcome.
    use zeno::ir::{GateInstr, Instr, Program, Reg};
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
            size: 2,
        }],
        cregs: vec![
            Reg {
                name: "m".into(),
                size: 1,
            },
            Reg {
                name: "out".into(),
                size: 1,
            },
        ],
        instrs: vec![
            g("x", &[0]),
            Instr::Measure { qubit: 0, clbit: 0 },
            Instr::If {
                creg: 0,
                value: 1,
                op: Box::new(g("x", &[1])),
            },
            Instr::Measure { qubit: 1, clbit: 1 },
        ],
    };
    let shots = 40_000;
    let r = zeno::run_program(
        &p,
        &RunOptions {
            shots,
            seed: Some(19),
            noise: Some(NoiseModel {
                readout_flip_1to0: 0.3,
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    // Key layout: "out m" (last-declared creg leftmost).
    six_sigma(r.counts.get("1 1"), shots, 0.49, "fired and recorded");
    six_sigma(r.counts.get("0 1"), shots, 0.21, "fired, out misread");
    six_sigma(r.counts.get("0 0"), shots, 0.30, "not fired");
    assert_eq!(r.counts.get("1 0"), 0, "out can't be 1 if m recorded 0");
}

// -------------------------------------------------- multi-qubit channels

#[test]
fn depolarizing_2q_after_cx_changes_the_outcome_at_12_of_15() {
    // cx |00⟩ = |00⟩. On a depolarizing hit (w.p. p) one of the 15
    // non-identity 2-qubit Pauli strings applies, uniformly (p/15 each).
    // In the measured (computational) basis a string leaves |00⟩ fixed
    // iff BOTH tensor factors are diagonal, i.e. in {I, Z}: that is
    // {I,Z}⊗{I,Z} minus II → IZ, ZI, ZZ = 3 strings. The other
    // 15 − 3 = 12 contain an X or Y on at least one qubit and flip that
    // qubit. Hence P(measured ≠ 00) = p · 12/15.
    let p = 0.2;
    let shots = 40_000;
    let mut c = Circuit::new(2);
    c.cx(0, 1).measure_all();
    let r = run_noisy(
        &c,
        NoiseModel {
            depolarizing_2q: p,
            ..Default::default()
        },
        shots,
        20,
    );
    six_sigma(
        shots - r.counts.get("00"),
        shots,
        p * 12.0 / 15.0,
        "depolarizing_2q state-change rate",
    );
}

#[test]
fn bell_zz_correlation_decays_by_16_15_p_and_ignores_1q_noise() {
    // Bell pair: h(0), cx(0,1). ⟨Z0Z1⟩ = P(even parity) − P(odd parity),
    // and the ideal state has ⟨Z0Z1⟩ = 1.
    //
    // depolarizing_2q (after cx): a Pauli string σa⊗σb flips the parity of
    // |Φ±⟩ iff exactly one factor is in {X, Y} (X/Y flip a bit, I/Z do
    // not). Count over the 15 non-identity strings: a∈{X,Y}, b∈{I,Z} → 4;
    // a∈{I,Z}, b∈{X,Y} → 4; total 8 parity-flipping, 7 parity-preserving.
    // P(odd) = p·8/15 ⇒ ⟨Z0Z1⟩ = 1 − 2·(8/15)p = 1 − (16/15)p.
    //
    // depolarizing_1q fires only after the 1-qubit h — BEFORE the cx — and
    // cannot break the parity: CX maps X0→X0X1, Y0→Y0X1, Z0→Z0, so every
    // single-qubit error becomes a string with zero or two bit-flipping
    // factors → parity even. Composition with a later 2q string XORs the
    // flips, so P(odd) stays exactly p·8/15 even at heavy 1q noise: the
    // assertion holds with depolarizing_1q = 0.25, pinning that
    // depolarizing_1q attaches to 1-qubit gates only.
    let p = 0.15;
    let shots = 40_000;
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1).measure_all();
    let r = run_noisy(
        &c,
        NoiseModel {
            depolarizing_1q: 0.25,
            depolarizing_2q: p,
            ..Default::default()
        },
        shots,
        21,
    );
    let odd = r.counts.get("01") + r.counts.get("10");
    six_sigma(odd, shots, p * 8.0 / 15.0, "Bell odd-parity rate");
    let zz = 1.0 - 2.0 * odd as f64 / shots as f64;
    let want = 1.0 - 16.0 * p / 15.0;
    assert!(
        (zz - want).abs() < 0.02,
        "⟨Z0Z1⟩ = {zz:.4}, analytic {want:.4}"
    );
}

// ------------------------------------------------- executor integration

#[test]
fn noise_forces_fusion_off_and_pushes_the_notice() {
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1).measure_all();

    // Ideal at fusion 5: h + cx fuse into one op.
    let ideal = zeno::run_program(
        &c.to_program(),
        &RunOptions {
            shots: 16,
            seed: Some(1),
            fusion_max: Some(5),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(ideal.stats.output_ops, 1, "sanity: bell fuses to one op");

    // Same request + noise: fusion must be off even though fusion_max = 5.
    let noisy = zeno::run_program(
        &c.to_program(),
        &RunOptions {
            shots: 16,
            seed: Some(1),
            fusion_max: Some(5),
            noise: Some(NoiseModel {
                depolarizing_2q: 0.1,
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        noisy.stats.output_ops, 2,
        "h and cx must stay separate under noise"
    );
    assert!(
        noisy
            .notices
            .iter()
            .any(|n| n == "noise: trajectory sampling — per-shot execution, fusion disabled"),
        "missing noise notice, got {:?}",
        noisy.notices
    );
}

#[test]
fn statevector_request_with_noise_is_rejected() {
    let mut c = Circuit::new(1);
    c.h(0);
    let model = NoiseModel {
        bit_flip: 0.1,
        ..Default::default()
    };
    let err = Simulator::new()
        .noise(model.clone())
        .statevector(&c)
        .unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("mixture") && msg.contains("no single final state vector"),
        "unhelpful error: {msg}"
    );

    let err = zeno::run_program(
        &c.to_program(),
        &RunOptions {
            shots: 0,
            want_statevector: true,
            seed: Some(1),
            noise: Some(model),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, zeno::Error::InvalidCircuit(_)), "{err}");
}

#[test]
fn trivial_model_is_exactly_ideal() {
    // A Some(all-zero) model must take the ideal path: identical counts to
    // no model at the same seed, fusion intact, statevector allowed.
    let mut c = Circuit::new(3);
    c.h(0).cx(0, 1).cx(1, 2).measure_all();
    let base = RunOptions {
        shots: 5000,
        seed: Some(3),
        ..Default::default()
    };
    let with_trivial = RunOptions {
        noise: Some(NoiseModel::default()),
        ..base.clone()
    };
    let a = zeno::run_program(&c.to_program(), &base).unwrap();
    let b = zeno::run_program(&c.to_program(), &with_trivial).unwrap();
    assert_eq!(a.counts, b.counts);

    let mut sv_c = Circuit::new(1);
    sv_c.h(0);
    let sv = Simulator::new()
        .noise(NoiseModel::default())
        .statevector(&sv_c)
        .unwrap();
    assert_eq!(sv.len(), 2);
}

#[test]
fn noisy_runs_are_deterministic_across_seeds_and_threads() {
    let mut c = zeno::random_circuit(5, 6, 99);
    c.measure_all();
    let model = NoiseModel {
        depolarizing_1q: 0.02,
        depolarizing_2q: 0.05,
        bit_flip: 0.01,
        phase_flip: 0.01,
        amplitude_damping: 0.03,
        readout_flip_0to1: 0.02,
        readout_flip_1to0: 0.03,
    };
    let mk = |threads: Option<usize>| RunOptions {
        shots: 4000,
        seed: Some(2718),
        noise: Some(model.clone()),
        threads,
        ..Default::default()
    };
    let a = zeno::run_program(&c.to_program(), &mk(None)).unwrap();
    let b = zeno::run_program(&c.to_program(), &mk(None)).unwrap();
    assert_eq!(a.counts, b.counts, "same seed must reproduce noisy counts");
    let t1 = zeno::run_program(&c.to_program(), &mk(Some(1))).unwrap();
    assert_eq!(
        a.counts, t1.counts,
        "threads=1 must match the default thread count"
    );
    assert!(a.counts.len() > 2, "noise should spread outcomes");
}

#[test]
fn invalid_models_error_cleanly_through_the_run_path() {
    let mut c = Circuit::new(1);
    c.x(0).measure(0, 0);
    let err = zeno::run_program(
        &c.to_program(),
        &RunOptions {
            noise: Some(NoiseModel {
                depolarizing_1q: 1.5,
                ..Default::default()
            }),
            ..Default::default()
        },
    )
    .unwrap_err();
    assert!(matches!(err, zeno::Error::Noise(_)), "{err}");
    assert!(err.to_string().starts_with("invalid noise model:"), "{err}");

    // JSON front door: range + unknown-field failures are equally clean.
    assert!(NoiseModel::from_json(r#"{"bit_flip": 2.0}"#).is_err());
    let msg = NoiseModel::from_json(r#"{"bitflip": 0.1}"#)
        .unwrap_err()
        .to_string();
    assert!(
        msg.contains("bitflip"),
        "unknown-field error must name the field: {msg}"
    );
}

#[test]
fn norm_stays_one_under_noisy_evolution() {
    // Drive the channel machinery directly on a backend and check ‖ψ‖² —
    // the sampler renormalizes anyway, so only an explicit check catches
    // a forgotten no-jump renormalization (see docs/NOISE.md).
    use rand::SeedableRng;
    let model = NoiseModel {
        depolarizing_1q: 0.2,
        depolarizing_2q: 0.3,
        bit_flip: 0.1,
        phase_flip: 0.1,
        amplitude_damping: 0.4,
        ..Default::default()
    };
    let h = std::f64::consts::FRAC_1_SQRT_2;
    let had = [
        zeno::C64::new(h, 0.0),
        zeno::C64::new(h, 0.0),
        zeno::C64::new(h, 0.0),
        zeno::C64::new(-h, 0.0),
    ];
    let mut be = CpuBackend::<f64>::new(3);
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(7);
    for step in 0..200u32 {
        let q = step % 3;
        be.apply_unitary(&[q], &had);
        apply_gate_noise(&mut be, &model, &[q], &mut rng);
        be.apply_cx(q, (q + 1) % 3);
        let qs = [q.min((q + 1) % 3), q.max((q + 1) % 3)];
        apply_gate_noise(&mut be, &model, &qs, &mut rng);
        let n = be.norm_sqr();
        assert!((n - 1.0).abs() < 1e-9, "step {step}: norm² drifted to {n}");
    }
}

// ------------------------------------------------------- qiskit-aer oracle

fn tvd(a: &Counts, b: &HashMap<String, u64>, shots: u64) -> f64 {
    let mut keys: std::collections::BTreeSet<&str> = a.iter().map(|(k, _)| k.as_str()).collect();
    keys.extend(b.keys().map(|k| k.as_str()));
    let s = shots as f64;
    keys.iter()
        .map(|k| {
            let pa = a.get(k) as f64 / s;
            let pb = b.get(*k).copied().unwrap_or(0) as f64 / s;
            (pa - pb).abs()
        })
        .sum::<f64>()
        / 2.0
}

/// Cross-check composite noise against qiskit-aer (the strongest oracle):
/// the same seven-channel model built from aer primitives must produce
/// the same distribution — TVD < 0.02 at 100k shots. Skips (loudly) when
/// the /tmp/qk-venv interpreter is absent so CI stays green without it.
#[test]
fn qiskit_aer_cross_check_tvd() {
    let py = std::path::Path::new("/tmp/qk-venv/bin/python");
    if !py.exists() {
        eprintln!("SKIP: /tmp/qk-venv/bin/python not found (qiskit-aer oracle unavailable)");
        return;
    }
    let model = NoiseModel {
        depolarizing_1q: 0.05,
        depolarizing_2q: 0.08,
        bit_flip: 0.02,
        phase_flip: 0.03,
        amplitude_damping: 0.06,
        readout_flip_0to1: 0.02,
        readout_flip_1to0: 0.05,
    };
    let shots = 100_000u64;
    // No circuit may contain adjacent self-inverse pairs (zeno's
    // cancellation pass would drop them; aer would keep them noisy).
    type Gates = Vec<(&'static str, Vec<u32>)>;
    let circuits: Vec<(&str, u32, Gates)> = vec![
        ("bell", 2, vec![("h", vec![0]), ("cx", vec![0, 1])]),
        (
            "chain",
            1,
            vec![
                ("x", vec![0]),
                ("h", vec![0]),
                ("x", vec![0]),
                ("h", vec![0]),
            ],
        ),
        (
            "ghz3",
            3,
            vec![("h", vec![0]), ("cx", vec![0, 1]), ("cx", vec![1, 2])],
        ),
    ];

    let spec = serde_json::json!({
        "shots": shots,
        "seed": 20260711u64,
        "model": model,
        "circuits": circuits.iter().map(|(name, n, gates)| serde_json::json!({
            "name": name,
            "n": n,
            "gates": gates.iter().map(|(g, qs)| serde_json::json!([g, qs])).collect::<Vec<_>>(),
        })).collect::<Vec<_>>(),
    });

    let script = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/aer_reference.py");
    let mut child = Command::new(py)
        .arg(script)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn the aer oracle");
    child
        .stdin
        .take()
        .expect("piped stdin")
        .write_all(spec.to_string().as_bytes())
        .expect("write the spec");
    let out = child.wait_with_output().expect("oracle exit");
    assert!(
        out.status.success(),
        "aer oracle failed:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let aer: HashMap<String, HashMap<String, u64>> =
        serde_json::from_slice(&out.stdout).expect("oracle JSON");

    for (name, n, gates) in &circuits {
        let mut c = Circuit::new(*n);
        for (g, qs) in gates {
            c.gate(g, qs, &[]);
        }
        c.measure_all();
        let r = run_noisy(&c, model.clone(), shots, 424242);
        let d = tvd(&r.counts, &aer[*name], shots);
        eprintln!("aer cross-check {name}: TVD = {d:.4}");
        assert!(d < 0.02, "{name}: TVD {d:.4} ≥ 0.02 vs qiskit-aer");
    }
}

// ------------------------------------------------------------ metal parity

/// CPU (f64) vs Metal (f32) noisy counts at the same seed: the same
/// trajectory machinery runs through the same Backend trait, but the
/// jump/branch decisions compare RNG draws against `prob_one` reductions
/// computed from f32 amplitudes on Metal and f64 on CPU, so occasional
/// trajectories branch differently — distributions agree (TVD), bits
/// need not.
#[cfg(feature = "metal")]
#[test]
fn metal_noisy_counts_match_cpu_in_distribution() {
    let mut c = Circuit::new(4);
    c.h(0).h(1).h(2).h(3);
    c.cx(0, 1).cx(1, 2).cx(2, 3);
    c.rz(0, 0.7).rz(3, 1.1);
    c.cx(0, 3);
    c.measure_all();
    let model = NoiseModel {
        depolarizing_1q: 0.03,
        depolarizing_2q: 0.06,
        bit_flip: 0.01,
        phase_flip: 0.02,
        amplitude_damping: 0.04,
        readout_flip_0to1: 0.01,
        readout_flip_1to0: 0.02,
    };
    let shots = 20_000u64;
    let mk = |backend| RunOptions {
        shots,
        seed: Some(31337),
        backend,
        noise: Some(model.clone()),
        ..Default::default()
    };
    let cpu = zeno::run_program(&c.to_program(), &mk(zeno::BackendChoice::Cpu)).unwrap();
    let gpu = zeno::run_program(&c.to_program(), &mk(zeno::BackendChoice::Metal)).unwrap();
    assert_eq!(gpu.backend, "metal-f32");
    let aer_like: HashMap<String, u64> = gpu.counts.iter().map(|(k, &v)| (k.clone(), v)).collect();
    let d = tvd(&cpu.counts, &aer_like, shots);
    eprintln!("metal noisy parity: TVD = {d:.4}");
    assert!(d < 0.05, "cpu vs metal noisy TVD {d:.4} ≥ 0.05");
}
