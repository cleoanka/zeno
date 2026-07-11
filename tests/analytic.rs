//! Analytic and statistical acceptance tests: exact known states (GHZ,
//! QFT against the DFT formula), sampling statistics against 6σ binomial
//! bounds, dynamic-circuit semantics (teleportation of a u3 state, reset,
//! mid-circuit measurement), compiler edge cases, and determinism across
//! runs and thread counts.
//!
//! Every test is seeded and deterministic. Statistical bounds are 6σ of the
//! exact binomial, so a failure is evidence of a real defect, not noise.
//! The QFT check does not guess bit conventions: the builder's own
//! instruction stream is replayed through a tiny local interpreter and only
//! then pinned to the closed-form DFT.

use kuantum::compiler::{compile, CompileOptions};
use kuantum::ir::{GateInstr, Instr, Program, Reg};
use kuantum::{qft, random_circuit, Circuit, RunOptions, Simulator, C64};

const PI: f64 = std::f64::consts::PI;

fn cpx(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// e^{iθ}
fn eip(theta: f64) -> C64 {
    cpx(theta.cos(), theta.sin())
}

fn max_delta(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

/// Textbook gate application for the tiny local interpreter (independent
/// of both `src/state.rs` and `tests/reference.rs`). `qs` is in argument
/// order: bit `b` of a local index is global qubit `qs[b]`.
fn apply_gate(amps: &mut [C64], qs: &[u32], mat: &[C64]) {
    let dim = 1usize << qs.len();
    let mask: usize = qs.iter().map(|&q| 1usize << q).sum();
    let offset: Vec<usize> = (0..dim)
        .map(|m| {
            qs.iter()
                .enumerate()
                .filter(|&(b, _)| (m >> b) & 1 == 1)
                .map(|(_, &q)| 1usize << q)
                .sum()
        })
        .collect();
    for base in 0..amps.len() {
        if base & mask != 0 {
            continue;
        }
        let old: Vec<C64> = offset.iter().map(|&o| amps[base + o]).collect();
        for (l, &o) in offset.iter().enumerate() {
            let mut acc = cpx(0.0, 0.0);
            for (m, &om) in old.iter().enumerate() {
                acc += mat[l * dim + m] * om;
            }
            amps[base + o] = acc;
        }
    }
}

/// The only gates the qft builder (plus |x⟩ preparation) may emit.
fn small_gate(name: &str, params: &[f64]) -> Vec<C64> {
    let z = cpx(0.0, 0.0);
    let o = cpx(1.0, 0.0);
    let sq = std::f64::consts::FRAC_1_SQRT_2;
    match name {
        "x" => vec![z, o, o, z],
        "h" => vec![cpx(sq, 0.0), cpx(sq, 0.0), cpx(sq, 0.0), cpx(-sq, 0.0)],
        "cp" => {
            // diag(1, 1, 1, e^{iλ}) — symmetric in its two qubits.
            let mut m = vec![z; 16];
            m[0] = o;
            m[5] = o;
            m[10] = o;
            m[15] = eip(params[0]);
            m
        }
        other => panic!("unexpected gate '{other}' in the qft program"),
    }
}

fn bitrev(j: usize, n: u32) -> usize {
    (0..n).fold(0usize, |acc, b| acc | (((j >> b) & 1) << (n - 1 - b)))
}

fn statevector_opts() -> RunOptions {
    RunOptions {
        shots: 0,
        want_statevector: true,
        seed: Some(1),
        ..Default::default()
    }
}

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

/// Teleport u3(θ,φ,λ)|0⟩ from q0 to q2 with mid-circuit measurement and
/// classically controlled corrections (three 1-bit cregs, QASM2 style).
/// Counts key layout: "out m1 m0" (last-declared creg leftmost).
fn teleport_program(theta: f64, phi: f64, lambda: f64) -> Program {
    let g = |name: &str, qubits: &[u32], params: &[f64]| {
        Instr::Gate(GateInstr {
            name: name.into(),
            params: params.to_vec(),
            qubits: qubits.to_vec(),
        })
    };
    Program {
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
            g("u3", &[0], &[theta, phi, lambda]),
            g("h", &[1], &[]),
            g("cx", &[1, 2], &[]),
            g("cx", &[0, 1], &[]),
            g("h", &[0], &[]),
            Instr::Measure { qubit: 0, clbit: 0 },
            Instr::Measure { qubit: 1, clbit: 1 },
            Instr::If {
                creg: 1,
                value: 1,
                op: Box::new(g("x", &[2], &[])),
            },
            Instr::If {
                creg: 0,
                value: 1,
                op: Box::new(g("z", &[2], &[])),
            },
            Instr::Measure { qubit: 2, clbit: 2 },
        ],
    }
}

// ------------------------------------------------------ analytic states

#[test]
fn ghz12_collapses_to_exactly_two_branches() {
    let shots = 20_000u64;
    let mut c = Circuit::new(12);
    c.h(0);
    for q in 0..11 {
        c.cx(q, q + 1);
    }
    c.measure_all();
    let r = Simulator::new().shots(shots).seed(41).run(&c).unwrap();
    let zeros = "0".repeat(12);
    let ones = "1".repeat(12);
    assert_eq!(
        r.counts.len(),
        2,
        "GHZ must produce exactly two keys, got {:?}",
        r.counts
    );
    assert_eq!(r.counts.get(&zeros) + r.counts.get(&ones), shots);
    six_sigma(r.counts.get(&zeros), shots, 0.5, "GHZ |0…0⟩ branch");
}

#[test]
fn qft5_matches_reference_replay_and_dft_formula() {
    let n = 5u32;
    let dim = 1usize << n;
    for &x in &[0usize, 1, 5, 19, 31] {
        // |x⟩ preparation + the crate's own qft(n) instruction stream.
        let mut p = qft(n).to_program();
        let prefix: Vec<Instr> = (0..n)
            .filter(|&q| (x >> q) & 1 == 1)
            .map(|q| {
                Instr::Gate(GateInstr {
                    name: "x".into(),
                    params: vec![],
                    qubits: vec![q],
                })
            })
            .collect();
        p.instrs.splice(0..0, prefix);

        let r = kuantum::run_program(&p, &statevector_opts()).unwrap();
        let sv = r.statevector.unwrap();

        // Replay the exact same instruction stream independently.
        let mut amps = vec![cpx(0.0, 0.0); dim];
        amps[0] = cpx(1.0, 0.0);
        for ins in &p.instrs {
            match ins {
                Instr::Gate(g) => apply_gate(&mut amps, &g.qubits, &small_gate(&g.name, &g.params)),
                other => panic!("unexpected instruction {other:?}"),
            }
        }
        let d = max_delta(&sv, &amps);
        assert!(d < 1e-12, "qft(5)|{x}⟩ vs replay: max |Δ| = {d:.3e}");

        // The cp ladder without final swaps writes the DFT bit-reversed:
        // sv[rev(j)] = e^{2πi·x·j/2^n}/√(2^n). (Verified against the replay
        // above, not assumed.)
        let scale = 1.0 / (dim as f64).sqrt();
        for j in 0..dim {
            let want = eip(2.0 * PI * (x * j) as f64 / dim as f64) * scale;
            let got = sv[bitrev(j, n)];
            assert!(
                (got - want).norm() < 1e-12,
                "qft(5)|{x}⟩, frequency {j}: got {got}, want {want}"
            );
        }
    }
}

// -------------------------------------------------- sampling statistics

#[test]
fn uniform_superposition_sampling_within_six_sigma() {
    let n = 10u32;
    let shots = 100_000u64;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    c.measure_all();
    let r = Simulator::new().shots(shots).seed(2024).run(&c).unwrap();
    assert_eq!(r.counts.total(), shots);

    let p = 1.0 / 1024.0;
    // Check all 1024 possible keys, including any that never occurred.
    for idx in 0..1024usize {
        let key = format!("{idx:010b}");
        six_sigma(r.counts.get(&key), shots, p, &format!("|+⟩^10 key {key}"));
    }
}

#[test]
fn bell_correlations_are_exact() {
    let shots = 50_000u64;
    let mut c = Circuit::new(2);
    c.h(0).cx(0, 1).measure_all();
    let r = Simulator::new().shots(shots).seed(11).run(&c).unwrap();
    assert_eq!(
        r.counts.get("00") + r.counts.get("11"),
        shots,
        "Bell state may only produce 00 and 11, got {:?}",
        r.counts
    );
    six_sigma(r.counts.get("00"), shots, 0.5, "Bell 00 branch");
}

// ------------------------------------------------------------ dynamics

#[test]
fn teleportation_transfers_u3_state() {
    let (theta, phi, lambda) = (0.7, 0.3, 0.1);
    let shots = 20_000u64;
    let p = teleport_program(theta, phi, lambda);

    let compiled = compile(&p, &CompileOptions::default()).unwrap();
    assert!(
        compiled.dynamic,
        "teleportation must compile as a dynamic circuit"
    );

    let opts = RunOptions {
        shots,
        seed: Some(7),
        ..Default::default()
    };
    let r = kuantum::run_program(&p, &opts).unwrap();
    assert_eq!(r.counts.total(), shots);

    // Marginal of the output qubit: P(1) = sin²(θ/2), independent of the
    // Bell measurement outcomes.
    let ones: u64 = r
        .counts
        .iter()
        .filter(|(k, _)| k.starts_with('1'))
        .map(|(_, n)| *n)
        .sum();
    let p1 = (theta / 2.0).sin().powi(2);
    six_sigma(ones, shots, p1, "teleported qubit P(1)");

    // The four Bell branches (m1, m0) are each 1/4.
    let mut branch = [0u64; 4];
    for (key, cnt) in r.counts.iter() {
        // Key layout: "out m1 m0" → bytes [0], [2], [4].
        let b = key.as_bytes();
        assert_eq!(b.len(), 5, "unexpected key format {key:?}");
        let m1 = usize::from(b[2] - b'0');
        let m0 = usize::from(b[4] - b'0');
        branch[(m1 << 1) | m0] += *cnt;
    }
    for (i, &cnt) in branch.iter().enumerate() {
        six_sigma(cnt, shots, 0.25, &format!("teleport Bell branch {i}"));
    }
}

#[test]
fn reset_always_yields_zero() {
    // From |1⟩.
    let mut c = Circuit::new(1);
    c.x(0).reset(0).measure(0, 0);
    let r = Simulator::new().shots(2000).seed(3).run(&c).unwrap();
    assert_eq!(r.counts.get("0"), 2000, "reset |1⟩ must always measure 0");

    // From a superposition (both collapse outcomes exercised).
    let mut c2 = Circuit::new(1);
    c2.h(0).reset(0).measure(0, 0);
    let r2 = Simulator::new().shots(2000).seed(4).run(&c2).unwrap();
    assert_eq!(r2.counts.get("0"), 2000, "reset |+⟩ must always measure 0");
}

#[test]
fn mid_circuit_measurement_statistics() {
    let shots = 20_000u64;
    // H, measure (bit 0), H again, measure (bit 1): the first bit is
    // 50/50 and the second is 50/50 independent of the first.
    let mut c = Circuit::new(2);
    c.h(0);
    c.measure(0, 0);
    c.h(0);
    c.measure(0, 1);
    let r = Simulator::new().shots(shots).seed(6).run(&c).unwrap();
    assert_eq!(r.counts.total(), shots);

    let k = |s: &str| r.counts.get(s);
    assert_eq!(k("00") + k("01") + k("10") + k("11"), shots);
    // Marginals (keys are "bit1 bit0" as one 2-bit creg).
    six_sigma(k("01") + k("11"), shots, 0.5, "first measurement P(1)");
    six_sigma(k("10") + k("11"), shots, 0.5, "second measurement P(1)");
    // Joint: uniform over the four outcomes ⇔ independence.
    for key in ["00", "01", "10", "11"] {
        six_sigma(k(key), shots, 0.25, &format!("joint outcome {key}"));
    }
}

// -------------------------------------------------- compiler edge cases

#[test]
fn self_inverse_pairs_cancel_to_identity() {
    let mut c = Circuit::new(3);
    c.h(0).h(0).cx(0, 1).cx(0, 1);
    let r = kuantum::run_program(&c.to_program(), &statevector_opts()).unwrap();
    assert_eq!(r.stats.input_gates, 4);
    assert_eq!(r.stats.cancelled, 4, "h h and cx cx must cancel");
    assert_eq!(r.stats.output_ops, 0);
    let sv = r.statevector.unwrap();
    assert!((sv[0] - cpx(1.0, 0.0)).norm() < 1e-12);
    assert!(sv[1..].iter().all(|a| a.norm() < 1e-12));
}

#[test]
fn barrier_blocks_cancellation_but_identity_holds_numerically() {
    let mut c = Circuit::new(2);
    c.h(0).barrier().h(0).cx(0, 1).barrier().cx(0, 1);
    let r = kuantum::run_program(&c.to_program(), &statevector_opts()).unwrap();
    assert_eq!(r.stats.cancelled, 0, "barriers must fence cancellation");
    assert!(r.stats.output_ops >= 1, "gates must actually execute");
    let sv = r.statevector.unwrap();
    assert!(
        (sv[0] - cpx(1.0, 0.0)).norm() < 1e-12,
        "numerical identity, got amp[0] = {}",
        sv[0]
    );
    assert!(sv[1..].iter().all(|a| a.norm() < 1e-12));
}

#[test]
fn empty_one_qubit_and_measure_only_circuits() {
    // Empty circuit: |0…0⟩, no counts.
    let c = Circuit::new(3);
    let mut opts = statevector_opts();
    opts.shots = 100;
    let r = kuantum::run_program(&c.to_program(), &opts).unwrap();
    assert!(r.counts.is_empty(), "no measurements → no counts");
    let sv = r.statevector.unwrap();
    assert_eq!(sv.len(), 8);
    assert!((sv[0] - cpx(1.0, 0.0)).norm() < 1e-15);
    assert!(sv[1..].iter().all(|a| a.norm() < 1e-15));

    // Smallest possible circuit.
    let mut c1 = Circuit::new(1);
    c1.h(0).measure(0, 0);
    let r1 = Simulator::new().shots(4000).seed(9).run(&c1).unwrap();
    assert_eq!(r1.counts.get("0") + r1.counts.get("1"), 4000);
    six_sigma(r1.counts.get("0"), 4000, 0.5, "1-qubit H");

    // Measure-only circuit: deterministic all-zeros.
    let mut cm = Circuit::new(4);
    cm.measure_all();
    let rm = Simulator::new().shots(256).seed(2).run(&cm).unwrap();
    assert_eq!(rm.counts.len(), 1);
    assert_eq!(rm.counts.get("0000"), 256);
}

// ---------------------------------------------------------- determinism

#[test]
fn determinism_sampled_path() {
    let mut c = random_circuit(8, 10, 1234);
    c.measure_all();
    let sim = Simulator::new().shots(5000).seed(77);
    let a = sim.run(&c).unwrap();
    let b = sim.run(&c).unwrap();
    assert_eq!(a.counts, b.counts, "same seed must reproduce counts");
    assert_eq!(a.seed, 77);
    assert!(a.counts.len() > 1, "random circuit should spread outcomes");

    let t1 = Simulator::new()
        .shots(5000)
        .seed(77)
        .threads(1)
        .run(&c)
        .unwrap();
    assert_eq!(
        a.counts, t1.counts,
        "threads=1 must match the default thread count"
    );
}

#[test]
fn determinism_dynamic_path() {
    let p = teleport_program(0.7, 0.3, 0.1);
    let opts = RunOptions {
        shots: 3000,
        seed: Some(123),
        ..Default::default()
    };
    let a = kuantum::run_program(&p, &opts).unwrap();
    let b = kuantum::run_program(&p, &opts).unwrap();
    assert_eq!(a.counts, b.counts, "same seed must reproduce counts");

    let opts1 = RunOptions {
        threads: Some(1),
        ..opts
    };
    let t1 = kuantum::run_program(&p, &opts1).unwrap();
    assert_eq!(
        a.counts, t1.counts,
        "threads=1 must match the default thread count"
    );
}
