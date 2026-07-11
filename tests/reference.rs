//! Adversarial reference verification of kuantum.
//!
//! `RefSim` is a from-scratch textbook dense state-vector simulator with
//! deliberately different implementation choices from `src/state.rs`: one
//! interleaved `Vec<C64>` of amplitudes and explicit bit arithmetic per
//! basis index — no split re/im arrays, no run-walk kernels, no fusion, no
//! rayon. Every native gate matrix is hand-coded here from the standard
//! definitions (qiskit U(θ,φ,λ) convention; controls first = low bits of
//! the local index) and first cross-checked against `kuantum::gates::build`;
//! the full simulator is then cross-checked gate-by-gate in multiple qubit
//! argument orders (including descending ones), and on random circuits at
//! every fusion level and both precisions.
//!
//! All randomness is seeded through a local splitmix64 so the tests are
//! deterministic and need no external crates.

use kuantum::gates::{self, GateMatrix};
use kuantum::ir::Instr;
use kuantum::{random_circuit, Circuit, Precision, Program, RunOptions, Simulator, C64};

const PI: f64 = std::f64::consts::PI;

fn cpx(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

/// e^{iθ}
fn eip(theta: f64) -> C64 {
    cpx(theta.cos(), theta.sin())
}

// ------------------------------------------------------------- test RNG

/// Tiny deterministic PRNG (splitmix64); the tests must not depend on any
/// external crate and must never be unseeded.
struct TestRng(u64);

impl TestRng {
    fn new(seed: u64) -> Self {
        TestRng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in [0, 1).
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    /// Uniform angle in (−π, π].
    fn angle(&mut self) -> f64 {
        (self.uniform() * 2.0 - 1.0) * PI
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// -------------------------------------------------- reference simulator

/// Textbook dense state-vector simulator (see module docs).
struct RefSim {
    n: u32,
    amps: Vec<C64>,
}

impl RefSim {
    fn new(n: u32) -> Self {
        let mut amps = vec![cpx(0.0, 0.0); 1usize << n];
        amps[0] = cpx(1.0, 0.0);
        RefSim { n, amps }
    }

    /// Apply a dense `2^k × 2^k` row-major matrix to qubits `qs`, given in
    /// argument order: bit `b` of a local index is global qubit `qs[b]`.
    fn apply(&mut self, qs: &[u32], mat: &[C64]) {
        let k = qs.len();
        let dim = 1usize << k;
        assert_eq!(mat.len(), dim * dim, "matrix size for {qs:?}");
        let mask: usize = qs.iter().map(|&q| 1usize << q).sum();
        assert_eq!(mask.count_ones() as usize, k, "duplicate qubits {qs:?}");
        // Offset added to a base index (all qs bits clear) by local index m.
        let offset: Vec<usize> = (0..dim)
            .map(|m| {
                qs.iter()
                    .enumerate()
                    .filter(|&(b, _)| (m >> b) & 1 == 1)
                    .map(|(_, &q)| 1usize << q)
                    .sum()
            })
            .collect();
        let mut old = vec![cpx(0.0, 0.0); dim];
        for base in 0..(1usize << self.n) {
            if base & mask != 0 {
                continue;
            }
            for (m, o) in old.iter_mut().enumerate() {
                *o = self.amps[base + offset[m]];
            }
            for (l, &off) in offset.iter().enumerate() {
                let mut acc = cpx(0.0, 0.0);
                for (m, o) in old.iter().enumerate() {
                    acc += mat[l * dim + m] * *o;
                }
                self.amps[base + off] = acc;
            }
        }
    }
}

/// Replay every unitary instruction of a program through the reference.
fn ref_of_program(p: &Program) -> Vec<C64> {
    let mut r = RefSim::new(p.n_qubits());
    for ins in &p.instrs {
        match ins {
            Instr::Gate(g) => r.apply(&g.qubits, &ref_gate(&g.name, &g.params)),
            Instr::Barrier(_) => {}
            other => panic!("reference walker got non-unitary instruction {other:?}"),
        }
    }
    r.amps
}

// ------------------- hand-coded matrices (standard definitions) --------

/// Dense matrix from a diagonal.
fn dmat(d: &[C64]) -> Vec<C64> {
    let dim = d.len();
    let mut m = vec![cpx(0.0, 0.0); dim * dim];
    for (j, v) in d.iter().enumerate() {
        m[j * dim + j] = *v;
    }
    m
}

/// Controlled-U in argument order [control, target]: control is bit 0 of
/// the local index, target is bit 1.
fn ctrl1(u: &[C64]) -> Vec<C64> {
    let mut m = vec![cpx(0.0, 0.0); 16];
    m[0] = cpx(1.0, 0.0); // |c=0,t=0⟩ fixed
    m[2 * 4 + 2] = cpx(1.0, 0.0); // |c=0,t=1⟩ fixed
    for t_out in 0..2usize {
        for t_in in 0..2usize {
            m[(1 + 2 * t_out) * 4 + (1 + 2 * t_in)] = u[t_out * 2 + t_in];
        }
    }
    m
}

/// qiskit convention U(θ, φ, λ).
fn u3m(theta: f64, phi: f64, lambda: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![
        cpx(ct, 0.0),
        -eip(lambda) * st,
        eip(phi) * st,
        eip(phi + lambda) * ct,
    ]
}

fn rxm(theta: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![cpx(ct, 0.0), cpx(0.0, -st), cpx(0.0, -st), cpx(ct, 0.0)]
}

fn rym(theta: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![cpx(ct, 0.0), cpx(-st, 0.0), cpx(st, 0.0), cpx(ct, 0.0)]
}

fn rzm(theta: f64) -> Vec<C64> {
    dmat(&[eip(-theta / 2.0), eip(theta / 2.0)])
}

/// Every native gate, hand-coded dense, argument-order convention.
fn ref_gate(name: &str, params: &[f64]) -> Vec<C64> {
    let z = cpx(0.0, 0.0);
    let o = cpx(1.0, 0.0);
    let i = cpx(0.0, 1.0);
    let sq = std::f64::consts::FRAC_1_SQRT_2;
    let h = || vec![cpx(sq, 0.0), cpx(sq, 0.0), cpx(sq, 0.0), cpx(-sq, 0.0)];
    let y = || vec![z, -i, i, z];
    match name {
        "id" | "u0" => vec![o, z, z, o],
        "x" => vec![z, o, o, z],
        "y" => y(),
        "z" => dmat(&[o, -o]),
        "h" => h(),
        "s" => dmat(&[o, i]),
        "sdg" => dmat(&[o, -i]),
        "t" => dmat(&[o, eip(PI / 4.0)]),
        "tdg" => dmat(&[o, eip(-PI / 4.0)]),
        "sx" => vec![cpx(0.5, 0.5), cpx(0.5, -0.5), cpx(0.5, -0.5), cpx(0.5, 0.5)],
        "sxdg" => vec![cpx(0.5, -0.5), cpx(0.5, 0.5), cpx(0.5, 0.5), cpx(0.5, -0.5)],
        "rx" => rxm(params[0]),
        "ry" => rym(params[0]),
        "rz" => rzm(params[0]),
        "u1" | "p" => dmat(&[o, eip(params[0])]),
        "u2" => u3m(PI / 2.0, params[0], params[1]),
        "u3" | "u" => u3m(params[0], params[1], params[2]),
        "cx" => {
            // |c,t⟩ → |c, t⊕c⟩; local index m = c + 2t.
            let mut m = vec![z; 16];
            for col in 0..4usize {
                let (cb, tb) = (col & 1, (col >> 1) & 1);
                let row = cb | ((tb ^ cb) << 1);
                m[row * 4 + col] = o;
            }
            m
        }
        "cy" => ctrl1(&y()),
        "cz" => dmat(&[o, o, o, -o]),
        "ch" => ctrl1(&h()),
        "swap" => {
            let mut m = vec![z; 16];
            for col in 0..4usize {
                let (a, b) = (col & 1, (col >> 1) & 1);
                m[(b | (a << 1)) * 4 + col] = o;
            }
            m
        }
        "cp" | "cu1" => dmat(&[o, o, o, eip(params[0])]),
        "crx" => ctrl1(&rxm(params[0])),
        "cry" => ctrl1(&rym(params[0])),
        "crz" => ctrl1(&rzm(params[0])),
        "cu3" => ctrl1(&u3m(params[0], params[1], params[2])),
        "rxx" => {
            // exp(−iθ/2·X⊗X) = cos(θ/2)·I − i·sin(θ/2)·(X⊗X)
            let (ct, st) = ((params[0] / 2.0).cos(), (params[0] / 2.0).sin());
            let mut m = vec![z; 16];
            for col in 0..4usize {
                m[col * 4 + col] = cpx(ct, 0.0);
                m[(col ^ 3) * 4 + col] = cpx(0.0, -st);
            }
            m
        }
        "rzz" => {
            // exp(−iθ/2·Z⊗Z): e^{-iθ/2} on even parity, e^{+iθ/2} on odd.
            let d: Vec<C64> = (0..4usize)
                .map(|m| {
                    let parity = (m & 1) ^ ((m >> 1) & 1);
                    eip(if parity == 1 {
                        params[0] / 2.0
                    } else {
                        -params[0] / 2.0
                    })
                })
                .collect();
            dmat(&d)
        }
        "ccx" => {
            // controls = bits 0,1; target = bit 2.
            let mut m = vec![z; 64];
            for col in 0..8usize {
                let row = if col & 3 == 3 { col ^ 4 } else { col };
                m[row * 8 + col] = o;
            }
            m
        }
        "cswap" => {
            // control = bit 0; swap bits 1 and 2 when it is set.
            let mut m = vec![z; 64];
            for col in 0..8usize {
                let (a, b) = ((col >> 1) & 1, (col >> 2) & 1);
                let row = if col & 1 == 1 {
                    1 | (b << 1) | (a << 2)
                } else {
                    col
                };
                m[row * 8 + col] = o;
            }
            m
        }
        other => panic!("no reference matrix for gate '{other}'"),
    }
}

/// The complete native gate set of `gates::lookup`: (name, arity, n_params).
const NATIVE: &[(&str, usize, usize)] = &[
    ("id", 1, 0),
    ("u0", 1, 0),
    ("x", 1, 0),
    ("y", 1, 0),
    ("z", 1, 0),
    ("h", 1, 0),
    ("s", 1, 0),
    ("sdg", 1, 0),
    ("t", 1, 0),
    ("tdg", 1, 0),
    ("sx", 1, 0),
    ("sxdg", 1, 0),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("u1", 1, 1),
    ("p", 1, 1),
    ("u2", 1, 2),
    ("u3", 1, 3),
    ("u", 1, 3),
    ("cx", 2, 0),
    ("cy", 2, 0),
    ("cz", 2, 0),
    ("ch", 2, 0),
    ("swap", 2, 0),
    ("cp", 2, 1),
    ("cu1", 2, 1),
    ("crx", 2, 1),
    ("cry", 2, 1),
    ("crz", 2, 1),
    ("rxx", 2, 1),
    ("rzz", 2, 1),
    ("cu3", 2, 3),
    ("ccx", 3, 0),
    ("cswap", 3, 0),
];

// ------------------------------------------------------------ helpers

fn dense_of(g: GateMatrix, arity: usize) -> Vec<C64> {
    let dim = 1usize << arity;
    match g {
        GateMatrix::Unitary(m) => m,
        GateMatrix::Diagonal(d) => {
            assert_eq!(d.len(), dim, "diagonal length");
            dmat(&d)
        }
    }
}

fn max_delta(a: &[C64], b: &[C64]) -> f64 {
    assert_eq!(a.len(), b.len(), "vector length mismatch");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).norm())
        .fold(0.0, f64::max)
}

fn norm_of(v: &[C64]) -> f64 {
    v.iter().map(|a| a.norm_sqr()).sum::<f64>().sqrt()
}

/// Relabel the two qubit arguments of a 4×4 matrix (bit 0 ↔ bit 1).
fn swap_two_qubit_args(m: &[C64]) -> Vec<C64> {
    let perm = |j: usize| ((j & 1) << 1) | ((j >> 1) & 1);
    let mut out = vec![cpx(0.0, 0.0); 16];
    for r in 0..4usize {
        for col in 0..4usize {
            out[perm(r) * 4 + perm(col)] = m[r * 4 + col];
        }
    }
    out
}

fn pick_distinct(rng: &mut TestRng, n: u32, k: usize) -> Vec<u32> {
    let mut qs: Vec<u32> = Vec::with_capacity(k);
    while qs.len() < k {
        let q = rng.below(n as usize) as u32;
        if !qs.contains(&q) {
            qs.push(q);
        }
    }
    qs
}

/// The adversarial gate pool: every native gate, including all diagonals.
const POOL: &[(&str, usize, usize)] = &[
    ("x", 1, 0),
    ("y", 1, 0),
    ("z", 1, 0),
    ("h", 1, 0),
    ("s", 1, 0),
    ("sdg", 1, 0),
    ("t", 1, 0),
    ("tdg", 1, 0),
    ("sx", 1, 0),
    ("sxdg", 1, 0),
    ("rx", 1, 1),
    ("ry", 1, 1),
    ("rz", 1, 1),
    ("p", 1, 1),
    ("u1", 1, 1),
    ("u2", 1, 2),
    ("u3", 1, 3),
    ("u", 1, 3),
    ("cx", 2, 0),
    ("cy", 2, 0),
    ("cz", 2, 0),
    ("ch", 2, 0),
    ("swap", 2, 0),
    ("cp", 2, 1),
    ("cu1", 2, 1),
    ("crx", 2, 1),
    ("cry", 2, 1),
    ("crz", 2, 1),
    ("rxx", 2, 1),
    ("rzz", 2, 1),
    ("cu3", 2, 3),
    ("ccx", 3, 0),
    ("cswap", 3, 0),
];

/// Diagonal-only pool (plus their controlled forms).
const DIAG_POOL: &[(&str, usize, usize)] = &[
    ("z", 1, 0),
    ("s", 1, 0),
    ("sdg", 1, 0),
    ("t", 1, 0),
    ("tdg", 1, 0),
    ("rz", 1, 1),
    ("p", 1, 1),
    ("cz", 2, 0),
    ("cp", 2, 1),
    ("crz", 2, 1),
    ("rzz", 2, 1),
];

fn append_random_gates(
    c: &mut Circuit,
    n: u32,
    count: u32,
    seed: u64,
    pool: &[(&str, usize, usize)],
    barriers: bool,
) {
    let mut rng = TestRng::new(seed);
    let pool: Vec<&(&str, usize, usize)> = pool
        .iter()
        .filter(|&&(_, arity, _)| arity <= n as usize)
        .collect();
    for _ in 0..count {
        let &(name, arity, np) = pool[rng.below(pool.len())];
        let qs = pick_distinct(&mut rng, n, arity);
        let params: Vec<f64> = (0..np).map(|_| rng.angle()).collect();
        c.gate(name, &qs, &params);
        if barriers && rng.below(8) == 0 {
            c.barrier();
        }
    }
}

fn random_mixed_circuit(n: u32, count: u32, seed: u64) -> Circuit {
    let mut c = Circuit::new(n);
    append_random_gates(&mut c, n, count, seed, POOL, true);
    c
}

fn check_fusion_levels(c: &Circuit, want: &[C64], what: &str) {
    for fusion in 0..=6u8 {
        let sv = Simulator::new().fusion(fusion).statevector(c).unwrap();
        let norm = norm_of(&sv);
        assert!(
            (norm - 1.0).abs() < 1e-12,
            "{what} fusion={fusion}: norm {norm} drifted from 1"
        );
        let d = max_delta(&sv, want);
        assert!(d < 1e-10, "{what} fusion={fusion}: max |Δ| = {d:.3e}");
    }
}

// ------------------------------------------------------------- part 1

/// Cross-check the gate library itself: every native matrix must equal the
/// hand-coded standard definition, elementwise, in argument-order
/// convention (controls first = low bits).
#[test]
fn hand_coded_matrices_match_gate_library() {
    let mut rng = TestRng::new(0x51DE_CAFE);
    for &(name, arity, n_params) in NATIVE {
        let def =
            gates::lookup(name).unwrap_or_else(|| panic!("'{name}' missing from gates::lookup"));
        assert_eq!(def.arity as usize, arity, "'{name}' arity");
        assert_eq!(def.n_params as usize, n_params, "'{name}' n_params");
        for _ in 0..3 {
            let params: Vec<f64> = (0..n_params).map(|_| rng.angle()).collect();
            let lib = dense_of(
                gates::build(name, &params)
                    .unwrap_or_else(|| panic!("build('{name}', {params:?}) returned None")),
                arity,
            );
            let mine = ref_gate(name, &params);
            let d = max_delta(&lib, &mine);
            if d > 1e-14 {
                let mut hint = String::new();
                if arity == 2 && max_delta(&lib, &swap_two_qubit_args(&mine)) < 1e-14 {
                    hint = " — the library matrix equals the standard definition with \
                            the two qubit arguments SWAPPED (control/target exchanged)"
                        .into();
                }
                panic!("gate '{name}' params {params:?}: matrix mismatch, max |Δ| = {d:.3e}{hint}");
            }
        }
    }
}

// ------------------------------------------------------------- part 2a

/// The single most important test: every native gate, applied through the
/// full kuantum pipeline to a random product state, must match the
/// reference for several argument orders including descending ones. This
/// catches qubit-permutation (control/target) bugs.
#[test]
fn every_native_gate_matches_reference_in_all_argument_orders() {
    fn orders_for(arity: usize) -> Vec<Vec<u32>> {
        match arity {
            1 => vec![vec![0], vec![2], vec![3]],
            2 => vec![vec![0, 1], vec![1, 0], vec![3, 1], vec![2, 3], vec![3, 0]],
            3 => vec![vec![0, 1, 2], vec![3, 0, 2], vec![2, 3, 0], vec![1, 3, 2]],
            _ => unreachable!("arity {arity}"),
        }
    }
    let mut rng = TestRng::new(0xA5A5_0001);
    for &(name, arity, n_params) in NATIVE {
        for qs in orders_for(arity) {
            let params: Vec<f64> = (0..n_params).map(|_| rng.angle()).collect();
            let prep: Vec<[f64; 3]> = (0..4)
                .map(|_| [rng.angle(), rng.angle(), rng.angle()])
                .collect();

            let mut c = Circuit::new(4);
            for (q, p) in prep.iter().enumerate() {
                c.u(q as u32, p[0], p[1], p[2]);
            }
            c.gate(name, &qs, &params);

            let mut r = RefSim::new(4);
            for (q, p) in prep.iter().enumerate() {
                r.apply(&[q as u32], &u3m(p[0], p[1], p[2]));
            }
            r.apply(&qs, &ref_gate(name, &params));

            for fusion in [0u8, 5] {
                let sv = Simulator::new().fusion(fusion).statevector(&c).unwrap();
                let d = max_delta(&sv, &r.amps);
                assert!(
                    d < 1e-12,
                    "gate '{name}' on qubits {qs:?} params {params:?} \
                     (fusion {fusion}): max |Δ| = {d:.3e}"
                );
            }
        }
    }
}

// --------------------------------------------------------- meta-test

/// Prove the harness is sensitive: a deliberately wrong phase (S vs S†)
/// and a deliberately swapped control/target (cx(0,1) vs cx(1,0)) must
/// produce a large visible difference against kuantum, while the correct
/// reference agrees to 1e-12. Loose tolerances or order-blind comparisons
/// would silently pass such errors; this test forbids that.
#[test]
fn harness_detects_injected_phase_and_order_errors() {
    // Phase sensitivity: |+⟩ then S.
    let mut c = Circuit::new(1);
    c.h(0).s(0);
    let sv = Simulator::new().statevector(&c).unwrap();

    let mut good = RefSim::new(1);
    good.apply(&[0], &ref_gate("h", &[]));
    good.apply(&[0], &ref_gate("s", &[]));
    assert!(max_delta(&sv, &good.amps) < 1e-12, "correct S must agree");

    let mut bad = RefSim::new(1);
    bad.apply(&[0], &ref_gate("h", &[]));
    // Injected sign error: diag(1, -i) = S† instead of S.
    bad.apply(&[0], &dmat(&[cpx(1.0, 0.0), cpx(0.0, -1.0)]));
    let d = max_delta(&sv, &bad.amps);
    assert!(d > 0.5, "a sign-flipped S must be detected, got |Δ| = {d}");

    // Argument-order sensitivity: cx(0,1) vs cx(1,0) on a generic
    // product state.
    let prep = [[0.9, 0.4, 1.3], [0.5, 2.1, -0.7]];
    let mut c2 = Circuit::new(2);
    for (q, p) in prep.iter().enumerate() {
        c2.u(q as u32, p[0], p[1], p[2]);
    }
    c2.cx(0, 1);
    let sv2 = Simulator::new().statevector(&c2).unwrap();

    let mut swapped = RefSim::new(2);
    for (q, p) in prep.iter().enumerate() {
        swapped.apply(&[q as u32], &u3m(p[0], p[1], p[2]));
    }
    swapped.apply(&[1, 0], &ref_gate("cx", &[]));
    let d2 = max_delta(&sv2, &swapped.amps);
    assert!(
        d2 > 0.1,
        "cx with swapped arguments must be detected, got |Δ| = {d2}"
    );
}

// ------------------------------------------------------------- part 2b

#[test]
fn kuantum_random_circuits_match_reference_at_all_fusion_levels() {
    for n in 2..=10u32 {
        let c = random_circuit(n, 20, 1000 + u64::from(n));
        let want = ref_of_program(&c.to_program());
        check_fusion_levels(&c, &want, &format!("random_circuit n={n}"));
    }
}

#[test]
fn adversarial_mixed_circuits_match_reference_at_all_fusion_levels() {
    for n in 2..=10u32 {
        let c = random_mixed_circuit(n, 60, 9000 + u64::from(n));
        let want = ref_of_program(&c.to_program());
        check_fusion_levels(&c, &want, &format!("mixed n={n}"));
    }
}

// ------------------------------------------------------------- part 2c

#[test]
fn f32_statevector_tracks_f64_reference() {
    for n in [4u32, 8, 10] {
        let cases = [
            random_circuit(n, 20, 1000 + u64::from(n)),
            random_mixed_circuit(n, 60, 9000 + u64::from(n)),
        ];
        for (ci, c) in cases.iter().enumerate() {
            let want = ref_of_program(&c.to_program());
            for fusion in [0u8, 5] {
                let sv = Simulator::new()
                    .precision(Precision::F32)
                    .fusion(fusion)
                    .statevector(c)
                    .unwrap();
                let d = max_delta(&sv, &want);
                assert!(
                    d < 2e-4,
                    "n={n} case={ci} fusion={fusion}: f32 max |Δ| = {d:.3e}"
                );
            }
        }
    }
}

// ------------------------------------------------------------- part 2g

/// Diagonal-heavy circuits (rz/cp/crz/rzz ladders behind an H wall,
/// including explicit descending argument orders) must match the reference
/// at every fusion level — this exercises the diagonal-fusion pass and
/// `permute_diag_to_sorted`.
#[test]
fn diagonal_ladders_match_reference() {
    let n = 6u32;
    let mut c = Circuit::new(n);
    for q in 0..n {
        c.h(q);
    }
    append_random_gates(&mut c, n, 40, 0xD1A6_0001, DIAG_POOL, false);
    // Explicit descending argument orders for the non-symmetric diagonals.
    c.cp(5, 2, 0.4);
    c.crz(4, 1, 0.8);
    c.rzz(3, 0, 1.1);
    c.gate("cu1", &[4, 0], &[0.6]);
    c.crz(1, 5, -0.9);

    let want = ref_of_program(&c.to_program());
    check_fusion_levels(&c, &want, "diagonal ladder n=6");
}

/// Alternating disjoint 4-qubit blocks on 8 qubits: the combined support
/// (8) exceeds every allowed fusion width, so the compiler must repeatedly
/// flush groups — and the result must stay exact.
#[test]
fn disjoint_wide_blocks_force_group_flushes() {
    let n = 8u32;
    let mut c = Circuit::new(n);
    for round in 0..4 {
        let th = 0.3 + f64::from(round) * 0.17;
        c.ccx(0, 1, 2);
        c.u(3, th, 0.2, 0.9);
        c.cx(2, 3);
        c.h(0);
        c.swap(1, 3);
        c.ccx(4, 5, 6);
        c.u(7, th + 0.05, 1.1, 0.4);
        c.cx(6, 7);
        c.h(4);
        c.swap(5, 7);
    }
    let want = ref_of_program(&c.to_program());
    check_fusion_levels(&c, &want, "disjoint 4q blocks n=8");

    // The fused width must never exceed the requested maximum, and the two
    // disjoint supports can never merge into a single op.
    let opts = RunOptions {
        shots: 0,
        want_statevector: true,
        seed: Some(1),
        fusion_max: 5,
        ..Default::default()
    };
    let r = kuantum::run_program(&c.to_program(), &opts).unwrap();
    assert!(
        r.stats.max_fused <= 5,
        "fusion_max 5 violated: max_fused = {}",
        r.stats.max_fused
    );
    assert!(
        r.stats.output_ops >= 2,
        "disjoint 8-qubit supports cannot be a single op"
    );
    let d = max_delta(&r.statevector.unwrap(), &want);
    assert!(d < 1e-10, "flush circuit at fusion 5: max |Δ| = {d:.3e}");
}

// -------------------------------------------------- heavyweight (release)

/// Heavyweight cross-check at n=12; run in release:
/// `cargo test --release --test reference -- --ignored`
#[test]
#[ignore = "heavyweight: run in release with -- --ignored"]
fn release_scale_random_circuits_match_reference() {
    let n = 12u32;
    let cases = [
        random_circuit(n, 30, 4242),
        random_mixed_circuit(n, 120, 2424),
    ];
    for (ci, c) in cases.iter().enumerate() {
        let want = ref_of_program(&c.to_program());
        check_fusion_levels(c, &want, &format!("release-scale case={ci}"));
        let sv32 = Simulator::new()
            .precision(Precision::F32)
            .statevector(c)
            .unwrap();
        let d = max_delta(&sv32, &want);
        assert!(d < 2e-4, "release-scale case={ci}: f32 max |Δ| = {d:.3e}");
    }
}
