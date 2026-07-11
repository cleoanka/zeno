//! Native gate set: names → matrices.
//!
//! Matrix convention: row-major, dimension `2^k`; bit `b` of a local index
//! corresponds to `qubits[b]` (the b-th gate argument). For controlled gates
//! the control(s) come first, so control = bit 0.
//!
//! Diagonal gates are returned as [`GateMatrix::Diagonal`] so the compiler
//! and kernels can use O(2^n) sweeps instead of full 2×2-block updates.

use crate::ir::C64;
use std::f64::consts::FRAC_PI_2;

/// A gate resolved to a concrete matrix.
#[derive(Debug, Clone, PartialEq)]
pub enum GateMatrix {
    /// Full `2^k × 2^k` row-major unitary.
    Unitary(Vec<C64>),
    /// Diagonal of a `2^k × 2^k` diagonal unitary.
    Diagonal(Vec<C64>),
}

/// Static description of a native gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GateDef {
    pub arity: u32,
    pub n_params: u32,
}

/// Names that are their own inverse (used by the cancellation pass).
pub const SELF_INVERSE: &[&str] = &[
    "x", "y", "z", "h", "cx", "cy", "cz", "ch", "swap", "ccx", "cswap",
];

fn c(re: f64, im: f64) -> C64 {
    C64::new(re, im)
}

fn expi(theta: f64) -> C64 {
    C64::new(theta.cos(), theta.sin())
}

/// qiskit-convention U(θ, φ, λ).
fn u3(theta: f64, phi: f64, lambda: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![
        c(ct, 0.0),
        -expi(lambda) * st,
        expi(phi) * st,
        expi(phi + lambda) * ct,
    ]
}

fn rx(theta: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![c(ct, 0.0), c(0.0, -st), c(0.0, -st), c(ct, 0.0)]
}

fn ry(theta: f64) -> Vec<C64> {
    let (ct, st) = ((theta / 2.0).cos(), (theta / 2.0).sin());
    vec![c(ct, 0.0), c(-st, 0.0), c(st, 0.0), c(ct, 0.0)]
}

fn rz_diag(theta: f64) -> Vec<C64> {
    vec![expi(-theta / 2.0), expi(theta / 2.0)]
}

/// Controlled version of a 1-qubit unitary: qubits = [control, target],
/// so control = bit 0, target = bit 1. Row-major 4×4.
fn controlled1(u: &[C64]) -> Vec<C64> {
    let mut m = vec![C64::default(); 16];
    for j in 0..4usize {
        let (ctrl, t) = (j & 1, (j >> 1) & 1);
        if ctrl == 0 {
            m[j * 4 + j] = c(1.0, 0.0);
        } else {
            for tp in 0..2usize {
                let i = 1 | (tp << 1);
                m[i * 4 + j] = u[tp * 2 + t];
            }
        }
    }
    m
}

/// Controlled version of a 1-qubit diagonal: qubits = [control, target],
/// so control = bit 0 and index j = control | target<<1.
fn cdiag1(d: &[C64]) -> Vec<C64> {
    vec![c(1.0, 0.0), d[0], c(1.0, 0.0), d[1]]
}

/// Permutation matrix from an index map `perm(j) = i` (column j → row i).
fn permutation(dim: usize, perm: impl Fn(usize) -> usize) -> Vec<C64> {
    let mut m = vec![C64::default(); dim * dim];
    for j in 0..dim {
        m[perm(j) * dim + j] = c(1.0, 0.0);
    }
    m
}

/// Look up a native gate's signature.
pub fn lookup(name: &str) -> Option<GateDef> {
    let (arity, n_params) = match name {
        "id" | "u0" => (1, 0),
        "x" | "y" | "z" | "h" | "s" | "sdg" | "t" | "tdg" | "sx" | "sxdg" => (1, 0),
        "rx" | "ry" | "rz" | "u1" | "p" => (1, 1),
        "u2" => (1, 2),
        "u3" | "u" => (1, 3),
        "cx" | "cy" | "cz" | "ch" | "swap" => (2, 0),
        "cp" | "cu1" | "crx" | "cry" | "crz" | "rxx" | "rzz" => (2, 1),
        "cu3" => (2, 3),
        "ccx" | "cswap" => (3, 0),
        _ => return None,
    };
    Some(GateDef { arity, n_params })
}

/// Build the matrix for a native gate. Returns `None` for unknown names;
/// panics are avoided — wrong param counts return `None` too.
pub fn build(name: &str, params: &[f64]) -> Option<GateMatrix> {
    use GateMatrix::{Diagonal, Unitary};
    let def = lookup(name)?;
    if params.len() != def.n_params as usize {
        return None;
    }
    let p = |i: usize| params[i];
    let one = c(1.0, 0.0);
    let sq = std::f64::consts::FRAC_1_SQRT_2;
    Some(match name {
        "id" | "u0" => Diagonal(vec![one, one]),
        "x" => Unitary(vec![c(0.0, 0.0), one, one, c(0.0, 0.0)]),
        "y" => Unitary(vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)]),
        "z" => Diagonal(vec![one, c(-1.0, 0.0)]),
        "h" => Unitary(vec![c(sq, 0.0), c(sq, 0.0), c(sq, 0.0), c(-sq, 0.0)]),
        "s" => Diagonal(vec![one, c(0.0, 1.0)]),
        "sdg" => Diagonal(vec![one, c(0.0, -1.0)]),
        "t" => Diagonal(vec![one, expi(std::f64::consts::FRAC_PI_4)]),
        "tdg" => Diagonal(vec![one, expi(-std::f64::consts::FRAC_PI_4)]),
        "sx" => Unitary(vec![c(0.5, 0.5), c(0.5, -0.5), c(0.5, -0.5), c(0.5, 0.5)]),
        "sxdg" => Unitary(vec![c(0.5, -0.5), c(0.5, 0.5), c(0.5, 0.5), c(0.5, -0.5)]),
        "rx" => Unitary(rx(p(0))),
        "ry" => Unitary(ry(p(0))),
        "rz" => Diagonal(rz_diag(p(0))),
        "u1" | "p" => Diagonal(vec![one, expi(p(0))]),
        "u2" => Unitary(u3(FRAC_PI_2, p(0), p(1))),
        "u3" | "u" => Unitary(u3(p(0), p(1), p(2))),
        "cx" => Unitary(permutation(4, |j| if j & 1 == 1 { j ^ 2 } else { j })),
        "cy" => {
            let y = vec![c(0.0, 0.0), c(0.0, -1.0), c(0.0, 1.0), c(0.0, 0.0)];
            Unitary(controlled1(&y))
        }
        "cz" => Diagonal(vec![one, one, one, c(-1.0, 0.0)]),
        "ch" => {
            let h = vec![c(sq, 0.0), c(sq, 0.0), c(sq, 0.0), c(-sq, 0.0)];
            Unitary(controlled1(&h))
        }
        "swap" => Unitary(permutation(4, |j| ((j & 1) << 1) | ((j >> 1) & 1))),
        "cp" | "cu1" => Diagonal(vec![one, one, one, expi(p(0))]),
        "crx" => Unitary(controlled1(&rx(p(0)))),
        "cry" => Unitary(controlled1(&ry(p(0)))),
        "crz" => Diagonal(cdiag1(&rz_diag(p(0)))),
        "cu3" => Unitary(controlled1(&u3(p(0), p(1), p(2)))),
        "rxx" => {
            let (ct, st) = ((p(0) / 2.0).cos(), (p(0) / 2.0).sin());
            let mut m = vec![C64::default(); 16];
            for j in 0..4usize {
                m[j * 4 + j] = c(ct, 0.0);
                m[(j ^ 3) * 4 + j] = c(0.0, -st);
            }
            Unitary(m)
        }
        "rzz" => {
            let e0 = expi(-p(0) / 2.0);
            let e1 = expi(p(0) / 2.0);
            Diagonal(vec![e0, e1, e1, e0])
        }
        // controls = bits 0,1; target = bit 2
        "ccx" => Unitary(permutation(8, |j| if j & 3 == 3 { j ^ 4 } else { j })),
        // control = bit 0; swap bits 1,2
        "cswap" => Unitary(permutation(8, |j| {
            if j & 1 == 1 {
                (j & 1) | ((j & 2) << 1) | ((j & 4) >> 1)
            } else {
                j
            }
        })),
        _ => return None,
    })
}

#[cfg(test)]
#[allow(clippy::identity_op, clippy::erasing_op)]
mod tests {
    use super::*;

    fn as_matrix(g: GateMatrix, k: usize) -> Vec<C64> {
        let dim = 1usize << k;
        match g {
            GateMatrix::Unitary(m) => m,
            GateMatrix::Diagonal(d) => {
                let mut m = vec![C64::default(); dim * dim];
                for j in 0..dim {
                    m[j * dim + j] = d[j];
                }
                m
            }
        }
    }

    #[test]
    fn all_native_gates_are_unitary() {
        let names: &[(&str, usize)] = &[
            ("x", 1),
            ("y", 1),
            ("z", 1),
            ("h", 1),
            ("s", 1),
            ("sdg", 1),
            ("t", 1),
            ("tdg", 1),
            ("sx", 1),
            ("sxdg", 1),
            ("cx", 2),
            ("cy", 2),
            ("cz", 2),
            ("ch", 2),
            ("swap", 2),
            ("ccx", 3),
            ("cswap", 3),
        ];
        for &(name, k) in names {
            check_unitary(name, &[], k);
        }
        for &(name, k, np) in &[
            ("rx", 1usize, 1usize),
            ("ry", 1, 1),
            ("rz", 1, 1),
            ("u1", 1, 1),
            ("p", 1, 1),
            ("u2", 1, 2),
            ("u3", 1, 3),
            ("u", 1, 3),
            ("cp", 2, 1),
            ("crx", 2, 1),
            ("cry", 2, 1),
            ("crz", 2, 1),
            ("cu3", 2, 3),
            ("rxx", 2, 1),
            ("rzz", 2, 1),
        ] {
            let params: Vec<f64> = (0..np).map(|i| 0.3 + 0.71 * i as f64).collect();
            check_unitary(name, &params, k);
        }
    }

    fn check_unitary(name: &str, params: &[f64], k: usize) {
        let dim = 1usize << k;
        let m = as_matrix(build(name, params).unwrap(), k);
        // M† M = I
        for i in 0..dim {
            for j in 0..dim {
                let mut acc = C64::default();
                for l in 0..dim {
                    acc += m[l * dim + i].conj() * m[l * dim + j];
                }
                let expect = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (acc - C64::new(expect, 0.0)).norm() < 1e-12,
                    "{name} not unitary at ({i},{j}): {acc}"
                );
            }
        }
    }

    #[test]
    fn cx_flips_target_when_control_set() {
        // qubits = [control, target]; control = bit 0.
        let m = as_matrix(build("cx", &[]).unwrap(), 2);
        // |c=1, t=0> = index 1 → |c=1, t=1> = index 3
        assert_eq!(m[3 * 4 + 1], c(1.0, 0.0));
        assert_eq!(m[1 * 4 + 3], c(1.0, 0.0));
        assert_eq!(m[0], c(1.0, 0.0));
        assert_eq!(m[2 * 4 + 2], c(1.0, 0.0));
    }
}
