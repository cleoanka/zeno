//! Ergonomic circuit builder (single `q` register, single `c` register).
//!
//! ```
//! use zeno::{Circuit, Simulator};
//!
//! let mut c = Circuit::new(2);
//! c.h(0).cx(0, 1).measure_all();
//! let result = Simulator::new().shots(1000).seed(7).run(&c).unwrap();
//! assert_eq!(result.counts.get("00") + result.counts.get("11"), 1000);
//! ```
//!
//! For multi-register programs or classical control, build an
//! [`crate::ir::Program`] directly or use the QASM front end.

use crate::ir::{GateInstr, Instr, Program, Reg};

#[derive(Debug, Clone)]
pub struct Circuit {
    n_qubits: u32,
    instrs: Vec<Instr>,
}

macro_rules! gate0 {
    ($($name:ident),*) => {
        $(pub fn $name(&mut self, q: u32) -> &mut Self {
            self.gate(stringify!($name), &[q], &[])
        })*
    };
}

macro_rules! gate0_2q {
    ($($name:ident),*) => {
        $(pub fn $name(&mut self, a: u32, b: u32) -> &mut Self {
            self.gate(stringify!($name), &[a, b], &[])
        })*
    };
}

macro_rules! gate1p {
    ($($name:ident),*) => {
        $(pub fn $name(&mut self, q: u32, theta: f64) -> &mut Self {
            self.gate(stringify!($name), &[q], &[theta])
        })*
    };
}

impl Circuit {
    pub fn new(n_qubits: u32) -> Self {
        Circuit {
            n_qubits,
            instrs: vec![],
        }
    }

    pub fn n_qubits(&self) -> u32 {
        self.n_qubits
    }

    /// Append a gate by name (see [`crate::gates`] for the native set).
    pub fn gate(&mut self, name: &str, qubits: &[u32], params: &[f64]) -> &mut Self {
        self.instrs.push(Instr::Gate(GateInstr {
            name: name.to_string(),
            params: params.to_vec(),
            qubits: qubits.to_vec(),
        }));
        self
    }

    gate0!(x, y, z, h, s, sdg, t, tdg, sx, sxdg);
    gate0_2q!(cx, cy, cz, ch, swap);
    gate1p!(rx, ry, rz, p);

    pub fn u(&mut self, q: u32, theta: f64, phi: f64, lambda: f64) -> &mut Self {
        self.gate("u3", &[q], &[theta, phi, lambda])
    }

    pub fn cp(&mut self, control: u32, target: u32, lambda: f64) -> &mut Self {
        self.gate("cp", &[control, target], &[lambda])
    }

    pub fn crz(&mut self, control: u32, target: u32, theta: f64) -> &mut Self {
        self.gate("crz", &[control, target], &[theta])
    }

    pub fn rzz(&mut self, a: u32, b: u32, theta: f64) -> &mut Self {
        self.gate("rzz", &[a, b], &[theta])
    }

    pub fn ccx(&mut self, c0: u32, c1: u32, target: u32) -> &mut Self {
        self.gate("ccx", &[c0, c1, target], &[])
    }

    pub fn cswap(&mut self, control: u32, a: u32, b: u32) -> &mut Self {
        self.gate("cswap", &[control, a, b], &[])
    }

    pub fn measure(&mut self, qubit: u32, clbit: u32) -> &mut Self {
        self.instrs.push(Instr::Measure { qubit, clbit });
        self
    }

    pub fn measure_all(&mut self) -> &mut Self {
        for q in 0..self.n_qubits {
            self.measure(q, q);
        }
        self
    }

    pub fn reset(&mut self, qubit: u32) -> &mut Self {
        self.instrs.push(Instr::Reset { qubit });
        self
    }

    /// Optimization fence (blocks cancellation and fusion across it).
    pub fn barrier(&mut self) -> &mut Self {
        self.instrs
            .push(Instr::Barrier((0..self.n_qubits).collect()));
        self
    }

    pub fn to_program(&self) -> Program {
        Program {
            qregs: vec![Reg {
                name: "q".into(),
                size: self.n_qubits,
            }],
            cregs: vec![Reg {
                name: "c".into(),
                size: self.n_qubits,
            }],
            instrs: self.instrs.clone(),
        }
    }
}

/// n-qubit quantum Fourier transform (no swaps at the end), a standard
/// fusion benchmark: the controlled-phase ladder is entirely diagonal.
pub fn qft(n: u32) -> Circuit {
    let mut c = Circuit::new(n);
    for i in (0..n).rev() {
        c.h(i);
        for j in (0..i).rev() {
            let angle = std::f64::consts::PI / f64::from(1u32 << (i - j));
            c.cp(j, i, angle);
        }
    }
    c
}

/// Seeded random circuit: `depth` layers of Haar-ish single-qubit rotations
/// followed by a shifting CX ladder. Used by `zeno bench` and the tests.
pub fn random_circuit(n: u32, depth: u32, seed: u64) -> Circuit {
    use rand::{Rng, SeedableRng};
    let mut rng = rand_xoshiro::Xoshiro256PlusPlus::seed_from_u64(seed);
    let mut c = Circuit::new(n);
    let tau = std::f64::consts::TAU;
    for layer in 0..depth {
        for q in 0..n {
            c.u(
                q,
                rng.gen::<f64>() * tau,
                rng.gen::<f64>() * tau,
                rng.gen::<f64>() * tau,
            );
        }
        let offset = layer % 2;
        let mut q = offset;
        while q + 1 < n {
            c.cx(q, q + 1);
            q += 2;
        }
    }
    c
}
