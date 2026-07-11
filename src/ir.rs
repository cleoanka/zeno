//! Circuit intermediate representation.
//!
//! Shared by the QASM front end ([`crate::qasm`]), the builder API
//! ([`crate::circuit`]) and the compiler ([`crate::compiler`]).
//!
//! Conventions (identical everywhere in this crate):
//! - Qubits are numbered globally, little-endian: bit `q` of a basis-state
//!   index holds the state of qubit `q`.
//! - A gate's `qubits` list is in *argument order* (e.g. `[control, target]`
//!   for `cx`). Matrices produced by [`crate::gates`] use the same order:
//!   bit `b` of a local matrix index corresponds to `qubits[b]`.

use num_complex::Complex64;

/// Complex amplitude type used throughout the crate.
pub type C64 = Complex64;

/// A quantum or classical register: name and size.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct Reg {
    pub name: String,
    pub size: u32,
}

/// A single gate application, unresolved (by name + params).
#[derive(Debug, Clone, PartialEq)]
pub struct GateInstr {
    pub name: String,
    pub params: Vec<f64>,
    /// Global qubit indices, in the gate's own argument order.
    pub qubits: Vec<u32>,
}

/// One program instruction.
#[derive(Debug, Clone, PartialEq)]
pub enum Instr {
    Gate(GateInstr),
    Measure {
        qubit: u32,
        clbit: u32,
    },
    Reset {
        qubit: u32,
    },
    Barrier(Vec<u32>),
    /// `if (creg == value) op;` — `creg` indexes into [`Program::cregs`].
    If {
        creg: usize,
        value: u64,
        op: Box<Instr>,
    },
}

/// A full program: registers plus a flat instruction list.
///
/// Qubit index space is the concatenation of `qregs` in declaration order;
/// same for clbits and `cregs`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Program {
    pub qregs: Vec<Reg>,
    pub cregs: Vec<Reg>,
    pub instrs: Vec<Instr>,
}

impl Program {
    pub fn n_qubits(&self) -> u32 {
        self.qregs.iter().map(|r| r.size).sum()
    }

    pub fn n_clbits(&self) -> u32 {
        self.cregs.iter().map(|r| r.size).sum()
    }

    /// Offset of qreg `idx` in the global qubit index space.
    pub fn qreg_offset(&self, idx: usize) -> u32 {
        self.qregs[..idx].iter().map(|r| r.size).sum()
    }

    /// Offset of creg `idx` in the global clbit index space.
    pub fn creg_offset(&self, idx: usize) -> u32 {
        self.cregs[..idx].iter().map(|r| r.size).sum()
    }
}
