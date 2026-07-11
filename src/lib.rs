//! # kuantum
//!
//! Apple Silicon-native quantum circuit **simulator, compiler and runner**.
//!
//! - Split-complex (SoA) state vector; NEON-friendly, memory-bandwidth-lean
//!   kernels parallelized with rayon across P+E cores.
//! - A fusion compiler (cancellation → diagonal fusion → ≤5-qubit gate
//!   fusion) so deep circuits become a handful of memory sweeps.
//! - RAM-aware capacity planning: pick f64/f32 per run, refuse politely
//!   when a state can't fit, and tell you what *would* fit.
//! - OpenQASM 2.0 front end and a plain Rust builder API.
//!
//! ```
//! use kuantum::{Circuit, Simulator};
//!
//! let mut bell = Circuit::new(2);
//! bell.h(0).cx(0, 1).measure_all();
//! let r = Simulator::new().shots(1000).seed(42).run(&bell).unwrap();
//! assert_eq!(r.counts.get("00") + r.counts.get("11"), 1000);
//! ```

pub mod circuit;
pub mod compiler;
pub mod exec;
pub mod gates;
pub mod ir;
pub mod mem;
#[cfg(feature = "metal")]
pub mod metal;
pub mod qasm;
pub mod sample;
pub mod state;

pub use circuit::{qft, random_circuit, Circuit};
pub use exec::{Backend, Counts, RunOptions, RunResult};
pub use ir::{Program, C64};

/// Floating-point width of the state vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Precision {
    F32,
    F64,
}

impl Precision {
    /// Bytes per amplitude (re + im).
    pub fn bytes_per_amp(self) -> u64 {
        match self {
            Precision::F32 => 8,
            Precision::F64 => 16,
        }
    }
}

impl std::fmt::Display for Precision {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Precision::F32 => "f32",
            Precision::F64 => "f64",
        })
    }
}

/// Which engine applies the gates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BackendChoice {
    #[default]
    Auto,
    Cpu,
    Metal,
}

/// Crate-wide error type.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("QASM parse error: {0}")]
    Qasm(#[from] qasm::QasmError),

    #[error(
        "state vector for {n_qubits} qubits at {precision} needs {} but the budget is {} \
         (max {max_qubits} qubits at this precision; raise --mem-limit or lower qubits)",
        human_bytes(*needed_bytes),
        human_bytes(*budget_bytes as u128)
    )]
    Memory {
        n_qubits: u32,
        precision: Precision,
        needed_bytes: u128,
        budget_bytes: u64,
        max_qubits: u32,
    },

    #[error("invalid circuit: {0}")]
    InvalidCircuit(String),

    #[error("{0}")]
    Unsupported(String),

    #[error(transparent)]
    Io(#[from] std::io::Error),
}

/// `1536 → "1.5 KiB"` — for error messages and the CLI.
pub fn human_bytes(b: u128) -> String {
    const UNITS: &[&str] = &["B", "KiB", "MiB", "GiB", "TiB"];
    let mut v = b as f64;
    let mut u = 0;
    while v >= 1024.0 && u + 1 < UNITS.len() {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{b} B")
    } else {
        format!("{v:.1} {}", UNITS[u])
    }
}

/// Builder-style front door around [`RunOptions`].
///
/// ```
/// use kuantum::{Circuit, Simulator, Precision};
/// let mut c = Circuit::new(3);
/// c.h(0).cx(0, 1).cx(1, 2).measure_all();
/// let r = Simulator::new()
///     .shots(2048)
///     .seed(7)
///     .precision(Precision::F32)
///     .run(&c)
///     .unwrap();
/// assert_eq!(r.counts.len(), 2); // GHZ: all zeros or all ones
/// ```
#[derive(Debug, Clone, Default)]
pub struct Simulator {
    opts: RunOptions,
}

impl Simulator {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn shots(mut self, shots: u64) -> Self {
        self.opts.shots = shots;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.opts.seed = Some(seed);
        self
    }

    pub fn precision(mut self, p: Precision) -> Self {
        self.opts.precision = Some(p);
        self
    }

    pub fn backend(mut self, b: BackendChoice) -> Self {
        self.opts.backend = b;
        self
    }

    /// Maximum dense fused-gate width in qubits (0 disables all fusion,
    /// including diagonal fusion). Default is automatic: 1 on CPU, 5 on
    /// Metal — measured on M4 Pro, dense fusion is a GPU win but a CPU
    /// loss, while diagonal fusion (always on when this is ≥ 1) wins
    /// everywhere.
    pub fn fusion(mut self, kmax: u8) -> Self {
        self.opts.fusion_max = Some(kmax);
        self
    }

    pub fn mem_limit(mut self, bytes: u64) -> Self {
        self.opts.mem_limit = Some(bytes);
        self
    }

    pub fn threads(mut self, n: usize) -> Self {
        self.opts.threads = Some(n);
        self
    }

    pub fn options(&self) -> &RunOptions {
        &self.opts
    }

    pub fn run(&self, c: &Circuit) -> Result<RunResult, Error> {
        self.run_program(&c.to_program())
    }

    pub fn run_program(&self, p: &Program) -> Result<RunResult, Error> {
        run_program(p, &self.opts)
    }

    pub fn run_qasm(&self, src: &str) -> Result<RunResult, Error> {
        run_program(&qasm::parse_str(src)?, &self.opts)
    }

    /// Final state vector of a measurement-free (or trailing-measure-only)
    /// circuit.
    pub fn statevector(&self, c: &Circuit) -> Result<Vec<C64>, Error> {
        let mut opts = self.opts.clone();
        opts.want_statevector = true;
        opts.shots = 0;
        let r = run_program_with(&c.to_program(), &opts)?;
        Ok(r.statevector.expect("requested statevector"))
    }
}

fn compile_options(opts: &RunOptions) -> compiler::CompileOptions {
    compiler::CompileOptions {
        fusion_max: opts.fusion_max.unwrap_or(match opts.backend {
            BackendChoice::Metal => 5,
            _ => 1,
        }),
        ..Default::default()
    }
}

fn run_program_with(p: &Program, opts: &RunOptions) -> Result<RunResult, Error> {
    let compiled = compiler::compile(p, &compile_options(opts))?;
    exec::run_compiled(&compiled, opts)
}

/// Compile and run a [`Program`].
pub fn run_program(p: &Program, opts: &RunOptions) -> Result<RunResult, Error> {
    run_program_with(p, opts)
}

/// Parse, compile and run OpenQASM 2.0 source.
pub fn run_qasm_str(src: &str, opts: &RunOptions) -> Result<RunResult, Error> {
    run_program(&qasm::parse_str(src)?, opts)
}

/// Parse, compile and run an OpenQASM 2.0 file.
pub fn run_qasm_file(path: &std::path::Path, opts: &RunOptions) -> Result<RunResult, Error> {
    run_program(&qasm::parse_file(path)?, opts)
}

/// Largest qubit count that fits the default budget (75% of RAM).
pub fn max_qubits_default(precision: Precision) -> u32 {
    mem::max_qubits(mem::budget_bytes(None, 0.75), precision)
}
