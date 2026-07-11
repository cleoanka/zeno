//! Metal GPU backend (f32 state vector in unified memory).
//!
//! Contract:
//! - `MetalBackend::new(n_qubits) -> Result<Self, Error>`
//! - implements [`crate::exec::Backend`]
//! - f32 only (Metal has no f64); the executor enforces precision.
//! - buffers are `StorageModeShared` so reductions/sampling can run on the
//!   CPU over the same memory without copies (Apple Silicon unified memory).

use crate::exec::Backend;
use crate::ir::C64;
use crate::Error;

pub struct MetalBackend {
    _private: (),
}

impl MetalBackend {
    pub fn new(_n_qubits: u32) -> Result<Self, Error> {
        Err(Error::Unsupported(
            "Metal backend not implemented yet".into(),
        ))
    }
}

impl Backend for MetalBackend {
    fn name(&self) -> &'static str {
        "metal-f32"
    }
    fn apply_unitary(&mut self, _qs: &[u32], _mat: &[C64]) {
        unimplemented!()
    }
    fn apply_diagonal(&mut self, _qs: &[u32], _d: &[C64]) {
        unimplemented!()
    }
    fn measure(&mut self, _q: u32, _u: f64) -> bool {
        unimplemented!()
    }
    fn reset_all(&mut self) {
        unimplemented!()
    }
    fn sample(&mut self, _shots: usize, _seed: u64) -> Vec<u64> {
        unimplemented!()
    }
    fn statevector(&mut self) -> Vec<C64> {
        unimplemented!()
    }
    fn norm_sqr(&mut self) -> f64 {
        unimplemented!()
    }
}
