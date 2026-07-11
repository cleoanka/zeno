//! OpenQASM 2.0 front end.
//!
//! Interface contract (the rest of the crate depends on exactly this):
//! - `parse_str(&str) -> Result<Program, QasmError>`
//! - `parse_file(&Path) -> Result<Program, Error>`
//! - `QasmError { line, col, msg }` with `Display`.

use crate::ir::Program;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QasmError {
    pub line: usize,
    pub col: usize,
    pub msg: String,
}

impl std::fmt::Display for QasmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}:{}: {}", self.line, self.col, self.msg)
    }
}

impl std::error::Error for QasmError {}

pub fn parse_str(_src: &str) -> Result<Program, QasmError> {
    Err(QasmError {
        line: 0,
        col: 0,
        msg: "QASM front end not yet wired".into(),
    })
}

pub fn parse_file(path: &Path) -> Result<Program, crate::Error> {
    let src = std::fs::read_to_string(path)?;
    Ok(parse_str(&src)?)
}
