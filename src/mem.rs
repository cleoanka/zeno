//! RAM detection and qubit-capacity planning.
//!
//! A state vector of `n` qubits needs `2^n` complex amplitudes. We store
//! amplitudes as split real/imaginary arrays, so the cost is
//! `2^n * 2 * sizeof(float)` bytes: 8 B/amplitude in f32, 16 B in f64.

use crate::Precision;

/// Total physical RAM in bytes. `ZENO_MEM_BYTES` overrides (useful for
/// tests and for pretending to be a smaller machine).
pub fn physical_ram_bytes() -> u64 {
    if let Ok(v) = std::env::var("ZENO_MEM_BYTES") {
        if let Ok(bytes) = v.parse::<u64>() {
            return bytes;
        }
    }
    #[cfg(target_os = "macos")]
    {
        let mut value: u64 = 0;
        let mut len = std::mem::size_of::<u64>();
        let name = std::ffi::CString::new("hw.memsize").unwrap();
        let rc = unsafe {
            libc::sysctlbyname(
                name.as_ptr(),
                &mut value as *mut u64 as *mut libc::c_void,
                &mut len,
                std::ptr::null_mut(),
                0,
            )
        };
        if rc == 0 && value > 0 {
            return value;
        }
    }
    // Conservative fallback for non-macOS builds (CI etc.).
    8 << 30
}

/// Best-effort "free right now" estimate from `vm_stat`
/// (free + inactive + purgeable + speculative pages).
#[cfg(target_os = "macos")]
pub fn available_now_bytes() -> Option<u64> {
    let out = std::process::Command::new("vm_stat").output().ok()?;
    let text = String::from_utf8_lossy(&out.stdout);
    let page: u64 = text
        .lines()
        .next()?
        .split("page size of")
        .nth(1)?
        .split_whitespace()
        .next()?
        .parse()
        .ok()?;
    let mut pages: u64 = 0;
    for line in text.lines() {
        let want = [
            "Pages free",
            "Pages inactive",
            "Pages purgeable",
            "Pages speculative",
        ]
        .iter()
        .any(|k| line.starts_with(k));
        if want {
            if let Some(v) = line.split(':').nth(1) {
                pages += v.trim().trim_end_matches('.').parse::<u64>().unwrap_or(0);
            }
        }
    }
    Some(pages * page)
}

#[cfg(not(target_os = "macos"))]
pub fn available_now_bytes() -> Option<u64> {
    None
}

/// State-vector size in bytes for `n` qubits at `prec`.
pub fn state_bytes(n_qubits: u32, prec: Precision) -> u128 {
    (1u128 << n_qubits) * prec.bytes_per_amp() as u128
}

/// Largest qubit count whose state vector fits into `budget` bytes.
pub fn max_qubits(budget: u64, prec: Precision) -> u32 {
    let mut n = 0u32;
    while state_bytes(n + 1, prec) <= budget as u128 {
        n += 1;
        if n >= 60 {
            break;
        }
    }
    n
}

/// Resolved memory plan for a run.
#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct MemPlan {
    pub n_qubits: u32,
    pub precision: Precision,
    pub needed_bytes: u64,
    pub budget_bytes: u64,
}

/// Compute the memory budget: an explicit limit wins, otherwise
/// `fraction` of physical RAM.
pub fn budget_bytes(mem_limit: Option<u64>, fraction: f64) -> u64 {
    match mem_limit {
        Some(b) => b,
        None => (physical_ram_bytes() as f64 * fraction) as u64,
    }
}

/// Check that `n_qubits` at `prec` fits in `budget`.
pub fn plan(n_qubits: u32, prec: Precision, budget: u64) -> Result<MemPlan, crate::Error> {
    let needed = state_bytes(n_qubits, prec);
    if needed > budget as u128 {
        return Err(crate::Error::Memory {
            n_qubits,
            precision: prec,
            needed_bytes: needed,
            budget_bytes: budget,
            max_qubits: max_qubits(budget, prec),
        });
    }
    Ok(MemPlan {
        n_qubits,
        precision: prec,
        needed_bytes: needed as u64,
        budget_bytes: budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_math() {
        // 24 GiB budget: 2^30 * 16 B = 16 GiB fits in f64, 2^31 doesn't.
        let budget = 24u64 << 30;
        assert_eq!(max_qubits(budget, Precision::F64), 30);
        assert_eq!(max_qubits(budget, Precision::F32), 31);
        assert_eq!(state_bytes(30, Precision::F64), 16 << 30);
        assert_eq!(state_bytes(30, Precision::F32), 8 << 30);
    }

    #[test]
    fn plan_rejects_oversize() {
        assert!(plan(30, Precision::F64, 1 << 30).is_err());
        assert!(plan(20, Precision::F64, 1 << 30).is_ok());
    }
}
