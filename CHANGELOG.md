# Changelog

## v0.1.0 — 2026-07-11

Initial release.

- Split-complex (SoA) state-vector core with NEON-friendly run-walk kernels:
  1-qubit fast path, fused k≤6-qubit gather/scatter kernel, single-sweep
  diagonal kernel; rayon parallelism across P+E cores.
- Fusion compiler: self-inverse cancellation → commuting diagonal fusion →
  greedy dense gate fusion, barrier fences. Dense-fusion width defaults
  per backend (1 on CPU, 5 on Metal — measured, not assumed). CX executes
  as a zero-arithmetic permutation sweep.
- Executor: analytic O(2ⁿ + shots·log shots) sampling for static circuits; per-shot
  dynamic path (mid-circuit measure, reset, `if`) with parallel shots when
  the memory budget allows; deterministic seeding everywhere.
- RAM-aware capacity planning: f64/f32 per run, auto-precision fallback,
  `kuantum info` capacity table, `--mem-limit`/`KUANTUM_MEM_BYTES` overrides.
- OpenQASM 2.0 front end (registers, user gates, broadcasting, expressions,
  `if`/measure/reset/barrier) with line:col error reporting.
- Optional Metal GPU backend (`--features metal`, f32, unified memory).
- CLI: `run` (histogram, `--json`, `--statevector`), `info`, `bench`,
  `compile` (fusion visualizer).
