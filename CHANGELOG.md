# Changelog

## v0.2.0 — 2026-07-11

- **Explicit NEON kernels** (bit-exact by construction; scalar paths kept
  as fallback + test oracle): dense fused kernel vectorized across groups
  (24q fusion-5: 1.68× f64 / 2.91× f32), diagonal kernel streams at DRAM
  bandwidth (QFT-24 end to end: 1.5× f64 / 1.4× f32), q<2 single-qubit
  slices 1.3–2.6×.
- **Noise channels** (trajectory sampling): depolarizing 1q/2q, bit/phase
  flip, amplitude damping (exact jump/no-jump unraveling), readout error;
  `--noise` accepts inline JSON, key=value pairs or a file; cross-validated
  against qiskit-aer noise models (TVD < 0.004 @ 100k shots). Noise forces
  per-shot execution with fusion disabled (docs/NOISE.md).
- **OpenQASM 3 front end** (documented subset): `qubit[n]`/`bit[n]`, both
  measure forms, `if` blocks, gate defs, broadcasting, `stdgates.inc`
  aliases, π/tau; version header auto-dispatches; unsupported features get
  named-feature errors (docs/QASM3.md).
- **`zeno demo`**: six built-in, self-explaining circuits (bell, ghz, qft,
  grover, teleport, noisy) — zero files; plus docs/TUTORIAL.md, a complete
  beginner's walkthrough with real transcripts.
- Metal threadgroup-memory staging: implemented, measured at 1.08× (below
  the 1.10× merge bar), reverted and documented in src/metal.rs — the
  roadmap item is measured-and-closed, not pending.
- `Backend::prob_one` (collapse-free) added for state-dependent channels.
- 270 tests (was 142), clippy-clean both feature sets.

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
  `zeno info` capacity table, `--mem-limit`/`ZENO_MEM_BYTES` overrides.
- OpenQASM 2.0 front end (registers, user gates, broadcasting, expressions,
  `if`/measure/reset/barrier) with line:col error reporting.
- Optional Metal GPU backend (`--features metal`, f32, unified memory).
- CLI: `run` (histogram, `--json`, `--statevector`), `info`, `bench`,
  `compile` (fusion visualizer).
