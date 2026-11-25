# Changelog

All notable changes are recorded here. Dates are release dates.

> **Note on the two 2026-07-11 releases.** `v0.1.0` and `v0.2.0` are tagged on
> the same day on purpose: `v0.1.0` was the first working, tagged core, and
> `v0.2.0` (NEON kernels, noise channels, OpenQASM 3, `zeno demo`) followed a
> few hours later as an intentional same-day double release. This is why the
> test count jumps from 142 (v0.1.0) to 270 (v0.2.0, full Metal build) between
> two entries with the same date.

## Unreleased

Inputs that used to be accepted now fail with a clean error (exit 1):

- **Non-finite gate parameters** (`rx(ln(0))`, `rx(1/0)`) are rejected by the
  compiler (`gate 'rx' parameter #0 is not finite`). Before, static circuits
  panicked in the sampler (exit 101) and dynamic ones silently measured 0.
- **More than 64 classical bits in total** are rejected at the declaration in
  both front ends. Before, `creg a[4294967295]; creg b[2];` wrapped to 1 in
  `u32` and slipped past the compiler's limit; `Program::n_qubits` /
  `n_clbits` now saturate instead of wrapping.
- **Gate expansion producing more than 10,000,000 instructions** in one file
  is rejected at the call site (line:col), before expanding. Before, a few
  hundred bytes of nested `gate` definitions could exhaust memory.
- **Invalid self-inverse pairs** (`h(99).h(99)`, `cx(0,0).cx(0,0)` from the
  Rust builder) now error like a single invalid gate does, instead of being
  cancelled away silently. The QASM front ends were not affected.

Fixes:

- `circuit::qft(n)` for `n >= 33`: controlled-phase angles are computed with
  `2f64.powi`, so `cp(0, 32)` gets π/2³² instead of a wrapped shift (π).
  Angles for `n <= 32` are bit-identical.
- Rustdoc and CLI help: fusion default (auto: 1 on CPU, 5 on Metal) and
  OpenQASM 3 (subset) support are now described correctly.

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
