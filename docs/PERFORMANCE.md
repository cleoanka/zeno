# Performance

All numbers: **Apple M4 Pro** (8P+4E, 12 threads), 24 GB, macOS 15,
`cargo build --release` (with the repo's `-C target-cpu=native`),
zeno v0.2.0, medians of 3 runs on a normally-loaded desktop.
Every command is reproducible as written. qiskit-aer's cells are its best
observed session (aer's own `time_taken` varied >2× between our sessions;
zeno's did not — we compare against aer at its best).

## Head-to-head vs qiskit-aer (identical OpenQASM files)

qiskit 2.5.0 + qiskit-aer 0.17.2, `AerSimulator(method="statevector")`
(f64, OpenMP multithreaded), transpile excluded from timing; zeno
`sim_time` (parse/compile excluded), 1024 shots both.

| circuit | aer f64 | zeno cpu f64 | zeno cpu f32 | zeno metal f32 |
|---|---|---|---|---|
| random 20q, depth 12 | 64 ms | **40 ms** (1.6×) | 32 ms | — |
| random 24q, depth 12 | 894 ms | 900 ms (1.0×) | 450 ms | **253 ms** (3.5×) |
| random 26q, depth 12 | 3.83 s | 3.86 s (1.0×) | 2.05 s | **0.95 s** (4.0×) |
| random 24q, depth 40 | 3.39 s | **2.99 s** (1.1×) | 1.58 s | **0.73 s** (4.7×) |
| QFT 24q (no final swaps, as `qft()` builds it) | 607 ms | **258 ms** (2.4×) | 210 ms | **140 ms** (4.3×) |

"random" = brickwork: a layer of Haar-ish `u3` on every qubit + an
alternating CX ladder per layer. Same file, same seed, loaded by both
engines. Distributions were separately cross-validated against aer to
max |Δamp| < 1e-9 (f64) and TVD < 0.01 at 100k shots on 60+ circuits.

f32 vs f64 note: aer's statevector method is double-precision, so the f32
columns are a *capacity/speed trade* zeno offers, not an apples-to-apples
engine comparison. The f64 column is the fair fight.

## CPU scaling (brickwork depth 12, defaults)

| qubits | f64 | f32 |
|---|---|---|
| 20 | 40 ms | 29 ms |
| 22 | 159 ms | 79 ms |
| 24 | 926 ms | 477 ms |
| 26 | 3.98 s | 2.09 s |
| 28 | 16.2 s | 8.6 s |

Reproduce with
`zeno bench --qubits 20,22,24,26,28 --depth 12 --precision f32`.

f32 runs ~1.9× faster than f64 at scale — the kernels are memory-bound,
so halving the bytes halves the time. It also buys one extra qubit at the
same RAM.

## Big states and dynamics

- **GHZ-29** (8 GiB state, f64): ~9.2 s end-to-end, peak RSS ≈ 7.2 GiB
  (`zeno run ghz29.qasm --shots 1024`). Capacity on 24 GB: 30 qubits
  f64 / 31 qubits f32 (`zeno info`).
- **Teleportation** (dynamic: mid-circuit measures + `if`):
  1M shots in 73 ms ≈ **13.6 M shots/s**
  (`zeno run examples/teleport.qasm --shots 1000000`).

## Why the defaults look the way they do

- **Dense fusion is off on CPU (auto width 1), on for Metal (width 5).**
  Measured: fusing to 5-qubit matrices made the CPU 2.6–4.3× *slower*
  (the dense 2^k×2^k matmul is compute-bound; the plain 1-qubit run-walk
  sweep is memory-bound and already fast), while the GPU gains ~2.2×
  from fatter dispatches. `--fusion K` overrides either way. v0.2.0's NEON
  fused kernel narrowed the gap (24q fusion-5: 5.6 s → 3.3 s f64) but
  fusion 1 still wins (0.93 s) — the default stands, re-measured.
- **Diagonal fusion is always on** (unless `--fusion 0`): a QFT's whole
  controlled-phase ladder collapses into a handful of table sweeps —
  that's most of the 1.6× QFT win.
- **CX is a permutation, not math**: it executes as a swap sweep over the
  control=1 half with zero arithmetic. This alone took the depth-12
  brickwork sweep from 1.97 s → 0.91 s at 24 qubits.
- **Sampling is analytic**: static circuits are simulated once; shots are
  drawn from |ψ|² in O(2ⁿ + shots·log shots). 1 shot and 10⁶ shots cost
  nearly the same.

## v0.2.0 kernel work (what shipped, what didn't)

- **NEON, bit-exact by construction.** Every vector lane computes one
  independent amplitude with the identical mul/add/sub expression tree as
  the scalar code (no FMA — it rounds differently); the scalar kernels
  remain as the non-aarch64 fallback and the `to_bits`-equality test
  oracle, and the Metal bit-parity suite still passes.
- **Dense fused kernel (`apply_kq`)**: vectorized across groups (lane =
  group). 24q fusion-5 bench: 1.68× f64 / 2.91× f32; per-k microbench
  1.5–4.6×.
- **Diagonal kernel**: run-walks the lowest support qubit and streams
  vector multiplies — isolated sweeps went from 36–105 GB/s to
  139–260 GB/s (DRAM-saturated). QFT-24 end to end: 375 → 258 ms f64,
  290 → 210 ms f32.
- **1-qubit sweeps**: already DRAM-bound at the default sizes (LLVM
  auto-vectorizes the contiguous runs); explicit NEON only pays on the
  q<2 slices (1.3–2.6×), which is exactly what the QFT path hits.
- **Metal threadgroup staging: measured and closed.** Best tuning (64-wide
  threadgroups, float4 cooperative copy, gather overlap) reached 1.083×
  on deep 24q fusion-5 — below the 1.10× merge bar. Root cause: the
  matrix reads are dynamically uniform and already served on-chip; the
  residual cost is state gather/scatter, which staging cannot touch. The
  experiment is documented in `src/metal.rs` next to the kernel template.
- Remaining honest headroom: `apply_cx` and the q≥2 1-qubit sweeps sit at
  DRAM bandwidth — CPU gains from here require algorithmic change (wider
  fusion break-even sweep), not micro-optimization.

## Noise cost model

Noise (`--noise`) switches to per-shot trajectory execution with fusion
disabled — cost scales with shots × gates × 2ⁿ instead of the ideal
2ⁿ + shots·log shots. The noise test suite (17 analytic cases +
qiskit-aer cross-checks at TVD < 0.004) runs in ~1.5 s; budget
accordingly for large n × high shots.

## Method notes

- Times are `sim_time` from `--json` (gates + sampling; parse/compile
  excluded — compile is ~60 ms for a 20k-gate file, linear).
- Throughput metric: `input_gates × 2ⁿ / sim_time` ("amp-updates/s") —
  it deliberately counts *input* gates so different fusion settings stay
  comparable on the same circuit.
- Run-to-run spread on a live desktop was up to ~15% at small sizes;
  medians of 3 are reported. Nothing was cherry-picked; the perf review
  that produced these protocols was adversarial (findings that didn't
  reproduce were dropped).
