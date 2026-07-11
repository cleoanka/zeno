# Performance

All numbers: **Apple M4 Pro** (8P+4E, 12 threads), 24 GB, macOS 15,
`cargo build --release` (with the repo's `-C target-cpu=native`),
kuantum v0.1.0, medians of 3 runs on a normally-loaded desktop.
Every command is reproducible as written.

## Head-to-head vs qiskit-aer (identical OpenQASM files)

qiskit 2.5.0 + qiskit-aer 0.17.2, `AerSimulator(method="statevector")`
(f64, OpenMP multithreaded), transpile excluded from timing; kuantum
`sim_time` (parse/compile excluded), 1024 shots both.

| circuit | aer f64 | kuantum cpu f64 | kuantum cpu f32 | kuantum metal f32 |
|---|---|---|---|---|
| random 20q, depth 12 | 64 ms | **40 ms** (1.6×) | 32 ms | — |
| random 24q, depth 12 | 894 ms | 900 ms (1.0×) | 450 ms | **253 ms** (3.5×) |
| random 26q, depth 12 | 3.83 s | 3.86 s (1.0×) | 2.05 s | **0.95 s** (4.0×) |
| random 24q, depth 40 | 3.39 s | **2.99 s** (1.1×) | 1.58 s | **0.73 s** (4.7×) |
| QFT 24q | 607 ms | **375 ms** (1.6×) | 290 ms | **140 ms** (4.3×) |

"random" = brickwork: a layer of Haar-ish `u3` on every qubit + an
alternating CX ladder per layer. Same file, same seed, loaded by both
engines. Distributions were separately cross-validated against aer to
max |Δamp| < 1e-9 (f64) and TVD < 0.01 at 100k shots on 60+ circuits.

f32 vs f64 note: aer's statevector method is double-precision, so the f32
columns are a *capacity/speed trade* kuantum offers, not an apples-to-apples
engine comparison. The f64 column is the fair fight.

## CPU scaling (brickwork depth 12, defaults)

| qubits | f64 | f32 | f32 throughput |
|---|---|---|---|
| 20 | 40 ms | 32 ms | 11.6 G amp-updates/s |
| 22 | 155 ms | 86 ms | 19.0 G |
| 24 | 908 ms | 450 ms | 15.8 G |
| 26 | 3.83 s | 2.05 s | 15.2 G |
| 28 | 16.2 s | 8.6 s | 15.3 G |

Reproduce with
`kuantum bench --qubits 20,22,24,26,28 --depth 12 --precision f32`.

f32 runs ~1.9× faster than f64 at scale — the kernels are memory-bound,
so halving the bytes halves the time. It also buys one extra qubit at the
same RAM.

## Big states and dynamics

- **GHZ-29** (8 GiB state, f64): ~9.2 s end-to-end, peak RSS ≈ 7.2 GiB
  (`kuantum run ghz29.qasm --shots 1024`). Capacity on 24 GB: 30 qubits
  f64 / 31 qubits f32 (`kuantum info`).
- **Teleportation** (dynamic: mid-circuit measures + `if`):
  1M shots in 73 ms ≈ **13.6 M shots/s**
  (`kuantum run examples/teleport.qasm --shots 1000000`).

## Why the defaults look the way they do

- **Dense fusion is off on CPU (auto width 1), on for Metal (width 5).**
  Measured: fusing to 5-qubit matrices made the CPU 2.6–4.3× *slower*
  (the dense 2^k×2^k matmul is compute-bound; the plain 1-qubit run-walk
  sweep is memory-bound and already fast), while the GPU gains ~2.2×
  from fatter dispatches. `--fusion K` overrides either way.
- **Diagonal fusion is always on** (unless `--fusion 0`): a QFT's whole
  controlled-phase ladder collapses into a handful of table sweeps —
  that's most of the 1.6× QFT win.
- **CX is a permutation, not math**: it executes as a swap sweep over the
  control=1 half with zero arithmetic. This alone took the depth-12
  brickwork sweep from 1.97 s → 0.91 s at 24 qubits.
- **Sampling is analytic**: static circuits are simulated once; shots are
  drawn from |ψ|² in O(2ⁿ + shots·log shots). 1 shot and 10⁶ shots cost
  nearly the same.

## Known headroom (roadmap, honestly)

- `kern_1q` sustains ~110 GB/s of the M4 Pro's ~273 GB/s — it is
  issue-limited, not bandwidth-saturated. Explicit NEON interleaving
  could plausibly buy up to ~2× and would shift the fusion break-even.
- The dense fused kernel (`apply_kq`) is scalar; vectorizing it would
  make wider CPU fusion worthwhile and is the main reason aer keeps
  parity at 24–26q.
- Metal currently wins 2× over cpu-f32 (4–4.7× over f64/aer); a fused
  threadgroup-memory kernel should widen that.

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
