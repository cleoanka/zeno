# Architecture

`zeno` is three machines bolted together: a **compiler** that shrinks a
circuit, a **kernel core** that sweeps a state vector, and an **executor**
that decides how many times and on what hardware the sweeps run.

```
QASM 2.0 / 3 ──parse──▶ ir::Program ──compile──▶ [COp] ──execute──▶ Counts / statevector
Rust builder ──────────▲                           │        ▲
                                                   │   noise channels (trajectory, per shot)
                                                   ├── CPU backend (f32/f64, rayon + explicit NEON)
                                                   └── Metal backend (f32, unified memory)
```

## Conventions (everything depends on these)

- **Little-endian qubits.** Bit `q` of a basis-state index is the state of
  qubit `q`.
- **Argument-order matrices.** A gate's qubit list is in call order
  (`cx = [control, target]`); a `2^k×2^k` matrix from `src/gates.rs` maps
  bit `b` of its local index to `qubits[b]`. The compiler permutes every
  matrix to **sorted-qubit convention** before it reaches a kernel.
- **Counts keys** follow qiskit: cregs joined by spaces, last-declared
  leftmost; within a creg, bit `k-1 … 0` left to right.

## State layout (`src/state.rs`)

Amplitudes are stored **split** — `re: Vec<T>`, `im: Vec<T>` (SoA), `T ∈
{f32, f64}`. Two reasons:

1. NEON loves it: complex multiply-add becomes plain FMA loops over two
   contiguous streams, no shuffles. LLVM autovectorizes the kernels with
   `-C target-cpu=native` (see `.cargo/config.toml`).
2. Probability math (`re² + im²`) reads each stream once.

On top of the autovectorized run-walk loops, the hot kernels carry explicit
NEON paths (aarch64 only; scalar kernels remain as the fallback *and* the
test oracle). The vectorization is **bit-exact by design**: each lane holds
one independent amplitude computed with the identical mul/add/sub expression
tree — no FMA, no reassociation — verified by `to_bits`-equality tests and
by the Metal bit-parity suite. The dense fused kernel vectorizes *across
groups* (lane = group) for the same reason: within-matvec vectorization
would reorder the accumulation.

Gate kernels are **run-walk** loops: for a target qubit `q` with stride
`s = 2^q`, the half-index space `0..2^(n-1)` is cut into contiguous chunks
(rayon tasks); inside a chunk the kernel walks maximal runs of length ≤ `s`
so that both sides of the butterfly (`i`, `i+s`) are contiguous slices —
vectorizable at every `q`, cache-line friendly, no gather except in the
fused path.

Three kernels cover everything:

| Kernel | Cost per amplitude | Used for |
|---|---|---|
| `apply_1q` | 4 FMA-pairs, 2 streams | leftover 1-qubit gates |
| `apply_kq` (k ≤ 6) | `2^k` cmul per amp, gather/scatter | fused gates |
| `apply_diag` | 1 cmul | fused diagonal runs |

Measurement (`prob_one` → `collapse`) accumulates probabilities in f64
even for f32 states.

## Compiler (`src/compiler.rs`)

Order matters; each pass feeds the next:

1. **Cancellation** — adjacent self-inverse pairs (`h h`, `cx cx`) die,
   looking through gates on disjoint qubits. Barriers fence.
2. **Diagonal fusion** — diagonal gates commute with each other and with
   anything disjoint, so runs of `rz/t/s/cz/cp/crz/rzz/…` collapse into one
   table of ≤ 2^10 phases by default (cap 2^12) applied in a single sweep.
   A QFT's whole controlled-phase ladder becomes one diagonal per layer.
   This pass runs whenever fusion isn't disabled outright — it is a
   memory-bound win on every backend.
3. **Dense fusion** — greedy absorb-while-support-≤-kmax: matrices are
   embedded into the union support and multiplied. Width is
   **backend-dependent by default**: 1 on CPU, 5 on Metal. Measured on
   M4 Pro, trading five memory sweeps for one 2^k×2^k matmul sweep wins
   ~2× on the GPU but *loses* ~3× on the CPU, where the plain 1-qubit
   run-walk kernel is already the fastest thing in the crate (`--fusion`
   overrides).
4. **Finalize** — trailing measurements split off; anything dynamic
   (mid-circuit measure, `reset`, `if`) flags the circuit for per-shot
   execution.

## Executor (`src/exec.rs`, `src/sample.rs`)

- **Static circuits** run once; shots are drawn from |ψ|² analytically in
  O(2ⁿ + shots·log shots): chunk masses → route sorted uniforms → one scan
  per touched chunk. 1 shot and 10⁶ shots cost nearly the same.
- **Dynamic circuits** re-run per shot. Shots parallelize across cores when
  `state_bytes × threads` fits the budget (they usually do — dynamic
  circuits tend to be small); each shot gets a `splitmix64(seed ^ shot)`
  RNG so results are deterministic under any thread schedule.
- **Memory planning** (`src/mem.rs`) reads `hw.memsize`, budgets 75% by
  default, auto-falls-back f64 → f32 with a notice, and errors with the
  exact capacity table otherwise.

## Metal backend (`src/metal.rs`, `--features metal`)

f32 state in two `StorageModeShared` buffers — Apple Silicon unified memory
means the CPU samples/measures **the same bytes** the GPU wrote, zero
copies. Gates are encoded lazily into command buffers and flushed only when
a read (measure/sample/statevector) needs the data. Reductions and sampling
run on the CPU over the shared buffers; the GPU does what it's good at:
embarrassingly parallel butterfly sweeps.

## Noise (`src/noise.rs`)

Noise is trajectory-sampled: each shot is an independent stochastic
evolution, with channels (depolarizing 1q/2q, bit/phase flip, amplitude
damping, readout flips) applied after each *compiled-as-written* gate — so
a non-trivial model forces fusion off and per-shot execution. Amplitude
damping uses the exact jump/no-jump unraveling (jump probability
`γ·P(q=1)` via `Backend::prob_one`); everything else is a state-independent
Pauli mixture, which is why trajectories are cheap. Semantics, RNG-stream
layout and the analytic derivations live in docs/NOISE.md; the suite
cross-checks qiskit-aer's noise models to TVD < 0.004 at 100k shots.

## Front ends (`src/qasm.rs`, `src/qasm3.rs`)

`qasm::parse_str` sniffs the version header (comment-aware) and routes to
the OpenQASM 2.0 parser or the OpenQASM 3 subset parser (`qubit[n]`/
`bit[n]`, both measure forms, `if` blocks, gate defs, `stdgates.inc`
aliases). Both emit the same `ir::Program`; unsupported OQ3 features are
rejected with named-feature errors (docs/QASM3.md).

## Testing strategy

`tests/reference.rs` contains an independently written dense simulator
(interleaved storage, no fusion, no threads — deliberately different) and
compares full state vectors against it for every native gate in multiple
argument orders, random circuits at every fusion level, both precisions.
`tests/analytic.rs` pins closed-form results (GHZ, QFT phases, Grover,
teleportation statistics). The Metal backend must match the CPU backend
bit-for-bit on counts (same seed) and to 1e-5 on amplitudes.
