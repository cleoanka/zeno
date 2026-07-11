# Architecture

`kuantum` is three machines bolted together: a **compiler** that shrinks a
circuit, a **kernel core** that sweeps a state vector, and an **executor**
that decides how many times and on what hardware the sweeps run.

```
QASM 2.0 ──parse──▶ ir::Program ──compile──▶ [COp] ──execute──▶ Counts / statevector
Rust builder ──────────▲                        │
                                                ├── CPU backend (f32/f64, rayon + NEON)
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
   table of ≤ 2^10 phases applied in a single sweep. A QFT's whole
   controlled-phase ladder becomes one diagonal per layer.
3. **General fusion** — greedy absorb-while-support-≤-kmax (default 5,
   qiskit-aer's default): matrices are embedded into the union support and
   multiplied. A depth-20 dense circuit typically compiles to ~n/4 fused
   ops per layer — each one memory sweep instead of five.
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

## Testing strategy

`tests/reference.rs` contains an independently written dense simulator
(interleaved storage, no fusion, no threads — deliberately different) and
compares full state vectors against it for every native gate in multiple
argument orders, random circuits at every fusion level, both precisions.
`tests/analytic.rs` pins closed-form results (GHZ, QFT phases, Grover,
teleportation statistics). The Metal backend must match the CPU backend
bit-for-bit on counts (same seed) and to 1e-5 on amplitudes.
