# Noise: trajectory-sampled channels

zeno models noise with **stochastic quantum trajectories on the state
vector** (what qiskit-aer calls per-shot noise sampling). Each shot is an
independent pure-state trajectory: after every executed gate, error
channels fire probabilistically and update the state vector; at every
measurement, a classical readout error may flip the recorded bit. Averaged
over shots this reproduces the modeled channels **exactly** — no density
matrices are materialized, no channel is truncated, and statistical (shot)
error is the only error.

The price is per-shot execution: a noisy run costs `O(shots · gates · 2^n)`
instead of one simulation plus analytic sampling. Shots run in parallel
across cores when per-thread state copies fit the memory budget.

## The model

```rust
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct NoiseModel {
    pub depolarizing_1q: f64,   // after each 1-qubit gate
    pub depolarizing_2q: f64,   // after each >=2-qubit gate (see semantics)
    pub bit_flip: f64,          // X error, per touched qubit, after each gate
    pub phase_flip: f64,        // Z error, per touched qubit, after each gate
    pub amplitude_damping: f64, // gamma, per touched qubit, after each gate
    pub readout_flip_0to1: f64, // P(record 1 | true 0) at each measurement
    pub readout_flip_1to0: f64, // P(record 0 | true 1)
}
```

- Exported as `zeno::NoiseModel` (module `zeno::noise`).
- **Validation** (`NoiseModel::validate() -> Result<(), Error>`): every
  field must be a probability in `[0, 1]`; NaN is rejected. Values are
  validated again at run time, so a hand-constructed invalid model fails
  cleanly. `depolarizing_1q ≤ 3/4` is the physically "sensible" range
  (3/4 is already the fully depolarizing channel), but values up to 1 are
  allowed with the standard "w.p. `p` apply a uniform non-identity Pauli"
  convention.
- **JSON** (`NoiseModel::from_json`): all seven fields optional, default 0;
  **unknown fields are rejected** (a typo must not silently produce an
  ideal simulation). Errors surface as `zeno::Error::Noise` and display as
  `invalid noise model: …`.
- `NoiseModel::is_trivial()` is true when every field is exactly 0. A
  trivial model (or `noise: None`) is treated as *no noise at all*: the
  ideal fused/analytically-sampled fast path runs, and statevector
  capture stays allowed.

```json
{
  "depolarizing_1q": 0.001,
  "depolarizing_2q": 0.01,
  "bit_flip": 0.0005,
  "phase_flip": 0.0005,
  "amplitude_damping": 0.002,
  "readout_flip_0to1": 0.01,
  "readout_flip_1to0": 0.03
}
```

## Semantics

- Channels attach to **compiled-as-written gates**, so noise runs force
  fusion OFF: when `RunOptions::noise` is `Some(non-trivial)`,
  `zeno::run_program` compiles with `fusion_max = 0` (no dense fusion, no
  diagonal fusion) and the executor runs every shot as its own trajectory.
  The run carries the notice:

  > `noise: trajectory sampling — per-shot execution, fusion disabled`

- **After each gate op** (`Unitary`/`Diagonal`/`Cx`), channels fire in this
  frozen order:
  1. **Depolarizing** — `depolarizing_1q` after 1-qubit gates,
     `depolarizing_2q` after `k ≥ 2`-qubit gates: with probability `p`
     apply a uniformly random **non-identity Pauli string** on the gate's
     `k` qubits. For `k = 1` that is X, Y or Z with probability `p/3`
     each; in general there are `4^k − 1` strings, each with probability
     `p / (4^k − 1)` (so `k = 2` gives 15 strings at `p/15`, `k = 3` gives
     63 at `p/63` — the `cx`/`ccx`-style generalization is automatic).
  2. **Per touched qubit, in ascending qubit order**:
     `bit_flip` (w.p. `p` apply X), then `phase_flip` (w.p. `p` apply Z),
     then `amplitude_damping` (see the math below).
- **Readout error**: at *every* measurement — mid-circuit and final — the
  recorded bit flips `0→1` w.p. `readout_flip_0to1` and `1→0` w.p.
  `readout_flip_1to0`. The state collapse is **unaffected**: this is a
  purely classical error on the stored clbit. Classical control
  (`if (c==v)`) reads the recorded — i.e. possibly flipped — value,
  exactly like a real control system acting on its own (noisy) record.
- **What counts as a gate**: only executed gate ops carry gate noise.
  `id`/`u0` are eliminated during lowering and carry no noise; `barrier`,
  `measure` and `reset` are not gates (reset's internal correction flip is
  an implementation detail, not a gate). Gates under an `if` carry noise
  only when the guard fires. Note that the compiler's **cancellation pass
  still runs**: adjacent self-inverse pairs on identical qubits (`h h`,
  `cx cx`, …) are removed *before* noise attaches — insert a `barrier`
  between them if you need both instances to execute (and be noisy).
- **Statevector requests + noise → error**: a noisy state is a mixture
  over trajectories; there is no single final state vector. Asking for one
  (`want_statevector`, `Simulator::statevector`, `--statevector`) with a
  non-trivial model returns `Error::InvalidCircuit` with a clear message.
- **Backends**: noise drives the same `Backend` trait as everything else,
  so it works unchanged on CPU (f32/f64) and Metal. CPU-vs-Metal noisy
  counts at the same seed agree in distribution but are **not
  bit-identical**: the jump/flip decisions compare RNG draws against
  probabilities (`prob_one` reductions) computed from f32 amplitudes on
  Metal and (by default) f64 on CPU, so an occasional trajectory branches
  differently. The Metal parity test bounds the disagreement by TVD, not
  equality.

## Amplitude damping: the trajectory math

The channel with parameter `γ` has Kraus operators

```
K0 = diag(1, √(1−γ))        K1 = √γ · |0⟩⟨1|
```

For a pure state `|ψ⟩` the trajectory step is:

- `P(jump) = ⟨ψ|K1†K1|ψ⟩ = γ · ⟨ψ|1⟩⟨1|ψ⟩ = γ · P(1)`, with `P(1)` from
  `backend.prob_one(q)` (no collapse).
- Draw `u ∈ [0,1)` from the shot's RNG stream.
- **Jump** (`u < P(jump)`): the new state is `K1|ψ⟩ / ‖K1|ψ⟩‖`. Since
  `K1 = √γ|0⟩⟨1|`, that is exactly *collapse qubit `q` to outcome 1, then
  apply X* (the `√γ` scalar cancels in the normalization). zeno collapses
  via `backend.measure(q, 0.0)` — safe because `u < γ·P(1)` with `u ≥ 0`
  implies `P(1) > 0` — then applies X.
- **No-jump** (`u ≥ P(jump)`): the new state is `K0|ψ⟩ / ‖K0|ψ⟩‖` with
  `‖K0|ψ⟩‖² = (1−P(1)) + (1−γ)P(1) = 1 − γ·P(1) = 1 − P(jump)`. zeno folds
  the renormalization into the operator and applies the single 1-qubit
  diagonal `diag(1, √(1−γ)) / √(1 − P(jump))`, keeping `‖ψ‖ = 1` in one
  sweep. (Equivalently: apply `K0`, then a global scalar — a global scalar
  is just the 1-qubit diagonal `[s, s]` on any qubit.)

Averaging the two branches reproduces
`E(ρ) = K0 ρ K0† + K1 ρ K1†` exactly. The renormalization in the no-jump
branch is *not* optional: `prob_one` returns absolute mass, so a drifting
norm would corrupt every later measurement and damping decision.
`tests/noise.rs` pins this with `P(1) = (1−γ)^k` after `k` chained gates on
`|1⟩`, `P(1) = (1−γ)/2` after one damped gate on `|+⟩`, and an explicit
`norm² ≈ 1` check after noisy evolution.

## Randomness and reproducibility

All noise randomness comes from the **per-shot Xoshiro256++ stream**,
seeded with `splitmix64(master_seed ^ shot_index)` — the same scheme the
dynamic-circuit executor already uses. Same seed ⇒ identical noisy counts,
independent of thread count or shot execution order.

The per-shot draw layout is **frozen** (so seeds stay stable across
releases): ops execute in compiled order, and

1. a gate op consumes, in order: one `f64` draw for depolarizing *if the
   applicable `p > 0`*, plus one uniform-integer draw in `[1, 4^k)`
   (`rand::Rng::gen_range`; rejection sampling may consume several raw
   outputs) *only on a hit*; then per touched qubit in ascending order:
   one `f64` draw for `bit_flip` if `> 0`, one for `phase_flip` if `> 0`,
   one for `amplitude_damping` if `> 0`.
2. a measurement consumes one `f64` draw for the collapse (also in the
   noiseless dynamic path), then one `f64` readout draw *only if* the flip
   probability for the observed value is `> 0`.
3. `reset` consumes one `f64` collapse draw; untaken `if` branches consume
   nothing.

Channels with probability 0 consume **no** draws. `prob_one` is computed
from the state, not the stream.

## Library API

```rust
use zeno::{Circuit, NoiseModel, Simulator};

let mut c = Circuit::new(2);
c.h(0).cx(0, 1).measure_all();

let model = NoiseModel { depolarizing_2q: 0.05, ..Default::default() };
let r = Simulator::new().shots(10_000).seed(7).noise(model).run(&c).unwrap();
// r.counts now contains "01"/"10" leakage; r.notices carries the noise notice.
```

`RunOptions` carries the model directly: `RunOptions { noise: Some(model), .. }`.

## CLI contract (`zeno run`)

The `run` subcommand exposes the model through one flag:

- `--noise <MODEL>` — three forms, decided by shape: **inline JSON** if the
  (trimmed) argument starts with `{`; comma-separated **`key=value` pairs**
  if it contains `=` (keys are the `NoiseModel` field names; an unknown key
  errors with the full key list); otherwise a **path** to a JSON file with
  the same schema. All three produce identical results at the same seed.
  Examples:

  ```sh
  zeno run bell.qasm --shots 20000 --seed 7 --noise '{"depolarizing_2q": 0.05}'
  zeno run bell.qasm --noise bit_flip=0.01,readout_flip_1to0=0.02
  zeno run bell.qasm --noise ibm-ish.json
  ```

- Implementation: read the file if needed, call
  `zeno::NoiseModel::from_json(&text)` (parses **and** validates), and set
  `RunOptions { noise: Some(model), … }`. Absent flag ⇒ `noise: None`.
  Parse/validation failures print the `zeno::Error` (Display begins with
  `invalid noise model: …`) and exit non-zero like every other run error.
- No further CLI-side logic is needed: the library forces fusion off,
  pushes the notice (already printed by the notice machinery), runs
  per-shot trajectories, and rejects `--statevector` combined with a
  non-trivial model (`invalid circuit: a noise model is active: …`).
  A trivial model (`--noise '{}'`) behaves exactly like no flag.
- JSON output (`--json`): when the flag was given, echo the parsed model
  as a `"noise"` object (`NoiseModel` implements `serde::Serialize`) so
  runs stay self-describing; omit the key otherwise.

## Verification

`tests/noise.rs` pins every channel against closed-form values at 6σ
binomial bounds (bit/phase flip visibility, depolarizing `2p/3` and
`12p/15` state-change rates, `⟨Z0Z1⟩ = 1 − 16p/15` Bell decay, damping
`(1−γ)^k`, readout independence, determinism across seeds and thread
counts) and cross-checks composite models against qiskit-aer
(`tests/aer_reference.py`, TVD < 0.02 at 100k shots) when
`/tmp/qk-venv/bin/python` exists.
