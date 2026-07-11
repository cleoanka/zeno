# Your first quantum circuit in 15 minutes

This tutorial assumes you know nothing about quantum computing and nothing
about the terminal beyond copy and paste. By the end you will have run five
famous quantum circuits, written one of your own, broken it on purpose, and
understood every line of output along the way.

All transcripts below are real output from zeno v0.2.0 (the header line and
the timing will show *your* machine and *your* milliseconds; everything else
reproduces exactly with the seeds shown).

## 1. Three words you need

A **qubit** is a quantum bit: like a coin that is both heads and tails until
you look at it. While a program runs, a qubit can be 0 and 1 at once — that
in-between condition is called **superposition**. **Measurement** is the
"looking": it forces the qubit to pick 0 or 1, randomly, with probabilities
set by the circuit. Because one measurement is a coin flip, we run the same
circuit many times (each run is called a **shot**) and look at the tally.
Two qubits can also be linked so their random answers always agree — that
link is **entanglement**, and it is the first thing we will build.

## 2. Install

Copy and paste this into Terminal (if `cargo` is missing, first install Rust
with `brew install rustup` and follow its prompts):

```sh
git clone https://github.com/cleoanka/zeno
cd zeno
cargo install --path .
```

If your Mac has Apple Silicon and you want the GPU backend too, use this
instead of the last line:

```sh
cargo install --path . --features metal
```

Success looks like this:

```sh
$ zeno --version
zeno 0.2.0
```

## 3. Your first run: `zeno demo`

No files, no setup — just type:

```sh
zeno demo
```

```text
zeno v0.2.0 · Apple M4 Pro · 12 threads
demo: bell — two entangled qubits, the smallest interesting circuit

A qubit is a quantum bit: between preparation and measurement it can
be 0 and 1 at once (superposition). Here the h gate puts qubit 0 into
an equal superposition, then cx ties qubit 1 to it, so the pair always
agrees when measured (entanglement). The histogram below should show
only 00 and 11, each near 50% — never 01 or 10.

circuit (OpenQASM 2.0):
  OPENQASM 2.0;
  include "qelib1.inc";
  qreg q[2];
  creg c[2];
  h q[0];
  cx q[0], q[1];
  measure q -> c;

2 qubits · 2 gates -> 2 ops · f64 · cpu-f64 · 674 µs

  00  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  507   50.7%
  11  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇   493   49.3%

seed 0x701a0d019f1a425e (reproduce with --seed)

Try next: zeno demo ghz
```

Every line, decoded:

- **`zeno v0.2.0 · Apple M4 Pro · 12 threads`** — the version, your chip,
  and how many CPU threads the simulation used. Yours will differ.
- **`demo: bell — …`** — which built-in demo ran. `bell` is the default.
- **The paragraph** — what the circuit is about to show, in plain words.
- **`circuit (OpenQASM 2.0):`** — the exact source code that is about to
  run. This is real OpenQASM, the standard quantum circuit language; in
  section 4 you will write this file yourself.
- **`2 qubits · 2 gates -> 2 ops · f64 · cpu-f64 · 674 µs`** — the run
  summary: 2 qubits were simulated, the 2 gates compiled into 2 operations,
  at double precision (`f64`), on the CPU backend, in 674 microseconds.
- **The histogram** — the heart of the output. Each row is one distinct
  measurement result: the bitstring (`00` = both qubits read 0), a bar, the
  number of shots that produced it (out of 1000), and the percentage. Note
  what is *missing*: no `01`, no `10`. The two qubits never disagree —
  that is entanglement, tallied.
- **`seed 0x701a0d019f1a425e (reproduce with --seed)`** — measurement is
  random, so each run's tallies differ slightly. The seed is the recipe for
  this run's randomness: `zeno demo --seed 0x701a0d019f1a425e` replays
  these exact counts. Every zeno run prints one, so no result is ever lost.
- **`Try next: zeno demo ghz`** — the guided tour continues. There are six
  demos; see them all with:

```sh
zeno demo --list
```

```text
built-in demos (zeno demo <name>):
  name      what it shows
  bell      two entangled qubits, the smallest interesting circuit
  ghz       8-qubit entanglement: one coin flip decides all eight bits
  qft       quantum Fourier transform + exact inverse: a self-checking round trip
  grover    Grover's search finds the marked state |101> in two queries
  teleport  quantum teleportation with mid-circuit measurement and feed-forward
  noisy     the Bell pair again, on simulated noisy hardware

all demos take --shots N, --seed S, --noise SPEC and --json
```

## 4. Write your own circuit

Time to write the Bell pair yourself. Create a file called `bell.qasm` (any
text editor works, or paste this whole block into Terminal):

```sh
cat > bell.qasm <<'EOF'
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
EOF
```

Line by line:

- `OPENQASM 2.0;` — every file starts with this version header.
- `include "qelib1.inc";` — pulls in the standard gate names (zeno has
  this library built in; no file is needed on disk).
- `qreg q[2];` — declare a quantum register: 2 qubits, named `q[0]`, `q[1]`.
- `creg c[2];` — declare 2 classical bits to hold the measurement results.
- `h q[0];` — the Hadamard gate: puts `q[0]` into an equal superposition
  of 0 and 1 (the spinning coin).
- `cx q[0], q[1];` — controlled-X: if `q[0]` is 1, flip `q[1]` — applied
  to a superposition this *entangles* the pair.
- `measure q -> c;` — measure every qubit in `q` into the matching bit
  of `c`.

Every statement ends with a semicolon. Now run it, seeded so your output
matches this page exactly:

```sh
zeno run bell.qasm --shots 1000 --seed 42
```

```text
zeno v0.2.0 · Apple M4 Pro · 12 threads
bell · 2 qubits · 2 gates -> 2 ops · f64 · cpu-f64 · 642 µs

  00  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  502   50.2%
  11  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  498   49.8%

seed 0x000000000000002a (reproduce with --seed)
```

Now change something and *predict first*. Move the `h` from `q[0]` to
`q[1]` (so the file says `h q[1];` and the `cx` line stays the same).
Prediction: the `cx` control `q[0]` is now always 0, so it never fires —
no entanglement. `q[1]` is a lone coin flip, `q[0]` always reads 0.
The bitstring prints left-to-right as `c[1] c[0]`, so we expect `00` and
`10`, uncorrelated. Run it:

```sh
zeno run bell.qasm --shots 1000 --seed 42
```

```text
zeno v0.2.0 · Apple M4 Pro · 12 threads
bell · 2 qubits · 2 gates -> 2 ops · f64 · cpu-f64 · 645 µs

  00  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  502   50.2%
  10  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  498   49.8%

seed 0x000000000000002a (reproduce with --seed)
```

Prediction confirmed: same 50/50 randomness, but the qubits no longer
agree — `11` became `10`. You just falsified entanglement with a
one-character edit.

## 5. One step further

**Three qubits.** Entanglement chains: add a qubit and a second `cx`.

```sh
cat > ghz3.qasm <<'EOF'
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0], q[1];
cx q[1], q[2];
measure q -> c;
EOF
zeno run ghz3.qasm --shots 1000 --seed 42
```

```text
zeno v0.2.0 · Apple M4 Pro · 12 threads
ghz3 · 3 qubits · 3 gates -> 3 ops · f64 · cpu-f64 · 1.85 ms

  000  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  502   50.2%
  111  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  498   49.8%

seed 0x000000000000002a (reproduce with --seed)
```

One coin flip now decides three bits at once (this is a GHZ state — the
8-qubit version is `zeno demo ghz`).

**Peek behind the curtain.** On small circuits, `--statevector` prints the
exact quantum state instead of just samples (`--quiet` here drops the
header to keep it short):

```sh
zeno run bell.qasm --seed 42 --statevector --quiet
```

```text
  00  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  517   50.5%
  11  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇   507   49.5%

statevector · 4 amplitudes · |amp|² < 1e-12 hidden
  idx  basis          re          im    |amp|²
    0  |00⟩    +0.707107   +0.000000  0.500000
    3  |11⟩    +0.707107   +0.000000  0.500000
```

Each row is a possible outcome and its **amplitude** — the number the
simulator actually tracks. Squaring an amplitude gives the outcome's
probability: 0.707107² = 0.5, the 50% you have been seeing. `|01⟩` and
`|10⟩` have amplitude zero, which is *why* they never appear.

**What `--shots` does.** Shots are repetitions of the experiment. Few
shots, noisy percentages; many shots, smooth ones. With only 10:

```sh
zeno run bell.qasm --shots 10 --seed 42 --quiet
```

```text
  00  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  5   50.0%
  11  ▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇▇  5   50.0%
```

(A tidy 5/5 at this seed — try `--seed 45` and get an 8/2 split from the
very same circuit; small samples wobble.)
Shots are cheap in zeno: for circuits like these, 10 shots and 1,000,000
shots cost nearly the same, because the tallies are drawn from the exact
state.

## 6. When things go wrong

The three mistakes everyone makes, and exactly what zeno says.

**A typo in a gate name** (`hh q[0];` instead of `h q[0];`):

```text
error: QASM parse error: line 5:1: unknown gate 'hh'
```

The `line 5:1` is line 5, column 1 of your file — go there and fix the
name.

**A missing semicolon** (writing `h q[0]` and nothing else on the line):

```text
error: QASM parse error: line 6:1: expected ';' after the gate call, found 'cx'
```

Note the pointer is at line 6 — the parser only realizes the semicolon is
missing when the *next* statement begins, so look one line above the
reported spot.

**Forgetting to measure** (deleting the `measure q -> c;` line). This is
not an error — the circuit runs — but there is nothing to tally, and zeno
tells you why the histogram is empty:

```text
zeno v0.2.0 · Apple M4 Pro · 12 threads
nomeasure · 2 qubits · 2 gates -> 2 ops · f64 · cpu-f64 · 243 µs

note: circuit has no measurements: counts are empty (use --statevector to inspect the state)
seed 0x000000000000002a (reproduce with --seed)
```

## Where to go next

- Finish the guided tour: `zeno demo qft`, `zeno demo grover`,
  `zeno demo teleport`, and `zeno demo noisy` — the last one shows what
  realistic hardware noise does and teaches you the `--noise` flag.
- [../README.md](../README.md) — everything zeno can do, and how fast.
- [QASM.md](QASM.md) — the full OpenQASM 2.0 reference: every gate,
  broadcasting, user-defined gates, classical control.
- [NOISE.md](NOISE.md) — the noise model in depth: every channel, the math,
  and the `--noise` flag's JSON schema.
- `zeno info` — how many qubits *your* machine can hold.
