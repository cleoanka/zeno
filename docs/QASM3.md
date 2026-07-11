# OpenQASM 3 support in zeno (documented subset)

zeno's OpenQASM 3 front end (`src/qasm3.rs`) is a hand-written lexer +
recursive-descent parser producing the same `zeno::ir::Program` as the
OpenQASM 2.0 front end, with the same error type and the same error-quality
bar. **This document is the exact support matrix — the subset is frozen: what
is listed here works, everything else in OpenQASM 3 is rejected with an
explicit error naming the feature.**

There is a single entry point for all QASM: `zeno::qasm::parse_str` (and
`parse_file`, and therefore the `zeno` CLI). It dispatches on the declared
version:

- `OPENQASM 2.0;` (or `2;`) → the OpenQASM 2.0 parser (see `docs/QASM.md`);
- `OPENQASM 3;` or `OPENQASM 3.0;` → this subset parser.

Both front ends produce identical IR for equivalent programs, so everything
downstream (compiler, CPU/Metal backends, counts formatting) is version-blind.

## Program structure

| Statement | Status | Notes |
|---|---|---|
| `OPENQASM 3;` / `OPENQASM 3.0;` | required | Must be the first statement (line and `/* */` block comments may precede it). Any other version spelling is an error. |
| `include "stdgates.inc";` | accepted, internal | No file is ever read; the native gate set (a superset) is always available. Any other include path — including `"qelib1.inc"` — is an error. |
| `qubit[n] name;` / `qubit name;` | supported | `n ≥ 1`; no size means size 1. Total qubits across all declarations ≤ **48**. |
| `bit[n] name;` / `bit name;` | supported | `n ≥ 1`; no size means size 1. Register names are unique across qubit *and* bit registers. (The compiler additionally caps total clbits at 64.) |
| `qreg` / `creg` | **error** | OpenQASM 2 syntax; the error tells you to write `qubit[n]` / `bit[n]`. |
| `gate name(params?) args { body }` | supported | Same semantics as the 2.0 front end: expanded inline at call sites, nesting depth ≤ **128**, body may contain only gate calls and `barrier`, no shadowing of native gates / aliases / earlier definitions, recursion impossible. |
| gate calls `g(p,…) a,b,…;` | supported | With OpenQASM 2-style broadcasting (below). |
| `c = measure q;` | supported | Assignment form. Bit = bit, or register = register of **equal size** (mismatch is an error naming both registers and their sizes). Only `measure` may appear on the right-hand side of `=`. |
| `c[i] = measure q[j];` | supported | Single-bit assignment form. |
| `measure q -> c;` | supported | The OpenQASM 2 arrow form is kept for migration; identical lowering. |
| `reset q;` | supported | Broadcasts across a whole register. |
| `barrier args;` | supported | Whole registers expand; duplicates de-duplicated. |
| `barrier;` | supported | No arguments = **all** qubits declared so far. |
| `if (creg == n) qop;` | supported | Same restrictions as 2.0: whole register only, `n` a `u64`, body is a gate call / measure (either form) / reset — never `barrier`, a declaration or another `if`. |
| `if (creg == n) { qop; qop; … }` | supported | Block form; lowered to **one `Instr::If` per contained op** (after broadcasting). Same body restrictions as the single-statement form; an empty block is a no-op. |
| comments | supported | `// …` to end of line and `/* … */` (non-nesting) blocks. |

Declarations are order-independent with respect to *other* statements but a
register must be declared before its own first use (use-before-declare is an
error, exactly like OpenQASM 2).

## Gate set and stdgates aliases

The full native table from `docs/QASM.md` is always in scope (35 names,
qiskit conventions, control-first argument order). On top of it the OpenQASM 3
builtin and the `stdgates.inc` compatibility aliases resolve to natives:

| OpenQASM 3 name | Lowers to | Signature |
|---|---|---|
| `U(θ,φ,λ)` (builtin) | `u3` | 1 qubit, 3 params |
| `CX` | `cx` | 2 qubits, 0 params |
| `phase(λ)` | `p` | 1 qubit, 1 param |
| `cphase(λ)` | `cp` | 2 qubits, 1 param |

`gphase` and the 4-parameter `cu` are **not** in the subset (`cu3` is native).
Alias names cannot be shadowed by `gate` definitions.

## Broadcasting

Identical to the 2.0 front end: all whole-register arguments of one statement
must have equal size `n`; the statement expands to `n` applications with
single-bit arguments repeated; a duplicated qubit inside one expanded
application is an error naming the qubit. Measurement broadcasts register =
register of equal size in **both** forms; a size-1 whole register is treated
as its single bit.

## Parameter expressions

The OpenQASM 2 expression grammar, evaluated to `f64` at parse time
(`pi`, `+ - * / ^` with `^` right-associative and binding tightest, unary
`-`, parentheses, `sin cos tan exp ln sqrt`, scientific literals), plus:

- `π` is accepted as `pi`;
- `tau` and `τ` are accepted as 2π (`std::f64::consts::TAU`).

## Explicitly rejected OpenQASM 3 features

Each of the following produces an error with an accurate line:column that
**names the feature** and points here — never a generic "unexpected token":

| Feature | Trigger words |
|---|---|
| for / while loops | `for`, `while` |
| subroutines | `def` |
| calibration blocks | `defcal`, `cal` |
| gate modifiers | `ctrl @`, `negctrl @`, `inv @`, `pow(k) @` |
| timing | `delay`, `duration`, `stretch` |
| typed classical declarations | `float`, `int`, `uint`, `angle`, `bool`, `complex` |
| const declarations | `const` |
| input/output parameters | `input`, `output` |
| arrays beyond 1-D registers | `array` (also register range indexing `q[a:b]`) |
| switch statements | `switch` |
| extern declarations | `extern` |
| pragmas | `pragma`, `#pragma` |
| annotations | `@name` |
| else clauses | `else` |
| aliasing / scoping / control flow | `let`, `box`, `return`, `break`, `continue` |

Example message:

```
line 3:1: for loops are not supported in zeno's OpenQASM 3 subset (see docs/QASM3.md)
```

## Errors

Same contract as the 2.0 front end: every error carries an accurate 1-based
line and column, names the offending token and says what was expected.
Additional real examples from this front end:

- `line 2:1: 'qreg' is OpenQASM 2 syntax; in OpenQASM 3 declare quantum registers with 'qubit[n] name;'`
- `line 4:1: measure size mismatch: quantum register 'q' has size 3 but classical register 'c' has size 5`
- `line 4:5: expected 'measure' after '=' (in this subset an assignment can only store a measurement result), found 'h'`
- `line 4:7: found a single '='; the comparison operator in an if condition is '=='`
- `line 2:9: cannot include "qelib1.inc": only "stdgates.inc" is supported (and it is satisfied internally — the native gate set is always available)`

## Runnable examples

Bell pair, both measurement forms (either file runs with
`zeno run bell3.qasm --shots 4096`):

```qasm
OPENQASM 3;
include "stdgates.inc";
qubit[2] q;
bit[2] c;
h q[0];
cx q[0], q[1];
c = measure q;          // assignment form
// measure q -> c;      // arrow form — identical lowering
```

GHZ with a bare barrier and π:

```qasm
OPENQASM 3.0;
qubit[3] q;
bit[3] c;
h q[0];
cx q[0], q[1];
cx q[1], q[2];
barrier;                // no args = all qubits
rz(τ/2) q[0];           // τ = 2π; same as rz(pi)
c = measure q;
```

Conditional block and a parameterized gate definition:

```qasm
OPENQASM 3;
qubit[2] q;
bit[2] c;

gate rot(t) a { rz(t/2) a; ry(t) a; rz(-t/2) a; }

rot(pi/4) q[0];
c[0] = measure q[0];
if (c == 1) {
    x q[1];             // lowered to one Instr::If per op
    reset q[0];
}
c[1] = measure q[1];
```

## Known deviations from the OpenQASM 3 spec

1. Everything under *Explicitly rejected features* — the subset is
   deliberately small and loud about its edges.
2. `stdgates.inc` is satisfied internally by the native superset (no file
   read, qiskit-convention matrices); `gphase`/`cu` are absent.
3. `if` conditions are OpenQASM 2-shaped: whole register `== u64` only — no
   bit tests, comparison operators, boolean expressions or `else`.
4. Measurement is the only expression that can be assigned; classical
   computation is out of scope.
5. Identifiers are ASCII (letters, digits, `_`, not starting with a digit)
   plus the literal constants `π`/`τ`; general unicode identifiers are not
   accepted.
6. Same limits as the 2.0 front end: ≤ 48 total qubits, ≤ 128 gate-expansion
   depth, ≤ 64 clbits (compiler).
7. `barrier` de-duplicates repeated qubits instead of erroring.
