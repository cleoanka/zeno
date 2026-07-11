# OpenQASM 2.0 support in zeno

The front end (`src/qasm.rs`, `zeno::qasm::{parse_str, parse_file}`) is a
hand-written lexer + recursive-descent parser producing `zeno::ir::Program`.
This document is the exact support matrix of what is implemented — no more, no
less.

## Program structure

| Statement | Status | Notes |
|---|---|---|
| `OPENQASM 2.0;` | required | Must be the first statement (comments may precede it). `OPENQASM 2;` is accepted as 2.0. Version 3/3.0 dispatches to the OpenQASM 3 front end (see QASM3.md); anything else is an error. |
| `include "qelib1.inc";` | accepted, internal | No file is ever read; the include is a no-op because the native gate set is always available (see *Deviations*). Any other include path is an error. |
| `qreg name[n];` | supported | `n ≥ 1`. Total qubits across all qregs ≤ **48**. |
| `creg name[n];` | supported | `n ≥ 1`. Register names are unique across qregs *and* cregs. (The compiler additionally caps total clbits at 64.) |
| `gate name(params?) args { body }` | supported | Expanded inline (macro expansion) at call sites; nesting depth ≤ **128**. Body may contain only gate calls and `barrier`. Empty bodies and empty `()` are allowed. |
| `opaque name ...;` | **error** | No simulation semantics; a clear "unsupported" error is raised. |
| gate calls `g(p,…) a,b,…;` | supported | With broadcasting, see below. |
| `U(θ,φ,λ) q;` | supported | Builtin; lowered to native `u3`. Case-sensitive. |
| `CX a,b;` | supported | Builtin; lowered to native `cx`. Case-sensitive. |
| `measure q -> c;` | supported | Bit → bit, or register → register of equal size (see broadcasting). |
| `reset q;` | supported | Broadcasts across a whole register. |
| `barrier args;` | supported | At least one argument; whole registers expand to all their qubits; duplicates are silently de-duplicated. |
| `if (creg == n) qop;` | supported | `creg` must be a *whole* classical register (never indexed); `n` is an unsigned integer (`u64`). `qop` is a gate call, `measure` or `reset` — never `barrier` or another `if`. Broadcast qops expand into one `if` per expanded instruction. In the IR, `Instr::If.creg` is the **index into `Program.cregs`**, not a bit offset. |
| comments `// …` | supported | To end of line. Free-form whitespace; semicolons required; no empty statements. |

## Native gate set

The whole table below is available **always**, with or without
`include "qelib1.inc";` (deviation: the spec only ships `U`/`CX` as
primitives). Names are case-sensitive and lowercase (except the `U`/`CX`
builtins). Qubit arguments are in *argument order*; for controlled gates the
control(s) come first. "Diag" marks gates the compiler applies as O(2ⁿ)
diagonal sweeps.

| Gate | Qubits | Params | Diag | Gate | Qubits | Params | Diag |
|---|---|---|---|---|---|---|---|
| `id`, `u0` | 1 | 0 | yes | `cx` (= `CX`) | 2 | 0 | no |
| `x`, `y` | 1 | 0 | no | `cy`, `ch` | 2 | 0 | no |
| `z` | 1 | 0 | yes | `cz` | 2 | 0 | yes |
| `h` | 1 | 0 | no | `swap` | 2 | 0 | no |
| `s`, `sdg` | 1 | 0 | yes | `cp(λ)`, `cu1(λ)` | 2 | 1 | yes |
| `t`, `tdg` | 1 | 0 | yes | `crx(θ)`, `cry(θ)` | 2 | 1 | no |
| `sx`, `sxdg` | 1 | 0 | no | `crz(θ)` | 2 | 1 | yes |
| `rx(θ)`, `ry(θ)` | 1 | 1 | no | `rxx(θ)` | 2 | 1 | no |
| `rz(θ)` | 1 | 1 | yes | `rzz(θ)` | 2 | 1 | yes |
| `u1(λ)`, `p(λ)` | 1 | 1 | yes | `cu3(θ,φ,λ)` | 2 | 3 | no |
| `u2(φ,λ)` | 1 | 2 | no | `ccx` | 3 | 0 | no |
| `u3(θ,φ,λ)`, `u` (= `U`) | 1 | 3 | no | `cswap` | 3 | 0 | no |

User-defined `gate` names may not shadow anything: colliding with a native
gate, the `U`/`CX` builtins or an existing definition is an error. Recursion
is impossible (a gate's name is not in scope inside its own body) and reported
with a dedicated message.

## Broadcasting

For gate calls (native, builtin and user-defined) and `reset`:

- every argument is either a single qubit `reg[i]` or a whole register `reg`;
- all whole-register arguments in one statement must have **equal size** `n`
  (mismatch is an error naming both registers);
- the statement expands to `n` applications (or 1 if there are no
  whole-register arguments); single-qubit arguments are repeated across the
  broadcast;
- after expansion, **no application may use the same qubit twice** — this is
  an error that names the duplicated qubit (e.g. `cx q[1],q;` fails on the
  instance `cx q[1],q[1]`).

`measure` broadcasts register → register of equal size only. A size-1 whole
register is treated as its single bit, so `measure q -> c[0];` works when
`|q| = 1`. Mixing a register of size > 1 with a single bit is an error.

`barrier` takes the union of all referenced qubits (no size constraint,
duplicates removed).

## Parameter expressions

Evaluated to `f64` **at parse time**. Grammar (highest precedence last):

```
expr   := term  { ('+' | '-') term }
term   := unary { ('*' | '/') unary }
unary  := '-' unary | power
power  := atom [ '^' unary ]            // right-associative, binds tightest
atom   := NUMBER | 'pi' | PARAM | FUNC '(' expr ')' | '(' expr ')'
FUNC   := 'sin' | 'cos' | 'tan' | 'exp' | 'ln' | 'sqrt'
```

- `^` binds tightest and is right-associative: `2^3^2 = 512`, `-2^2 = -4`,
  `2^-3 = 0.125`.
- `NUMBER` is an integer or real literal; reals may use scientific notation
  (`1.5e2`, `2E-2`, `.5`).
- `PARAM` is a formal parameter of the enclosing `gate` definition — the only
  place identifiers are bound in expressions. At top level any identifier
  (other than `pi` and the functions) is an error listing what is in scope.
- Arithmetic follows IEEE-754 `f64` semantics; division by zero or `ln(0)` is
  not trapped (you get ±inf/NaN in the resulting matrix).

## Errors

Error quality is a headline feature. Every `QasmError` carries an accurate
**1-based line and column** of the offending token, names that token, and says
what was expected. Examples of real messages:

- `line 3:1: unknown gate 'H' (gate names are case-sensitive; did you mean 'h'?)`
- `line 4:1: broadcast size mismatch: register 'a' has size 2 but register 'b' has size 3`
- `line 3:1: gate 'cx' applied to duplicate qubit 'q[1]' after broadcasting`
- `line 3:5: index 5 out of range for register 'q' of size 2`
- `line 4:1: expected ';' after the gate call, found 'x'`
- `line 4:7: found a single '='; the only equality operator is '==' (as in 'if (c == 1)')`

Parsing is fail-fast: the first error aborts. Integer literals (register
sizes, indices, `if` values) are parsed with overflow checking — a huge index
is a clean error, never a panic.

## Known deviations from the OpenQASM 2.0 spec

Deliberate, all on the permissive side except where noted:

1. **Native-superset gate set.** The spec provides only `U` and `CX` as
   primitives; everything else comes from `qelib1.inc`. Here the full native
   table above is always in scope, and `include "qelib1.inc";` is a no-op
   satisfied internally (it is *not* re-parsed from an embedded copy).
   Consequently zeno's `rz`, `u1`, … are the crate's native definitions
   (qiskit conventions), not the textual qelib1 macro expansions.
2. **Only `qelib1.inc` may be included.** Any other path is an error; there is
   no filesystem access from `include`.
3. **Identifiers** may start with any ASCII letter or `_` (the spec requires a
   lowercase first letter). Reserved words (`OPENQASM`, `include`, `qreg`,
   `creg`, `gate`, `opaque`, `measure`, `reset`, `barrier`, `if`, `pi`, `U`,
   `CX`) cannot be used as names.
4. **Namespaces:** registers and gates live in separate namespaces (the spec
   has a single one). Register names are unique across qreg/creg; gate names
   are unique across native + builtin + user-defined.
5. `OPENQASM 2;` is accepted as version 2.0; `OPENQASM 3;`/`3.0;` routes to the OpenQASM 3 subset parser (QASM3.md).
6. **Limits:** ≤ 48 total qubits (front-end check, matching the compiler),
   ≤ 128 gate-expansion nesting depth. The compiler additionally enforces
   ≤ 64 classical bits.
7. `barrier` de-duplicates repeated qubits instead of erroring.
8. An `if` value larger than the register can represent is accepted (the
   condition is simply never true).
9. `opaque` is rejected rather than treated as a declaration.
