# Contributing

Issues and PRs are welcome.

## Ground rules

- `cargo fmt`, `cargo clippy --all-targets -- -D warnings` and `cargo test`
  must be green. CI runs on Apple Silicon (`macos-14`).
- Numerical changes need evidence: the `tests/reference.rs` suite compares
  against an independent dense simulator — extend it when you touch kernels
  or the compiler.
- Performance claims belong in `docs/PERFORMANCE.md` with the exact command,
  chip, and numbers.

## Layout

| Path | What lives there |
|---|---|
| `src/state.rs` | state vector + gate kernels (the hot code) |
| `src/compiler.rs` | cancellation / diagonal fusion / gate fusion |
| `src/exec.rs` | backends, shots, sampling orchestration |
| `src/qasm.rs` | OpenQASM 2.0 front end |
| `src/metal.rs` | optional Metal GPU backend (`--features metal`) |
| `tests/` | adversarial reference + analytic suites |
