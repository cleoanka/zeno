//! `kuantum` command-line interface.
//!
//! Four subcommands over the library in `src/lib.rs`:
//!
//! - `run`     — parse, compile and run an OpenQASM 2.0 file; histogram or JSON.
//! - `info`    — machine capacity: RAM, budget, max qubits per precision.
//! - `bench`   — seeded random-circuit throughput sweep (amp-updates/s).
//! - `compile` — fusion visualizer: stats + the compiled op list.
//!
//! Output conventions: dim ANSI for meta lines, bold for keys, one accent
//! color for bars — and only when stdout is a TTY and `NO_COLOR` is unset.
//! `--json` output is a single plain JSON object with `schema_version: 1`.

use clap::{Parser, Subcommand, ValueEnum};
use kuantum::compiler::{self, COp, CompileOptions};
use kuantum::ir::Reg;
use kuantum::{human_bytes, mem, BackendChoice, Counts, Precision, RunOptions, RunResult};
use std::io::IsTerminal;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

const VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "kuantum",
    version,
    about = "Apple Silicon-native quantum circuit simulator, compiler and runner",
    after_help = "Examples:\n  \
        kuantum run bell.qasm --shots 4096\n  \
        kuantum run qft.qasm --statevector --seed 7\n  \
        kuantum info\n  \
        kuantum bench --qubits 20,24 --compare-fusion\n  \
        kuantum compile grover.qasm"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run an OpenQASM 2.0 circuit and print measurement counts
    Run(RunArgs),
    /// Show machine capacity (RAM, budget, max qubits by precision)
    Info {
        /// Machine-readable JSON output
        #[arg(long)]
        json: bool,
    },
    /// Benchmark random circuits (throughput in amp-updates/s)
    Bench(BenchArgs),
    /// Parse and compile only: fusion stats and the compiled op list
    Compile(CompileArgs),
}

#[derive(clap::Args)]
struct RunArgs {
    /// OpenQASM 2.0 file
    file: PathBuf,
    /// Number of measurement shots
    #[arg(long, default_value_t = 1024)]
    shots: u64,
    /// RNG seed (decimal or 0x-hex); omit for a random seed
    #[arg(long, value_parser = parse_seed)]
    seed: Option<u64>,
    /// State-vector precision
    #[arg(long, value_enum, default_value_t = PrecisionArg::Auto)]
    precision: PrecisionArg,
    /// Simulation backend
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
    /// Maximum fused-gate width in qubits (0 disables fusion)
    #[arg(long, value_name = "K", default_value_t = 5)]
    fusion: u8,
    /// Memory budget, e.g. 8g, 512m, 1024k, or raw bytes
    #[arg(long, value_name = "SIZE", value_parser = parse_size)]
    mem_limit: Option<u64>,
    /// Fraction of physical RAM usable when --mem-limit is unset
    #[arg(long, value_name = "F", default_value_t = 0.75, value_parser = parse_fraction)]
    mem_fraction: f64,
    /// Override the rayon thread count
    #[arg(long, value_name = "N")]
    threads: Option<usize>,
    /// Also print the final state vector (needs a non-dynamic circuit)
    #[arg(long)]
    statevector: bool,
    /// Machine-readable JSON output
    #[arg(long)]
    json: bool,
    /// Histogram only: no header, summary or seed line
    #[arg(long)]
    quiet: bool,
}

#[derive(clap::Args)]
struct BenchArgs {
    /// Qubit counts to sweep, comma-separated
    #[arg(long, value_delimiter = ',', default_value = "20,22,24,26")]
    qubits: Vec<u32>,
    /// Layers in the random circuit
    #[arg(long, default_value_t = 12)]
    depth: u32,
    /// Maximum fused-gate width in qubits (0 disables fusion)
    #[arg(long, value_name = "K", default_value_t = 5)]
    fusion: u8,
    /// State-vector precision
    #[arg(long, value_enum, default_value_t = PrecisionArg::Auto)]
    precision: PrecisionArg,
    /// Simulation backend
    #[arg(long, value_enum, default_value_t = BackendArg::Auto)]
    backend: BackendArg,
    /// Seed for the random circuits
    #[arg(long, value_parser = parse_seed, default_value = "7")]
    seed: u64,
    /// Also run with fusion disabled and print the speedup
    #[arg(long)]
    compare_fusion: bool,
    /// Machine-readable JSON output
    #[arg(long)]
    json: bool,
}

#[derive(clap::Args)]
struct CompileArgs {
    /// OpenQASM 2.0 file
    file: PathBuf,
    /// Maximum fused-gate width in qubits (0 disables fusion)
    #[arg(long, value_name = "K", default_value_t = 5)]
    fusion: u8,
    /// Machine-readable JSON output
    #[arg(long)]
    json: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum PrecisionArg {
    Auto,
    F32,
    F64,
}

impl PrecisionArg {
    fn to_option(self) -> Option<Precision> {
        match self {
            PrecisionArg::Auto => None,
            PrecisionArg::F32 => Some(Precision::F32),
            PrecisionArg::F64 => Some(Precision::F64),
        }
    }

    fn label(self) -> &'static str {
        match self {
            PrecisionArg::Auto => "auto",
            PrecisionArg::F32 => "f32",
            PrecisionArg::F64 => "f64",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, ValueEnum)]
enum BackendArg {
    Auto,
    Cpu,
    Metal,
}

impl BackendArg {
    fn to_choice(self) -> BackendChoice {
        match self {
            BackendArg::Auto => BackendChoice::Auto,
            BackendArg::Cpu => BackendChoice::Cpu,
            BackendArg::Metal => BackendChoice::Metal,
        }
    }

    fn label(self) -> &'static str {
        match self {
            BackendArg::Auto => "auto",
            BackendArg::Cpu => "cpu",
            BackendArg::Metal => "metal",
        }
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Run(a) => cmd_run(&a),
        Cmd::Info { json } => cmd_info(json),
        Cmd::Bench(a) => cmd_bench(&a),
        Cmd::Compile(a) => cmd_compile(&a),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

// ---------------------------------------------------------------------------
// Shared formatting helpers
// ---------------------------------------------------------------------------

/// ANSI styling, enabled only on a TTY with `NO_COLOR` unset.
#[derive(Clone, Copy)]
struct Style {
    on: bool,
}

impl Style {
    fn detect() -> Self {
        Style {
            on: std::io::stdout().is_terminal() && std::env::var_os("NO_COLOR").is_none(),
        }
    }

    fn paint(&self, code: &str, s: &str) -> String {
        if self.on {
            format!("\x1b[{code}m{s}\x1b[0m")
        } else {
            s.to_string()
        }
    }

    fn dim(&self, s: &str) -> String {
        self.paint("2", s)
    }

    fn bold(&self, s: &str) -> String {
        self.paint("1", s)
    }

    fn accent(&self, s: &str) -> String {
        self.paint("36", s)
    }
}

/// `8g` / `512m` / `1024k` / raw bytes → bytes (case-insensitive; fractional
/// values like `1.5g` are accepted with a binary suffix).
fn parse_size(s: &str) -> Result<u64, String> {
    let t = s.trim().to_ascii_lowercase();
    let bad = || format!("invalid size '{s}' (expected e.g. 8g, 512m, 1024k, or raw bytes)");
    if t.is_empty() {
        return Err(bad());
    }
    let split = t
        .find(|c: char| !(c.is_ascii_digit() || c == '.'))
        .unwrap_or(t.len());
    let (num, suffix) = t.split_at(split);
    if num.is_empty() {
        return Err(bad());
    }
    let mult: u64 = match suffix {
        "" => return num.parse::<u64>().map_err(|_| bad()),
        "k" | "kb" | "kib" => 1 << 10,
        "m" | "mb" | "mib" => 1 << 20,
        "g" | "gb" | "gib" => 1 << 30,
        "t" | "tb" | "tib" => 1 << 40,
        _ => return Err(bad()),
    };
    let v: f64 = num.parse().map_err(|_| bad())?;
    let bytes = v * mult as f64;
    if !bytes.is_finite() || bytes < 0.0 || bytes >= u64::MAX as f64 {
        return Err(bad());
    }
    Ok(bytes.round() as u64)
}

/// Seed in decimal or `0x`-hex (so `seed 0x…` lines can be pasted back).
fn parse_seed(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let parsed = match t.strip_prefix("0x").or_else(|| t.strip_prefix("0X")) {
        Some(h) => u64::from_str_radix(h, 16),
        None => t.parse(),
    };
    parsed.map_err(|_| format!("invalid seed '{s}' (decimal or 0x-hex u64)"))
}

fn parse_fraction(s: &str) -> Result<f64, String> {
    let v: f64 = s.parse().map_err(|_| format!("invalid fraction '{s}'"))?;
    if v > 0.0 && v <= 1.0 {
        Ok(v)
    } else {
        Err(format!("--mem-fraction must be in (0, 1], got {s}"))
    }
}

/// `1 gate` / `3 gates` — tiny grammar for summary lines.
fn plural(n: usize, word: &str) -> String {
    let s = if n == 1 { "" } else { "s" };
    format!("{n} {word}{s}")
}

/// Three significant figures, no scientific notation.
fn sig3(x: f64) -> String {
    if !x.is_finite() || x <= 0.0 {
        return format!("{x:.0}");
    }
    let mag = x.abs().log10().floor() as i32;
    let decimals = (2 - mag).max(0) as usize;
    format!("{x:.decimals$}")
}

/// Amplitude-updates per second with an adaptive k/M/G suffix (G is the
/// norm at benchmark sizes).
fn fmt_amp_per_s(x: f64) -> String {
    if x >= 1e9 {
        format!("{} G", sig3(x / 1e9))
    } else if x >= 1e6 {
        format!("{} M", sig3(x / 1e6))
    } else if x >= 1e3 {
        format!("{} k", sig3(x / 1e3))
    } else {
        sig3(x)
    }
}

/// Adaptive units (ns/µs/ms/s), three significant figures.
fn fmt_duration(d: Duration) -> String {
    let s = d.as_secs_f64();
    if s < 1e-6 {
        format!("{} ns", sig3(s * 1e9))
    } else if s < 1e-3 {
        format!("{} µs", sig3(s * 1e6))
    } else if s < 1.0 {
        format!("{} ms", sig3(s * 1e3))
    } else {
        format!("{} s", sig3(s))
    }
}

/// Read + parse a QASM file, prefixing I/O errors with the path (a bare
/// "No such file or directory" helps nobody).
fn parse_qasm(path: &std::path::Path) -> Result<kuantum::Program, kuantum::Error> {
    let src = std::fs::read_to_string(path).map_err(|e| {
        kuantum::Error::Io(std::io::Error::new(
            e.kind(),
            format!("{}: {e}", path.display()),
        ))
    })?;
    Ok(kuantum::qasm::parse_str(&src)?)
}

fn chip_name() -> String {
    std::process::Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

fn machine_header(threads: usize, sty: &Style) -> String {
    sty.dim(&format!(
        "kuantum v{VERSION} · {} · {threads} threads",
        chip_name()
    ))
}

/// Column-aligned table: dim headers, `right[i]` right-aligns column `i`.
fn print_table(headers: &[&str], rows: &[Vec<String>], right: &[bool], sty: &Style) {
    let cols = headers.len();
    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (i, cell) in row.iter().enumerate().take(cols) {
            widths[i] = widths[i].max(cell.chars().count());
        }
    }
    let render = |cells: &[String]| -> String {
        let mut line = String::new();
        for (i, cell) in cells.iter().enumerate() {
            let w = widths[i];
            let pad = w.saturating_sub(cell.chars().count());
            line.push_str("  ");
            if right.get(i).copied().unwrap_or(false) {
                line.push_str(&" ".repeat(pad));
                line.push_str(cell);
            } else {
                line.push_str(cell);
                if i + 1 < cells.len() {
                    line.push_str(&" ".repeat(pad));
                }
            }
        }
        line
    };
    let hdr: Vec<String> = headers.iter().map(|h| h.to_string()).collect();
    println!("{}", sty.dim(&render(&hdr)));
    for row in rows {
        println!("{}", render(row));
    }
}

// ---------------------------------------------------------------------------
// kuantum run
// ---------------------------------------------------------------------------

fn cmd_run(a: &RunArgs) -> Result<(), kuantum::Error> {
    let program = parse_qasm(&a.file)?;
    let n_qubits = program.n_qubits();
    // In human mode only small states are printed, so skip the copy above 8
    // qubits; JSON always carries the full vector when requested.
    let want_sv = a.statevector && (a.json || n_qubits <= 8);
    let opts = RunOptions {
        shots: a.shots,
        seed: a.seed,
        precision: a.precision.to_option(),
        backend: a.backend.to_choice(),
        fusion_max: a.fusion,
        mem_fraction: a.mem_fraction,
        mem_limit: a.mem_limit,
        want_statevector: want_sv,
        threads: a.threads,
    };
    let r = kuantum::run_program(&program, &opts)?;
    if a.json {
        print_run_json(a, &r);
    } else {
        print_run_human(a, &r);
    }
    Ok(())
}

fn print_run_json(a: &RunArgs, r: &RunResult) {
    let mut obj = serde_json::json!({
        "schema_version": 1,
        "file": a.file.display().to_string(),
        "n_qubits": r.n_qubits,
        "shots": r.shots,
        "seed": r.seed,
        "precision": r.precision,
        "backend": r.backend,
        "sim_time_ms": r.sim_time.as_secs_f64() * 1e3,
        "mem_bytes": r.mem_bytes,
        "stats": r.stats,
        "counts": r.counts,
        "notices": r.notices,
    });
    if let Some(sv) = &r.statevector {
        obj["statevector"] = serde_json::Value::Array(
            sv.iter()
                .map(|amp| serde_json::json!([amp.re, amp.im]))
                .collect(),
        );
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&obj).expect("valid json")
    );
}

fn print_run_human(a: &RunArgs, r: &RunResult) {
    let sty = Style::detect();
    if !a.quiet {
        let threads = a.threads.unwrap_or_else(rayon::current_num_threads);
        println!("{}", machine_header(threads, &sty));
        let stem = a
            .file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("circuit");
        println!(
            "{} · {} · {} -> {} · {} · {} · {}",
            sty.bold(stem),
            plural(r.n_qubits as usize, "qubit"),
            plural(r.stats.input_gates, "gate"),
            plural(r.stats.output_ops, "op"),
            r.precision,
            r.backend,
            fmt_duration(r.sim_time),
        );
        println!();
    }
    print_histogram(&r.counts, &sty);
    if a.statevector {
        print_statevector_human(r, &sty);
    }
    for note in &r.notices {
        println!("{}", sty.dim(&format!("note: {note}")));
    }
    if !a.quiet {
        println!(
            "{}",
            sty.dim(&format!("seed 0x{:016x} (reproduce with --seed)", r.seed))
        );
    }
}

fn print_histogram(counts: &Counts, sty: &Style) {
    const TOP: usize = 16;
    const BAR_W: usize = 30;
    if counts.is_empty() {
        return;
    }
    let total = counts.total().max(1);
    let mut items: Vec<(&String, &u64)> = counts.iter().collect();
    items.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
    let shown = &items[..items.len().min(TOP)];
    let max_count = (*shown[0].1).max(1) as f64;
    let key_w = shown.iter().map(|(k, _)| k.chars().count()).max().unwrap();
    let count_w = shown
        .iter()
        .map(|(_, c)| c.to_string().len())
        .max()
        .unwrap();
    for &(k, &c) in shown {
        let len = ((c as f64 / max_count) * BAR_W as f64).round().max(1.0) as usize;
        let pct = c as f64 / total as f64 * 100.0;
        println!(
            "  {}  {}{}  {c:>count_w$}  {pct:>5.1}%",
            sty.bold(&format!("{k:<key_w$}")),
            sty.accent(&"▇".repeat(len)),
            " ".repeat(BAR_W - len),
        );
    }
    if items.len() > TOP {
        println!(
            "{}",
            sty.dim(&format!(
                "  … and {} more (use --json for all)",
                items.len() - TOP
            ))
        );
    }
    println!();
}

fn print_statevector_human(r: &RunResult, sty: &Style) {
    let n = r.n_qubits;
    match &r.statevector {
        Some(sv) => {
            println!(
                "{}",
                sty.dim(&format!(
                    "statevector · {} amplitudes · |amp|² < 1e-12 hidden",
                    sv.len()
                ))
            );
            let idx_w = (sv.len().saturating_sub(1)).to_string().len().max(3);
            let basis_w = (n as usize + 2).max("basis".len());
            println!(
                "{}",
                sty.dim(&format!(
                    "  {:>idx_w$}  {:<basis_w$}  {:>10}  {:>10}  {:>8}",
                    "idx", "basis", "re", "im", "|amp|²"
                ))
            );
            for (i, amp) in sv.iter().enumerate() {
                let p = amp.norm_sqr();
                if p < 1e-12 {
                    continue;
                }
                let bits: String = (0..n)
                    .rev()
                    .map(|q| char::from(b'0' + ((i >> q) & 1) as u8))
                    .collect();
                println!(
                    "  {i:>idx_w$}  {}  {:>+10.6}  {:>+10.6}  {p:>8.6}",
                    sty.bold(&format!("{:<basis_w$}", format!("|{bits}⟩"))),
                    amp.re,
                    amp.im,
                );
            }
            println!();
        }
        None => {
            println!(
                "statevector: {} amplitudes (printing suppressed; use --json)",
                1u128 << n
            );
            println!();
        }
    }
}

// ---------------------------------------------------------------------------
// kuantum info
// ---------------------------------------------------------------------------

const DEFAULT_FRACTION: f64 = 0.75;

struct PrecisionCapacity {
    precision: Precision,
    max_qubits: u32,
    state_bytes: u64,
    next_qubit_bytes: u64,
}

fn capacity(budget: u64, precision: Precision) -> PrecisionCapacity {
    let max_qubits = mem::max_qubits(budget, precision);
    let clamp = |b: u128| -> u64 { b.min(u64::MAX as u128) as u64 };
    PrecisionCapacity {
        precision,
        max_qubits,
        state_bytes: clamp(mem::state_bytes(max_qubits, precision)),
        next_qubit_bytes: clamp(mem::state_bytes(max_qubits + 1, precision)),
    }
}

fn cmd_info(json: bool) -> Result<(), kuantum::Error> {
    let chip = chip_name();
    let cores = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1);
    let ram = mem::physical_ram_bytes();
    let available = mem::available_now_bytes();
    let budget = mem::budget_bytes(None, DEFAULT_FRACTION);
    let caps = [
        capacity(budget, Precision::F64),
        capacity(budget, Precision::F32),
    ];

    if json {
        let mut precisions = serde_json::Map::new();
        for c in &caps {
            precisions.insert(
                c.precision.to_string(),
                serde_json::json!({
                    "max_qubits": c.max_qubits,
                    "state_bytes": c.state_bytes,
                    "next_qubit_bytes": c.next_qubit_bytes,
                }),
            );
        }
        let obj = serde_json::json!({
            "schema_version": 1,
            "chip": chip,
            "cores": cores,
            "ram_bytes": ram,
            "available_now_bytes": available,
            "budget_bytes": budget,
            "budget_fraction": DEFAULT_FRACTION,
            "precisions": precisions,
            "overrides": ["--mem-limit", "KUANTUM_MEM_BYTES"],
        });
        println!(
            "{}",
            serde_json::to_string_pretty(&obj).expect("valid json")
        );
        return Ok(());
    }

    let sty = Style::detect();
    println!("{}", sty.dim(&format!("kuantum v{VERSION} · {chip}")));
    println!("  cores   {cores}");
    let free = available
        .map(|b| human_bytes(b as u128))
        .unwrap_or_else(|| "n/a".to_string());
    println!(
        "  memory  {} RAM · free now ~{} · budget {} ({:.0}% of RAM)",
        human_bytes(ram as u128),
        free,
        human_bytes(budget as u128),
        DEFAULT_FRACTION * 100.0,
    );
    println!();
    let rows: Vec<Vec<String>> = caps
        .iter()
        .map(|c| {
            vec![
                c.precision.to_string(),
                c.max_qubits.to_string(),
                human_bytes(c.state_bytes as u128),
                human_bytes(c.next_qubit_bytes as u128),
            ]
        })
        .collect();
    print_table(
        &["precision", "max qubits", "state size", "next qubit needs"],
        &rows,
        &[false, true, true, true],
        &sty,
    );
    println!();
    println!(
        "{}",
        sty.dim("--mem-limit and KUANTUM_MEM_BYTES override the budget")
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// kuantum bench
// ---------------------------------------------------------------------------

enum BenchRow {
    Done {
        n: u32,
        gates: usize,
        ops: usize,
        time: Duration,
        amp_per_s: f64,
        baseline: Option<(Duration, f64)>,
        precision: Precision,
        backend: &'static str,
    },
    Skipped {
        n: u32,
        needed: u128,
        budget: u64,
    },
}

fn cmd_bench(a: &BenchArgs) -> Result<(), kuantum::Error> {
    let budget = mem::budget_bytes(None, DEFAULT_FRACTION);
    // Auto precision falls back to f32, so a size only truly exceeds the
    // budget when even the smallest representation does.
    let check_precision = match a.precision {
        PrecisionArg::F64 => Precision::F64,
        _ => Precision::F32,
    };
    let mut rows: Vec<BenchRow> = Vec::with_capacity(a.qubits.len());
    let mut notes: Vec<String> = vec![];
    for &n in &a.qubits {
        let needed = mem::state_bytes(n, check_precision);
        if n > 48 || needed > budget as u128 {
            rows.push(BenchRow::Skipped { n, needed, budget });
            continue;
        }
        let program = kuantum::random_circuit(n, a.depth, a.seed).to_program();
        let opts = RunOptions {
            shots: 1,
            seed: Some(a.seed),
            precision: a.precision.to_option(),
            backend: a.backend.to_choice(),
            fusion_max: a.fusion,
            ..Default::default()
        };
        let r = kuantum::run_program(&program, &opts)?;
        for note in &r.notices {
            // A shots=1 bench on a measurement-free circuit is by design.
            if !note.starts_with("circuit has no measurements") && !notes.contains(note) {
                notes.push(note.clone());
            }
        }
        let secs = r.sim_time.as_secs_f64().max(1e-9);
        let amp_per_s = r.stats.input_gates as f64 * 2f64.powi(n as i32) / secs;
        let baseline = if a.compare_fusion {
            let base_opts = RunOptions {
                fusion_max: 0,
                ..opts
            };
            let r0 = kuantum::run_program(&program, &base_opts)?;
            let speedup = r0.sim_time.as_secs_f64().max(1e-9) / secs;
            Some((r0.sim_time, speedup))
        } else {
            None
        };
        rows.push(BenchRow::Done {
            n,
            gates: r.stats.input_gates,
            ops: r.stats.output_ops,
            time: r.sim_time,
            amp_per_s,
            baseline,
            precision: r.precision,
            backend: r.backend,
        });
    }

    if a.json {
        print_bench_json(a, &rows, &notes);
    } else {
        print_bench_human(a, &rows, &notes);
    }
    Ok(())
}

fn print_bench_json(a: &BenchArgs, rows: &[BenchRow], notes: &[String]) {
    let results: Vec<serde_json::Value> = rows
        .iter()
        .map(|row| match row {
            BenchRow::Done {
                n,
                gates,
                ops,
                time,
                amp_per_s,
                baseline,
                precision,
                backend,
            } => {
                let mut v = serde_json::json!({
                    "qubits": n,
                    "skipped": false,
                    "input_gates": gates,
                    "output_ops": ops,
                    "sim_time_ms": time.as_secs_f64() * 1e3,
                    "amp_updates_per_s": amp_per_s,
                    "precision": precision,
                    "backend": backend,
                });
                if let Some((base, speedup)) = baseline {
                    v["nofusion_time_ms"] = serde_json::json!(base.as_secs_f64() * 1e3);
                    v["speedup"] = serde_json::json!(speedup);
                }
                v
            }
            BenchRow::Skipped { n, needed, budget } => serde_json::json!({
                "qubits": n,
                "skipped": true,
                "reason": format!(
                    "needs {}, budget {}",
                    human_bytes(*needed),
                    human_bytes(*budget as u128)
                ),
            }),
        })
        .collect();
    let obj = serde_json::json!({
        "schema_version": 1,
        "chip": chip_name(),
        "threads": rayon::current_num_threads(),
        "depth": a.depth,
        "fusion": a.fusion,
        "seed": a.seed,
        "precision": a.precision.label(),
        "backend": a.backend.label(),
        "results": results,
        "notices": notes,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&obj).expect("valid json")
    );
}

fn print_bench_human(a: &BenchArgs, rows: &[BenchRow], notes: &[String]) {
    let sty = Style::detect();
    println!("{}", machine_header(rayon::current_num_threads(), &sty));
    println!(
        "{}",
        sty.dim(&format!(
            "bench · depth {} · fusion ≤{} · seed {} · {} · {}",
            a.depth,
            a.fusion,
            a.seed,
            a.precision.label(),
            a.backend.label(),
        ))
    );
    println!();

    let mut headers = vec!["qubits", "gates -> fused ops", "time", "amp-updates/s"];
    let mut right = vec![true, true, true, true];
    if a.compare_fusion {
        headers.push("speedup");
        right.push(true);
    }
    let mut table: Vec<Vec<String>> = vec![];
    let mut skips: Vec<String> = vec![];
    for row in rows {
        match row {
            BenchRow::Done {
                n,
                gates,
                ops,
                time,
                amp_per_s,
                baseline,
                ..
            } => {
                let mut cells = vec![
                    n.to_string(),
                    format!("{gates} -> {ops}"),
                    fmt_duration(*time),
                    fmt_amp_per_s(*amp_per_s),
                ];
                if let Some((_, speedup)) = baseline {
                    cells.push(format!("{}×", sig3(*speedup)));
                }
                table.push(cells);
            }
            BenchRow::Skipped { n, needed, budget } => {
                skips.push(format!(
                    "  {n} qubits skipped: needs {}, budget {}",
                    human_bytes(*needed),
                    human_bytes(*budget as u128)
                ));
            }
        }
    }
    if !table.is_empty() {
        print_table(&headers, &table, &right, &sty);
    }
    for s in &skips {
        println!("{}", sty.dim(s));
    }
    if !table.is_empty() || !skips.is_empty() {
        println!();
    }
    for note in notes {
        println!("{}", sty.dim(&format!("note: {note}")));
    }
}

// ---------------------------------------------------------------------------
// kuantum compile
// ---------------------------------------------------------------------------

fn cmd_compile(a: &CompileArgs) -> Result<(), kuantum::Error> {
    let program = parse_qasm(&a.file)?;
    let compiled = compiler::compile(
        &program,
        &CompileOptions {
            fusion_max: a.fusion,
            ..Default::default()
        },
    )?;
    if a.json {
        print_compile_json(a, &compiled);
    } else {
        print_compile_human(a, &compiled);
    }
    Ok(())
}

fn op_kind(op: &COp) -> &'static str {
    match op {
        COp::Unitary { .. } => "unitary",
        COp::Diagonal { .. } => "diagonal",
        COp::Measure { .. } => "measure",
        COp::Reset { .. } => "reset",
        COp::If { .. } => "if",
        COp::Barrier => "barrier",
    }
}

fn op_qubits(op: &COp) -> Vec<u32> {
    match op {
        COp::Unitary { qubits, .. } | COp::Diagonal { qubits, .. } => qubits.clone(),
        COp::Measure { qubit, .. } | COp::Reset { qubit } => vec![*qubit],
        COp::If { inner, .. } => op_qubits(inner),
        COp::Barrier => vec![],
    }
}

/// `clbit 3` → `c[1]` given the declared cregs.
fn clbit_label(cregs: &[Reg], clbit: u32) -> String {
    let mut offset = 0u32;
    for reg in cregs {
        if clbit < offset + reg.size {
            return format!("{}[{}]", reg.name, clbit - offset);
        }
        offset += reg.size;
    }
    format!("clbit {clbit}")
}

fn creg_label(cregs: &[Reg], creg_offset: u32, creg_len: u32) -> String {
    let mut offset = 0u32;
    for reg in cregs {
        if offset == creg_offset && reg.size == creg_len {
            return reg.name.clone();
        }
        offset += reg.size;
    }
    format!("clbits[{creg_offset}..{}]", creg_offset + creg_len)
}

fn op_detail(op: &COp, cregs: &[Reg]) -> String {
    match op {
        COp::Unitary { qubits, .. } => {
            let dim = 1usize << qubits.len();
            format!("{dim}×{dim} matrix")
        }
        COp::Diagonal { diag, .. } => format!("{}-entry table", diag.len()),
        COp::Measure { clbit, .. } => format!("-> {}", clbit_label(cregs, *clbit)),
        COp::Reset { .. } => "-> |0⟩".to_string(),
        COp::If {
            creg_offset,
            creg_len,
            value,
            inner,
        } => format!(
            "{}=={value} -> {} {}",
            creg_label(cregs, *creg_offset, *creg_len),
            op_kind(inner),
            op_detail(inner, cregs),
        ),
        COp::Barrier => "fence".to_string(),
    }
}

fn compile_rows(c: &compiler::Compiled) -> Vec<(String, Vec<u32>, String, bool)> {
    let mut rows: Vec<(String, Vec<u32>, String, bool)> = c
        .ops
        .iter()
        .map(|op| {
            (
                op_kind(op).to_string(),
                op_qubits(op),
                op_detail(op, &c.cregs),
                false,
            )
        })
        .collect();
    for &(q, cb) in &c.final_measures {
        rows.push((
            "measure".to_string(),
            vec![q],
            format!("-> {}", clbit_label(&c.cregs, cb)),
            true,
        ));
    }
    rows
}

fn print_compile_human(a: &CompileArgs, c: &compiler::Compiled) {
    let sty = Style::detect();
    let stem = a
        .file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("circuit");
    println!(
        "{}",
        sty.dim(&format!("kuantum v{VERSION} · fusion ≤{}", a.fusion))
    );
    println!(
        "{} · {} · {}",
        sty.bold(stem),
        plural(c.n_qubits as usize, "qubit"),
        if c.dynamic { "dynamic" } else { "static" },
    );
    println!(
        "input gates {} · cancelled {} · output ops {} · max fused {}",
        c.stats.input_gates, c.stats.cancelled, c.stats.output_ops, c.stats.max_fused,
    );
    println!();
    let rows = compile_rows(c);
    if rows.is_empty() {
        println!("{}", sty.dim("  (no executable ops)"));
        return;
    }
    let table: Vec<Vec<String>> = rows
        .iter()
        .enumerate()
        .map(|(i, (kind, qubits, detail, _))| {
            let qs = qubits
                .iter()
                .map(|q| q.to_string())
                .collect::<Vec<_>>()
                .join(",");
            vec![i.to_string(), kind.clone(), qs, detail.clone()]
        })
        .collect();
    print_table(
        &["#", "kind", "qubits", "detail"],
        &table,
        &[true, false, false, false],
        &sty,
    );
    println!();
    if c.dynamic {
        println!(
            "{}",
            sty.dim("dynamic circuit: re-executed per shot (measure/reset/if mid-circuit)")
        );
    } else if !c.final_measures.is_empty() {
        println!(
            "{}",
            sty.dim("trailing measurements are sampled analytically from |ψ|²")
        );
    }
}

fn print_compile_json(a: &CompileArgs, c: &compiler::Compiled) {
    let ops: Vec<serde_json::Value> = compile_rows(c)
        .iter()
        .enumerate()
        .map(|(i, (kind, qubits, detail, final_measure))| {
            serde_json::json!({
                "index": i,
                "kind": kind,
                "qubits": qubits,
                "detail": detail,
                "final": final_measure,
            })
        })
        .collect();
    let obj = serde_json::json!({
        "schema_version": 1,
        "file": a.file.display().to_string(),
        "n_qubits": c.n_qubits,
        "fusion": a.fusion,
        "dynamic": c.dynamic,
        "stats": c.stats,
        "ops": ops,
    });
    println!(
        "{}",
        serde_json::to_string_pretty(&obj).expect("valid json")
    );
}

// ---------------------------------------------------------------------------
// Unit tests (parsers and formatters)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_suffixes() {
        assert_eq!(parse_size("8g").unwrap(), 8 << 30);
        assert_eq!(parse_size("512m").unwrap(), 512 << 20);
        assert_eq!(parse_size("1024k").unwrap(), 1 << 20);
        assert_eq!(parse_size("123456").unwrap(), 123_456);
        assert_eq!(parse_size("1t").unwrap(), 1 << 40);
    }

    #[test]
    fn size_case_and_long_forms() {
        assert_eq!(parse_size("8G").unwrap(), 8 << 30);
        assert_eq!(parse_size("8GiB").unwrap(), 8 << 30);
        assert_eq!(parse_size("8gb").unwrap(), 8 << 30);
        assert_eq!(parse_size(" 2m ").unwrap(), 2 << 20);
    }

    #[test]
    fn size_fractional() {
        assert_eq!(parse_size("1.5g").unwrap(), 3 << 29);
        assert_eq!(parse_size("0.5k").unwrap(), 512);
    }

    #[test]
    fn size_rejects_garbage() {
        for s in ["", "g", "12x", "1.5", "-3m", "1..2g", "8 g g"] {
            assert!(parse_size(s).is_err(), "should reject {s:?}");
        }
    }

    #[test]
    fn seed_decimal_and_hex() {
        assert_eq!(parse_seed("42").unwrap(), 42);
        assert_eq!(parse_seed("0xff").unwrap(), 255);
        assert_eq!(parse_seed("0XDEADBEEF").unwrap(), 0xDEAD_BEEF);
        assert!(parse_seed("zz").is_err());
        assert!(parse_seed("0x").is_err());
    }

    #[test]
    fn fraction_bounds() {
        assert_eq!(parse_fraction("0.75").unwrap(), 0.75);
        assert_eq!(parse_fraction("1").unwrap(), 1.0);
        assert!(parse_fraction("0").is_err());
        assert!(parse_fraction("1.5").is_err());
        assert!(parse_fraction("x").is_err());
    }

    #[test]
    fn duration_units() {
        assert_eq!(fmt_duration(Duration::from_nanos(950)), "950 ns");
        assert_eq!(fmt_duration(Duration::from_micros(12)), "12.0 µs");
        assert_eq!(fmt_duration(Duration::from_millis(3)), "3.00 ms");
        assert_eq!(fmt_duration(Duration::from_secs_f64(2.5)), "2.50 s");
    }

    #[test]
    fn three_sig_figs() {
        assert_eq!(sig3(123.4), "123");
        assert_eq!(sig3(12.34), "12.3");
        assert_eq!(sig3(1.234), "1.23");
        assert_eq!(sig3(0.1234), "0.123");
        assert_eq!(sig3(0.01234), "0.0123");
    }

    #[test]
    fn amp_per_s_units() {
        assert_eq!(fmt_amp_per_s(3.21e9), "3.21 G");
        assert_eq!(fmt_amp_per_s(4.5e6), "4.50 M");
        assert_eq!(fmt_amp_per_s(1.82e3), "1.82 k");
        assert_eq!(fmt_amp_per_s(12.0), "12.0");
    }

    #[test]
    fn clbit_labels() {
        let cregs = vec![
            Reg {
                name: "a".into(),
                size: 2,
            },
            Reg {
                name: "b".into(),
                size: 3,
            },
        ];
        assert_eq!(clbit_label(&cregs, 0), "a[0]");
        assert_eq!(clbit_label(&cregs, 3), "b[1]");
        assert_eq!(creg_label(&cregs, 2, 3), "b");
        assert_eq!(creg_label(&cregs, 1, 1), "clbits[1..2]");
    }

    #[test]
    fn cli_parses() {
        use clap::CommandFactory;
        Cli::command().debug_assert();
    }

    #[test]
    fn plurals() {
        assert_eq!(plural(1, "op"), "1 op");
        assert_eq!(plural(2, "op"), "2 ops");
        assert_eq!(plural(0, "gate"), "0 gates");
    }

    /// Drives the compile visualizer (static and dynamic circuits) so table
    /// rendering and creg labelling are verified before the parser lands.
    #[test]
    fn compile_printers_do_not_panic() {
        let args = CompileArgs {
            file: PathBuf::from("demo.qasm"),
            fusion: 5,
            json: false,
        };
        // Static: trailing measurements split from the fused body.
        let mut ghz = kuantum::Circuit::new(3);
        ghz.h(0).cx(0, 1).cx(1, 2).measure_all();
        let compiled = compiler::compile(&ghz.to_program(), &CompileOptions::default()).unwrap();
        assert!(!compiled.dynamic);
        print_compile_human(&args, &compiled);
        print_compile_json(&args, &compiled);
        // Dynamic: a mid-circuit reset forces per-shot execution.
        let mut dyn_c = kuantum::Circuit::new(2);
        dyn_c.h(0).measure(0, 0).reset(0).cx(0, 1).measure_all();
        let compiled = compiler::compile(&dyn_c.to_program(), &CompileOptions::default()).unwrap();
        assert!(compiled.dynamic);
        print_compile_human(&args, &compiled);
        print_compile_json(&args, &compiled);
    }

    /// The QASM path is exercised end-to-end in tests/cli.rs; this drives
    /// every run printer (histogram, statevector, JSON) through the library
    /// so formatting bugs surface before the parser lands.
    #[test]
    fn run_printers_do_not_panic() {
        let mut c = kuantum::Circuit::new(3);
        c.h(0).cx(0, 1).cx(1, 2).measure_all();
        let opts = RunOptions {
            shots: 256,
            seed: Some(9),
            want_statevector: true,
            ..Default::default()
        };
        let r = kuantum::run_program(&c.to_program(), &opts).unwrap();
        for (json, quiet, statevector) in [
            (false, false, true),
            (false, true, false),
            (true, false, true),
        ] {
            let args = RunArgs {
                file: PathBuf::from("ghz.qasm"),
                shots: 256,
                seed: Some(9),
                precision: PrecisionArg::Auto,
                backend: BackendArg::Auto,
                fusion: 5,
                mem_limit: None,
                mem_fraction: 0.75,
                threads: None,
                statevector,
                json,
                quiet,
            };
            if json {
                print_run_json(&args, &r);
            } else {
                print_run_human(&args, &r);
            }
        }
    }
}
