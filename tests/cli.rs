//! End-to-end tests of the `zeno` binary, run via CARGO_BIN_EXE_zeno:
//! help surfaces, `info`/`bench`/`compile` JSON shapes, error paths, and
//! QASM end-to-end runs (bell counts, determinism, histogram, statevector).

use std::path::PathBuf;
use std::process::{Command, Output};

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_zeno"))
}

fn run(args: &[&str]) -> Output {
    bin().args(args).output().expect("binary runs")
}

fn stdout(out: &Output) -> String {
    String::from_utf8_lossy(&out.stdout).into_owned()
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn json(out: &Output) -> serde_json::Value {
    serde_json::from_slice(&out.stdout)
        .unwrap_or_else(|e| panic!("stdout is not valid JSON ({e}):\n{}", stdout(out)))
}

/// Write a Bell-pair circuit into the cargo tempdir and return its path.
fn bell_qasm() -> PathBuf {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tempdir");
    let path = dir.join("bell.qasm");
    std::fs::write(
        &path,
        "OPENQASM 2.0;\n\
         include \"qelib1.inc\";\n\
         qreg q[2];\n\
         creg c[2];\n\
         h q[0];\n\
         cx q[0],q[1];\n\
         measure q -> c;\n",
    )
    .expect("write bell.qasm");
    path
}

// ---------------------------------------------------------------------------
// Help / usability (green now)
// ---------------------------------------------------------------------------

#[test]
fn help_top_level_lists_all_subcommands() {
    let out = run(&["--help"]);
    assert!(out.status.success());
    let text = stdout(&out);
    for sub in ["run", "info", "bench", "compile"] {
        assert!(text.contains(sub), "help must mention `{sub}`:\n{text}");
    }
}

#[test]
fn help_run_shows_all_flags() {
    let out = run(&["run", "--help"]);
    assert!(out.status.success());
    let text = stdout(&out);
    for flag in [
        "--shots",
        "--seed",
        "--precision",
        "--backend",
        "--fusion",
        "--mem-limit",
        "--mem-fraction",
        "--threads",
        "--statevector",
        "--json",
        "--quiet",
    ] {
        assert!(
            text.contains(flag),
            "run --help must mention `{flag}`:\n{text}"
        );
    }
}

#[test]
fn help_other_subcommands() {
    for (sub, flag) in [
        ("info", "--json"),
        ("bench", "--compare-fusion"),
        ("compile", "--fusion"),
    ] {
        let out = run(&[sub, "--help"]);
        assert!(out.status.success(), "{sub} --help failed");
        let text = stdout(&out);
        assert!(
            text.contains(flag),
            "{sub} --help must mention `{flag}`:\n{text}"
        );
    }
}

#[test]
fn version_flag() {
    let out = run(&["--version"]);
    assert!(out.status.success());
    assert!(stdout(&out).contains("zeno"));
}

// ---------------------------------------------------------------------------
// info (green now)
// ---------------------------------------------------------------------------

#[test]
fn info_json_has_expected_shape() {
    // 16 GiB pretend-RAM -> budget 12 GiB -> 29 qubits f64, 30 qubits f32.
    let out = bin()
        .args(["info", "--json"])
        .env("ZENO_MEM_BYTES", (16u64 << 30).to_string())
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    assert_eq!(v["schema_version"], 1);
    assert!(v["chip"].is_string());
    assert!(v["cores"].as_u64().unwrap() >= 1);
    assert_eq!(v["ram_bytes"].as_u64().unwrap(), 16 << 30);
    assert_eq!(v["budget_bytes"].as_u64().unwrap(), 12 << 30);
    assert_eq!(v["budget_fraction"], 0.75);
    assert_eq!(v["precisions"]["f64"]["max_qubits"], 29);
    assert_eq!(v["precisions"]["f32"]["max_qubits"], 30);
    assert_eq!(
        v["precisions"]["f64"]["state_bytes"].as_u64().unwrap(),
        (1u64 << 29) * 16
    );
    assert_eq!(
        v["precisions"]["f64"]["next_qubit_bytes"].as_u64().unwrap(),
        (1u64 << 30) * 16
    );
    let overrides = v["overrides"].as_array().unwrap();
    assert!(overrides.iter().any(|o| o == "ZENO_MEM_BYTES"));
}

#[test]
fn info_human_mentions_overrides_and_stays_plain_when_piped() {
    let out = run(&["info"]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("ZENO_MEM_BYTES"));
    assert!(text.contains("--mem-limit"));
    assert!(text.contains("f64") && text.contains("f32"));
    assert!(
        !out.stdout.contains(&0x1b),
        "piped output must not contain ANSI escapes"
    );
}

// ---------------------------------------------------------------------------
// bench (green now)
// ---------------------------------------------------------------------------

#[test]
fn bench_json_small_sweep() {
    let out = run(&["bench", "--json", "--qubits", "4,5", "--depth", "3"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["depth"], 3);
    assert_eq!(v["seed"], 7);
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    for (i, expect_n) in [(0usize, 4u64), (1, 5)] {
        let r = &results[i];
        assert_eq!(r["qubits"].as_u64().unwrap(), expect_n);
        assert_eq!(r["skipped"], false);
        assert!(r["input_gates"].as_u64().unwrap() > 0);
        assert!(r["output_ops"].as_u64().unwrap() > 0);
        assert!(r["sim_time_ms"].as_f64().unwrap() >= 0.0);
        assert!(r["amp_updates_per_s"].as_f64().unwrap() > 0.0);
    }
}

#[test]
fn bench_json_compare_fusion_adds_speedup() {
    let out = run(&[
        "bench",
        "--json",
        "--qubits",
        "4",
        "--depth",
        "2",
        "--compare-fusion",
    ]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    let r = &v["results"][0];
    assert!(r["speedup"].as_f64().unwrap() > 0.0);
    assert!(r["nofusion_time_ms"].as_f64().unwrap() >= 0.0);
}

#[test]
fn bench_skips_sizes_over_the_budget() {
    // Pretend 1 MiB of RAM: 4 qubits fit, 26 do not — must not error out.
    let out = bin()
        .args(["bench", "--json", "--qubits", "4,26", "--depth", "2"])
        .env("ZENO_MEM_BYTES", (1u64 << 20).to_string())
        .output()
        .unwrap();
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    let results = v["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["skipped"], false);
    assert_eq!(results[1]["skipped"], true);
    assert!(results[1]["reason"].as_str().unwrap().contains("budget"));
}

#[test]
fn bench_human_table_is_aligned_and_plain_when_piped() {
    let out = run(&["bench", "--qubits", "4", "--depth", "2"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("amp-updates/s"), "missing header:\n{text}");
    assert!(text.contains("qubits"));
    assert!(
        !out.stdout.contains(&0x1b),
        "piped output must not contain ANSI escapes"
    );
}

// ---------------------------------------------------------------------------
// Error paths (green now)
// ---------------------------------------------------------------------------

#[test]
fn run_missing_file_is_a_clean_error() {
    let out = run(&["run", "/definitely/not/here.qasm"]);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.starts_with("error:"), "stderr: {err}");
    assert!(err.contains("here.qasm"), "error must name the file: {err}");
    assert!(stdout(&out).is_empty());
}

#[test]
fn compile_missing_file_is_a_clean_error() {
    let out = run(&["compile", "/definitely/not/here.qasm"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(stderr(&out).starts_with("error:"));
}

#[test]
fn bad_mem_limit_is_a_clap_usage_error() {
    let out = run(&["run", "x.qasm", "--mem-limit", "12xq"]);
    assert_eq!(out.status.code(), Some(2), "clap usage errors exit 2");
    assert!(stderr(&out).contains("invalid size"));
}

#[test]
fn bad_mem_fraction_is_a_clap_usage_error() {
    let out = run(&["run", "x.qasm", "--mem-fraction", "1.5"]);
    assert_eq!(out.status.code(), Some(2));
}

#[test]
fn unknown_subcommand_exits_2() {
    let out = run(&["frobnicate"]);
    assert_eq!(out.status.code(), Some(2));
}

// ---------------------------------------------------------------------------
// QASM end-to-end
// ---------------------------------------------------------------------------

#[test]
fn run_bell_json_counts() {
    let bell = bell_qasm();
    let out = run(&[
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "42",
        "--shots",
        "1024",
        "--json",
    ]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["n_qubits"], 2);
    assert_eq!(v["shots"], 1024);
    assert_eq!(v["seed"], 42);
    let counts = v["counts"].as_object().unwrap();
    let mut total = 0u64;
    for (key, n) in counts {
        assert!(
            key == "00" || key == "11",
            "bell run may only produce 00/11, got {key}"
        );
        total += n.as_u64().unwrap();
    }
    assert_eq!(total, 1024);
    assert!(v["stats"]["input_gates"].as_u64().unwrap() >= 2);
    assert!(!out.stdout.contains(&0x1b), "no ANSI in --json output");
}

#[test]
fn run_bell_json_is_seed_deterministic() {
    let bell = bell_qasm();
    let args = [
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "7",
        "--shots",
        "512",
        "--json",
    ];
    let a = json(&run(&args));
    let b = json(&run(&args));
    assert_eq!(a["counts"], b["counts"]);
}

#[test]
fn run_bell_human_histogram() {
    let bell = bell_qasm();
    let out = run(&["run", bell.to_str().unwrap(), "--seed", "42"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("▇"), "histogram bars expected:\n{text}");
    assert!(text.contains("qubits"), "summary line expected:\n{text}");
    assert!(
        text.contains("seed 0x000000000000002a"),
        "seed line expected:\n{text}"
    );
    assert!(!out.stdout.contains(&0x1b), "no ANSI when piped");
}

#[test]
fn run_bell_quiet_is_just_the_histogram() {
    let bell = bell_qasm();
    let out = run(&["run", bell.to_str().unwrap(), "--seed", "42", "--quiet"]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("▇"));
    assert!(!text.contains("seed 0x"), "quiet must drop the seed line");
    assert!(!text.contains("zeno v"), "quiet must drop the header");
}

#[test]
fn run_bell_statevector_json() {
    let bell = bell_qasm();
    let out = run(&[
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "1",
        "--statevector",
        "--json",
    ]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    let sv = v["statevector"].as_array().unwrap();
    assert_eq!(sv.len(), 4);
    let amp = |i: usize| {
        let re = sv[i][0].as_f64().unwrap();
        let im = sv[i][1].as_f64().unwrap();
        (re * re + im * im).sqrt()
    };
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    assert!((amp(0) - inv_sqrt2).abs() < 1e-9);
    assert!((amp(3) - inv_sqrt2).abs() < 1e-9);
    assert!(amp(1) < 1e-12 && amp(2) < 1e-12);
}

#[test]
fn compile_bell_shows_stats_and_ops() {
    let bell = bell_qasm();
    let out = run(&["compile", bell.to_str().unwrap()]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("input gates"), "stats expected:\n{text}");
    assert!(text.contains("measure"), "op list expected:\n{text}");

    let out = run(&["compile", bell.to_str().unwrap(), "--json"]);
    assert!(out.status.success());
    let v = json(&out);
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["n_qubits"], 2);
    assert!(v["stats"]["input_gates"].as_u64().unwrap() >= 2);
    assert!(!v["ops"].as_array().unwrap().is_empty());
}
