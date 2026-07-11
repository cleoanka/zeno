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
///
/// Each call gets a unique file: tests run in parallel, and a shared path
/// would let one test truncate the file while another test's child process
/// is reading it (observed as a flake on 2-core CI runners).
fn bell_qasm() -> PathBuf {
    use std::sync::atomic::{AtomicUsize, Ordering};
    static N: AtomicUsize = AtomicUsize::new(0);
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tempdir");
    let path = dir.join(format!(
        "bell_{}_{}.qasm",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ));
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
    .expect("write bell qasm");
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

// ---------------------------------------------------------------------------
// zeno demo
// ---------------------------------------------------------------------------

/// Seeded JSON run of a demo, returning the parsed output.
fn demo_json(name: &str, extra: &[&str]) -> serde_json::Value {
    let mut args = vec!["demo", name, "--seed", "7", "--json"];
    args.extend_from_slice(extra);
    let out = run(&args);
    assert!(out.status.success(), "demo {name}: {}", stderr(&out));
    json(&out)
}

#[test]
fn demo_default_is_bell_and_shows_source_and_next_step() {
    let out = run(&["demo", "--seed", "7"]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(text.contains("OPENQASM 2.0;"), "must show source:\n{text}");
    assert!(text.contains("cx q[0], q[1];"), "must show gates:\n{text}");
    assert!(text.contains("▇"), "histogram expected:\n{text}");
    assert!(
        text.contains("Try next: zeno demo ghz"),
        "next-step line expected:\n{text}"
    );
    assert!(text.contains("seed 0x"), "seed line expected:\n{text}");
    assert!(!out.stdout.contains(&0x1b), "no ANSI when piped");
}

#[test]
fn demo_bell_counts_are_only_00_and_11() {
    let v = demo_json("bell", &[]);
    assert_eq!(v["demo"], "bell");
    let counts = v["counts"].as_object().unwrap();
    let mut total = 0u64;
    for (key, n) in counts {
        assert!(key == "00" || key == "11", "unexpected key {key}");
        total += n.as_u64().unwrap();
    }
    assert_eq!(total, 1000, "demo default is 1000 shots");
}

#[test]
fn demo_ghz_counts_are_only_all_zeros_and_all_ones() {
    let v = demo_json("ghz", &[]);
    for key in v["counts"].as_object().unwrap().keys() {
        assert!(
            key == "00000000" || key == "11111111",
            "unexpected key {key}"
        );
    }
}

#[test]
fn demo_qft_round_trip_is_deterministic_101() {
    let v = demo_json("qft", &[]);
    let counts = v["counts"].as_object().unwrap();
    assert_eq!(counts.len(), 1, "round trip must be exact: {counts:?}");
    assert_eq!(counts["101"].as_u64().unwrap(), 1000);
}

#[test]
fn demo_grover_101_dominates() {
    let v = demo_json("grover", &[]);
    let counts = v["counts"].as_object().unwrap();
    let marked = counts["101"].as_u64().unwrap();
    for (key, n) in counts {
        if key != "101" {
            assert!(
                n.as_u64().unwrap() < marked,
                "'101' must dominate, but {key} has {n}"
            );
        }
    }
    // ~94.5% expected; 850/1000 is > 9 sigma below, so this cannot flake.
    assert!(marked > 850, "grover success rate too low: {marked}/1000");
}

#[test]
fn demo_teleport_out_bit_is_always_1() {
    let v = demo_json("teleport", &[]);
    for key in v["counts"].as_object().unwrap().keys() {
        // Key is "out m1 m0" (last-declared creg leftmost).
        assert!(
            key.starts_with('1'),
            "teleported payload must read 1, got key {key}"
        );
    }
}

#[test]
fn demo_noisy_leaks_beyond_00_and_11_and_reports_the_model() {
    let v = demo_json("noisy", &[]);
    let counts = v["counts"].as_object().unwrap();
    assert!(
        counts.len() > 2,
        "noise must produce keys beyond 00/11: {counts:?}"
    );
    assert_eq!(v["noise"]["depolarizing_2q"], 0.05);
    let notices = v["notices"].as_array().unwrap();
    assert!(
        notices
            .iter()
            .any(|n| n.as_str().unwrap().contains("trajectory sampling")),
        "trajectory notice expected: {notices:?}"
    );

    // Human mode teaches the real flag with the equivalent JSON.
    let out = run(&["demo", "noisy", "--seed", "7"]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(
        text.contains("--noise '{"),
        "must print the equivalent --noise flag:\n{text}"
    );
    assert!(
        text.contains("note: noise: trajectory sampling"),
        "human output must surface the notice:\n{text}"
    );
}

#[test]
fn demo_shots_and_seed_pass_through() {
    let v = demo_json("bell", &["--shots", "200"]);
    assert_eq!(v["shots"], 200);
    let total: u64 = v["counts"]
        .as_object()
        .unwrap()
        .values()
        .map(|n| n.as_u64().unwrap())
        .sum();
    assert_eq!(total, 200);
    assert_eq!(v["seed"], 7);
    // Same seed, same counts.
    assert_eq!(
        demo_json("bell", &["--shots", "200"])["counts"],
        v["counts"]
    );
}

#[test]
fn demo_json_shape() {
    let v = demo_json("bell", &[]);
    assert_eq!(v["schema_version"], 1);
    assert_eq!(v["demo"], "bell");
    assert_eq!(v["n_qubits"], 2);
    let source = v["source"].as_str().unwrap();
    assert!(source.starts_with("OPENQASM 2.0;"));
    assert!(source.contains("measure q -> c;"));
    assert!(v["noise"].is_null(), "ideal demo carries no noise key");
    assert!(v["stats"]["input_gates"].as_u64().unwrap() >= 2);
}

#[test]
fn demo_list_names_every_demo() {
    let out = run(&["demo", "--list"]);
    assert!(out.status.success());
    let text = stdout(&out);
    for name in ["bell", "ghz", "qft", "grover", "teleport", "noisy"] {
        assert!(text.contains(name), "--list must mention {name}:\n{text}");
    }
}

#[test]
fn demo_unknown_name_is_a_friendly_exit_1() {
    let out = run(&["demo", "warp-drive"]);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("warp-drive"), "must echo the name: {err}");
    for name in ["bell", "ghz", "qft", "grover", "teleport", "noisy"] {
        assert!(err.contains(name), "must list option {name}: {err}");
    }
}

// ---------------------------------------------------------------------------
// zeno run --noise
// ---------------------------------------------------------------------------

#[test]
fn run_help_mentions_noise() {
    let out = run(&["run", "--help"]);
    assert!(out.status.success());
    let text = stdout(&out);
    assert!(text.contains("--noise"), "run --help:\n{text}");
    assert!(
        text.contains("bit_flip=0.01"),
        "help must carry an example:\n{text}"
    );
}

#[test]
fn run_noise_key_value_changes_bell_counts() {
    let bell = bell_qasm();
    let base = &[
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "7",
        "--shots",
        "2000",
        "--json",
    ];
    let ideal = json(&run(base));
    let mut noisy_args = base.to_vec();
    noisy_args.extend_from_slice(&["--noise", "bit_flip=0.1"]);
    let noisy = json(&run(&noisy_args));
    assert_ne!(
        ideal["counts"], noisy["counts"],
        "bit_flip=0.1 must change counts at the same seed"
    );
    assert!(
        noisy["counts"].as_object().unwrap().len() > 2,
        "bit flips must leak into 01/10: {}",
        noisy["counts"]
    );
    assert_eq!(noisy["noise"]["bit_flip"], 0.1);
    assert!(ideal["noise"].is_null(), "no flag, no noise key");
}

#[test]
fn run_noise_invalid_key_lists_the_valid_keys() {
    let bell = bell_qasm();
    let out = run(&["run", bell.to_str().unwrap(), "--noise", "bitflip=0.1"]);
    assert_eq!(out.status.code(), Some(1));
    let err = stderr(&out);
    assert!(err.contains("bitflip"), "must echo the typo: {err}");
    for key in [
        "depolarizing_1q",
        "depolarizing_2q",
        "bit_flip",
        "phase_flip",
        "amplitude_damping",
        "readout_flip_0to1",
        "readout_flip_1to0",
    ] {
        assert!(err.contains(key), "must list valid key {key}: {err}");
    }
}

#[test]
fn run_noise_json_file_path() {
    // Unique per-test filename: a shared name is a parallel-test race.
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR"));
    std::fs::create_dir_all(&dir).expect("tempdir");
    let path = dir.join(format!("noise_file_path_{}.json", std::process::id()));
    std::fs::write(
        &path,
        r#"{"readout_flip_0to1": 1.0, "readout_flip_1to0": 1.0}"#,
    )
    .expect("write noise json");
    let bell = bell_qasm();
    let out = run(&[
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "7",
        "--shots",
        "300",
        "--noise",
        path.to_str().unwrap(),
        "--json",
    ]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let v = json(&out);
    assert_eq!(v["noise"]["readout_flip_0to1"], 1.0);
    // Both readout bits always flip: the Bell keys 00/11 become 11/00,
    // so the key set is unchanged — but the notice must be present.
    let notices = v["notices"].as_array().unwrap();
    assert!(notices
        .iter()
        .any(|n| n.as_str().unwrap().contains("trajectory sampling")));
    for key in v["counts"].as_object().unwrap().keys() {
        assert!(key == "00" || key == "11", "unexpected key {key}");
    }
}

#[test]
fn run_noise_human_output_surfaces_the_notice() {
    let bell = bell_qasm();
    let out = run(&[
        "run",
        bell.to_str().unwrap(),
        "--seed",
        "7",
        "--noise",
        "phase_flip=0.05",
    ]);
    assert!(out.status.success(), "stderr: {}", stderr(&out));
    let text = stdout(&out);
    assert!(
        text.contains("note: noise: trajectory sampling"),
        "notice expected in human output:\n{text}"
    );
}
