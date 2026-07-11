use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "kuantum",
    version,
    about = "Apple Silicon-native quantum circuit simulator"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Run an OpenQASM 2.0 circuit
    Run {
        file: std::path::PathBuf,
        #[arg(long, default_value_t = 1024)]
        shots: u64,
        #[arg(long)]
        seed: Option<u64>,
    },
    /// Show machine capacity (max qubits by precision)
    Info,
}

fn main() {
    let cli = Cli::parse();
    let code = match cli.cmd {
        Cmd::Run { file, shots, seed } => {
            let opts = kuantum::RunOptions {
                shots,
                seed,
                ..Default::default()
            };
            match kuantum::run_qasm_file(&file, &opts) {
                Ok(r) => {
                    for (k, v) in r.counts.iter() {
                        println!("{k} {v}");
                    }
                    0
                }
                Err(e) => {
                    eprintln!("error: {e}");
                    1
                }
            }
        }
        Cmd::Info => {
            let ram = kuantum::mem::physical_ram_bytes();
            println!("RAM: {}", kuantum::human_bytes(ram as u128));
            for p in [kuantum::Precision::F64, kuantum::Precision::F32] {
                println!("max qubits ({p}): {}", kuantum::max_qubits_default(p));
            }
            0
        }
    };
    std::process::exit(code);
}
