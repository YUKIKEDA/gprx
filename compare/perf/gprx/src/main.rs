//! gprx side of the `compare/perf` harnesses: one cell per process, one JSON
//! row on the last stdout line.
//!
//! ```text
//! gprx-perf exact CASE.json [--memory]
//! gprx-perf online CASE.json [--stages | --delete]
//! gprx-perf sparse CASE.json
//! gprx-perf sparse-online CASE.json incremental|full
//! ```
//!
//! A cell that cannot run still prints a row, with `status: "na"`.

mod case;
mod exact;
mod online;
mod rss;
mod shared;
mod sparse;
mod sparse_online;
mod timing;

use std::env;
use std::fs;
use std::process::ExitCode;

use serde::Serialize;
use serde::de::DeserializeOwned;

use case::{Case, ResultRow, SparseCase, SparseOnlineCase, SparseOnlineResult};

const USAGE: &str = "usage: gprx-perf exact|online|sparse|sparse-online CASE.json [flags]\n  \
     exact [--memory]\n  online [--stages|--delete]\n  sparse-online incremental|full";

fn read_case<T: DeserializeOwned>(path: &str) -> Result<T, String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    serde_json::from_str(&text).map_err(|e| e.to_string())
}

fn print_row<R: Serialize>(row: &R) -> ExitCode {
    match serde_json::to_string(row) {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::from(1)
        }
    }
}

fn exact(path: &str, flags: &[String]) -> Result<ExitCode, String> {
    let case: Case = read_case(path)?;
    let memory = flags.iter().any(|flag| flag == "--memory");
    let lib = if memory { "gprx-memory" } else { "gprx" };
    let row = exact::run(&case, memory).unwrap_or_else(|e| ResultRow::na(lib, &case.name, e));
    Ok(print_row(&row))
}

fn online(path: &str, flags: &[String]) -> Result<ExitCode, String> {
    let case: Case = read_case(path)?;
    let row = if flags.iter().any(|flag| flag == "--delete") {
        online::run_delete(&case)
    } else {
        online::run_insert(&case, flags.iter().any(|flag| flag == "--stages"))
    };
    let row = row.unwrap_or_else(|e| ResultRow::na("gprx", &case.name, e));
    Ok(print_row(&row))
}

fn sparse(path: &str) -> Result<ExitCode, String> {
    let case: SparseCase = read_case(path)?;
    let row = sparse::run(&case).unwrap_or_else(|e| ResultRow::na("gprx", &case.name, e));
    Ok(print_row(&row))
}

fn sparse_online(path: &str, flags: &[String]) -> Result<ExitCode, String> {
    let Some(mode) = flags.first() else {
        return Err(USAGE.to_string());
    };
    let case: SparseOnlineCase = read_case(path)?;
    let (lib, row) = match mode.as_str() {
        "incremental" => ("gprx-inc", sparse_online::run_incremental(&case)),
        "full" => ("gprx-full", sparse_online::run_full(&case)),
        other => ("gprx-inc", Err(format!("unknown mode {other}"))),
    };
    let row = row.unwrap_or_else(|e| SparseOnlineResult::na(lib, &case.name, e));
    Ok(print_row(&row))
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let (Some(mode), Some(path)) = (args.first(), args.get(1)) else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let flags = &args[2..];
    let result = match mode.as_str() {
        "exact" => exact(path, flags),
        "online" => online(path, flags),
        "sparse" => sparse(path),
        "sparse-online" => sparse_online(path, flags),
        _ => {
            eprintln!("{USAGE}");
            return ExitCode::from(2);
        }
    };
    result.unwrap_or_else(|e| {
        eprintln!("{e}");
        ExitCode::from(1)
    })
}
