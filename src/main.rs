//! Crucible-LLM entry point.
//!
//! Dispatch (plan Chunk 7):
//! - `crucible-llm` → banner (no target specified: no `--url`/env/config
//!   file URL)
//! - `crucible-llm --tui …` → the ratatui dashboard (Chunk 8)
//! - `crucible-llm --url … [flags]` → **headless** single-stream
//!   benchmark (Engine A): N iterations, the prototype's result box
//!   and/or `--json` (same field set as `llmspeedtest.py`), a
//!   multi-iteration summary, and exit 1 if all runs failed.

use std::io::IsTerminal;
use std::panic;
use std::process;

use crucible_llm::config::{Config, ConfigError, ExportFormat};
use crucible_llm::engines::speed::{
    all_failed, format_result_box, format_summary, json_report, SpeedEngine,
};
use crucible_llm::storage::export::{self, ExportPayload, PacketSample};
use crucible_llm::storage::{BenchmarkSession, Database, StreamMetricRow};
use crucible_llm::ui::app::App;
use crucible_llm::ui::event::EventLoop;

fn main() {
    let cfg = match Config::from_cli() {
        Ok(cfg) => cfg,
        // `--help` / `--version` / bad flags: clap prints the right text to
        // the right stream and exits with its own code (0 for help).
        Err(ConfigError::Clap(e)) => e.exit(),
        Err(e) => {
            eprintln!("crucible-llm: {e}");
            process::exit(2);
        }
    };

    if cfg.tui {
        let ok = run_tui(&cfg);
        EventLoop::restore(); // best-effort terminal restore on all paths
        if !ok {
            process::exit(1);
        }
        return;
    }

    if cfg.target_explicit {
        process::exit(run_headless(&cfg));
    }

    print_banner();
}

/// The headless (non-TUI) path: run N single-stream iterations (Engine A)
/// and print the prototype's result box(es) / summary — or `--json` on
/// stdout. Returns the process exit code (1 iff every run failed).
fn run_headless(cfg: &Config) -> i32 {
    let rt = match tokio::runtime::Builder::new_current_thread()
        .enable_time()
        .enable_io() // the stream worker does real network I/O
        .build()
    {
        Ok(rt) => rt,
        Err(e) => {
            eprintln!("crucible-llm: failed to build runtime: {e}");
            return 1;
        }
    };
    rt.block_on(async {
        let engine = match SpeedEngine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("crucible-llm: {e}");
                return 1;
            }
        };

        let term = Term {
            color: !cfg.no_color && std::io::stdout().is_terminal(),
            tty: std::io::stderr().is_terminal(),
        };

        let prompt = engine.generate_prompt();
        let iterations = cfg.iterations.max(1) as usize;

        if !cfg.json {
            term.info(&format!(
                "Mode: {} | Prompt: {} chars (~{} tok) | Target: {}",
                cfg.mode.label(),
                prompt.text.chars().count(),
                prompt.token_count,
                cfg.url
            ));
            if cfg.nocache {
                term.dim("  KV-cache bypass enabled");
            }
            if iterations > 1 {
                term.dim(&format!("  Running {iterations} iterations"));
            }
            if let Some(fmt) = cfg.export {
                term.dim(&format!("  --export {} enabled", fmt.label()));
            }
        }

        // With `--export`, also keep each iteration's raw worker events —
        // the per-packet arrival timestamps the CSV export dumps (Chunk 13).
        // Without it the channel is drained and dropped (zero extra cost).
        let capture_events = cfg.export.is_some();
        let mut results = Vec::with_capacity(iterations);
        let mut packets: Vec<PacketSample> = Vec::new();
        for i in 0..iterations {
            if !cfg.json && iterations > 1 {
                term.progress(&format!("  [{}/{}] Running...", i + 1, iterations));
            }
            let result = if capture_events {
                let (result, events) = engine.run_iteration_events(&prompt).await;
                packets.extend(export::samples_from_events(&events, (i + 1) as u64));
                result
            } else {
                engine.run_iteration(&prompt).await
            };
            if cfg.verbose && !cfg.json {
                term.dim(&format!(
                    "  chunks={} content={} reasoning={}",
                    result.total_chunks, result.content_chunks, result.reasoning_chunks
                ));
            }
            results.push(result);
        }
        if !cfg.json && iterations > 1 {
            term.clear_progress();
        }

        if cfg.json {
            println!("{v}", v = json_report(cfg, &results));
        } else {
            for (i, r) in results.iter().enumerate() {
                match format_result_box(r, i + 1, iterations, term.color) {
                    Some(box_) => print!("{box_}"),
                    None => term.error(&format!(
                        "Test {}/{} FAILED: {}",
                        i + 1,
                        iterations,
                        r.error.as_deref().unwrap_or("unknown error")
                    )),
                }
                if let Some(note) = &r.error {
                    if r.completion_tokens > 0 {
                        term.warning(&format!("Note: {note}"));
                    }
                }
            }
            if let Some(summary) = format_summary(&results, term.color) {
                print!("{summary}");
            }
        }

        // Persist the completed run (Chunk 12): one `benchmark_sessions`
        // row + one `stream_metrics` row per iteration, in the platform
        // data dir. A storage failure degrades gracefully — it must never
        // break (or change the exit code of) a benchmark run.
        let session = BenchmarkSession {
            session_id: uuid::Uuid::new_v4().to_string(),
            timestamp: None, // SQLite `DEFAULT CURRENT_TIMESTAMP` fills it
            target_url: cfg.url.clone(),
            model_name: cfg.model.clone(),
            backend_type: None,
            quantization: None,
            system_gpu: None,
            total_duration_sec: Some(results.iter().map(|r| r.stream_time).sum()),
        };
        // Single-stream headless engine: every iteration runs at
        // concurrency level 1.
        let rows: Vec<StreamMetricRow> = results
            .iter()
            .map(|r| StreamMetricRow::from_speed_result(r, &session.session_id, 1))
            .collect();
        match Database::open_default() {
            Ok(mut db) => {
                match db.persist_run(&session, &rows) {
                    Ok(()) => {
                        let id = &session.session_id;
                        term.dim(&format!(
                            "  saved → {} (session {})",
                            db.path().display(),
                            &id[..8]
                        ))
                    }
                    Err(e) => term.warning(&format!("persist failed: {e}")),
                }
                // `--export` (Chunk 13): consume the run back from the
                // SQLite storage layer and write the requested format.
                if let Some(fmt) = cfg.export {
                    run_export(
                        &db,
                        &session,
                        &packets,
                        fmt,
                        cfg.export_path.as_deref(),
                        &term,
                    );
                }
            }
            Err(e) => term.warning(&format!("persist failed: {e}")),
        }

        // Prototype exit-code rule: all runs failed → exit 1.
        if all_failed(&results) {
            if !cfg.json {
                term.error("All test runs failed.");
            }
            return 1;
        }
        0
    })
}

/// Chunk 13: write the `--export` file for a just-persisted run,
/// consuming the data back from the SQLite storage layer (session row +
/// `stream_metrics` rows), plus the live per-packet samples for the CSV.
///
/// Destination: `--export-path`, or
/// `data_dir()/exports/crucible-<session-id[:8]>.<ext>` by default.
/// Degrades gracefully — an export failure never breaks (or changes the
/// exit code of) a benchmark run.
fn run_export(
    db: &Database,
    session: &BenchmarkSession,
    packets: &[PacketSample],
    format: ExportFormat,
    path: Option<&std::path::Path>,
    term: &Term,
) {
    let mut payload = match ExportPayload::from_stored(db, &session.session_id) {
        Ok(p) => p,
        Err(e) => {
            term.warning(&format!("export failed: {e}"));
            return;
        }
    };
    payload.packets = packets.to_vec();
    let dest = path
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| export::default_path(&session.session_id, format));
    match export::write(&dest, format, &payload) {
        Ok(p) => term.dim(&format!("  exported → {}", p.display())),
        Err(e) => term.warning(&format!("export failed: {e}")),
    }
}

/// Run the TUI on a dedicated single-thread runtime, guaranteeing the
/// terminal is restored even on panic.
///
/// The event loop is async (60Hz tokio tick) but the app itself is a
/// foreground terminal program, so a current-thread runtime is all it
/// needs; the worker pool will run on its own multi-thread runtime in
/// later chunks.
fn run_tui(cfg: &Config) -> bool {
    let export_format = cfg.export.unwrap_or_default();
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        runtime.block_on(async {
            let mut event_loop = EventLoop::new()?;
            let mut app = App::new()
                .with_export_format(export_format)
                .with_niah_config(cfg);
            event_loop.run(&mut app).await?;
            event_loop.teardown()
        })
    }));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            eprintln!("crucible-llm: TUI error: {e}");
            false
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "unknown panic".to_string());
            eprintln!("crucible-llm: TUI panicked: {msg}");
            false
        }
    }
}

fn print_banner() {
    let version = env!("CARGO_PKG_VERSION");
    println!("============================================================");
    println!("  Crucible-LLM v{version}");
    println!("  High-performance terminal LLM benchmark & inference profiler");
    println!("============================================================");
    println!();
    println!("  modes:");
    println!("    --url <endpoint> [flags]   headless single-stream benchmark");
    println!("                               (result box, --json, --iterations …)");
    println!("    --tui                      open the interactive ratatui dashboard");
    println!();
    println!("  run `crucible-llm --help` for all flags (a superset of");
    println!("  llmspeedtest.py / llmspeedtest2.py).");
}

/// TTY-aware output handler (parity with the prototype's `Output`):
/// `ℹ`/`⚠`/`✗` markers on stderr, bold lines on stdout, live progress
/// only on a TTY.
struct Term {
    color: bool,
    tty: bool,
}

impl Term {
    fn info(&self, msg: &str) {
        self.emit("36", "ℹ ", msg);
    }

    fn warning(&self, msg: &str) {
        self.emit("33", "⚠ ", msg);
    }

    fn error(&self, msg: &str) {
        self.emit("31", "✗ ", msg);
    }

    fn dim(&self, msg: &str) {
        self.emit("2", "", msg);
    }

    /// Live progress (stderr, overwrites the line; skipped off-TTY).
    fn progress(&self, msg: &str) {
        if self.tty {
            let s = if self.color {
                format!("\u{1b}[2m{msg}\u{1b}[0m")
            } else {
                msg.to_string()
            };
            eprintln!("\r\u{1b}[K{s}");
        }
    }

    fn clear_progress(&self) {
        if self.tty {
            eprint!("\r\u{1b}[K");
        }
    }

    fn emit(&self, code: &str, prefix: &str, msg: &str) {
        if self.color {
            eprintln!("\u{1b}[{code}m{prefix}{msg}\u{1b}[0m");
        } else {
            eprintln!("{prefix}{msg}");
        }
    }
}
