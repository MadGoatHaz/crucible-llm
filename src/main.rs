//! Crucible-LLM entry point.
//!
//! Dispatch (Chunk 20 — **the TUI is the default mode**):
//! - `crucible-llm` → the ratatui dashboard (Chunk 8). A bare invocation
//!   launches the TUI — it no longer prints the banner.
//! - `crucible-llm --headless …` / `--json …` → **headless** single-stream
//!   benchmark (Engine A): N iterations, the prototype's result box
//!   and/or `--json` (same field set as `llmspeedtest.py`), a
//!   multi-iteration summary, and exit 1 if all runs failed.
//!   (`--json` implies headless; `--headless` is the explicit marker.)
//! - `crucible-llm --tui …` → forces the TUI (compat: e.g. on a non-TTY,
//!   where the run would otherwise fall back to headless).
//! - `crucible-llm --banner` → the old startup banner, demoted to a
//!   hidden flag.
//!
//! A non-TTY stdout (pipe / CI) also falls back to the headless run
//! unless `--tui` is given explicitly.

use std::io::IsTerminal;
use std::panic;
use std::process;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crucible_llm::client::StreamEvent;
use crucible_llm::config::{Config, ConfigError, ExportFormat, DEFAULT_MODEL};
use crucible_llm::engines::hardware::{joules_per_token, profile};
use crucible_llm::engines::speed::{
    all_failed, format_result_box, format_summary, json_report, SpeedEngine, MAX_GEN_TOKENS,
};
use crucible_llm::engines::{
    build_sweep, summarize_sweep, NiahEngine, ReasoningEngine, StructuredEngine,
};
use crucible_llm::hw::{HwPoller, HW_POLL_INTERVAL_MS};
use crucible_llm::log::{Context, RunLogger};
use crucible_llm::storage::export::{self, ExportPayload, PacketSample};
use crucible_llm::storage::{BenchmarkSession, Database, StreamMetricRow};
use crucible_llm::timing::MonotonicInstant;
use crucible_llm::ui::app::App;
use crucible_llm::ui::event::EventLoop;
use crucible_llm::ui::theme::Theme;

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

    // The file-based run log: `latest.log` (overwritten each run) + a
    // `run-<timestamp>.log` archive in the log dir (default
    // `~/.local/share/crucible/logs/`, configurable via `--log-dir` /
    // `CRUCIBLE_LOG_DIR` / the config file). Created before dispatch so
    // both the TUI and headless paths — and every engine they spawn —
    // share the same run log. A logger that cannot create its files
    // degrades to a silent no-op (logging never breaks a run).
    let logger = RunLogger::start(cfg.log_dir.as_deref());

    // `--banner` (hidden, Chunk 20): the old bare-invocation banner path,
    // demoted to an explicit flag.
    if cfg.banner {
        print_banner();
        return;
    }

    // Chunk 20 — the TUI is the default mode. The run goes headless when
    // an explicit headless marker is present (`--headless`, or `--json`
    // which implies it)…
    let headless = cfg.headless || cfg.json;
    // …or when stdout is not a TTY (pipe / CI) and the user did not
    // explicitly ask for the dashboard with `--tui`. A TUI on a pipe
    // cannot render; the headless run is the sane fallback.
    let tty = std::io::stdout().is_terminal();
    if headless || (!tty && !cfg.tui) {
        if !headless {
            eprintln!("crucible-llm: stdout is not a TTY — running headless (use --tui to force the dashboard)");
        }
        logger.info(
            Context::Setup,
            format!(
                "run started — headless — url={} model={} mode={} iterations={} timeout={}s",
                cfg.url,
                cfg.model,
                cfg.mode.label(),
                cfg.iterations,
                cfg.timeout
            ),
        );
        let code = run_headless(&cfg, logger.clone());
        logger.info(
            Context::Setup,
            format!("run ended — headless — exit code {code}"),
        );
        logger.finish(); // flush before `process::exit` skips drops
        process::exit(code);
    }

    // Everything else — including a bare `crucible-llm` — launches the
    // TUI (Chunk 8).
    logger.info(
        Context::Tui,
        format!(
            "run started — TUI — url={} model={} (log: {})",
            cfg.url,
            cfg.model,
            logger
                .path()
                .map(|p| p.display().to_string())
                .unwrap_or_default()
        ),
    );
    let ok = run_tui(&cfg, logger.clone());
    EventLoop::restore(); // best-effort terminal restore on all paths
    logger.info(Context::Tui, format!("run ended — TUI — ok={ok}"));
    logger.finish(); // flush before exit (drops are skipped by process::exit)
    if !ok {
        process::exit(1);
    }
}

/// The headless (non-TUI) path: run N single-stream iterations (Engine A)
/// and print the prototype's result box(es) / summary — or `--json` on
/// stdout. Returns the process exit code (1 iff every run failed).
fn run_headless(cfg: &Config, logger: Arc<RunLogger>) -> i32 {
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
    // Chunk 17: the 100 ms hardware sampler (blueprint §4.3) runs for the
    // whole headless run; its power trace is sliced per-iteration into the
    // Silicon Efficiency Metric (Joules/Token). Never fails — a driverless
    // host simply yields N/A and the run proceeds normally.
    let hw = Arc::new(Mutex::new(HwPoller::new()));
    rt.spawn({
        let hw = hw.clone();
        async move {
            let mut interval = tokio::time::interval(Duration::from_millis(HW_POLL_INTERVAL_MS));
            loop {
                interval.tick().await;
                if let Ok(mut p) = hw.lock() {
                    p.poll();
                }
            }
        }
    });

    rt.block_on(async {
        let engine = match SpeedEngine::new(cfg) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("crucible-llm: {e}");
                logger.error(Context::EngineA, format!("Engine A init failed: {e}"));
                return 1;
            }
        };
        let engine = engine.logger(logger.clone());
        logger.info(
            Context::EngineA,
            format!(
                "Engine A (Speed) started — {} iteration(s), {} max tokens",
                cfg.iterations, MAX_GEN_TOKENS
            ),
        );

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

        // Keep each iteration's raw worker events (Chunk 13 CSV export +
        // Chunk 17 energy windows): the per-packet arrival timestamps the
        // CSV dumps, and the (T0, Tn) lifecycle window that slices the
        // hardware power trace into per-iteration Joules/Token.
        let capture_packets = cfg.export.is_some();
        let mut results = Vec::with_capacity(iterations);
        let mut packets: Vec<PacketSample> = Vec::new();
        let mut windows: Vec<Option<(MonotonicInstant, MonotonicInstant)>> =
            Vec::with_capacity(iterations);
        for i in 0..iterations {
            if !cfg.json && iterations > 1 {
                term.progress(&format!("  [{}/{}] Running...", i + 1, iterations));
            }
            let (result, events) = engine.run_iteration_events(&prompt).await;
            if capture_packets {
                packets.extend(export::samples_from_events(&events, (i + 1) as u64));
            }
            windows.push(run_window(&events));
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
        logger.info(
            Context::EngineA,
            format!(
                "Engine A complete — {}/{} run(s) succeeded",
                results.iter().filter(|r| !r.is_failed()).count(),
                results.len()
            ),
        );

        // Chunk 17: harvest the sampler's power trace (brief lock on the
        // run-completion path — never on the stream workers' timing path).
        let hw_trace = hw
            .lock()
            .ok()
            .map(|g| g.trace().to_vec())
            .unwrap_or_default();
        let gpu_name = hw
            .lock()
            .ok()
            .and_then(|g| g.gpu_name().map(str::to_string));

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
            // Chunk 17: run-level energy summary (blueprint §5D) — shown
            // only when power telemetry was present (a driverless host
            // prints nothing: the N/A rule, never a spurious 0.0).
            let energy = profile(
                &hw_trace,
                None,
                results.iter().map(|r| r.completion_tokens).sum(),
            );
            if let Some(jpt) = energy.joules_per_token {
                let peak = energy
                    .peak_power_w
                    .map(|w| format!(" · peak {w:.0} W"))
                    .unwrap_or_default();
                term.dim(&format!(
                    "  Energy: {:.1} J ({:.3} J/token{peak})",
                    energy.joules, jpt
                ));
            }
            if let Some(w) = &energy.fragmentation_warning {
                term.warning(&format!("VRAM: {w}"));
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
            system_gpu: gpu_name,
            total_duration_sec: Some(results.iter().map(|r| r.stream_time).sum()),
        };
        // Single-stream headless engine: every iteration runs at
        // concurrency level 1. Chunk 17: each row carries the
        // Silicon Efficiency Metric for that iteration's (T0, Tn) window
        // (N/A when no power telemetry was available).
        let rows: Vec<StreamMetricRow> = results
            .iter()
            .enumerate()
            .map(|(i, r)| {
                let mut row = StreamMetricRow::from_speed_result(r, &session.session_id, 1);
                row.joules_per_token = windows[i]
                    .and_then(|w| joules_per_token(&hw_trace, Some(w), r.completion_tokens));
                row
            })
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
                    // v0.1.1: the JSON export's `loop_guard` block comes
                    // from the runs the decode-loop guard excluded.
                    let loop_guard = {
                        let detected = results.iter().filter(|r| r.looping).count();
                        let excluded: u64 = results
                            .iter()
                            .filter(|r| r.looping)
                            .map(|r| r.completion_tokens)
                            .sum();
                        (detected > 0).then_some(crucible_llm::metrics::LoopGuardSummary {
                            detected_streams: detected,
                            excluded_tokens: excluded,
                        })
                    };
                    run_export(
                        &db,
                        &session,
                        &packets,
                        fmt,
                        cfg.export_path.as_deref(),
                        &term,
                        loop_guard,
                    );
                }
            }
            Err(e) => term.warning(&format!("persist failed: {e}")),
        }

        // Chunk 18: run any *additionally-selected* engines (B / C1 / C2 /
        // C3) and report a one-line summary for each. Engine A (speed) is
        // the run above; Engine D (hardware) is the continuous poller
        // started earlier. A selection of only `speed` (the default) skips
        // this block entirely — headless behavior is unchanged.
        let extra = cfg.engines;
        if extra.concurrency || extra.niah || extra.reasoning || extra.structured {
            if !cfg.json {
                term.dim(&format!(
                    "  running selected engines: {}",
                    extra.iter_labels().collect::<Vec<_>>().join(", ")
                ));
            }
            if extra.concurrency {
                if let Some(sweep) = build_sweep(cfg, None, Some(logger.clone())) {
                    if !cfg.json {
                        term.dim(&format!(
                            "  [B] concurrency sweep [{}]…",
                            cfg.ladder
                                .iter()
                                .map(|n| n.to_string())
                                .collect::<Vec<_>>()
                                .join(",")
                        ));
                    }
                    let r = sweep.run().await;
                    // FIX 3: the headless summary leads with the
                    // practical (per-stream usability) sweet spot, with
                    // the pure-throughput knee as reference.
                    if r.is_empty() {
                        term.warning("  [B] sweep produced no usable levels");
                    } else {
                        term.info(&format!("  [B] {}", summarize_sweep(&r)));
                    }
                }
            }
            if extra.niah {
                match NiahEngine::new(cfg) {
                    Ok(engine) => {
                        if !cfg.json {
                            term.dim("  [C1] NIAH matrix (7 sizes × 11 depths)…");
                        }
                        let r = engine.logger(logger.clone()).run().await;
                        term.info(&format!("  [C1] {}", r.accuracy_label()));
                    }
                    Err(e) => {
                        term.warning(&format!("[C1] niah init failed: {e}"));
                        logger.error(Context::EngineC1, format!("niah init failed: {e}"));
                    }
                }
            }
            if extra.reasoning {
                match ReasoningEngine::new(cfg) {
                    Ok(engine) => {
                        if !cfg.json {
                            term.dim("  [C2] reasoning bank (13 challenges)…");
                        }
                        let r = engine.logger(logger.clone()).run().await;
                        term.info(&format!(
                            "  [C2] {} · avg {:.1} t/s",
                            r.score.label(),
                            r.avg_tg_speed()
                        ));
                    }
                    Err(e) => {
                        term.warning(&format!("[C2] reasoning init failed: {e}"));
                        logger.error(Context::EngineC2, format!("reasoning init failed: {e}"));
                    }
                }
            }
            if extra.structured {
                match StructuredEngine::new(cfg) {
                    Ok(engine) => {
                        if !cfg.json {
                            term.dim("  [C3] structured output (free-form vs constrained)…");
                        }
                        let r = engine.logger(logger.clone()).run().await;
                        term.info(&format!("  [C3] {}", r.summary_line()));
                    }
                    Err(e) => {
                        term.warning(&format!("[C3] structured init failed: {e}"));
                        logger.error(Context::EngineC3, format!("structured init failed: {e}"));
                    }
                }
            }
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

/// Chunk 17: the (T0, Tn) lifecycle window of one iteration, extracted
/// from its worker events — the slice of the hardware power trace over
/// which the Silicon Efficiency Metric integrates. `None` when the run
/// never recorded a full window (e.g. it failed before both milestones).
fn run_window(events: &[StreamEvent]) -> Option<(MonotonicInstant, MonotonicInstant)> {
    let t0 = events.iter().find_map(|e| match e {
        StreamEvent::Frame { timestamps, .. } => timestamps.t0,
        _ => None,
    });
    let t_end = events.iter().rev().find_map(|e| match e {
        StreamEvent::Complete { timestamps, .. } | StreamEvent::Failed { timestamps, .. } => {
            timestamps.t_end
        }
        _ => None,
    });
    match (t0, t_end) {
        (Some(start), Some(end)) => Some((start, end)),
        _ => None,
    }
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
    loop_guard: Option<crucible_llm::metrics::LoopGuardSummary>,
) {
    let mut payload = match ExportPayload::from_stored(db, &session.session_id) {
        Ok(p) => p,
        Err(e) => {
            term.warning(&format!("export failed: {e}"));
            return;
        }
    };
    payload.packets = packets.to_vec();
    if let Some(lg) = loop_guard {
        payload = payload.with_loop_guard(lg);
    }
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
fn run_tui(cfg: &Config, logger: Arc<RunLogger>) -> bool {
    let export_format = cfg.export.unwrap_or_default();
    // Chunk 17: probe the hardware once (NVML feature-gated + sysinfo).
    // Never fails — a driverless host simply runs with N/A GPU fields.
    let hw = Arc::new(Mutex::new(HwPoller::new()));
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .enable_io() // the TUI's model discovery (reqwest HTTP) needs the IO driver
            .build()
            .expect("failed to build tokio runtime");
        runtime.block_on(async {
            let mut event_loop = EventLoop::new()?;
            // Chunk 18: the App is seeded with the full resolved config —
            // the editable Configuration form (View 5) and the engine
            // selection all draw from the same `Config` the headless and
            // export paths use.
            let mut app = App::new()
                .with_export_format(export_format)
                .with_config(cfg)
                .with_theme(Theme::from_id(&cfg.theme).unwrap_or_default())
                .with_logger(logger.clone());
            // The target is "fully given" when an explicit URL + a
            // non-placeholder model are present; otherwise the Setup
            // takeover is needed to fill them in (the placeholder model
            // name "default" counts as absent).
            let needs_setup = !(cfg.target_explicit && cfg.model != DEFAULT_MODEL);
            // **First run** (no theme ever chosen — `theme_explicit` is
            // false): the full-screen theme picker comes *before* Setup.
            // A subsequent run (a theme is saved in the config file) skips
            // the picker and goes straight to Setup / the Dashboard.
            if !cfg.theme_explicit {
                app = app.with_theme_picker(needs_setup);
            } else if needs_setup {
                app = app.with_setup(cfg);
            }
            // The 100 ms hardware telemetry task (blueprint §4.3): it
            // polls and merges into the `ArcSwap<MetricsSnapshot>` the
            // views read lock-free. It never touches the stream
            // workers' quanta timing path (measurement isolation,
            // blueprint §4); a lock failure is a no-op. Chunk 18: only
            // spawned (and only attached to the App) when the
            // hardware/energy telemetry is enabled — `app.hw.is_some()`
            // then means "the poller is live", which the views use to
            // decide whether an Energy line belongs on screen.
            if cfg.hardware {
                app = app.with_hw(hw.clone());
                let state = app.metrics.clone();
                let poller = hw;
                tokio::spawn(async move {
                    let mut interval =
                        tokio::time::interval(Duration::from_millis(HW_POLL_INTERVAL_MS));
                    loop {
                        interval.tick().await;
                        // FIX 2: the run is over (the metrics pipeline
                        // is frozen) → the poller stops: no hardware
                        // reads, no snapshot updates. The task goes
                        // idle (one atomic load per tick) until a new
                        // run re-arms the pipeline (`unfreeze()`).
                        if state.is_frozen() {
                            continue;
                        }
                        if let Ok(mut p) = poller.lock() {
                            p.tick(&state);
                        }
                    }
                });
            }
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
    println!("  modes (Chunk 20: the TUI is the default):");
    println!("    (no mode flags)              open the interactive ratatui dashboard");
    println!("    --headless [flags]           classic one-shot headless benchmark");
    println!("                               (result box, --iterations …)");
    println!("    --json [flags]               headless benchmark, JSON on stdout");
    println!("    --banner                     this banner");
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
