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
//!
//! CF-42: the headless orchestration itself lives in the library
//! ([`crucible_llm::run_headless`]); this binary is a thin dispatch + a
//! thin TUI wrapper.

use std::io::IsTerminal;
use std::panic;
use std::process;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crucible_llm::config::{Config, ConfigError, DEFAULT_MODEL};
use crucible_llm::hw::{
    detect_gpu, gpu_display_name, CpuPowerBackend, GpuBackend, HwPoller, HW_POLL_INTERVAL_MS,
};
use crucible_llm::log::{Context, RunLogger};
// CF-42: the headless orchestration is a public lib function — `main` is a
// thin caller.
use crucible_llm::run_headless;
use crucible_llm::ui::app::App;
use crucible_llm::ui::event::EventLoop;
use crucible_llm::ui::theme::Theme;

fn main() {
    let mut cfg = match Config::from_cli() {
        Ok(cfg) => cfg,
        // `--help` / `--version` / bad flags: clap prints the right text to
        // the right stream and exits with its own code (0 for help).
        Err(ConfigError::Clap(e)) => e.exit(),
        Err(e) => {
            eprintln!("crucible-llm: {e}");
            process::exit(2);
        }
    };

    // R2 (traceability): the session UUID is generated **up front**, before
    // the logger is created, so the run-log (header + archive file name)
    // and the persisted `benchmark_sessions` row share the *same* id — a
    // deterministic log ↔ DB join. The same id is threaded into both the
    // headless and TUI run paths.
    let session_id = uuid::Uuid::new_v4().to_string();

    // The file-based run log: `latest.log` (overwritten each run) + a
    // `run-<timestamp>-<session>.log` archive in the log dir (default
    // `~/.local/share/crucible/logs/`, configurable via `--log-dir` /
    // `CRUCIBLE_LOG_DIR` / the config file). Created before dispatch so
    // both the TUI and headless paths — and every engine they spawn —
    // share the same run log. A logger that cannot create its files
    // degrades to a silent no-op (logging never breaks a run).
    let logger = RunLogger::start_with_session(cfg.log_dir.as_deref(), Some(&session_id));

    // `--banner` (hidden, Chunk 20): the old bare-invocation banner path,
    // demoted to an explicit flag. Checked before the (unnecessary for a
    // banner) GPU probe.
    if cfg.banner {
        print_banner();
        return;
    }

    // GPU auto-detection (early): NVIDIA → AMD → Intel. The single result
    // is shared with the App (the GPU panel + the "D: Energy" label) and
    // Engine D (the telemetry). Logged once so the run record shows what
    // hardware was found. `None` (no GPU) is the graceful path — the app
    // runs with CPU/RAM telemetry only, never a panic.
    let gpu: Option<Arc<dyn GpuBackend>> = detect_gpu().map(Arc::from);
    match &gpu {
        Some(g) => {
            let name = gpu_display_name(g.as_ref());
            eprintln!("[HW] Detected: {name}");
            logger.info(Context::Setup, format!("[HW] Detected: {name}"));
        }
        None => {
            eprintln!("[HW] No GPU telemetry available");
            logger.info(
                Context::Setup,
                "[HW] No GPU telemetry available".to_string(),
            );
        }
    }

    // CPU power detection (early, alongside the GPU): the cascade is
    // RAPL → Super I/O → a clearly-labeled fixed estimate. Logged once so
    // the run record shows which source drives the CPU draw (the system
    // power / energy / `$/1M` cost math now includes it — GPU alone
    // underreports the real draw). Never `None`: the cascade ends in the
    // estimate.
    let cpu_power: Option<Box<CpuPowerBackend>> = CpuPowerBackend::try_init();
    match &cpu_power {
        Some(c) => {
            let name = c.method_name();
            eprintln!("[HW] CPU power: {name}");
            logger.info(Context::Setup, format!("[HW] CPU power: {name}"));
        }
        None => {
            eprintln!("[HW] No CPU power telemetry available");
            logger.info(
                Context::Setup,
                "[HW] No CPU power telemetry available".to_string(),
            );
        }
    }

    // CF-39: unify the two independent hardware booleans (the poller
    // toggle `cfg.hardware` and the Engine D *selection*
    // `cfg.engines.hardware`) into one derived rule so they can never
    // disagree. The energy engine runs only when the poller is enabled
    // **and** a GPU is present — so `--no-hardware` (poller off) and a
    // driverless host (no GPU) both turn Engine D off. An explicit
    // `--engine` selection wins over detection (the operator named Engine
    // D on purpose, so we respect it). See [`resolve_engine_d`].
    cfg.engines.hardware = resolve_engine_d(
        cfg.hardware,
        gpu.is_some(),
        cfg.engines_explicit,
        cfg.engines.hardware,
    );

    // Chunk 20 — the TUI is the default mode. The run goes headless when
    // an explicit headless marker is present (`--headless`, `--json`
    // which implies it, or `--matrix-report` which prints the
    // `[COMMUNITY-SUBMISSION]` block on stdout after the run)…
    let headless = cfg.headless || cfg.json || cfg.matrix_report;
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
        let code = run_headless(&cfg, logger.clone(), gpu, cpu_power, session_id.clone());
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
    let ok = run_tui(&cfg, logger.clone(), gpu, cpu_power, session_id.clone());
    EventLoop::restore(); // best-effort terminal restore on all paths
    logger.info(Context::Tui, format!("run ended — TUI — ok={ok}"));
    logger.finish(); // flush before exit (drops are skipped by process::exit)
    if !ok {
        process::exit(1);
    }
}

/// CF-39: the unified Engine D (energy) selection rule. The energy engine
/// runs only when the hardware telemetry **poller** is enabled **and** a
/// GPU is present — so the two booleans (`Config.hardware` and
/// `EngineSelection.hardware`) can never disagree:
/// * `--no-hardware` (poller off) → Engine D off (no starved N/A);
/// * a driverless host (no GPU) → Engine D off (the README's "no GPU →
///   off" rule — a remote user gets N/A anyway).
#[must_use]
fn unified_hardware(hardware: bool, gpu_present: bool) -> bool {
    hardware && gpu_present
}

/// CF-39: resolve the Engine D (energy) selection for a run, respecting an
/// explicit `--engine` choice.
///
/// When the operator named their engines explicitly (`engines_explicit` —
/// CLI `--engine` / env / config file), the selection is left exactly as
/// they set it (`explicit_engine_d`): we respect what they asked for, even
/// if it names Engine D on a host where the poller would turn it off.
///
/// Otherwise it is derived as [`unified_hardware`] (the poller toggle AND
/// GPU presence), so `--no-hardware` and a driverless host both turn
/// Engine D off — closing the two-boolean ambiguity (DO-10/CF-39) and the
/// headless/TUI `--no-hardware` asymmetry (DO-08/CF-20).
#[must_use]
fn resolve_engine_d(
    hardware: bool,
    gpu_present: bool,
    engines_explicit: bool,
    explicit_engine_d: bool,
) -> bool {
    if engines_explicit {
        explicit_engine_d
    } else {
        unified_hardware(hardware, gpu_present)
    }
}

/// Run the TUI on a dedicated single-thread runtime, guaranteeing the
/// terminal is restored even on panic.
///
/// The event loop is async (60Hz tokio tick) but the app itself is a
/// foreground terminal program, so a current-thread runtime is all it
/// needs; the worker pool will run on its own multi-thread runtime in
/// later chunks.
fn run_tui(
    cfg: &Config,
    logger: Arc<RunLogger>,
    gpu: Option<Arc<dyn GpuBackend>>,
    cpu_power: Option<Box<CpuPowerBackend>>,
    session_id: String,
) -> bool {
    let export_format = cfg.export.unwrap_or_default();
    // Chunk 17: the detected backend (any vendor) + the CPU power backend +
    // sysinfo drive the 100 ms poller. Never fails — a driverless host
    // simply runs with N/A GPU fields.
    let hw = Arc::new(Mutex::new(
        HwPoller::with_backend(gpu.clone())
            .with_cpu_power(cpu_power)
            .with_rate(cfg.rate_per_kwh),
    ));
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .enable_io() // the TUI's model discovery (reqwest HTTP) needs the IO driver
            .build()
            .expect("failed to build tokio runtime");
        runtime.block_on(async {
            let mut event_loop = EventLoop::new()?;
            // Chunk 18: the App is seeded with the full resolved config —
            // the editable Configuration form (View 6) and the engine
            // selection all draw from the same `Config` the headless and
            // export paths use.
            let mut app = App::new()
                .with_export_format(export_format)
                .with_config(cfg)
                .with_theme(Theme::from_id(&cfg.theme).unwrap_or_default())
                .with_logger(logger.clone())
                .with_session_id(session_id)
                .with_gpu(gpu);
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
            // The 100 ms hardware telemetry (blueprint §4.3): it polls and
            // merges into the `ArcSwap<MetricsSnapshot>` the views read
            // lock-free. It never touches the stream workers' quanta
            // timing path (measurement isolation, blueprint §4); a lock
            // failure is a no-op. Chunk 18: only spawned (and only
            // attached to the App) when the hardware/energy telemetry is
            // enabled — `app.hw.is_some()` then means "the poller is
            // live", which the views use to decide whether an Energy line
            // belongs on screen.
            if cfg.hardware {
                app = app.with_hw(hw.clone());
                let state = app.metrics.clone();
                let poller = hw;
                // CF-40: the 100 ms telemetry poller runs on a **dedicated
                // OS thread**, not a `tokio::spawn` task on the
                // `current_thread` runtime that also drives the 60 Hz render
                // loop. A task on that single thread only ticks when the
                // render loop yields, so the cadence was best-effort and the
                // power trace (and the View 4 energy integral) slipped under
                // load. A dedicated thread decouples it — the "four
                // decoupled rings" the blueprint promises — without stealing
                // time from the measured streams.
                std::thread::spawn(move || {
                    loop {
                        std::thread::sleep(Duration::from_millis(HW_POLL_INTERVAL_MS));
                        // FIX 2: the run is over (the metrics pipeline
                        // is frozen) → the poller stops: no hardware
                        // reads, no snapshot updates. The thread idles
                        // (one atomic load per tick) until a new run
                        // re-arms the pipeline (`unfreeze()`).
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

#[cfg(test)]
mod tests {
    use super::*;

    // CF-39: the poller toggle AND GPU presence derive the Engine D
    // selection, so the two booleans can never disagree.

    #[test]
    fn no_gpu_turns_engine_d_off_even_if_the_poller_is_on() {
        assert!(!unified_hardware(true, false), "no GPU → Engine D off");
        assert!(
            unified_hardware(true, true),
            "poller on + GPU → Engine D on"
        );
    }

    #[test]
    fn no_hardware_turns_engine_d_off_even_if_a_gpu_is_present() {
        assert!(
            !unified_hardware(false, true),
            "`--no-hardware` → Engine D off"
        );
        assert!(!unified_hardware(false, false), "both off → Engine D off");
    }

    #[test]
    fn a_detection_based_run_derives_engine_d_from_poller_and_gpu() {
        // Not explicit → derive. GPU present + poller on → on.
        assert!(resolve_engine_d(true, true, false, false));
        // Not explicit → derive. No GPU → off (the README "no GPU → off"),
        // even if a stale selection said on.
        assert!(!resolve_engine_d(true, false, false, true));
        // Not explicit → derive. `--no-hardware` (poller off) → off, even
        // with a GPU (the starved-N/A case CF-39 closes).
        assert!(!resolve_engine_d(false, true, false, true));
    }

    #[test]
    fn an_explicit_engine_selection_is_respected_not_overridden() {
        // `--engine hardware` (explicit) on a no-GPU, no-poller host: we
        // respect the operator's explicit choice (Engine D stays on — it
        // will simply report N/A without a poller).
        assert!(resolve_engine_d(false, false, true, true));
        // `--engine niah` (explicit, omits D): Engine D stays off even with
        // a GPU + poller on.
        assert!(!resolve_engine_d(true, true, true, false));
    }
}
