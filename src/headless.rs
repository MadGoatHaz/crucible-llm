//! The headless (non-TUI) benchmark path (plan Chunk 7 / Chunk 17).
//!
//! [`run_headless`] is the single-stream orchestration the `--headless` /
//! `--json` dispatch uses: it runs Engine A (N iterations), any
//! additionally-selected engines (B / C1 / C2 / C3 / F), harvests the 100 ms
//! hardware power trace into the Silicon Efficiency Metric (Joules/Token),
//! persists the run to the platform database, and writes the `--export` file
//! and/or the `--json` document (stdout). It is a **public lib function**
//! ([`crate::run_headless`]) so it is testable and reusable — `main.rs` is a
//! thin caller (CF-42).
//!
//! **CF-40** — the 100 ms hardware sampler runs on a **dedicated OS thread**,
//! decoupled from the benchmark workload (which owns the `current_thread`
//! tokio runtime). A `tokio::spawn` task on that single runtime thread only
//! ticks when the measured work yields, so its cadence was best-effort and
//! slipped under load, degrading the energy integral. A dedicated thread gives
//! the poller its own scheduling — the "decoupled ring" the blueprint
//! promises — without stealing time from the measured streams.
//!
//! **P3-A / CF-53** — the `--json` energy object carries the
//! `cpu_power_estimated` flag ([`build_energy_json`]) so a CI/CD consumer can
//! tell a measured power reading from the labeled fixed-estimate fallback.

use std::io::{IsTerminal, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::client::StreamEvent;
use crate::config::{Config, ExportFormat};
use crate::engines::hardware::{joules_per_token, profile, EnergyResult};
use crate::engines::sequence::{
    hardware_summary, speed_mean_is_blended, speed_mean_tps, sum_all_engines_tokens,
    summarize_speed,
};
use crate::engines::speed::{
    all_failed, format_split_report, format_summary, iteration_complete_line, json_report_split,
    perceived_ttft_seconds, PrefillMetric, SpeedEngine, SpeedResult, MAX_GEN_TOKENS,
};
use crate::engines::{
    build_sweep, context_size_label, flatout_streams_for, summarize_sweep, Engine, FlatOutEngine,
    FlatOutResult, NiahEngine, NiahResult, ReasoningEngine, ReasoningResult, StreamCountSource,
    StructuredEngine, StructuredResult, SweepResult,
};
use crate::hw::{
    arm_power_monitor, CpuPowerBackend, GpuBackend, HwPoller, IDLE_WINDOW_SECS,
    HW_POLL_INTERVAL_MS,
};
use crate::log::{read_log_file, Context, RunLogger};
use crate::report::MatrixReport;
use crate::storage::export::{self, ExportPayload, PacketSample};
use crate::storage::{
    BenchmarkSession, ConcurrencySummary, Database, EnergySummary, EngineResults, FlatOutSummary,
    ReasoningSummary, StreamMetricRow, StructuredSummary,
};
use crate::timing::MonotonicInstant;

/// Build the `--json` `energy` object from an [`EnergyResult`].
///
/// Serializes the [`EnergySummary`] (the raw `EnergyResult` is not
/// `Serialize` — it carries non-`Clone`-friendly trace internals), which
/// carries the **P3-A / I4 estimated-power flag**: `cpu_power_estimated` is
/// `true` when the CPU component of the system draw is (or included) the
/// labeled fixed-estimate fallback — no RAPL / Super-I-O sensor on the
/// host, *or* a measured backend that fell back mid-run (the sample-level
/// latch). A `--json` consumer (CI/CD regression gating) can tell a purely
/// measured power reading from one that includes the fabricated 40 W CPU
/// estimate — the J/token and `$/1M` derived from it inherit that
/// uncertainty. The flag now lives on the summary itself (I4), so the
/// `--json` object and the persisted DB summary can never disagree.
///
/// Pure and testable (no I/O).
#[must_use]
pub fn build_energy_json(energy: &EnergyResult) -> serde_json::Value {
    serde_json::to_value(EnergySummary::from_energy(energy)).unwrap_or(serde_json::Value::Null)
}

/// The Engine D energy result for the headless path: profile the poller's
/// trace (or the empty trace when no poller is running) over the all-engine
/// token denominator, then apply the load-window numerator (H3 — the whole
/// run in the headless poll()-only path, which never opens a load window).
///
/// **B2:** extracted so the D *gate* (the selection, not the poller flag)
/// and the no-poller N/A case are testable in isolation. When `hw` is
/// `None` (the poller is off — `--no-hardware` / a stale config) the trace is
/// empty and the result is the labeled N/A (the module N/A rule): a selected
/// Engine D is never silently skipped, and never a spurious 0.0 J/token —
/// symmetric with the TUI, which degrades to a labeled N/A without a poller.
#[must_use]
pub fn headless_energy(hw: &Option<Arc<Mutex<HwPoller>>>, total_tokens: u64) -> EnergyResult {
    let (trace_d, load_joules, idle_w, duration, cpu_est) = hw
        .as_ref()
        .and_then(|h| h.lock().ok())
        .map(|g| {
            (
                g.trace().to_vec(),
                g.load_window_joules().or(Some(g.energy_joules())),
                // F9: the monitor's idle floor — the trace profile and the
                // load-window numerator are on the same load-attributable
                // basis (0.0 in the poll()-only path until F7 opens the
                // idle window).
                g.monitor().idle_system_power_w(),
                (g.monitor().duration_sec() > 0.0).then(|| g.monitor().duration_sec()),
                // I4: read **after** the run — a measured backend that
                // fell back to the estimate mid-run is labeled too.
                g.cpu_power_estimated(),
            )
        })
        .unwrap_or_default();
    let mut energy = profile(&trace_d, None, total_tokens, idle_w);
    // I4: the D result carries the estimate flag (the `with_cpu_power_
    // estimated` builder is no longer dead — every production path sets
    // the field before the result reaches `--json` / the DB).
    energy.cpu_power_estimated = cpu_est;
    // H3/F9: the shared helper (the same one the TUI sequence's
    // `persist_results` and the TUI `e`-export use) applies the
    // load-window numerator and keeps `avg_power_w` consistent with the
    // idle-subtracted `joules` — one definition across every surface.
    crate::engines::sequence::apply_load_window(&mut energy, load_joules, total_tokens, duration);
    energy
}

/// The `--matrix-report` assembly: pure over the in-scope run variables
/// (testable like [`headless_energy`] — the `println!` in
/// [`run_headless`] is the only I/O in the flag's path).
///
/// The user fields (backend / quant / use case / notes) are left
/// **empty** (the task spec): the block is a copy-pasteable starting
/// point the user completes before submitting to
/// <https://madgoathaz.github.io/llm-matrix/>.
///
/// - `gpu_name` / `gpu_count` — from the detected backend (the same
///   sources the run's `--json` / DB persist; `None` / `0` on a
///   driverless host — the honest RAM-only line).
/// - `vram_total_mib` — **binary** MiB (the harvested trace's last
///   sample, `vram_total_bytes ÷ 1024²`); `0` when the poller is off →
///   the `vram` field is **blank** (the N/A rule), never a fabricated
///   tier.
/// - `ram_total_bytes` — a fresh `sysinfo` read (the caller).
/// - `speed` — the real Engine A valid-mean `tg_speed`
///   ([`speed_mean_tps`]); `0.0` when there are no valid runs → blank
///   (never a silent `0`, never an estimate).
#[must_use]
pub fn headless_matrix_report(
    cfg: &Config,
    results: &[SpeedResult],
    gpu_name: Option<&str>,
    gpu_count: u32,
    vram_total_mib: u64,
    ram_total_bytes: u64,
) -> MatrixReport {
    let mut report = MatrixReport::build(
        gpu_name.unwrap_or(""),
        gpu_count,
        vram_total_mib,
        ram_total_bytes,
        &cfg.model,
        speed_mean_tps(results),
        "",
        "",
        "",
        "",
    );
    // B1: every valid run pure-CoT → the speed is a blended full-window
    // rate, so the note says so honestly (no-op otherwise).
    if speed_mean_is_blended(results) {
        report.fill_pure_cot_note();
    }
    report
}

/// The `--show-log` read seam (pure over the logger; testable like
/// [`headless_matrix_report`] — the `println!`/`eprintln!` in
/// [`run_headless`] is the only I/O in the flag's path):
///
/// * `Some(Ok(text))` — the run log's content (the lines joined with
///   `\n`), to print on stdout (pipeable);
/// * `Some(Err(e))` — the file is missing/unreadable: print the read
///   notice on **stderr** (N/A-never-fail — the exit-code rule is
///   untouched);
/// * `None` — the logger is disabled (no path): print the unavailable
///   notice on **stderr**.
#[must_use]
pub fn show_log_output(logger: &RunLogger) -> Option<Result<String, String>> {
    logger
        .path()
        .map(|p| read_log_file(p).map(|lines| lines.join("\n")))
}

/// C2/C4 parity: the headless Engine A per-iteration **completion** line —
/// the same pure [`iteration_complete_line`] the TUI logs, so headless and
/// TUI emit the identical format (content/reasoning split + the answer /
/// full-window blended decode rates + TTFT). `None` for a failed iteration
/// (the caller logs the `Iteration {i}/{n} failed` error line instead,
/// mirroring the TUI).
///
/// Pure and testable (no I/O): [`run_headless`] routes each successful
/// iteration through this and fires it on the non-blocking log channel
/// (an mpsc send — never the stream timing path).
#[must_use]
pub fn headless_iteration_complete_line(
    i: usize,
    iterations: usize,
    r: &SpeedResult,
) -> Option<String> {
    (!r.is_failed()).then(|| iteration_complete_line(i, iterations, r))
}

/// The headless (non-TUI) path: run N single-stream iterations (Engine A)
/// and print the prototype's result box(es) / summary — or `--json` on
/// stdout. Returns the process exit code (1 iff every run failed).
///
/// CF-42: the extracted, public lib orchestration (was a private `main`-bin
/// fn). `main.rs` calls it as a thin wrapper.
pub fn run_headless(
    cfg: &Config,
    logger: Arc<RunLogger>,
    gpu: Option<Arc<dyn GpuBackend>>,
    cpu_power: Option<Box<CpuPowerBackend>>,
    session_id: String,
) -> i32 {
    // Empty-selection guard (mirrors the TUI's `App::start_run`): a run
    // with no engines selected has nothing to do. `--engine` names a
    // *complete* selection, so an explicit empty list is an operator
    // mistake we surface, not a silent no-op.
    if cfg.engines.is_empty() {
        eprintln!(
            "crucible-llm: no engines selected — use --engine all, or select some in the Config view (View 6)"
        );
        logger.warn(Context::Setup, "no engines selected — nothing to run");
        return 0;
    }
    // C1: the run-start `[CONFIG]` block — the resolved inputs (one
    // compact line near the top of the run log; a channel send, never a
    // blocking write).
    logger.log_config(cfg);
    // C5: the run wall clock (the digest's `wall Ns`).
    let run_started = std::time::Instant::now();
    // C5: the digest's error lines — the engines' own error surfaces
    // (the logger is fire-and-forget: no record buffer by design).
    let mut run_errors: Vec<String> = Vec::new();

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
    // `--matrix-report`: the (vendor-agnostic) GPU count, captured before
    // `gpu` is moved into the poller below (the trait `poll_all`; NVML
    // overrides per-device).
    let gpu_count = gpu.as_ref().map(|g| g.poll_all().len() as u32).unwrap_or(0);
    // CF-40: the 100 ms hardware sampler runs for the whole headless run
    // **only when the hardware telemetry is enabled** (`--no-hardware` off,
    // or a selection that omits D). It runs on a **dedicated OS thread** —
    // not a `tokio::spawn` task on the `current_thread` runtime below. The
    // benchmark work owns that single runtime thread, and a task shared with
    // it only ticks when the work yields, so the 100 ms cadence was
    // best-effort and slipped under load, degrading the energy integral.
    // A dedicated thread gives the poller its own scheduling (the
    // "decoupled ring" the blueprint promises) without stealing time from
    // the measured streams. Its power trace is sliced into the Silicon
    // Efficiency Metric (Joules/Token). The detected backend (any vendor)
    // drives the GPU fields; the CPU power backend drives the CPU draw; a
    // driverless host yields N/A and the run proceeds. The thread is
    // fire-and-forget: it lives until the process exits.
    let hw: Option<Arc<Mutex<HwPoller>>> = if cfg.hardware {
        let h = Arc::new(Mutex::new(
            HwPoller::with_backend(gpu)
                .with_cpu_power(cpu_power)
                .with_rate(cfg.rate_per_kwh),
        ));
        let poller = h.clone();
        std::thread::spawn(move || loop {
            std::thread::sleep(Duration::from_millis(HW_POLL_INTERVAL_MS));
            if let Ok(mut p) = poller.lock() {
                p.poll();
            }
        });
        Some(h)
    } else {
        None
    };

    rt.block_on(async {
        let term = Term {
            color: !cfg.no_color && std::io::stdout().is_terminal(),
            tty: std::io::stderr().is_terminal(),
        };

        // F7: the headless path opens the **same idle window** as the TUI
        // sequence (begin_run → IDLE_WINDOW_SECS no-load baseline →
        // start_load), so the cumulative integral subtracts the measured
        // no-load floor — headless and TUI define energy identically
        // (before F7 the headless poll()-only path ran with a 0.0 floor,
        // so headless J ≥ TUI J for the same workload). Skipped when the
        // poller is off (the D N/A path is unaffected).
        if let Some(h) = &hw {
            arm_power_monitor(h, IDLE_WINDOW_SECS).await;
        }

        // Engine A (speed) — gated on the selection (R1). `--engine niah`
        // runs *only* C1; a selection that omits `speed` skips Engine A
        // entirely (previously A always ran). The results/packets/windows
        // are declared outside the gate so the persistence + export below
        // can use them (they stay empty when A did not run).
        let mut results: Vec<SpeedResult> = Vec::new();
        // CF-05: the per-run client-perceived first-token latency (T3−T0) for
        // the short-prompt runs, alongside the server-side value in each
        // `SpeedResult.ttft`.
        let mut perceived: Vec<f64> = Vec::new();
        // CF-05: the long-prompt pre-fill throughput (the bandwidth half).
        // Filled after the short-prompt loop; an empty/invalid metric when
        // every short run failed (a dead server) so no wasted prefill.
        let mut prefill = PrefillMetric {
            tok_s: 0.0,
            prompt_tokens: 0,
            server_ttft_s: 0.0,
            estimated: false,
        };
        let mut packets: Vec<PacketSample> = Vec::new();
        let mut windows: Vec<Option<(MonotonicInstant, MonotonicInstant)>> = Vec::new();

        if cfg.engines.speed {
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

            // Keep each iteration's raw worker events (Chunk 13 CSV export
            // + Chunk 17 energy windows): the per-packet arrival timestamps
            // the CSV dumps, and the (T0, Tn) lifecycle window that slices
            // the hardware power trace into per-iteration Joules/Token.
            let capture_packets = cfg.export.is_some();
            results = Vec::with_capacity(iterations);
            for i in 0..iterations {
                if !cfg.json && iterations > 1 {
                    term.progress(&format!("  [{}/{}] Running...", i + 1, iterations));
                }
                let (result, events) = engine.run_iteration_events(&prompt).await;
                if capture_packets {
                    packets.extend(export::samples_from_events(&events, (i + 1) as u64));
                }
                windows.push(run_window(&events));
                perceived.push(perceived_ttft_seconds(&events));
                if cfg.verbose && !cfg.json {
                    term.dim(&format!(
                        "  chunks={} content={} reasoning={}",
                        result.total_chunks, result.content_chunks, result.reasoning_chunks
                    ));
                }
                // C2/C4 parity: the per-iteration completion line — the same
                // `iteration_complete_line()` the TUI logs, so headless and
                // TUI emit the identical format (content/reasoning split +
                // answer / blended rates + TTFT). The non-blocking log
                // channel (an mpsc send) — never the stream timing path. A
                // failed iteration logs the `Iteration {i}/{n} failed`
                // error line instead (mirroring the TUI).
                match headless_iteration_complete_line(i, iterations, &result) {
                    Some(line) => logger.info(Context::EngineA, line),
                    None => logger.error(
                        Context::EngineA,
                        format!(
                            "Iteration {}/{} failed: {}",
                            i + 1,
                            iterations,
                            result.error.as_deref().unwrap_or("unknown")
                        ),
                    ),
                }
                results.push(result);
            }
            if !cfg.json && iterations > 1 {
                term.clear_progress();
            }
            // CF-05: the pre-fill *throughput* half — a long-prompt
            // measurement, distinct from the short-prompt latency above.
            // Skipped when every short run failed (a dead server would only
            // waste a multi-thousand-token prefill).
            if !all_failed(&results) {
                if !cfg.json {
                    term.info(&format!(
                        "Measuring pre-fill throughput on a {}-token long prompt…",
                        cfg.tokens
                    ));
                }
                prefill = engine.run_prefill().await;
            }
            logger.info(
                Context::EngineA,
                format!(
                    "Engine A complete — {}/{} run(s) succeeded; pre-fill {:.1} tok/s ({} tok)",
                    results.iter().filter(|r| !r.is_failed()).count(),
                    results.len(),
                    prefill.tok_s,
                    prefill.prompt_tokens
                ),
            );
            // C5: the failed iterations are the digest's A error lines.
            for r in results.iter().filter(|r| r.is_failed()) {
                run_errors.push(format!(
                    "A: {}",
                    r.error.clone().unwrap_or_else(|| "unknown".to_string())
                ));
            }
        }

        // Chunk 17: harvest the sampler's power trace (brief lock on the
        // run-completion path — never on the stream workers' timing path).
        // Empty / `None` when Engine D was not selected (the poller is off).
        let hw_trace = hw
            .as_ref()
            .and_then(|h| h.lock().ok())
            .map(|g| g.trace().to_vec())
            .unwrap_or_default();
        let gpu_name = hw
            .as_ref()
            .and_then(|h| h.lock().ok())
            .and_then(|g| g.gpu_name().map(str::to_string));

        // The `--json` document: Engine A's results (empty when A was not
        // selected), with the `flat_out` object appended once Engine F has
        // run (the document is printed *after* the additional-engine block
        // below, so `flat_out` is included in one JSON document on stdout).
        let mut json_doc = if cfg.json {
            Some(json_report_split(cfg, &results, &perceived, &prefill))
        } else {
            None
        };
        if !cfg.json && !results.is_empty() {
            if all_failed(&results) {
                for (i, r) in results.iter().enumerate() {
                    term.error(&format!(
                        "Test {}/{} FAILED: {}",
                        i + 1,
                        results.len(),
                        r.error.as_deref().unwrap_or("unknown error")
                    ));
                }
            } else {
                // CF-05: the two distinct, labeled metrics (first-token
                // latency from the short prompt; pre-fill throughput from the
                // long prompt) plus the intact decode numbers.
                print!(
                    "{}",
                    format_split_report(&results, &perceived, &prefill, term.color)
                );
                for r in results.iter() {
                    if let Some(note) = &r.error {
                        if r.completion_tokens > 0 {
                            term.warning(&format!("Note: {note}"));
                        }
                    }
                }
            }
            if let Some(summary) = format_summary(&results, term.color) {
                print!("{summary}");
            }
        }
        // ── Result capture (R2: the B/C2/C3/F headline numbers) ──
        // The additional engines publish their results here so the
        // persistence block (after them) can fold them into the session's
        // `engine_results` column (CF-14: they used to be lost — only A +
        // C1 were persisted).
        let mut sweep_result: Option<SweepResult> = None;
        // C1 NIAH: captured so the persistence block can write the
        // `needle_evaluations` rows (the headless path used to run C1 and
        // drop the result — the TUI/sequence path persists it).
        let mut niah_result: Option<NiahResult> = None;
        let mut reasoning_result: Option<ReasoningResult> = None;
        let mut structured_result: Option<StructuredResult> = None;
        let mut flatout_result: Option<FlatOutResult> = None;
        // Engine D (hardware/energy) — computed after B/C1/C2/C3 and *before*
        // F (the TUI sequence order A→B→C1→C2→C3→D→F) so the headless and TUI
        // J/token agree (H5). `None` until D runs (or when D is not
        // selected). Reused for the (non-JSON) display, the `--json` energy
        // object (DO-09/CF-21), and the `engine_results` column; a
        // driverless host yields the N/A rule (no spurious 0.0).
        let mut energy_result: Option<EnergyResult> = None;

        // Chunk 18: run any *additionally-selected* engines (B / C1 / C2 /
        // C3 / F) and report a one-line summary for each. Engine A (speed)
        // is the run above; Engine D (hardware) is the continuous poller
        // started earlier. A selection of only `speed` skips this block
        // entirely.
        let extra = cfg.engines;
        if !cfg.json
            && (extra.concurrency
                || extra.niah
                || extra.reasoning
                || extra.structured
                || extra.flatout)
        {
            term.dim(&format!(
                "  running selected engines: {}",
                extra.iter_labels().collect::<Vec<_>>().join(", ")
            ));
        }
        // Engine F (Flat Out) reads Engine B's sweet spot as its stream
        // count (the real-world full-load number).
        let mut flatout_streams: Option<(usize, StreamCountSource)> = None;
        // The non-F additional engines (B, C1, C2, C3) run first — H5: D runs
        // after them and *before* F (the TUI sequence order A→B→C1→C2→C3→D→F),
        // so the headless and TUI J/token share the same denominator.
        if extra.concurrency {
            match build_sweep(cfg, None, Some(logger.clone())) {
                Some(sweep) => {
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
                    // The shared helper (also used by the TUI sequence path)
                    // guarantees headless and TUI consume the same max-
                    // throughput level (the envelope).
                    flatout_streams = Some(flatout_streams_for(&r));
                    // FIX 3: the headless [B] summary leads with the
                    // practical (per-stream usability) sweet spot, with
                    // the pure-throughput knee as reference.
                    if r.is_empty() {
                        term.warning("  [B] sweep produced no usable levels");
                    } else {
                        term.info(&format!("  [B] {}", summarize_sweep(&r)));
                    }
                    sweep_result = Some(r);
                }
                None => {
                    // C5: the TUI path logs this (sequence.rs
                    // `run_concurrency`); the headless path surfaced
                    // nothing — an init failure vanishes from the log
                    // and the digest alike.
                    logger.error(
                        Context::EngineB,
                        "Engine B init failed (client build) — sweep skipped",
                    );
                    run_errors.push("B: init failed (client build) — sweep skipped".to_string());
                }
            }
        }
        if extra.niah {
            match NiahEngine::new(cfg) {
                Ok(engine) => {
                    if !cfg.json {
                        // Issue #3 + Issue #4: render the *configured*
                        // ladders with the shared size formatter (was a
                        // hard-coded "7 sizes × 11 depths", and the
                        // per-cell worker tags logged `128000k`). Now the
                        // headline matches the TUI grid (`2k→128k`, `1M`,
                        // …) and names the count-test mode — and carries
                        // the `(approx.)` qualifier when no `--tokenizer`
                        // is set (`niah_size_desc`).
                        term.dim(&niah_size_desc(cfg));
                    }
                    let r = engine.logger(logger.clone()).run().await;
                    // Issue #6: the `--json` document carries the NIAH
                    // result (per-size recall, `reliable_up_to`, per-cell
                    // detail with the `context_label` qualifier) so a
                    // headless `--json` run is self-contained. Additive —
                    // the Engine-A JSON shape is untouched.
                    if let Some(v) = &mut json_doc {
                        v["niah"] = r.to_dict();
                    }
                    term.info(&format!("  [C1] {}", r.accuracy_label()));
                    niah_result = Some(r);
                }
                Err(e) => {
                    term.warning(&format!("[C1] niah init failed: {e}"));
                    logger.error(Context::EngineC1, format!("niah init failed: {e}"));
                    run_errors.push(format!("C1: niah init failed: {e}"));
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
                    reasoning_result = Some(r);
                }
                Err(e) => {
                    term.warning(&format!("[C2] reasoning init failed: {e}"));
                    logger.error(Context::EngineC2, format!("reasoning init failed: {e}"));
                    run_errors.push(format!("C2: reasoning init failed: {e}"));
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
                    // R2: the `--json` document carries the C3
                    // structured result (the score split incl. the
                    // distinct `truncated_by_reasoning` count, the
                    // per-case `truncated_by_reasoning` flags, the
                    // CF-17 compliance label, the CF-36-gated penalty,
                    // the constraint mode) so a headless `--json` run
                    // is self-contained. Additive — the Engine-A JSON
                    // shape is untouched (same pattern as the `niah` /
                    // `flat_out` sections).
                    if let Some(v) = &mut json_doc {
                        v["structured"] = r.to_dict();
                    }
                    term.info(&format!("  [C3] {}", r.summary_line()));
                    structured_result = Some(r);
                }
                Err(e) => {
                    term.warning(&format!("[C3] structured init failed: {e}"));
                    logger.error(Context::EngineC3, format!("structured init failed: {e}"));
                    run_errors.push(format!("C3: structured init failed: {e}"));
                }
            }
        }

        // Engine D (hardware/energy) — after A+B+C1+C2+C3, before F (the TUI
        // sequence order). **B2:** gated on the *selection*
        // (`extra.hardware`), not the poller flag (`cfg.hardware`) — every
        // other engine is gated on its selection, and an explicit
        // `--engine hardware` with the poller off (`--no-hardware` / a stale
        // `config.json`) must not be *silently skipped*: it runs and reports
        // a labeled N/A (symmetric with the TUI), satisfying the `--json`
        // energy-object contract. **H5:** the shared `sum_all_engines_tokens`
        // helper, so the headless and TUI denominators agree (F is not yet
        // run → it contributes 0, consistent with CF-03). **H3:** the
        // numerator is the load window (the whole run in the headless
        // poll()-only path, which never opens a load window), not the bounded
        // 6-min trace.
        if extra.hardware {
            let total_tokens = sum_all_engines_tokens(
                Some(results.as_slice()),
                sweep_result.as_ref(),
                niah_result.as_ref(),
                reasoning_result.as_ref(),
                structured_result.as_ref(),
                None,
            );
            energy_result = Some(headless_energy(&hw, total_tokens));
        }
        if let Some(energy) = &energy_result {
            if !cfg.json {
                // I5: the energy line is a pure function of the result —
                // the measured figure with its estimate label, or the
                // labeled N/A line (B3: never a silent gap, never a
                // spurious 0.0 J/token).
                if let Some(line) = headless_energy_line(energy) {
                    term.dim(&line);
                }
                if let Some(w) = &energy.fragmentation_warning {
                    term.warning(&format!("VRAM: {w}"));
                }
            }
            // DO-09/CF-21: the `--json` document carries the energy object
            // (the serializable summary, not the non-`Serialize` raw
            // `EnergyResult`). P3-A/CF-53/I4: the summary carries the
            // `cpu_power_estimated` flag (set in [`headless_energy`] from
            // the backend *after* the run, including a mid-run sample-level
            // fallback) so the output reflects when the power is the
            // labeled fixed-estimate fallback.
            if let Some(v) = &mut json_doc {
                v["energy"] = build_energy_json(energy);
            }
        }

        // F runs last, after D.
        if extra.flatout {
            match FlatOutEngine::new(cfg) {
                Ok(mut engine) => {
                    // The stream count is Engine B's max-throughput level
                    // (the sweep envelope) when the sweep ran (real-world
                    // full load), else the built-in default.
                    if let Some((n, source)) = flatout_streams {
                        engine = engine.stream_count(n, source);
                    }
                    if !cfg.json {
                        term.dim(
                            "  [F] flat out (max-throughput full load — real-world max t/s)…",
                        );
                    }
                    let r = engine.logger(logger.clone()).run().await;
                    // The `--json` document carries the flat_out
                    // result object.
                    if let Some(v) = &mut json_doc {
                        v["flat_out"] = r.to_dict();
                    }
                    term.info(&format!("  [F] {}", r.summary_line()));
                    flatout_result = Some(r);
                }
                Err(e) => {
                    term.warning(&format!("[F] flat out init failed: {e}"));
                    logger.error(Context::EngineF, format!("flat out init failed: {e}"));
                    run_errors.push(format!("F: flat out init failed: {e}"));
                }
            }
        }

        // F7: close the load window once the workload is done, so the
        // frozen value (end − start) is the numerator every surface
        // (terminal display, `--json`, DB, export) reads.
        if let Some(h) = &hw {
            if let Ok(mut p) = h.lock() {
                p.end_load();
            }
        }

        // R2 (traceability): fold the non-stream engine results
        // (B/C2/C3/D/F) into the session's `engine_results` JSON column
        // (CF-14: they used to be lost — only A + C1 were persisted).
        let engine_results = EngineResults {
            b_concurrency: sweep_result.as_ref().map(ConcurrencySummary::from_sweep),
            c2_reasoning: reasoning_result
                .as_ref()
                .map(ReasoningSummary::from_reasoning),
            c3_structured: structured_result
                .as_ref()
                .map(StructuredSummary::from_structured),
            d_energy: energy_result.as_ref().map(EnergySummary::from_energy),
            f_flatout: flatout_result.as_ref().map(FlatOutSummary::from_flatout),
        }
        .to_json();

        // Persist the completed run (Chunk 12): one `benchmark_sessions`
        // row + one `stream_metrics` row per iteration, in the platform
        // data dir. A storage failure degrades gracefully — it must never
        // break (or change the exit code of) a benchmark run.
        // R2: the session id is the one generated up front in `main` and
        // stamped into the run-log — so the DB row and the log join on it.
        let session = BenchmarkSession {
            session_id,
            timestamp: None, // SQLite `DEFAULT CURRENT_TIMESTAMP` fills it
            target_url: cfg.url.clone(),
            model_name: cfg.model.clone(),
            backend_type: None,
            quantization: None,
            system_gpu: gpu_name.clone(),
            total_duration_sec: Some(results.iter().map(|r| r.stream_time).sum()),
            // R2 (traceability): record which engines this run selected so
            // a stored session joins back to its run / log / export.
            engines: Some(cfg.engines.selected_names_json()),
            // R2: the B/C2/C3/D/F headline results (CF-14).
            engine_results,
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
                // CF-02: populate `stream_metrics.reasoning_tokens` when the
                // server reported the detail (the seam is now closed). A
                // reported `0` (thinking suppressed) is distinct from `None`
                // (the backend did not report it), so only the `Some` case is
                // written — the `NULL` no-longer.
                if let Some(t) = r.reasoning_tokens {
                    row = row.with_reasoning_tokens(t);
                }
                row.joules_per_token = windows[i]
                    .and_then(|w| joules_per_token(&hw_trace, Some(w), r.completion_tokens));
                row
            })
            .collect();
        match Database::open_default() {
            Ok(mut db) => {
                match db.persist_run(&session, &rows) {
                    Ok(()) => {
                        // Engine C1 (NIAH): persist the needle evaluations —
                        // mirroring the TUI/`sequence.rs` path. The headless
                        // run used to execute C1 and drop the result, leaving
                        // `needle_evaluations` empty (and the DB-built export
                        // without per-size retrieval reliability).
                        let needles = niah_result
                            .as_ref()
                            .map(|n| n.to_needle_rows(&session.session_id))
                            .unwrap_or_default();
                        for n in &needles {
                            if let Err(e) = db.insert_needle_evaluation(n) {
                                logger
                                    .warn(Context::EngineC1, format!("persist needle failed: {e}"));
                                break;
                            }
                        }
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
                        (detected > 0).then_some(crate::metrics::LoopGuardSummary {
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

        // C5: the run-end digest — the compact shareable block (the
        // `[CONFIG]` line + the `✓` per-engine headlines + any errors +
        // the totals), referenced by the session id, appended to the run
        // log. Emitted **before** the `--show-log` print, and
        // `logger.finish()` joins the writer thread so the file is
        // complete when the print reads it (main.rs's later `finish()`
        // is an idempotent no-op). All channel sends — run-end path
        // only, never the timing path.
        {
            let mut headlines: Vec<(String, String)> = Vec::new();
            if cfg.engines.speed {
                headlines.push((Engine::Speed.label().to_string(), summarize_speed(&results)));
            }
            if let Some(r) = &sweep_result {
                headlines.push((Engine::Concurrency.label().to_string(), summarize_sweep(r)));
            }
            if let Some(r) = &niah_result {
                headlines.push((Engine::Niah.label().to_string(), r.accuracy_label()));
            }
            if let Some(r) = &reasoning_result {
                headlines.push((
                    Engine::Reasoning.label().to_string(),
                    format!("{} · avg {:.1} t/s", r.score.label(), r.avg_tg_speed()),
                ));
            }
            if let Some(r) = &structured_result {
                headlines.push((Engine::Structured.label().to_string(), r.summary_line()));
            }
            if let Some(e) = &energy_result {
                headlines.push((Engine::Hardware.label().to_string(), hardware_summary(e)));
            }
            if let Some(r) = &flatout_result {
                headlines.push((Engine::FlatOut.label().to_string(), r.summary_line()));
            }
            let total_tokens = sum_all_engines_tokens(
                Some(results.as_slice()),
                sweep_result.as_ref(),
                niah_result.as_ref(),
                reasoning_result.as_ref(),
                structured_result.as_ref(),
                flatout_result.as_ref(),
            );
            // Peak VRAM from the already-harvested trace (binary MiB);
            // `None` (no telemetry) → the digest's honest `n/a`.
            let peak_vram_mib = hw_trace
                .iter()
                .filter_map(|s| s.vram_used_bytes)
                .max()
                .map(|b| b / (1024 * 1024));
            logger.log_digest(&crate::log::digest_lines(
                cfg,
                &session.session_id,
                &headlines,
                &run_errors,
                total_tokens,
                run_started.elapsed().as_secs_f64(),
                peak_vram_mib,
            ));
            logger.finish();
        }

        // The `--json` document (printed after the additional engines so
        // the `flat_out` + `energy` objects are included): stdout stays
        // pure JSON.
        if let Some(v) = json_doc {
            println!("{v}");
        }

        // `--matrix-report`: print the `[COMMUNITY-SUBMISSION]` block on
        // stdout (CI-pipeable via the non-TTY auto-fallback, main.rs — a
        // pipe run is headless, so the block lands on the same stream).
        // With `--json`, the JSON document prints *first*: the block's
        // markers cannot appear inside the JSON, so a pipe consumer
        // slices stdout on them. The user fields are left blank — the
        // block is a copy-pasteable starting point.
        if cfg.matrix_report {
            // VRAM: the already-harvested trace's last sample
            // (bytes → **binary** MiB) — empty when the poller is off
            // (the honest blank, never a fabricated tier).
            let vram_mib = hw_trace
                .last()
                .and_then(|s| s.vram_total_bytes)
                .map(|b| b / (1024 * 1024))
                .unwrap_or(0);
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let report = headless_matrix_report(
                cfg,
                &results,
                gpu_name.as_deref(),
                gpu_count,
                vram_mib,
                sys.total_memory(),
            );
            println!("{}", report.render());
        }

        // `--show-log`: print the full run log on stdout (pipeable —
        // `| less`, `| grep`). The file is complete at this point: the
        // writer flushes on every newline and `logger.finish()` runs
        // *after* `run_headless` returns (main.rs), so nothing is lost.
        // With `--json`, the JSON document prints *first* (the same
        // ordering as the `--matrix-report` block above): the log's
        // `[ts] [LEVEL] [CTX]` line shape is what a pipe consumer slices
        // on. N/A-never-fail: read errors / a disabled logger go to
        // **stderr** — the exit-code rule below is untouched.
        if cfg.show_log {
            match show_log_output(&logger) {
                Some(Ok(text)) => {
                    let mut stdout = std::io::stdout();
                    let _ = writeln!(stdout, "{text}");
                }
                Some(Err(e)) => eprintln!("crucible-llm: cannot read run log: {e}"),
                None => eprintln!("crucible-llm: run log unavailable (logger disabled)"),
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

/// I5: the headless terminal energy line — a pure function of the
/// [`EnergyResult`] (unit-testable, no TTY):
///
/// * the **measured** figure (`J` + `J/token` + peak), with the
///   **estimate label** `(CPU ~)` appended when the CPU component of the
///   system draw is the labeled fixed-estimate fallback (I4: including a
///   measured backend that fell back to the estimate mid-run) — the
///   TUI's `~` / `(CPU ~)` marker, now on the terminal line too;
/// * a **labeled N/A** line for each root cause (B3: "no power telemetry"
///   vs "no tokens generated") — never a silent gap, never a spurious
///   `0.0 J/token`.
///
/// `None` only when there is nothing to say (no energy result at all).
#[must_use]
pub fn headless_energy_line(energy: &EnergyResult) -> Option<String> {
    let line = if let Some(jpt) = energy.joules_per_token {
        let peak = energy
            .peak_power_w
            .map(|w| format!(" · peak {w:.0} W"))
            .unwrap_or_default();
        // I5: the estimate label (the energy includes the labeled CPU
        // estimate, so the number is flagged, not hidden).
        let estimated = if energy.cpu_power_estimated { " (CPU ~)" } else { "" };
        format!(
            "  Energy: {:.1} J ({:.3} J/token{peak}){estimated}",
            energy.joules, jpt
        )
    } else {
        let reason = energy.na_reason().unwrap_or("no power telemetry");
        match reason {
            "no power telemetry" => {
                // No power telemetry (driverless host, or a sensor with no
                // usable reading — e.g. the RAPL quirk on a shared host).
                "  Energy: N/A — no power telemetry (no valid power sensor on this host)"
                    .to_string()
            }
            // I7: telemetry + tokens present, but the (idle-subtracted)
            // draw never rose above the floor — the `0.0` N/A value,
            // labeled (never a silent 0.0 J/token).
            "no energy measured" => {
                "  Energy: N/A — no energy measured above the idle floor (power telemetry present)"
                    .to_string()
            }
            // Power telemetry is present, but the run produced zero tokens
            // (a D-only / no-work run) — nothing to divide the energy by.
            _ => "  Energy: N/A — no tokens generated this run (power telemetry present)"
                .to_string(),
        }
    };
    Some(line)
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

/// The `[C1] NIAH (...)` header line for a headless NIAH run (Issue #3
/// sizes + Issue #4 approx qualifier). Pure and testable: the sizes are
/// rendered with the shared [`context_size_label`] formatter, which carries
/// the `(approx.)` qualifier when no `--tokenizer` is set (the `chars/4`
/// estimate) — consistent with the TUI grid, so a headless reader sees the
/// same token-honesty the TUI shows.
fn niah_size_desc(cfg: &Config) -> String {
    let approx = cfg.tokenizer.is_none();
    let size_desc = match cfg.niah_custom {
        Some(size) => format!("count test at {}", context_size_label(size, approx)),
        None => {
            let sizes = &cfg.niah_contexts;
            if sizes.len() == 1 {
                format!("1 size {}", context_size_label(sizes[0], approx))
            } else {
                format!(
                    "{} sizes {}→{}",
                    sizes.len(),
                    context_size_label(sizes[0], approx),
                    context_size_label(*sizes.last().unwrap(), approx)
                )
            }
        }
    };
    format!(
        "  [C1] NIAH ({size_desc} × {} depths)…",
        cfg.niah_depths.len()
    )
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
    loop_guard: Option<crate::metrics::LoopGuardSummary>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineSelection;
    use crate::hw::GpuSample;
    use crate::sse::chunk::Chunk;
    use crate::sse::parser::ParsedFrame;
    use crate::timing::StreamTimestamps;

    /// CF-42: `run_headless` is a public lib fn callable with the same
    /// signature the bin uses. The empty-selection guard returns early
    /// (no server, no network I/O) — a server-free proof the extraction is
    /// testable/reusable.
    #[test]
    fn run_headless_is_callable_and_guard_returns_on_empty_selection() {
        let cfg = Config {
            engines: EngineSelection {
                speed: false,
                concurrency: false,
                niah: false,
                reasoning: false,
                structured: false,
                hardware: false,
                flatout: false,
            },
            ..Config::default()
        };
        let logger = RunLogger::start(None);
        let code = run_headless(&cfg, logger, None, None, "p3b-test".to_string());
        assert_eq!(code, 0, "empty selection → the guard returns 0 (no run)");
    }

    /// CF-42: the extracted function is the public lib contract
    /// (`crucible_llm::run_headless`), with the bin's exact signature.
    #[test]
    #[allow(clippy::type_complexity)] // the 5-arg fn-pointer *is* the contract
    fn run_headless_has_the_public_lib_signature() {
        let _f: fn(
            &Config,
            Arc<RunLogger>,
            Option<Arc<dyn GpuBackend>>,
            Option<Box<CpuPowerBackend>>,
            String,
        ) -> i32 = run_headless;
    }

    /// C2/C4 parity: the headless Engine A per-iteration completion line is
    /// the SAME pure `iteration_complete_line` the TUI logs — so headless
    /// and TUI emit the identical format (content/reasoning split +
    /// answer / blended rates + TTFT). A failed iteration (an error with
    /// zero completion tokens) yields `None` (the caller logs the
    /// `Iteration {i}/{n} failed` error line instead).
    #[test]
    fn headless_iteration_line_matches_the_tui_format() {
        // A pure-CoT run: all 256 completion tokens are reasoning (0
        // content) — the v0.1.7 shape.
        let pure_cot = SpeedResult {
            ttft: 0.18,
            prompt_tokens: 60,
            completion_tokens: 256,
            reasoning_tokens: Some(256),
            pp_speed: 333.3,
            tg_speed: 56.4,
            mtp_efficiency: 1.0,
            stream_time: 4.6,
            total_chunks: 256,
            content_chunks: 0,
            reasoning_chunks: 256,
            other_chunks: 0,
            estimated: false,
            model: "m".into(),
            mode: "chat".into(),
            error: None,
            looping: false,
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
            pure_cot: true,
            blended_tps: 56.4,
        };
        // The headless line is byte-identical to the TUI's pure builder.
        let line = headless_iteration_complete_line(0, 3, &pure_cot).expect("a completion line");
        assert_eq!(line, iteration_complete_line(0, 3, &pure_cot));
        assert_eq!(
            line,
            "Iteration 1/3 complete: 256 tok (0 content / 256 reasoning), \
             56.4 t/s answer, 56.4 t/s blended, TTFT 180 ms"
        );

        // A failed iteration (error + zero completion tokens) → `None` (the
        // caller logs the `Iteration {i}/{n} failed` error line instead).
        let failed = SpeedResult {
            completion_tokens: 0,
            reasoning_tokens: None,
            tg_speed: 0.0,
            blended_tps: 0.0,
            pure_cot: false,
            error: Some("connection refused".into()),
            ..pure_cot
        };
        assert!(
            headless_iteration_complete_line(0, 3, &failed).is_none(),
            "a failed iteration must not emit a completion line"
        );
    }

    /// P3-A / CF-53 / I4: the `--json` energy object carries the
    /// `cpu_power_estimated` flag (now from the `EnergyResult` itself —
    /// the same field the persisted `EnergySummary` copies, so `--json`
    /// and the DB can never disagree), alongside the `EnergySummary`
    /// fields.
    #[test]
    fn energy_json_carries_the_estimated_flag() {
        let base = EnergyResult {
            joules: 100.0,
            joules_per_token: Some(0.5),
            avg_power_w: Some(200.0),
            peak_power_w: Some(300.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(90.0),
            vendor: Some("NVIDIA".into()),
            model: Some("RTX A4000".into()),
            cpu_power_estimated: false,
        };

        // Estimated CPU power → the flag is `true` in the JSON object.
        let e_est = EnergyResult {
            cpu_power_estimated: true,
            ..base.clone()
        };
        let v = build_energy_json(&e_est);
        assert_eq!(v["cpu_power_estimated"], true);
        // The `EnergySummary` fields are present too (not clobbered).
        assert_eq!(v["joules"], 100.0);
        assert_eq!(v["joules_per_token"], 0.5);
        assert_eq!(v["avg_power_w"], 200.0);
        assert_eq!(v["vendor"], "NVIDIA");
        assert_eq!(v["model"], "RTX A4000");

        // Measured CPU power → the flag is `false` (omitted from the JSON,
        // the `false`-is-absent convention).
        let v2 = build_energy_json(&base);
        assert!(v2.get("cpu_power_estimated").is_none(), "{v2}");
        assert_eq!(v2["joules"], 100.0);
    }

    /// I5: the terminal energy line carries the **estimate label** when the
    /// CPU component is the labeled fixed-estimate fallback — and no label
    /// for a fully-measured run (the TUI's `(CPU ~)` marker, on the
    /// terminal line).
    #[test]
    fn headless_energy_line_labels_the_estimated_cpu_component() {
        let base = EnergyResult {
            joules: 12_345.6,
            joules_per_token: Some(0.098),
            avg_power_w: Some(229.0),
            peak_power_w: Some(521.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(91.0),
            vendor: Some("NVIDIA".into()),
            model: Some("RTX 3080 Ti".into()),
            cpu_power_estimated: false,
        };
        // Measured CPU power → no estimate marker.
        let line = headless_energy_line(&base).expect("a line");
        assert!(line.contains("12345.6 J"), "{line}");
        assert!(line.contains("0.098 J/token"), "{line}");
        assert!(line.contains("peak 521 W"), "{line}");
        assert!(!line.contains("~"), "no estimate marker: {line}");

        // Estimated CPU power → the marker is present.
        let est = EnergyResult {
            cpu_power_estimated: true,
            ..base
        };
        let line_est = headless_energy_line(&est).expect("a line");
        assert!(line_est.contains("(CPU ~)"), "the estimate label: {line_est}");
    }

    /// I5: the power-present / 0-token case prints a **labeled** N/A line
    /// (B3's root cause) — never a silent gap, never a spurious 0.0.
    #[test]
    fn headless_energy_line_is_labeled_na_for_zero_tokens() {
        // Power present, zero tokens (a D-only / no-work run).
        let no_tokens = EnergyResult {
            joules: 321.0,
            joules_per_token: None, // 0 tokens → N/A, power IS present
            avg_power_w: Some(120.0),
            peak_power_w: Some(200.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(90.0),
            vendor: Some("NVIDIA".into()),
            model: Some("Mock".into()),
            cpu_power_estimated: false,
        };
        let line = headless_energy_line(&no_tokens).expect("a line");
        assert!(
            line.contains("N/A — no tokens generated"),
            "the real cause is named: {line}"
        );

        // No power telemetry at all (driverless host).
        let no_power = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: None,
            peak_power_w: None,
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: None,
            vendor: None,
            model: None,
            cpu_power_estimated: false,
        };
        let line2 = headless_energy_line(&no_power).expect("a line");
        assert!(
            line2.contains("N/A — no power telemetry"),
            "the no-telemetry cause: {line2}"
        );
    }

    /// I7: the zero-energy-above-the-floor case prints its own **labeled**
    /// N/A line (telemetry present, draw never rose above the floor) —
    /// never a silent gap, never a spurious 0.0.
    #[test]
    fn headless_energy_line_labels_no_energy_measured() {
        let e = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: Some(0.0),
            peak_power_w: Some(80.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(90.0),
            vendor: Some("NVIDIA".into()),
            model: Some("Mock".into()),
            cpu_power_estimated: false,
        };
        let line = headless_energy_line(&e).expect("a line");
        assert!(line.contains("N/A — no energy measured"), "{line}");
    }

    /// I4: `headless_energy` reads the estimate flag from the poller's
    /// CPU backend **after** the run — a measured (RAPL) backend that
    /// latched a sample-level estimate fallback mid-run is labeled, not
    /// presented as fully-measured.
    #[test]
    fn headless_energy_labels_a_mid_run_estimate_fallback() {
        use crate::hw::cpu_power::{CpuPowerBackend, CpuPowerMethod};
        // A RAPL-accepted backend with a latched estimated sample (the
        // mid-run fallback the I4 fix must surface).
        let cpu = CpuPowerBackend::from_method_for_test(CpuPowerMethod::Rapl, true);
        let hw = Some(Arc::new(Mutex::new(
            HwPoller::with_backend(Some(Arc::new(PowerGpuBackend { watts: 150.0 })))
                .with_cpu_power(Some(Box::new(cpu))),
        )));
        {
            let mut p = hw.as_ref().unwrap().lock().unwrap();
            p.poll();
            std::thread::sleep(Duration::from_millis(5));
            p.poll();
        }
        let energy = headless_energy(&hw, 1000);
        assert!(
            energy.cpu_power_estimated,
            "the mid-run fallback labels the result"
        );
        // …and the `--json` object carries it.
        let v = build_energy_json(&energy);
        assert_eq!(v["cpu_power_estimated"], true);
        // A fully-measured backend stays unlabeled.
        let cpu_measured = CpuPowerBackend::from_method_for_test(CpuPowerMethod::Rapl, false);
        let hw2 = Some(Arc::new(Mutex::new(
            HwPoller::with_backend(Some(Arc::new(PowerGpuBackend { watts: 150.0 })))
                .with_cpu_power(Some(Box::new(cpu_measured))),
        )));
        {
            let mut p = hw2.as_ref().unwrap().lock().unwrap();
            p.poll();
            std::thread::sleep(Duration::from_millis(5));
            p.poll();
        }
        let energy2 = headless_energy(&hw2, 1000);
        assert!(!energy2.cpu_power_estimated, "measured stays unlabeled");
    }

    /// A deterministic GPU backend with a real power reading (B2 test
    /// fixture): a poller built on it accumulates a power-bearing trace.
    #[derive(Debug)]
    struct PowerGpuBackend {
        watts: f64,
    }

    impl GpuBackend for PowerGpuBackend {
        fn vendor(&self) -> &str {
            "NVIDIA"
        }
        fn model(&self) -> Option<&str> {
            Some("Mock GPU")
        }
        fn poll(&self) -> GpuSample {
            GpuSample {
                power_watts: Some(self.watts),
                utilization_pct: Some(90),
                ..GpuSample::default()
            }
        }
    }

    /// B2: a selected Engine D with **no poller** (`hw = None` —
    /// `--no-hardware` / a stale `config.json`) is not silently skipped: it
    /// yields the labeled N/A, and the `--json` energy object is present with
    /// `na_reason` set and `joules_per_token` `null` (the module N/A rule,
    /// symmetric with the TUI).
    #[test]
    fn headless_d_no_poller_is_a_labeled_na_in_json() {
        let hw: Option<Arc<Mutex<HwPoller>>> = None;
        // D-only: no other engine produced tokens.
        let energy = headless_energy(&hw, 0);
        assert!(energy.is_na(), "no poller → no power telemetry → N/A");
        assert_eq!(energy.joules_per_token, None, "never a spurious 0.0");

        // The `--json` energy object is present and carries the reason.
        let v = build_energy_json(&energy);
        assert!(v.get("joules").is_some(), "energy object present: {v}");
        assert!(
            v.get("na_reason")
                .and_then(|r| r.as_str())
                .is_some_and(|r| !r.is_empty()),
            "na_reason is a non-empty label: {v}"
        );
        // `joules_per_token` is `None` → omitted from the object (the N/A
        // rule: never a spurious 0.0).
        assert!(
            v.get("joules_per_token").is_none(),
            "joules_per_token is absent (N/A), never a spurious 0.0: {v}"
        );
    }

    /// A GPU backend whose draw is controlled per-phase (the F7 test
    /// fixture: low no-load floor, high load draw).
    #[derive(Debug)]
    struct RampGpuBackend {
        watts: std::sync::Arc<std::sync::Mutex<f64>>,
    }

    impl GpuBackend for RampGpuBackend {
        fn vendor(&self) -> &str {
            "NVIDIA"
        }
        fn model(&self) -> Option<&str> {
            Some("Ramp GPU")
        }
        fn poll(&self) -> GpuSample {
            GpuSample {
                power_watts: Some(*self.watts.lock().unwrap()),
                utilization_pct: Some(90),
                ..GpuSample::default()
            }
        }
    }

    /// F7: the headless path measures and subtracts the **same** no-load
    /// idle floor as the TUI — the arming (`begin_run` → idle window →
    /// `start_load`) happens in `run_headless`, and the cumulative
    /// integral subtracts the measured floor. A 250 W load on a 50 W
    /// floor integrates ~200 W (load-attributable), not ~250 W.
    #[test]
    fn headless_energy_subtracts_the_measured_idle_floor() {
        let watts = Arc::new(Mutex::new(50.0));
        let hw = Some(Arc::new(Mutex::new(HwPoller::with_backend(Some(
            Arc::new(RampGpuBackend {
                watts: watts.clone(),
            }),
        )))));
        // The idle window (what `run_headless` opens via arm_power_monitor):
        // no-load draw at 50 W.
        {
            hw.as_ref().unwrap().lock().unwrap().begin_run();
        }
        for _ in 0..5 {
            *watts.lock().unwrap() = 50.0;
            hw.as_ref().unwrap().lock().unwrap().poll();
            std::thread::sleep(Duration::from_millis(5));
        }
        // Open the load window (finalizes the 50 W floor).
        hw.as_ref().unwrap().lock().unwrap().start_load();
        // The workload: 250 W draw.
        let load_start = MonotonicInstant::now();
        *watts.lock().unwrap() = 250.0;
        for _ in 0..5 {
            hw.as_ref().unwrap().lock().unwrap().poll();
            std::thread::sleep(Duration::from_millis(5));
        }
        // Close the window (what `run_headless` does after the workload).
        hw.as_ref().unwrap().lock().unwrap().end_load();
        let load_end = MonotonicInstant::now();
        let span = load_start.delta_nanos(&load_end) as f64 / 1e9;

        let energy = headless_energy(&hw, 1000);
        // The floor was measured at ~50 W (not the old 0.0)…
        let idle_floor = {
            let guard = hw.as_ref().unwrap().lock().unwrap();
            guard.monitor().idle_system_power_w()
        };
        assert!(
            (idle_floor - 50.0).abs() < 5.0,
            "the measured no-load floor is ~50 W: {idle_floor}"
        );
        // …and the numerator is load-attributable: ~200 W × span, NOT
        // ~250 W × span (the pre-F7 raw basis).
        assert!(
            (energy.joules - 200.0 * span).abs() < 250.0 * span * 0.15,
            "idle-subtracted energy ≈ 200 W × {span}s: {}",
            energy.joules
        );
        assert!(
            energy.joules < 250.0 * span * 0.9,
            "the raw (unsubtracted) basis would be ~250 W × span: {}",
            energy.joules
        );
    }

    /// B2: the normal (poller-present) path is unchanged — a power-bearing
    /// trace with tokens yields a real J/token (not N/A).
    #[test]
    fn headless_d_poller_present_yields_joules_per_token() {
        let hw = Some(Arc::new(Mutex::new(HwPoller::with_backend(Some(
            Arc::new(PowerGpuBackend { watts: 150.0 }),
        )))));
        // Build up a power-bearing trace (two samples, distinct timestamps).
        {
            let mut p = hw.as_ref().unwrap().lock().unwrap();
            p.poll();
            std::thread::sleep(Duration::from_millis(5));
            p.poll();
        }
        let energy = headless_energy(&hw, 1000);
        assert!(
            !energy.is_na(),
            "poller present → power telemetry → not N/A"
        );
        assert!(
            energy.joules_per_token.is_some(),
            "tokens + power → a real J/token"
        );
    }

    /// CF-40 support: `run_window` extracts the (T0, Tn) lifecycle window
    /// from a run's worker events; `None` when a full window is absent.
    #[test]
    fn run_window_extracts_t0_and_tend() {
        let t0 = MonotonicInstant::now();
        let t_end = MonotonicInstant::now();
        let frame = StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content("x".into()),
                t_nanos: 0,
                done: false,
            },
            at: t0,
            timestamps: StreamTimestamps {
                t0: Some(t0),
                ..Default::default()
            },
        };
        let complete = StreamEvent::Complete {
            timestamps: StreamTimestamps {
                t_end: Some(t_end),
                ..Default::default()
            },
            usage: None,
            premature: false,
            malformed_frames: 0,
            looping: false,
            loop_excluded_tokens: 0,
        };
        assert_eq!(
            run_window(&[frame.clone(), complete.clone()]),
            Some((t0, t_end))
        );

        // No full window → `None`: empty, a lone terminal event (no T0), or
        // a lone frame (no Tn).
        assert_eq!(run_window(&[]), None);
        assert_eq!(run_window(&[complete]), None);
        assert_eq!(run_window(&[frame]), None);
    }

    /// Issue #4: the headless NIAH header surfaces the `(approx.)` qualifier
    /// when no `--tokenizer` is set (the `chars/4` estimate), and plain sizes
    /// when a tokenizer is configured — consistent with the TUI grid.
    #[test]
    fn niah_size_desc_marks_approx_without_tokenizer() {
        // No tokenizer → approx qualifier (count-test mode).
        let cfg = Config {
            niah_custom: Some(32_000),
            niah_depths: vec![0, 50, 100],
            tokenizer: None,
            ..Config::default()
        };
        let desc = niah_size_desc(&cfg);
        assert!(desc.contains("32k (approx.)"), "{desc}");
        assert!(desc.contains("3 depths"), "{desc}");

        // Multi-size ladder, no tokenizer → every size qualified.
        let cfg2 = Config {
            niah_contexts: vec![16_000, 64_000],
            niah_depths: vec![0, 50, 100],
            tokenizer: None,
            ..Config::default()
        };
        let desc2 = niah_size_desc(&cfg2);
        assert!(desc2.contains("16k (approx.)→64k (approx.)"), "{desc2}");

        // With a tokenizer → plain sizes (no qualifier).
        let cfg3 = Config {
            niah_custom: Some(32_000),
            niah_depths: vec![0, 50, 100],
            tokenizer: Some(std::path::PathBuf::from("tok.json")),
            ..Config::default()
        };
        let desc3 = niah_size_desc(&cfg3);
        assert!(desc3.contains("32k"), "{desc3}");
        assert!(!desc3.contains("approx."), "{desc3}");
    }

    // ── `--matrix-report` assembly (the pure part of the flag) ──────────

    /// A minimal Engine A result fixture (the report's speed-mean input).
    fn speed_result(tg: f64, failed: bool, looping: bool) -> SpeedResult {
        SpeedResult {
            ttft: 0.2,
            prompt_tokens: 100,
            completion_tokens: if failed { 0 } else { 50 },
            reasoning_tokens: None,
            pp_speed: 1000.0,
            tg_speed: tg,
            mtp_efficiency: 1.0,
            stream_time: 1.0,
            total_chunks: 10,
            content_chunks: 10,
            reasoning_chunks: 0,
            other_chunks: 0,
            estimated: false,
            model: "m".into(),
            mode: "short".into(),
            error: failed.then(|| "boom".to_string()),
            looping,
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
            pure_cot: false,
            blended_tps: 0.0,
        }
    }

    #[test]
    fn headless_matrix_report_assembles_the_block() {
        let cfg = Config {
            model: "test-model".to_string(),
            ..Config::default()
        };
        // Valid mean of 45 + 55 = 50 (the failed + looping runs are
        // excluded).
        let results = vec![
            speed_result(45.0, false, false),
            speed_result(0.0, true, false),
            speed_result(55.0, false, false),
            speed_result(99.0, false, true),
        ];
        // 24576 MiB = 24 GiB (binary) → the `16-24` tier; 64 decimal GB
        // RAM.
        let r = headless_matrix_report(
            &cfg,
            &results,
            Some("NVIDIA GeForce RTX 3080 Ti"),
            1,
            24576,
            64 * 1_000_000_000,
        );
        assert_eq!(r.hardware, "1x NVIDIA GeForce RTX 3080 Ti 24GB + 64GB RAM");
        assert_eq!(r.vram, "24GB — 16-24");
        assert_eq!(r.model, "test-model");
        assert_eq!(r.speed, "50 tok/s");
        // The user fields are blank (the copy-pasteable starting point
        // the user completes before submitting).
        assert_eq!(r.backend, "");
        assert_eq!(r.quant, "");
        assert_eq!(r.use_case, "");
        assert_eq!(r.notes, "");
        // The exact block the `println!` emits (the parser contract:
        // markers + 8 ordered lines + the 10-column value alignment).
        let rendered = r.render();
        assert!(rendered.ends_with('\n'), "trailing newline (file-friendly)");
        let lines: Vec<&str> = rendered.lines().collect();
        assert_eq!(lines[0], "[COMMUNITY-SUBMISSION]");
        assert_eq!(
            lines[1],
            "hardware: 1x NVIDIA GeForce RTX 3080 Ti 24GB + 64GB RAM"
        );
        assert_eq!(lines[2], "vram:     24GB — 16-24");
        assert_eq!(lines[3], "model:    test-model");
        assert_eq!(lines[4], "backend:  ");
        assert_eq!(lines[5], "quant:    ");
        assert_eq!(lines[6], "speed:    50 tok/s");
        assert_eq!(lines[7], "use_case: ");
        assert_eq!(lines[8], "notes:    ");
        assert_eq!(lines[9], "[/COMMUNITY-SUBMISSION]");
    }

    #[test]
    fn headless_matrix_report_no_gpu_is_honest() {
        let cfg = Config {
            model: "test-model".to_string(),
            ..Config::default()
        };
        // No GPU (name `None`, count 0), no VRAM telemetry, no runs.
        let r = headless_matrix_report(&cfg, &[], None, 0, 0, 32 * 1_000_000_000);
        assert_eq!(r.hardware, "32GB RAM", "RAM-only line, never a 1x …");
        assert_eq!(r.vram, "", "no telemetry → blank (the N/A rule)");
        assert_eq!(r.speed, "", "no runs → blank (never a silent 0)");
        assert_eq!(r.model, "test-model");
    }

    #[test]
    fn headless_matrix_report_speed_blank_when_no_valid_runs() {
        let cfg = Config {
            model: "m".to_string(),
            ..Config::default()
        };
        // A failed run + a looping run: no *valid* runs → blank speed.
        let results = vec![
            speed_result(0.0, true, false),
            speed_result(88.0, false, true),
        ];
        let r = headless_matrix_report(&cfg, &results, Some("G"), 1, 8192, 16 * 1_000_000_000);
        assert_eq!(r.speed, "", "no valid runs → blank, never a silent 0");
        // 8192 MiB = 8 GiB (binary) → the `≤16` tier.
        assert_eq!(r.vram, "8GB — ≤16");
        assert_eq!(r.hardware, "1x G 8GB + 16GB RAM");
    }

    // ── `--show-log` (the read seam) ─────────────────────────────────────

    #[test]
    fn show_log_output_disabled_logger_is_none() {
        // A disabled logger (no path) → the `None` arm: the caller
        // prints the "logger disabled" stderr notice.
        let logger = RunLogger::disabled();
        assert!(show_log_output(&logger).is_none());
    }

    #[test]
    fn show_log_output_reads_the_run_log() {
        let dir =
            std::env::temp_dir().join(format!("crucible-showlog-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        logger.info(Context::Setup, "run started");
        logger.warn(Context::EngineB, "step 2 slow");
        logger.finish(); // the writer thread flushes both files

        let text = show_log_output(&logger)
            .expect("an active logger has a path")
            .expect("the file is readable");
        // The header is the first line (the writer emits it before
        // draining the channel), then the records in order.
        assert!(text.starts_with("=== crucible-llm run log"), "{text}");
        assert!(text.contains("[INFO] [SETUP] run started"), "{text}");
        assert!(text.contains("[WARN] [ENGINE_B] step 2 slow"), "{text}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn show_log_output_unreadable_file_is_an_err_not_a_panic() {
        // The path survives (the logger opened the file at start); the
        // read fails (the file is gone) → `Err` (the caller prints the
        // stderr notice), never a panic.
        let dir = std::env::temp_dir()
            .join(format!("crucible-showlog-missing-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        logger.finish();
        std::fs::remove_file(dir.join("latest.log")).ok();
        let out = show_log_output(&logger).expect("an active logger has a path");
        assert!(out.is_err(), "a missing file is Err: {out:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
