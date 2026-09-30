//! The four benchmark engines: A (speed), B (concurrency), C (capability),
//! and D (hardware).
//!
//! * [`speed`] — Engine A (Chunk 7): single-stream TTFT / PP / TG / MTP
//!   orchestration; the headless result box / `--json` report / summary
//!   (parity with `llmspeedtest.py`).
//! * [`concurrency`] — Engine B (Chunk 10 + 11): the multi-stream ladder
//!   sweep (`1→2→4→8→16→32→64`), collecting per-level aggregate tokens/sec
//!   and client-perceived p90 TPOT across all concurrent streams, plus
//!   saturation knee-point detection and the "Optimal Operational
//!   Envelope" (the recommended sweet spot).
//! * [`capability`] — Engine C (Chunk 15 + 16): C1 needle-in-a-haystack
//!   context retention, C2 deterministic reasoning/code verification,
//!   and C3 structured-output/JSON-grammar compliance.
//! * [`hardware`] — Engine D (Chunk 17): the silicon-efficiency profiler —
//!   `Joules/Token = ∫P(t)dt / Total_Generated_Tokens` over the 100 ms
//!   hardware power trace, plus the VRAM fragmentation warning.
//!
//! **Chunk 18 — full integration:** [`run_selected`] orchestrates *any*
//! combination of the four engines from a single [`Config`] (the
//! [`EngineSelection`] in `cfg.engines`), and [`ResultSlot`] is the
//! lock-free `ArcSwap` double-buffer seam the TUI uses to read a
//! completed engine result without ever touching the timing path
//! (measurement-isolation invariant, blueprint §4).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::config::Config;
use crate::metrics::state::MetricsState;

pub mod capability;
pub mod concurrency;
pub mod hardware;
pub mod speed;

pub use capability::{
    build_document, classify, is_json_compliant, score_responses, Challenge, Checker, Needle,
    NiahCell, NiahCellState, NiahDocument, NiahEngine, NiahEngineConfig, NiahResult, NiahSlot,
    ReasoningEngine, ReasoningResult, ReasoningScore, StructuredEngine, StructuredResult,
    NIAH_DEPTHS, NIAH_MAX_GEN_TOKENS, NIAH_SIZES, PREFILL_THROTTLE_FACTOR, REASONING_BANK,
    REASONING_MAX_GEN_TOKENS, REQUIRED_FIELDS, STRUCTURED_MAX_GEN_TOKENS, STRUCTURED_TASK,
};
pub use concurrency::{
    normalize_ladder, Envelope, KneePoint, Sweep, SweepLevel, SweepResult, DEFAULT_LADDER,
    KNEE_GAIN_THRESHOLD, KNEE_SPIKE_THRESHOLD,
};
pub use hardware::{
    fragmentation_warning, integrate_joules, joules_per_token, profile, EnergyResult,
    VRAM_FRAGMENTATION_THRESHOLD,
};
pub use speed::{
    aggregate, all_failed, format_result_box, format_summary, json_report, EngineError,
    SpeedEngine, SpeedResult, MAX_GEN_TOKENS,
};

// Re-export the config-domain selection so `engines::EngineSelection` is the
// canonical path (the type itself lives in `crate::config`, the single
// source of truth for run configuration).
pub use crate::config::EngineSelection;

// ── Lock-free result slot (TUI seam, Chunk 18) ────────────────────────────

/// A lock-free holder for a completed engine result plus a running flag.
///
/// The same `ArcSwap` double-buffer pattern as the Chunk 6
/// [`crate::metrics::state::MetricsState`] and the Chunk 15 [`NiahSlot`]:
/// a background runner (spawned from a key press) *publishes* via
/// [`store`], and a TUI view *reads* via [`load`] — the render loop never
/// blocks, never takes a mutex, and never touches the quanta timing path
/// (measurement-isolation invariant, blueprint §4).
///
/// `T` is the engine's result type (e.g. `Vec<SpeedResult>`,
/// [`ReasoningResult`], [`StructuredResult`]).
#[derive(Debug)]
pub struct ResultSlot<T> {
    result: ArcSwap<Option<T>>,
    running: AtomicBool,
}

impl<T> ResultSlot<T> {
    /// An empty slot (no result yet, nothing running).
    pub fn new() -> Self {
        Self {
            result: ArcSwap::from_pointee(None),
            running: AtomicBool::new(false),
        }
    }

    /// Lock-free read of the current result (`None` until the first run
    /// completes).
    pub fn load(&self) -> Arc<Option<T>> {
        self.result.load_full()
    }

    /// Publish a completed result.
    pub fn store(&self, result: T) {
        self.result.store(Arc::new(Some(result)));
    }

    /// Mark a run in progress (key path) / finished (runner path).
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// `true` while a run is in progress.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl<T> Default for ResultSlot<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ── Engine orchestration (Chunk 18) ───────────────────────────────────────

/// The combined result of a multi-engine run: whichever of A/B/C1/C2/C3
/// were selected in [`EngineSelection`] have their results here; the rest
/// stay empty/`None`. Engine D (hardware) is continuous telemetry, not a
/// one-shot result, so it is driven by the [`crate::hw::HwPoller`] rather
/// than collected here.
#[derive(Debug, Default)]
pub struct RunReport {
    /// Engine A — one [`SpeedResult`] per iteration.
    pub speed: Vec<SpeedResult>,
    /// Engine B — the ladder sweep curve (knee + envelope derived from it).
    pub concurrency: Option<SweepResult>,
    /// Engine C1 — the NIAH matrix.
    pub niah: Option<NiahResult>,
    /// Engine C2 — the deterministic reasoning score + speed.
    pub reasoning: Option<ReasoningResult>,
    /// Engine C3 — the structured-output penalty + compliance.
    pub structured: Option<StructuredResult>,
}

impl RunReport {
    /// A human-readable one-line summary of what ran (for logs / the TUI).
    pub fn summary_line(&self) -> String {
        let mut parts = Vec::new();
        if !self.speed.is_empty() {
            parts.push(format!("A: {} run(s)", self.speed.len()));
        }
        if let Some(s) = &self.concurrency {
            parts.push(format!("B: {} level(s)", s.levels.len()));
        }
        if let Some(n) = &self.niah {
            parts.push(format!("C1: {}", n.accuracy_label()));
        }
        if let Some(r) = &self.reasoning {
            parts.push(format!("C2: {}", r.score.label()));
        }
        if let Some(s) = &self.structured {
            parts.push(format!(
                "C3: {:+.1}% penalty, compliant={}",
                s.penalty_pct, s.compliant
            ));
        }
        if parts.is_empty() {
            "no engines selected".to_string()
        } else {
            parts.join(" · ")
        }
    }
}

/// Run the engines selected in `cfg.engines` (plan Chunk 18: "a single
/// command/run can trigger any combination" of A/B/C/D).
///
/// Engines run **sequentially** (one at a time) so each measures the
/// endpoint alone — no queueing cross-talk — and a failure in one engine
/// degrades gracefully (its field stays empty) without aborting the rest.
/// This is the headless / one-shot orchestration; the TUI instead spawns
/// the selected engines as independent background tasks, each publishing to
/// its own [`ResultSlot`] (see `App::start_run`).
pub async fn run_selected(cfg: &Config) -> RunReport {
    let sel = cfg.engines;
    let mut report = RunReport::default();

    if sel.speed {
        if let Ok(engine) = SpeedEngine::new(cfg) {
            let (_, results) = engine.run().await;
            report.speed = results;
        }
    }

    if sel.concurrency {
        if let Some(sweep) = build_sweep(cfg, None) {
            report.concurrency = Some(sweep.run().await);
        }
    }

    if sel.niah {
        if let Ok(engine) = NiahEngine::new(cfg) {
            report.niah = Some(engine.run().await);
        }
    }

    if sel.reasoning {
        if let Ok(engine) = ReasoningEngine::new(cfg) {
            report.reasoning = Some(engine.run().await);
        }
    }

    if sel.structured {
        if let Ok(engine) = StructuredEngine::new(cfg) {
            report.structured = Some(engine.run().await);
        }
    }

    report
}

/// Build an Engine B [`Sweep`] from a config (shared client, the
/// configured ladder, and the configured prompt). `metrics` (optional)
/// wires the sweep to a [`MetricsState`] so it publishes live snapshots to
/// the TUI while running (blueprint §4.2). `None` when the HTTP client (or
/// an explicit tokenizer) cannot be built — the engine simply doesn't run
/// (graceful degradation, never a panic).
pub fn build_sweep(cfg: &Config, metrics: Option<Arc<MetricsState>>) -> Option<Sweep> {
    use crate::client::pool::WorkerPool;
    use std::time::Duration;

    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
        .build()
        .ok()?;
    // Engine B reuses the configured prompt mode/tokens for its workload.
    let generator = match &cfg.tokenizer {
        Some(path) => crate::prompt::PromptGenerator::new(Some(
            crate::prompt::Tokenizer::from_file(path).ok()?,
        )),
        None => crate::prompt::PromptGenerator::new(None),
    };
    let prompt = match cfg.mode {
        crate::config::Mode::Short => generator.short(),
        crate::config::Mode::Long => generator.long(cfg.tokens),
    };
    let pool = WorkerPool::new(
        client,
        &cfg.url,
        &cfg.model,
        &prompt.text,
        speed::MAX_GEN_TOKENS,
    )
    .read_timeout(Duration::from_secs(cfg.timeout.max(1)));
    let pool = match &cfg.api_key {
        Some(key) => pool.api_key(key.clone()),
        None => pool,
    };
    let mut sweep = Sweep::new(pool, cfg.ladder.clone());
    if let Some(state) = metrics {
        sweep = sweep.metrics(state);
    }
    Some(sweep)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Mode;

    fn cfg_with(sel: EngineSelection) -> Config {
        Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "test".to_string(),
            mode: Mode::Short,
            engines: sel,
            ..Config::default()
        }
    }

    #[test]
    fn result_slot_starts_empty_and_publishes() {
        let slot: ResultSlot<u64> = ResultSlot::new();
        assert!(slot.load().is_none());
        assert!(!slot.is_running());

        slot.set_running(true);
        assert!(slot.is_running());
        slot.store(42);
        slot.set_running(false);

        assert!(!slot.is_running());
        match &*slot.load() {
            Some(v) => assert_eq!(*v, 42),
            None => panic!("expected a stored value"),
        }
    }

    #[test]
    fn result_slot_is_lock_free_readable_while_running() {
        let slot: Arc<ResultSlot<String>> = Arc::new(ResultSlot::new());
        let reader = slot.clone();
        slot.set_running(true);
        // A reader can load (None) without blocking while a run is marked
        // in progress — the measurement-isolation contract.
        assert!(reader.load().is_none());
        slot.store("done".to_string());
        assert_eq!(reader.load().as_deref(), Some("done"));
    }

    #[test]
    fn run_report_summary_line_names_the_engines_that_ran() {
        let empty = RunReport::default();
        assert_eq!(empty.summary_line(), "no engines selected");

        let r = RunReport {
            speed: vec![SpeedResult {
                ttft: 0.1,
                prompt_tokens: 100,
                completion_tokens: 10,
                pp_speed: 1000.0,
                tg_speed: 10.0,
                mtp_efficiency: 1.0,
                stream_time: 1.0,
                total_chunks: 10,
                content_chunks: 10,
                reasoning_chunks: 0,
                other_chunks: 0,
                estimated: false,
                model: "m".into(),
                mode: "short".into(),
                error: None,
            }],
            ..Default::default()
        };
        let line = r.summary_line();
        assert!(line.contains("A: 1 run(s)"), "{line}");
        assert!(!line.contains("no engines"), "{line}");
    }

    #[test]
    fn run_report_summary_includes_capability_scores() {
        let r = RunReport {
            reasoning: Some(ReasoningResult {
                responses: vec![],
                ttfts: vec![],
                tg_speeds: vec![],
                score: ReasoningScore {
                    total: 13,
                    solved: 9,
                    by_category: [(4, 5), (3, 5), (2, 3)],
                },
            }),
            structured: Some(StructuredResult {
                free_tps: 100.0,
                constrained_tps: 70.0,
                penalty_pct: 30.0,
                free_ttft: 0.1,
                constrained_ttft: 0.12,
                compliant: true,
                constrained_body: "{}".into(),
                free_body: "hi".into(),
            }),
            ..Default::default()
        };
        let line = r.summary_line();
        assert!(line.contains("C2: 9/13 solved"), "{line}");
        assert!(line.contains("C3: +30.0% penalty"), "{line}");
    }

    #[test]
    fn selection_drives_which_engines_run() {
        // A selection with only Engine C2 set: the report shape reflects
        // that only the reasoning field would be populated (we assert the
        // selection logic, not the live network run — that is the Chunk 19
        // e2e concern).
        let sel = EngineSelection {
            speed: false,
            concurrency: false,
            niah: false,
            reasoning: true,
            structured: false,
            hardware: false,
        };
        assert_eq!(sel.count(), 1);
        assert!(sel.reasoning);
        let _ = cfg_with(sel); // config carries the selection
    }
}
