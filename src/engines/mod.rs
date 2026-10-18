//! The seven benchmark engines: A (speed), B (concurrency), C1/C2/C3
//! (capability), D (hardware), and F (flat out).
//!
//! * [`speed`] — Engine A (Chunk 7): single-stream TTFT / PP / TG / MTP
//!   orchestration; the headless result box / `--json` report / summary
//!   (parity with `llmspeedtest.py`).
//! * [`concurrency`] — Engine B (Chunk 10 + 11): the multi-stream ladder
//!   sweep (default `1→2→3→4→8→12→16→24→32`), collecting per-level
//!   aggregate tokens/sec and client-perceived p90 TPOT across all
//!   concurrent streams, plus saturation knee-point detection, the
//!   per-stream usability profile, and the "Optimal Operational Envelope"
//!   (the recommended sweet spot).
//! * [`capability`] — Engine C (Chunk 15 + 16): C1 needle-in-a-haystack
//!   context retention, C2 deterministic reasoning/code verification,
//!   and C3 structured-output/JSON-grammar compliance.
//! * [`hardware`] — Engine D (Chunk 17): the silicon-efficiency profiler —
//!   `Joules/Token = ∫P(t)dt / Total_Generated_Tokens` over the 100 ms
//!   hardware power trace, plus the VRAM fragmentation warning.
//! * [`flatout`] — Engine F: real-world maximum throughput at the
//!   sweet-spot concurrency — `n` concurrent streams (Engine B's
//!   recommendation) for one 60-second window, all at once.
//!
//! **Chunk 18 — full integration:** [`ResultSlot`] is the lock-free
//! `ArcSwap` double-buffer seam the TUI uses to read a completed engine
//! result without ever touching the timing path (measurement-isolation
//! invariant, blueprint §4).
//!
//! **One orchestrator per mode (R1, closes the dead-code finding DO-07):**
//! there is exactly *one* code path that runs "this subset of engines"
//! per mode, so they cannot drift:
//!
//! * **Headless** — `main::run_headless` runs the selected engines
//!   sequentially in the foreground (Engine A gated on the selection, the
//!   additional engines B/C1/C2/C3/F gated on it, Engine D as the
//!   background poller).
//! * **TUI** — [`sequence::BenchmarkSequence`] runs the selected engines
//!   one at a time, each publishing to its lock-free [`ResultSlot`].
//!
//! The earlier standalone `run_selected`/`RunReport` helper (a third,
//! divergent runner that was never called) was removed: it duplicated the
//! headless block without the `--json` / logger / sweet-spot wiring, so
//! keeping it only invited drift.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use arc_swap::ArcSwap;

use crate::config::Config;
use crate::metrics::state::MetricsState;

pub mod capability;
pub mod concurrency;
pub mod flatout;
pub mod hardware;
pub mod sequence;
pub mod speed;

pub use capability::{
    build_document, classify, context_size_label, evaluate_case, human_size, is_json_compliant,
    score_responses, CaseCheck, CaseVerdict, Challenge, Checker, Needle, NiahCell, NiahCellState,
    NiahDocument, NiahEngine, NiahEngineConfig, NiahResult, NiahSlot, ReasoningEngine,
    ReasoningResult, ReasoningScore, StructuredCase, StructuredCaseResult, StructuredEngine,
    StructuredResult, NIAH_DEPTHS, NIAH_MAX_GEN_TOKENS, NIAH_SIZES, PREFILL_THROTTLE_FACTOR,
    REASONING_BANK, REASONING_MAX_GEN_TOKENS, STRUCTURED_CASES, STRUCTURED_MAX_GEN_TOKENS,
};
pub use concurrency::{
    normalize_ladder, recommended_streams_for, ConcurrencyMatrix, Envelope, KneePoint, MatrixCell,
    Sweep, SweepLevel, SweepResult, UsabilityProfile, DEFAULT_LADDER, IDEAL_PER_STREAM_TPS,
    KNEE_GAIN_THRESHOLD, KNEE_SPIKE_THRESHOLD, PRACTICAL_PER_STREAM_TPS, SWEET_SPOT_MARGIN,
    USABLE_PER_STREAM_TPS,
};
pub use flatout::{
    flatout_streams_for, FlatOutEngine, FlatOutResult, StreamCountSource, DEFAULT_STREAM_COUNT,
    MAX_TOKENS, MINIMAL_PROMPT, WINDOW_SECS,
};
pub use hardware::{
    fragmentation_warning, integrate_joules, joules_per_token, profile, EnergyResult,
    VRAM_FRAGMENTATION_THRESHOLD,
};
pub use sequence::{
    queue_for, summarize_sweep, BenchmarkSequence, Engine, EngineProgress, ProgressBus, RunPause,
    RunSlots, SeqPhase, SeqState, SeqStateSlot,
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
//
// The "run this subset" logic lives in the two mode-specific orchestrators
// (see the module docs): `main::run_headless` (headless) and
// `sequence::BenchmarkSequence` (TUI). There is deliberately *no* third
// general-purpose runner here — the earlier `run_selected`/`RunReport`
// helper was dead code that had drifted from both, and was removed
// (DO-07). `build_sweep` below is the one shared Engine B constructor both
// orchestrators use.

/// Build an Engine B [`Sweep`] from a config (shared client, the
/// configured ladder, and the configured prompt). `metrics` (optional)
/// wires the sweep to a [`MetricsState`] so it publishes live snapshots to
/// the TUI while running (blueprint §4.2). `logger` (optional) wires the
/// run logger so the sweep records its per-step lifecycle. `None` when the
/// HTTP client (or an explicit tokenizer) cannot be built — the engine
/// simply doesn't run (graceful degradation, never a panic).
///
/// **The freeze fix is wired here:** every worker gets the configured
/// `timeout` as its max lifetime (`worker_timeout`), so a hung
/// connection can never hold a sweep level open forever.
pub fn build_sweep(
    cfg: &Config,
    metrics: Option<Arc<MetricsState>>,
    logger: Option<Arc<crate::log::RunLogger>>,
) -> Option<Sweep> {
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
    // CF-13: a shared tokenizer for the pool's per-stream token counts —
    // when the server omits a terminal `usage` frame, the sweep re-counts
    // the text with it (authoritative) instead of falling to a silent
    // `chars/4` estimate (flagged `estimated`). Loaded independently of the
    // generator (which consumes an owned `Tokenizer`); the `?` in the
    // generator construction above guarantees the file is valid here.
    let pool_tokenizer: Option<Arc<crate::prompt::Tokenizer>> = match &cfg.tokenizer {
        Some(path) => crate::prompt::Tokenizer::from_file(path).ok().map(Arc::new),
        None => None,
    };
    let pool = WorkerPool::new(
        client,
        &cfg.url,
        &cfg.model,
        &prompt.text,
        speed::MAX_GEN_TOKENS,
    )
    .read_timeout(Duration::from_secs(cfg.timeout.max(1)))
    // Per-worker max lifetime (CF-28): scaled with the *expected*
    // generation time (the configured timeout as a floor, grown with
    // `max_tokens`) so a slow-but-healthy long-context stream is not
    // killed mid-generation — a genuinely hung stream is still bounded,
    // and the stall detector distinguishes it from a slow one. This is
    // what prevents a hung connection from freezing the sweep.
    .worker_timeout(WorkerPool::scaled_worker_timeout(
        Duration::from_secs(cfg.timeout.max(1)),
        speed::MAX_GEN_TOKENS,
    ))
    // CF-13: thread the optional tokenizer into the pool so per-stream
    // token counts can be authoritative.
    .with_tokenizer(pool_tokenizer);
    let pool = match &cfg.api_key {
        Some(key) => pool.api_key(key.clone()),
        None => pool,
    };
    let pool = match &logger {
        Some(l) => pool.logger(l.clone()),
        None => pool,
    };
    let mut sweep = Sweep::new(pool, cfg.ladder.clone());
    // v0.1.1 2D concurrency × context matrix: with more than one context
    // size the sweep runs one ladder per context and collects the grid.
    // A single context (including `[0]`) keeps the classic 1D sweep.
    //
    // CF-27: the recommended sweet spot is pinned to the **zero-context**
    // leg (`context == 0`) regardless of its position in `matrix_contexts`
    // (see `Sweep::run_matrix`), so reordering the contexts can no longer
    // silently change the headline decode measurement.
    if cfg.matrix_contexts.len() > 1 {
        sweep = sweep.with_matrix(
            Some(generator),
            cfg.matrix_contexts.clone(),
            prompt.token_count,
        );
    } else {
        sweep = sweep.with_matrix(None, Vec::new(), prompt.token_count);
    }
    if let Some(state) = metrics {
        sweep = sweep.metrics(state);
    }
    if let Some(l) = logger {
        sweep = sweep.logger(l);
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
            flatout: false,
        };
        assert_eq!(sel.count(), 1);
        assert!(sel.reasoning);
        let _ = cfg_with(sel); // config carries the selection
    }
}
