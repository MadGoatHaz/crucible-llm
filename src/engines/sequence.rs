//! The **Benchmark Sequence** executor: runs the selected engines strictly
//! **one at a time** (A → B → C1 → C2 → C3 → D → F) and publishes a
//! lock-free progress state that the TUI reads every frame.
//!
//! This is the TUI-side orchestration (the headless path keeps its own
//! sequential order in `main::run_headless`). The sequence is a
//! `tokio::spawn`ed task that:
//!
//! 1. walks the ordered engine queue (only the engines the user selected
//!    in the Configuration form / setup flow);
//! 2. runs each engine to completion before starting the next — one
//!    engine hits the endpoint at a time, no queueing cross-talk;
//! 3. mirrors each engine's [`EngineProgress`] (published to the
//!    [`ProgressBus`] by the engine's own run loop) into a
//!    [`SeqState`] on the [`SeqStateSlot`] at a steady 10 Hz cadence —
//!    the render loop reads it lock-free (measurement-isolation
//!    invariant, blueprint §4);
//! 4. publishes each engine's result to the existing lock-free result
//!    slots (so Views 2/3 keep working) and shows a brief summary between
//!    engines;
//! 5. streams real events to the App's log via an `mpsc` channel (the
//!    App drains it on its tick path).
//!
//! Engine D (hardware/energy) is continuous telemetry, not a one-shot:
//! when selected, the sequence runs a short final sampling window after
//! the one-shot engines and folds the power trace into the
//! Silicon Efficiency Metric (Joules/Token, `hardware::profile`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use arc_swap::ArcSwapOption;

use crate::config::Config;
use crate::engines::capability::{
    NiahEngine, NiahResult, ReasoningEngine, ReasoningResult, StructuredEngine, StructuredResult,
    NIAH_MAX_GEN_TOKENS,
};
use crate::engines::concurrency::SweepResult;
use crate::engines::flatout::StreamCountSource;
use crate::engines::flatout::{FlatOutEngine, FlatOutResult, DEFAULT_STREAM_COUNT, WINDOW_SECS};
use crate::engines::hardware::{profile, EnergyResult};
use crate::engines::speed::{SpeedEngine, SpeedResult};
use crate::engines::{build_sweep, NiahSlot, ResultSlot};
use crate::hw::{HwPoller, IDLE_WINDOW_SECS};
use crate::log::{Context, RunLogger};
use crate::metrics::state::MetricsState;
use crate::storage::db::Database;
use crate::storage::models::{
    BenchmarkSession, ConcurrencySummary, EnergySummary, EngineResults, FlatOutSummary,
    NeedleEvaluation, ReasoningSummary, StreamMetricRow, StructuredSummary,
};

/// The cooperative pause gate (the `Space` key, TUI): shared between the
/// App's key path and every engine run loop.
///
/// While set, each engine's run loop yields (100 ms poll) **before
/// spawning the next request unit** — the in-flight stream(s) complete
/// normally, and no new requests hit the endpoint. Clearing the gate
/// resumes exactly where the loop left off (the iteration/level/cell
/// counter is untouched — no state is corrupted by a pause).
///
/// Measurement-isolation note (blueprint §4): the gate is *polled by the
/// engine tasks*, never by the render loop or the tick path — a pause
/// can never perturb the quanta timing path.
#[derive(Debug, Default)]
pub struct RunPause {
    inner: AtomicBool,
}

impl RunPause {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the pause state (key path: `Space` toggles it).
    pub fn set(&self, paused: bool) {
        self.inner.store(paused, Ordering::Relaxed);
    }

    /// `true` while the user has paused the run.
    pub fn is_paused(&self) -> bool {
        self.inner.load(Ordering::Relaxed)
    }

    /// Yield until the gate is cleared. Called by each engine run loop
    /// immediately before spawning a new request unit (iteration / ladder
    /// level / matrix cell / challenge / structured run).
    pub async fn wait_while_paused(&self) {
        while self.is_paused() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
}

/// The six benchmark engines (blueprint §5) in the canonical run order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Engine {
    /// Engine A — Speed & Latency (single-stream iterations).
    Speed,
    /// Engine B — Concurrency & Saturation sweep.
    Concurrency,
    /// Engine C1 — Needle-in-a-Haystack matrix.
    Niah,
    /// Engine C2 — Deterministic reasoning / code verification.
    Reasoning,
    /// Engine C3 — Structured output / JSON-grammar compliance.
    Structured,
    /// Engine D — Hardware & Energy profiler (continuous sampling).
    Hardware,
    /// Engine F — Flat Out (sustained max-speed, one continuous 60s
    /// stream).
    FlatOut,
}

impl Engine {
    /// The queue-panel label (e.g. `A: Speed`).
    pub fn label(self) -> &'static str {
        match self {
            Engine::Speed => "A: Speed",
            Engine::Concurrency => "B: Concurrency",
            Engine::Niah => "C1: NIAH",
            Engine::Reasoning => "C2: Reasoning",
            Engine::Structured => "C3: Structured",
            Engine::Hardware => "D: Energy",
            Engine::FlatOut => "F: Flat Out",
        }
    }

    /// The short uppercase name for the header (e.g. `SPEED`).
    pub fn name(self) -> &'static str {
        match self {
            Engine::Speed => "SPEED",
            Engine::Concurrency => "CONCURRENCY",
            Engine::Niah => "NIAH",
            Engine::Reasoning => "REASONING",
            Engine::Structured => "STRUCTURED",
            Engine::Hardware => "ENERGY",
            Engine::FlatOut => "FLAT OUT",
        }
    }

    /// The full header title (e.g. `ENGINE A: SPEED`).
    pub fn title(self) -> &'static str {
        match self {
            Engine::Speed => "ENGINE A: SPEED",
            Engine::Concurrency => "ENGINE B: CONCURRENCY",
            Engine::Niah => "ENGINE C1: NIAH",
            Engine::Reasoning => "ENGINE C2: REASONING",
            Engine::Structured => "ENGINE C3: STRUCTURED",
            Engine::Hardware => "ENGINE D: ENERGY",
            Engine::FlatOut => "ENGINE F: FLAT OUT",
        }
    }

    /// All six engines in the canonical run order (the default queue
    /// shown before a run starts).
    pub const ALL: [Engine; 7] = [
        Engine::Speed,
        Engine::Concurrency,
        Engine::Niah,
        Engine::Reasoning,
        Engine::Structured,
        Engine::Hardware,
        Engine::FlatOut,
    ];

    /// A concise, user-facing description of what this engine measures.
    ///
    /// The TUI shows this under an `ℹ` marker (dimmed) in the setup /
    /// config engine-selection phase and beside each engine's results, so
    /// users can choose engines knowingly and interpret the numbers.
    ///
    /// Kept to at most three lines, each pre-wrapped at ≤58 columns, so it
    /// fits a terminal panel without crowding the data.
    pub fn description(self) -> &'static str {
        match self {
            Engine::Speed => {
                "Single-stream throughput. Measures tokens/sec and\nTTFT (time to first token) for one user — how fast\nthe model generates text in isolation."
            }
            Engine::Concurrency => {
                "Multi-stream sweep (1→2→3→4→8→12→16→24→32) finds\nwhere per-user speed degrades: the sweet spot keeps ≥30\nt/s per user (2% margin) — how many users you can serve."
            }
            Engine::Niah => {
                "Long-context retrieval. Hides a fact in a 2k–128k\ndocument and asks the model to find it — measures\ncontext retention for RAG / document QA."
            }
            Engine::Reasoning => {
                "Logical reasoning accuracy. 13 deterministic challenges\n(math, logic, code) — measures problem-solving\nindependent of speed."
            }
            Engine::Structured => {
                "JSON compliance. Tests whether the model follows\nthe response_format instruction — reliability for API\nand agent tool-calling."
            }
            Engine::Hardware => {
                "GPU power profiling (watts, J/token, $/1M\ntokens). MUST run on the GPU box — NVIDIA\n(NVML)/AMD/Intel, multi-GPU. Remote users: N/A"
            }
            Engine::FlatOut => {
                "Real-world maximum throughput. Runs at your server's\nsweet-spot concurrency for 60 seconds. The number to\nquote when comparing setups."
            }
        }
    }
}

/// The progress of the *current* engine, as reported by that engine's own
/// run loop (the TUI renders this as the header text + progress bar).
#[derive(Debug, Clone, PartialEq)]
pub enum EngineProgress {
    /// Engine A: `Iteration {current}/{total}` + tokens generated so far.
    Speed {
        iteration: usize,
        total: usize,
        tokens: u64,
    },
    /// Engine B: `Concurrency level {current} (step {n}/{total_steps})`
    /// + streams active.
    Concurrency {
        level: usize,
        step: usize,
        total_steps: usize,
        active: usize,
    },
    /// Engine C1: `Size {size}, Depth {d}/{total_depths}` (cell `c` of the
    /// full matrix).
    Niah {
        size: u32,
        depth: usize,
        total_depths: usize,
        cell: usize,
        total_cells: usize,
    },
    /// Engine C2: `Challenge {n}/{total}`.
    Reasoning { challenge: usize, total: usize },
    /// Engine C3: `Run {n}/{total}` (free-form → constrained).
    Structured { run: usize, total: usize },
    /// Engine D: `Sampling... {elapsed}s`.
    Sampling { elapsed: f64 },
    /// Engine F: `{elapsed}s / 60s — {streams} streams — {tokens} tok
    /// ({tps} t/s aggregate)` (the sweet-spot full-load run).
    FlatOut {
        elapsed_secs: f64,
        streams: usize,
        tokens: u64,
        tps: f64,
    },
}

impl EngineProgress {
    /// `0.0..=1.0` completion of the current engine (the progress bar).
    ///
    /// The convention: *completed* units over total (so `Iteration 1/5`
    /// starts at 0.0); the final 100% comes from the sequence's `Complete`
    /// state, not the engine.
    pub fn fraction(&self) -> f64 {
        match self {
            EngineProgress::Speed {
                iteration, total, ..
            } => {
                let (iteration, total) = (*iteration, *total);
                iteration.saturating_sub(1).min(total) as f64 / total.max(1) as f64
            }
            EngineProgress::Concurrency {
                step,
                total_steps,
                active,
                level,
            } => {
                let (step, total_steps, active, level) = (*step, *total_steps, *active, *level);
                let done = step.saturating_sub(1).min(total_steps) as f64;
                let within = if level > 0 {
                    (active as f64 / level as f64) * 0.5
                } else {
                    0.0
                };
                ((done + within.min(1.0)) / total_steps.max(1) as f64).clamp(0.0, 1.0)
            }
            EngineProgress::Niah {
                cell, total_cells, ..
            } => {
                let (cell, total_cells) = (*cell, *total_cells);
                cell.saturating_sub(1) as f64 / total_cells.max(1) as f64
            }
            EngineProgress::Reasoning { challenge, total } => {
                let (challenge, total) = (*challenge, *total);
                challenge.saturating_sub(1) as f64 / total.max(1) as f64
            }
            EngineProgress::Structured { run, total } => {
                let (run, total) = (*run, *total);
                run.saturating_sub(1) as f64 / total.max(1) as f64
            }
            // Engine D: no known end — an indeterminate (0.0) bar; the
            // elapsed seconds carry the "alive" signal.
            EngineProgress::Sampling { .. } => 0.0,
            // Engine F: elapsed time over the 60s window.
            EngineProgress::FlatOut { elapsed_secs, .. } => {
                (*elapsed_secs / WINDOW_SECS).clamp(0.0, 1.0)
            }
        }
    }

    /// The human-readable progress line for the header (the
    /// `— Iteration 2/5` part of `ENGINE A: SPEED — Running — …`).
    pub fn label(&self) -> String {
        match self {
            EngineProgress::Speed {
                iteration,
                total,
                tokens,
            } => format!("Iteration {iteration}/{total} · {tokens} tok"),
            EngineProgress::Concurrency {
                level,
                step,
                total_steps,
                active,
            } => format!("Concurrency level {level} (step {step}/{total_steps}) · {active} active"),
            EngineProgress::Niah {
                size,
                depth,
                total_depths,
                ..
            } => format!("Size {}k, Depth {depth}/{total_depths}", size / 1000),
            EngineProgress::Reasoning { challenge, total } => {
                format!("Challenge {challenge}/{total}")
            }
            EngineProgress::Structured { run, total } => format!("Run {run}/{total}"),
            EngineProgress::Sampling { elapsed } => format!("Sampling… {elapsed:.1}s"),
            EngineProgress::FlatOut {
                elapsed_secs,
                streams,
                tokens,
                tps,
            } => format!(
                "{elapsed_secs:.0}s / {WINDOW_SECS:.0}s — {streams} streams — {} tok ({tps:.1} t/s)",
                thousands(*tokens)
            ),
        }
    }
}

/// The overall sequence phase (the TUI's state machine).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeqPhase {
    /// No run has started (or the queue is empty).
    Idle,
    /// An engine is running (the progress bar is live).
    Running,
    /// The current engine just finished — its summary is shown briefly
    /// before the next one starts.
    Complete,
    /// Every selected engine has finished.
    AllComplete,
}

impl SeqPhase {
    pub fn label(self) -> &'static str {
        match self {
            SeqPhase::Idle => "Idle",
            SeqPhase::Running => "Running",
            SeqPhase::Complete => "Complete",
            SeqPhase::AllComplete => "All Complete",
        }
    }
}

/// One published state of the benchmark sequence — the *only* data the
/// Live view's sequence header / progress bar / queue panel read.
///
/// Plain `Clone` data, no interior mutability: a fresh `Arc<SeqState>` is
/// swapped in atomically (the same double-buffer pattern as the Chunk 6
/// metrics snapshot), so the render loop never blocks.
#[derive(Debug, Clone)]
pub struct SeqState {
    /// The current phase.
    pub phase: SeqPhase,
    /// The ordered engine queue for this run (the selected engines).
    pub queue: Vec<Engine>,
    /// The engine the phase refers to (the running / just-completed one).
    pub engine: Engine,
    /// The current engine's live progress (`None` while the phase is
    /// `Idle` / between the bus reset and the first publish).
    pub progress: Option<EngineProgress>,
    /// The summary line of the current (or overall, when `AllComplete`)
    /// result — shown in the header on `Complete` / `AllComplete`.
    pub summary: String,
    /// The completed engines so far, in run order, with their summary
    /// lines (the queue panel's `✓` rows).
    pub completed: Vec<(Engine, String)>,
    /// Unix milliseconds when the *current* engine started (`0` when no
    /// engine is running): the Live view's key-metrics `Duration` row
    /// renders `now − engine_started_ms` (a plain wall-clock read on the
    /// render path — never the quanta timing path).
    pub engine_started_ms: u64,
}

impl SeqState {
    /// The index of `engine` in the queue (`None` when not queued).
    pub fn queue_index(&self, engine: Engine) -> Option<usize> {
        self.queue.iter().position(|e| *e == engine)
    }
}

/// Lock-free holder for the current [`SeqState`] (the TUI seam).
///
/// The sequence executor (a background `tokio` task) *publishes* via
/// [`store`]; the render loop *reads* via [`load`] — never blocking, never
/// touching the quanta timing path (measurement-isolation invariant,
/// blueprint §4). The `running` flag is set synchronously by the key path
/// so re-pressing `r` mid-sequence is a no-op.
#[derive(Debug)]
pub struct SeqStateSlot {
    inner: ArcSwapOption<SeqState>,
    running: AtomicBool,
}

impl SeqStateSlot {
    pub fn new() -> Self {
        Self {
            inner: ArcSwapOption::empty(),
            running: AtomicBool::new(false),
        }
    }

    /// Lock-free read (`None` until the first sequence starts).
    pub fn load(&self) -> Option<Arc<SeqState>> {
        self.inner.load_full()
    }

    /// Publish a new sequence state.
    pub fn store(&self, state: SeqState) {
        self.inner.store(Some(Arc::new(state)));
    }

    /// Mark the sequence in progress (key path) / finished (executor).
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// `true` while a sequence is in progress.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }
}

impl Default for SeqStateSlot {
    fn default() -> Self {
        Self::new()
    }
}

/// The per-engine progress channel: the engine's run loop publishes
/// [`EngineProgress`] here as work advances; the sequence's 10 Hz ticker
/// mirrors it into the [`SeqStateSlot`] (and the TUI reads *that*
/// lock-free). Kept separate from the result slot so a progress publish
/// never has to build (or stall on) a full result.
#[derive(Debug)]
pub struct ProgressBus {
    inner: ArcSwapOption<EngineProgress>,
}

impl ProgressBus {
    pub fn new() -> Self {
        Self {
            inner: ArcSwapOption::empty(),
        }
    }

    /// Publish the current progress (the engine's hot path — a single
    /// atomic store, no allocation churn beyond the small enum).
    pub fn publish(&self, progress: EngineProgress) {
        self.inner.store(Some(Arc::new(progress)));
    }

    /// Lock-free read (`None` until the first publish / after a reset).
    pub fn load(&self) -> Option<Arc<EngineProgress>> {
        self.inner.load_full()
    }

    /// Clear the bus (the executor calls this between engines so the
    /// ticker does not publish a stale engine's progress).
    pub fn reset(&self) {
        self.inner.store(None);
    }
}

impl Default for ProgressBus {
    fn default() -> Self {
        Self::new()
    }
}

/// The existing lock-free result slots the sequence publishes completed
/// engine results into (Views 2/3 read these exactly as before).
#[derive(Debug)]
pub struct RunSlots {
    pub speed: Arc<ResultSlot<Vec<SpeedResult>>>,
    pub concurrency: Arc<ResultSlot<SweepResult>>,
    pub niah: Arc<NiahSlot>,
    pub reasoning: Arc<ResultSlot<ReasoningResult>>,
    pub structured: Arc<ResultSlot<StructuredResult>>,
    pub flatout: Arc<ResultSlot<FlatOutResult>>,
}

/// The sequential benchmark executor (one engine at a time).
///
/// Built from the resolved [`Config`] (the engine selection comes from
/// `cfg.engines`) plus the shared TUI seams: the metrics snapshot
/// publisher, the hardware poller (Engine D), the sequence state slot,
/// the progress bus, the result slots, and the App log channel.
pub struct BenchmarkSequence {
    cfg: Config,
    metrics: Arc<MetricsState>,
    hw: Option<Arc<Mutex<HwPoller>>>,
    slot: Arc<SeqStateSlot>,
    bus: Arc<ProgressBus>,
    slots: RunSlots,
    /// The App log pipe (the executor sends, the App drains on tick).
    log: Option<mpsc::Sender<String>>,
    /// The `Space`-key pause gate shared with every engine (the engines
    /// wait on it before spawning each new request unit).
    pause: Arc<RunPause>,
    /// The file-based run logger (the executor records engine
    /// transitions; a `disabled` logger makes this a no-op).
    logger: Arc<RunLogger>,
    /// R2 (traceability): this run's session UUID — the same id the
    /// run-log carries, so the persisted `benchmark_sessions` row joins to
    /// its log.
    session_id: String,
    /// The ordered engine queue (selection-filtered, canonical order).
    engines: Vec<Engine>,
    /// The queue index the ticker is mirroring (the current engine).
    current: Arc<AtomicUsize>,
    /// Completed engines so far (the queue panel's `✓` rows) — a brief
    /// lock on the completion path only, never the timing path.
    completed: Arc<Mutex<Vec<(Engine, String)>>>,
}

/// The ordered engine queue for a selection: the selected engines in the
/// canonical A → B → C1 → C2 → C3 → D order (an empty selection yields an
/// empty queue).
pub fn queue_for(sel: &crate::config::EngineSelection) -> Vec<Engine> {
    [
        (sel.speed, Engine::Speed),
        (sel.concurrency, Engine::Concurrency),
        (sel.niah, Engine::Niah),
        (sel.reasoning, Engine::Reasoning),
        (sel.structured, Engine::Structured),
        (sel.hardware, Engine::Hardware),
        (sel.flatout, Engine::FlatOut),
    ]
    .into_iter()
    .filter(|(on, _)| *on)
    .map(|(_, e)| e)
    .collect()
}

impl BenchmarkSequence {
    /// Build the executor for a run. The engine queue is the selected
    /// engines in the canonical A → B → C1 → C2 → C3 → D → F order; an
    /// empty selection yields an empty queue (the run is a no-op that logs
    /// a warning).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cfg: Config,
        metrics: Arc<MetricsState>,
        hw: Option<Arc<Mutex<HwPoller>>>,
        slot: Arc<SeqStateSlot>,
        bus: Arc<ProgressBus>,
        slots: RunSlots,
        log: Option<mpsc::Sender<String>>,
        pause: Arc<RunPause>,
        logger: Arc<RunLogger>,
        session_id: String,
    ) -> Self {
        let engines = queue_for(&cfg.engines);
        Self {
            cfg,
            metrics,
            hw,
            slot,
            bus,
            slots,
            log,
            pause,
            logger,
            session_id,
            engines,
            current: Arc::new(AtomicUsize::new(0)),
            completed: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// The ordered engine queue for this run.
    pub fn engines(&self) -> &[Engine] {
        &self.engines
    }

    /// Send one line to the App log (a dropped App is a no-op).
    fn log_line(&self, line: String) {
        if let Some(tx) = &self.log {
            let _ = tx.send(line);
        }
    }

    /// The current unix-millisecond wall clock (for the
    /// [`SeqState::engine_started_ms`] stamp — a display concern, never
    /// the quanta timing path).
    fn now_ms() -> u64 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    }

    /// The `engine_started_ms` to carry into a fresh `Running` state: the
    /// previous engine's start stamp survives (the ticker re-publishes the
    /// same engine many times), a new engine gets a fresh stamp, and
    /// non-running phases clear it.
    fn started_ms_for(&self, phase: SeqPhase, engine: Engine) -> u64 {
        if phase != SeqPhase::Running {
            return 0;
        }
        match self.slot.load() {
            Some(prev) if prev.engine == engine && prev.engine_started_ms > 0 => {
                prev.engine_started_ms
            }
            _ => Self::now_ms(),
        }
    }

    /// Store a `SeqState` built from the current shared pieces.
    fn publish_state(&self, phase: SeqPhase, engine: Engine, summary: &str) {
        let progress = if phase == SeqPhase::Running {
            self.bus.load().as_deref().cloned()
        } else {
            None
        };
        let completed = self.completed.lock().map(|g| g.clone()).unwrap_or_default();
        self.slot.store(SeqState {
            phase,
            queue: self.engines.clone(),
            engine,
            progress,
            summary: summary.to_string(),
            completed,
            engine_started_ms: self.started_ms_for(phase, engine),
        });
    }

    /// Run the whole sequence: one engine at a time, in order.
    ///
    /// A 10 Hz ticker task mirrors the [`ProgressBus`] into the
    /// [`SeqStateSlot`] while an engine runs (steady visual feedback);
    /// the main path stores the `Complete` / `AllComplete` states and
    /// resets the bus between engines so the ticker never shows a stale
    /// engine. Each engine's result is published to its lock-free result
    /// slot; a failed init degrades gracefully (a `N/A` summary) and the
    /// sequence continues.
    pub async fn run(self) {
        if self.engines.is_empty() {
            self.log_line("[seq] no engines selected — enable some in the Config view".into());
            self.logger
                .warn(Context::Sequence, "no engines selected — nothing to run");
            self.slot.set_running(false);
            return;
        }
        self.logger.info(
            Context::Sequence,
            format!(
                "sequence started — {} engine(s): {}",
                self.engines.len(),
                self.engines
                    .iter()
                    .map(|e| e.label())
                    .collect::<Vec<_>>()
                    .join(" → ")
            ),
        );

        // The 10 Hz ticker: mirror the progress bus into the state slot
        // while a run is active. It exits when the sequence finishes.
        let ticker_done = Arc::new(AtomicBool::new(false));
        {
            let bus = self.bus.clone();
            let slot = self.slot.clone();
            let engines = self.engines.clone();
            let current = self.current.clone();
            let completed = self.completed.clone();
            let done = ticker_done.clone();
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(Duration::from_millis(100));
                loop {
                    interval.tick().await;
                    if done.load(Ordering::Relaxed) {
                        break;
                    }
                    let Some(engine) = engines.get(current.load(Ordering::Relaxed)) else {
                        continue;
                    };
                    let progress = bus.load().as_deref().cloned();
                    let Ok(g) = completed.lock() else {
                        continue;
                    };
                    let completed = g.clone();
                    drop(g);
                    // The ticker re-publishes the *same* engine at 10 Hz:
                    // keep its start stamp (a new engine gets one from
                    // `publish_state` before the ticker catches up).
                    let started_ms = slot
                        .load()
                        .as_ref()
                        .filter(|p| p.engine == *engine && p.engine_started_ms > 0)
                        .map(|p| p.engine_started_ms)
                        .unwrap_or_else(Self::now_ms);
                    slot.store(SeqState {
                        phase: SeqPhase::Running,
                        queue: engines.clone(),
                        engine: *engine,
                        progress,
                        summary: String::new(),
                        completed,
                        engine_started_ms: started_ms,
                    });
                }
            });
        }
        // Arm the GPU & Power monitor (View 4) and measure the idle power
        // baseline *before* any load hits the GPU, so the "compute power"
        // (load − idle) and the 1 Hz power history both start from a real
        // no-load floor. No-op without a hardware poller (N/A rule).
        self.prepare_power_monitor().await;
        // The overall "Total Duration" clock anchors here: the first engine
        // begins *now* (after the idle baseline window), never at app open
        // (the forever 100 ms poller's ticks) and never at the `r` keypress
        // (5 s earlier, mid-idle-window).
        self.metrics.mark_run_started();

        for (i, engine) in self.engines.iter().enumerate() {
            self.current.store(i, Ordering::Relaxed);
            self.bus.reset();
            self.log_line(format!(
                "[seq] ▶ {} — starting (engine {} of {})",
                engine.label(),
                i + 1,
                self.engines.len()
            ));
            self.logger.info(
                Context::Sequence,
                format!(
                    "▶ {} — starting (engine {} of {})",
                    engine.label(),
                    i + 1,
                    self.engines.len()
                ),
            );
            self.publish_state(SeqPhase::Running, *engine, "");

            // Mark the engine's result slot running (the views show the
            // in-progress state while its engine is the one on the bench).
            let summary = match engine {
                Engine::Speed => self.run_speed().await,
                Engine::Concurrency => self.run_concurrency().await,
                Engine::Niah => self.run_niah().await,
                Engine::Reasoning => self.run_reasoning().await,
                Engine::Structured => self.run_structured().await,
                // Engine D samples the whole-run power trace; its J/token
                // denominator spans every engine that has produced tokens by
                // this point (A+B+C1+C2+C3). Engine F runs *after* D, so its
                // tokens are not yet known here — the whole-run figure that
                // includes F is computed in `persist_results` (CF-03/CF-54).
                Engine::Hardware => self.run_hardware().await,
                // Engine F (Flat Out) runs after D: the max decode speed finale.
                Engine::FlatOut => self.run_flatout().await,
            };

            self.bus.reset();
            {
                let mut g = self.completed.lock().unwrap_or_else(|e| e.into_inner());
                g.push((*engine, summary.clone()));
            }
            self.log_line(format!("[seq] ✓ {} — {summary}", engine.label()));
            self.logger.info(
                Context::Sequence,
                format!("✓ {} — {summary}", engine.label()),
            );
            self.publish_state(SeqPhase::Complete, *engine, &summary);

            // FIX 2: the last engine is done — close the monitor's load
            // window, publish that final state to the snapshot, and
            // *then* freeze the metrics pipeline, before the summary
            // hold. The 100 ms hardware poller (the monitor's only
            // publisher) would otherwise keep stamping samples and
            // growing `elapsed_sec` for the whole 2 s hold (and the
            // graph would keep animating), so the frozen state is the
            // true final state from this point on. The order matters:
            // once the pipeline is frozen the poller stops calling
            // `tick()` and `update()` is a no-op — a publish *after* the
            // freeze would never reach the UI, and View 4 would keep
            // showing the pre-`end_load` monitor (duration counting
            // forever, avg power decaying forever). (The close + publish
            // + freeze at `AllComplete` below is the idempotent safety
            // net.)
            if i + 1 == self.engines.len() {
                self.end_power_monitor();
                self.publish_power_monitor();
                self.metrics.freeze();
            }

            // Brief summary hold so the user sees the result before the
            // queue advances (the header shows `✓ … — {summary}`).
            tokio::time::sleep(Duration::from_secs(2)).await;
        }

        // A compact final line: the per-engine summaries already logged;
        // the header shows the count.
        let final_summary = format!(
            "{} of {} engines complete",
            self.completed.lock().map(|g| g.len()).unwrap_or(0),
            self.engines.len()
        );
        self.slot.set_running(false);
        ticker_done.store(true, Ordering::Relaxed);
        self.publish_state(
            SeqPhase::AllComplete,
            *self.engines.last().unwrap(),
            &final_summary,
        );
        // The run is over: close the monitor's load window, publish the
        // final monitor state, and freeze the metrics pipeline (all
        // idempotent — the last engine's `Complete` already did this
        // before the summary hold). The freeze stops the
        // forever-running 100 ms hardware poller (and any other writer)
        // from re-stamping `0.0` throughput samples / growing `elapsed`
        // / dragging the overall averages after the last engine
        // finishes — so the OVERALL METRICS panel and the throughput
        // graph hold their final values and stop animating. The publish
        // runs *before* the freeze (see `publish_power_monitor`): the
        // frozen pipeline would swallow it, and View 4 would keep
        // showing the pre-`end_load` monitor (duration counting
        // forever, avg power decaying forever).
        self.end_power_monitor();
        self.publish_power_monitor();
        self.metrics.freeze();
        self.log_line(format!("[seq] ✓ all benchmarks complete — {final_summary}"));
        self.logger.info(
            Context::Sequence,
            format!("sequence complete — {final_summary}"),
        );
        // Persist the completed run to SQLite (the History view reads
        // these rows). A storage failure degrades gracefully — it never
        // changes the run's outcome.
        self.persist_results();
    }

    /// Persist the completed run to the default SQLite database: one
    /// `benchmark_sessions` row plus `stream_metrics` rows (from Engine A
    /// speed results) and `needle_evaluations` rows (from Engine C1 NIAH
    /// results). A failure is logged but never propagated — the run is
    /// already complete and the TUI is showing its final state.
    fn persist_results(&self) {
        // R2 (CF-14): fold the non-stream engine results (B/C2/C3/D/F) into
        // the session's `engine_results` JSON column. A → `stream_metrics`
        // and C1 → `needle_evaluations` (below); these are the engines that
        // previously had *no* persistence at all.
        //
        // Engine D is continuous telemetry (no result slot), so its profile
        // is recomputed here from the poller's trace (the same N/A rule as
        // `run_hardware` / the headless path).
        let d_energy: Option<EnergyResult> =
            self.hw.as_ref().and_then(|h| h.lock().ok()).and_then(|g| {
                let trace = g.trace().to_vec();
                if trace.is_empty() {
                    return None;
                }
                // CF-03: the persisted (whole-run) J/token denominator spans
                // every engine that ran — by persist time all slots (A..F) are
                // filled, matching the full load window the numerator uses.
                let total_tokens = self.all_engines_tokens();
                let mut e = profile(&trace, None, total_tokens);
                if let Some(b) = g.backend() {
                    e.vendor = Some(b.vendor().to_string());
                    e.model = b.model().map(str::to_string);
                }
                // H3: the persisted numerator is the full load window, not the
                // bounded display trace.
                apply_load_window(&mut e, g.load_window_joules(), total_tokens);
                Some(e)
            });
        let engine_results = EngineResults {
            b_concurrency: (*self.slots.concurrency.load())
                .clone()
                .as_ref()
                .map(ConcurrencySummary::from_sweep),
            c2_reasoning: (*self.slots.reasoning.load())
                .clone()
                .as_ref()
                .map(ReasoningSummary::from_reasoning),
            c3_structured: (*self.slots.structured.load())
                .clone()
                .as_ref()
                .map(StructuredSummary::from_structured),
            d_energy: d_energy.as_ref().map(EnergySummary::from_energy),
            f_flatout: (*self.slots.flatout.load())
                .clone()
                .as_ref()
                .map(FlatOutSummary::from_flatout),
        }
        .to_json();

        // R2: the session id is the one shared with the run-log (stamped in
        // its header/archive name by `main`), so the stored row joins to it.
        let session = BenchmarkSession {
            session_id: self.session_id.clone(),
            timestamp: None, // SQLite `DEFAULT CURRENT_TIMESTAMP` fills it
            target_url: self.cfg.url.clone(),
            model_name: self.cfg.model.clone(),
            backend_type: None,
            quantization: None,
            system_gpu: None,
            total_duration_sec: None,
            // R2 (traceability): record which engines this run selected so
            // a stored session joins back to its run / log / export.
            engines: Some(self.cfg.engines.selected_names_json()),
            // R2: the B/C2/C3/D/F headline results (CF-14).
            engine_results,
        };

        // Engine A: speed results → stream_metrics rows.
        let mut metrics: Vec<StreamMetricRow> = Vec::new();
        if let Some(results) = (*self.slots.speed.load()).clone() {
            for r in &results {
                let mut row = StreamMetricRow::from_speed_result(r, &session.session_id, 1);
                // CF-02 parity with the headless path: populate
                // `reasoning_tokens` when the server reported the detail.
                if let Some(t) = r.reasoning_tokens {
                    row = row.with_reasoning_tokens(t);
                }
                metrics.push(row);
            }
        }

        // Engine C1: NIAH results → needle_evaluations rows.
        let mut needles: Vec<NeedleEvaluation> = Vec::new();
        if let Some(niah) = (*self.slots.niah.load()).as_ref() {
            for cell in &niah.cells {
                needles.push(NeedleEvaluation {
                    eval_id: None,
                    session_id: session.session_id.clone(),
                    context_length: Some(cell.target_tokens as i64),
                    depth_percent: Some(cell.depth_percent as f64),
                    retrieved_successfully: Some(cell.retrieved),
                    // Option C: persist the per-cell truncated-by-reasoning
                    // state (inconclusive (budget), not a retrieval fail).
                    truncated_by_reasoning: Some(cell.truncated_by_reasoning),
                    latency_ms: Some(cell.ttft_s * 1000.0),
                });
            }
        }

        match Database::open_default() {
            Ok(mut db) => {
                // Insert the session + stream metrics in one transaction.
                if let Err(e) = db.persist_run(&session, &metrics) {
                    self.logger
                        .warn(Context::Sequence, format!("persist session failed: {e}"));
                } else {
                    // Insert the needle evaluations (separate inserts).
                    for n in &needles {
                        if let Err(e) = db.insert_needle_evaluation(n) {
                            self.logger
                                .warn(Context::Sequence, format!("persist needle failed: {e}"));
                            break;
                        }
                    }
                    let id = &session.session_id;
                    self.log_line(format!(
                        "[seq] saved → {} (session {})",
                        db.path().display(),
                        &id[..8]
                    ));
                    self.logger.info(
                        Context::Sequence,
                        format!(
                            "run persisted → {} (session {}, {} metric rows, {} needle rows)",
                            db.path().display(),
                            &id[..8],
                            metrics.len(),
                            needles.len()
                        ),
                    );
                }
            }
            Err(e) => {
                self.logger
                    .warn(Context::Sequence, format!("open DB failed: {e}"));
            }
        }
    }

    // ── Per-engine runners (one at a time) ─────────────────────────────

    async fn run_speed(&self) -> String {
        self.slots.speed.set_running(true);
        let engine = match SpeedEngine::new(&self.cfg) {
            Ok(e) => e,
            Err(e) => {
                self.slots.speed.set_running(false);
                self.logger
                    .error(Context::EngineA, format!("Engine A init failed: {e}"));
                return format!("init failed: {e}");
            }
        };
        let engine = engine
            .metrics(self.metrics.clone())
            .progress(self.bus.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        let (_, results) = engine.run().await;
        self.slots.speed.set_running(false);
        self.slots.speed.store(results.clone());
        summarize_speed(&results)
    }

    async fn run_concurrency(&self) -> String {
        let Some(mut sweep) = build_sweep(
            &self.cfg,
            Some(self.metrics.clone()),
            Some(self.logger.clone()),
        ) else {
            self.logger.error(
                Context::EngineB,
                "Engine B init failed (client build) — sweep skipped",
            );
            return "sweep init failed (client build)".to_string();
        };
        sweep = sweep
            .progress(self.bus.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        self.slots.concurrency.set_running(true);
        let result = sweep.run().await;
        self.slots.concurrency.set_running(false);
        self.slots.concurrency.store(result.clone());
        summarize_sweep(&result)
    }

    async fn run_niah(&self) -> String {
        self.slots.niah.set_running(true);
        let engine = match NiahEngine::new(&self.cfg) {
            Ok(e) => e,
            Err(e) => {
                self.slots.niah.set_running(false);
                self.logger
                    .error(Context::EngineC1, format!("Engine C1 init failed: {e}"));
                return format!("init failed: {e}");
            }
        };
        let engine = engine
            .progress(self.bus.clone())
            .metrics(self.metrics.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        let result = engine.run().await;
        self.slots.niah.set_running(false);
        self.slots.niah.store(result.clone());
        result.accuracy_label()
    }

    async fn run_reasoning(&self) -> String {
        self.slots.reasoning.set_running(true);
        let engine = match ReasoningEngine::new(&self.cfg) {
            Ok(e) => e,
            Err(e) => {
                self.slots.reasoning.set_running(false);
                self.logger
                    .error(Context::EngineC2, format!("Engine C2 init failed: {e}"));
                return format!("init failed: {e}");
            }
        };
        let engine = engine
            .progress(self.bus.clone())
            .metrics(self.metrics.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        let result = engine.run().await;
        self.slots.reasoning.set_running(false);
        self.slots.reasoning.store(result.clone());
        format!(
            "{} · avg {:.1} t/s",
            result.score.label(),
            result.avg_tg_speed()
        )
    }

    async fn run_structured(&self) -> String {
        self.slots.structured.set_running(true);
        let engine = match StructuredEngine::new(&self.cfg) {
            Ok(e) => e,
            Err(e) => {
                self.slots.structured.set_running(false);
                self.logger
                    .error(Context::EngineC3, format!("Engine C3 init failed: {e}"));
                return format!("init failed: {e}");
            }
        };
        let engine = engine
            .progress(self.bus.clone())
            .metrics(self.metrics.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        let result = engine.run().await;
        self.slots.structured.set_running(false);
        self.slots.structured.store(result.clone());
        result.summary_line()
    }

    /// Arm the GPU & Power monitor (View 4) and measure the **idle power
    /// baseline** before any load hits the GPU.
    ///
    /// `begin_run()` resets the monitor and opens the idle clock; the
    /// background 100 ms poller task then accumulates `IDLE_WINDOW_SECS` of
    /// no-load power into the monitor's idle baseline while we simply sleep
    /// (a pure async yield — no mutex held, so the poller task keeps
    /// ticking on the current-thread runtime). Once the window elapses,
    /// `start_load()` finalizes the baseline (mean idle power) and opens the
    /// load window, from which the 1 Hz power history is recorded.
    ///
    /// This is what makes the View 4 **compute power** (`total − idle`) and
    /// energy numbers meaningful: they are measured relative to the
    /// machine's real no-load floor, not from zero. No-op without a hardware
    /// poller (a driver-less host — the N/A rule).
    async fn prepare_power_monitor(&self) {
        let Some(poller) = &self.hw else {
            return;
        };
        // Arm the monitor (a brief lock — never held across an await).
        if let Ok(mut p) = poller.lock() {
            p.begin_run();
        }
        self.log_line(format!(
            "[GPU] measuring idle power baseline ({}s) before load…",
            IDLE_WINDOW_SECS
        ));
        self.logger.info(
            Context::Sequence,
            format!(
                "measuring idle power baseline ({}s) before the benchmark load",
                IDLE_WINDOW_SECS
            ),
        );
        // Sleep the idle window: the background poller accumulates the
        // baseline while we wait.
        tokio::time::sleep(Duration::from_secs(IDLE_WINDOW_SECS)).await;
        // Open the load window (finalize the idle baseline; history begins).
        if let Ok(mut p) = poller.lock() {
            p.start_load();
        }
    }

    /// Close the GPU & Power monitor's (View 4) **load window** when the run
    /// ends: the duration (and the duration-derived averages) freeze at
    /// their final values instead of keeping count on the frozen panel.
    /// Idempotent — the sequence calls it at both freeze sites (the last
    /// engine's `Complete` and the `AllComplete` safety net). No-op without
    /// a hardware poller (a driver-less host — the N/A rule).
    fn end_power_monitor(&self) {
        let Some(poller) = &self.hw else {
            return;
        };
        // A brief lock — never held across an await (the key/render path
        // never touches it, measurement-isolation invariant, blueprint §4).
        if let Ok(mut p) = poller.lock() {
            p.end_load();
        }
    }

    /// Publish the monitor's final state to the metrics snapshot (the
    /// run-end seam that [`Self::end_power_monitor`] needs).
    ///
    /// The 100 ms poller task is the monitor's normal publisher, but it
    /// stops calling `tick()` the moment [`MetricsState`] is frozen (and
    /// `update()` is a no-op while frozen) — so this must run **before**
    /// the freeze: it carries the `end_load()`-closed monitor state
    /// (`load_ended` set → duration final, energy / avg power windowed to
    /// the test) into the lock-free snapshot View 4 renders. Without it
    /// the panel keeps showing the pre-`end_load` monitor: the duration
    /// counts on forever and the avg power decays forever (the "GPU &
    /// Power tab keeps running after the tests complete" bug). Idempotent
    /// (the `AllComplete` safety net re-publishes the same state); a
    /// no-op without a hardware poller.
    fn publish_power_monitor(&self) {
        let Some(poller) = &self.hw else {
            return;
        };
        // A brief lock — never held across an await (the key/render path
        // never touches it, measurement-isolation invariant, blueprint §4).
        if let Ok(p) = poller.lock() {
            p.publish_monitor(&self.metrics);
        }
    }

    /// CF-03 / H5: the cumulative **input + output** token count across every
    /// engine that has produced tokens so far (A+B+C1+C2+C3+F), read from the
    /// result slots — the honest J/token denominator. Delegates to the
    /// **shared** [`sum_all_engines_tokens`] helper so the TUI and the
    /// headless path compute the *same* denominator (H5) — the old code
    /// divided by A+B only (and, in headless, A only), inflating the headline
    /// J/token by excluding the tokens of the most energy-hungry work.
    ///
    /// **Real counts** are used where the result type carries them: A
    /// `completion_tokens`, B `levels[].total_tokens`, **C1 per-cell
    /// `prompt_tokens` + `completion_tokens`** (H2 — the server-reported
    /// usage; the `chars/4` doc estimate stands in for the prompt only when
    /// no usage block arrived), F `total_tokens`. C2/C3 (and a C1 cell with
    /// no usage block) contribute a clearly-labeled `chars/4` / gen-cap
    /// estimate.
    ///
    /// Slots that have not run yet contribute `0`, so calling this at D's
    /// sampling point (before F) and at `persist_results` (after F) each stay
    /// consistent with the power window they pair with.
    fn all_engines_tokens(&self) -> u64 {
        sum_all_engines_tokens(
            (*self.slots.speed.load()).as_deref(), // Option<Vec<_>> → &[SpeedResult]
            (*self.slots.concurrency.load()).as_ref(),
            (*self.slots.niah.load()).as_ref(),
            (*self.slots.reasoning.load()).as_ref(),
            (*self.slots.structured.load()).as_ref(),
            (*self.slots.flatout.load()).as_ref(),
        )
    }

    /// Engine D — the continuous energy profiler: a short final sampling
    /// window (the 100 ms poller keeps running underneath), then the
    /// Silicon Efficiency Metric over the whole trace. Returns the
    /// one-line summary (`J/token`, or `N/A` on a driverless host).
    async fn run_hardware(&self) -> String {
        // CF-03: the J/token denominator spans every engine that has produced
        // tokens by the time D samples (A+B+C1+C2+C3; F runs after D). This
        // matches the power trace the numerator integrates, so numerator and
        // denominator span the same engine set (no more A+B-only inflation).
        let total_tokens = self.all_engines_tokens();
        self.logger.info(
            Context::EngineD,
            format!(
                "Engine D (Energy) started — 3s sampling window over {total_tokens} tokens (all engines)"
            ),
        );
        const WINDOW: Duration = Duration::from_secs(3);
        let start = std::time::Instant::now();
        while start.elapsed() < WINDOW {
            self.bus.publish(EngineProgress::Sampling {
                elapsed: start.elapsed().as_secs_f64(),
            });
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        // Harvest the poller's accumulated power trace plus the backend's
        // vendor + model (the 100 ms task filled the trace; we only read it
        // here, never on the timing path).
        let (trace, vendor, model, load_joules) = match &self.hw {
            Some(poller) => {
                let g = poller.lock().ok();
                match g {
                    Some(p) => {
                        let (vendor, model) = p
                            .backend()
                            .map(|b| (Some(b.vendor().to_string()), b.model().map(str::to_string)))
                            .unwrap_or((None, None));
                        (p.trace().to_vec(), vendor, model, p.load_window_joules())
                    }
                    None => (Vec::new(), None, None, None),
                }
            }
            None => (Vec::new(), None, None, None),
        };
        let mut energy = profile(&trace, None, total_tokens);
        energy.vendor = vendor;
        energy.model = model;
        // H3: the numerator is the full load window (the cumulative counter
        // sliced to [load_started, load_ended]), not the bounded 6-min
        // display trace — so a run longer than the trace capacity integrates
        // the whole window (CF-03). The trace still supplies the stats.
        apply_load_window(&mut energy, load_joules, total_tokens);
        let summary = match energy.joules_per_token {
            Some(jpt) => {
                let peak = energy
                    .peak_power_w
                    .map(|w| format!(" · peak {w:.0} W"))
                    .unwrap_or_default();
                let util = energy
                    .avg_utilization_pct
                    .map(|u| format!(" · {u:.0}% util"))
                    .unwrap_or_default();
                format!("{jpt:.3} J/token{peak}{util}")
            }
            None => "N/A (no power telemetry)".to_string(),
        };
        self.logger
            .info(Context::EngineD, format!("Engine D complete — {summary}"));
        summary
    }

    /// Engine F — Flat Out: the real-world maximum throughput finale.
    /// Runs at Engine B's **max-throughput level** (the sweep *envelope* —
    /// the last level before the saturation knee, else the peak-aggregate
    /// level) for 60 seconds: the server's full-load number to quote.
    /// When a sweep exists but has no envelope (no usable curve) it falls
    /// back to the practical sweet spot; with no sweep at all, to
    /// [`DEFAULT_STREAM_COUNT`] streams. Returns the one-line summary.
    async fn run_flatout(&self) -> String {
        self.slots.flatout.set_running(true);
        let (stream_count, source) = self
            .slots
            .concurrency
            .load()
            .as_ref()
            .as_ref()
            .map(|r| {
                // The max-throughput level (the envelope); the practical
                // sweet spot only as the fallback when no curve exists.
                let n = r
                    .envelope()
                    .map(|e| e.sweet_spot)
                    .unwrap_or(r.sweet_spot);
                (n, StreamCountSource::ConcurrencyMaxThroughput)
            })
            .unwrap_or((DEFAULT_STREAM_COUNT, StreamCountSource::Default));
        let engine = match FlatOutEngine::new(&self.cfg) {
            Ok(e) => e,
            Err(e) => {
                self.slots.flatout.set_running(false);
                self.logger
                    .error(Context::EngineF, format!("Engine F init failed: {e}"));
                return format!("init failed: {e}");
            }
        };
        let engine = engine
            .stream_count(stream_count, source)
            .metrics(self.metrics.clone())
            .progress(self.bus.clone())
            .pause(self.pause.clone())
            .logger(self.logger.clone());
        let result = engine.run().await;
        self.slots.flatout.set_running(false);
        self.slots.flatout.store(result.clone());
        result.summary_line()
    }
}

// ── Summary helpers (pure) ────────────────────────────────────────────────

/// `1,847` — thousands separators (the Flat Out progress label's token
/// count).
fn thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Engine A's one-line summary: the averaged decode speed across the
/// valid runs (`N/A` when every run failed), plus the P2-A stream-integrity
/// flags when any valid run carries them (CF-44 E2E-only, CF-10 usage
/// discrepancy, CF-24 clamped ITL samples).
pub fn summarize_speed(results: &[SpeedResult]) -> String {
    let valid: Vec<&SpeedResult> = results.iter().filter(|r| !r.is_failed()).collect();
    match valid.first() {
        Some(_) => {
            let n = valid.len() as f64;
            let tg = valid.iter().map(|r| r.tg_speed).sum::<f64>() / n;
            let ttft = valid.iter().map(|r| r.ttft).sum::<f64>() / n * 1000.0;
            let mut line = format!(
                "{:.1} t/s decode · TTFT {:.0} ms · {} run(s)",
                tg,
                ttft,
                valid.len()
            );
            // CF-44 (P2-A): a non-streaming (plain-JSON) result is
            // E2E-only — its "TTFT" is total response latency.
            if valid.iter().any(|r| r.e2e_only) {
                line.push_str(" · E2E-only (non-streaming)");
            }
            // CF-10 (P2-A): the server's usage exceeded ~4× the observed
            // frames — the count was corrected toward the wire truth.
            if valid.iter().any(|r| r.usage_discrepancy) {
                line.push_str(" · ⚠ usage > 4× frames (corrected)");
            }
            // CF-24 (P2-A): ITL samples clamped at the histogram bound —
            // the percentile figures are capped, not exact.
            let clamped: u64 = valid.iter().map(|r| r.itl_clamped).sum();
            if clamped > 0 {
                line.push_str(&format!(" · {clamped} ITL sample(s) clamped"));
            }
            line
        }
        None => "all runs failed".to_string(),
    }
}

/// Engine B's one-line summary: the **practical sweet spot** (the
/// highest concurrency where every user still gets ≥ 29.4 t/s — 30 t/s
/// ideal with a 2% margin, what matters for real use), with the
/// pure-throughput knee as reference, or a note when the curve was empty
/// / nothing cleared the usability bar.
#[must_use]
pub fn summarize_sweep(result: &SweepResult) -> String {
    if result.is_empty() {
        return "no usable levels".to_string();
    }
    let us = result.usability();
    let knee = result
        .envelope()
        .and_then(|e| e.knee)
        .map(|k| format!(" · throughput knee at {}", k.concurrency))
        .unwrap_or_default();
    match us.practical_sweet_spot {
        Some(n) => format!(
            "practical sweet spot {} users (~{:.0} t/s each){}",
            n, us.practical_per_stream, knee
        ),
        None => {
            // Even the lightest load is below the 30 t/s/user comfort bar
            // (29.4 effective): report the usability edge instead.
            match us.max_usable {
                Some(n) => format!(
                    "no level reaches 30 t/s/user — usable up to {} users (~{:.0} t/s each){}",
                    n, us.max_usable_per_stream, knee
                ),
                None => format!("unusable for interactive use (below 15 t/s/user even at 1){knee}"),
            }
        }
    }
}

/// The cumulative **input + output** token count across every engine (the
/// honest J/token denominator, CF-03) — a **shared** pure helper used by
/// both the TUI ([`BenchmarkSequence::all_engines_tokens`]) and the headless
/// path ([`crate::run_headless`]) so the two can never diverge (H5).
///
/// Each argument is `None` for an engine that has not run yet (it
/// contributes `0`). **H2:** C1 (NIAH) contributes the real per-cell
/// `prompt_tokens` + `completion_tokens`; the labeled `NIAH_MAX_GEN_TOKENS`
/// gen-cap stands in only for a cell with no usage block. C2/C3 contribute
/// the codebase's standard no-tokenizer `chars/4` estimate. The numerator
/// (the power window the energy math integrates) spans the same engine set
/// as this denominator — a number must mean what it says.
#[must_use]
pub fn sum_all_engines_tokens(
    speed: Option<&[SpeedResult]>,
    sweep: Option<&SweepResult>,
    niah: Option<&NiahResult>,
    reasoning: Option<&ReasoningResult>,
    structured: Option<&StructuredResult>,
    flatout: Option<&FlatOutResult>,
) -> u64 {
    let mut total = 0u64;
    // Engine A — real completion tokens.
    if let Some(results) = speed {
        total += results.iter().map(|r| r.completion_tokens).sum::<u64>();
    }
    // Engine B — real sweep total tokens.
    if let Some(result) = sweep {
        total += result.levels.iter().map(|l| l.total_tokens).sum::<u64>();
    }
    // Engine C1 (NIAH) — real per-cell prompt + completion tokens (H2); the
    // labeled gen-cap stands in only for a cell with no usage block.
    if let Some(r) = niah {
        for c in r.cells.iter().filter(|c| c.target_tokens > 0) {
            let completion = match c.completion_tokens {
                Some(n) => n,
                None => NIAH_MAX_GEN_TOKENS as u64, // labeled gen-cap fallback
            };
            total += c.prompt_tokens + completion;
        }
    }
    // Engine C2 (Reasoning) — estimate: chars/4 over the response text.
    if let Some(r) = reasoning {
        let chars: usize = r.responses.iter().map(|s| s.chars().count()).sum();
        total += (chars / 4) as u64;
    }
    // Engine C3 (Structured) — estimate: chars/4 over free + constrained.
    if let Some(s) = structured {
        let chars: usize = s.free_body.chars().count()
            + s.cases
                .iter()
                .map(|c| c.output.chars().count())
                .sum::<usize>();
        total += (chars / 4) as u64;
    }
    // Engine F (Flat Out) — real total tokens.
    if let Some(f) = flatout {
        total += f.total_tokens;
    }
    total
}

/// H3: apply the load-window numerator to a trace-profiled [`EnergyResult`].
///
/// The `joules` (and the derived `joules_per_token`) come from the poller's
/// cumulative `load_window_joules` (the *full* load window), not the bounded
/// 6-min display trace — so a run longer than the trace capacity integrates
/// the whole window and the numerator spans the same interval as the "all
/// engines" denominator (CF-03). The other stats (peak / avg / utilization /
/// VRAM) stay trace-based (the display ring). `None` (no load window — the
/// N/A case) leaves the trace-profiled values untouched.
fn apply_load_window(energy: &mut EnergyResult, load_joules: Option<f64>, total_tokens: u64) {
    if let Some(j) = load_joules {
        energy.joules = j;
        energy.joules_per_token = (total_tokens > 0).then(|| j / total_tokens as f64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineSelection;
    use crate::engines::capability::{NiahResult, ReasoningScore};
    use crate::engines::concurrency::SweepLevel;

    fn config_with(sel: EngineSelection) -> Config {
        Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "test".to_string(),
            engines: sel,
            ..Config::default()
        }
    }

    fn seq(sel: EngineSelection) -> BenchmarkSequence {
        BenchmarkSequence::new(
            config_with(sel),
            Arc::new(MetricsState::new()),
            None,
            Arc::new(SeqStateSlot::new()),
            Arc::new(ProgressBus::new()),
            RunSlots {
                speed: Arc::new(ResultSlot::new()),
                concurrency: Arc::new(ResultSlot::new()),
                niah: Arc::new(NiahSlot::new()),
                reasoning: Arc::new(ResultSlot::new()),
                structured: Arc::new(ResultSlot::new()),
                flatout: Arc::new(ResultSlot::new()),
            },
            None,
            Arc::new(RunPause::new()),
            crate::log::RunLogger::disabled(),
            "test-session".to_string(),
        )
    }

    // ── queue construction ──────────────────────────────────────────────

    #[test]
    fn queue_is_the_selected_engines_in_canonical_order() {
        let sel = EngineSelection {
            speed: true,
            concurrency: true,
            niah: true,
            reasoning: true,
            structured: true,
            hardware: true,
            flatout: true,
        };
        assert_eq!(
            seq(sel).engines(),
            &[
                Engine::Speed,
                Engine::Concurrency,
                Engine::Niah,
                Engine::Reasoning,
                Engine::Structured,
                Engine::Hardware,
                Engine::FlatOut
            ]
        );
    }

    #[test]
    fn queue_contains_only_selected_engines() {
        let sel = EngineSelection {
            speed: false,
            concurrency: true,
            niah: false,
            reasoning: true,
            structured: false,
            hardware: false,
            flatout: false,
        };
        assert_eq!(
            seq(sel).engines(),
            &[Engine::Concurrency, Engine::Reasoning]
        );
    }

    #[test]
    fn empty_selection_yields_an_empty_queue() {
        let sel = EngineSelection {
            speed: false,
            concurrency: false,
            niah: false,
            reasoning: false,
            structured: false,
            hardware: false,
            flatout: false,
        };
        assert!(seq(sel).engines().is_empty());
    }

    // ── user-facing descriptions ─────────────────────────────────────────

    #[test]
    fn description_is_concise_for_every_engine() {
        for engine in Engine::ALL {
            let desc = engine.description();
            let lines: Vec<&str> = desc.lines().collect();
            assert!(!lines.is_empty(), "{engine:?} has a description");
            assert!(
                lines.len() <= 3,
                "{engine:?} description fits in 3 lines: {desc:?}"
            );
            for l in &lines {
                assert!(
                    l.chars().count() <= 58,
                    "{engine:?} line stays ≤58 cols (panel width): {l:?}"
                );
            }
        }
    }

    #[test]
    fn descriptions_are_distinct_per_engine() {
        let mut descs: Vec<&str> = Engine::ALL.iter().map(|e| e.description()).collect();
        descs.sort();
        for w in descs.windows(2) {
            assert_ne!(w[0], w[1], "two engines share a description");
        }
    }

    // ── progress fractions & labels ─────────────────────────────────────

    #[test]
    fn speed_fraction_tracks_completed_iterations() {
        assert_eq!(
            EngineProgress::Speed {
                iteration: 1,
                total: 5,
                tokens: 0
            }
            .fraction(),
            0.0
        );
        assert!(
            (EngineProgress::Speed {
                iteration: 3,
                total: 5,
                tokens: 100
            }
            .fraction()
                - 0.4)
                .abs()
                < 1e-9
        );
        // 5 of 5 *started* ⇒ 4 completed ⇒ 80%; the final 100% comes
        // from the sequence's `Complete` state, not the engine.
        assert_eq!(
            EngineProgress::Speed {
                iteration: 5,
                total: 5,
                tokens: 500
            }
            .fraction(),
            0.8
        );
    }

    #[test]
    fn concurrency_fraction_tracks_the_ladder() {
        assert_eq!(
            EngineProgress::Concurrency {
                level: 1,
                step: 1,
                total_steps: 7,
                active: 0
            }
            .fraction(),
            0.0
        );
        // Step 3 of 7, all 8 streams active: (2 + 0.5)/7.
        let f = EngineProgress::Concurrency {
            level: 8,
            step: 3,
            total_steps: 7,
            active: 8,
        }
        .fraction();
        assert!((f - 2.5 / 7.0).abs() < 1e-9);
    }

    #[test]
    fn niah_fraction_tracks_the_matrix_cells() {
        let f = EngineProgress::Niah {
            size: 4000,
            depth: 3,
            total_depths: 11,
            cell: 13,
            total_cells: 77,
        }
        .fraction();
        assert!((f - 12.0 / 77.0).abs() < 1e-9);
        assert_eq!(
            EngineProgress::Niah {
                size: 4000,
                depth: 3,
                total_depths: 11,
                cell: 1,
                total_cells: 77,
            }
            .fraction(),
            0.0
        );
    }

    #[test]
    fn progress_labels_match_the_spec() {
        assert_eq!(
            EngineProgress::Speed {
                iteration: 3,
                total: 5,
                tokens: 248
            }
            .label(),
            "Iteration 3/5 · 248 tok"
        );
        assert_eq!(
            EngineProgress::Concurrency {
                level: 8,
                step: 3,
                total_steps: 7,
                active: 8,
            }
            .label(),
            "Concurrency level 8 (step 3/7) · 8 active"
        );
        assert_eq!(
            EngineProgress::Niah {
                size: 4000,
                depth: 3,
                total_depths: 11,
                cell: 13,
                total_cells: 77,
            }
            .label(),
            "Size 4k, Depth 3/11"
        );
        assert_eq!(
            EngineProgress::Reasoning {
                challenge: 5,
                total: 13,
            }
            .label(),
            "Challenge 5/13"
        );
        assert_eq!(
            EngineProgress::Structured { run: 2, total: 2 }.label(),
            "Run 2/2"
        );
        assert_eq!(
            EngineProgress::Sampling { elapsed: 3.0 }.label(),
            "Sampling… 3.0s"
        );
        assert_eq!(
            EngineProgress::FlatOut {
                elapsed_secs: 32.0,
                streams: 8,
                tokens: 1847,
                tps: 57.7,
            }
            .label(),
            "32s / 60s — 8 streams — 1,847 tok (57.7 t/s)"
        );
    }

    #[test]
    fn flatout_fraction_tracks_the_60s_window() {
        assert_eq!(
            EngineProgress::FlatOut {
                elapsed_secs: 0.0,
                streams: 3,
                tokens: 0,
                tps: 0.0,
            }
            .fraction(),
            0.0
        );
        let f = EngineProgress::FlatOut {
            elapsed_secs: 30.0,
            streams: 8,
            tokens: 1000,
            tps: 33.3,
        }
        .fraction();
        assert!((f - 0.5).abs() < 1e-9, "fraction: {f}");
        // Overruns clamp to 1.0 (the window is the only stop).
        assert_eq!(
            EngineProgress::FlatOut {
                elapsed_secs: 90.0,
                streams: 8,
                tokens: 1000,
                tps: 11.1,
            }
            .fraction(),
            1.0
        );
    }

    #[test]
    fn thousands_groups_digits() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1847), "1,847");
        assert_eq!(thousands(1_234_567), "1,234,567");
    }

    // ── slot / bus mechanics ────────────────────────────────────────────

    #[test]
    fn seq_slot_starts_empty_and_publishes() {
        let slot = SeqStateSlot::new();
        assert!(slot.load().is_none());
        assert!(!slot.is_running());

        slot.set_running(true);
        assert!(slot.is_running());
        slot.store(SeqState {
            phase: SeqPhase::Running,
            queue: vec![Engine::Speed],
            engine: Engine::Speed,
            progress: Some(EngineProgress::Speed {
                iteration: 1,
                total: 3,
                tokens: 0,
            }),
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        });
        let s = slot.load().unwrap();
        assert_eq!(s.phase, SeqPhase::Running);
        assert_eq!(s.engine, Engine::Speed);
        assert!(s.queue_index(Engine::Speed) == Some(0));
    }

    #[test]
    fn progress_bus_publishes_and_resets() {
        let bus = ProgressBus::new();
        assert!(bus.load().is_none());
        bus.publish(EngineProgress::Reasoning {
            challenge: 2,
            total: 13,
        });
        assert_eq!(
            bus.load().as_deref(),
            Some(&EngineProgress::Reasoning {
                challenge: 2,
                total: 13,
            })
        );
        bus.reset();
        assert!(bus.load().is_none());
    }

    // ── summaries ───────────────────────────────────────────────────────

    fn speed_result(tg: f64, ttft: f64, tokens: u64, failed: bool) -> SpeedResult {
        SpeedResult {
            ttft,
            prompt_tokens: 100,
            completion_tokens: tokens,
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
            looping: false,
            // P2-A surfacing (CF-10/24/44): a clean fixture run.
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
        }
    }

    #[test]
    fn speed_summary_averages_the_valid_runs() {
        let line = summarize_speed(&[
            speed_result(100.0, 0.2, 256, false),
            speed_result(120.0, 0.4, 256, false),
        ]);
        assert!(line.contains("110.0 t/s decode"), "{line}");
        assert!(line.contains("TTFT 300 ms"), "{line}");
        assert!(line.contains("2 run(s)"), "{line}");
    }

    #[test]
    fn speed_summary_all_failed_is_explicit() {
        assert_eq!(
            summarize_speed(&[speed_result(0.0, 0.0, 0, true)]),
            "all runs failed"
        );
    }

    #[test]
    fn speed_summary_surfaces_the_p2a_integrity_flags() {
        let mut r = speed_result(100.0, 0.2, 256, false);
        r.e2e_only = true;
        r.usage_discrepancy = true;
        r.itl_clamped = 3;
        let line = summarize_speed(&[r]);
        assert!(line.contains("E2E-only (non-streaming)"), "{line}");
        assert!(line.contains("⚠ usage > 4× frames (corrected)"), "{line}");
        assert!(line.contains("3 ITL sample(s) clamped"), "{line}");
        // A clean run carries no flag suffix.
        let clean = summarize_speed(&[speed_result(100.0, 0.2, 256, false)]);
        assert!(!clean.contains("E2E-only"), "{clean}");
        assert!(!clean.contains("usage >"), "{clean}");
        assert!(!clean.contains("clamped"), "{clean}");
    }

    #[test]
    fn sweep_summary_reports_the_practical_sweet_spot() {
        use crate::engines::concurrency::SweepLevel;
        let level = |c: usize, tps: f64, p90: u64| SweepLevel {
            concurrency: c,
            aggregate_tps: tps,
            p50_tpot_ns: p90 / 2,
            p90_tpot_ns: p90,
            p99_tpot_ns: p90 * 2,
            ttft_p50_ns: 10_000_000,
            ttft_p90_ns: 20_000_000,
            total_tokens: 100,
            completed_streams: c,
            failed_streams: 0,
            timed_out_streams: 0,
            timed_out_tokens: 0,
            aborted: false,
            wall_ns: 1_000_000_000,
            streams: Vec::new(),

            context: 0,
            per_stream_tps: 0.0,
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            estimated: false,
        };
        // Per-stream: 100 @ 1 user, 175 @ 2 → the practical sweet spot
        // (≥29.4 t/s each — 30 with a 2% margin) is 2. No knee on a
        // 2-level climb.
        let result = SweepResult::from_levels(
            vec![level(1, 100.0, 5_000_000), level(2, 350.0, 6_000_000)],
            None,
        );
        let line = summarize_sweep(&result);
        assert!(line.contains("practical sweet spot 2 users"), "{line}");
        assert!(line.contains("~175 t/s each"), "{line}");
        assert_eq!(summarize_sweep(&SweepResult::default()), "no usable levels");
    }

    // ── CF-03: the J/token denominator spans every engine, not just A+B ──

    fn sweep_with_total_tokens(tokens: u64) -> SweepResult {
        let level = SweepLevel {
            concurrency: 1,
            aggregate_tps: 100.0,
            p50_tpot_ns: 1_000_000,
            p90_tpot_ns: 2_000_000,
            p99_tpot_ns: 3_000_000,
            ttft_p50_ns: 10_000_000,
            ttft_p90_ns: 20_000_000,
            total_tokens: tokens,
            completed_streams: 1,
            failed_streams: 0,
            timed_out_streams: 0,
            timed_out_tokens: 0,
            aborted: false,
            wall_ns: 1_000_000_000,
            streams: Vec::new(),
            context: 0,
            per_stream_tps: 100.0,
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            estimated: false,
        };
        SweepResult::from_levels(vec![level], None)
    }

    fn niah_with_run_cells(run: usize) -> NiahResult {
        let mut r = NiahResult::new(vec![2000], vec![50, 60]);
        for c in r.cells.iter_mut().take(run) {
            c.target_tokens = 2000; // mark the cell as run (an unrun cell is 0)
        }
        r
    }

    fn reasoning_with_response_chars(chars: usize) -> ReasoningResult {
        ReasoningResult {
            responses: vec!["x".repeat(chars)],
            ttfts: vec![0.2],
            tg_speeds: vec![50.0],
            score: ReasoningScore {
                total: 1,
                solved: 1,
                by_category: [(1, 1), (0, 0), (0, 0)],
            },
        }
    }

    fn structured_with_body_chars(chars: usize) -> StructuredResult {
        StructuredResult {
            free_tps: 65.2,
            constrained_tps: 61.8,
            penalty_pct: 5.2,
            free_ttft: 0.2,
            constrained_ttft: 0.21,
            cases: vec![],
            constrained_body: String::new(),
            free_body: "x".repeat(chars),
        }
    }

    fn flatout_with_total_tokens(tokens: u64) -> FlatOutResult {
        FlatOutResult {
            stream_count: 3,
            stream_count_source: StreamCountSource::Default,
            total_tokens: tokens,
            duration_secs: 60.0,
            aggregate_tps: tokens as f64 / 60.0,
            per_stream_tps: tokens as f64 / 180.0,
            // CF-04 (P2-D): no failures in this fixture — both views agree.
            tps_per_completed_stream: tokens as f64 / 180.0,
            tps_per_requested_stream: tokens as f64 / 180.0,
            failed_stream_ratio: 0.0,
            // CF-29 (P2-D): a normal run, unclamped cap.
            config_error: None,
            max_tokens_used: crate::engines::flatout::MAX_TOKENS,
            ttft_avg_ms: 100.0,
            itl_p50_ms: 20.0,
            itl_p99_ms: 40.0,
            // CF-07 fields (WS2): this fixture is an authoritative count.
            tokens_estimated: false,
            counting_method: "usage".into(),
        }
    }

    #[test]
    fn all_engines_tokens_spans_every_engine_not_just_a_b() {
        let s = seq(EngineSelection {
            speed: true,
            concurrency: true,
            niah: true,
            reasoning: true,
            structured: true,
            hardware: true,
            flatout: true,
        });

        // Populate every result slot with a known token contribution:
        //   A: 256 (real)  B: 1000 (real)  C1: 1 run cell (est.)
        //   C2: 400 chars (est.)  C3: 800 chars (est.)  F: 5000 (real)
        s.slots
            .speed
            .store(vec![speed_result(100.0, 0.2, 256, false)]);
        s.slots.concurrency.store(sweep_with_total_tokens(1000));
        s.slots.niah.store(niah_with_run_cells(1));
        s.slots.reasoning.store(reasoning_with_response_chars(400));
        s.slots.structured.store(structured_with_body_chars(800));
        s.slots.flatout.store(flatout_with_total_tokens(5000));

        // The old A+B-only denominator would be 256 + 1000 = 1256.
        let a_b_only = 256u64 + 1000;
        // The honest all-engine denominator (the C1 contribution is
        // `run_cells × NIAH_MAX_GEN_TOKENS` — 2048 after the R3 budget
        // raise):
        //   256 + 1000 + (1×2048) + (400/4) + (800/4) + 5000 = 8604.
        let all = s.all_engines_tokens();
        assert_eq!(all, 8604, "denominator spans A+B+C1+C2+C3+F, not just A+B");
        assert!(
            all > a_b_only,
            "the all-engine denominator ({all}) must exceed the old A+B-only ({a_b_only})"
        );
    }

    #[test]
    fn all_engines_tokens_is_zero_when_no_engine_has_run() {
        // An empty selection → every slot empty → the denominator is 0
        // (the N/A rule: `profile` yields `None` J/token, never a spurious 0.0).
        let s = seq(EngineSelection {
            speed: false,
            concurrency: false,
            niah: false,
            reasoning: false,
            structured: false,
            hardware: true,
            flatout: false,
        });
        assert_eq!(s.all_engines_tokens(), 0);
    }

    /// H2: a NIAH result with known per-cell prompt + completion tokens
    /// yields the *real* sum in the C1 denominator, not the old
    /// `run_cells × gen-cap` product (which over-counted the ~2.3k real
    /// output tokens ~68× and dropped the ~2.79 M real prompt tokens).
    #[test]
    fn all_engines_tokens_uses_real_niah_prompt_and_completion() {
        let s = seq(EngineSelection {
            speed: false,
            concurrency: false,
            niah: true,
            reasoning: false,
            structured: false,
            hardware: true,
            flatout: false,
        });
        // Two run cells with real usage: (prompt 8000, completion 15) and
        // (prompt 32000, completion 20) → the real C1 sum is 40035, not the
        // old 2 × 2048 = 4096.
        let mut r = NiahResult::new(vec![8000, 32000], vec![0]);
        r.cells[0].target_tokens = 8000;
        r.cells[0].prompt_tokens = 8000;
        r.cells[0].completion_tokens = Some(15);
        r.cells[1].target_tokens = 32000;
        r.cells[1].prompt_tokens = 32000;
        r.cells[1].completion_tokens = Some(20);
        s.slots.niah.store(r);
        let all = s.all_engines_tokens();
        assert_eq!(
            all,
            8000 + 15 + 32000 + 20,
            "the real prompt+completion sum: {all}"
        );
        assert_ne!(
            all,
            2 * NIAH_MAX_GEN_TOKENS as u64,
            "not the old gen-cap product"
        );
    }

    /// H2: a NIAH cell with no usage block (`completion_tokens == None`)
    /// falls back to the labeled `NIAH_MAX_GEN_TOKENS` gen-cap for its output
    /// side (defensive — the server reported neither a usage block).
    #[test]
    fn all_engines_tokens_falls_back_to_the_gen_cap_when_usage_is_absent() {
        let s = seq(EngineSelection {
            speed: false,
            concurrency: false,
            niah: true,
            reasoning: false,
            structured: false,
            hardware: true,
            flatout: false,
        });
        // One run cell with a known prompt but no completion (no usage block).
        let mut r = NiahResult::new(vec![8000], vec![0]);
        r.cells[0].target_tokens = 8000;
        r.cells[0].prompt_tokens = 8000;
        r.cells[0].completion_tokens = None;
        s.slots.niah.store(r);
        let all = s.all_engines_tokens();
        assert_eq!(
            all,
            8000 + NIAH_MAX_GEN_TOKENS as u64,
            "the labeled gen-cap fallback: {all}"
        );
    }

    /// H5: the TUI's `all_engines_tokens` method and the shared
    /// `sum_all_engines_tokens` helper (used by the headless path) compute the
    /// **same** denominator for the same stored results — headless and TUI can
    /// never diverge (the old headless path used an A-only denominator).
    #[test]
    fn all_engines_tokens_tui_and_shared_helper_agree() {
        let s = seq(EngineSelection {
            speed: true,
            concurrency: true,
            niah: true,
            reasoning: true,
            structured: true,
            hardware: true,
            flatout: true,
        });
        // Populate every slot with a known contribution.
        s.slots
            .speed
            .store(vec![speed_result(100.0, 0.2, 256, false)]);
        s.slots.concurrency.store(sweep_with_total_tokens(1000));
        let mut niah = NiahResult::new(vec![8000], vec![0]);
        niah.cells[0].target_tokens = 8000;
        niah.cells[0].prompt_tokens = 8000;
        niah.cells[0].completion_tokens = Some(15);
        s.slots.niah.store(niah);
        s.slots.reasoning.store(reasoning_with_response_chars(400));
        s.slots.structured.store(structured_with_body_chars(800));
        s.slots.flatout.store(flatout_with_total_tokens(5000));

        // The TUI method (reads the slots) and the shared helper (the
        // headless path) must agree on the same inputs.
        let via_method = s.all_engines_tokens();
        let via_helper = sum_all_engines_tokens(
            (*s.slots.speed.load()).as_deref(),
            (*s.slots.concurrency.load()).as_ref(),
            (*s.slots.niah.load()).as_ref(),
            (*s.slots.reasoning.load()).as_ref(),
            (*s.slots.structured.load()).as_ref(),
            (*s.slots.flatout.load()).as_ref(),
        );
        assert_eq!(
            via_method, via_helper,
            "TUI and headless denominators agree"
        );
        // 256 (A) + 1000 (B) + (8000+15) (C1) + (400/4) (C2) + (800/4) (C3)
        // + 5000 (F) = 15015.
        assert_eq!(
            via_method,
            256 + 1000 + 8015 + 100 + 200 + 5000,
            "the full all-engine sum"
        );
    }

    // ── the sequence run itself (no network: empty engines no-op) ───────

    #[tokio::test]
    async fn empty_selection_run_is_a_clean_noop() {
        let sel = EngineSelection {
            speed: false,
            concurrency: false,
            niah: false,
            reasoning: false,
            structured: false,
            hardware: false,
            flatout: false,
        };
        // The key path sets the slot running before spawning; the run
        // must clear it again on the no-op path.
        let slot = Arc::new(SeqStateSlot::new());
        slot.set_running(true);
        let s = BenchmarkSequence::new(
            config_with(sel),
            Arc::new(MetricsState::new()),
            None,
            slot.clone(),
            Arc::new(ProgressBus::new()),
            RunSlots {
                speed: Arc::new(ResultSlot::new()),
                concurrency: Arc::new(ResultSlot::new()),
                niah: Arc::new(NiahSlot::new()),
                reasoning: Arc::new(ResultSlot::new()),
                structured: Arc::new(ResultSlot::new()),
                flatout: Arc::new(ResultSlot::new()),
            },
            None,
            Arc::new(RunPause::new()),
            crate::log::RunLogger::disabled(),
            "test-session".to_string(),
        );
        s.run().await;
        assert!(!slot.is_running(), "the slot clears when the run ends");
    }

    // ── the Space-key pause gate ─────────────────────────────────────────

    #[tokio::test]
    async fn pause_gate_blocks_until_cleared() {
        let gate = Arc::new(RunPause::new());
        gate.set(true);
        let waiter = {
            let g = gate.clone();
            tokio::spawn(async move {
                g.wait_while_paused().await;
            })
        };
        // While set, the waiter must still be blocked (no work proceeds).
        tokio::time::sleep(Duration::from_millis(250)).await;
        assert!(!waiter.is_finished(), "gate set → engine work is held");
        // Clearing the gate releases it (resume where it left off).
        gate.set(false);
        tokio::time::timeout(Duration::from_secs(2), waiter)
            .await
            .expect("waiter released")
            .expect("waiter task panicked");
    }

    #[test]
    fn pause_gate_defaults_to_running() {
        let gate = RunPause::new();
        assert!(!gate.is_paused());
        gate.set(true);
        assert!(gate.is_paused());
        gate.set(false);
        assert!(!gate.is_paused());
    }
}
