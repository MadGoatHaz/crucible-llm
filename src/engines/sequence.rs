//! The **Benchmark Sequence** executor: runs the selected engines strictly
//! **one at a time** (A → B → C1 → C2 → C3 → D) and publishes a lock-free
//! progress state that the TUI reads every frame.
//!
//! This is the TUI-side orchestration (the headless path keeps its own
//! sequential order via `run_selected` / `main::run_headless`). The
//! sequence is a `tokio::spawn`ed task that:
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
    NiahEngine, ReasoningEngine, ReasoningResult, StructuredEngine, StructuredResult,
};
use crate::engines::concurrency::SweepResult;
use crate::engines::hardware::profile;
use crate::engines::speed::{SpeedEngine, SpeedResult};
use crate::engines::{build_sweep, NiahSlot, ResultSlot};
use crate::hw::HwPoller;
use crate::log::{Context, RunLogger};
use crate::metrics::state::MetricsState;

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
        }
    }

    /// All six engines in the canonical run order (the default queue
    /// shown before a run starts).
    pub const ALL: [Engine; 6] = [
        Engine::Speed,
        Engine::Concurrency,
        Engine::Niah,
        Engine::Reasoning,
        Engine::Structured,
        Engine::Hardware,
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
                "Multi-stream sweep. Gradually raises parallel requests\n(1→2→3→4→8→12→16→24→32) to find where per-user\nspeed degrades — how many users you can actually serve."
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
                "GPU power profiling (watts, joules/token). MUST run on the\nmachine with the GPU. NVIDIA: built-in (NVML). AMD/Intel:\npending support. Remote users: reports N/A."
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
    ]
    .into_iter()
    .filter(|(on, _)| *on)
    .map(|(_, e)| e)
    .collect()
}

impl BenchmarkSequence {
    /// Build the executor for a run. The engine queue is the selected
    /// engines in the canonical A → B → C1 → C2 → C3 → D order; an empty
    /// selection yields an empty queue (the run is a no-op that logs a
    /// warning).
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

        let mut total_tokens = 0u64;

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
                // Engine D runs last in the canonical order, so the
                // cumulative token count is final when it samples.
                Engine::Hardware => self.run_hardware(total_tokens).await,
            };
            total_tokens += match engine {
                Engine::Speed => summary_tokens_speed(&self.slots.speed),
                Engine::Concurrency => summary_tokens_sweep(&self.slots.concurrency),
                _ => 0u64,
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

            // FIX 2: the last engine is done — freeze the metrics
            // pipeline *now*, before the summary hold. The 100 ms
            // hardware poller would otherwise keep stamping samples and
            // growing `elapsed_sec` for the whole 2 s hold (and the
            // graph would keep animating), so the frozen state is the
            // true final state from this point on. (The freeze at
            // `AllComplete` below is the idempotent safety net.)
            if i + 1 == self.engines.len() {
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
        // The run is over: freeze the metrics pipeline (idempotent — the
        // last engine's `Complete` already froze it before the summary
        // hold). This stops the forever-running 100 ms hardware poller
        // (and any other writer) from re-stamping `0.0` throughput
        // samples / growing `elapsed` / dragging the overall averages
        // after the last engine finishes — so the OVERALL METRICS panel
        // and the throughput graph hold their final values and stop
        // animating.
        self.metrics.freeze();
        self.log_line(format!("[seq] ✓ all benchmarks complete — {final_summary}"));
        self.logger.info(
            Context::Sequence,
            format!("sequence complete — {final_summary}"),
        );
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

    /// Engine D — the continuous energy profiler: a short final sampling
    /// window (the 100 ms poller keeps running underneath), then the
    /// Silicon Efficiency Metric over the whole trace. Returns the
    /// one-line summary (`J/token`, or `N/A` on a driverless host).
    async fn run_hardware(&self, total_tokens: u64) -> String {
        self.logger.info(
            Context::EngineD,
            format!("Engine D (Energy) started — 3s sampling window over {total_tokens} tokens"),
        );
        const WINDOW: Duration = Duration::from_secs(3);
        let start = std::time::Instant::now();
        while start.elapsed() < WINDOW {
            self.bus.publish(EngineProgress::Sampling {
                elapsed: start.elapsed().as_secs_f64(),
            });
            tokio::time::sleep(Duration::from_millis(250)).await;
        }
        let (trace, _gpu) = match &self.hw {
            Some(poller) => poller
                .lock()
                .ok()
                .map(|g| (g.trace().to_vec(), g.gpu_name().map(str::to_string)))
                .unwrap_or_default(),
            None => (Vec::new(), None),
        };
        let energy = profile(&trace, None, total_tokens);
        let summary = match energy.joules_per_token {
            Some(jpt) => {
                let peak = energy
                    .peak_power_w
                    .map(|w| format!(" · peak {w:.0} W"))
                    .unwrap_or_default();
                format!("{jpt:.3} J/token{peak}")
            }
            None => "N/A (no power telemetry)".to_string(),
        };
        self.logger
            .info(Context::EngineD, format!("Engine D complete — {summary}"));
        summary
    }
}

// ── Summary helpers (pure) ────────────────────────────────────────────────

/// Engine A's one-line summary: the averaged decode speed across the
/// valid runs (`N/A` when every run failed).
pub fn summarize_speed(results: &[SpeedResult]) -> String {
    let valid: Vec<&SpeedResult> = results.iter().filter(|r| !r.is_failed()).collect();
    match valid.first() {
        Some(_) => {
            let n = valid.len() as f64;
            let tg = valid.iter().map(|r| r.tg_speed).sum::<f64>() / n;
            let ttft = valid.iter().map(|r| r.ttft).sum::<f64>() / n * 1000.0;
            format!(
                "{:.1} t/s decode · TTFT {:.0} ms · {} run(s)",
                tg,
                ttft,
                valid.len()
            )
        }
        None => "all runs failed".to_string(),
    }
}

/// Engine B's one-line summary: the **practical sweet spot** (the
/// highest concurrency where every user still gets ≥ 40 t/s — what
/// matters for real use), with the pure-throughput knee as reference,
/// or a note when the curve was empty / nothing cleared the usability
/// bar.
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
            // Even the lightest load is below the 40 t/s/user comfort
            // bar: report the usability edge instead.
            match us.max_usable {
                Some(n) => format!(
                    "no level reaches 40 t/s/user — usable up to {} users (~{:.0} t/s each){}",
                    n, us.max_usable_per_stream, knee
                ),
                None => format!("unusable for interactive use (below 15 t/s/user even at 1){knee}"),
            }
        }
    }
}

/// Tokens generated by a stored Engine A run (the Energy denominator).
fn summary_tokens_speed(slot: &ResultSlot<Vec<SpeedResult>>) -> u64 {
    match slot.load().as_ref() {
        Some(results) => results.iter().map(|r| r.completion_tokens).sum(),
        None => 0,
    }
}

/// Tokens generated by a stored Engine B sweep (the Energy denominator).
fn summary_tokens_sweep(slot: &ResultSlot<SweepResult>) -> u64 {
    match slot.load().as_ref() {
        Some(result) => result.levels.iter().map(|l| l.total_tokens).sum(),
        None => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::EngineSelection;

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
            },
            None,
            Arc::new(RunPause::new()),
            crate::log::RunLogger::disabled(),
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
        };
        assert_eq!(
            seq(sel).engines(),
            &[
                Engine::Speed,
                Engine::Concurrency,
                Engine::Niah,
                Engine::Reasoning,
                Engine::Structured,
                Engine::Hardware
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
            aborted: false,
            wall_ns: 1_000_000_000,
            streams: Vec::new(),
        };
        // Per-stream: 100 @ 1 user, 175 @ 2 → the practical sweet spot
        // (≥40 t/s each) is 2. No knee on a 2-level climb.
        let result = SweepResult {
            levels: vec![level(1, 100.0, 5_000_000), level(2, 350.0, 6_000_000)],
        };
        let line = summarize_sweep(&result);
        assert!(line.contains("practical sweet spot 2 users"), "{line}");
        assert!(line.contains("~175 t/s each"), "{line}");
        assert_eq!(summarize_sweep(&SweepResult::default()), "no usable levels");
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
            },
            None,
            Arc::new(RunPause::new()),
            crate::log::RunLogger::disabled(),
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
