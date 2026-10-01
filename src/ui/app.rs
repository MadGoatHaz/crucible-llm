//! `App` state machine: the `View` enum (`Live`, `Concurrency`, `Needle`,
//! `History`, `Config`) and key handling (blueprint §6 footer: `1`-`5` switch
//! views, `Space` pause/resume, `+` step concurrency, `n` new needle,
//! `e` export, `q`/`Esc` quit).
//!
//! **State-aware key guards** (the TUI interaction audit): a benchmark
//! is a *server-load* event, so while a `BenchmarkSequence` (or any
//! standalone engine run) is in progress the keys that would add load —
//! `n` (NIAH matrix), `+` (concurrency step), `c` (setup takeover, whose
//! URL stage fires an HTTP discovery) — are locked and log why. The
//! always-available keys are view switching (`1`-`5`), `Space`
//! (pause/resume: the shared [`RunPause`] gate holds each engine before
//! its next request, in-flight streams complete, resume continues where
//! it left off), `e` (local file export), and `q`/`Esc` (quit). The
//! `n` key additionally requires a `[Y/N]` confirmation that shows the
//! request count (default matrix: 7 sizes × 11 depths = 77), and the
//! footer + status bar reflect the current state (Idle / BUSY / Paused /
//! confirming).
//!
//! The `App` holds the shared `Arc<MetricsState>` — the `ArcSwap`
//! double-buffered [`MetricsSnapshot`] pipeline (Chunk 6). The render loop
//! only ever *reads* the snapshot via `MetricsState::load()` (a lock-free
//! atomic read); the stream worker *writes* it via `MetricsState::update()`.
//! The render loop never writes the snapshot and never touches the timing
//! path (measurement-isolation invariant, blueprint §4).

use std::sync::{Arc, Mutex};

use arc_swap::ArcSwapOption;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::client::models::ModelInfo;
use crate::config::{Config, ExportFormat};
use crate::engines::{
    fragmentation_warning, queue_for, BenchmarkSequence, NiahEngineConfig, NiahSlot, ProgressBus,
    ReasoningResult, ResultSlot, RunPause, RunSlots, SeqPhase, SeqState, SeqStateSlot, SpeedResult,
    StructuredResult, SweepResult, NIAH_DEPTHS, NIAH_SIZES, VRAM_FRAGMENTATION_THRESHOLD,
};
use crate::hw::HwPoller;
use crate::log::{Context, RunLogger};
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::storage::db::Database;
use crate::storage::export::{self, ExportPayload};
use crate::storage::models::{BenchmarkSession, StreamMetricRow};
use crate::ui::theme::{palette, style};
use crate::ui::views;
use crate::ui::views::config::{ConfigKeyResult, ConfigMode, ConfigState};
use crate::ui::views::history::HistoryState;
use crate::ui::views::setup::{SetupKeyResult, SetupState};

/// The five dashboard views (blueprint §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    /// View 1 — Live Monitor & Telemetry.
    Live,
    /// View 2 — Concurrency & Saturation matrix.
    Concurrency,
    /// View 3 — Needle-in-a-Haystack matrix.
    Needle,
    /// View 4 — Historical Comparison & Diff.
    History,
    /// View 5 — Configuration.
    Config,
}

impl View {
    /// All views in tab-bar order.
    pub const ALL: [View; 5] = [
        View::Live,
        View::Concurrency,
        View::Needle,
        View::History,
        View::Config,
    ];

    /// 0-based index (position in the tab bar).
    pub fn index(self) -> usize {
        match self {
            View::Live => 0,
            View::Concurrency => 1,
            View::Needle => 2,
            View::History => 3,
            View::Config => 4,
        }
    }

    /// Tab-bar label (e.g. "Live Monitor").
    pub fn label(self) -> &'static str {
        match self {
            View::Live => "Live Monitor",
            View::Concurrency => "Concurrency Matrix",
            View::Needle => "Needle (NIAH)",
            View::History => "History Diff",
            View::Config => "Config",
        }
    }

    /// Map a digit (`1`..=`5`) to a view.
    pub fn from_digit(d: u8) -> Option<View> {
        match d {
            1 => Some(View::Live),
            2 => Some(View::Concurrency),
            3 => Some(View::Needle),
            4 => Some(View::History),
            5 => Some(View::Config),
            _ => None,
        }
    }
}

/// The TUI phase: the pre-dashboard **Setup** takeover, or the
/// five-view dashboard itself.
///
/// Setup is *not* a sixth tab: it is a full-screen flow (URL → model
/// discovery/selection → benchmark config → launch) that occupies the
/// whole frame with its own top bar and footer. `App::new()` starts in
/// [`Phase::Dashboard`] (the previous behavior); the entry point moves it
/// to [`Phase::Setup`] when the target (URL + model) was not given
/// explicitly, and the `c` key re-opens Setup from any dashboard view
/// (except View 5, where `c` is a typeable character).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    /// The interactive setup flow (full-screen takeover).
    Setup,
    /// The five-view dashboard (Live / Concurrency / Needle / History /
    /// Config).
    #[default]
    Dashboard,
}

/// Semantic key actions the engine layer consumes (plumbed to the worker
/// pool / storage in later chunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Nothing to do — keep running.
    Continue,
    /// User asked to quit (`q` / `Esc` / Ctrl-C).
    Quit,
    /// `Space` — pause/resume the benchmark.
    PauseResume,
    /// `+` — step the concurrency ladder up.
    StepConcurrency,
    /// `n` — open the NIAH confirmation prompt (idle only; the `y`
    /// answer is what actually spawns the run).
    NewNeedle,
    /// `e` — export results (JSON/MD/CSV).
    Export,
    /// `r` / `F5` — run the engines selected in the Config view (Chunk 18).
    Run,
    /// `j` / `↓` (History view) — move the session cursor down.
    HistoryNext,
    /// `k` / `↑` (History view) — move the session cursor up.
    HistoryPrev,
    /// `a` (History view) — set the cursor's session as diff run A.
    HistorySelectA,
    /// `b` (History view) — set the cursor's session as diff run B.
    HistorySelectB,
}

/// Application state machine: current view, navigation, pause state, and the
/// shared `ArcSwap`-backed metric snapshot the views render.
///
/// `metrics` is the lock-free [`MetricsState`] (Chunk 6): the stream worker
/// publishes a fresh [`MetricsSnapshot`] via `update()`, and the render loop
/// reads it via `load()` without ever blocking or touching the timing path.
#[derive(Debug)]
pub struct App {
    pub running: bool,
    pub paused: bool,
    pub view: View,
    /// Concurrency target the `+` key steps up (plumbed to the worker pool
    /// in later chunks).
    pub concurrency_target: usize,
    /// Monotonic 60Hz render tick counter.
    pub tick: u64,
    /// Shared lock-free metrics snapshot (the `ArcSwap` double buffer).
    pub metrics: Arc<MetricsState>,
    pub log: Vec<Line<'static>>,
    /// The last completed concurrency sweep (plan Chunk 10) — a lock-free
    /// [`ResultSlot`] (Chunk 18) so a background `r`-key sweep can publish
    /// its [`SweepResult`] and View 2 reads it without blocking.
    pub sweep: Arc<ResultSlot<SweepResult>>,
    /// Format the `e` key exports (Chunk 13; JSON by default, set from
    /// `--export` by the entry point).
    pub export_format: ExportFormat,
    /// History view state (Chunk 14): the stored session list plus the
    /// A/B selection and precomputed diff. `None` until the History view
    /// is first opened (lazy DB load) — or the DB is unavailable, in
    /// which case the view renders its placeholder.
    pub history: Option<HistoryState>,
    /// Lock-free NIAH result holder (Chunk 15): the `n`-key background
    /// runner publishes a completed matrix here; View 3 reads it
    /// lock-free (measurement-isolation invariant, blueprint §4).
    pub niah: Arc<NiahSlot>,
    /// The NIAH-relevant config snapshot for spawning background runs
    /// (`n` key). `None` until the entry point supplies one.
    pub niah_config: Option<NiahEngineConfig>,
    /// Lock-free Engine A (speed) result holder (Chunk 18): the `r`-key
    /// runner publishes the per-iteration [`SpeedResult`]s here.
    pub speed_slot: Arc<ResultSlot<Vec<SpeedResult>>>,
    /// Lock-free Engine C2 (reasoning) result holder (Chunk 18).
    pub reasoning_slot: Arc<ResultSlot<ReasoningResult>>,
    /// Lock-free Engine C3 (structured) result holder (Chunk 18).
    pub structured_slot: Arc<ResultSlot<StructuredResult>>,
    /// The editable Configuration form (View 5, Chunk 18). Seeded from the
    /// resolved [`Config`]; edited on the key path; `F2` persists it and
    /// `F5`/`r` runs the selected engines from it.
    pub config: ConfigState,
    /// The 100 ms hardware telemetry poller (Chunk 17): a background
    /// task publishes VRAM / power / clock / J-token into the shared
    /// `ArcSwap<MetricsSnapshot>`; the views read it lock-free. `None`
    /// until the entry point attaches one.
    pub hw: Option<Arc<Mutex<HwPoller>>>,
    /// Lock-free model-discovery slot (Chunk 20): a background
    /// `tokio::spawn`ed `GET {base}/v1/models` (OpenAI-compatible)
    /// publishes the discovered [`ModelInfo`] list here; the setup flow
    /// / Config view read it lock-free via
    /// [`model_list`](Self::model_list) (measurement-isolation
    /// invariant, blueprint §4). `None` until the first discovery
    /// completes; a failed discovery leaves the previous list (if any)
    /// in place — the picker degrades to free text (N/A-never-fail).
    pub models: Arc<ResultSlot<Vec<ModelInfo>>>,
    /// The TUI phase: the Setup takeover or the dashboard (the
    /// entry point picks; `c` re-opens Setup from the dashboard).
    pub phase: Phase,
    /// The interactive setup flow state (the four-stage URL → model →
    /// config → launch sequence). Mutated only in the key/tick path.
    pub setup: SetupState,
    /// The last model-discovery error message (lock-free; `None` on
    /// success). The setup flow surfaces it in the manual-entry stage
    /// (N/A-never-fail: a failed discovery degrades to free text).
    pub discovery_error: Arc<ArcSwapOption<String>>,
    /// Lock-free benchmark-sequence state (the sequential executor's TUI
    /// seam): the Live view's header / progress bar / queue panel read it
    /// lock-free via [`SeqStateSlot::load`]. `None` until a sequence
    /// starts (measurement-isolation invariant, blueprint §4).
    pub seq: Arc<SeqStateSlot>,
    /// The per-engine progress bus: the sequence's engines publish their
    /// [`crate::engines::EngineProgress`] here, and the executor's 10 Hz
    /// ticker mirrors it into [`seq`](Self::seq).
    pub seq_bus: Arc<ProgressBus>,
    /// The `Space`-key cooperative pause gate shared with every engine
    /// run loop: while set, engines hold *before* spawning the next
    /// request unit (in-flight streams complete; resume continues exactly
    /// where it left off — no state is corrupted). The key path toggles
    /// it; the render loop never touches it (measurement-isolation
    /// invariant, blueprint §4).
    pub pause: Arc<RunPause>,
    /// `true` while the `n`-key confirmation prompt is on screen
    /// ("Run NIAH test? [Y/N]") — a modal that swallows every key but
    /// `y`/`Enter` (confirm) and anything else (cancel). Set only when
    /// the system is idle (never during a benchmark sequence), so a
    /// single confirmed press can spawn at most one NIAH run.
    pub pending_niah: bool,
    /// The executor's log pipe: the sequence task *sends* real events;
    /// the tick path *drains* them into [`log`](Self::log) (the bounded
    /// line list the Live view renders). `None` until the first run.
    pub log_rx: Option<std::sync::mpsc::Receiver<String>>,
    /// Hysteresis latch for the VRAM fragmentation warning (blueprint
    /// §5D): the log line fires once when occupancy crosses the
    /// threshold and re-arms once it falls 5 points below it.
    vram_warned: bool,
    /// The file-based run logger (shared with every engine the App
    /// spawns): setup / discovery / launch events land in
    /// `latest.log` + the run archive. A `disabled` logger (the
    /// `App::new()` default for tests) makes every call a no-op.
    pub logger: Arc<RunLogger>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Fresh app in the Live view, running, seeded with the blueprint-mock
    /// sample snapshot so the dashboard is verifiable before a real stream
    /// worker publishes data.
    pub fn new() -> Self {
        let log = Vec::new();
        let metrics = Arc::new(MetricsState::new());
        Self {
            running: true,
            paused: false,
            view: View::Live,
            concurrency_target: 1,
            tick: 0,
            metrics,
            log,
            sweep: Arc::new(ResultSlot::new()),
            export_format: ExportFormat::default(),
            history: None,
            niah: Arc::new(NiahSlot::new()),
            niah_config: None,
            speed_slot: Arc::new(ResultSlot::new()),
            reasoning_slot: Arc::new(ResultSlot::new()),
            structured_slot: Arc::new(ResultSlot::new()),
            config: ConfigState::default(),
            hw: None,
            models: Arc::new(ResultSlot::new()),
            // `App::new()` keeps the classic behavior (dashboard first);
            // the entry point switches to Setup via `with_setup` when the
            // target was not fully pre-configured.
            phase: Phase::Dashboard,
            setup: SetupState::new(),
            discovery_error: Arc::new(ArcSwapOption::empty()),
            seq: Arc::new(SeqStateSlot::new()),
            seq_bus: Arc::new(ProgressBus::new()),
            pause: Arc::new(RunPause::new()),
            pending_niah: false,
            log_rx: None,
            vram_warned: false,
            // Tests use `App::new()` directly: a disabled logger keeps
            // them off the filesystem (the entry point attaches the real
            // one via `with_logger`).
            logger: RunLogger::disabled(),
        }
    }

    /// Attach the run logger (the entry point calls this with the
    /// process's `RunLogger::start(...)` instance).
    pub fn with_logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = logger;
        self
    }

    /// Supply the NIAH-relevant config snapshot (Chunk 15) so the `n`
    /// key can spawn a background matrix run against the target endpoint.
    pub fn with_niah_config(mut self, cfg: &Config) -> Self {
        self.niah_config = Some(NiahEngineConfig::from_config(cfg));
        self
    }

    /// Seed the editable Configuration form (View 5, Chunk 18) from the
    /// resolved [`Config`]. The form is the TUI's edit surface over the
    /// same config the headless/export paths consume; `F2` persists it and
    /// `F5`/`r` run the selected engines from it. Also refreshes the NIAH
    /// config snapshot so the `n` key stays in sync with the form.
    pub fn with_config(mut self, cfg: &Config) -> Self {
        self.config = ConfigState::from_config(cfg);
        self.niah_config = Some(NiahEngineConfig::from_config(cfg));
        // Seed the metrics snapshot with the real target identity so the
        // status bar shows the correct model/endpoint before any engine runs.
        self.metrics.update(MetricsSnapshot {
            endpoint: cfg.url.clone(),
            model: cfg.model.clone(),
            mode: cfg.mode.label().to_string(),
            backend: "vLLM".to_string(),
            ..Default::default()
        });
        self
    }

    /// Attach the hardware telemetry poller (Chunk 17). The entry point
    /// spawns the 100 ms task that drives `hw.tick(&app.metrics)`; the
    /// render path only ever reads the resulting snapshot.
    pub fn with_hw(mut self, hw: Arc<Mutex<HwPoller>>) -> Self {
        self.hw = Some(hw);
        self
    }

    /// Enter the interactive **Setup** phase (the full-screen
    /// pre-dashboard takeover), pre-filling from the resolved [`Config`]:
    ///
    /// * an *explicitly* provided URL is pre-filled in stage 1 (a
    ///   built-in default is not — the setup flow starts with an empty
    ///   prompt, per the recon's "no leaked LAN IP" rule);
    /// * a non-placeholder model name is pre-filled as the stage-2
    ///   manual-entry text.
    ///
    /// The entry point calls this only when the target (URL + model) was
    /// not fully given on the command line / env / config file.
    pub fn with_setup(mut self, cfg: &Config) -> Self {
        self.phase = Phase::Setup;
        if cfg.target_explicit {
            self.setup.url = cfg.url.clone();
            self.setup.url_cursor = self.setup.url.chars().count();
        }
        if cfg.model != crate::config::DEFAULT_MODEL {
            self.setup.model_query = cfg.model.clone();
            self.setup.model_query_cursor = self.setup.model_query.chars().count();
        }
        self
    }

    /// Re-open the Setup phase from the dashboard (the `c` key):
    /// re-seed the flow from the current Configuration form and start at
    /// the benchmark-configuration stage — `Esc` walks back through
    /// model selection and the URL prompt.
    pub fn open_setup(&mut self) {
        self.setup.url = self.config.url.clone();
        self.setup.url_cursor = self.setup.url.chars().count();
        self.setup.models = self
            .model_list()
            .map(|v| v.into_iter().map(|m| m.id).collect())
            .unwrap_or_default();
        self.setup.model_query = self.config.model.clone();
        self.setup.model_query_cursor = self.setup.model_query.chars().count();
        self.setup.model_cursor = 0;
        self.setup.error = None;
        self.setup.form_field = 0;
        self.setup.phase = crate::ui::views::setup::SetupPhase::Config;
        self.phase = Phase::Setup;
    }

    /// Stage-4 `Enter`: leave the Setup takeover, land on the Live view,
    /// and start the benchmark (the selected engines run against the
    /// shared Configuration form the setup flow just filled in).
    pub fn launch(&mut self) {
        self.phase = Phase::Dashboard;
        self.view = View::Live;
        self.paused = false;
        // Refresh the snapshot identity from the current config so the
        // status bar and panels show the real target (not stale/zeroed
        // values from before the setup flow filled in the form).
        let cfg = self.config.to_config();
        self.metrics.update(MetricsSnapshot {
            endpoint: cfg.url.clone(),
            model: cfg.model.clone(),
            mode: cfg.mode.label().to_string(),
            backend: "vLLM".to_string(),
            ..Default::default()
        });
        self.push_log(
            format!("[setup] launching: {} / {}", cfg.url, cfg.model),
            style::value_ok(),
        );
        // The setup summary the run log is reviewed for: the exact target
        // and parameters the run was launched with.
        self.logger.info(
            Context::Setup,
            format!(
                "setup complete — url={} model={} mode={} tokens={} iterations={} timeout={}s nocache={} ladder=[{}] engines=[{}]",
                cfg.url,
                cfg.model,
                cfg.mode.label(),
                cfg.tokens,
                cfg.iterations,
                cfg.timeout,
                cfg.nocache,
                cfg.ladder
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(","),
                cfg.engines.iter_labels().collect::<Vec<_>>().join(",")
            ),
        );
        self.start_run();
    }

    /// The last model-discovery error message (lock-free read; `None`
    /// until a discovery fails, cleared by a successful one).
    pub fn discovery_error(&self) -> Option<String> {
        self.discovery_error.load().as_deref().cloned()
    }

    /// The discovered model list (Chunk 20): a lock-free read of the
    /// [`models`](Self::models) slot — `None` until the first discovery
    /// completes. The render path may call this; it never blocks.
    pub fn model_list(&self) -> Option<Vec<ModelInfo>> {
        (*self.models.load()).clone()
    }

    /// `true` while a model-discovery request is in flight (Chunk 20).
    pub fn discovery_running(&self) -> bool {
        self.models.is_running()
    }

    /// Lazily load the History view state (Chunk 14): open the default
    /// SQLite DB (`data_dir()/crucible/benchmarks.db`) and list the stored
    /// sessions.
    ///
    /// Called only from the key path (first entry into the History view,
    /// or a history key press) — never from the render path, which stays a
    /// pure `&App` read (measurement-isolation invariant, blueprint §4).
    /// A storage failure leaves the state `None`: the view renders its
    /// "no stored runs" placeholder and never panics.
    pub fn ensure_history(&mut self) {
        if self.history.is_some() {
            return;
        }
        let path = Database::default_path();
        self.history = HistoryState::load(&path).ok();
    }

    /// Set the `e`-key export format (`--export` wiring, Chunk 13).
    pub fn with_export_format(mut self, format: ExportFormat) -> Self {
        self.export_format = format;
        self
    }

    /// The `n` key (blueprint §6 footer: "[N] New Needle Test"): spawn a
    /// background NIAH matrix run (Chunk 15, Engine C1) — **after** the
    /// key path shows the `[Y/N]` confirmation prompt (the default matrix
    /// is 7 sizes × 11 depths = 77 sequential requests, so it never
    /// fires blindly).
    ///
    /// This is a *key-path* action (never the render path): the runner
    /// task publishes its result to the lock-free [`NiahSlot`] when done,
    /// and View 3 reads it — so a running matrix never perturbs the 60Hz
    /// render loop or the timing path (measurement-isolation invariant,
    /// blueprint §4).
    ///
    /// **Guards** (defense in depth — `handle_key` enforces them first):
    /// * a `BenchmarkSequence` in progress locks `n` (the matrix would
    ///   otherwise hit the endpoint *alongside* the running engine);
    /// * an in-progress standalone NIAH run locks `n` (one run at a
    ///   time);
    /// * the runner shares the `Space`-key [`RunPause`] gate, so a
    ///   confirmed standalone run pauses/resumes like the sequence.
    pub fn start_niah(&mut self) {
        if self.seq.is_running() {
            self.push_log(
                "[niah] locked while a benchmark sequence is running — \
                 it runs as Engine C1 in the queue"
                    .to_string(),
                style::value_warn(),
            );
            return;
        }
        if self.niah.is_running() {
            self.push_log(
                "[niah] already running — one size × depth cell at a time".to_string(),
                style::value_warn(),
            );
            return;
        }
        // Chunk 18: the `n` key runs against the *current* Configuration
        // form, so edits made in View 5 apply immediately.
        let config = NiahEngineConfig::from_config(&self.config.to_config());
        let engine = match config.engine() {
            Ok(e) => e,
            Err(e) => {
                self.push_log(
                    format!("[niah] engine init failed: {e}"),
                    style::value_err(),
                );
                return;
            }
        };
        let sizes = engine.sizes_list().to_vec();
        let depths = engine.depths_list().to_vec();
        let requests = sizes.len() * depths.len();
        let slot = self.niah.clone();
        self.niah.set_running(true);
        self.push_log(
            format!(
                "[niah] started: {requests} requests ({} sizes × {} depths) → {}",
                sizes.len(),
                depths.len(),
                config.url
            ),
            style::value_ok(),
        );
        // The runner owns the slot handle; on completion it clears the
        // running flag and publishes the scored grid.
        let logger = self.logger.clone();
        let engine = engine.pause(self.pause.clone()).logger(logger.clone());
        tokio::spawn(async move {
            logger.info(
                Context::EngineC1,
                format!(
                    "standalone NIAH run started — {requests} requests → {url}",
                    url = config.url
                ),
            );
            let result = engine.run().await;
            slot.set_running(false);
            slot.store(result);
        });
    }

    /// Trigger model discovery (Chunk 20): `tokio::spawn` a
    /// `GET {base}/v1/models` call (OpenAI-compatible; the base is taken
    /// from the current Configuration form) and publish the result to the
    /// lock-free [`models`](Self::models) slot.
    ///
    /// This is a *key-path* action (the setup flow calls it once the user
    /// confirms the target URL — never the render path): the discovery
    /// task publishes the deduped, id-sorted [`ModelInfo`] list when done,
    /// and the views read it lock-free — so an in-flight HTTP call never
    /// perturbs the 60Hz render loop or the timing path
    /// (measurement-isolation invariant, blueprint §4). Re-triggering
    /// while a discovery is in flight is a no-op (logged). A failed
    /// discovery leaves the previous list (if any) in place — the model
    /// field degrades to free text (N/A-never-fail rule).
    pub fn start_discovery(&mut self) {
        if self.models.is_running() {
            self.push_log(
                "[discovery] already in progress — one request at a time".to_string(),
                style::value_warn(),
            );
            return;
        }
        // The form (View 5 / the setup flow) is the source of the target
        // URL and API key, so edits apply immediately.
        let cfg = self.config.to_config();
        let url = cfg.url.clone();
        let api_key = cfg.api_key.clone();
        let slot = self.models.clone();
        let err_slot = self.discovery_error.clone();
        let logger = self.logger.clone();
        slot.set_running(true);
        self.push_log(
            format!("[discovery] listing models at {url}/models"),
            style::value_ok(),
        );
        logger.info(Context::Discovery, format!("→ GET {url}/models"));
        tokio::spawn(async move {
            let started = std::time::Instant::now();
            match crate::client::models::list_models(&url, api_key.as_deref()).await {
                Ok(models) => {
                    slot.set_running(false);
                    slot.store(models.clone());
                    // A successful discovery clears any stale error.
                    err_slot.store(None);
                    logger.info(
                        Context::Discovery,
                        format!(
                            "← 200 ({} ms) — {} model(s) found: {}",
                            started.elapsed().as_millis(),
                            models.len(),
                            models
                                .iter()
                                .map(|m| m.id.clone())
                                .collect::<Vec<_>>()
                                .join(", ")
                        ),
                    );
                }
                Err(e) => {
                    // Clear the running flag; keep any previously
                    // discovered list (the view falls back to free text).
                    // The message lands in the lock-free error slot so
                    // the setup flow can surface it in the UI.
                    slot.set_running(false);
                    err_slot.store(Some(Arc::new(e.to_string())));
                    logger.error(
                        Context::Discovery,
                        format!("← failed after {} ms: {e}", started.elapsed().as_millis()),
                    );
                    eprintln!("[discovery] failed: {e}");
                }
            }
        });
    }

    /// `r` / `F5` (and the setup flow's launch): run **every engine
    /// selected** in the Configuration form — **strictly one at a time**,
    /// in the canonical A → B → C1 → C2 → C3 → D order.
    ///
    /// The [`BenchmarkSequence`] runs as a single background `tokio` task:
    ///
    /// * it awaits each engine's completion before starting the next (one
    ///   engine hits the endpoint at a time — no queueing cross-talk);
    /// * each engine reports its [`crate::engines::EngineProgress`] to
    ///   [`seq_bus`](Self::seq_bus); the executor's 10 Hz ticker mirrors
    ///   it into the lock-free [`seq`](Self::seq) slot, which the Live
    ///   view's header / progress bar / queue panel read every frame;
    /// * live [`crate::metrics::state::MetricsSnapshot`]s keep flowing to
    ///   the shared metrics seam, so the telemetry gauges show the
    ///   *current* engine's real data;
    /// * each completed result is published to the existing lock-free
    ///   result slots (Views 2/3 read them exactly as before);
    /// * real events stream to [`log`](Self::log) through the mpsc pipe
    ///   the tick path drains (no pre-generated fake logs).
    ///
    /// The render loop never blocks and never touches the quanta timing
    /// path (measurement-isolation invariant, blueprint §4). Engine D
    /// (hardware) is continuous telemetry (the 100 ms poller spawned by
    /// the entry point); when selected, the sequence ends with a short
    /// sampling window that folds the power trace into Joules/Token.
    /// Re-pressing `r` while a sequence is in progress is a no-op (logged).
    pub fn start_run(&mut self) {
        if self.seq.is_running() {
            self.push_log(
                "[seq] already running — engines execute one at a time".to_string(),
                style::value_warn(),
            );
            return;
        }
        let cfg = self.config.to_config();
        if cfg.engines.is_empty() {
            self.push_log(
                "[seq] no engines selected — enable some in the Config view (View 5)".to_string(),
                style::value_warn(),
            );
            return;
        }
        self.seq.set_running(true);
        // Publish the initial sequence state *synchronously* so the Live
        // view's header / progress bar / queue panel are correct from the
        // very first frame (before the spawned task's first 10 Hz tick).
        let queue = queue_for(&cfg.engines);
        self.seq.store(SeqState {
            phase: SeqPhase::Running,
            queue: queue.clone(),
            engine: queue[0],
            progress: None,
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
        });
        self.push_log(
            format!(
                "[seq] starting {} engine(s): {}",
                cfg.engines.count(),
                cfg.engines.iter_labels().collect::<Vec<_>>().join(" → ")
            ),
            style::value_ok(),
        );
        let (tx, rx) = std::sync::mpsc::channel();
        self.log_rx = Some(rx);
        self.logger.info(
            Context::Sequence,
            format!(
                "run started — {} engine(s): {}",
                cfg.engines.count(),
                cfg.engines.iter_labels().collect::<Vec<_>>().join(" → ")
            ),
        );
        let seq = BenchmarkSequence::new(
            cfg,
            self.metrics.clone(),
            self.hw.clone(),
            self.seq.clone(),
            self.seq_bus.clone(),
            RunSlots {
                speed: self.speed_slot.clone(),
                concurrency: self.sweep.clone(),
                niah: self.niah.clone(),
                reasoning: self.reasoning_slot.clone(),
                structured: self.structured_slot.clone(),
            },
            Some(tx),
            self.pause.clone(),
            self.logger.clone(),
        );
        tokio::spawn(async move {
            seq.run().await;
        });
    }

    /// Handle one terminal key event (blueprint §6 footer key map).
    pub fn handle_key(&mut self, key: &KeyEvent) -> KeyAction {
        // Ctrl-C quits globally — in the dashboard *and* the Setup
        // takeover (where `q` and `Esc` are phase keys, not quit keys).
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.logger.info(Context::Tui, "quit (Ctrl-C)");
            self.running = false;
            return KeyAction::Quit;
        }

        // Setup phase (full-screen takeover): every other key routes to
        // the four-stage flow. `q` types into a field; `Esc` walks back
        // one stage (quitting only at stage 1).
        if self.phase == Phase::Setup {
            return match self.setup.handle_key(key, &mut self.config) {
                SetupKeyResult::Inert => KeyAction::Continue,
                SetupKeyResult::Quit => {
                    self.running = false;
                    KeyAction::Quit
                }
                // Stage 1 `Enter` / stage 2 `d`: sync the entered target
                // into the shared form, then fire the async discovery.
                SetupKeyResult::Discover | SetupKeyResult::Retry => {
                    self.config.url = self.setup.url.trim().to_string();
                    self.logger
                        .info(Context::Setup, format!("URL entered: {}", self.config.url));
                    self.start_discovery();
                    KeyAction::Continue
                }
                // Stage 2 `Enter`: the picked (or typed) model is the target.
                SetupKeyResult::Selected => {
                    self.config.model = self.setup.confirmed_model().unwrap_or_default();
                    self.logger.info(
                        Context::Setup,
                        format!("model selected: {}", self.config.model),
                    );
                    self.config.cursor = 0;
                    self.setup.form_field = 0;
                    KeyAction::Continue
                }
                // Stage 4 `Enter`: leave the takeover and run.
                SetupKeyResult::Launched => {
                    self.launch();
                    KeyAction::Run
                }
            };
        }

        // Dashboard: the NIAH confirmation prompt is a **modal** — while
        // it is on screen, `y`/`Enter` confirms (spawning exactly one
        // run) and every other key cancels it (so a stray keypress can
        // never both answer the prompt *and* trigger a second action).
        // It is checked before the quit keys: `Esc` cancels the prompt
        // instead of quitting.
        if self.pending_niah {
            match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    self.pending_niah = false;
                    self.start_niah();
                    return KeyAction::NewNeedle;
                }
                _ => {
                    self.pending_niah = false;
                    self.push_log(
                        "[niah] cancelled — no requests sent".to_string(),
                        style::value_warn(),
                    );
                    return KeyAction::Continue;
                }
            }
        }

        // `q` quits globally — even inside the Config view (where `q` would
        // otherwise be typed into a field).
        if key.code == KeyCode::Char('q') {
            self.logger.info(Context::Tui, "quit (q)".to_string());
            self.running = false;
            return KeyAction::Quit;
        }
        // `Esc` quits — EXCEPT in the Config view, where it leaves the
        // config (to the gate / Live) instead of killing the app (FIX 4:
        // the view must never trap the user; `q` is the quit key there).
        if key.code == KeyCode::Esc && self.view != View::Config {
            self.logger.info(Context::Tui, "quit (Esc)".to_string());
            self.running = false;
            return KeyAction::Quit;
        }

        // `c` — re-open the interactive Setup takeover. Scoped away from
        // View 5, where `c` is a typeable character in a field. Locked
        // while a benchmark sequence runs: the takeover's URL stage
        // fires an HTTP discovery request, which must not land on the
        // endpoint mid-benchmark.
        if key.code == KeyCode::Char('c') && self.view != View::Config {
            if self.seq.is_running() {
                self.push_log(
                    "[setup] locked while a benchmark is running — press Space to pause, Q to quit"
                        .to_string(),
                    style::value_warn(),
                );
                return KeyAction::Continue;
            }
            self.open_setup();
            return KeyAction::Continue;
        }

        // Config view (Chunk 18 + FIX 4): the view has an **edit gate**.
        // On entry it is [`ConfigMode::Viewing`] — a read-only "press
        // Enter to edit" screen. The number keys `1`–`4` *always* switch
        // views (they are never captured by field editing), `Esc` leaves
        // the config, and `q` quits (handled above) — so the user can
        // never get stuck. Only after `Enter` does [`ConfigMode::Editing`]
        // make the fields live (there `Esc` / `1`–`4` save and exit).
        if self.view == View::Config {
            // PageDown / PageUp cycle to the neighbouring view (always
            // available, even at the gate) and reset to the read-only gate.
            if key.code == KeyCode::PageDown || key.code == KeyCode::PageUp {
                let delta = if key.code == KeyCode::PageDown {
                    1
                } else {
                    View::ALL.len() - 1
                };
                self.view = View::ALL[(self.view.index() + delta) % View::ALL.len()];
                self.config.edit_mode = ConfigMode::Viewing;
                if self.view == View::History {
                    self.ensure_history();
                }
                return KeyAction::Continue;
            }

            match self.config.edit_mode {
                // ── The gate (read-only). ──
                ConfigMode::Viewing => match key.code {
                    // `1`–`4` always switch views (never typed into a field).
                    KeyCode::Char(c @ '1'..='4') => {
                        if let Some(d) = c.to_digit(10) {
                            if let Some(view) = View::from_digit(d as u8) {
                                self.view = view;
                                self.config.edit_mode = ConfigMode::Viewing;
                                if view == View::History {
                                    self.ensure_history();
                                }
                            }
                        }
                        return KeyAction::Continue;
                    }
                    // `5` stays at the gate.
                    KeyCode::Char('5') => return KeyAction::Continue,
                    // `Enter` opens the editor.
                    KeyCode::Enter => {
                        self.config.edit_mode = ConfigMode::Editing;
                        return KeyAction::Continue;
                    }
                    // `Esc` leaves the config (back to Live).
                    KeyCode::Esc => {
                        self.view = View::Live;
                        self.config.edit_mode = ConfigMode::Viewing;
                        return KeyAction::Continue;
                    }
                    // Quick actions on the current form (no editing needed).
                    KeyCode::F(2) => {
                        if self.config.save().is_ok() {
                            self.push_log(
                                format!("[config] saved → {}", self.config.config_path.display()),
                                style::value_ok(),
                            );
                        }
                        return KeyAction::Continue;
                    }
                    KeyCode::F(5) => {
                        self.start_run();
                        return KeyAction::Run;
                    }
                    // Everything else is ignored at the gate.
                    _ => return KeyAction::Continue,
                },
                // ── Editing (the fields are live). ──
                ConfigMode::Editing => {
                    // `1`–`4` ALWAYS exit the config and switch views
                    // (saving first) — they are never typed into a field.
                    if let KeyCode::Char(c @ '1'..='4') = key.code {
                        let _ = self.config.save();
                        if let Some(d) = c.to_digit(10) {
                            if let Some(view) = View::from_digit(d as u8) {
                                self.view = view;
                                self.config.edit_mode = ConfigMode::Viewing;
                                if view == View::History {
                                    self.ensure_history();
                                }
                            }
                        }
                        return KeyAction::Continue;
                    }
                    // `5` / `Esc`: save and return to the gate.
                    if key.code == KeyCode::Char('5') || key.code == KeyCode::Esc {
                        let _ = self.config.save();
                        self.config.edit_mode = ConfigMode::Viewing;
                        return KeyAction::Continue;
                    }
                    // Everything else delegates to the field editor
                    // (Tab / arrows / F2 / F5 / typing / backspace).
                    match self.config.handle_key(key) {
                        ConfigKeyResult::Saved => {
                            self.push_log(
                                format!("[config] saved → {}", self.config.config_path.display()),
                                style::value_ok(),
                            );
                            self.logger.info(
                                Context::Setup,
                                format!("config saved → {}", self.config.config_path.display()),
                            );
                            return KeyAction::Continue;
                        }
                        ConfigKeyResult::Run => {
                            self.start_run();
                            return KeyAction::Run;
                        }
                        ConfigKeyResult::Inert => return KeyAction::Continue,
                    }
                }
            }
        }

        match key.code {
            KeyCode::Char(c @ '1'..='5') => {
                // `c as u8` is the Unicode code point (49 for '1') — use
                // the digit's *value* (1) for the view lookup.
                if let Some(view) = View::from_digit(c.to_digit(10).unwrap() as u8) {
                    self.view = view;
                    // Entering the Config view always lands on the read-only
                    // gate (FIX 4), never mid-edit.
                    if view == View::Config {
                        self.config.edit_mode = ConfigMode::Viewing;
                    }
                    // Chunk 14: entering the History view loads the stored
                    // session list once (key path, not the render path).
                    if view == View::History {
                        self.ensure_history();
                    }
                }
                KeyAction::Continue
            }
            // `Space` — pause/resume. The shared [`RunPause`] gate is the
            // real mechanism: every engine run loop holds *before*
            // spawning its next request unit, so in-flight streams
            // complete and the run resumes exactly where it left off.
            // (The render clock also freezes, as before.)
            KeyCode::Char(' ') => {
                self.paused = !self.paused;
                self.pause.set(self.paused);
                let msg = if self.paused {
                    "[pause] run paused — in-flight requests finish, no new ones start".to_string()
                } else {
                    "[pause] run resumed".to_string()
                };
                self.push_log(msg, style::value_warn());
                KeyAction::PauseResume
            }
            // `+` — step the concurrency target. Locked while a sequence
            // runs: the Engine B sweep owns the concurrency ladder, and
            // a mid-run edit would desynchronize the run from the form.
            KeyCode::Char('+') | KeyCode::Char('=') => {
                if self.seq.is_running() {
                    self.push_log(
                        "[concurrency] locked while a benchmark is running — the sweep owns the ladder"
                            .to_string(),
                        style::value_warn(),
                    );
                    return KeyAction::Continue;
                }
                self.concurrency_target = (self.concurrency_target * 2).min(128);
                KeyAction::StepConcurrency
            }
            // `n` — request a standalone NIAH matrix run. It never fires
            // directly: the key opens a `[Y/N]` confirmation that shows
            // the request count (default 7 sizes × 11 depths = 77), and
            // it is locked entirely while a benchmark sequence runs
            // (the matrix would otherwise hit the endpoint alongside
            // the running engine) or while another NIAH run is in
            // progress (one run at a time).
            KeyCode::Char('n') => {
                if self.seq.is_running() {
                    self.push_log(
                        "[niah] locked while a benchmark is running — it runs as Engine C1 in the queue"
                            .to_string(),
                        style::value_warn(),
                    );
                    return KeyAction::Continue;
                }
                if self.niah.is_running() {
                    self.push_log(
                        "[niah] already running — one size × depth cell at a time".to_string(),
                        style::value_warn(),
                    );
                    return KeyAction::Continue;
                }
                self.pending_niah = true;
                KeyAction::NewNeedle
            }
            KeyCode::Char('e') => KeyAction::Export,
            // Chunk 18: `r` runs the engines selected in the Config view.
            KeyCode::Char('r') => {
                self.start_run();
                KeyAction::Run
            }
            // Chunk 14 — History view navigation/selection. These keys are
            // scoped to the History view; elsewhere they are inert.
            KeyCode::Char('j')
            | KeyCode::Down
            | KeyCode::Char('k')
            | KeyCode::Up
            | KeyCode::Char('a')
            | KeyCode::Char('b')
                if self.view == View::History && self.history.is_some() =>
            {
                let h = self
                    .history
                    .as_mut()
                    .expect("history is Some (guard above)");
                match key.code {
                    KeyCode::Char('j') | KeyCode::Down => {
                        h.move_cursor(1);
                        KeyAction::HistoryNext
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        h.move_cursor(-1);
                        KeyAction::HistoryPrev
                    }
                    KeyCode::Char('a') => {
                        h.select_a();
                        KeyAction::HistorySelectA
                    }
                    _ => {
                        h.select_b();
                        KeyAction::HistorySelectB
                    }
                }
            }
            _ => KeyAction::Continue,
        }
    }

    /// The `e` key (blueprint §6 footer, Chunk 13): export the current
    /// metrics snapshot to `data_dir()/exports/` in the configured format
    /// and log the destination.
    ///
    /// The TUI has no persisted session to re-read, so the payload is built
    /// **live** from the lock-free [`MetricsSnapshot`] (the headless path
    /// instead consumes the SQLite layer after persistence — same
    /// [`ExportPayload`] shape). Per-packet CSV detail only exists on the
    /// headless path, where the raw stream events are captured; a TUI CSV
    /// export carries the metrics rows only.
    pub fn export(&mut self) -> Result<std::path::PathBuf, String> {
        let snap = self.metrics.load();
        // Chunk 17: with the hardware poller attached, the export rows
        // carry the live silicon-efficiency reading (cumulative joules ÷
        // tokens generated so far) and the session carries the GPU name.
        // A poisoned/absent poller degrades to N/A (`None`) — never a
        // failure of the export itself. The guard is scoped so it drops
        // before the `push_log` below reborrows `self` mutably.
        let (live_jpt, gpu_name) = {
            let poller = self.hw.as_ref().and_then(|h| h.lock().ok());
            (
                poller
                    .as_ref()
                    .and_then(|g| g.live_joules_per_token(snap.completion_tokens)),
                poller
                    .as_ref()
                    .and_then(|g| g.gpu_name().map(str::to_string)),
            )
        };
        let session = BenchmarkSession {
            session_id: uuid::Uuid::new_v4().to_string(),
            timestamp: None,
            target_url: snap.endpoint.clone(),
            model_name: snap.model.clone(),
            backend_type: (!snap.backend.is_empty()).then(|| snap.backend.clone()),
            quantization: None,
            system_gpu: gpu_name,
            total_duration_sec: None,
        };
        let metrics: Vec<StreamMetricRow> = snap
            .streams
            .iter()
            .map(|s| StreamMetricRow {
                metric_id: None,
                session_id: session.session_id.clone(),
                concurrency_level: Some(self.concurrency_target as i64),
                prompt_tokens: s.pp_tokens.map(|v| v as i64),
                completion_tokens: s.tg_tokens.map(|v| v as i64),
                reasoning_tokens: None,
                ttft_ms: s.ttft_s.map(|v| v * 1000.0),
                tpot_ms: s.gen_tps.filter(|v| *v > 0.0).map(|v| 1000.0 / v),
                mtp_efficiency: s.mtp,
                joules_per_token: live_jpt,
                cache_hit: None,
            })
            .collect();
        let payload = ExportPayload::from_live(session, metrics, Vec::new(), Vec::new());
        let dest = export::default_path_live(self.export_format);
        let path = export::write(&dest, self.export_format, &payload).map_err(|e| e.to_string())?;
        self.push_log(
            format!(
                "[export] {} → {}",
                self.export_format.label(),
                path.display()
            ),
            style::value_ok(),
        );
        Ok(path)
    }

    /// Append a line to the event log (bounded: the oldest lines drop when
    /// it grows past 200).
    pub fn push_log(&mut self, msg: String, st: Style) {
        self.log.push(Line::from(Span::styled(msg, st)));
        if self.log.len() > 200 {
            self.log.drain(0..self.log.len() - 200);
        }
    }

    /// Advance the 60Hz render clock.
    ///
    /// The metric snapshot itself is *not* written here: it is published by
    /// the stream worker / engine / hardware poller via
    /// `MetricsState::update()` and read lock-free in the render path
    /// (`render` → views → `MetricsState::load`). Keeping the write out of
    /// the render loop is the measurement-isolation invariant (blueprint
    /// §4). `Space` (paused) freezes the clock.
    ///
    /// The one derived event this tick path emits (Chunk 17): the
    /// VRAM-fragmentation warning (blueprint §5D) — logged once when
    /// occupancy crosses [`VRAM_FRAGMENTATION_THRESHOLD`], re-armed after
    /// a 5-point drop. Reading the snapshot here is a plain lock-free
    /// load; no timing path is touched.
    pub fn on_tick(&mut self) {
        // Drain the executor's log pipe first (even while paused, so the
        // bounded log catches up instead of growing unbounded in the
        // channel): the sequence task sends *real* events as they happen,
        // and the Live view renders them. Collect before mutating `self`
        // (the receiver lives inside `self`).
        let mut pending = Vec::new();
        if let Some(rx) = &self.log_rx {
            while let Ok(line) = rx.try_recv() {
                pending.push(line);
            }
        }
        for line in pending {
            self.push_log(line, style::value());
        }

        // The render clock freezes on `Space` only in the dashboard; the
        // setup flow keeps ticking (the discovery spinner advances and
        // the stage-2 hand-off below must fire).
        if self.paused && self.phase == Phase::Dashboard {
            return;
        }
        self.tick += 1;

        // Setup stage 2: the discovery request finished — materialize
        // the model list and move to the picker (tick path, never the
        // render path). A failed discovery degrades to manual entry
        // (N/A-never-fail rule).
        if self.phase == Phase::Setup
            && self.setup.phase == crate::ui::views::setup::SetupPhase::Discover
            && !self.discovery_running()
        {
            self.setup
                .complete_discovery(self.model_list(), self.discovery_error());
        }

        let m = self.metrics.load();
        let ratio = if m.vram_total_gb > 0.0 {
            (m.vram_used_gb / m.vram_total_gb).clamp(0.0, 1.0)
        } else {
            0.0
        };
        if ratio >= VRAM_FRAGMENTATION_THRESHOLD && !self.vram_warned {
            self.vram_warned = true;
            if let Some(w) = fragmentation_warning(
                (m.vram_used_gb * 1e9) as u64,
                (m.vram_total_gb * 1e9) as u64,
            ) {
                self.push_log(format!("Warning: {w}"), style::value_warn());
            }
        } else if ratio < VRAM_FRAGMENTATION_THRESHOLD * 0.95 {
            self.vram_warned = false;
        }
    }

    /// Draw the full frame.
    ///
    /// The Setup phase is a **full-screen takeover** (not one of the five
    /// tabs): it draws its own top bar, centered panel, and key-hint
    /// footer across the entire frame. The dashboard keeps the classic
    /// status bar / tab bar / view / footer layout.
    pub fn render(&self, f: &mut Frame) {
        if self.phase == Phase::Setup {
            views::setup::render(f.area(), self, f);
            return;
        }
        let area = f.area();
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // status bar
                Constraint::Length(1), // tab bar
                Constraint::Min(3),    // view content
                Constraint::Length(1), // footer
            ])
            .split(area);

        self.render_status_bar(chunks[0], f);
        self.render_tab_bar(chunks[1], f);
        views::render_current(chunks[2], self, f);
        self.render_footer(chunks[3], f);
    }

    /// Top status bar: version, backend, target model, mode (blueprint §6).
    ///
    /// Reads the shared snapshot lock-free; a fresh `Arc` is taken each frame
    /// so the bar always reflects the latest published metrics.
    fn render_status_bar(&self, area: Rect, f: &mut Frame) {
        let m = self.metrics.load();
        let mut spans = vec![
            Span::styled(
                format!(" Crucible-LLM v{} ", env!("CARGO_PKG_VERSION")),
                Style::default()
                    .fg(palette::HIGHLIGHT)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("[{}] ", m.backend), style::value()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled(format!("Target: {} ", m.model), style::label()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled(format!("Mode: {} ", m.mode), style::label()),
        ];
        // State indicators: `▸ BUSY` while any benchmark load is in
        // flight (sequence running, or a standalone NIAH run), `|| PAUSED`
        // when the Space gate is holding.
        if self.seq.is_running() || self.niah.is_running() {
            spans.push(Span::styled(
                " ▸ BUSY ",
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if self.paused {
            spans.push(Span::styled(" || PAUSED ", style::value_warn()));
        }
        f.render_widget(
            Paragraph::new(Line::from(spans)).style(style::status_bar()),
            area,
        );
    }

    /// Tab bar: `[1] Live Monitor | [2] Concurrency Matrix | ...`.
    fn render_tab_bar(&self, area: Rect, f: &mut Frame) {
        let mut spans = vec![Span::raw(" ")];
        for (i, view) in View::ALL.iter().enumerate() {
            let label = format!("[{}] {}", i + 1, view.label());
            let st = if *view == self.view {
                style::tab_active()
            } else {
                style::tab_inactive()
            };
            spans.push(Span::styled(label, st));
            if i + 1 < View::ALL.len() {
                spans.push(Span::styled(" | ", style::tab_separator()));
            }
        }
        f.render_widget(Paragraph::new(Line::from(spans)), area);
    }

    /// Bottom key-hint footer (blueprint §6) — **state-aware**: it shows
    /// exactly which keys are live in the current state, and locks the
    /// server-load keys (`N` / `C` / `+` / `R`) with a visible marker
    /// while a benchmark is in flight.
    fn render_footer(&self, area: Rect, f: &mut Frame) {
        f.render_widget(Paragraph::new(self.footer_line()), area);
    }

    /// The footer line for the current state (pure — testable without a
    /// terminal):
    ///
    /// * **confirming** (`n` pressed, `[Y/N]` on screen): the prompt
    ///   itself, with the request count and the two live keys;
    /// * **running** (sequence or standalone NIAH in flight): views,
    ///   pause/resume, export, quit — with `N`/`C`/`+`/`R` shown locked;
    /// * **idle**: views, run, NIAH (with its request count), config,
    ///   export, quit.
    pub fn footer_line(&self) -> Line<'static> {
        let sep = Span::styled(" | ", style::tab_separator());
        let live = |t: &str| Span::styled(t.to_string(), style::footer());
        let locked = |t: &str| {
            Span::styled(
                t.to_string(),
                Style::default()
                    .fg(palette::MUTED)
                    .add_modifier(Modifier::DIM),
            )
        };
        // The current view, highlighted, so a user who switched with
        // `1`-`5` always knows which dashboard panel is on screen.
        let view_tag = Span::styled(
            format!(" ▸ {} ", self.view.label()),
            Style::default()
                .fg(palette::HIGHLIGHT)
                .add_modifier(Modifier::BOLD),
        );

        if self.pending_niah {
            let requests = NIAH_SIZES.len() * NIAH_DEPTHS.len();
            return Line::from(vec![
                view_tag,
                live(&format!(
                    " [Y] Run NIAH test? {requests} requests ({s} sizes × {d} depths) ",
                    s = NIAH_SIZES.len(),
                    d = NIAH_DEPTHS.len()
                )),
                sep.clone(),
                live(" [N/Esc] Cancel "),
                sep.clone(),
                live("[Q] Quit"),
            ]);
        }

        let busy = self.seq.is_running() || self.niah.is_running();
        if busy {
            let space_key = if self.paused {
                live("[Space] Resume")
            } else {
                live("[Space] Pause")
            };
            return Line::from(vec![
                view_tag,
                live("[1-5] Views"),
                sep.clone(),
                space_key,
                sep.clone(),
                live("[E] Export"),
                sep.clone(),
                live("[Q] Quit"),
                sep.clone(),
                locked("[N] [C] [+] [R] locked"),
            ]);
        }

        let requests = NIAH_SIZES.len() * NIAH_DEPTHS.len();
        Line::from(vec![
            view_tag,
            live("[1-5] Views"),
            sep.clone(),
            live("[R] Run Benchmark"),
            sep.clone(),
            live(&format!("[N] NIAH ({requests} req, confirms)")),
            sep.clone(),
            live("[C] Config"),
            sep.clone(),
            live("[E] Export"),
            sep.clone(),
            live("[Q] Quit"),
        ])
    }
}

/// Small formatting helpers shared by the views.
pub mod fmt {
    /// `2048` or `--` when absent.
    pub fn tokens(v: Option<u64>) -> String {
        v.map(|v| v.to_string()).unwrap_or_else(|| "--".to_string())
    }

    /// `0.182 s` or `--` when absent.
    pub fn sec(v: Option<f64>) -> String {
        v.map(|v| format!("{v:.3} s"))
            .unwrap_or_else(|| "--".to_string())
    }

    /// `72.4 t/s` or `--` when absent.
    pub fn tps(v: Option<f64>) -> String {
        v.map(|v| format!("{v:.1} t/s"))
            .unwrap_or_else(|| "--".to_string())
    }

    /// `1.84 x` or `--` when absent.
    pub fn mult(v: Option<f64>) -> String {
        v.map(|v| format!("{v:.2} x"))
            .unwrap_or_else(|| "--".to_string())
    }

    /// ASCII progress bar body, e.g. `========>      ` (head is `>`).
    pub fn progress_bar(p: f64, width: usize) -> String {
        let width = width.max(1);
        let filled = (p.clamp(0.0, 1.0) * width as f64).round() as usize;
        let mut s = String::with_capacity(width);
        for i in 0..width {
            if i < filled {
                s.push(if i == filled - 1 && filled < width {
                    '>'
                } else {
                    '='
                });
            } else {
                s.push(' ');
            }
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::views::setup::SetupPhase;
    use crossterm::event::KeyEventKind;
    use crossterm::event::KeyEventState;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn char_key(c: char) -> KeyEvent {
        key(KeyCode::Char(c))
    }

    // ── phase selection at launch ───────────────────────────────────────

    #[test]
    fn new_app_starts_in_the_dashboard() {
        assert_eq!(App::new().phase, Phase::Dashboard);
    }

    #[test]
    fn with_setup_bare_config_starts_at_url_with_empty_prompt() {
        // A bare `crucible-llm` (built-in default URL, placeholder model)
        // must not leak the default into the setup prompt.
        let app = App::new().with_setup(&Config::default());
        assert_eq!(app.phase, Phase::Setup);
        assert_eq!(app.setup.phase, SetupPhase::Url);
        assert!(app.setup.url.is_empty());
    }

    #[test]
    fn with_setup_prefills_an_explicit_url_and_model() {
        let cfg = Config {
            url: "http://myhost:8000/v1".to_string(),
            model: "qwen3-72b".to_string(),
            target_explicit: true,
            ..Config::default()
        };
        let app = App::new().with_setup(&cfg);
        assert_eq!(app.setup.url, "http://myhost:8000/v1");
        assert_eq!(app.setup.url_cursor, app.setup.url.chars().count());
        assert_eq!(app.setup.model_query, "qwen3-72b");
    }

    #[test]
    fn with_setup_leaves_the_placeholder_model_blank() {
        let cfg = Config {
            target_explicit: false,
            ..Config::default()
        }; // model == "default" (placeholder)
        let app = App::new().with_setup(&cfg);
        assert!(app.setup.model_query.is_empty());
    }

    // ── `c` re-opens setup from the dashboard ───────────────────────────

    #[test]
    fn c_key_opens_setup_at_the_config_stage() {
        let mut app = App::new();
        app.config.url = "http://host:1/v1".to_string();
        app.config.model = "m1".to_string();
        let r = app.handle_key(&char_key('c'));
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.phase, Phase::Setup);
        assert_eq!(app.setup.phase, SetupPhase::Config);
        assert_eq!(app.setup.url, "http://host:1/v1");
        assert_eq!(app.setup.model_query, "m1");
    }

    #[test]
    fn c_key_in_the_config_view_is_scoped_to_editing() {
        // FIX 4: at the gate (Viewing), `c` is ignored (and never opens
        // Setup). In Editing, `c` types into the focused field.
        let mut app = App::new();
        app.view = View::Config; // gate
        app.config.cursor = 0; // URL
        app.config.url.clear(); // start from a known-empty field
        app.handle_key(&char_key('c'));
        assert_eq!(
            app.phase,
            Phase::Dashboard,
            "`c` must not open Setup from the gate"
        );
        assert_eq!(app.config.url, "", "the gate ignores typing");

        app.config.edit_mode = ConfigMode::Editing;
        app.handle_key(&char_key('c'));
        assert_eq!(
            app.phase,
            Phase::Dashboard,
            "`c` must not open Setup in Editing"
        );
        assert_eq!(app.config.url, "c", "`c` types into the URL in Editing");
    }

    // ── setup-phase key routing ─────────────────────────────────────────

    #[test]
    fn setup_q_types_into_the_url_instead_of_quitting() {
        let mut app = App::new().with_setup(&Config::default());
        let r = app.handle_key(&char_key('q'));
        assert_eq!(r, KeyAction::Continue);
        assert!(app.running, "q is printable in setup, not a quit");
        assert_eq!(app.setup.url, "q");
    }

    #[test]
    fn setup_ctrl_c_still_quits() {
        let mut app = App::new().with_setup(&Config::default());
        let r = app.handle_key(&key(KeyCode::Char('c')));
        // (Ctrl-C check requires the CONTROL modifier; a bare `c` types.)
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.setup.url, "c");
        let mut ctrl = key(KeyCode::Char('c'));
        ctrl.modifiers = KeyModifiers::CONTROL;
        let r = app.handle_key(&ctrl);
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running);
    }

    #[test]
    fn setup_esc_at_url_stage_quits() {
        let mut app = App::new().with_setup(&Config::default());
        let r = app.handle_key(&key(KeyCode::Esc));
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running);
    }

    #[test]
    fn setup_esc_in_model_stage_walks_back_to_url() {
        let mut app = App::new().with_setup(&Config::default());
        app.setup.phase = SetupPhase::Model;
        let r = app.handle_key(&key(KeyCode::Esc));
        assert_eq!(r, KeyAction::Continue);
        assert!(app.running);
        assert_eq!(app.setup.phase, SetupPhase::Url);
    }

    #[test]
    fn setup_esc_in_confirm_stage_walks_back_to_config() {
        let mut app = App::new().with_setup(&Config::default());
        app.setup.phase = SetupPhase::Confirm;
        app.handle_key(&key(KeyCode::Esc));
        assert_eq!(app.setup.phase, SetupPhase::Config);
    }

    // ── discovery hand-off (tick path) ──────────────────────────────────

    #[tokio::test]
    async fn url_enter_fires_the_async_discovery() {
        let mut app = App::new().with_setup(&Config::default());
        for c in "http://127.0.0.1:1/v1".chars() {
            app.handle_key(&char_key(c));
        }
        let r = app.handle_key(&key(KeyCode::Enter));
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.setup.phase, SetupPhase::Discover);
        // The URL synced into the shared form before the spawn.
        assert_eq!(app.config.url, "http://127.0.0.1:1/v1");
        assert!(app.discovery_running());
    }

    #[test]
    fn on_tick_completes_a_successful_discovery_into_the_picker() {
        let mut app = App::new().with_setup(&Config::default());
        app.setup.phase = SetupPhase::Discover;
        app.models.set_running(false);
        app.models.store(vec![ModelInfo {
            id: "qwen3-72b".into(),
            ..Default::default()
        }]);
        app.on_tick();
        assert_eq!(app.setup.phase, SetupPhase::Model);
        assert_eq!(app.setup.models, vec!["qwen3-72b"]);
        assert!(app.setup.error.is_none());
    }

    #[test]
    fn on_tick_degrades_a_failed_discovery_to_manual_entry() {
        let mut app = App::new().with_setup(&Config::default());
        app.setup.phase = SetupPhase::Discover;
        app.models.set_running(false);
        app.discovery_error
            .store(Some(Arc::new("connection failed: refused".to_string())));
        app.on_tick();
        assert_eq!(app.setup.phase, SetupPhase::Model);
        assert!(app.setup.models.is_empty());
        assert_eq!(
            app.setup.error.as_deref(),
            Some("connection failed: refused")
        );
    }

    // ── launch (stage 4 `Enter`) ────────────────────────────────────────

    #[tokio::test]
    async fn confirm_enter_leaves_setup_and_runs_the_benchmark() {
        let mut app = App::new().with_setup(&Config::default());
        app.config.url = "http://127.0.0.1:1/v1".to_string();
        app.config.model = "m1".to_string();
        app.setup.phase = SetupPhase::Confirm;
        let r = app.handle_key(&key(KeyCode::Enter));
        assert_eq!(r, KeyAction::Run);
        assert_eq!(app.phase, Phase::Dashboard);
        assert_eq!(app.view, View::Live);
        assert!(app.running);
        // The sequential executor is in progress now (set synchronously
        // by `start_run` before the `tokio::spawn`).
        assert!(app.seq.is_running());
    }

    // ── full walk-through: URL → model → config → launch ────────────────

    #[tokio::test]
    async fn full_setup_walk_through_reaches_the_live_view() {
        let mut app = App::new().with_setup(&Config::default());

        // Stage 1: type the URL, Enter.
        for c in "http://127.0.0.1:1/v1".chars() {
            app.handle_key(&char_key(c));
        }
        app.handle_key(&key(KeyCode::Enter));
        assert_eq!(app.setup.phase, SetupPhase::Discover);

        // The discovery "completes" (simulated slot publish).
        app.models.set_running(false);
        app.models.store(vec![
            ModelInfo {
                id: "llama-3-70b".into(),
                ..Default::default()
            },
            ModelInfo {
                id: "qwen3-72b".into(),
                ..Default::default()
            },
        ]);
        app.on_tick();
        assert_eq!(app.setup.phase, SetupPhase::Model);

        // Stage 2: pick the second model.
        app.handle_key(&key(KeyCode::Down));
        app.handle_key(&key(KeyCode::Enter));
        assert_eq!(app.setup.phase, SetupPhase::Config);
        assert_eq!(app.config.model, "qwen3-72b");

        // Stage 3: bump the iterations, Enter.
        app.setup.form_field = 2; // Iterations
        app.handle_key(&key(KeyCode::Right));
        assert_eq!(app.config.iterations, 2);
        app.handle_key(&key(KeyCode::Enter));
        assert_eq!(app.setup.phase, SetupPhase::Confirm);

        // Stage 4: Esc back, then Enter to launch.
        app.handle_key(&key(KeyCode::Esc));
        assert_eq!(app.setup.phase, SetupPhase::Config);
        app.handle_key(&key(KeyCode::Enter));
        app.handle_key(&key(KeyCode::Enter));
        assert_eq!(app.phase, Phase::Dashboard);
        assert_eq!(app.view, View::Live);
        assert!(app.seq.is_running());
    }

    // ── TUI interaction guards (the key-spam audit) ─────────────────────

    /// The footer line's plain text (spans concatenated).
    fn line_text(line: &Line) -> String {
        line.spans
            .iter()
            .map(|s| s.content.as_ref().to_string())
            .collect()
    }

    #[test]
    fn n_key_when_idle_opens_the_confirmation_not_a_run() {
        let mut app = App::new();
        let r = app.handle_key(&char_key('n'));
        assert_eq!(r, KeyAction::NewNeedle);
        assert!(app.pending_niah, "idle n opens the [Y/N] prompt");
        assert!(
            !app.niah.is_running(),
            "the prompt alone must never spawn requests"
        );
    }

    #[test]
    fn n_key_is_locked_while_a_sequence_runs() {
        let mut app = App::new();
        app.seq.set_running(true); // e.g. Engine A in progress
        app.handle_key(&char_key('n'));
        assert!(!app.pending_niah, "no confirmation may open mid-sequence");
        assert!(
            !app.niah.is_running(),
            "a second NIAH runner must not hit the server alongside the sequence"
        );
        assert!(app
            .log
            .iter()
            .any(|l| line_text(l).contains("locked while a benchmark is running")));
    }

    #[test]
    fn n_key_is_locked_while_a_standalone_niah_run_is_active() {
        let mut app = App::new();
        app.niah.set_running(true);
        app.handle_key(&char_key('n'));
        assert!(!app.pending_niah, "one NIAH run at a time");
    }

    #[tokio::test]
    async fn n_confirmation_y_fires_exactly_one_run() {
        let mut app = App::new();
        // Unreachable endpoint: the spawned run fails fast, no load.
        app.config.url = "http://127.0.0.1:9/v1".to_string();
        app.handle_key(&char_key('n'));
        assert!(app.pending_niah);
        app.handle_key(&char_key('y'));
        assert!(!app.pending_niah);
        assert!(app.niah.is_running(), "confirmed y spawns the run");
        // A further press while running stays locked.
        app.handle_key(&char_key('n'));
        assert!(!app.pending_niah);
    }

    #[test]
    fn n_confirmation_n_cancels_and_sends_nothing() {
        let mut app = App::new();
        app.handle_key(&char_key('n'));
        assert!(app.pending_niah);
        app.handle_key(&char_key('n')); // the prompt's own cancel key
        assert!(!app.pending_niah);
        assert!(!app.niah.is_running());
    }

    #[test]
    fn n_confirmation_esc_cancels_instead_of_quitting() {
        let mut app = App::new();
        app.handle_key(&char_key('n'));
        assert!(app.pending_niah);
        let r = app.handle_key(&key(KeyCode::Esc));
        assert_eq!(
            r,
            KeyAction::Continue,
            "Esc answers the prompt, not the app"
        );
        assert!(!app.pending_niah);
        assert!(app.running, "the app keeps running");
        assert!(!app.niah.is_running());
    }

    #[test]
    fn n_confirmation_other_keys_cancel_and_are_swallowed() {
        let mut app = App::new();
        app.handle_key(&char_key('n'));
        assert!(app.pending_niah);
        app.handle_key(&char_key('1')); // a stray key
        assert!(!app.pending_niah);
        assert!(!app.niah.is_running());
        assert_eq!(
            app.view,
            View::Live,
            "the swallowed key must not double-act (no view switch)"
        );
    }

    #[test]
    fn plus_key_is_locked_while_a_sequence_runs() {
        let mut app = App::new();
        app.seq.set_running(true);
        let before = app.concurrency_target;
        app.handle_key(&char_key('+'));
        assert_eq!(
            app.concurrency_target, before,
            "the sweep owns the ladder mid-run"
        );
    }

    #[test]
    fn plus_key_steps_up_when_idle() {
        let mut app = App::new();
        app.handle_key(&char_key('+'));
        assert_eq!(app.concurrency_target, 2);
        app.handle_key(&char_key('='));
        assert_eq!(app.concurrency_target, 4);
    }

    #[test]
    fn c_key_is_locked_while_a_sequence_runs() {
        let mut app = App::new();
        app.seq.set_running(true);
        app.handle_key(&char_key('c'));
        assert_eq!(
            app.phase,
            Phase::Dashboard,
            "no setup takeover (and no discovery request) mid-run"
        );
    }

    #[test]
    fn space_toggles_the_shared_pause_gate() {
        let mut app = App::new();
        assert!(!app.pause.is_paused());
        let r = app.handle_key(&char_key(' '));
        assert_eq!(r, KeyAction::PauseResume);
        assert!(app.paused);
        assert!(
            app.pause.is_paused(),
            "the engine gate is set with the flag"
        );
        app.handle_key(&char_key(' '));
        assert!(!app.paused);
        assert!(!app.pause.is_paused(), "resume clears the gate");
    }

    // ── view switching (1-5) — the #1 priority ───────────────────────────

    #[test]
    fn digits_switch_views_from_every_dashboard_view() {
        for view in View::ALL {
            let mut app = App::new();
            app.view = view;
            // In the Config view the cursor starts on the URL (a text
            // field where digits type); park it on a step field so the
            // digits switch views.
            if view == View::Config {
                app.config.cursor = 2; // Field::Mode
            }
            for (digit, expected) in [
                ('1', View::Live),
                ('2', View::Concurrency),
                ('3', View::Needle),
                ('4', View::History),
                ('5', View::Config),
            ] {
                app.handle_key(&char_key(digit));
                assert_eq!(app.view, expected, "from {view:?}, {digit} → {expected:?}");
                // After leaving and re-entering the Config view mid-loop,
                // the cursor keeps its (non-text) position.
                if app.view == View::Config {
                    app.config.cursor = 2;
                }
            }
        }
    }

    #[test]
    fn digits_switch_views_while_a_sequence_runs() {
        // Views must be switchable at ALL times — even mid-benchmark.
        let mut app = App::new();
        app.seq.set_running(true);
        app.view = View::Live;
        app.handle_key(&char_key('2'));
        assert_eq!(app.view, View::Concurrency, "mid-run view switch works");
        app.handle_key(&char_key('4'));
        assert_eq!(app.view, View::History);
        app.handle_key(&char_key('1'));
        assert_eq!(app.view, View::Live);
    }

    #[test]
    fn config_view_digits_switch_views_from_step_fields() {
        // From a non-text field (Tokens = index 3) the digits switch views
        // instead of typing.
        let mut app = App::new();
        app.view = View::Config;
        app.config.cursor = 3; // Field::Tokens
        app.handle_key(&char_key('2'));
        assert_eq!(app.view, View::Concurrency);
        assert_eq!(
            app.config.tokens,
            crate::config::DEFAULT_TOKENS,
            "the digit must not edit the field"
        );
    }

    #[test]
    fn config_view_digits_always_switch_views() {
        // FIX 4: number keys 1-4 are NEVER captured by the config editor —
        // they always switch views, from the gate *and* from edit mode.
        // From the gate, `1` leaves straight to Live.
        let mut app = App::new();
        app.view = View::Config; // gate (Viewing)
        app.handle_key(&char_key('1'));
        assert_eq!(app.view, View::Live, "digit 1 exits the gate to Live");

        // In Editing mode, a non-navigation digit (8) still types into the
        // focused field, but a 1-4 digit exits and switches views.
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        app.config.cursor = 9; // Field::Ladder
        app.config.ladder.clear();
        // Point the save (digit-exit saves) at a temp path so the test
        // never touches the real config file.
        app.config.config_path =
            std::env::temp_dir().join(format!("crucible-digit-{}.json", std::process::id()));
        app.handle_key(&char_key('8'));
        assert_eq!(app.view, View::Config, "8 types into the field");
        assert_eq!(app.config.ladder, "8");
        app.handle_key(&char_key('1'));
        assert_eq!(app.view, View::Live, "1 exits the config and switches");
        let _ = std::fs::remove_file(&app.config.config_path);
    }

    #[test]
    fn footer_highlights_the_current_view() {
        let mut app = App::new();
        app.view = View::Concurrency;
        let t = line_text(&app.footer_line());
        assert!(
            t.contains("Concurrency Matrix"),
            "footer names the view: {t}"
        );
    }

    // ── state-aware footer ───────────────────────────────────────────────

    #[test]
    fn footer_is_idle_when_nothing_runs() {
        let app = App::new();
        let t = line_text(&app.footer_line());
        assert!(t.contains("[R] Run Benchmark"), "{t}");
        assert!(t.contains("[N] NIAH (77 req, confirms)"), "{t}");
        assert!(t.contains("[C] Config"), "{t}");
        assert!(!t.contains("locked"), "{t}");
    }

    #[test]
    fn footer_locks_the_server_keys_while_running() {
        let app = App::new();
        app.seq.set_running(true);
        let t = line_text(&app.footer_line());
        assert!(t.contains("[Space] Pause"), "{t}");
        assert!(t.contains("[1-5] Views"), "{t}");
        assert!(t.contains("locked"), "{t}");
    }

    #[test]
    fn footer_shows_resume_when_paused() {
        let mut app = App::new();
        app.seq.set_running(true);
        app.handle_key(&char_key(' '));
        let t = line_text(&app.footer_line());
        assert!(t.contains("[Space] Resume"), "{t}");
        assert!(t.contains("locked"), "{t}");
    }

    #[test]
    fn footer_shows_the_confirmation_prompt() {
        let mut app = App::new();
        app.handle_key(&char_key('n'));
        let t = line_text(&app.footer_line());
        assert!(
            t.contains("[Y] Run NIAH test? 77 requests (7 sizes × 11 depths)"),
            "{t}"
        );
        assert!(t.contains("[N/Esc] Cancel"), "{t}");
    }
}
