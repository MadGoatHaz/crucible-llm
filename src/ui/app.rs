//! `App` state machine: the `View` enum (`Live`, `Concurrency`, `Needle`,
//! `Gpu`, `History`, `Config`) and key handling (blueprint §6 footer:
//! `1`-`6` switch views, `Space` pause/resume, `+` step concurrency,
//! `n` new needle, `e` export, `q` quit — with a `[y/N]` confirmation
//! overlay).
//!
//! **State-aware key guards** (the TUI interaction audit): a benchmark
//! is a *server-load* event, so while a `BenchmarkSequence` (or any
//! standalone engine run) is in progress the keys that would add load —
//! `n` (NIAH matrix), `+` (concurrency step), `c` (setup takeover, whose
//! URL stage fires an HTTP discovery) — are locked and log why. The
//! always-available keys are view switching (`1`-`6`), `Space`
//! (pause/resume: the shared [`RunPause`] gate holds each engine before
//! its next request, in-flight streams complete, resume continues where
//! it left off), `e` (local file export), and `q` (quit — the *only* quit
//! key: it opens a `[y/N]` confirmation overlay, and only a confirmed `y`
//! ends the app). `Ctrl-C` is inert (the terminal's copy selection owns
//! it) and `Esc` never quits outright (in Setup stage 1 it opens the same
//! confirmation; in the dashboard it leaves the Config view). The
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
use std::time::{SystemTime, UNIX_EPOCH};

use arc_swap::ArcSwapOption;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::client::models::ModelInfo;
use crate::config::{Config, ExportFormat};
use crate::engines::hardware::profile;
use crate::engines::sequence::sum_all_engines_tokens;
use crate::engines::{
    fragmentation_warning, queue_for, speed_mean_tps, BenchmarkSequence, EngineProgress,
    NiahEngineConfig, NiahSlot, ProgressBus, ReasoningResult, ResultSlot, RunPause, RunSlots,
    SeqPhase, SeqState, SeqStateSlot, SpeedResult, StructuredResult, SweepResult, NIAH_DEPTHS,
    NIAH_SIZES, VRAM_FRAGMENTATION_THRESHOLD, WINDOW_SECS,
};
use crate::hw::{arm_power_monitor, GpuBackend, HwPoller, IDLE_WINDOW_SECS, MANUAL_IDLE_WINDOW_SECS};
use crate::log::{read_log_file, Context, RunLogger};
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::report::MatrixReport;
use crate::storage::db::Database;
use crate::storage::export::{self, ExportPayload};
use crate::storage::models::{
    BenchmarkSession, ConcurrencySummary, EngineResults, EnergySummary, FlatOutSummary,
    NeedleEvaluation, ReasoningSummary, StreamMetricRow, StructuredSummary,
};
use crate::ui::theme::{self, glyph, style, Theme};
use crate::ui::views;
use crate::ui::views::config::{ConfigKeyResult, ConfigMode, ConfigState};
use crate::ui::views::history::{HistoryMode, HistoryState};
use crate::ui::views::log::{LogKeyResult, LogState};
use crate::ui::views::report::{ReportKeyResult, ReportState};
use crate::ui::views::setup::{SetupKeyResult, SetupState};

/// The six dashboard views (blueprint §6).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum View {
    /// View 1 — Live Monitor & Telemetry.
    Live,
    /// View 2 — Concurrency & Saturation matrix.
    Concurrency,
    /// View 3 — Needle-in-a-Haystack matrix.
    Needle,
    /// View 4 — GPU & Power Monitor (the dedicated hardware / energy panel).
    Gpu,
    /// View 5 — Historical Comparison & Diff.
    History,
    /// View 6 — Configuration.
    Config,
}

impl View {
    /// All views in tab-bar order.
    pub const ALL: [View; 6] = [
        View::Live,
        View::Concurrency,
        View::Needle,
        View::Gpu,
        View::History,
        View::Config,
    ];

    /// 0-based index (position in the tab bar).
    pub fn index(self) -> usize {
        match self {
            View::Live => 0,
            View::Concurrency => 1,
            View::Needle => 2,
            View::Gpu => 3,
            View::History => 4,
            View::Config => 5,
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
            View::Gpu => "GPU & Power",
        }
    }

    /// Map a digit (`1`..=`6`) to a view (the tab-bar order).
    pub fn from_digit(d: u8) -> Option<View> {
        match d {
            1 => Some(View::Live),
            2 => Some(View::Concurrency),
            3 => Some(View::Needle),
            4 => Some(View::Gpu),
            5 => Some(View::History),
            6 => Some(View::Config),
            _ => None,
        }
    }
}

/// The TUI phase: the pre-dashboard **Setup** takeover, or the
/// six-view dashboard itself.
///
/// Setup is *not* a seventh tab: it is a full-screen flow (URL → model
/// discovery/selection → benchmark config → launch) that occupies the
/// whole frame with its own top bar and footer. `App::new()` starts in
/// [`Phase::Dashboard`] (the previous behavior); the entry point moves it
/// to [`Phase::Setup`] when the target (URL + model) was not given
/// explicitly, and the `c` key re-opens Setup from any dashboard view
/// (except View 6, where `c` is a typeable character).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Phase {
    /// The first-run theme picker (full-screen takeover, shown before
    /// Setup when no theme has ever been chosen).
    ThemePicker,
    /// The interactive setup flow (full-screen takeover).
    Setup,
    /// The post-run Matrix Report takeover (full-screen): opens via `m`
    /// after a completed run; `Esc` closes it.
    Report,
    /// The run-log viewer (full-screen takeover): opens via `l` from any
    /// dashboard view with **no completion gate** (the log exists from
    /// process start); `Esc` closes it.
    Log,
    /// The six-view dashboard (Live / Concurrency / Needle / GPU / History
    /// / Config).
    #[default]
    Dashboard,
}

/// Semantic key actions the engine layer consumes (plumbed to the worker
/// pool / storage in later chunks).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyAction {
    /// Nothing to do — keep running.
    Continue,
    /// The quit was confirmed (`q` → the `[y/N]` overlay → `y`). Ctrl-C is
    /// inert and `Esc` never quits outright.
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
    /// The active color theme. Every view reads its colors from this —
    /// switching it re-skins the whole TUI (first-run picker or the
    /// Config view's Theme field).
    pub active_theme: Theme,
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
    /// Lock-free Engine F (Flat Out) result holder.
    pub flatout_slot: Arc<ResultSlot<crate::engines::flatout::FlatOutResult>>,
    /// The editable Configuration form (View 6, Chunk 18). Seeded from the
    /// resolved [`Config`]; edited on the key path; `F2` persists it and
    /// `F5`/`r` runs the selected engines from it.
    pub config: ConfigState,
    /// The 100 ms hardware telemetry poller (Chunk 17): a background
    /// task publishes VRAM / power / clock / J-token into the shared
    /// `ArcSwap<MetricsSnapshot>`; the views read it lock-free. `None`
    /// until the entry point attaches one.
    pub hw: Option<Arc<Mutex<HwPoller>>>,
    /// The detected GPU backend (any vendor), shared with the poller.
    /// The views read its vendor + model for the GPU panel title and the
    /// "D: Energy" label; the live telemetry itself lives in the
    /// `MetricsSnapshot.gpu` field (updated by the 100 ms poller).
    /// `None` when no GPU is present (the GPU panel is hidden).
    pub gpu: Option<Arc<dyn GpuBackend>>,
    /// Lock-free model-discovery slot (Chunk 20): a background
    /// `tokio::spawn`ed `GET {base}/v1/models` (OpenAI-compatible)
    /// publishes the discovered [`ModelInfo`] list here; the setup flow
    /// / Config view read it lock-free via
    /// [`model_list`](Self::model_list) (measurement-isolation
    /// invariant, blueprint §4). `None` until the first discovery
    /// completes; a failed discovery leaves the previous list (if any)
    /// in place — the picker degrades to free text (N/A-never-fail).
    pub models: Arc<ResultSlot<Vec<ModelInfo>>>,
    /// The TUI phase: the Setup / Matrix Report takeovers or the
    /// dashboard (the entry point picks; `c` re-opens Setup and `m`
    /// re-opens the report from the dashboard).
    pub phase: Phase,
    /// The interactive setup flow state (the four-stage URL → model →
    /// config → launch sequence). Mutated only in the key/tick path.
    pub setup: SetupState,
    /// The post-run Matrix Report takeover state (the `m` key after a
    /// completed run): the assembled [`MatrixReport`] + the user-fill
    /// field buffers. Mutated only in the key path (the report is
    /// re-assembled on every open).
    pub report: ReportState,
    /// The run-log viewer takeover state (the `l` key — no completion
    /// gate: the log exists from process start): a read-only snapshot of
    /// the run's `latest.log` (or a stored session's archive log from the
    /// History view). Mutated only in the key path (re-read on every
    /// open).
    pub log_view: LogState,
    /// The first-run theme picker state (full-screen takeover shown before
    /// Setup when no theme has ever been chosen).
    pub theme_picker: crate::ui::views::theme_picker::ThemePickerState,
    /// `true` when the target URL was provided explicitly (CLI / env /
    /// config file) — captured at startup so the theme picker knows
    /// whether to fall through to Setup (fresh target) or the Dashboard
    /// (target already given) after a theme is chosen.
    pub target_explicit: bool,
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
    /// `true` while the quit confirmation overlay ("Quit crucible-llm?
    /// [y] Yes  [n] No") is on screen — a modal that swallows every key
    /// but `y`/`Y`/`Enter` (confirm → `running = false`) and anything
    /// else (cancel). `q` is the *only* quit key (Ctrl-C is inert so the
    /// terminal's copy selection works, and Esc never quits outright).
    /// It works in both the dashboard and the Setup takeover (where
    /// `Esc` at stage 1 opens it, since `q` is a typeable character
    /// there).
    pub pending_quit: bool,
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
    /// R2 (traceability): this run's session UUID, shared with the
    /// run-log (stamped in its header/archive name by `main`) and used as
    /// the `benchmark_sessions.session_id` the sequence persists — so the
    /// TUI's stored run joins to its log. `App::new()` mints a fresh one
    /// (tests); the entry point overrides it via [`Self::with_session_id`].
    pub session_id: String,
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
            active_theme: Theme::default(),
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
            flatout_slot: Arc::new(ResultSlot::new()),
            config: ConfigState::default(),
            hw: None,
            gpu: None,
            models: Arc::new(ResultSlot::new()),
            // `App::new()` keeps the classic behavior (dashboard first);
            // the entry point switches to Setup via `with_setup` when the
            // target was not fully pre-configured.
            phase: Phase::Dashboard,
            setup: SetupState::new(),
            report: ReportState::default(),
            log_view: LogState::default(),
            theme_picker: crate::ui::views::theme_picker::ThemePickerState::default(),
            target_explicit: false,
            discovery_error: Arc::new(ArcSwapOption::empty()),
            seq: Arc::new(SeqStateSlot::new()),
            seq_bus: Arc::new(ProgressBus::new()),
            pause: Arc::new(RunPause::new()),
            pending_niah: false,
            pending_quit: false,
            log_rx: None,
            vram_warned: false,
            // Tests use `App::new()` directly: a disabled logger keeps
            // them off the filesystem (the entry point attaches the real
            // one via `with_logger`).
            logger: RunLogger::disabled(),
            // A fresh session id per app (the entry point overrides it
            // with the one shared by the run-log via `with_session_id`).
            session_id: uuid::Uuid::new_v4().to_string(),
        }
    }

    /// R2 (traceability): set this run's session UUID (the one `main`
    /// generated and stamped into the run-log) so the persisted
    /// `benchmark_sessions` row joins to the log.
    pub fn with_session_id(mut self, session_id: String) -> Self {
        self.session_id = session_id;
        self
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

    /// Seed the editable Configuration form (View 6, Chunk 18) from the
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
            // The backend is unknown (vLLM, llama.cpp, LM Studio, SGLang,
            // …) — never assume one; the status bar shows the endpoint
            // host instead.
            backend: String::new(),
            ..Default::default()
        });
        self
    }

    /// `true` when the NIAH context sizes are the `chars/4` estimate —
    /// i.e. no `--tokenizer` was supplied (Issue #4).
    ///
    /// The NIAH grid (View 3) and the Live view's Long Context capability
    /// bar label sizes with the `(approx.)` qualifier in that case — the
    /// same qualifier the headless run log and the `--json` `niah` section
    /// render via the shared `context_size_label`, so a no-tokenizer run is
    /// never presented as exact anywhere. The NIAH engine consumes
    /// [`niah_config`](Self::niah_config) (its `tokenizer` field is the
    /// source of truth); before the entry point attaches one, the editable
    /// form's tokenizer field is the fallback.
    #[must_use]
    pub fn niah_approximate(&self) -> bool {
        match &self.niah_config {
            Some(c) => c.tokenizer.is_none(),
            None => self
                .config
                .tokenizer
                .as_deref()
                .is_none_or(|t| t.trim().is_empty()),
        }
    }

    /// Attach the hardware telemetry poller (Chunk 17). The entry point
    /// spawns the 100 ms task that drives `hw.tick(&app.metrics)`; the
    /// render path only ever reads the resulting snapshot.
    pub fn with_hw(mut self, hw: Arc<Mutex<HwPoller>>) -> Self {
        self.hw = Some(hw);
        self
    }

    /// Attach the detected GPU backend (any vendor). The entry point
    /// shares the single detection result here and in the poller; the
    /// views read its vendor + model for the GPU panel title and the
    /// "D: Energy" label.
    pub fn with_gpu(mut self, gpu: Option<Arc<dyn GpuBackend>>) -> Self {
        self.gpu = gpu;
        self
    }

    /// Set the active color theme (the entry point calls this with the
    /// theme resolved from `--theme` / env / config file / default).
    pub fn with_theme(mut self, theme: Theme) -> Self {
        self.active_theme = theme;
        self
    }

    /// Enter the first-run **theme picker** (the full-screen takeover shown
    /// before Setup when no theme has ever been chosen). `needs_setup`
    /// records whether, once a theme is confirmed, the flow should fall
    /// through to the Setup takeover (fresh target) or straight to the
    /// Dashboard (the target URL + model were already given).
    pub fn with_theme_picker(mut self, needs_setup: bool) -> Self {
        self.phase = Phase::ThemePicker;
        self.theme_picker = crate::ui::views::theme_picker::ThemePickerState {
            cursor: 0,
            needs_setup,
        };
        self
    }

    /// Seed the Setup flow from the current Configuration form: an
    /// *explicitly* provided URL is pre-filled in stage 1 (a built-in
    /// default is not — the setup flow starts with an empty prompt, per the
    /// "no leaked LAN IP" rule), and a non-placeholder model name is
    /// pre-filled as the stage-2 manual-entry text.
    fn enter_setup(&mut self) {
        self.phase = Phase::Setup;
        if self.target_explicit {
            self.setup.url = self.config.url.clone();
            self.setup.url_cursor = self.setup.url.chars().count();
        }
        if self.config.model != crate::config::DEFAULT_MODEL {
            self.setup.model_query = self.config.model.clone();
            self.setup.model_query_cursor = self.setup.model_query.chars().count();
        }
    }

    /// Enter the interactive **Setup** phase (the full-screen
    /// pre-dashboard takeover), pre-filling from the resolved [`Config`].
    ///
    /// The entry point calls this only when the target (URL + model) was
    /// not fully given on the command line / env / config file.
    pub fn with_setup(mut self, cfg: &Config) -> Self {
        self.target_explicit = cfg.target_explicit;
        // Seed the config form from the resolved config so the setup
        // prefill reads the right URL/model (the entry point also calls
        // `with_config`, but this keeps `with_setup` self-sufficient).
        self.config = ConfigState::from_config(cfg);
        self.enter_setup();
        self
    }

    /// Confirm a theme in the first-run picker (the `Enter` key path):
    /// apply it to the active theme, persist it to the config file, and
    /// fall through to Setup (fresh target) or the Dashboard (target
    /// already given).
    pub fn confirm_theme(&mut self) {
        let chosen = Theme::ALL[self.theme_picker.cursor.min(Theme::ALL.len() - 1)];
        self.active_theme = chosen;
        self.config.theme = chosen.id().to_string();
        // Persist so subsequent runs skip the picker (`theme_explicit` is
        // now satisfied by the saved file).
        let _ = self.config.save();
        if self.theme_picker.needs_setup {
            self.enter_setup();
        } else {
            self.phase = Phase::Dashboard;
            self.view = View::Live;
        }
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

    /// Open the **Matrix Report** takeover (the `m` key, gated on a
    /// completed run): assemble the report from the run's result slots +
    /// hardware telemetry (the auto fields freeze at open; the user
    /// fields start empty), pre-fill the save path, and take over the
    /// screen. Re-opening after `Esc` re-assembles the auto fields and
    /// keeps the user's in-progress edits (the buffers are re-seeded
    /// from the fresh report only for the auto fields — see
    /// [`build_matrix_report`](Self::build_matrix_report)).
    pub fn open_report(&mut self) {
        let report = self.build_matrix_report();
        let ts = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let save_path = format!(
            "{}/exports/matrix-{ts}.md",
            crate::config::data_dir().display()
        );
        // Preserve the user's in-progress edits across close/re-open:
        // the fresh assembly carries the auto fields; the four user
        // buffers keep whatever the user has already typed.
        let (backend, quant, use_case, notes) = (
            self.report.backend.clone(),
            self.report.quant.clone(),
            self.report.use_case.clone(),
            self.report.notes.clone(),
        );
        self.report = ReportState::new(report, save_path);
        self.report.backend = backend;
        self.report.backend_cursor = self.report.backend.chars().count();
        self.report.quant = quant;
        self.report.quant_cursor = self.report.quant.chars().count();
        self.report.use_case = use_case;
        self.report.use_case_cursor = self.report.use_case.chars().count();
        self.report.notes = notes;
        self.report.notes_cursor = self.report.notes.chars().count();
        self.report.sync_report();
        self.phase = Phase::Report;
    }

    /// Open the **run-log viewer** (the `l` key): read the current run's
    /// `latest.log` into the takeover and take over the screen.
    ///
    /// **No completion gate** (unlike `m`): the run log exists from
    /// process start, so `l` works before, during, and after a run. The
    /// content is a *snapshot* taken at open (the file keeps growing;
    /// re-`l` after `Esc` re-reads a fresh copy). N/A-never-fail: a
    /// disabled logger (no path) or an unreadable file yields the
    /// placeholder view, never a crash.
    pub fn open_log(&mut self) {
        match self.logger.path() {
            Some(p) => match read_log_file(p) {
                Ok(lines) => {
                    self.log_view =
                        LogState::new(lines, p.display().to_string(), Self::log_save_path(ts_secs()));
                }
                Err(e) => {
                    self.log_view = LogState::unavailable(format!(
                        "cannot read {}: {e}",
                        p.display()
                    ));
                }
            },
            None => self.log_view =
                LogState::unavailable("run log unavailable (logger disabled)".to_string()),
        }
        self.phase = Phase::Log;
    }

    /// Open a **stored session's** archive log (the History view's `l`
    /// key): find `run-<stamp>-<session8>.log` in the run-log directory
    /// and read it into the takeover.
    ///
    /// The directory is the current run's log dir (`logger.path()`'s
    /// parent) — archives from past sessions live alongside it — falling
    /// back to `data_dir()/logs` when the logger is disabled. The match
    /// is on the archive name's `-{session8}` suffix (the first 8 chars
    /// of the session UUID, `log.rs` `short_id` semantics); when more
    /// than one matches (a same-second collision, `-2` suffix —
    /// practically impossible for a unique UUID), the lexicographically
    /// greatest name wins (deterministic). N/A-never-fail: no match
    /// yields the placeholder view, never a crash.
    pub fn open_log_for_session(&mut self, session_id: &str) {
        let short8: String = session_id.chars().take(8).collect();
        let dir = self
            .logger
            .path()
            .and_then(|p| p.parent().map(std::path::PathBuf::from))
            .unwrap_or_else(|| crate::config::data_dir().join("logs"));
        let needle = format!("-{short8}");
        let mut best: Option<std::path::PathBuf> = None;
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();
                if name.starts_with("run-") && name.ends_with(".log") && name.contains(&needle) {
                    let path = dir.join(&name);
                    let greater = best
                        .as_ref()
                        .map_or(true, |b| path.to_string_lossy() > b.to_string_lossy());
                    if greater {
                        best = Some(path);
                    }
                }
            }
        }
        match best {
            Some(p) => match read_log_file(&p) {
                Ok(lines) => {
                    self.log_view =
                        LogState::new(lines, p.display().to_string(), Self::log_save_path(ts_secs()));
                }
                Err(e) => {
                    self.log_view = LogState::unavailable(format!(
                        "cannot read {}: {e}",
                        p.display()
                    ));
                }
            },
            None => self.log_view =
                LogState::unavailable(format!("no archive log found for session {short8}")),
        }
        self.phase = Phase::Log;
    }

    /// The `c` key's log message for a run-log clipboard write outcome
    /// (the N/A-never-fail rule, mirroring [`report_copy_message`] — a
    /// terminal that drops the OSC 52 sequence is a logged warning,
    /// never a crash; save-to-file is the guaranteed path). Pure over
    /// the outcome so the handler logic is testable without a terminal.
    #[must_use]
    pub fn log_copy_message(res: &Result<(), String>) -> String {
        match res {
            Ok(()) => "[log] copied to clipboard (OSC 52)".to_string(),
            Err(_) => "[log] clipboard unsupported by terminal — use [S]ave".to_string(),
        }
    }

    /// The `s` key's save destination: `data_dir()/exports/run-log-<ts>.log`
    /// (`exports` is the established user-save location — the same dir
    /// the Matrix Report saves to). Pure over the timestamp so it is
    /// unit-testable.
    #[must_use]
    pub fn log_save_path(ts: u64) -> String {
        format!(
            "{}/exports/run-log-{ts}.log",
            crate::config::data_dir().display()
        )
    }

    /// Stage-4 `Enter`: leave the Setup takeover, land on the Live view,
    /// and start the benchmark (the selected engines run against the
    /// shared Configuration form the setup flow just filled in).
    pub fn launch(&mut self) {
        let th = self.active_theme;
        self.phase = Phase::Dashboard;
        self.view = View::Live;
        self.paused = false;
        // FIX 2: a new run re-arms the metrics pipeline (the previous
        // run's freeze must not leak into this one).
        self.metrics.unfreeze();
        // Refresh the snapshot identity from the current config so the
        // status bar and panels show the real target (not stale/zeroed
        // values from before the setup flow filled in the form).
        let cfg = self.config.to_config();
        self.metrics.update(MetricsSnapshot {
            endpoint: cfg.url.clone(),
            model: cfg.model.clone(),
            mode: cfg.mode.label().to_string(),
            // The backend is unknown (vLLM, llama.cpp, LM Studio, SGLang,
            // …) — never assume one; the status bar shows the endpoint
            // host instead.
            backend: String::new(),
            ..Default::default()
        });
        self.push_log(
            format!("[setup] launching: {} / {}", cfg.url, cfg.model),
            style::value_ok(th),
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
        let path = Database::default_path();
        if let Some(h) = self.history.as_mut() {
            // Reload the session list (picks up any runs persisted since
            // the last visit). The mode/cursor are preserved.
            h.db_path = path;
            h.reload();
        } else {
            self.history = HistoryState::load(&path).ok();
        }
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
        let th = self.active_theme;
        if self.seq.is_running() {
            self.push_log(
                "[niah] locked while a benchmark sequence is running — \
                  it runs as Engine C1 in the queue"
                    .to_string(),
                style::value_warn(th),
            );
            return;
        }
        if self.niah.is_running() {
            self.push_log(
                "[niah] already running — one size × depth cell at a time".to_string(),
                style::value_warn(th),
            );
            return;
        }
        // FIX 2: a standalone run is a run — re-arm the metrics pipeline
        // (a previous completed run froze it), and freeze it again when
        // this one finishes (below) so the numbers stop drifting.
        self.metrics.unfreeze();
        // Chunk 18: the `n` key runs against the *current* Configuration
        // form, so edits made in View 6 apply immediately.
        let config = NiahEngineConfig::from_config(&self.config.to_config());
        let engine = match config.engine() {
            Ok(e) => e,
            Err(e) => {
                self.push_log(
                    format!("[niah] engine init failed: {e}"),
                    style::value_err(th),
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
            style::value_ok(th),
        );
        // The runner owns the slot handle; on completion it clears the
        // running flag and publishes the scored grid.
        let logger = self.logger.clone();
        let engine = engine.pause(self.pause.clone()).logger(logger.clone());
        let metrics = self.metrics.clone();
        let hw = self.hw.clone();
        tokio::spawn(async move {
            logger.info(
                Context::EngineC1,
                format!(
                    "standalone NIAH run started — {requests} requests → {url}",
                    url = config.url
                ),
            );
            // B5: arm the power monitor like the sequence path (begin_run →
            // idle window → start_load) so View 4's load window / duration is
            // fresh for the standalone run — it used to only *close* the
            // window at the end without ever opening one, leaving the
            // duration stale.
            if let Some(p) = &hw {
                arm_power_monitor(p, IDLE_WINDOW_SECS).await;
            }
            let result = engine.run().await;
            slot.set_running(false);
            slot.store(result);
            // The run is over: close the GPU monitor's load window (the
            // duration freezes at its final value — a standalone run is a
            // run) and publish that final state *before* freezing the
            // metrics pipeline: the 100 ms poller task (the monitor's only
            // publisher) stops calling `tick()` once frozen, and
            // `update()` is a no-op while frozen, so a post-freeze publish
            // would never reach View 4 (the panel would keep showing the
            // pre-`end_load` monitor: duration counting forever, avg power
            // decaying forever).
            if let Some(p) = &hw {
                if let Ok(mut poller) = p.lock() {
                    poller.end_load();
                    poller.publish_monitor(&metrics);
                }
            }
            // FIX 2: freeze the metrics pipeline (the hardware poller goes
            // idle; the UI shows final numbers).
            metrics.freeze();
        });
    }

    /// Arm a manual idle-baseline measurement (the `[i]` GPU-tab action):
    /// the 100 ms hardware poller sits the system no-load for
    /// [`MANUAL_IDLE_WINDOW_SECS`] and records the mean as the new idle
    /// floor — a clean reference point the user controls, not just the
    /// automatic 5 s pre-test window. Pressing `[i]` again cancels an
    /// in-progress measurement. Locked while a benchmark sequence or a
    /// standalone NIAH run is in progress (those own the power window); a
    /// driver-less host (no poller) degrades to a log line — the
    /// N/A-never-fail rule.
    pub fn measure_idle(&mut self) {
        let th = self.active_theme;
        if self.seq.is_running() || self.niah.is_running() {
            self.push_log(
                "[idle] locked while a benchmark is running — finish it first".to_string(),
                style::value_warn(th),
            );
            return;
        }
        let Some(poller) = &self.hw else {
            self.push_log(
                "[idle] no GPU telemetry — nothing to measure".to_string(),
                style::value_warn(th),
            );
            return;
        };
        // Perform the action inside the lock, capture the outcome, and drop
        // the guard *before* touching `self` again (the immutable `poller`
        // borrow of `self.hw` must end first).
        let cancelled = match poller.lock() {
            Ok(mut p) => {
                if p.monitor().is_measuring_idle() {
                    p.cancel_idle_measurement();
                    true
                } else {
                    p.begin_idle_measurement();
                    false
                }
            }
            Err(_) => return,
        };
        if cancelled {
            self.push_log(
                "[idle] measurement cancelled".to_string(),
                style::value_warn(th),
            );
        } else {
            self.push_log(
                format!(
                    "[idle] measuring idle baseline ({}s) — press [i] again to cancel",
                    MANUAL_IDLE_WINDOW_SECS
                ),
                style::value_ok(th),
            );
        }
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
        let th = self.active_theme;
        if self.models.is_running() {
            self.push_log(
                "[discovery] already in progress — one request at a time".to_string(),
                style::value_warn(th),
            );
            return;
        }
        // The form (View 6 / the setup flow) is the source of the target
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
            style::value_ok(th),
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
        let th = self.active_theme;
        if self.seq.is_running() {
            self.push_log(
                "[seq] already running — engines execute one at a time".to_string(),
                style::value_warn(th),
            );
            return;
        }
        // FIX 2: re-arm the metrics pipeline for the new run — a
        // completed run freezes it (final numbers stop drifting), and
        // the freeze must not leak into the next one. This also wakes
        // the idle hardware poller.
        self.metrics.unfreeze();
        let cfg = self.config.to_config();
        if cfg.engines.is_empty() {
            self.push_log(
                "[seq] no engines selected — enable some in the Config view (View 6)".to_string(),
                style::value_warn(th),
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
            style::value_ok(th),
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
                flatout: self.flatout_slot.clone(),
            },
            Some(tx),
            self.pause.clone(),
            self.logger.clone(),
            self.session_id.clone(),
        );
        tokio::spawn(async move {
            seq.run().await;
        });
    }

    /// Handle one terminal key event (blueprint §6 footer key map).
    pub fn handle_key(&mut self, key: &KeyEvent) -> KeyAction {
        let th = self.active_theme;
        // Ctrl-C is **inert**: the terminal's copy selection (mouse-select
        // + Ctrl-C) owns it, so it must never quit the app — and the same
        // swallow covers every other Ctrl combo, so none of them can fire
        // a key action (the old global `Ctrl-C quits` path is gone).
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return KeyAction::Continue;
        }

        // The first-run theme picker is a **modal** full-screen takeover:
        // `↑`/`↓` move the cursor, `Enter` confirms (applies + persists the
        // theme, then falls through to Setup or the Dashboard), and `q`
        // opens the usual quit confirmation. Every other key is swallowed
        // (the picker owns the screen until a theme is chosen).
        if self.phase == Phase::ThemePicker {
            return match key.code {
                KeyCode::Up => {
                    self.theme_picker.cursor =
                        (self.theme_picker.cursor + Theme::ALL.len() - 1) % Theme::ALL.len();
                    KeyAction::Continue
                }
                KeyCode::Down => {
                    self.theme_picker.cursor = (self.theme_picker.cursor + 1) % Theme::ALL.len();
                    KeyAction::Continue
                }
                KeyCode::Enter => {
                    self.confirm_theme();
                    KeyAction::Continue
                }
                KeyCode::Char('q') => {
                    self.pending_quit = true;
                    KeyAction::Continue
                }
                _ => KeyAction::Continue,
            };
        }

        // The quit confirmation is a **modal overlay** (dashboard *and*
        // Setup takeover): `y`/`Y`/`Enter` confirms — the *only* quit
        // path — and every other key cancels and is swallowed (a stray
        // keypress can never both answer the prompt and trigger a second
        // action).
        if self.pending_quit {
            return match key.code {
                KeyCode::Char('y') | KeyCode::Char('Y') | KeyCode::Enter => {
                    self.pending_quit = false;
                    self.logger.info(Context::Tui, "quit (q confirmed)");
                    self.running = false;
                    KeyAction::Quit
                }
                _ => {
                    self.pending_quit = false;
                    self.push_log(
                        "[quit] cancelled — the app keeps running".to_string(),
                        style::value_warn(th),
                    );
                    KeyAction::Continue
                }
            };
        }

        // Matrix Report phase (full-screen takeover, modeled on the Setup
        // phase below): `c` copy (OSC 52), `s` save, `Esc` close. It
        // returns before the `q`-quit arm, so `q` types into the focused
        // field (exactly as in Setup) and can never open the quit overlay
        // from the report.
        if self.phase == Phase::Report {
            return match self.report.handle_key(key) {
                ReportKeyResult::Inert => KeyAction::Continue,
                // N/A-never-fail: a terminal that drops the OSC 52
                // sequence is a logged warning, never a crash (the save
                // path is the guaranteed one).
                ReportKeyResult::Copied => {
                    let res = crate::clipboard::copy_to_clipboard(&self.report.report.render());
                    self.push_log(
                        Self::report_copy_message(&res),
                        if res.is_ok() {
                            style::value_ok(th)
                        } else {
                            style::value_warn(th)
                        },
                    );
                    KeyAction::Continue
                }
                ReportKeyResult::Saved => {
                    match Self::write_report_file(
                        &self.report.report.render(),
                        &self.report.save_path,
                    ) {
                        Ok(p) => self.push_log(
                            format!("[report] saved → {}", p.display()),
                            style::value_ok(th),
                        ),
                        Err(e) => self
                            .push_log(format!("[report] save failed: {e}"), style::value_warn(th)),
                    }
                    KeyAction::Continue
                }
                ReportKeyResult::Closed => {
                    // The run state is untouched (`seq` still reads
                    // `AllComplete`), so `m` re-opens the report with the
                    // same auto fields and the user's in-progress edits.
                    self.phase = Phase::Dashboard;
                    KeyAction::Continue
                }
            };
        }

        // Run-log viewer phase (full-screen takeover, modeled on the
        // Report phase above): `c` copy (OSC 52), `s` save, `Esc` close,
        // scroll with j/k/arrows/PageUp/PageDown/Home/End. It returns
        // before the `q`-quit arm, so `q` can never open the quit
        // overlay from the log view.
        if self.phase == Phase::Log {
            return match self.log_view.handle_key(key) {
                LogKeyResult::Inert => KeyAction::Continue,
                // N/A-never-fail: a terminal that drops the OSC 52
                // sequence is a logged warning, never a crash (the save
                // path is the guaranteed one).
                LogKeyResult::Copied => {
                    let res = crate::clipboard::copy_to_clipboard(&self.log_view.lines.join("\n"));
                    self.push_log(
                        Self::log_copy_message(&res),
                        if res.is_ok() {
                            style::value_ok(th)
                        } else {
                            style::value_warn(th)
                        },
                    );
                    KeyAction::Continue
                }
                LogKeyResult::Saved => {
                    // The save destination is pre-filled at open
                    // (data_dir()/exports — the established user-save
                    // location); the view is read-only, so it is not
                    // user-editable.
                    match Self::write_report_file(&self.log_view.lines.join("\n"), &self.log_view.save_path)
                    {
                        Ok(p) => self.push_log(
                            format!("[log] saved → {}", p.display()),
                            style::value_ok(th),
                        ),
                        Err(e) => self
                            .push_log(format!("[log] save failed: {e}"), style::value_warn(th)),
                    }
                    KeyAction::Continue
                }
                LogKeyResult::Closed => {
                    // The run state is untouched; the view is back on
                    // the dashboard. The log content is a snapshot —
                    // re-`l` re-reads the (growing) file.
                    self.phase = Phase::Dashboard;
                    KeyAction::Continue
                }
            };
        }

        // Setup phase (full-screen takeover): every other key routes to
        // the four-stage flow. `q` types into a field; `Esc` walks back
        // one stage (quitting only at stage 1).
        if self.phase == Phase::Setup {
            return match self.setup.handle_key(key, &mut self.config) {
                SetupKeyResult::Inert => KeyAction::Continue,
                // `Esc` at stage 1 no longer quits outright: it opens the
                // `[y/N]` confirmation overlay (the only quit path — in
                // Setup `q` is a typeable character, not a quit key).
                SetupKeyResult::Quit => {
                    self.pending_quit = true;
                    KeyAction::Continue
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
                        style::value_warn(th),
                    );
                    return KeyAction::Continue;
                }
            }
        }

        // `q` is the **only** quit key — even inside the Config view
        // (where `q` would otherwise be typed into a field). It opens the
        // `[y/N]` confirmation overlay; only a confirmed `y` quits.
        if key.code == KeyCode::Char('q') {
            self.pending_quit = true;
            self.logger
                .info(Context::Tui, "quit confirmation opened (q)");
            return KeyAction::Continue;
        }
        // `Esc` never quits: in the dashboard views it is a no-op (the
        // Config view handles it itself — leave to the gate / Live — and
        // Setup stage 1 routes it to the confirmation above).

        // `c` — re-open the interactive Setup takeover. Scoped away from
        // View 6, where `c` is a typeable character in a field. Locked
        // while a benchmark sequence runs: the takeover's URL stage
        // fires an HTTP discovery request, which must not land on the
        // endpoint mid-benchmark.
        if key.code == KeyCode::Char('c') && self.view != View::Config && self.view != View::History
        {
            if self.seq.is_running() {
                self.push_log(
                    "[setup] locked while a benchmark is running — press Space to pause, Q to quit"
                        .to_string(),
                    style::value_warn(th),
                );
                return KeyAction::Continue;
            }
            self.open_setup();
            return KeyAction::Continue;
        }

        // `l` — open the run-log viewer. **No completion gate** (the log
        // exists from process start — it works before, during, and after
        // a run). Scoped away from View 6, where `l` is a typeable
        // character in a field (the same scope as `c` above), and from
        // the History view, which has its own `l` (the cursor session's
        // archive log — the History block below).
        if (key.code == KeyCode::Char('l') || key.code == KeyCode::Char('L'))
            && self.view != View::Config
            && self.view != View::History
        {
            self.open_log();
            return KeyAction::Continue;
        }

        // Config view (Chunk 18 + FIX 4): the view has an **edit gate**.
        // On entry it is [`ConfigMode::Viewing`] — a read-only "press
        // Enter to edit" screen. The number keys `1`–`6` *always* switch
        // views (they are never captured by field editing), `Esc` leaves
        // the config, and `q` quits (handled above) — so the user can
        // never get stuck. Only after `Enter` does [`ConfigMode::Editing`]
        // make the fields live (there `Esc` / `1`–`6` save and exit).
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
                    // `1`–`6` always switch views (never typed into a field);
                    // `6` is the Config view itself (stays at the gate).
                    KeyCode::Char(c @ '1'..='6') => {
                        if let Some(d) = c.to_digit(10) {
                            if let Some(view) = View::from_digit(d as u8) {
                                if view != View::Config {
                                    self.view = view;
                                    self.config.edit_mode = ConfigMode::Viewing;
                                    if view == View::History {
                                        self.ensure_history();
                                    }
                                }
                            }
                        }
                        return KeyAction::Continue;
                    }
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
                                style::value_ok(th),
                            );
                        }
                        return KeyAction::Continue;
                    }
                    KeyCode::F(5) => {
                        self.start_run();
                        // After launching a run, switch to Live.
                        self.view = View::Live;
                        self.config.edit_mode = ConfigMode::Viewing;
                        return KeyAction::Run;
                    }
                    // `R` — reset to built-in defaults.
                    KeyCode::Char('R') => {
                        self.config.reset_to_defaults();
                        self.push_log(
                            "[config] reset to defaults".to_string(),
                            style::value_warn(th),
                        );
                        return KeyAction::Continue;
                    }
                    // Everything else is ignored at the gate.
                    _ => return KeyAction::Continue,
                },
                // ── Editing (the fields are live). ──
                ConfigMode::Editing => {
                    // `Esc`: save and return to the gate (stay on tab 6).
                    if key.code == KeyCode::Esc {
                        let _ = self.config.save();
                        self.config.edit_mode = ConfigMode::Viewing;
                        self.active_theme = Theme::from_id(&self.config.theme).unwrap_or_default();
                        return KeyAction::Continue;
                    }
                    // Everything else (including `1`–`9`, `0`, letters, Tab,
                    // arrows, F2, F5) delegates to the field editor.
                    let result = self.config.handle_key(key);
                    // Live preview: a theme change in the form re-skins the
                    // whole TUI immediately (the render reads `active_theme`).
                    self.active_theme = Theme::from_id(&self.config.theme).unwrap_or_default();
                    match result {
                        ConfigKeyResult::Saved => {
                            self.push_log(
                                format!("[config] saved → {}", self.config.config_path.display()),
                                style::value_ok(th),
                            );
                            self.logger.info(
                                Context::Setup,
                                format!("config saved → {}", self.config.config_path.display()),
                            );
                            return KeyAction::Continue;
                        }
                        ConfigKeyResult::Run => {
                            self.start_run();
                            // After launching a run, switch to Live.
                            self.view = View::Live;
                            self.config.edit_mode = ConfigMode::Viewing;
                            return KeyAction::Run;
                        }
                        ConfigKeyResult::Inert => return KeyAction::Continue,
                    }
                }
            }
        }

        // History view (Chunk 14 + rewrite): full keyboard navigation.
        // Scoped to the History view; the `c` key is consumed here (not
        // routed to the Setup takeover) and `Esc` returns to the List.
        // Only the specific History keys are consumed here; all other
        // keys (digit 1-6, Space, etc.) fall through to the general
        // handler so view switching and quit always work.
        if self.view == View::History {
            if let Some(h) = self.history.as_mut() {
                match key.code {
                    // Navigate: j/k/arrows (List and Compare-select modes).
                    KeyCode::Char('j') | KeyCode::Down | KeyCode::Char('k') | KeyCode::Up => {
                        let delta = if matches!(key.code, KeyCode::Char('j') | KeyCode::Down) {
                            1
                        } else {
                            -1
                        };
                        match &h.mode {
                            HistoryMode::List | HistoryMode::Compare { second: None, .. } => {
                                h.move_cursor(delta);
                                return KeyAction::HistoryNext;
                            }
                            _ => return KeyAction::Continue,
                        }
                    }
                    // Enter: view details (List) or select second run (Compare).
                    KeyCode::Enter => {
                        match &h.mode {
                            HistoryMode::List => h.enter_detail(),
                            HistoryMode::Compare { .. } => h.select_compare_second(),
                            _ => {}
                        }
                        return KeyAction::Continue;
                    }
                    // C: enter compare mode (List only).
                    KeyCode::Char('c') | KeyCode::Char('C') => {
                        if matches!(h.mode, HistoryMode::List) {
                            h.enter_compare();
                        }
                        return KeyAction::Continue;
                    }
                    // D: delete the selected run (List only, with confirmation).
                    KeyCode::Char('d') | KeyCode::Char('D') => {
                        if matches!(h.mode, HistoryMode::List) {
                            h.enter_delete_confirm();
                        }
                        return KeyAction::Continue;
                    }
                    // L: open the cursor session's archive log
                    // (run-<ts>-<session8>.log) in the run-log viewer.
                    // All three list-bearing modes have a cursor, so one
                    // arm covers List / Detail / Compare; `Esc` in the
                    // viewer returns here (the view state is untouched by
                    // the phase change — the same rule as the Report
                    // close).
                    KeyCode::Char('l') | KeyCode::Char('L') => {
                        let id = h.sessions.get(h.cursor).map(|s| s.session_id.clone());
                        if let Some(id) = id {
                            self.open_log_for_session(&id);
                        }
                        return KeyAction::Continue;
                    }
                    // Esc: back to the List (from Detail / Compare / DeleteConfirm).
                    KeyCode::Esc => {
                        h.back_to_list();
                        return KeyAction::Continue;
                    }
                    // y: confirm the delete.
                    KeyCode::Char('y') | KeyCode::Char('Y') => {
                        if matches!(h.mode, HistoryMode::DeleteConfirm(_)) {
                            h.confirm_delete();
                        }
                        return KeyAction::Continue;
                    }
                    // n: cancel the delete.
                    KeyCode::Char('n') => {
                        if matches!(h.mode, HistoryMode::DeleteConfirm(_)) {
                            h.back_to_list();
                        }
                        return KeyAction::Continue;
                    }
                    // Any other key: fall through to the general handler below.
                    _ => {}
                }
            }
        }

        match key.code {
            KeyCode::Char(c @ '1'..='6') => {
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
                self.push_log(msg, style::value_warn(th));
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
                        style::value_warn(th),
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
                        style::value_warn(th),
                    );
                    return KeyAction::Continue;
                }
                if self.niah.is_running() {
                    self.push_log(
                        "[niah] already running — one size × depth cell at a time".to_string(),
                        style::value_warn(th),
                    );
                    return KeyAction::Continue;
                }
                self.pending_niah = true;
                KeyAction::NewNeedle
            }
            KeyCode::Char('e') => KeyAction::Export,
            // `m` — open the Matrix Report (the post-run
            // `[COMMUNITY-SUBMISSION]` block). Gated on a **completed**
            // run: the auto fields are assembled from the run's result
            // slots + hardware telemetry, and opening with stale/empty
            // data would fabricate a report (the `m` key is free on the
            // dashboard — `q`/`c`/`e`/`n`/`r`/`i`/`+` are the only
            // letter keys in use here).
            KeyCode::Char('m') => {
                let complete =
                    self.seq.load().map(|s| s.phase == SeqPhase::AllComplete) == Some(true);
                if complete {
                    self.open_report();
                } else {
                    self.push_log(
                        "[report] complete a run first (press [R] to start)".to_string(),
                        style::value_warn(th),
                    );
                }
                KeyAction::Continue
            }
            // `i` — measure a fresh idle power baseline (the GPU tab's
            // manual idle test): the 100 ms poller sits the system no-load
            // for [`MANUAL_IDLE_WINDOW_SECS`] and records the mean as the
            // new idle floor. Pressing it again cancels. Locked while a
            // benchmark runs (those own the power window).
            KeyCode::Char('i') => {
                self.measure_idle();
                KeyAction::Continue
            }
            // Chunk 18: `r` runs the engines selected in the Config view.
            KeyCode::Char('r') => {
                self.start_run();
                KeyAction::Run
            }
            _ => KeyAction::Continue,
        }
    }

    /// URC-5: build the [`ExportPayload`] from the **completed result
    /// slots** (the final per-engine results), *not* the last live
    /// [`MetricsSnapshot`] (a mid-run / last-frame sample that drifts once
    /// the run ends). The session carries the run's `session_id` (so it
    /// joins to the run-log), the target/model from the Config form, the
    /// selected `engines`, and the B/C2/C3/D/F `engine_results`.
    ///
    /// Pure (no file I/O) so it is unit-testable; [`Self::export`] writes
    /// the result. Per-packet CSV detail only exists on the headless path;
    /// a TUI CSV export carries the metric rows only.
    fn build_export_payload(&self) -> ExportPayload {
        let cfg = self.config.to_config();

        // Engine D energy + the GPU name: the **same** definition the TUI
        // DB path (`BenchmarkSequence::persist_results` /
        // `engine_results_json`) uses (I1) — the shared
        // `sum_all_engines_tokens` denominator (all engines, not
        // Engine-A-only) + the shared `apply_load_window` (full
        // load-window numerator, idle-subtracted, `avg_power_w` on the
        // same basis) — so the export's `d_energy` equals the DB's
        // `d_energy` for the same run. A poisoned/absent poller degrades
        // to `None` — never an export failure.
        let (energy, gpu_name) = {
            let poller = self.hw.as_ref().and_then(|h| h.lock().ok());
            match poller {
                Some(g) => {
                    let trace = g.trace().to_vec();
                    let e = (!trace.is_empty()).then(|| {
                        // I1: the all-engine token denominator — the shared
                        // helper the DB path uses (the old export divided by
                        // Engine-A-only completion tokens).
                        let total_tokens = sum_all_engines_tokens(
                            (*self.speed_slot.load()).as_deref(),
                            (*self.sweep.load()).as_ref(),
                            (*self.niah.load()).as_ref(),
                            (*self.reasoning_slot.load()).as_ref(),
                            (*self.structured_slot.load()).as_ref(),
                            (*self.flatout_slot.load()).as_ref(),
                        );
                        // F9: the monitor's idle floor keeps `avg_power_w`
                        // on the same basis as the idle-subtracted joules.
                        let idle_w = g.monitor().idle_system_power_w();
                        let mut e = profile(&trace, None, total_tokens, idle_w);
                        if let Some(b) = g.backend() {
                            e.vendor = Some(b.vendor().to_string());
                            e.model = b.model().map(str::to_string);
                        }
                        // I1: the shared load-window application — the
                        // export's `d_energy` is the DB's `d_energy`.
                        let duration = (g.monitor().duration_sec() > 0.0)
                            .then(|| g.monitor().duration_sec());
                        crate::engines::sequence::apply_load_window(
                            &mut e,
                            g.load_window_joules(),
                            total_tokens,
                            duration,
                        );
                        e
                    });
                    (e, g.gpu_name().map(str::to_string))
                }
                None => (None, None),
            }
        };
        let energy_summary = energy.as_ref().map(EnergySummary::from_energy);

        // R2: the `engine_results` object (B/C2/C3/D/F) from the slots.
        let engine_results = EngineResults {
            b_concurrency: (*self.sweep.load())
                .clone()
                .as_ref()
                .map(ConcurrencySummary::from_sweep),
            c2_reasoning: (*self.reasoning_slot.load())
                .clone()
                .as_ref()
                .map(ReasoningSummary::from_reasoning),
            c3_structured: (*self.structured_slot.load())
                .clone()
                .as_ref()
                .map(StructuredSummary::from_structured),
            d_energy: energy_summary.clone(),
            f_flatout: (*self.flatout_slot.load())
                .clone()
                .as_ref()
                .map(FlatOutSummary::from_flatout),
        };
        let engine_results_json = engine_results.to_json();
        let has_engine_results = engine_results_json.is_some();

        let session = BenchmarkSession {
            session_id: self.session_id.clone(),
            timestamp: None,
            target_url: cfg.url.clone(),
            model_name: cfg.model.clone(),
            backend_type: None,
            quantization: None,
            system_gpu: gpu_name,
            total_duration_sec: None,
            // R2 (traceability): the engines this run selected + the
            // B/C2/C3/D/F headline results.
            engines: Some(cfg.engines.selected_names_json()),
            engine_results: engine_results_json,
        };

        // Engine A: the speed slot's per-iteration results → metric rows.
        // I1: `joules_per_token` stays **consistently NULL** on TUI rows —
        // the TUI path has no per-iteration power windows (the speed
        // engine drains its events internally), and stamping every row
        // with the single run-level number (the old A-only behavior)
        // fabricated a per-stream metric. The run-level D figure lives in
        // `payload.energy` / `engine_results.d_energy`, which this export
        // carries; the headless path (which *does* capture per-iteration
        // windows) stamps its rows with the per-iteration value.
        let metrics: Vec<StreamMetricRow> = (*self.speed_slot.load())
            .as_ref()
            .map(|rs| {
                rs.iter()
                    .map(|r| StreamMetricRow::from_speed_result(r, &session.session_id, 1))
                    .collect()
            })
            .unwrap_or_default();

        // Engine C1: the NIAH slot's cells → needle rows.
        let needles: Vec<NeedleEvaluation> = (*self.niah.load())
            .as_ref()
            .map(|n| {
                n.cells
                    .iter()
                    .map(|cell| NeedleEvaluation {
                        eval_id: None,
                        session_id: session.session_id.clone(),
                        context_length: Some(cell.target_tokens as i64),
                        depth_percent: Some(cell.depth_percent as f64),
                        retrieved_successfully: Some(cell.retrieved),
                        // Option C: persist the per-cell truncated-by-
                        // reasoning state (inconclusive (budget), not a
                        // retrieval fail).
                        truncated_by_reasoning: Some(cell.truncated_by_reasoning),
                        latency_ms: Some(cell.ttft_s * 1000.0),
                    })
                    .collect()
            })
            .unwrap_or_default();

        let mut payload = ExportPayload::from_live(session, metrics, Vec::new(), needles);
        // R2: attach the B/C2/C3/D/F results + the energy object.
        payload.engine_results = has_engine_results.then_some(engine_results);
        payload.energy = energy_summary;
        // v0.1.1: carry the 2D concurrency × context matrix (when Engine B
        // ran one) into the JSON export's `matrix` block.
        if let Some(matrix) = (*self.sweep.load())
            .as_ref()
            .and_then(|r| r.matrix.as_ref())
        {
            payload = payload.with_matrix(matrix.clone());
        }
        payload
    }

    /// The Matrix Report auto fields, assembled from the run's result
    /// slots + hardware telemetry (mirrors [`build_export_payload`]
    /// (Self::build_export_payload); the headless path assembles the same
    /// block from its in-scope run variables — design §1.4/§4.2).
    ///
    /// Sources (the honest-degradation rules, design §1.4/§1.5):
    /// - **model** — the Configuration form (`config.to_config()`);
    /// - **GPU name** — the poller's `gpu_display_name` form, then the
    ///   shared backend, then `""` (no GPU);
    /// - **GPU count** — vendor-agnostic (`GpuBackend::poll_all().len()`;
    ///   NVML overrides per-device) — *not* the NVML-only `device_count`;
    /// - **VRAM** — the lock-free aggregate [`GpuSample`]
    ///   (`memory_total_mb`, **binary MiB**) — *not* the decimal
    ///   `vram_total_gb` (a "24 GiB" card reads 25.77 in decimal GB and
    ///   would cross its tier boundary);
    /// - **RAM** — a fresh `sysinfo` read (the poller's copy is ≤100 ms
    ///   stale);
    /// - **speed** — Engine A's mean `tg_speed` over the valid runs
    ///   ([`speed_mean_tps`]) — blank, never a silent 0, when there are
    ///   no valid runs.
    ///
    /// The user fields (backend / quant / use case / notes) are passed
    /// **empty** — the user fills them in the takeover.
    fn build_matrix_report(&self) -> MatrixReport {
        let cfg = self.config.to_config();
        let (gpu_name, gpu_count, vram_mib, ram_bytes) = {
            let poller = self.hw.as_ref().and_then(|h| h.lock().ok());
            // Name: the poller first (the `gpu_display_name` form it
            // captured at construction), then the shared backend.
            let gpu_name = poller
                .as_ref()
                .and_then(|g| g.gpu_name().map(str::to_string))
                .or_else(|| {
                    self.gpu
                        .as_ref()
                        .map(|g| crate::hw::gpu_display_name(g.as_ref()))
                })
                .unwrap_or_default();
            // Count: vendor-agnostic (the trait `poll_all`); a
            // pollerless host falls back to the shared backend.
            let gpu_count = match poller.as_ref().and_then(|g| g.backend()) {
                Some(b) => b.poll_all().len() as u32,
                None => self
                    .gpu
                    .as_ref()
                    .map(|g| g.poll_all().len() as u32)
                    .unwrap_or(0),
            };
            // VRAM: the lock-free aggregate sample (binary MiB) — the
            // decimal `vram_total_gb` is deliberately *not* used (the
            // 24-GiB-card-reads-25.77 trap, design §1.2).
            let vram_mib = self
                .metrics
                .load()
                .gpu
                .as_ref()
                .and_then(|g| g.memory_total_mb)
                .unwrap_or(0);
            // RAM: a fresh `sysinfo` read (the poller's copy is ≤100 ms
            // stale; this is the key path, never the 60 Hz render path).
            let mut sys = sysinfo::System::new();
            sys.refresh_memory();
            let ram_bytes = sys.total_memory();
            (gpu_name, gpu_count, vram_mib, ram_bytes)
        };
        // Engine A's mean decode speed (the valid filter; 0.0 = no valid
        // runs → the report's blank speed field).
        let speed = speed_mean_tps((*self.speed_slot.load()).as_deref().unwrap_or(&[]));
        // The user fields start empty (the user fills them in the
        // takeover).
        MatrixReport::build(
            &gpu_name, gpu_count, vram_mib, ram_bytes, &cfg.model, speed, "", "", "", "",
        )
    }

    /// The `c` key's log message for a clipboard write outcome (the
    /// N/A-never-fail rule, design invariant 3 — a terminal that drops the
    /// OSC 52 sequence is a logged warning, never a crash; save-to-file is
    /// the guaranteed path). Pure over the outcome so the handler logic is
    /// testable without a terminal.
    #[must_use]
    pub fn report_copy_message(res: &Result<(), String>) -> String {
        match res {
            Ok(()) => "[report] copied to clipboard (OSC 52)".to_string(),
            Err(_) => "[report] clipboard unsupported by terminal — use [S]ave".to_string(),
        }
    }

    /// The `s` key: write the Matrix Report block verbatim to `path`
    /// (creating the parent directory) — the `storage/export.rs` `write`
    /// pattern over a raw string (`export::write` is `ExportPayload`-typed,
    /// so this is the small raw-string variant, not a change to it).
    fn write_report_file(content: &str, path: &str) -> Result<std::path::PathBuf, String> {
        let p = std::path::PathBuf::from(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(&p, content).map_err(|e| e.to_string())?;
        Ok(p)
    }

    /// The `e` key (blueprint §6 footer, Chunk 13): export the run's
    /// results (built from the result slots, URC-5) to
    /// `data_dir()/exports/` in the configured format and log the
    /// destination.
    pub fn export(&mut self) -> Result<std::path::PathBuf, String> {
        let th = self.active_theme;
        let payload = self.build_export_payload();
        let dest = export::default_path_live(self.export_format);
        let path = export::write(&dest, self.export_format, &payload).map_err(|e| e.to_string())?;
        self.push_log(
            format!(
                "[export] {} → {}",
                self.export_format.label(),
                path.display()
            ),
            style::value_ok(th),
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
        let th = self.active_theme;
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
            self.push_log(line, style::value(th));
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

        // FIX 2: the run is over (the metrics pipeline is frozen) → no
        // metric-driven tick work at all. The snapshot is static, so the
        // VRAM-fragmentation check (the only metric read here) has
        // nothing new to warn about; skipping it keeps the tick path
        // from touching a finished run.
        if self.metrics.is_frozen() {
            return;
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
                self.push_log(format!("Warning: {w}"), style::value_warn(th));
            }
        } else if ratio < VRAM_FRAGMENTATION_THRESHOLD * 0.95 {
            self.vram_warned = false;
        }
    }

    /// Draw the full frame.
    ///
    /// The Setup phase is a **full-screen takeover** (not one of the six
    /// tabs): it draws its own top bar, centered panel, and key-hint
    /// footer across the entire frame. The dashboard keeps the classic
    /// status bar / tab bar / view / footer layout.
    pub fn render(&self, f: &mut Frame) {
        // Edge case: the layout needs at least 80 columns. On a narrower
        // terminal show a clear message instead of a broken, clipped frame.
        if f.area().width < 80 {
            self.render_too_narrow(f.area(), f);
            return;
        }
        if self.phase == Phase::ThemePicker {
            views::theme_picker::render(f.area(), self, f);
        } else if self.phase == Phase::Report {
            views::report::render(f.area(), self, f);
        } else if self.phase == Phase::Log {
            views::log::render(f.area(), self, f);
        } else if self.phase == Phase::Setup {
            views::setup::render(f.area(), self, f);
        } else {
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
        // The quit confirmation is a small centered overlay drawn on top
        // of whatever phase is behind it (dashboard or Setup).
        if self.pending_quit {
            self.render_quit_overlay(f);
        }
    }

    /// The "terminal too narrow" full-frame message (shown when the width is
    /// under the 80-column minimum the layout needs).
    fn render_too_narrow(&self, area: Rect, f: &mut Frame) {
        let th = self.active_theme;
        let block = theme::block(theme::panel_title(th, "CRUCIBLE-LLM"), style::value_err(th));
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "Terminal too narrow (minimum 80 columns)",
                style::value_err(th),
            )))
            .block(block)
            .alignment(Alignment::Center),
            area,
        );
    }

    /// The `[y/N]` quit-confirmation overlay: a small centered box on top
    /// of the current frame. `q` (dashboard) / `Esc` (Setup stage 1) opens
    /// it; `y` confirms (the only quit path), anything else cancels.
    fn render_quit_overlay(&self, f: &mut Frame) {
        let th = self.active_theme;
        let area = f.area();
        const W: u16 = 34;
        const H: u16 = 5;
        let w = W.min(area.width);
        let h = H.min(area.height);
        let x = area.x + (area.width.saturating_sub(w)) / 2;
        let y = area.y + (area.height.saturating_sub(h)) / 2;
        let overlay = Rect::new(x, y, w, h);
        let block = Block::default()
            .borders(Borders::ALL)
            .border_type(ratatui::widgets::BorderType::Rounded)
            .border_style(
                Style::default()
                    .fg(th.accent())
                    .add_modifier(Modifier::BOLD),
            )
            .title(" QUIT? ");
        let text = Text::from(vec![
            Line::raw(""),
            Line::from(Span::styled(
                "Quit crucible-llm?",
                Style::default()
                    .fg(th.accent())
                    .add_modifier(Modifier::BOLD),
            )),
            Line::from(vec![
                Span::styled("  [y] Yes", style::value(th)),
                Span::styled("   [n] No", style::value(th)),
            ]),
        ]);
        f.render_widget(Paragraph::new(text).block(block), overlay);
    }

    /// Top status bar: version, server address (`host:port`), target
    /// model, mode (blueprint §6).
    ///
    /// The backend is deliberately **not** shown: we don't know what is
    /// serving the endpoint (vLLM, llama.cpp, LM Studio, Unsloth, SGLang,
    /// …), so the bar shows the actual server address instead.
    ///
    /// Reads the shared snapshot lock-free; a fresh `Arc` is taken each frame
    /// so the bar always reflects the latest published metrics.
    fn render_status_bar(&self, area: Rect, f: &mut Frame) {
        let th = self.active_theme;
        let m = self.metrics.load();
        // The server address (host:port) from the endpoint URL — the
        // `—` placeholder when no target has been set yet.
        let host = fmt::url_host(&m.endpoint);
        let mut spans = vec![
            Span::styled(" ▐ ", style::tab_separator(th)),
            Span::styled("CRUCIBLE", style::title(th)),
            Span::styled(
                "·LLM",
                Style::default()
                    .fg(th.secondary())
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" v{}", env!("CARGO_PKG_VERSION")),
                style::footer(th),
            ),
            Span::styled(
                format!("  ▐ [{}]", if host.is_empty() { "—" } else { &host }),
                style::value(th),
            ),
            Span::styled(" ▐ ", style::tab_separator(th)),
            Span::styled(format!("Target: {}", m.model), style::label(th)),
            Span::styled(" ▐ ", style::tab_separator(th)),
            Span::styled(format!("Mode: {}", m.mode), style::label(th)),
        ];
        // State indicators: `◉ RUNNING` while any benchmark load is in
        // flight (sequence running, or a standalone NIAH run), `‖ PAUSED`
        // when the Space gate is holding.
        if self.seq.is_running() || self.niah.is_running() {
            spans.push(Span::styled(
                format!("  {} RUNNING", glyph::RUN),
                Style::default()
                    .fg(th.primary())
                    .add_modifier(Modifier::BOLD),
            ));
        }
        if self.paused {
            spans.push(Span::styled(" ‖ PAUSED ", style::value_warn(th)));
        }
        f.render_widget(
            Paragraph::new(Line::from(spans)).style(style::status_bar(th)),
            area,
        );
    }

    /// Tab bar: `[1] Live Monitor | [2] Concurrency Matrix | ...`.
    fn render_tab_bar(&self, area: Rect, f: &mut Frame) {
        let th = self.active_theme;
        let mut spans = vec![Span::raw(" ")];
        for (i, view) in View::ALL.iter().enumerate() {
            let active = *view == self.view;
            let label = if active {
                format!("{} [{}] {}", glyph::PREFIX, i + 1, view.label())
            } else {
                format!("[{}] {}", i + 1, view.label())
            };
            let st = if active {
                style::tab_active(th)
            } else {
                style::tab_inactive(th)
            };
            spans.push(Span::styled(label, st));
            if i + 1 < View::ALL.len() {
                spans.push(Span::styled(" │ ", style::tab_separator(th)));
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
    /// terminal). A single clean line:
    ///
    /// ```text
    /// [1]Live [2]Conc [3]NIAH [4]GPU [5]Hist [6]Cfg │ ● Connected │ Engine B: Step 5/9 │ [Space]Pause [q]Quit
    /// ```
    ///
    /// * the six view tabs — the current view in **accent** (bold), the
    ///   rest dim;
    /// * a connection-status dot (green `● Connected`, red `● Disconnected`);
    /// * the running engine's status (`Engine B: Step 5/9`) — shown only
    ///   while a sequence with live progress is in flight;
    /// * the action keys (`[Space]Pause` / `[Space]Resume` while running,
    ///   `[R]Run` when idle, `[q]Quit` always).
    ///
    /// The `[Y/N]` NIAH confirmation is a modal footer (its own shape).
    pub fn footer_line(&self) -> Line<'static> {
        let th = self.active_theme;
        if self.pending_niah {
            return self.niah_confirm_footer();
        }
        let mut spans: Vec<Span> = Vec::new();
        spans.extend(self.footer_view_tabs());
        spans.push(footer_sep(th));
        spans.push(self.footer_connection());
        if let Some(st) = self.footer_engine_status() {
            spans.push(footer_sep(th));
            spans.push(st);
        }
        spans.push(footer_sep(th));
        spans.extend(self.footer_action_keys());
        Line::from(spans)
    }

    /// The `[Y/N]` NIAH-confirmation modal footer: the request count and
    /// the two live keys.
    fn niah_confirm_footer(&self) -> Line<'static> {
        let th = self.active_theme;
        let sep = footer_sep(th);
        let live = |t: &str| Span::styled(t.to_string(), style::footer(th));
        let requests = NIAH_SIZES.len() * NIAH_DEPTHS.len();
        Line::from(vec![
            live(&format!(
                "[Y] Run NIAH test? {requests} requests ({s} sizes × {d} depths) ",
                s = NIAH_SIZES.len(),
                d = NIAH_DEPTHS.len()
            )),
            sep.clone(),
            live("[N/Esc] Cancel "),
            sep.clone(),
            live("[q] Quit"),
        ])
    }

    /// The six view tabs: `[1]Live [2]Conc [3]NIAH [4]GPU [5]Hist [6]Cfg` —
    /// the current view in **accent** (bold), the rest dim.
    fn footer_view_tabs(&self) -> Vec<Span<'static>> {
        let th = self.active_theme;
        View::ALL
            .iter()
            .enumerate()
            .flat_map(|(i, v)| {
                let active = *v == self.view;
                let label = format!("[{}]{}", i + 1, footer_view_label(*v));
                let st = if active {
                    style::tab_active(th)
                } else {
                    style::footer(th)
                };
                let mut out: Vec<Span<'static>> = Vec::new();
                if active {
                    out.push(Span::styled(glyph::PREFIX.to_string(), st));
                }
                out.push(Span::styled(label, st));
                out.push(Span::raw(" "));
                out
            })
            .collect()
    }

    /// The connection-status dot: green `● Connected` when a target URL is
    /// set, red `● Disconnected` otherwise.
    fn footer_connection(&self) -> Span<'static> {
        let th = self.active_theme;
        if self.config.url.trim().is_empty() {
            Span::styled("● Disconnected", style::value_err(th))
        } else {
            Span::styled("● Connected", style::value_ok(th))
        }
    }

    /// The running-engine status segment (`Engine B: Step 5/9`) — present
    /// only while a sequence with live progress is in flight.
    fn footer_engine_status(&self) -> Option<Span<'static>> {
        let th = self.active_theme;
        let state = self.seq.load()?;
        if state.phase != SeqPhase::Running {
            return None;
        }
        let letter = state.engine.label().split(':').next().unwrap_or("").trim();
        let step = engine_step_label(&state.progress);
        Some(Span::styled(
            format!("Engine {letter}: {step}"),
            Style::default()
                .fg(th.primary())
                .add_modifier(Modifier::BOLD),
        ))
    }

    /// The action keys: `[Space]Pause` / `[Space]Resume` while running,
    /// `[R]Run` when idle, `[M]Report` once a run has completed, and
    /// `[q]Quit` always.
    fn footer_action_keys(&self) -> Vec<Span<'static>> {
        let th = self.active_theme;
        let running = self.seq.is_running() || self.niah.is_running();
        let primary = if running {
            if self.paused {
                "[Space]Resume"
            } else {
                "[Space]Pause"
            }
        } else {
            "[R]Run"
        };
        // Action keys: electric purple (the "do something" keys); the
        // navigation tabs above stay dim blue.
        let key_style = Style::default()
            .fg(th.secondary())
            .add_modifier(Modifier::BOLD);
        let mut spans = vec![Span::styled(primary.to_string(), key_style), Span::raw(" ")];
        // The `m` key opens the Matrix Report once every selected engine
        // has finished (the same gate the key itself uses).
        let complete = self.seq.load().map(|s| s.phase == SeqPhase::AllComplete) == Some(true);
        if complete {
            spans.push(Span::styled("[M]Report".to_string(), key_style));
            spans.push(Span::raw(" "));
        }
        // The `l` key opens the run-log viewer at any time (no completion
        // gate — the log exists from process start).
        spans.push(Span::styled("[L]og".to_string(), key_style));
        spans.push(Span::raw(" "));
        spans.push(Span::styled("[q]Quit".to_string(), key_style));
        spans
    }
}

/// The footer's `│` separator span.
fn footer_sep(th: Theme) -> Span<'static> {
    Span::styled(" │ ", style::tab_separator(th))
}

/// The current Unix seconds (the save-destination timestamp stamp, e.g.
/// `run-log-<ts>.log`). `0` when the clock cannot be read (it never
/// will in practice).
fn ts_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The compact tab label for a view (the footer's `[n]Label` tabs).
fn footer_view_label(v: View) -> &'static str {
    match v {
        View::Live => "Live",
        View::Concurrency => "Conc",
        View::Needle => "NIAH",
        View::History => "Hist",
        View::Config => "Cfg",
        View::Gpu => "GPU",
    }
}

/// The compact `Step X/Y` (or `Sampling…`) for a running engine's progress.
fn engine_step_label(progress: &Option<EngineProgress>) -> String {
    match progress {
        Some(EngineProgress::Speed {
            iteration, total, ..
        }) => format!("Step {iteration}/{total}"),
        Some(EngineProgress::Concurrency {
            step, total_steps, ..
        }) => format!("Step {step}/{total_steps}"),
        Some(EngineProgress::Niah {
            cell, total_cells, ..
        }) => format!("Step {cell}/{total_cells}"),
        Some(EngineProgress::Reasoning { challenge, total }) => format!("Step {challenge}/{total}"),
        Some(EngineProgress::Structured { run, total }) => format!("Step {run}/{total}"),
        Some(EngineProgress::Sampling { .. }) => "Sampling…".to_string(),
        Some(EngineProgress::FlatOut { elapsed_secs, .. }) => {
            format!("{elapsed_secs:.0}s / {WINDOW_SECS:.0}s")
        }
        None => "Starting…".to_string(),
    }
}

/// Number-formatting helpers shared by every view (the "consistent number
/// formatting" rule). Pure string builders — unit-testable, no terminal.
///
/// * [`format_rate`]   — `142.3 t/s` (one decimal).
/// * [`format_duration`] — `4m 32s` for a minute or more, else `1.2s`.
/// * [`format_tokens`] — `33,108` (thousands separator) or `1.2M` for ≥1M.
/// * [`format_pct`]    — `92.3%` (one decimal).
/// * [`url_host`]      — the `host:port` of a URL (status bar).
pub mod fmt {
    /// `142.3 t/s` — tokens/sec, one decimal.
    pub fn format_rate(tps: f64) -> String {
        format!("{tps:.1} t/s")
    }

    /// `4m 32s` for a duration of a minute or more, else `1.2s` (one
    /// decimal). `--` for a non-positive duration (never blank).
    pub fn format_duration(secs: f64) -> String {
        if secs <= 0.0 {
            return "--".to_string();
        }
        if secs < 60.0 {
            return format!("{secs:.1}s");
        }
        let total = secs as u64;
        let m = total / 60;
        let s = total % 60;
        format!("{m}m {s}s")
    }

    /// `33,108` with thousands separators, or `1.2M` once a count reaches a
    /// million. Zero renders as `0` (never blank).
    pub fn format_tokens(n: u64) -> String {
        if n >= 1_000_000 {
            return format!("{:.1}M", n as f64 / 1_000_000.0);
        }
        let s = n.to_string();
        let bytes = s.as_bytes();
        let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
        for (i, b) in bytes.iter().enumerate() {
            if i > 0 && (bytes.len() - i).is_multiple_of(3) {
                out.push(',');
            }
            out.push(*b as char);
        }
        out
    }

    /// `92.3%` — one decimal.
    pub fn format_pct(p: f64) -> String {
        format!("{p:.1}%")
    }

    /// Truncate `s` to at most `max` columns, appending `…` when it is
    /// shortened (char-boundary safe — a multi-byte char is never split).
    /// Used for long model names / URLs that must fit a fixed panel width.
    pub fn truncate(s: &str, max: usize) -> String {
        if s.len() <= max {
            return s.to_string();
        }
        let mut end = max;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }

    /// The server address (`host:port`) of a URL —
    /// `http://127.0.0.1:8888/v1` → `127.0.0.1:8888`. The status
    /// bar shows this (never a backend name: the server could be vLLM,
    /// llama.cpp, LM Studio, Unsloth, SGLang, … — we don't know). A bare
    /// host passes through; the empty string stays empty.
    pub fn url_host(url: &str) -> String {
        let trimmed = url.trim();
        let without_scheme = trimmed.split("://").last().unwrap_or(trimmed).trim();
        without_scheme
            .split('/')
            .next()
            .unwrap_or("")
            .trim()
            .to_string()
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

    // ── URC-5: the TUI export builds from the result slots, not the
    //     last live snapshot ─────────────────────────────────────────────

    #[test]
    fn tui_export_builds_from_result_slots_not_live_snapshot() {
        let mut app = App::new();
        // The live [`MetricsSnapshot`] is left at its default (no streams),
        // so an export that (wrongly) read the snapshot would be empty.
        // The result slots carry the real, distinct data.
        app.config.url = "http://slot-target:9999/v1".to_string();
        app.config.model = "slot-model".to_string();
        app.config.engine_speed = true;
        app.speed_slot.store(vec![SpeedResult {
            ttft: 0.25,
            prompt_tokens: 100,
            completion_tokens: 50,
            reasoning_tokens: None,
            pp_speed: 400.0,
            tg_speed: 20.0,
            mtp_efficiency: 1.0,
            stream_time: 2.5,
            total_chunks: 50,
            content_chunks: 50,
            reasoning_chunks: 0,
            other_chunks: 0,
            estimated: false,
            model: "slot-model".into(),
            mode: "short".into(),
            error: None,
            looping: false,
            // P2-A surfacing (CF-10/24/44): a clean fixture run.
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
        }]);
        // A C2 reasoning result in its slot (no dedicated table → it lives
        // in `engine_results`).
        app.reasoning_slot.store(ReasoningResult {
            responses: vec!["42".into()],
            ttfts: vec![0.1],
            tg_speeds: vec![10.0],
            score: crate::engines::ReasoningScore {
                total: 13,
                solved: 9,
                by_category: [(4, 5), (3, 5), (2, 3)],
            },
            estimated: false,
        });

        let payload = app.build_export_payload();

        // The session carries the Config form's target/model + the run's
        // session id (not the live snapshot's identity).
        assert_eq!(payload.session.target_url, "http://slot-target:9999/v1");
        assert_eq!(payload.session.model_name, "slot-model");
        assert_eq!(payload.session.session_id, app.session_id);
        // R2: the selected engines are recorded on the session.
        assert!(
            payload
                .session
                .engines
                .as_deref()
                .unwrap()
                .contains("speed"),
            "engines recorded"
        );

        // URC-5: the metric rows come from the speed SLOT (one row) — the
        // (empty) live snapshot would yield zero rows.
        assert_eq!(payload.metrics.len(), 1, "metrics from the speed slot");
        assert_eq!(payload.metrics[0].completion_tokens, Some(50));

        // URC-5: the C2 result is captured in `engine_results`.
        let er = payload
            .engine_results
            .as_ref()
            .expect("engine_results present");
        assert_eq!(er.c2_reasoning.as_ref().unwrap().solved, 9);
    }

    // ── I1: the TUI export's d_energy == the TUI DB's d_energy ──────────

    /// A deterministic mock GPU (150 W) so the export's poller has a
    /// power-bearing trace without any hardware.
    #[derive(Debug)]
    struct MockExportGpu {
        watts: f64,
    }

    impl GpuBackend for MockExportGpu {
        fn vendor(&self) -> &str {
            "NVIDIA"
        }
        fn model(&self) -> Option<&str> {
            Some("Mock GPU")
        }
        fn poll(&self) -> crate::hw::GpuSample {
            crate::hw::GpuSample {
                power_watts: Some(self.watts),
                utilization_pct: Some(90),
                ..Default::default()
            }
        }
    }

    /// I1: the TUI `e`-export's `d_energy` is computed with the **same**
    /// definition as the TUI DB path (`persist_results` /
    /// `engine_results_json`): the shared all-engine token denominator +
    /// the shared `apply_load_window` (full load-window numerator). The
    /// old export recomputed D from the bounded trace with an
    /// Engine-A-only completion denominator — orders of magnitude apart on
    /// a NIAH-heavy run. Stream rows stay consistently NULL (the TUI path
    /// has no per-iteration windows; a single stamped number was
    /// fabricated).
    #[test]
    fn tui_export_d_energy_matches_the_db_definition() {
        let mut app = App::new();
        app.config.url = "http://slot-target:9999/v1".to_string();
        app.config.model = "slot-model".to_string();
        app.config.engine_speed = true;
        // Engine A: 100 completion tokens (50 prompt — not in A's
        // contribution to the all-engine denominator).
        app.speed_slot.store(vec![SpeedResult {
            ttft: 0.25,
            prompt_tokens: 50,
            completion_tokens: 100,
            reasoning_tokens: None,
            pp_speed: 400.0,
            tg_speed: 20.0,
            mtp_efficiency: 1.0,
            stream_time: 2.5,
            total_chunks: 100,
            content_chunks: 100,
            reasoning_chunks: 0,
            other_chunks: 0,
            estimated: false,
            model: "slot-model".into(),
            mode: "short".into(),
            error: None,
            looping: false,
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
        }]);
        // Engine C2: 400 response chars → 100 tokens (the chars/4
        // estimate) in the all-engine denominator.
        app.reasoning_slot.store(ReasoningResult {
            responses: vec!["x".repeat(400)],
            ttfts: vec![0.1],
            tg_speeds: vec![10.0],
            score: crate::engines::ReasoningScore {
                total: 13,
                solved: 9,
                by_category: [(4, 5), (3, 5), (2, 3)],
            },
            estimated: false,
        });
        // The hardware poller: armed (begin_run → start_load), a few
        // 150 W samples, closed (end_load) — a completed run's window.
        let hw = Arc::new(Mutex::new(
            HwPoller::with_backend(Some(Arc::new(MockExportGpu { watts: 150.0 })))
                .with_capacity(1000),
        ));
        {
            let mut p = hw.lock().unwrap();
            p.begin_run();
            p.start_load();
            for _ in 0..5 {
                p.poll();
                std::thread::sleep(std::time::Duration::from_millis(20));
            }
            p.end_load();
        }
        app = app.with_hw(hw.clone());

        let payload = app.build_export_payload();

        // The export carries a d_energy (in both the convenience `energy`
        // object and `engine_results.d_energy` — the same summary).
        let energy = payload.energy.as_ref().expect("export energy present");
        let db_energy = payload
            .engine_results
            .as_ref()
            .and_then(|er| er.d_energy.clone())
            .expect("engine_results.d_energy present");
        assert_eq!(energy.joules, db_energy.joules, "export == DB joules");
        assert_eq!(
            energy.joules_per_token, db_energy.joules_per_token,
            "export == DB J/token"
        );

        // The numerator is the poller's full load window…
        let full_j = hw
            .lock()
            .unwrap()
            .load_window_joules()
            .expect("closed load window");
        assert!(full_j > 0.0, "the window integrated real energy: {full_j}");
        assert_eq!(energy.joules, full_j, "export joules == full load window");

        // …and the denominator is the **all-engine** total (100 A + 100
        // C2 = 200) — not the old Engine-A-only 100.
        let expected_jpt = full_j / 200.0;
        let a_only_jpt = full_j / 100.0;
        let jpt = energy.joules_per_token.expect("power + tokens → a value");
        assert!(
            (jpt - expected_jpt).abs() < 1e-9,
            "J/token divides by all tokens: {jpt} vs {expected_jpt}"
        );
        assert!(
            (jpt - a_only_jpt).abs() > 1e-9,
            "the A-only denominator is not the one used: {jpt} vs {a_only_jpt}"
        );

        // I1: the stream rows are consistently NULL — no single stamped
        // number masquerading as a per-stream metric.
        assert_eq!(payload.metrics.len(), 1);
        assert_eq!(
            payload.metrics[0].joules_per_token, None,
            "TUI stream rows stay NULL (no per-iteration window exists)"
        );
    }

    // ── status-bar server address (never a backend name) ────────────────

    #[test]
    fn url_host_extracts_host_and_port() {
        assert_eq!(fmt::url_host("http://127.0.0.1:8888/v1"), "127.0.0.1:8888");
        assert_eq!(
            fmt::url_host("https://example.com/v1/chat/completions"),
            "example.com"
        );
        assert_eq!(fmt::url_host("localhost:8000"), "localhost:8000");
        assert_eq!(fmt::url_host("http://localhost:8000"), "localhost:8000");
        assert_eq!(fmt::url_host(""), "");
        assert_eq!(fmt::url_host("  "), "");
    }

    #[test]
    fn the_status_bar_never_shows_a_hardcoded_backend() {
        // The snapshot's backend is empty (we don't know the server); the
        // rendered status bar carries the endpoint host instead.
        let app = App::new();
        app.metrics.update(MetricsSnapshot {
            endpoint: "http://127.0.0.1:8888/v1".to_string(),
            model: "swift-27b".to_string(),
            mode: "short".to_string(),
            backend: String::new(),
            ..Default::default()
        });
        let text = rendered_text(&app, 120, 40);
        assert!(
            text.contains("127.0.0.1:8888"),
            "status bar shows the server address: {text}"
        );
        assert!(
            !text.to_ascii_uppercase().contains("VLLM"),
            "no hardcoded vLLM label anywhere in the frame: {text}"
        );
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

    // ── first-run theme picker: confirm path ─────────────────────────────

    #[test]
    fn confirm_theme_applies_and_persists_vampire() {
        let mut app = App::new().with_theme_picker(true);
        app.config.config_path = std::path::PathBuf::from("/tmp/crucible-test-theme-pick.json");
        assert_eq!(app.phase, Phase::ThemePicker);
        app.theme_picker.cursor = 1; // Vampire
        app.confirm_theme();
        assert_eq!(app.active_theme, Theme::Vampire);
        assert_eq!(app.config.theme, "vampire");
        // needs_setup == true → fall through to the Setup takeover.
        assert_eq!(app.phase, Phase::Setup);
    }

    #[test]
    fn confirm_theme_goes_to_dashboard_when_target_given() {
        let mut app = App::new().with_theme_picker(false);
        app.config.config_path = std::path::PathBuf::from("/tmp/crucible-test-theme-pick2.json");
        app.theme_picker.cursor = 2; // Monochrome
        app.confirm_theme();
        assert_eq!(app.active_theme, Theme::Monochrome);
        assert_eq!(app.config.theme, "monochrome");
        // needs_setup == false → straight to the Dashboard (Live).
        assert_eq!(app.phase, Phase::Dashboard);
        assert_eq!(app.view, View::Live);
    }

    #[test]
    fn confirm_theme_defaults_to_cyberpunk() {
        let mut app = App::new().with_theme_picker(true);
        app.config.config_path = std::path::PathBuf::from("/tmp/crucible-test-theme-pick3.json");
        // Cursor 0 = Cyberpunk (the default).
        app.confirm_theme();
        assert_eq!(app.active_theme, Theme::Cyberpunk);
        assert_eq!(app.config.theme, "cyberpunk");
    }

    #[test]
    fn with_theme_sets_the_active_theme() {
        let app = App::new().with_theme(Theme::Vampire);
        assert_eq!(app.active_theme, Theme::Vampire);
    }

    /// End-to-end: switching the active theme re-colors the rendered frame.
    /// A Vampire-themed dashboard shows crimson (the Vampire primary) where
    /// a Cyberpunk one shows neon cyan — proving the whole TUI re-skins.
    #[test]
    fn active_theme_reskins_the_rendered_frame() {
        use ratatui::style::Color;

        fn buffer_colors(app: &App) -> Vec<Color> {
            let backend = ratatui::backend::TestBackend::new(120, 40);
            let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
            terminal.draw(|f| app.render(f)).expect("frame");
            terminal
                .backend()
                .buffer()
                .content()
                .iter()
                .map(|c| c.fg)
                .collect()
        }

        let cyber = App::new().with_theme(Theme::Cyberpunk);
        let vampire = App::new().with_theme(Theme::Vampire);

        // Cyberpunk's neon cyan primary is present in its frame…
        assert!(
            buffer_colors(&cyber).contains(&Color::Rgb(0, 255, 255)),
            "cyberpunk frame has neon cyan"
        );
        // …and is absent from the Vampire frame, which carries crimson instead.
        assert!(
            !buffer_colors(&vampire).contains(&Color::Rgb(0, 255, 255)),
            "vampire frame has no neon cyan"
        );
        assert!(
            buffer_colors(&vampire).contains(&Color::Rgb(220, 20, 60)),
            "vampire frame has crimson"
        );
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
    fn ctrl_c_is_inert_in_setup() {
        // A bare `c` still types into the URL; Ctrl-C (the terminal's
        // copy key) is swallowed — it never quits the app.
        let mut app = App::new().with_setup(&Config::default());
        let r = app.handle_key(&key(KeyCode::Char('c')));
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.setup.url, "c");
        let mut ctrl = key(KeyCode::Char('c'));
        ctrl.modifiers = KeyModifiers::CONTROL;
        let r = app.handle_key(&ctrl);
        assert_eq!(r, KeyAction::Continue, "Ctrl-C is inert");
        assert!(app.running, "Ctrl-C must not quit");
        assert_eq!(app.setup.url, "c", "the Ctrl combo is swallowed, not typed");
    }

    #[test]
    fn setup_esc_at_url_stage_opens_the_quit_confirmation() {
        // Esc at the outermost stage no longer quits outright: it opens
        // the [y/N] overlay. `n` cancels; a confirmed `y` quits.
        let mut app = App::new().with_setup(&Config::default());
        let r = app.handle_key(&key(KeyCode::Esc));
        assert_eq!(r, KeyAction::Continue);
        assert!(app.pending_quit, "Esc opens the confirmation overlay");
        assert!(app.running, "no quit until confirmed");

        app.handle_key(&char_key('n'));
        assert!(!app.pending_quit, "n cancels");
        assert!(app.running);

        app.handle_key(&key(KeyCode::Esc));
        assert!(app.pending_quit);
        let r = app.handle_key(&char_key('y'));
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running, "confirmed y quits");
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

    // ── quit: `q` is the only quit key (confirmed) ─────────────────────

    #[test]
    fn q_opens_the_quit_confirmation_not_a_quit() {
        let mut app = App::new();
        let r = app.handle_key(&char_key('q'));
        assert_eq!(r, KeyAction::Continue, "q alone must not quit");
        assert!(app.pending_quit, "q opens the [y/N] overlay");
        assert!(app.running);
    }

    #[test]
    fn q_confirm_y_quits_the_app() {
        let mut app = App::new();
        app.handle_key(&char_key('q'));
        assert!(app.pending_quit);
        let r = app.handle_key(&char_key('y'));
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running);
        assert!(!app.pending_quit);
    }

    #[test]
    fn q_confirm_uppercase_y_also_quits() {
        let mut app = App::new();
        app.handle_key(&char_key('q'));
        let r = app.handle_key(&char_key('Y'));
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running);
    }

    #[test]
    fn q_confirm_n_cancels_and_stays_running() {
        let mut app = App::new();
        app.handle_key(&char_key('q'));
        assert!(app.pending_quit);
        let r = app.handle_key(&char_key('n'));
        assert_eq!(r, KeyAction::Continue);
        assert!(!app.pending_quit, "n cancels the overlay");
        assert!(app.running, "the app keeps running");
    }

    #[test]
    fn q_confirm_other_keys_cancel_and_are_swallowed() {
        let mut app = App::new();
        app.handle_key(&char_key('q'));
        assert!(app.pending_quit);
        app.handle_key(&char_key('1')); // a stray key
        assert!(!app.pending_quit);
        assert!(app.running);
        assert_eq!(
            app.view,
            View::Live,
            "the swallowed key must not double-act (no view switch)"
        );
    }

    #[test]
    fn ctrl_c_is_inert_in_the_dashboard() {
        let mut app = App::new();
        let mut ctrl = key(KeyCode::Char('c'));
        ctrl.modifiers = KeyModifiers::CONTROL;
        let r = app.handle_key(&ctrl);
        assert_eq!(r, KeyAction::Continue, "Ctrl-C is inert");
        assert!(app.running, "the terminal's copy key must not quit");
    }

    #[test]
    fn esc_no_longer_quits_in_the_dashboard_views() {
        for view in [View::Live, View::Concurrency, View::Needle, View::History] {
            let mut app = App::new();
            app.view = view;
            let r = app.handle_key(&key(KeyCode::Esc));
            assert_eq!(r, KeyAction::Continue, "{view:?}");
            assert!(app.running, "Esc must not quit {view:?}");
        }
    }

    #[test]
    fn q_opens_the_confirmation_from_the_config_view() {
        // Even in the Config view (where `q` would otherwise type), `q`
        // is the quit key: it opens the overlay and never edits a field.
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        app.config.cursor = 0; // URL
        let url_before = app.config.url.clone();
        let r = app.handle_key(&char_key('q'));
        assert_eq!(r, KeyAction::Continue);
        assert!(app.pending_quit);
        assert_eq!(app.config.url, url_before, "q must not type into the field");
        assert!(app.running);
    }

    #[test]
    fn quit_confirmation_works_while_a_benchmark_runs() {
        // `q` is available mid-run (it was before too); confirming quits.
        let mut app = App::new();
        app.seq.set_running(true);
        app.handle_key(&char_key('q'));
        assert!(app.pending_quit);
        let r = app.handle_key(&char_key('y'));
        assert_eq!(r, KeyAction::Quit);
        assert!(!app.running);
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
                ('4', View::Gpu),
                ('5', View::History),
                ('6', View::Config),
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
        assert_eq!(app.view, View::Gpu);
        app.handle_key(&char_key('5'));
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
    fn config_view_digits_type_in_edit_mode() {
        // From the gate (Viewing), `1` leaves straight to Live.
        let mut app = App::new();
        app.view = View::Config; // gate (Viewing)
        app.handle_key(&char_key('1'));
        assert_eq!(app.view, View::Live, "digit 1 exits the gate to Live");

        // In Editing mode, digits type into the focused field (not switch
        // views). Only Esc exits edit mode.
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        app.config.cursor = 9; // Field::Ladder
        app.config.ladder.clear();
        // Point the save (Esc saves) at a temp path so the test
        // never touches the real config file.
        app.config.config_path =
            std::env::temp_dir().join(format!("crucible-digit-{}.json", std::process::id()));
        app.handle_key(&char_key('8'));
        assert_eq!(app.view, View::Config, "8 types into the field");
        assert_eq!(app.config.ladder, "8");
        app.handle_key(&char_key('1'));
        assert_eq!(
            app.view,
            View::Config,
            "1 types into the field (not switch)"
        );
        assert_eq!(app.config.ladder, "81");
        // Esc saves and returns to the gate (stay on tab 6).
        app.handle_key(&key(KeyCode::Esc));
        assert_eq!(app.view, View::Config, "Esc stays on tab 6");
        assert_eq!(
            app.config.edit_mode,
            ConfigMode::Viewing,
            "Esc returns to gate"
        );
        let _ = std::fs::remove_file(&app.config.config_path);
    }

    #[tokio::test]
    async fn config_f5_from_viewing_switches_to_live() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Viewing;
        app.handle_key(&key(KeyCode::F(5)));
        assert_eq!(app.view, View::Live, "F5 from the gate switches to Live");
        assert_eq!(
            app.config.edit_mode,
            ConfigMode::Viewing,
            "edit mode stays at the gate"
        );
    }

    #[tokio::test]
    async fn config_f5_from_editing_switches_to_live() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        app.handle_key(&key(KeyCode::F(5)));
        assert_eq!(app.view, View::Live, "F5 from editing switches to Live");
        assert_eq!(
            app.config.edit_mode,
            ConfigMode::Viewing,
            "edit mode resets to the gate"
        );
    }

    #[test]
    fn config_esc_in_editing_saves_and_stays_on_tab6() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        // Point the save at a temp path so the test never touches the
        // real config file.
        app.config.config_path =
            std::env::temp_dir().join(format!("crucible-esc-{}.json", std::process::id()));
        app.handle_key(&key(KeyCode::Esc));
        assert_eq!(app.view, View::Config, "Esc stays on tab 6");
        assert_eq!(
            app.config.edit_mode,
            ConfigMode::Viewing,
            "Esc returns to the gate"
        );
        assert!(app.config.saved, "Esc saves the form");
        let _ = std::fs::remove_file(&app.config.config_path);
    }

    #[test]
    fn footer_highlights_the_current_view() {
        let mut app = App::new();
        app.view = View::Concurrency;
        let t = line_text(&app.footer_line());
        assert!(t.contains("[2]Conc"), "footer names the current view: {t}");
        // The current view's tab is the accent-highlighted one.
        let line = app.footer_line();
        let span = line
            .spans
            .iter()
            .find(|s| s.content.as_ref() == "[2]Conc")
            .expect("the Concurrency tab span");
        assert_eq!(
            span.style.fg,
            Some(Theme::Cyberpunk.primary()),
            "current view is accent"
        );
    }

    // ── state-aware footer ───────────────────────────────────────────────

    #[test]
    fn footer_is_idle_when_nothing_runs() {
        let app = App::new();
        let t = line_text(&app.footer_line());
        // All six view tabs, the connection dot, and the idle action keys.
        assert!(t.contains("[1]Live"), "{t}");
        assert!(t.contains("[4]GPU"), "{t}");
        assert!(t.contains("[5]Hist"), "{t}");
        assert!(t.contains("[6]Cfg"), "{t}");
        assert!(t.contains("●"), "connection dot present: {t}");
        assert!(t.contains("[R]Run"), "idle shows the run key: {t}");
        assert!(t.contains("[q]Quit"), "{t}");
    }

    #[test]
    fn footer_shows_the_connection_state() {
        // A target URL is set (the default config has one) → connected.
        let app = App::new();
        let t = line_text(&app.footer_line());
        assert!(t.contains("Connected"), "connected dot: {t}");
        // No target → disconnected.
        let mut app2 = App::new();
        app2.config.url.clear();
        let t2 = line_text(&app2.footer_line());
        assert!(t2.contains("Disconnected"), "disconnected dot: {t2}");
    }

    #[test]
    fn footer_locks_the_server_keys_while_running() {
        let app = App::new();
        app.seq.set_running(true);
        let t = line_text(&app.footer_line());
        assert!(t.contains("[Space]Pause"), "running shows pause: {t}");
        assert!(t.contains("[q]Quit"), "{t}");
    }

    #[test]
    fn footer_shows_the_engine_status_while_running() {
        // A sequence with live progress → `Engine B: Step 5/9`.
        let app = App::new();
        app.seq.store(crate::engines::SeqState {
            phase: crate::engines::SeqPhase::Running,
            queue: vec![crate::engines::Engine::Concurrency],
            engine: crate::engines::Engine::Concurrency,
            progress: Some(crate::engines::EngineProgress::Concurrency {
                level: 16,
                step: 5,
                total_steps: 9,
                active: 16,
            }),
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        });
        let t = line_text(&app.footer_line());
        assert!(t.contains("Engine B: Step 5/9"), "engine status: {t}");
    }

    #[test]
    fn footer_shows_resume_when_paused() {
        let mut app = App::new();
        app.seq.set_running(true);
        app.handle_key(&char_key(' '));
        let t = line_text(&app.footer_line());
        assert!(t.contains("[Space]Resume"), "paused shows resume: {t}");
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

    // ── terminal-size edge case ──────────────────────────────────────────

    fn rendered_text(app: &App, w: u16, h: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(w, h);
        let mut terminal = ratatui::Terminal::new(backend).expect("terminal");
        terminal.draw(|f| app.render(f)).expect("draw frame");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn render_shows_the_too_narrow_message_under_80_cols() {
        let app = App::new();
        let text = rendered_text(&app, 60, 24);
        assert!(text.contains("Terminal too narrow"), "{text}");
        assert!(text.contains("minimum 80 columns"), "{text}");
    }

    #[test]
    fn render_shows_the_full_ui_at_80_cols_and_wider() {
        let app = App::new();
        for (w, h) in [(80, 24), (120, 40)] {
            let text = rendered_text(&app, w, h);
            assert!(!text.contains("Terminal too narrow"), "{w}x{h}: {text}");
            assert!(text.contains("BENCHMARK SEQUENCE"), "dashboard: {text}");
        }
    }

    // ── Matrix Report (M3) ───────────────────────────────────────────────

    use crate::hw::GpuSample;

    /// A deterministic GPU backend (the Matrix Report assembly fixture):
    /// a fixed marketing name + a fixed **binary-MiB** VRAM total.
    #[derive(Debug)]
    struct MockGpu {
        vram_mib: u64,
    }

    impl GpuBackend for MockGpu {
        fn vendor(&self) -> &str {
            "Mock"
        }
        fn model(&self) -> Option<&str> {
            Some("Mock GPU 24G")
        }
        fn poll(&self) -> GpuSample {
            GpuSample {
                memory_total_mb: Some(self.vram_mib),
                ..GpuSample::default()
            }
        }
    }

    /// A minimal Engine A result fixture (the report's speed-mean input).
    fn speed_fixture(tg: f64, failed: bool, looping: bool) -> SpeedResult {
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
            model: "test-model".into(),
            mode: "short".into(),
            error: failed.then(|| "boom".to_string()),
            looping,
            usage_discrepancy: false,
            e2e_only: false,
            itl_clamped: 0,
        }
    }

    #[test]
    fn speed_mean_tps_is_over_the_valid_runs_only() {
        // The canonical filter (`!is_failed() && !looping`): the failed
        // and looping runs are excluded from the mean.
        assert_eq!(
            speed_mean_tps(&[
                speed_fixture(90.0, false, false),
                speed_fixture(0.0, true, false),
                speed_fixture(30.0, false, true),
            ]),
            90.0
        );
        assert_eq!(
            speed_mean_tps(&[
                speed_fixture(40.0, false, false),
                speed_fixture(60.0, false, false),
            ]),
            50.0
        );
        // No valid runs → 0.0 (the report renders a blank, never a
        // silent 0).
        assert_eq!(speed_mean_tps(&[]), 0.0);
        assert_eq!(speed_mean_tps(&[speed_fixture(0.0, true, false)]), 0.0);
    }

    #[test]
    fn build_matrix_report_assembles_the_auto_fields() {
        let mut app = App::new();
        app.config.model = "test-model".to_string();
        let vram_mib = 24576; // 24 GiB (binary)
        app.gpu = Some(Arc::new(MockGpu { vram_mib }) as Arc<dyn GpuBackend>);
        app.hw = Some(Arc::new(Mutex::new(HwPoller::with_backend(Some(
            Arc::new(MockGpu { vram_mib }) as Arc<dyn GpuBackend>,
        )))));
        // The lock-free aggregate sample (the VRAM source, binary MiB).
        app.metrics.update(MetricsSnapshot {
            gpu: Some(GpuSample {
                memory_total_mb: Some(vram_mib),
                ..GpuSample::default()
            }),
            ..MetricsSnapshot::default()
        });
        // Engine A: one valid run (90 t/s) + one failed + one looping
        // (30) → the mean is the valid run alone.
        app.speed_slot.store(vec![
            speed_fixture(90.0, false, false),
            speed_fixture(0.0, true, false),
            speed_fixture(30.0, false, true),
        ]);

        let r = app.build_matrix_report();
        assert_eq!(r.model, "test-model");
        // VRAM via the **MiB→tier** path (binary): 24576 MiB = 24 GiB →
        // `16-24`. (The decimal trap: the same bytes read 25.77 GB and
        // would cross into `24-32`.)
        assert_eq!(r.vram, "24GB — 16-24");
        // The real Engine A mean — no tilde, no estimate.
        assert_eq!(r.speed, "90 tok/s");
        // Hardware: the display name + count + binary-GB VRAM + the
        // (real) host-RAM tail.
        assert!(
            r.hardware.starts_with("1x Mock GPU 24G 24GB + "),
            "hardware prefix: {}",
            r.hardware
        );
        assert!(
            r.hardware.ends_with("GB RAM"),
            "hardware RAM tail: {}",
            r.hardware
        );
        // The user fields start empty (the user fills them in the
        // takeover).
        assert_eq!(r.backend, "");
        assert_eq!(r.quant, "");
        assert_eq!(r.use_case, "");
        assert_eq!(r.notes, "");
    }

    #[test]
    fn build_matrix_report_degrades_honestly_without_a_gpu() {
        let mut app = App::new();
        app.config.model = "test-model".to_string();
        app.speed_slot
            .store(vec![speed_fixture(45.4, false, false)]);

        let r = app.build_matrix_report();
        // No GPU: the hardware is the host RAM only (never a fabricated
        // `1x …`), the vram is blank (the N/A rule — never a tier), and
        // the speed is the real mean.
        assert!(
            r.hardware.ends_with("GB RAM"),
            "RAM-only hardware: {}",
            r.hardware
        );
        assert!(!r.hardware.contains('x'), "{}", r.hardware);
        assert_eq!(r.vram, "");
        assert_eq!(r.speed, "45 tok/s");
        assert_eq!(r.model, "test-model");
    }

    #[test]
    fn report_copy_message_is_ok_and_na_never_fail() {
        // The `c` key's handler decision, pure over the outcome (the
        // N/A-never-fail rule — a broken sink is a warning, never a
        // crash).
        assert_eq!(
            App::report_copy_message(&Ok(())),
            "[report] copied to clipboard (OSC 52)"
        );
        assert_eq!(
            App::report_copy_message(&Err("broken sink".to_string())),
            "[report] clipboard unsupported by terminal — use [S]ave"
        );
    }

    #[test]
    fn report_copy_key_logs_success_and_stays_open() {
        let mut app = App::new();
        app.phase = Phase::Report;
        app.report = ReportState::new(
            MatrixReport {
                model: "test-model".into(),
                ..Default::default()
            },
            "/tmp/exports/matrix-1.md".to_string(),
        );
        // In a piped `cargo test` the OSC 52 write to stdout succeeds
        // (inert bytes on the pipe — the clipboard.rs smoke test
        // establishes the convention) → the success log. The app never
        // crashes and stays in the report.
        app.handle_key(&key(KeyCode::Char('c')));
        let log_text: Vec<String> = app.log.iter().map(line_text).collect();
        assert!(
            log_text
                .iter()
                .any(|l| l == "[report] copied to clipboard (OSC 52)"),
            "the c key logs the copy: {log_text:?}"
        );
        assert!(app.running, "copy never quits the app");
        assert_eq!(app.phase, Phase::Report, "stays in the report after copy");
    }

    #[test]
    fn report_save_key_writes_the_block_to_the_save_path() {
        let mut app = App::new();
        app.phase = Phase::Report;
        let path = std::env::temp_dir().join(format!("crucible-report-{}.md", std::process::id()));
        app.report = ReportState::new(
            MatrixReport {
                hardware: "1x Mock GPU 24G 24GB + 64GB RAM".into(),
                vram: "24GB — 16-24".into(),
                model: "test-model".into(),
                ..Default::default()
            },
            path.to_string_lossy().into_owned(),
        );
        // A typed user field must land in the written block (write-back).
        app.report.backend = "vLLM".into();
        app.report.sync_report();
        app.handle_key(&key(KeyCode::Char('s')));

        let content = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("the s key wrote the file: {e}"));
        assert!(content.contains("[COMMUNITY-SUBMISSION]"), "{content}");
        assert!(content.contains("backend:  vLLM"), "{content}");
        assert!(content.contains("[/COMMUNITY-SUBMISSION]"), "{content}");
        let log_text: Vec<String> = app.log.iter().map(line_text).collect();
        assert!(
            log_text
                .iter()
                .any(|l| l.starts_with("[report] saved → ") && l.contains("crucible-report-")),
            "the s key logs the path: {log_text:?}"
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn report_esc_returns_to_the_dashboard() {
        let mut app = App::new();
        app.phase = Phase::Report;
        app.handle_key(&key(KeyCode::Esc));
        assert_eq!(app.phase, Phase::Dashboard, "Esc closes the report");
        assert!(app.running);
    }

    // ── Run-log viewer (Phase::Log, the `l` key) ─────────────────────────

    /// A temp run-log dir + a real (non-disabled) logger writing into it
    /// (the offline `open_log` / archive-scan fixture — no real
    /// `~/.local/share` touched).
    fn temp_logger(name: &str) -> (std::path::PathBuf, std::sync::Arc<RunLogger>) {
        let dir = std::env::temp_dir().join(format!("crucible-logview-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        (dir, logger)
    }

    #[test]
    fn l_key_opens_the_log_view_without_a_completion_gate() {
        // The log exists from process start — no run is needed (this
        // app's logger is disabled: the view opens in its N/A placeholder
        // mode, never a crash).
        let mut app = App::new();
        let r = app.handle_key(&char_key('l'));
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.phase, Phase::Log, "l opens the log view immediately");
        assert!(app.running);
    }

    #[test]
    fn l_key_opens_the_log_view_mid_run() {
        // No completion gate even while a benchmark sequence is running
        // (unlike `m`): the log is being written live.
        let mut app = App::new();
        app.seq.set_running(true);
        app.handle_key(&char_key('l'));
        assert_eq!(app.phase, Phase::Log, "mid-run l opens the log view");
        assert!(app.seq.is_running(), "the run is untouched by the view");
    }

    #[test]
    fn l_key_in_the_config_view_is_scoped_to_editing() {
        // View 6 is a text form: at the gate `l` is ignored (and never
        // opens the log view); in Editing, `l` types into the focused
        // field — the same scope as the `c` key.
        let mut app = App::new();
        app.view = View::Config; // gate
        app.config.cursor = 0; // URL
        app.config.url.clear(); // start from a known-empty field
        app.handle_key(&char_key('l'));
        assert_eq!(
            app.phase,
            Phase::Dashboard,
            "`l` must not open the log view from the gate"
        );
        assert_eq!(app.config.url, "", "the gate ignores typing");

        app.config.edit_mode = ConfigMode::Editing;
        app.handle_key(&char_key('l'));
        assert_eq!(
            app.phase,
            Phase::Dashboard,
            "`l` must not open the log view in Editing"
        );
        assert_eq!(app.config.url, "l", "`l` types into the URL in Editing");
    }

    #[test]
    fn l_key_in_the_history_view_opens_the_cursor_session_archive() {
        // A.5: from the History view, `l` opens the cursor session's
        // archive log (run-<ts>-<session8>.log).
        let (dir, logger) = temp_logger("history-l");
        let session_id = "12345678-aaaa-bbbb-cccc-dddddddddddd".to_string();
        let archive_name = "run-20261019-010000-12345678.log";
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join(archive_name),
            "[01:00:00.000] [INFO] [SETUP] archived run\n",
        )
        .unwrap();
        logger.finish();

        let mut app = App::new().with_logger(logger);
        app.view = View::History;
        app.history = Some(HistoryState {
            sessions: vec![crate::storage::models::BenchmarkSession {
                session_id: session_id.clone(),
                ..Default::default()
            }],
            cursor: 0,
            ..Default::default()
        });
        app.handle_key(&char_key('l'));
        assert_eq!(app.phase, Phase::Log, "history l opens the log view");
        assert_eq!(
            app.log_view.path,
            dir.join(archive_name).display().to_string(),
            "the archive path is shown"
        );
        assert!(
            app.log_view.lines.iter().any(|l| l.contains("archived run")),
            "the archive content is read: {:?}",
            app.log_view.lines
        );
        assert!(!app.log_view.unavailable);
        // The History view state is untouched by the phase change.
        assert_eq!(app.view, View::History);
        assert!(app.history.is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_log_reads_the_file_log_content() {
        // A real logger in a temp dir: `latest.log` carries the records
        // (the writer thread formats them).
        let (dir, logger) = temp_logger("open-log");
        logger.info(Context::Setup, "run started — url=http://x");
        logger.warn(Context::EngineB, "step 2 slow");
        logger.finish();

        let mut app = App::new().with_logger(logger);
        app.open_log();
        assert_eq!(app.phase, Phase::Log);
        assert_eq!(
            app.log_view.path,
            dir.join("latest.log").display().to_string()
        );
        assert!(
            app.log_view
                .lines
                .iter()
                .any(|l| l.contains("[INFO] [SETUP] run started")),
            "the file log content is in the state: {:?}",
            app.log_view.lines
        );
        assert!(
            app.log_view
                .lines
                .iter()
                .any(|l| l.contains("[WARN] [ENGINE_B] step 2 slow"))
        );
        assert!(!app.log_view.unavailable);
        // The save destination is pre-filled under the exports dir.
        assert!(app.log_view.save_path.contains("exports/run-log-"), "{}", app.log_view.save_path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn open_log_with_a_disabled_logger_is_the_placeholder() {
        let mut app = App::new(); // App::new() → the disabled logger
        app.open_log();
        assert_eq!(app.phase, Phase::Log, "the view opens regardless");
        assert!(app.log_view.unavailable, "N/A-never-fail placeholder mode");
        assert!(
            app.log_view
                .lines
                .iter()
                .any(|l| l.contains("logger disabled")),
            "{:?}",
            app.log_view.lines
        );
    }

    #[test]
    fn open_log_for_session_not_found_is_the_placeholder() {
        let (dir, logger) = temp_logger("no-archive");
        logger.finish();
        let mut app = App::new().with_logger(logger);
        app.open_log_for_session("ffffffff-eeee-dddd-cccc-aaaaaaaaaaaa");
        assert_eq!(app.phase, Phase::Log);
        assert!(app.log_view.unavailable);
        assert!(
            app.log_view
                .lines
                .iter()
                .any(|l| l.contains("no archive log found for session ffffffff")),
            "{:?}",
            app.log_view.lines
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn log_esc_returns_to_the_dashboard() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = LogState::new(
            vec!["line one".to_string()],
            "/tmp/latest.log".to_string(),
            "/tmp/run-log.log".to_string(),
        );
        let r = app.handle_key(&key(KeyCode::Esc));
        assert_eq!(r, KeyAction::Continue);
        assert_eq!(app.phase, Phase::Dashboard, "Esc closes the log view");
        assert!(app.running);
    }

    #[test]
    fn log_scrolling_via_the_app_key_path() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = LogState::new(
            (0..20).map(|i| format!("line {i}")).collect(),
            "/tmp/latest.log".to_string(),
            "/tmp/run-log.log".to_string(),
        );
        app.handle_key(&char_key('j'));
        assert_eq!(app.log_view.scroll, 1);
        app.handle_key(&key(KeyCode::Down));
        assert_eq!(app.log_view.scroll, 2);
        app.handle_key(&char_key('k'));
        assert_eq!(app.log_view.scroll, 1);
        app.handle_key(&key(KeyCode::PageDown));
        assert_eq!(app.log_view.scroll, 11);
        app.handle_key(&key(KeyCode::PageUp));
        assert_eq!(app.log_view.scroll, 1);
        app.handle_key(&key(KeyCode::End));
        assert_eq!(app.log_view.scroll, 19, "End = the last line index");
        app.handle_key(&key(KeyCode::Home));
        assert_eq!(app.log_view.scroll, 0);
    }

    #[test]
    fn log_q_is_inert_and_never_opens_the_quit_overlay() {
        // The Log branch returns before the q-quit arm (the same rule as
        // the report).
        let mut app = App::new();
        app.phase = Phase::Log;
        let r = app.handle_key(&char_key('q'));
        assert_eq!(r, KeyAction::Continue);
        assert!(!app.pending_quit, "q must not open the quit overlay");
        assert_eq!(app.phase, Phase::Log, "the view stays open");
        assert!(app.running);
    }

    #[test]
    fn log_copy_message_is_ok_and_na_never_fail() {
        // The `c` key's handler decision, pure over the outcome (the
        // N/A-never-fail rule — a broken sink is a warning, never a
        // crash).
        assert_eq!(
            App::log_copy_message(&Ok(())),
            "[log] copied to clipboard (OSC 52)"
        );
        assert_eq!(
            App::log_copy_message(&Err("broken sink".to_string())),
            "[log] clipboard unsupported by terminal — use [S]ave"
        );
    }

    #[test]
    fn log_copy_key_logs_success_and_stays_open() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = LogState::new(
            vec![
                "[01:00:00.000] [INFO] [SETUP] run started".to_string(),
                "[01:00:01.000] [INFO] [SSE] token #1".to_string(),
            ],
            "/tmp/latest.log".to_string(),
            "/tmp/run-log.log".to_string(),
        );
        // In a piped `cargo test` the OSC 52 write to stdout succeeds
        // (inert bytes on the pipe — the clipboard.rs smoke test
        // establishes the convention) → the success log. The app never
        // crashes and stays in the log view.
        app.handle_key(&key(KeyCode::Char('c')));
        let log_text: Vec<String> = app.log.iter().map(line_text).collect();
        assert!(
            log_text
                .iter()
                .any(|l| l == "[log] copied to clipboard (OSC 52)"),
            "the c key logs the copy: {log_text:?}"
        );
        assert!(app.running, "copy never quits the app");
        assert_eq!(app.phase, Phase::Log, "stays in the log view after copy");
    }

    #[test]
    fn log_save_key_writes_the_log_to_the_save_path() {
        let mut app = App::new();
        app.phase = Phase::Log;
        let path = std::env::temp_dir().join(format!("crucible-logview-{}.log", std::process::id()));
        app.log_view = LogState::new(
            vec![
                "=== crucible-llm run log — 2026-10-19 01:00:00 ===".to_string(),
                "[01:00:00.000] [INFO] [SETUP] run started".to_string(),
            ],
            "/tmp/logs/latest.log".to_string(),
            path.to_string_lossy().into_owned(),
        );
        app.handle_key(&key(KeyCode::Char('s')));

        let content =
            std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("the s key wrote the file: {e}"));
        assert!(content.contains("=== crucible-llm run log"), "{content}");
        assert!(content.contains("run started"), "{content}");
        let log_text: Vec<String> = app.log.iter().map(line_text).collect();
        assert!(
            log_text
                .iter()
                .any(|l| l.starts_with("[log] saved → ") && l.contains("crucible-logview-")),
            "the s key logs the path: {log_text:?}"
        );
        assert_eq!(app.phase, Phase::Log, "the view stays open after save");
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn log_save_path_is_under_the_exports_dir() {
        // `exports` is the established user-save location (the same dir
        // the Matrix Report saves to).
        let p = App::log_save_path(123);
        assert!(p.contains("exports/run-log-123.log"), "{p}");
        assert!(p.starts_with(&crate::config::data_dir().display().to_string()), "{p}");
    }

    #[test]
    fn footer_shows_the_log_key() {
        let app = App::new();
        let t = line_text(&app.footer_line());
        assert!(t.contains("[L]og"), "the footer names the log key: {t}");
    }

    #[test]
    fn render_phase_log_shows_the_takeover() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = LogState::new(
            vec![
                "=== crucible-llm run log — 2026-10-19 01:00:00 (session 12345678) ===".to_string(),
                "[01:00:00.000] [INFO] [SETUP] run started".to_string(),
            ],
            "/tmp/logs/latest.log".to_string(),
            "/tmp/exports/run-log-1.log".to_string(),
        );
        let text = rendered_text(&app, 120, 40);
        assert!(text.contains("RUN LOG"), "the top bar: {text}");
        assert!(text.contains("run started"), "the log line: {text}");
        assert!(text.contains("[C]opy"), "{text}");
        assert!(text.contains("[S]ave"), "{text}");
        assert!(text.contains("[Esc]lose"), "{text}");
        assert!(text.contains("/tmp/logs/latest.log"), "the path: {text}");
        // The dashboard must not render underneath the takeover.
        assert!(!text.contains("BENCHMARK SEQUENCE"), "{text}");
    }
}
