//! `App` state machine: the `View` enum (`Live`, `Concurrency`, `Needle`,
//! `History`, `Config`) and key handling (blueprint §6 footer: `1`-`5` switch
//! views, `Space` pause/resume, `+` step concurrency, `n` new needle,
//! `e` export, `q`/`Esc` quit).
//!
//! The `App` holds the shared `Arc<MetricsState>` — the `ArcSwap`
//! double-buffered [`MetricsSnapshot`] pipeline (Chunk 6). The render loop
//! only ever *reads* the snapshot via `MetricsState::load()` (a lock-free
//! atomic read); the stream worker *writes* it via `MetricsState::update()`.
//! The render loop never writes the snapshot and never touches the timing
//! path (measurement-isolation invariant, blueprint §4).

use std::sync::Arc;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::config::{Config, ExportFormat};
use crate::engines::{NiahEngineConfig, NiahSlot, SweepResult};
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::storage::db::Database;
use crate::storage::export::{self, ExportPayload};
use crate::storage::models::{BenchmarkSession, StreamMetricRow};
use crate::ui::theme::{palette, style};
use crate::ui::views;
use crate::ui::views::history::HistoryState;

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
    /// `n` — queue a new needle test.
    NewNeedle,
    /// `e` — export results (JSON/MD/CSV).
    Export,
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
    /// The last completed concurrency sweep (plan Chunk 10). `None` until a
    /// sweep has run; View 2 renders the curve from this.
    pub sweep: Option<Arc<SweepResult>>,
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
        let log = vec![
            Line::from(Span::styled(
                "[12:44:02] Stream #04 completed: 840 tokens in 12.19s (68.9 t/s). Speculative Acceptance: 79%",
                style::value_ok(),
            )),
            Line::from(Span::styled(
                "[12:44:03] Warning: Stream #03 prefill context reached 4096 tokens. Server KV memory allocation +400MB",
                style::value_warn(),
            )),
            Line::from(Span::styled(
                "[12:44:04] Concurrency step up: Spawning batch 17..24.",
                style::label(),
            )),
        ];
        let metrics = Arc::new(MetricsState::new());
        // Seed the initial snapshot with blueprint-mock sample data. A real
        // stream worker overwrites this via `MetricsState::update()`.
        metrics.update(MetricsSnapshot::sample());
        Self {
            running: true,
            paused: false,
            view: View::Live,
            concurrency_target: 1,
            tick: 0,
            metrics,
            log,
            sweep: None,
            export_format: ExportFormat::default(),
            history: None,
            niah: Arc::new(NiahSlot::new()),
            niah_config: None,
        }
    }

    /// Supply the NIAH-relevant config snapshot (Chunk 15) so the `n`
    /// key can spawn a background matrix run against the target endpoint.
    pub fn with_niah_config(mut self, cfg: &Config) -> Self {
        self.niah_config = Some(NiahEngineConfig::from_config(cfg));
        self
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
    /// background NIAH matrix run (Chunk 15, Engine C1).
    ///
    /// This is a *key-path* action (never the render path): the runner
    /// task publishes its result to the lock-free [`NiahSlot`] when done,
    /// and View 3 reads it — so a running matrix never perturbs the 60Hz
    /// render loop or the timing path (measurement-isolation invariant,
    /// blueprint §4). Re-pressing `n` while a run is in progress is a
    /// no-op (logged).
    pub fn start_niah(&mut self) {
        if self.niah.is_running() {
            self.push_log(
                "[niah] already running — one size × depth cell at a time".to_string(),
                style::value_warn(),
            );
            return;
        }
        let config = match self.niah_config.clone() {
            Some(c) => c,
            None => {
                self.push_log(
                    "[niah] no target configured — set --url first".to_string(),
                    style::value_warn(),
                );
                return;
            }
        };
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
        let slot = self.niah.clone();
        self.niah.set_running(true);
        self.push_log(
            format!(
                "[niah] started: {} sizes × {} depths → {}",
                sizes.len(),
                depths.len(),
                config.url
            ),
            style::value_ok(),
        );
        // The runner owns the slot handle; on completion it clears the
        // running flag and publishes the scored grid.
        tokio::spawn(async move {
            let result = engine.run().await;
            slot.set_running(false);
            slot.store(result);
        });
    }

    /// Handle one terminal key event (blueprint §6 footer key map).
    pub fn handle_key(&mut self, key: &KeyEvent) -> KeyAction {
        // Terminal etiquette: Ctrl-C quits like any other quit key.
        if key.modifiers.contains(KeyModifiers::CONTROL) && key.code == KeyCode::Char('c') {
            self.running = false;
            return KeyAction::Quit;
        }
        match key.code {
            KeyCode::Char('q') | KeyCode::Esc => {
                self.running = false;
                KeyAction::Quit
            }
            KeyCode::Char(c @ '1'..='5') => {
                if let Some(view) = View::from_digit(c as u8) {
                    self.view = view;
                    // Chunk 14: entering the History view loads the stored
                    // session list once (key path, not the render path).
                    if view == View::History {
                        self.ensure_history();
                    }
                }
                KeyAction::Continue
            }
            KeyCode::Char(' ') => {
                self.paused = !self.paused;
                KeyAction::PauseResume
            }
            KeyCode::Char('+') | KeyCode::Char('=') => {
                self.concurrency_target = (self.concurrency_target * 2).min(128);
                KeyAction::StepConcurrency
            }
            KeyCode::Char('n') => KeyAction::NewNeedle,
            KeyCode::Char('e') => KeyAction::Export,
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
        let session = BenchmarkSession {
            session_id: uuid::Uuid::new_v4().to_string(),
            timestamp: None,
            target_url: snap.endpoint.clone(),
            model_name: snap.model.clone(),
            backend_type: (!snap.backend.is_empty()).then(|| snap.backend.clone()),
            quantization: None,
            system_gpu: None,
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
                joules_per_token: None,
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
    /// The metric snapshot itself is *not* touched here: it is published by
    /// the stream worker / engine via `MetricsState::update()` and read
    /// lock-free in the render path (`render` → views → `MetricsState::load`).
    /// Keeping the write out of the render loop is the measurement-isolation
    /// invariant (blueprint §4). `Space` (paused) freezes the clock.
    pub fn on_tick(&mut self) {
        if self.paused {
            return;
        }
        self.tick += 1;
    }

    /// Draw the full frame: status bar, tab bar, current view, footer.
    pub fn render(&self, f: &mut Frame) {
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

    /// Bottom key-hint footer (blueprint §6).
    fn render_footer(&self, area: Rect, f: &mut Frame) {
        let line = Line::from(vec![
            Span::styled(" [Space] Pause/Resume", style::footer()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled("[+] Step Concurrency", style::footer()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled("[N] New Needle Test", style::footer()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled("[E] Export", style::footer()),
            Span::styled(" | ", style::tab_separator()),
            Span::styled("[Q] Quit", style::footer()),
        ]);
        f.render_widget(Paragraph::new(line), area);
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
