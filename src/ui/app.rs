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

use crate::engines::SweepResult;
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::ui::theme::{palette, style};
use crate::ui::views;

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
        }
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
            _ => KeyAction::Continue,
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
