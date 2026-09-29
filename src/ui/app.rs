//! `App` state machine: the `View` enum (`Live`, `Concurrency`, `Needle`,
//! `History`, `Config`) and key handling (blueprint §6 footer: `1`-`5` switch
//! views, `Space` pause/resume, `+` step concurrency, `n` new needle,
//! `e` export, `q`/`Esc` quit).
//!
//! Also carries the placeholder metric state the views render until the real
//! `ArcSwap<MetricsSnapshot>` pipeline (Chunk 6) is wired in. The render
//! loop only ever *reads* this state — it never touches the timing path
//! (measurement-isolation invariant, blueprint §4).

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

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

/// One row of the Active Streams Monitor (blueprint §6, View 1).
#[derive(Debug, Clone)]
pub struct StreamRow {
    pub id: u32,
    /// `Reasoning` | `Content` | `Tool-Call`.
    pub kind: &'static str,
    /// `Streaming` | `Waiting` | `Done`.
    pub state: &'static str,
    pub pp_tokens: Option<usize>,
    pub tg_tokens: Option<usize>,
    pub ttft_s: Option<f64>,
    pub gen_tps: Option<f64>,
    /// MTP multiplier (`tokens / packets`).
    pub mtp: Option<f64>,
    /// 0.0..=1.0.
    pub progress: f64,
}

/// Placeholder metric snapshot the views render while the real
/// `ArcSwap<MetricsSnapshot>` (Chunk 6) is not yet wired in. Values mirror
/// the blueprint §6 mockup so the layout is verifiable against it.
#[derive(Debug, Clone)]
pub struct PlaceholderMetrics {
    pub endpoint: &'static str,
    pub backend: &'static str,
    pub model: &'static str,
    pub mode: &'static str,
    pub aggregate_tps: f64,
    pub active_streams: usize,
    pub total_streams: usize,
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub power_w: f64,
    pub joules_per_token: f64,
    pub itl_p50_ms: f64,
    pub itl_p90_ms: f64,
    pub itl_p99_ms: f64,
    /// Normalized ITL histogram bins (0.0..=1.0), low → high latency.
    pub itl_bins: [f64; 24],
    pub streams: Vec<StreamRow>,
    /// Rolling aggregate-throughput window (one sample per second, last 60).
    pub throughput_series: Vec<f64>,
}

impl PlaceholderMetrics {
    /// Sample values straight from the blueprint §6 mockup.
    pub fn sample() -> Self {
        let itl_bins = [
            0.92, 0.86, 0.79, 0.71, 0.62, 0.53, 0.44, 0.36, 0.29, 0.23, 0.18, 0.14, 0.11, 0.08,
            0.06, 0.045, 0.033, 0.024, 0.017, 0.012, 0.008, 0.005, 0.003, 0.002,
        ];
        Self {
            endpoint: "http://127.0.0.1:8000/v1",
            backend: "vLLM",
            model: "Qwen3.6-35B-A3B-UD-Q4_K_XL",
            mode: "Concurrency",
            aggregate_tps: 842.3,
            active_streams: 16,
            total_streams: 16,
            vram_used_gb: 21.4,
            vram_total_gb: 24.0,
            power_w: 285.0,
            joules_per_token: 0.338,
            itl_p50_ms: 12.1,
            itl_p90_ms: 16.4,
            itl_p99_ms: 41.2,
            itl_bins,
            streams: vec![
                StreamRow {
                    id: 1,
                    kind: "Reasoning",
                    state: "Streaming",
                    pp_tokens: Some(2048),
                    tg_tokens: Some(312),
                    ttft_s: Some(0.182),
                    gen_tps: Some(72.4),
                    mtp: Some(1.84),
                    progress: 0.55,
                },
                StreamRow {
                    id: 2,
                    kind: "Content",
                    state: "Streaming",
                    pp_tokens: Some(512),
                    tg_tokens: Some(180),
                    ttft_s: Some(0.045),
                    gen_tps: Some(88.1),
                    mtp: Some(1.02),
                    progress: 0.70,
                },
                StreamRow {
                    id: 3,
                    kind: "Tool-Call",
                    state: "Waiting",
                    pp_tokens: Some(4096),
                    tg_tokens: None,
                    ttft_s: None,
                    gen_tps: None,
                    mtp: None,
                    progress: 0.0,
                },
                StreamRow {
                    id: 4,
                    kind: "Reasoning",
                    state: "Done",
                    pp_tokens: Some(2048),
                    tg_tokens: Some(840),
                    ttft_s: Some(0.191),
                    gen_tps: Some(68.9),
                    mtp: Some(1.79),
                    progress: 1.0,
                },
            ],
            throughput_series: (0..60)
                .map(|i| 842.3 + 18.0 * ((i as f64) * 0.31).sin())
                .collect(),
        }
    }
}

/// Application state machine: current view, navigation, pause state, and
/// the (placeholder) metric snapshot the views render.
#[derive(Debug)]
pub struct App {
    pub running: bool,
    pub paused: bool,
    pub view: View,
    /// Concurrency target the `+` key steps up (plumbed to the worker pool
    /// in later chunks).
    pub concurrency_target: usize,
    /// Monotonic 60Hz tick counter (placeholder animation clock).
    pub tick: u64,
    pub metrics: PlaceholderMetrics,
    pub log: Vec<Line<'static>>,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    /// Fresh app in the Live view, running, with blueprint-mock sample data.
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
        Self {
            running: true,
            paused: false,
            view: View::Live,
            concurrency_target: 1,
            tick: 0,
            metrics: PlaceholderMetrics::sample(),
            log,
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

    /// Advance the 60Hz placeholder clock. Until the real engine snapshot
    /// exists (Chunk 6), this animates the sample metrics so the redraw
    /// cycle is visible; `Space` (paused) freezes it.
    pub fn on_tick(&mut self) {
        if self.paused {
            return;
        }
        self.tick += 1;
        let t = self.tick as f64;
        self.metrics.aggregate_tps = 842.3 + 18.0 * t.sin() + 6.0 * (t * 0.31).sin();
        self.metrics.power_w = 285.0 + 4.0 * (t * 0.5).sin();
        self.metrics.vram_used_gb = 21.4 + 0.05 * (t * 0.2).sin();
        self.metrics
            .throughput_series
            .push(self.metrics.aggregate_tps);
        if self.metrics.throughput_series.len() > 60 {
            self.metrics
                .throughput_series
                .drain(0..self.metrics.throughput_series.len() - 60);
        }
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
    fn render_status_bar(&self, area: Rect, f: &mut Frame) {
        let m = &self.metrics;
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
    pub fn tokens(v: Option<usize>) -> String {
        v.map(|v| v.to_string())
            .unwrap_or_else(|| "--".to_string())
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
                s.push(if i == filled - 1 && filled < width { '>' } else { '=' });
            } else {
                s.push(' ');
            }
        }
        s
    }
}
