//! The interactive Setup phase — a **full-screen takeover** shown before the
//! five-view dashboard (plan: "TUI Setup phase").
//!
//! Four stages guide a first-time user from "bare `crucible-llm`" to a
//! running benchmark:
//!
//! 1. **URL** — cursor-based text entry for the target server URL
//!    (works fully offline; the current/default URL is pre-filled when one
//!    was given explicitly).
//! 2. **Discover / Model** — `Enter` fires the async `GET {base}/models`
//!    discovery (the `App`'s lock-free `ResultSlot<Vec<ModelInfo>>` seam);
//!    a "Discovering…" spinner stage yields to a scrollable, type-to-filter
//!    model picker. A failed discovery degrades to free-text model entry
//!    (N/A-never-fail) with `[Tab]` to retry (the spinner stage uses `[d]`).
//! 3. **Config** — the benchmark form (mode, tokens, iterations, ladder,
//!    the six engine switches): `Tab`/`↑↓` move the focus, `Space` toggles,
//!    `←→`/`+-` step numbers, typing edits. Edits write through to the
//!    shared [`ConfigState`] so `F5`/`r` semantics carry over unchanged.
//!    The focused field shows a dimmed `ℹ` explanation below the form
//!    (scalar fields carry their own help text; engine fields reuse
//!    `Engine::description()`).
//! 4. **Confirm** — a summary of every selected setting; `Enter` launches
//!    the benchmark (transition to the Live view + `start_run`), `Esc`
//!    goes back to modify.
//!
//! **Measurement isolation** (blueprint §4): rendering is a pure `&App`
//! read (the model list and discovery status come from lock-free slots);
//! every mutation happens in the key path (`SetupState::handle_key`) or the
//! tick path (the `Discover`→`Model` completion in `App::on_tick`) — never
//! in the render path.

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::client::models::ModelInfo;
use crate::config::EngineSelection;
use crate::ui::app::App;
use crate::ui::theme::{palette, style};
use crate::ui::views::config::ConfigState;

/// The four stages of the setup flow (stage 2 has two sub-states: the
/// in-flight `Discover` spinner and the `Model` picker).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupPhase {
    /// Stage 1 — target URL entry.
    Url,
    /// Stage 2a — model discovery in flight (spinner).
    Discover,
    /// Stage 2b — model selection (discovered list / manual entry).
    Model,
    /// Stage 3 — benchmark configuration form.
    Config,
    /// Stage 4 — summary and launch.
    Confirm,
}

impl SetupPhase {
    /// 0-based step index for the top-bar progress indicator
    /// (`Discover` and `Model` are both step 2).
    pub fn step_index(self) -> usize {
        match self {
            SetupPhase::Url => 0,
            SetupPhase::Discover | SetupPhase::Model => 1,
            SetupPhase::Config => 2,
            SetupPhase::Confirm => 3,
        }
    }
}

/// The outcome of a setup key press (consumed by `App::handle_key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupKeyResult {
    /// The key was consumed by the flow; nothing special to do.
    Inert,
    /// `Esc` at stage 1 — the user asked to quit.
    Quit,
    /// Stage 1 `Enter` (non-empty URL) — start model discovery.
    Discover,
    /// Stage 2 — retry the discovery against the current URL (`d` in the
    /// spinner stage, `Tab` in the picker stage where `d` stays typeable).
    Retry,
    /// Stage 2 `Enter` — a model was selected (see [`SetupState::confirmed_model`]).
    Selected,
    /// Stage 4 `Enter` — launch the benchmark.
    Launched,
}

/// The stage-3 form fields, in cursor order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SetupField {
    Mode,
    Tokens,
    Iterations,
    Ladder,
    EngineSpeed,
    EngineConcurrency,
    EngineNiah,
    EngineReasoning,
    EngineStructured,
    EngineHardware,
}

impl SetupField {
    /// Every field in cursor order.
    pub const ALL: [SetupField; 10] = [
        SetupField::Mode,
        SetupField::Tokens,
        SetupField::Iterations,
        SetupField::Ladder,
        SetupField::EngineSpeed,
        SetupField::EngineConcurrency,
        SetupField::EngineNiah,
        SetupField::EngineReasoning,
        SetupField::EngineStructured,
        SetupField::EngineHardware,
    ];

    /// The short form label (value column carries the detail).
    pub fn label(self) -> &'static str {
        match self {
            SetupField::Mode => "Mode",
            SetupField::Tokens => "Tokens",
            SetupField::Iterations => "Iterations",
            SetupField::Ladder => "Ladder",
            SetupField::EngineSpeed => "Engine A",
            SetupField::EngineConcurrency => "Engine B",
            SetupField::EngineNiah => "Engine C1",
            SetupField::EngineReasoning => "Engine C2",
            SetupField::EngineStructured => "Engine C3",
            SetupField::EngineHardware => "Engine D",
        }
    }

    /// The [`Engine`] this field toggles (`None` for the non-engine
    /// fields) — the config stage shows the focused engine's description
    /// so the user can choose engines knowingly.
    pub fn engine(self) -> Option<crate::engines::sequence::Engine> {
        use crate::engines::sequence::Engine;
        match self {
            SetupField::EngineSpeed => Some(Engine::Speed),
            SetupField::EngineConcurrency => Some(Engine::Concurrency),
            SetupField::EngineNiah => Some(Engine::Niah),
            SetupField::EngineReasoning => Some(Engine::Reasoning),
            SetupField::EngineStructured => Some(Engine::Structured),
            SetupField::EngineHardware => Some(Engine::Hardware),
            _ => None,
        }
    }

    /// The dimmed `ℹ` explanation shown below the field while it has
    /// focus (the scalar fields; the engine fields return `None` and
    /// reuse [`Engine::description`](Self::engine) instead).
    ///
    /// Kept to at most three lines — the config panel wraps to fit.
    pub fn explanation(self) -> Option<&'static str> {
        match self {
            SetupField::Mode => Some(
                "\"short\" = brief prompt (~100 tokens) for quick TTFT measurement.\n\"long\" = extended prompt (your token target) for sustained throughput.\nUse \"short\" to test responsiveness, \"long\" to test sustained generation speed.",
            ),
            SetupField::Tokens => Some(
                "Target number of tokens to generate per request. Higher = longer test,\nmore stable averages. 256 = quick test, 2000 = standard, 8192+ = stress.\nThis is the MAX_tokens sent to the server — actual output may vary.",
            ),
            SetupField::Iterations => Some(
                "How many times to repeat the benchmark. More iterations = more reliable\naverages (reduces variance from scheduling, caching, thermal throttling).\n1 = quick check, 5 = reliable, 10+ = publication-grade.",
            ),
            SetupField::Ladder => Some(
                "Concurrency levels to test, in order. Each level spawns that many\nsimultaneous requests. The sweep finds where per-user speed degrades.\nDefault: 1,2,3,4,8,12,16,24,32 (fine at the low end); custom: 1,4,16,64.",
            ),
            _ => None,
        }
    }
}

/// The interactive setup flow state.
///
/// Owned by [`App`] and mutated only in the key path. Text fields carry an
/// explicit character cursor so `←`/`→`/`Home`/`End`/`Delete` editing works
/// (ratatui has no built-in text input widget; this is the buffer-based
/// pattern from the blueprint's TUI notes).
#[derive(Debug, Clone)]
pub struct SetupState {
    pub phase: SetupPhase,
    /// Stage 1: the target URL text.
    pub url: String,
    /// Stage 1: character cursor inside `url`.
    pub url_cursor: usize,
    /// Stage 2: discovered model ids (materialized from the lock-free
    /// slot by `App::on_tick` when discovery completes).
    pub models: Vec<String>,
    /// Stage 2: cursor within the (filtered) model list.
    pub model_cursor: usize,
    /// Stage 2: typed filter — or the manual model name when no list is
    /// available (the discovery-failure fallback).
    pub model_query: String,
    /// Stage 2: character cursor inside `model_query`.
    pub model_query_cursor: usize,
    /// Stage 2: the last discovery error (`None` on success).
    pub error: Option<String>,
    /// Stage 3: the focused [`SetupField`].
    pub form_field: usize,
}

impl SetupState {
    /// A fresh flow at stage 1 with an empty URL.
    pub fn new() -> Self {
        Self {
            phase: SetupPhase::Url,
            url: String::new(),
            url_cursor: 0,
            models: Vec::new(),
            model_cursor: 0,
            model_query: String::new(),
            model_query_cursor: 0,
            error: None,
            form_field: 0,
        }
    }

    /// Handle one key press, routed to the active stage.
    ///
    /// `cfg` is the shared [`ConfigState`] the stage-3 form edits
    /// (write-through: the values the form shows are the values `F5`/`r`
    /// will run with).
    pub fn handle_key(&mut self, key: &KeyEvent, cfg: &mut ConfigState) -> SetupKeyResult {
        match self.phase {
            SetupPhase::Url => self.handle_url_key(key),
            SetupPhase::Discover => self.handle_discover_key(key),
            SetupPhase::Model => self.handle_model_key(key),
            SetupPhase::Config => self.handle_config_key(key, cfg),
            SetupPhase::Confirm => self.handle_confirm_key(key),
        }
    }

    // ── Stage 1: URL entry ───────────────────────────────────────────────

    fn handle_url_key(&mut self, key: &KeyEvent) -> SetupKeyResult {
        match key.code {
            // Stage 1 is the outermost stage: `Esc` quits the app.
            KeyCode::Esc => SetupKeyResult::Quit,
            KeyCode::Enter => {
                if self.url.trim().is_empty() {
                    self.error = Some(
                        "A server URL is required (e.g. http://localhost:8000/v1)".to_string(),
                    );
                    SetupKeyResult::Inert
                } else {
                    self.url = self.url.trim().to_string();
                    self.url_cursor = self.url.chars().count();
                    self.error = None;
                    self.phase = SetupPhase::Discover;
                    SetupKeyResult::Discover
                }
            }
            KeyCode::Char(c) if c.is_ascii_graphic() => {
                type_at(&mut self.url, &mut self.url_cursor, c);
                SetupKeyResult::Inert
            }
            KeyCode::Backspace => {
                delete_before(&mut self.url, &mut self.url_cursor);
                SetupKeyResult::Inert
            }
            KeyCode::Delete => {
                delete_at(&mut self.url, &mut self.url_cursor);
                SetupKeyResult::Inert
            }
            KeyCode::Left => {
                nudge_cursor(&mut self.url_cursor, self.url.chars().count(), -1);
                SetupKeyResult::Inert
            }
            KeyCode::Right => {
                nudge_cursor(&mut self.url_cursor, self.url.chars().count(), 1);
                SetupKeyResult::Inert
            }
            KeyCode::Home => {
                self.url_cursor = 0;
                SetupKeyResult::Inert
            }
            KeyCode::End => {
                self.url_cursor = self.url.chars().count();
                SetupKeyResult::Inert
            }
            _ => SetupKeyResult::Inert,
        }
    }

    // ── Stage 2a: discovery spinner ─────────────────────────────────────

    fn handle_discover_key(&mut self, key: &KeyEvent) -> SetupKeyResult {
        match key.code {
            KeyCode::Esc => {
                self.phase = SetupPhase::Url;
                SetupKeyResult::Inert
            }
            // `d` — retry (also covers the failed-discovery case; the app
            // re-fires `start_discovery` against the current URL).
            KeyCode::Char('d') => SetupKeyResult::Retry,
            _ => SetupKeyResult::Inert,
        }
    }

    // ── Stage 2b: model picker / manual entry ───────────────────────────

    /// The model ids matching the current filter (all of them when the
    /// filter is empty).
    pub fn filtered_models(&self) -> Vec<String> {
        let q = self.model_query.trim().to_ascii_lowercase();
        if q.is_empty() {
            self.models.clone()
        } else {
            self.models
                .iter()
                .filter(|m| m.to_ascii_lowercase().contains(&q))
                .cloned()
                .collect()
        }
    }

    /// The model `Enter` confirms in this stage: the filtered cursor item,
    /// or the typed name verbatim when nothing matches (manual entry).
    /// `None` when there is nothing to select.
    pub fn confirmed_model(&self) -> Option<String> {
        let filtered = self.filtered_models();
        if !filtered.is_empty() {
            Some(filtered[self.model_cursor.min(filtered.len() - 1)].clone())
        } else if !self.model_query.trim().is_empty() {
            Some(self.model_query.trim().to_string())
        } else {
            None
        }
    }

    fn handle_model_key(&mut self, key: &KeyEvent) -> SetupKeyResult {
        match key.code {
            KeyCode::Esc => {
                self.phase = SetupPhase::Url;
                SetupKeyResult::Inert
            }
            // `Tab` retries the discovery. It is a non-printable key on
            // purpose: `d` must stay typeable (model names like
            // "deepseek-…" contain `d`) — the spinner stage keeps `d`.
            KeyCode::Tab => SetupKeyResult::Retry,
            KeyCode::Enter => {
                if self.confirmed_model().is_some() {
                    self.phase = SetupPhase::Config;
                    SetupKeyResult::Selected
                } else {
                    SetupKeyResult::Inert
                }
            }
            // List navigation is arrow-only: every printable char (j, k, …)
            // goes into the filter / manual name.
            KeyCode::Down => {
                let n = self.filtered_models().len();
                if n > 0 {
                    self.model_cursor = (self.model_cursor + 1) % n;
                }
                SetupKeyResult::Inert
            }
            KeyCode::Up => {
                let n = self.filtered_models().len();
                if n > 0 {
                    self.model_cursor = (self.model_cursor + n - 1) % n;
                }
                SetupKeyResult::Inert
            }
            KeyCode::Char(c) if c.is_ascii_graphic() => {
                type_at(&mut self.model_query, &mut self.model_query_cursor, c);
                self.clamp_model_cursor();
                SetupKeyResult::Inert
            }
            KeyCode::Backspace => {
                delete_before(&mut self.model_query, &mut self.model_query_cursor);
                self.clamp_model_cursor();
                SetupKeyResult::Inert
            }
            KeyCode::Delete => {
                delete_at(&mut self.model_query, &mut self.model_query_cursor);
                self.clamp_model_cursor();
                SetupKeyResult::Inert
            }
            KeyCode::Left => {
                nudge_cursor(
                    &mut self.model_query_cursor,
                    self.model_query.chars().count(),
                    -1,
                );
                SetupKeyResult::Inert
            }
            KeyCode::Right => {
                nudge_cursor(
                    &mut self.model_query_cursor,
                    self.model_query.chars().count(),
                    1,
                );
                SetupKeyResult::Inert
            }
            KeyCode::Home => {
                self.model_query_cursor = 0;
                SetupKeyResult::Inert
            }
            KeyCode::End => {
                self.model_query_cursor = self.model_query.chars().count();
                SetupKeyResult::Inert
            }
            _ => SetupKeyResult::Inert,
        }
    }

    /// Keep the list cursor valid after the filter changes.
    fn clamp_model_cursor(&mut self) {
        let n = self.filtered_models().len();
        if n == 0 {
            self.model_cursor = 0;
        } else {
            self.model_cursor = self.model_cursor.min(n - 1);
        }
    }

    /// The discovery finished (called from `App::on_tick` — the tick path,
    /// never the render path): materialize the list and move to the model
    /// stage. A failure (or an empty server list) degrades to manual entry.
    pub fn complete_discovery(&mut self, models: Option<Vec<ModelInfo>>, error: Option<String>) {
        self.models = models
            .map(|v| v.into_iter().map(|m| m.id).collect())
            .unwrap_or_default();
        self.model_cursor = 0;
        self.model_query.clear();
        self.model_query_cursor = 0;
        self.error = if self.models.is_empty() {
            error.or_else(|| Some("No models found on this server".to_string()))
        } else {
            None
        };
        self.phase = SetupPhase::Model;
    }

    // ── Stage 3: benchmark configuration form ───────────────────────────

    fn current_field(&self) -> SetupField {
        SetupField::ALL[self.form_field.min(SetupField::ALL.len() - 1)]
    }

    fn handle_config_key(&mut self, key: &KeyEvent, cfg: &mut ConfigState) -> SetupKeyResult {
        match key.code {
            KeyCode::Esc => {
                self.phase = SetupPhase::Model;
                SetupKeyResult::Inert
            }
            // `Enter` advances to the summary (use `Space` to toggle).
            KeyCode::Enter => {
                self.phase = SetupPhase::Confirm;
                SetupKeyResult::Inert
            }
            KeyCode::Tab => {
                self.form_field = (self.form_field + 1) % SetupField::ALL.len();
                SetupKeyResult::Inert
            }
            KeyCode::BackTab => {
                self.form_field =
                    (self.form_field + SetupField::ALL.len() - 1) % SetupField::ALL.len();
                SetupKeyResult::Inert
            }
            KeyCode::Down => {
                self.form_field = (self.form_field + 1).min(SetupField::ALL.len() - 1);
                SetupKeyResult::Inert
            }
            KeyCode::Up => {
                self.form_field = self.form_field.saturating_sub(1);
                SetupKeyResult::Inert
            }
            // `Space` toggles booleans / cycles the mode (matches View 5).
            KeyCode::Char(' ') => {
                self.form_toggle(cfg);
                SetupKeyResult::Inert
            }
            KeyCode::Left | KeyCode::Char('-') => {
                self.form_step(cfg, -1);
                SetupKeyResult::Inert
            }
            KeyCode::Right | KeyCode::Char('+') => {
                self.form_step(cfg, 1);
                SetupKeyResult::Inert
            }
            KeyCode::Backspace => {
                self.form_backspace(cfg);
                SetupKeyResult::Inert
            }
            // Typing edits the focused text/number field.
            KeyCode::Char(c) if c.is_ascii_graphic() => {
                self.form_type(cfg, c);
                SetupKeyResult::Inert
            }
            _ => SetupKeyResult::Inert,
        }
    }

    /// `Space` on the focused field: cycle the mode, flip an engine switch.
    fn form_toggle(&mut self, cfg: &mut ConfigState) {
        match self.current_field() {
            SetupField::Mode => {
                cfg.mode = match cfg.mode {
                    crate::config::Mode::Short => crate::config::Mode::Long,
                    crate::config::Mode::Long => crate::config::Mode::Short,
                };
            }
            SetupField::EngineSpeed => cfg.engine_speed = !cfg.engine_speed,
            SetupField::EngineConcurrency => cfg.engine_concurrency = !cfg.engine_concurrency,
            SetupField::EngineNiah => cfg.engine_niah = !cfg.engine_niah,
            SetupField::EngineReasoning => cfg.engine_reasoning = !cfg.engine_reasoning,
            SetupField::EngineStructured => cfg.engine_structured = !cfg.engine_structured,
            SetupField::EngineHardware => cfg.hardware = !cfg.hardware,
            _ => {}
        }
    }

    /// `←`/`→` (or `-`/`+`) on the focused field: step a number / cycle.
    fn form_step(&mut self, cfg: &mut ConfigState, dir: i32) {
        match self.current_field() {
            SetupField::Mode => {
                cfg.mode = match (cfg.mode, dir) {
                    (crate::config::Mode::Short, 1) => crate::config::Mode::Long,
                    (crate::config::Mode::Long, -1) => crate::config::Mode::Short,
                    (m, _) => m,
                };
            }
            SetupField::Tokens => {
                let step = 100u32;
                cfg.tokens = if dir > 0 {
                    cfg.tokens.saturating_add(step)
                } else {
                    cfg.tokens.saturating_sub(step)
                };
            }
            SetupField::Iterations => {
                cfg.iterations = if dir > 0 {
                    cfg.iterations.saturating_add(1)
                } else {
                    cfg.iterations.saturating_sub(1)
                };
            }
            _ => {}
        }
    }

    /// Type a character into the focused field (digits for numbers, any
    /// graphic ASCII for the ladder).
    fn form_type(&mut self, cfg: &mut ConfigState, c: char) {
        match self.current_field() {
            SetupField::Ladder => cfg.ladder.push(c),
            SetupField::Tokens => {
                if let Some(d) = c.to_digit(10) {
                    cfg.tokens = cfg.tokens.saturating_mul(10).saturating_add(d);
                }
            }
            SetupField::Iterations => {
                if let Some(d) = c.to_digit(10) {
                    cfg.iterations = cfg.iterations.saturating_mul(10).saturating_add(d);
                }
            }
            _ => {}
        }
    }

    /// Backspace on the focused field.
    fn form_backspace(&mut self, cfg: &mut ConfigState) {
        match self.current_field() {
            SetupField::Ladder => {
                cfg.ladder.pop();
            }
            SetupField::Tokens => {
                cfg.tokens /= 10;
            }
            SetupField::Iterations => {
                cfg.iterations /= 10;
            }
            _ => {}
        }
    }

    // ── Stage 4: confirm & launch ────────────────────────────────────────

    fn handle_confirm_key(&mut self, key: &KeyEvent) -> SetupKeyResult {
        match key.code {
            KeyCode::Enter => SetupKeyResult::Launched,
            KeyCode::Esc => {
                self.phase = SetupPhase::Config;
                SetupKeyResult::Inert
            }
            _ => SetupKeyResult::Inert,
        }
    }
}

impl Default for SetupState {
    fn default() -> Self {
        Self::new()
    }
}

// ── Buffer-based text editing helpers (char-index cursors) ────────────────

/// Insert `c` at the cursor.
fn type_at(s: &mut String, cursor: &mut usize, c: char) {
    let i = (*cursor).min(s.chars().count());
    s.insert(i, c);
    *cursor = i + 1;
}

/// Delete the character before the cursor (`Backspace`).
fn delete_before(s: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    *cursor -= 1;
    s.remove(*cursor);
}

/// Delete the character at the cursor (`Delete`).
fn delete_at(s: &mut String, cursor: &mut usize) {
    if *cursor >= s.chars().count() {
        return;
    }
    s.remove(*cursor);
}

/// Move the cursor by `dir` within `[0, len]`.
fn nudge_cursor(cursor: &mut usize, len: usize, dir: i32) {
    *cursor = (*cursor as i32 + dir).clamp(0, len as i32) as usize;
}

// ── Rendering (pure `&App` reads) ─────────────────────────────────────────

/// The top-bar step labels, in order.
const STEP_LABELS: [&str; 4] = ["1 URL", "2 MODEL", "3 CONFIG", "4 LAUNCH"];

/// Spinner frames for the discovery stage (advanced by the 60Hz tick).
const SPINNER: [&str; 4] = ["|", "/", "−", "\\"];

/// Render the full-screen setup takeover into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let s = &app.setup;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // top bar (title + step indicator)
            Constraint::Min(3),    // body panel
            Constraint::Length(1), // key-hint footer
        ])
        .split(area);

    render_top_bar(chunks[0], s, f);

    // Center the body panel (50% width) so long URLs don't stretch the box.
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(50),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);
    let body = mid[1];
    match s.phase {
        SetupPhase::Url => render_url(body, s, f),
        SetupPhase::Discover => render_discover(body, s, app, f),
        SetupPhase::Model => render_model(body, s, f),
        SetupPhase::Config => render_config(body, s, app, f),
        SetupPhase::Confirm => render_confirm(body, s, app, f),
    }

    render_footer(chunks[2], s, f);
}

/// Top bar: brand + the four-step progress indicator.
fn render_top_bar(area: Rect, s: &SetupState, f: &mut Frame) {
    let step = s.phase.step_index();
    let mut spans = vec![Span::styled(
        " CRUCIBLE-LLM — SETUP ",
        Style::default()
            .fg(palette::HIGHLIGHT)
            .add_modifier(Modifier::BOLD),
    )];
    for (i, label) in STEP_LABELS.iter().enumerate() {
        let st = if i == step {
            style::highlight()
        } else if i < step {
            style::value_ok()
        } else {
            style::tab_inactive()
        };
        spans.push(Span::styled(format!("{label:<12}"), st));
        if i + 1 < STEP_LABELS.len() {
            spans.push(Span::styled(" ▸ ", style::tab_separator()));
        }
    }
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The panel border used by every stage.
fn panel(title: impl Into<Line<'static>>) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_style(style::active_border())
        .title(title)
}

/// Stage 1: the URL prompt with a visible text cursor.
fn render_url(area: Rect, s: &SetupState, f: &mut Frame) {
    let before: String = s.url.chars().take(s.url_cursor).collect();
    let after: String = s.url.chars().skip(s.url_cursor).collect();
    let mut lines = vec![
        Line::from(Span::styled("Enter server URL", style::title())),
        Line::from(Span::styled(
            "(e.g. http://localhost:8000/v1)",
            style::footer(),
        )),
        Line::raw(""),
        Line::from(vec![
            Span::styled("> ", style::highlight()),
            Span::styled(before, style::value()),
            Span::styled("█", Style::default().fg(palette::ACCENT)),
            Span::styled(after, style::value()),
        ]),
        Line::raw(""),
    ];
    if let Some(e) = &s.error {
        lines.push(Line::from(Span::styled(e, style::value_err())));
        lines.push(Line::raw(""));
    }
    lines.push(Line::from(Span::styled(
        "Press Enter to discover the models served here.",
        style::footer(),
    )));
    f.render_widget(
        Paragraph::new(Text::from(lines)).block(panel("SETUP — CONNECTION")),
        area,
    );
}

/// Stage 2a: the discovery spinner.
fn render_discover(area: Rect, s: &SetupState, app: &App, f: &mut Frame) {
    let frame = (app.tick / 6) as usize % SPINNER.len();
    let lines = vec![
        Line::from(Span::styled(
            format!("{} Discovering models…", SPINNER[frame]),
            style::highlight(),
        )),
        Line::raw(""),
        Line::from(vec![
            Span::styled("Target:  ", style::label()),
            Span::styled(s.url.clone(), style::value()),
        ]),
        Line::from(Span::styled(
            "GET {base}/models (OpenAI-compatible)",
            style::footer(),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "[d] retry now   ·   [Esc] back to URL",
            style::footer(),
        )),
    ];
    f.render_widget(
        Paragraph::new(Text::from(lines)).block(panel("SETUP — MODEL DISCOVERY")),
        area,
    );
}

/// Stage 2b: the model picker (scrollable + filterable) — or the free-text
/// model entry when discovery produced nothing.
fn render_model(area: Rect, s: &SetupState, f: &mut Frame) {
    let filtered = s.filtered_models();
    let has_list = !s.models.is_empty();

    let title = if has_list {
        format!("SETUP — SELECT MODEL ({} found)", s.models.len())
    } else {
        "SETUP — ENTER MODEL NAME".to_string()
    };
    let title: Line<'static> = Line::from(Span::styled(title, style::title()));

    let mut lines: Vec<Line> = Vec::new();
    if !has_list {
        if let Some(e) = &s.error {
            lines.push(Line::from(Span::styled(
                format!("Discovery failed: {e}"),
                style::value_err(),
            )));
            lines.push(Line::from(Span::styled(
                "Type the model name to use, or press [d] to retry.",
                style::footer(),
            )));
            lines.push(Line::raw(""));
        }
        let before: String = s.model_query.chars().take(s.model_query_cursor).collect();
        let after: String = s.model_query.chars().skip(s.model_query_cursor).collect();
        lines.push(Line::from(vec![
            Span::styled("Model: ", style::label()),
            Span::styled(before, style::value()),
            Span::styled("█", Style::default().fg(palette::ACCENT)),
            Span::styled(after, style::value()),
        ]));
    } else {
        lines.push(Line::from(Span::styled(
            "Available models:",
            style::muted_title(),
        )));
        // Scroll window keeping the cursor in view.
        let height = area.height.saturating_sub(2) as usize;
        let visible = height.saturating_sub(4).max(1);
        let start = (s.model_cursor + 1)
            .saturating_sub(visible / 2)
            .min(filtered.len().saturating_sub(1));
        for (i, m) in filtered.iter().enumerate().skip(start).take(visible) {
            let st = if i == s.model_cursor {
                style::highlight()
            } else {
                style::value()
            };
            lines.push(Line::from(vec![
                Span::styled(if i == s.model_cursor { "> " } else { "  " }, st),
                Span::styled(m.clone(), st),
            ]));
        }
        if filtered.is_empty() {
            lines.push(Line::from(Span::styled(
                "(no match — Enter uses the typed name)",
                style::value_warn(),
            )));
        }
        if !s.model_query.is_empty() {
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![
                Span::styled("filter: ", style::footer()),
                Span::styled(s.model_query.clone(), style::value_warn()),
            ]));
        }
    }
    f.render_widget(Paragraph::new(Text::from(lines)).block(panel(title)), area);
}

/// Stage 3: the benchmark configuration form (write-through to
/// `ConfigState`; the target URL/model are shown as read-only context).
fn render_config(area: Rect, s: &SetupState, app: &App, f: &mut Frame) {
    let c = &app.config;
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("Target   ", style::footer()),
            Span::styled(c.url.clone(), style::value()),
        ]),
        Line::from(vec![
            Span::styled("Model    ", style::footer()),
            Span::styled(c.model.clone(), style::value_ok()),
        ]),
        Line::raw(""),
    ];
    for &field in &SetupField::ALL {
        let is_cursor = field == s.current_field();
        let (value, vstyle) = form_value(field, c);
        let prefix = if is_cursor {
            Span::styled("> ", style::highlight())
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            prefix,
            Span::styled(format!("{:<11}", field.label()), style::label()),
            Span::styled(value, vstyle),
        ]));
    }
    // The focused field's explanation (dimmed `ℹ` note): the engine
    // fields show `Engine::description()` (what a benchmark measures),
    // the scalar fields show their own help text — every field explains
    // itself as the user tabs through the form.
    let field = s.current_field();
    if let Some(engine) = field.engine() {
        lines.push(Line::raw(""));
        lines.extend(crate::ui::views::engine_info_lines(engine));
    } else if let Some(text) = field.explanation() {
        lines.push(Line::raw(""));
        lines.extend(crate::ui::views::info_lines(text));
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(panel("SETUP — BENCHMARK CONFIGURATION"))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// `(value, value_style)` for one stage-3 form field.
fn form_value(field: SetupField, c: &ConfigState) -> (String, Style) {
    match field {
        SetupField::Mode => (c.mode.label().to_string(), style::value()),
        SetupField::Tokens => (c.tokens.to_string(), style::value()),
        SetupField::Iterations => (c.iterations.to_string(), style::value()),
        SetupField::Ladder => (c.ladder.clone(), style::value()),
        SetupField::EngineSpeed => (
            format!("[{}] Speed", tick(c.engine_speed)),
            bool_style(c.engine_speed),
        ),
        SetupField::EngineConcurrency => (
            format!("[{}] Concurrency", tick(c.engine_concurrency)),
            bool_style(c.engine_concurrency),
        ),
        SetupField::EngineNiah => (
            format!("[{}] NIAH", tick(c.engine_niah)),
            bool_style(c.engine_niah),
        ),
        SetupField::EngineReasoning => (
            format!("[{}] Reasoning", tick(c.engine_reasoning)),
            bool_style(c.engine_reasoning),
        ),
        SetupField::EngineStructured => (
            format!("[{}] Structured", tick(c.engine_structured)),
            bool_style(c.engine_structured),
        ),
        SetupField::EngineHardware => (
            format!("[{}] Hardware / Energy", tick(c.hardware)),
            bool_style(c.hardware),
        ),
    }
}

fn tick(b: bool) -> &'static str {
    if b {
        "x"
    } else {
        " "
    }
}

fn bool_style(b: bool) -> Style {
    if b {
        style::value_ok()
    } else {
        Style::default().fg(palette::MUTED)
    }
}

/// Stage 4: the launch summary.
fn render_confirm(area: Rect, s: &SetupState, app: &App, f: &mut Frame) {
    let c = &app.config;
    let engines = EngineSelection {
        speed: c.engine_speed,
        concurrency: c.engine_concurrency,
        niah: c.engine_niah,
        reasoning: c.engine_reasoning,
        structured: c.engine_structured,
        hardware: c.hardware,
    };
    let engine_labels: String = engines
        .iter_labels()
        .map(|l| l.to_string())
        .collect::<Vec<_>>()
        .join(" · ");
    let lines = vec![
        Line::from(Span::styled("Benchmark summary", style::title())),
        Line::raw(""),
        summary_line("Target URL", &s.url, style::value()),
        summary_line("Model", &c.model, style::value_ok()),
        summary_line("Mode", c.mode.label(), style::value()),
        summary_line("Tokens", &c.tokens.to_string(), style::value()),
        summary_line("Iterations", &c.iterations.to_string(), style::value()),
        summary_line("Ladder", &c.ladder, style::value()),
        summary_line("Engines", &engine_labels, style::highlight()),
        Line::raw(""),
        Line::from(Span::styled(
            "Press Enter to start the benchmark.",
            style::value_ok(),
        )),
        Line::from(Span::styled(
            "Press Esc to go back and modify.",
            style::footer(),
        )),
    ];
    f.render_widget(
        Paragraph::new(Text::from(lines)).block(panel("SETUP — CONFIRM & LAUNCH")),
        area,
    );
}

fn summary_line(label: &str, value: &str, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<12}", label), style::label()),
        Span::styled(value.to_string(), value_style),
    ])
}

/// Bottom key-hint footer (per stage).
fn render_footer(area: Rect, s: &SetupState, f: &mut Frame) {
    let hint = match s.phase {
        SetupPhase::Url => {
            "[type] URL  ·  [←→/Home/End] cursor  ·  [⌫/Del] delete  ·  [Enter] discover models  ·  [Esc] quit"
        }
        SetupPhase::Discover => "[d] retry  ·  [Esc] back to URL",
        SetupPhase::Model => {
            "[↑↓] select  ·  [type] filter  ·  [Enter] choose  ·  [Tab] retry discovery  ·  [Esc] back"
        }
        SetupPhase::Config => {
            "[Tab/↑↓] move  ·  [Space] toggle  ·  [←→/+/-] step  ·  [type] edit  ·  [Enter] next  ·  [Esc] back"
        }
        SetupPhase::Confirm => "[Enter] start benchmark  ·  [Esc] modify",
    };
    f.render_widget(
        Paragraph::new(Line::from(Span::styled(hint, style::footer()))),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Mode;
    use crossterm::event::{KeyEventKind, KeyEventState};

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent {
            code,
            modifiers: crossterm::event::KeyModifiers::NONE,
            kind: KeyEventKind::Press,
            state: KeyEventState::NONE,
        }
    }

    fn char_key(c: char) -> KeyEvent {
        key(KeyCode::Char(c))
    }

    fn cfg() -> ConfigState {
        ConfigState::default()
    }

    fn setup_with_models() -> SetupState {
        let mut s = SetupState::new();
        s.complete_discovery(
            Some(vec![
                ModelInfo {
                    id: "llama-3-70b".into(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "mistral-7b".into(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "qwen3-72b".into(),
                    ..Default::default()
                },
            ]),
            None,
        );
        s
    }

    // ── text editing helpers ─────────────────────────────────────────────

    #[test]
    fn type_at_inserts_at_cursor() {
        let mut s = String::from("ac");
        let mut c = 1usize;
        type_at(&mut s, &mut c, 'b');
        assert_eq!(s, "abc");
        assert_eq!(c, 2);
    }

    #[test]
    fn delete_before_and_delete_at() {
        let mut s = String::from("abc");
        let mut c = 2usize;
        delete_before(&mut s, &mut c);
        assert_eq!(s, "ac");
        assert_eq!(c, 1);
        delete_at(&mut s, &mut c);
        assert_eq!(s, "a");
        assert_eq!(c, 1);
        // At the end: delete_at is a no-op; backspace at 0 is a no-op.
        delete_at(&mut s, &mut c);
        assert_eq!(s, "a");
        let mut c0 = 0usize;
        delete_before(&mut s, &mut c0);
        assert_eq!(s, "a");
    }

    // ── stage 1: URL entry ───────────────────────────────────────────────

    #[test]
    fn url_typing_backspace_and_cursor_edit() {
        let mut s = SetupState::new();
        for c in "abc".chars() {
            assert_eq!(
                s.handle_key(&char_key(c), &mut cfg()),
                SetupKeyResult::Inert
            );
        }
        assert_eq!(s.url, "abc");
        assert_eq!(s.url_cursor, 3);
        // Left, insert in the middle, backspace the middle char.
        assert_eq!(
            s.handle_key(&key(KeyCode::Left), &mut cfg()),
            SetupKeyResult::Inert
        );
        assert_eq!(s.url_cursor, 2);
        s.handle_key(&char_key('X'), &mut cfg());
        assert_eq!(s.url, "abXc");
        s.handle_key(&key(KeyCode::Backspace), &mut cfg());
        assert_eq!(s.url, "abc");
        assert_eq!(s.url_cursor, 2);
    }

    #[test]
    fn url_enter_with_text_advances_to_discover() {
        let mut s = SetupState::new();
        s.url = "  http://localhost:8000/v1  ".into();
        let r = s.handle_key(&key(KeyCode::Enter), &mut cfg());
        assert_eq!(r, SetupKeyResult::Discover);
        assert_eq!(s.phase, SetupPhase::Discover);
        // The URL is trimmed on the way out.
        assert_eq!(s.url, "http://localhost:8000/v1");
    }

    #[test]
    fn url_enter_empty_stays_and_sets_error() {
        let mut s = SetupState::new();
        let r = s.handle_key(&key(KeyCode::Enter), &mut cfg());
        assert_eq!(r, SetupKeyResult::Inert);
        assert_eq!(s.phase, SetupPhase::Url);
        assert!(s.error.is_some());
    }

    #[test]
    fn url_esc_quits() {
        let mut s = SetupState::new();
        assert_eq!(
            s.handle_key(&key(KeyCode::Esc), &mut cfg()),
            SetupKeyResult::Quit
        );
    }

    #[test]
    fn q_is_a_plain_character_in_url_phase() {
        let mut s = SetupState::new();
        let r = s.handle_key(&char_key('q'), &mut cfg());
        assert_eq!(r, SetupKeyResult::Inert, "q must not quit in setup");
        assert_eq!(s.url, "q");
    }

    // ── stage 2a: discovery spinner ─────────────────────────────────────

    #[test]
    fn discover_esc_returns_to_url() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Discover;
        let r = s.handle_key(&key(KeyCode::Esc), &mut cfg());
        assert_eq!(r, SetupKeyResult::Inert);
        assert_eq!(s.phase, SetupPhase::Url);
    }

    #[test]
    fn discover_d_retries() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Discover;
        assert_eq!(
            s.handle_key(&char_key('d'), &mut cfg()),
            SetupKeyResult::Retry
        );
    }

    // ── stage 2 completion (the on_tick hand-off) ────────────────────────

    #[test]
    fn complete_discovery_success_populates_the_picker() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Discover;
        // The slot list is already id-sorted by `clean_models`; the
        // materialization preserves that order.
        s.complete_discovery(
            Some(vec![
                ModelInfo {
                    id: "llama-3-70b".into(),
                    ..Default::default()
                },
                ModelInfo {
                    id: "qwen3-72b".into(),
                    ..Default::default()
                },
            ]),
            None,
        );
        assert_eq!(s.phase, SetupPhase::Model);
        assert_eq!(s.models, vec!["llama-3-70b", "qwen3-72b"]);
        assert!(s.error.is_none());
    }

    #[test]
    fn complete_discovery_failure_degrades_to_manual_entry() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Discover;
        s.complete_discovery(None, Some("connection failed: refused".into()));
        assert_eq!(s.phase, SetupPhase::Model);
        assert!(s.models.is_empty());
        assert_eq!(s.error.as_deref(), Some("connection failed: refused"));
    }

    #[test]
    fn complete_discovery_empty_list_gets_a_generic_error() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Discover;
        s.complete_discovery(Some(Vec::new()), None);
        assert_eq!(s.phase, SetupPhase::Model);
        assert!(s.models.is_empty());
        assert!(s.error.is_some());
    }

    // ── stage 2b: model picker ───────────────────────────────────────────

    #[test]
    fn model_navigation_wraps() {
        let mut s = setup_with_models();
        s.handle_key(&key(KeyCode::Down), &mut cfg());
        assert_eq!(s.model_cursor, 1);
        s.handle_key(&key(KeyCode::Up), &mut cfg());
        assert_eq!(s.model_cursor, 0);
        s.handle_key(&key(KeyCode::Up), &mut cfg());
        assert_eq!(s.model_cursor, 2, "up wraps to the last entry");
    }

    #[test]
    fn model_filter_narrows_the_list() {
        let mut s = setup_with_models();
        for c in "qw".chars() {
            s.handle_key(&char_key(c), &mut cfg());
        }
        assert_eq!(s.filtered_models(), vec!["qwen3-72b"]);
        assert_eq!(s.model_cursor, 0);
    }

    #[test]
    fn model_enter_selects_the_cursor_item() {
        let mut s = setup_with_models();
        s.model_cursor = 2;
        let r = s.handle_key(&key(KeyCode::Enter), &mut cfg());
        assert_eq!(r, SetupKeyResult::Selected);
        assert_eq!(s.phase, SetupPhase::Config);
        assert_eq!(s.confirmed_model().as_deref(), Some("qwen3-72b"));
    }

    #[test]
    fn model_enter_uses_typed_name_when_nothing_matches() {
        let mut s = setup_with_models();
        for c in "my-custom-model".chars() {
            s.handle_key(&char_key(c), &mut cfg());
        }
        assert!(s.filtered_models().is_empty());
        let r = s.handle_key(&key(KeyCode::Enter), &mut cfg());
        assert_eq!(r, SetupKeyResult::Selected);
        assert_eq!(s.confirmed_model().as_deref(), Some("my-custom-model"));
        // The typed name survived trim on confirmation.
        assert_eq!(s.model_query, "my-custom-model");
    }

    #[test]
    fn model_enter_with_nothing_available_is_inert() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Model; // no list, empty query
        let r = s.handle_key(&key(KeyCode::Enter), &mut cfg());
        assert_eq!(r, SetupKeyResult::Inert);
        assert_eq!(s.phase, SetupPhase::Model);
    }

    #[test]
    fn model_tab_retries_discovery_and_d_stays_typeable() {
        let mut s = setup_with_models();
        assert_eq!(
            s.handle_key(&key(KeyCode::Tab), &mut cfg()),
            SetupKeyResult::Retry
        );
        // `d` is a plain character in this stage (no retry interception).
        s.handle_key(&char_key('d'), &mut cfg());
        assert_eq!(s.model_query, "d");
    }

    #[test]
    fn manual_model_entry_types_and_backspaces() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Model;
        s.error = Some("connection failed: refused".into());
        for c in "ab".chars() {
            s.handle_key(&char_key(c), &mut cfg());
        }
        s.handle_key(&key(KeyCode::Backspace), &mut cfg());
        assert_eq!(s.model_query, "a");
        assert_eq!(s.confirmed_model().as_deref(), Some("a"));
    }

    // ── stage 3: config form ─────────────────────────────────────────────

    #[test]
    fn form_tab_and_backtab_wrap() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        s.handle_key(&key(KeyCode::Tab), &mut c);
        assert_eq!(s.form_field, 1);
        s.handle_key(&key(KeyCode::BackTab), &mut c);
        assert_eq!(s.form_field, 0);
        for _ in 0..SetupField::ALL.len() {
            s.handle_key(&key(KeyCode::Tab), &mut c);
        }
        assert_eq!(s.form_field, 0, "full lap wraps to the start");
    }

    #[test]
    fn form_space_toggles_engine_and_cycles_mode() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        // Focus Engine D (index 9) — off by default (FIX 4: everything
        // except Energy is selected), so the toggle is observable.
        s.form_field = 9;
        assert!(!c.hardware);
        s.handle_key(&char_key(' '), &mut c);
        assert!(c.hardware);
        s.handle_key(&char_key(' '), &mut c);
        assert!(!c.hardware);
        // Focus Mode (index 0): short → long.
        s.form_field = 0;
        assert_eq!(c.mode, Mode::Short);
        s.handle_key(&char_key(' '), &mut c);
        assert_eq!(c.mode, Mode::Long);
    }

    #[test]
    fn form_digits_and_backspace_edit_numbers() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        s.form_field = 1; // Tokens (zero the default first).
        c.tokens = 0;
        for ch in "37".chars() {
            s.handle_key(&char_key(ch), &mut c);
        }
        assert_eq!(c.tokens, 37);
        s.handle_key(&key(KeyCode::Backspace), &mut c);
        assert_eq!(c.tokens, 3);
        // Non-digits are ignored in number fields.
        s.handle_key(&char_key('x'), &mut c);
        assert_eq!(c.tokens, 3);
    }

    #[test]
    fn form_step_moves_numbers() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        // Tokens: ±100 per step.
        s.form_field = 1;
        let before = c.tokens;
        s.handle_key(&key(KeyCode::Right), &mut c);
        assert_eq!(c.tokens, before + 100);
        s.handle_key(&key(KeyCode::Left), &mut c);
        assert_eq!(c.tokens, before);
        // Iterations: ±1 per step.
        s.form_field = 2;
        let before_i = c.iterations;
        s.handle_key(&key(KeyCode::Right), &mut c);
        assert_eq!(c.iterations, before_i + 1);
        s.handle_key(&key(KeyCode::Left), &mut c);
        assert_eq!(c.iterations, before_i);
    }

    #[test]
    fn form_ladder_takes_any_printable() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        c.ladder.clear();
        s.form_field = 3; // Ladder
        for ch in "1,4,16".chars() {
            s.handle_key(&char_key(ch), &mut c);
        }
        assert_eq!(c.ladder, "1,4,16");
        s.handle_key(&key(KeyCode::Backspace), &mut c);
        assert_eq!(c.ladder, "1,4,1");
    }

    #[test]
    fn form_enter_advances_to_confirm_and_esc_returns() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Config;
        let mut c = cfg();
        s.handle_key(&key(KeyCode::Enter), &mut c);
        assert_eq!(s.phase, SetupPhase::Confirm);
        s.handle_key(&key(KeyCode::Esc), &mut c);
        assert_eq!(s.phase, SetupPhase::Config);
    }

    // ── stage 4: confirm & launch ────────────────────────────────────────

    #[test]
    fn confirm_enter_launches() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Confirm;
        assert_eq!(
            s.handle_key(&key(KeyCode::Enter), &mut cfg()),
            SetupKeyResult::Launched
        );
    }

    #[test]
    fn confirm_esc_returns_to_config() {
        let mut s = SetupState::new();
        s.phase = SetupPhase::Confirm;
        let r = s.handle_key(&key(KeyCode::Esc), &mut cfg());
        assert_eq!(r, SetupKeyResult::Inert);
        assert_eq!(s.phase, SetupPhase::Config);
    }

    // ── stage 3: rendering the focused engine's description ──────────────

    fn render_setup_text(app: &crate::ui::app::App, w: u16, h: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(w, h);
        let mut terminal = ratatui::Terminal::new(backend).expect("TestBackend terminal");
        terminal
            .draw(|f| render(f.area(), app, f))
            .expect("render frame");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    fn setup_app_at_config(field: usize) -> crate::ui::app::App {
        let mut app = crate::ui::app::App::new();
        app.phase = crate::ui::app::Phase::Setup;
        app.setup.phase = SetupPhase::Config;
        app.setup.form_field = field;
        app
    }

    #[test]
    fn config_stage_shows_the_focused_engine_description() {
        // form_field 4 = Engine A (Speed).
        let app = setup_app_at_config(4);
        let text = render_setup_text(&app, 100, 30);
        assert!(text.contains("SETUP — BENCHMARK CONFIGURATION"), "{text}");
        assert!(
            text.contains('ℹ'),
            "focused engine gets an info note: {text}"
        );
        assert!(text.contains("Single-stream throughput"), "{text}");
    }

    #[test]
    fn config_stage_tracks_the_focus_across_engines() {
        // form_field 9 = Engine D (Hardware).
        let app = setup_app_at_config(9);
        let text = render_setup_text(&app, 120, 30);
        assert!(text.contains("GPU power profiling"), "{text}");
        assert!(text.contains("MUST run"), "{text}");
        assert!(text.contains("NVML"), "{text}");
        assert!(text.contains("Remote users"), "{text}");
    }

    // ── stage 3: rendering the focused scalar field's explanation ──────

    #[test]
    fn config_stage_shows_the_focused_mode_explanation() {
        // form_field 0 = Mode.
        let app = setup_app_at_config(0);
        let text = render_setup_text(&app, 120, 30);
        assert!(
            text.contains('ℹ'),
            "focused scalar field gets an info note: {text}"
        );
        assert!(text.contains("brief prompt"), "{text}");
        assert!(text.contains("TTFT"), "{text}");
        assert!(text.contains("sustained generation speed"), "{text}");
    }

    #[test]
    fn config_stage_shows_the_focused_tokens_explanation() {
        // form_field 1 = Tokens.
        let app = setup_app_at_config(1);
        let text = render_setup_text(&app, 120, 30);
        assert!(text.contains('ℹ'), "{text}");
        assert!(text.contains("more stable averages"), "{text}");
        assert!(text.contains("MAX_tokens"), "{text}");
    }

    #[test]
    fn config_stage_shows_the_focused_iterations_explanation() {
        // form_field 2 = Iterations.
        let app = setup_app_at_config(2);
        let text = render_setup_text(&app, 120, 30);
        assert!(text.contains('ℹ'), "{text}");
        assert!(text.contains("thermal throttling"), "{text}");
        assert!(text.contains("publication-grade"), "{text}");
    }

    #[test]
    fn config_stage_shows_the_focused_ladder_explanation() {
        // form_field 3 = Ladder.
        let app = setup_app_at_config(3);
        let text = render_setup_text(&app, 120, 30);
        assert!(text.contains('ℹ'), "{text}");
        assert!(text.contains("1,2,3,4,8,12,16,24,32"), "{text}");
        assert!(text.contains("1,4,16,64"), "{text}");
    }

    // ── step indicator ───────────────────────────────────────────────────

    #[test]
    fn step_index_groups_discover_and_model() {
        assert_eq!(SetupPhase::Url.step_index(), 0);
        assert_eq!(SetupPhase::Discover.step_index(), 1);
        assert_eq!(SetupPhase::Model.step_index(), 1);
        assert_eq!(SetupPhase::Config.step_index(), 2);
        assert_eq!(SetupPhase::Confirm.step_index(), 3);
    }
}
