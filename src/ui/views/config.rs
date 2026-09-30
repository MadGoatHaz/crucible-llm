//! View 5 — Configuration (blueprint §6 / plan Chunk 18): an *editable*
//! form for the target URL, model, mode, tokens, iterations, timeout,
//! API key, tokenizer path, cache-bypass, concurrency ladder, the hardware
//! (Engine D) toggle, and the per-engine enable switches (A/B/C1/C2/C3).
//!
//! **Editing model** — a single-row cursor walks the field list:
//!
//! * `Tab` / `Shift-Tab` / `↑` / `↓` — move the cursor between fields;
//! * typing — appends to the focused field (text fields take any graphic
//!   ASCII; number fields take digits);
//! * `Backspace` — deletes from the focused field;
//! * `Space` / `Enter` — toggles a boolean / cycles the mode;
//! * `←` / `→` / `-` / `+` — steps a number / cycles the mode;
//! * `F2` — **save** the form to the platform config file
//!   (`~/.config/crucible/config.json`);
//! * `F5` — **run** the selected engines against the edited config.
//!
//! The form is the TUI's edit surface over the same [`Config`] the headless
//! and export paths consume (cross-cutting "Config as single source of
//! truth", plan §5): [`ConfigState::to_config`] produces a [`Config`], and
//! [`ConfigState::save`] persists it through the existing
//! [`ConfigFile`] serde layer, so a value edited here flows end-to-end into
//! the next run, the SQLite persistence, and the `--export` output.
//!
//! **Measurement isolation** (blueprint §4): rendering is a pure `&App`
//! read; the only mutation is the user-driven key path (cursor/field
//! edits), which never touches the quanta timing path or the lock-free
//! metrics snapshot.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::config::{default_config_path, parse_ladder, Config, ConfigFile, EngineSelection, Mode};
use crate::engines::concurrency::DEFAULT_LADDER;
use crate::ui::app::App;
use crate::ui::theme::{palette, style};

/// The editable fields, in display / cursor order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Field {
    Url,
    Model,
    Mode,
    Tokens,
    Iterations,
    Timeout,
    ApiKey,
    Nocache,
    Tokenizer,
    Ladder,
    Hardware,
    EngineSpeed,
    EngineConcurrency,
    EngineNiah,
    EngineReasoning,
    EngineStructured,
}

impl Field {
    /// Every field in cursor order.
    pub const ALL: [Field; 16] = [
        Field::Url,
        Field::Model,
        Field::Mode,
        Field::Tokens,
        Field::Iterations,
        Field::Timeout,
        Field::ApiKey,
        Field::Nocache,
        Field::Tokenizer,
        Field::Ladder,
        Field::Hardware,
        Field::EngineSpeed,
        Field::EngineConcurrency,
        Field::EngineNiah,
        Field::EngineReasoning,
        Field::EngineStructured,
    ];

    /// The form label for this field.
    pub fn label(self) -> &'static str {
        match self {
            Field::Url => "Target URL",
            Field::Model => "Model",
            Field::Mode => "Mode",
            Field::Tokens => "Target tokens",
            Field::Iterations => "Iterations",
            Field::Timeout => "Timeout (s)",
            Field::ApiKey => "API key",
            Field::Nocache => "Cache bypass",
            Field::Tokenizer => "Tokenizer",
            Field::Ladder => "Concurrency ladder",
            Field::Hardware => "Hardware telemetry",
            Field::EngineSpeed => "Engine A — Speed",
            Field::EngineConcurrency => "Engine B — Concurrency",
            Field::EngineNiah => "Engine C1 — NIAH",
            Field::EngineReasoning => "Engine C2 — Reasoning",
            Field::EngineStructured => "Engine C3 — Structured",
        }
    }
}

/// The outcome of a Config-view key press (consumed by `App::handle_key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigKeyResult {
    /// The key edited/nailed the form; nothing special to do.
    Inert,
    /// `F2` — the form was saved to the config file.
    Saved,
    /// `F5` — the user asked to run the selected engines.
    Run,
}

/// The editable Configuration form (View 5).
///
/// Seeded from a resolved [`Config`] via [`from_config`]; edited in place by
/// the key path; turned back into a [`Config`] via [`to_config`] (for runs)
/// or persisted via [`save`] (the `F2` path).
#[derive(Debug, Clone)]
pub struct ConfigState {
    pub url: String,
    pub model: String,
    pub mode: Mode,
    pub tokens: u32,
    pub iterations: u32,
    pub timeout: u64,
    pub api_key: Option<String>,
    pub nocache: bool,
    pub tokenizer: Option<String>,
    /// Comma-separated ladder (`"1,2,4,8,16,32,64"`).
    pub ladder: String,
    /// Engine D — the hardware/energy telemetry poller.
    pub hardware: bool,
    pub engine_speed: bool,
    pub engine_concurrency: bool,
    pub engine_niah: bool,
    pub engine_reasoning: bool,
    pub engine_structured: bool,
    /// The cursor's position in [`Field::ALL`].
    pub cursor: usize,
    /// Where `F2` writes the form.
    pub config_path: PathBuf,
    /// `true` right after a successful save (drives the "saved" flash).
    pub saved: bool,
}

impl ConfigState {
    /// The field the cursor is currently on.
    pub fn current(&self) -> Field {
        Field::ALL[self.cursor.min(Field::ALL.len() - 1)]
    }

    /// Seed the form from a resolved [`Config`] (the TUI entry point).
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            mode: cfg.mode,
            tokens: cfg.tokens,
            iterations: cfg.iterations,
            timeout: cfg.timeout,
            api_key: cfg.api_key.clone(),
            nocache: cfg.nocache,
            tokenizer: cfg
                .tokenizer
                .as_deref()
                .map(|p| p.to_string_lossy().into_owned()),
            ladder: cfg
                .ladder
                .iter()
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(","),
            hardware: cfg.hardware,
            engine_speed: cfg.engines.speed,
            engine_concurrency: cfg.engines.concurrency,
            engine_niah: cfg.engines.niah,
            engine_reasoning: cfg.engines.reasoning,
            engine_structured: cfg.engines.structured,
            cursor: 0,
            config_path: default_config_path().unwrap_or_else(|| PathBuf::from("config.json")),
            saved: false,
        }
    }

    /// Turn the form back into a [`Config`] (the run path: `F5` / `r`).
    pub fn to_config(&self) -> Config {
        let ladder = parse_ladder(&self.ladder)
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| DEFAULT_LADDER.to_vec());
        let mut cfg = Config {
            url: self.url.trim().to_string(),
            model: self.model.trim().to_string(),
            mode: self.mode,
            tokens: self.tokens.max(1),
            iterations: self.iterations.max(1),
            timeout: self.timeout.max(1),
            nocache: self.nocache,
            ladder,
            hardware: self.hardware,
            engines: EngineSelection {
                speed: self.engine_speed,
                concurrency: self.engine_concurrency,
                niah: self.engine_niah,
                reasoning: self.engine_reasoning,
                structured: self.engine_structured,
                hardware: self.hardware,
            },
            ..Config::default()
        };
        cfg.api_key = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(String::from);
        cfg.tokenizer = self
            .tokenizer
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);
        cfg
    }

    /// `F2` — persist the form to the platform config file.
    ///
    /// Writes a [`ConfigFile`] (the existing serde layer) so the values
    /// survive into the next invocation via the normal CLI > env > file >
    /// defaults resolution. Returns the written path.
    pub fn save(&mut self) -> Result<PathBuf, String> {
        let ladder = parse_ladder(&self.ladder).filter(|l| !l.is_empty());
        let file = ConfigFile {
            url: Some(self.url.trim().to_string()),
            model: Some(self.model.trim().to_string()),
            mode: Some(self.mode),
            tokens: Some(self.tokens),
            iterations: Some(self.iterations),
            api_key: self
                .api_key
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(String::from),
            timeout: Some(self.timeout),
            nocache: Some(self.nocache),
            json: None,
            verbose: None,
            no_color: None,
            tokenizer: self
                .tokenizer
                .as_deref()
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
            tui: None,
            // Mode markers are CLI concerns — never persisted (Chunk 20).
            headless: None,
            export: None,
            export_path: None,
            ladder,
            hardware: Some(self.hardware),
            engines: Some(EngineSelection {
                speed: self.engine_speed,
                concurrency: self.engine_concurrency,
                niah: self.engine_niah,
                reasoning: self.engine_reasoning,
                structured: self.engine_structured,
                hardware: self.hardware,
            }),
        };
        let json = serde_json::to_string_pretty(&file).map_err(|e| e.to_string())?;
        if let Some(parent) = self.config_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).map_err(|e| e.to_string())?;
            }
        }
        std::fs::write(&self.config_path, json).map_err(|e| e.to_string())?;
        self.saved = true;
        Ok(self.config_path.clone())
    }

    /// Handle one key press on the form (the user-driven key path only).
    pub fn handle_key(&mut self, key: &KeyEvent) -> ConfigKeyResult {
        match key.code {
            // Navigation.
            KeyCode::Tab => {
                self.cursor = (self.cursor + 1) % Field::ALL.len();
                ConfigKeyResult::Inert
            }
            KeyCode::BackTab => {
                self.cursor = (self.cursor + Field::ALL.len() - 1) % Field::ALL.len();
                ConfigKeyResult::Inert
            }
            KeyCode::Down => {
                self.cursor = (self.cursor + 1).min(Field::ALL.len() - 1);
                ConfigKeyResult::Inert
            }
            KeyCode::Up => {
                self.cursor = self.cursor.saturating_sub(1);
                ConfigKeyResult::Inert
            }
            // Save / run.
            KeyCode::F(2) => match self.save() {
                Ok(_) => ConfigKeyResult::Saved,
                Err(_) => ConfigKeyResult::Inert,
            },
            KeyCode::F(5) => ConfigKeyResult::Run,
            // Toggle / cycle.
            KeyCode::Enter | KeyCode::Char(' ') => {
                self.toggle_current();
                ConfigKeyResult::Inert
            }
            // Step / cycle.
            KeyCode::Left | KeyCode::Char('-') => {
                self.adjust_current(-1);
                ConfigKeyResult::Inert
            }
            KeyCode::Right | KeyCode::Char('+') => {
                self.adjust_current(1);
                ConfigKeyResult::Inert
            }
            // Delete.
            KeyCode::Backspace => {
                self.backspace_current();
                ConfigKeyResult::Inert
            }
            // Insert (any graphic ASCII; fields that don't take text ignore it).
            KeyCode::Char(c) if c.is_ascii_graphic() => {
                self.type_current(c);
                ConfigKeyResult::Inert
            }
            _ => ConfigKeyResult::Inert,
        }
    }

    /// Type a character into the focused field.
    fn type_current(&mut self, c: char) {
        match self.current() {
            Field::Url => self.url.push(c),
            Field::Model => self.model.push(c),
            Field::ApiKey => self.api_key.get_or_insert_with(String::new).push(c),
            Field::Tokenizer => self.tokenizer.get_or_insert_with(String::new).push(c),
            Field::Ladder => self.ladder.push(c),
            Field::Tokens => {
                if let Some(d) = c.to_digit(10) {
                    self.tokens = self.tokens.saturating_mul(10).saturating_add(d);
                }
            }
            Field::Iterations => {
                if let Some(d) = c.to_digit(10) {
                    self.iterations = self.iterations.saturating_mul(10).saturating_add(d);
                }
            }
            Field::Timeout => {
                if let Some(d) = c.to_digit(10) {
                    self.timeout = self.timeout.saturating_mul(10).saturating_add(d as u64);
                }
            }
            // Mode / booleans don't take typed text.
            _ => {}
        }
    }

    /// Delete from the focused field.
    fn backspace_current(&mut self) {
        match self.current() {
            Field::Url => {
                self.url.pop();
            }
            Field::Model => {
                self.model.pop();
            }
            Field::ApiKey => {
                if let Some(s) = &mut self.api_key {
                    s.pop();
                    if s.is_empty() {
                        self.api_key = None;
                    }
                }
            }
            Field::Tokenizer => {
                if let Some(s) = &mut self.tokenizer {
                    s.pop();
                    if s.is_empty() {
                        self.tokenizer = None;
                    }
                }
            }
            Field::Ladder => {
                self.ladder.pop();
            }
            Field::Tokens => {
                self.tokens /= 10;
            }
            Field::Iterations => {
                self.iterations /= 10;
            }
            Field::Timeout => {
                self.timeout /= 10;
            }
            _ => {}
        }
    }

    /// Toggle / cycle the focused field (`Space` / `Enter`).
    fn toggle_current(&mut self) {
        match self.current() {
            Field::Mode => {
                self.mode = match self.mode {
                    Mode::Short => Mode::Long,
                    Mode::Long => Mode::Short,
                };
            }
            Field::Nocache => self.nocache = !self.nocache,
            Field::Hardware => self.hardware = !self.hardware,
            Field::EngineSpeed => self.engine_speed = !self.engine_speed,
            Field::EngineConcurrency => self.engine_concurrency = !self.engine_concurrency,
            Field::EngineNiah => self.engine_niah = !self.engine_niah,
            Field::EngineReasoning => self.engine_reasoning = !self.engine_reasoning,
            Field::EngineStructured => self.engine_structured = !self.engine_structured,
            _ => {}
        }
    }

    /// Step / cycle the focused field by `dir` (`←`/`-` = -1, `→`/`+` = +1).
    fn adjust_current(&mut self, dir: i32) {
        match self.current() {
            Field::Mode => {
                self.mode = match (self.mode, dir) {
                    (Mode::Short, 1) => Mode::Long,
                    (Mode::Long, -1) => Mode::Short,
                    (m, _) => m,
                };
            }
            Field::Tokens => {
                let step = 100u32;
                self.tokens = if dir > 0 {
                    self.tokens.saturating_add(step)
                } else {
                    self.tokens.saturating_sub(step)
                };
            }
            Field::Iterations => {
                self.iterations = if dir > 0 {
                    self.iterations.saturating_add(1)
                } else {
                    self.iterations.saturating_sub(1)
                };
            }
            Field::Timeout => {
                self.timeout = if dir > 0 {
                    self.timeout.saturating_add(5)
                } else {
                    self.timeout.saturating_sub(5)
                };
            }
            _ => {}
        }
    }
}

impl Default for ConfigState {
    fn default() -> Self {
        Self::from_config(&Config::default())
    }
}

/// Render the Configuration view into `area` (a pure `&App` read).
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let c = &app.config;
    let mut lines: Vec<Line> = Vec::with_capacity(Field::ALL.len() + 3);
    for &field in &Field::ALL {
        let is_cursor = field == c.current();
        let (label, value, vstyle) = field_display(field, c);
        let prefix = if is_cursor {
            Span::styled("> ", style::highlight())
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            prefix,
            Span::styled(format!("{label:<22} "), style::label()),
            Span::styled(value, vstyle),
        ]));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "[Tab/↑↓] move · [Space/Enter] toggle · [←→/+/-] step · [type] edit · [⌫] delete",
        style::footer(),
    )));
    lines.push(Line::from(vec![Span::styled(
        "[F2] Save to config file  ·  [F5] Run selected engines",
        style::footer(),
    )]));
    if c.saved {
        lines.push(Line::from(Span::styled("✓ saved", style::value_ok())));
    }
    let border = if c.saved {
        style::active_border()
    } else {
        style::border()
    };
    f.render_widget(
        Paragraph::new(Text::from(lines)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(border)
                .title("CONFIGURATION (edit a field, then F2 to save)"),
        ),
        area,
    );
}

/// `(label, value, value_style)` for one form field.
fn field_display(field: Field, c: &ConfigState) -> (String, String, Style) {
    match field {
        Field::Url => ("Target URL".to_string(), c.url.clone(), style::value()),
        Field::Model => ("Model".to_string(), c.model.clone(), style::value()),
        Field::Mode => (
            "Mode".to_string(),
            c.mode.label().to_string(),
            style::value(),
        ),
        Field::Tokens => (
            "Target tokens".to_string(),
            c.tokens.to_string(),
            style::value(),
        ),
        Field::Iterations => (
            "Iterations".to_string(),
            c.iterations.to_string(),
            style::value(),
        ),
        Field::Timeout => (
            "Timeout (s)".to_string(),
            c.timeout.to_string(),
            style::value(),
        ),
        Field::ApiKey => (
            "API key".to_string(),
            c.api_key
                .as_deref()
                .map(str::to_string)
                .unwrap_or_else(|| "(none)".to_string()),
            style::value(),
        ),
        Field::Nocache => (
            "Cache bypass".to_string(),
            bool_str(c.nocache),
            bool_style(c.nocache),
        ),
        Field::Tokenizer => (
            "Tokenizer".to_string(),
            c.tokenizer
                .as_deref()
                .map(str::to_string)
                .unwrap_or_else(|| "(chars/4, estimated)".to_string()),
            if c.tokenizer.is_some() {
                style::value()
            } else {
                style::value_warn()
            },
        ),
        Field::Ladder => (
            "Concurrency ladder".to_string(),
            c.ladder.clone(),
            style::value(),
        ),
        Field::Hardware => (
            "Hardware telemetry".to_string(),
            bool_str(c.hardware),
            bool_style(c.hardware),
        ),
        Field::EngineSpeed => (
            "Engine A — Speed".to_string(),
            bool_str(c.engine_speed),
            bool_style(c.engine_speed),
        ),
        Field::EngineConcurrency => (
            "Engine B — Concurrency".to_string(),
            bool_str(c.engine_concurrency),
            bool_style(c.engine_concurrency),
        ),
        Field::EngineNiah => (
            "Engine C1 — NIAH".to_string(),
            bool_str(c.engine_niah),
            bool_style(c.engine_niah),
        ),
        Field::EngineReasoning => (
            "Engine C2 — Reasoning".to_string(),
            bool_str(c.engine_reasoning),
            bool_style(c.engine_reasoning),
        ),
        Field::EngineStructured => (
            "Engine C3 — Structured".to_string(),
            bool_str(c.engine_structured),
            bool_style(c.engine_structured),
        ),
    }
}

fn bool_str(b: bool) -> String {
    if b {
        "on".to_string()
    } else {
        "off".to_string()
    }
}

fn bool_style(b: bool) -> Style {
    if b {
        style::value_ok()
    } else {
        Style::default().fg(palette::MUTED)
    }
}
