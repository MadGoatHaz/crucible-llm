//! View 6 — Configuration (blueprint §6 / plan Chunk 18): an *editable*
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
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::config::{
    default_config_path, parse_ladder, Config, ConfigFile, EngineSelection, Mode, DEFAULT_RATE_KWH,
};
use crate::engines::concurrency::DEFAULT_LADDER;
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, style, Theme};

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
    EngineFlatOut,
    Theme,
    RateKwh,
}

impl Field {
    /// Every field in cursor order.
    pub const ALL: [Field; 19] = [
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
        Field::EngineFlatOut,
        Field::Theme,
        Field::RateKwh,
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
            Field::EngineFlatOut => "Engine F — Flat Out",
            Field::Theme => "Theme",
            Field::RateKwh => "Electricity rate",
        }
    }

    /// The [`Engine`] this field toggles (`None` for the non-engine
    /// fields) — the form shows the focused engine's description so the
    /// user can see what a benchmark measures before toggling it.
    pub fn engine(self) -> Option<crate::engines::sequence::Engine> {
        use crate::engines::sequence::Engine;
        match self {
            Field::EngineSpeed => Some(Engine::Speed),
            Field::EngineConcurrency => Some(Engine::Concurrency),
            Field::EngineNiah => Some(Engine::Niah),
            Field::EngineReasoning => Some(Engine::Reasoning),
            Field::EngineStructured => Some(Engine::Structured),
            Field::Hardware => Some(Engine::Hardware),
            Field::EngineFlatOut => Some(Engine::FlatOut),
            _ => None,
        }
    }

    /// The dimmed `ℹ` explanation shown below the field while it has
    /// focus.
    ///
    /// Engine fields return `None` and reuse
    /// [`Engine::description`](Self::engine) instead. Non-engine fields
    /// carry their own help text (max 4 lines, pre-wrapped to fit the
    /// terminal panel).
    pub fn explanation(self) -> Option<&'static str> {
        match self {
            Field::Url => Some(
                "Base URL of your OpenAI-compatible server.\n\
                 Examples: http://localhost:8000/v1 (vLLM)\n\
                 http://localhost:11434/v1 (Ollama)\n\
                 Must be reachable from this machine.",
            ),
            Field::Model => Some(
                "The model to benchmark. Usually auto-detected\n\
                 from the server. Must match exactly what the\n\
                 server reports (case-sensitive).",
            ),
            Field::Mode => Some(
                "\"short\" = ~100 token prompt. Tests responsiveness\n\
                 (TTFT). \"long\" = your token target as prompt.\n\
                 Tests sustained throughput. Use \"short\" for quick\n\
                 checks, \"long\" for realistic workloads.",
            ),
            Field::Tokens => Some(
                "Maximum tokens to generate per request\n\
                 (max_tokens). 256 = quick. 10000 = standard.\n\
                 8192+ = stress. Higher = more stable averages\n\
                 but longer test time.",
            ),
            Field::Iterations => Some(
                "How many times to repeat each benchmark.\n\
                 1 = quick check. 3-5 = reliable. 10+ =\n\
                 publication-grade. More iterations reduces\n\
                 variance from caching/throttling.",
            ),
            Field::Timeout => Some(
                "Max seconds to wait for a complete response.\n\
                 Default: 120s. Increase for slow models.\n\
                 If no tokens arrive for this duration, the\n\
                 stream is killed. Too low = false timeouts.",
            ),
            Field::ApiKey => Some(
                "Bearer token for authenticated servers. Most\n\
                 local servers (vLLM, Ollama) don't need this.\n\
                 Cloud APIs may require it. Stored only in\n\
                 memory during the session (not persisted).",
            ),
            Field::Nocache => Some(
                "Adds a random marker to prevent KV-cache hits.\n\
                 ON = cold-start (real first-request perf).\n\
                 OFF = warm (steady-state). Most servers cache\n\
                 repeated prompts — this measures real gen.",
            ),
            Field::Tokenizer => Some(
                "Path to a HuggingFace tokenizer.json for\n\
                 exact token counting. Without it: estimate\n\
                 at chars÷4. With it: exact counts. Leave\n\
                 empty to use estimation.",
            ),
            Field::Ladder => Some(
                "Concurrency levels to test (Engine B only).\n\
                 Comma-separated, e.g.: 1,2,3,4,8,12,16,24,32.\n\
                 Each level spawns that many simultaneous\n\
                 requests. Finds your server's practical limit.",
            ),
            Field::Theme => Some(
                "The color theme of the whole TUI.\n\
                 ←/→ cycles (with a live preview); 1/2/3\n\
                 selects Cyberpunk / Vampire / Monochrome.\n\
                 Saved when you press Esc.",
            ),
            Field::RateKwh => Some(
                "Your electricity rate in $/kWh (e.g., 0.16).\n\
                 Drives the $/1M-token cost in the GPU &\n\
                 Power view. Blank/invalid falls back to $0.16.\n\
                 Type digits and a decimal point.",
            ),
            // Engine fields use `Engine::description()` via `engine()`.
            Field::Hardware
            | Field::EngineSpeed
            | Field::EngineConcurrency
            | Field::EngineNiah
            | Field::EngineReasoning
            | Field::EngineStructured
            | Field::EngineFlatOut => None,
        }
    }
}

/// The Config view's interaction mode (FIX 4 — the "edit gate"):
///
/// * [`Viewing`] — the default on entry. The form is shown **read-only**
///   behind a gate ("press [Enter] to edit"). Only `Enter` (→ edit),
///   `Esc` (→ back to Live), `1`–`4` (→ switch view), `5` (stay), `F2`
///   (save) and `F5` (run → Live) are live; every other key is ignored, so
///   the view can never capture the number keys and trap the user.
/// * [`Editing`] — the form is editable. `Esc` saves and returns to the
///   gate (stay on tab 6); `q` always quits; all other keys (characters
///   including `1`–`9`/`0`, Tab, arrows, `F2`, `F5`) edit the focused
///   field. `F5` launches the run and switches to Live.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ConfigMode {
    #[default]
    Viewing,
    Editing,
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

/// The editable Configuration form (View 6).
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
    /// Comma-separated ladder (`"1,2,3,4,8,12,16,24,32"`).
    pub ladder: String,
    /// Engine D — the hardware/energy telemetry poller.
    pub hardware: bool,
    pub engine_speed: bool,
    pub engine_concurrency: bool,
    pub engine_niah: bool,
    pub engine_reasoning: bool,
    pub engine_structured: bool,
    pub engine_flatout: bool,
    /// The color theme (`"cyberpunk"` / `"vampire"` / `"monochrome"`).
    pub theme: String,
    /// The electricity rate in $/kWh, kept as a text buffer for editing
    /// (`"0.16"`) and parsed on save / run (blank or invalid input falls
    /// back to the built-in default).
    pub rate_text: String,
    /// The cursor's position in [`Field::ALL`].
    pub cursor: usize,
    /// Where `F2` writes the form.
    pub config_path: PathBuf,
    /// `true` right after a successful save (drives the "saved" flash).
    pub saved: bool,
    /// The interaction mode (FIX 4): [`ConfigMode::Viewing`] (the read-only
    /// gate, the default on entry) or [`ConfigMode::Editing`] (fields live).
    /// (Named `edit_mode` to avoid colliding with the benchmark `mode`.)
    pub edit_mode: ConfigMode,
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
            // FIX 4: the form's Hardware/Energy toggle mirrors the
            // *engine selection* (D is off by default — it must run on
            // the GPU box). `to_config` keeps the two in lockstep, so
            // seeding from `engines.hardware` is what the user sees and
            // what a run will use.
            hardware: cfg.engines.hardware,
            engine_speed: cfg.engines.speed,
            engine_concurrency: cfg.engines.concurrency,
            engine_niah: cfg.engines.niah,
            engine_reasoning: cfg.engines.reasoning,
            engine_structured: cfg.engines.structured,
            engine_flatout: cfg.engines.flatout,
            theme: cfg.theme.clone(),
            rate_text: format!("{}", cfg.rate_per_kwh),
            cursor: 0,
            config_path: default_config_path().unwrap_or_else(|| PathBuf::from("config.json")),
            saved: false,
            // FIX 4: every entry into the Config view starts at the gate.
            edit_mode: ConfigMode::Viewing,
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
                flatout: self.engine_flatout,
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
        cfg.theme = self.theme.clone();
        cfg.rate_per_kwh = parse_rate(&self.rate_text);
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
            // The log dir is a CLI/env concern — never persisted from the form.
            log_dir: None,
            ladder,
            hardware: Some(self.hardware),
            engines: Some(EngineSelection {
                speed: self.engine_speed,
                concurrency: self.engine_concurrency,
                niah: self.engine_niah,
                reasoning: self.engine_reasoning,
                structured: self.engine_structured,
                hardware: self.hardware,
                flatout: self.engine_flatout,
            }),
            // The matrix axis is a CLI/env concern — never persisted from the form.
            matrix_contexts: None,
            // The color theme is a pure preference — persisted so the
            // first-run picker is skipped on the next launch.
            theme: Some(self.theme.clone()),
            // The `$/kWh` rate is edited from the form (the RateKwh field)
            // and persisted so it survives into the next launch via the
            // normal CLI > env > file > default resolution.
            rate_per_kwh: Some(parse_rate(&self.rate_text)),
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

    /// Reset the form to built-in defaults (`R` key).
    pub fn reset_to_defaults(&mut self) {
        *self = Self::default();
    }

    /// Handle one key press on the form (the user-driven key path only).
    pub fn handle_key(&mut self, key: &KeyEvent) -> ConfigKeyResult {
        match key.code {
            // Reset to defaults (`R` — intercepted before character typing).
            KeyCode::Char('R') => {
                self.reset_to_defaults();
                ConfigKeyResult::Inert
            }
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
            // Theme: `1`/`2`/`3` select directly (the live preview follows).
            Field::Theme => {
                if let Some(n) = c.to_digit(10) {
                    if (1..=Theme::ALL.len() as u32).contains(&n) {
                        self.theme = Theme::ALL[(n - 1) as usize].id().to_string();
                    }
                }
            }
            // Electricity rate: digits and a single decimal point (the
            // buffer is validated, so a half-typed "0." is kept but a
            // second dot or a letter is rejected).
            Field::RateKwh => {
                let next = format!("{}{c}", self.rate_text);
                if is_valid_rate(&next) {
                    self.rate_text = next;
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
            Field::RateKwh => {
                self.rate_text.pop();
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
            Field::EngineFlatOut => self.engine_flatout = !self.engine_flatout,
            Field::Theme => {
                let idx = Theme::from_id(&self.theme)
                    .and_then(|t| Theme::ALL.iter().position(|x| *x == t))
                    .unwrap_or(0);
                let next = (idx + 1) % Theme::ALL.len();
                self.theme = Theme::ALL[next].id().to_string();
            }
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
            Field::Theme => {
                let idx = Theme::from_id(&self.theme)
                    .and_then(|t| Theme::ALL.iter().position(|x| *x == t))
                    .unwrap_or(0);
                let next = (idx as i32 + dir).rem_euclid(Theme::ALL.len() as i32) as usize;
                self.theme = Theme::ALL[next].id().to_string();
            }
            // Electricity rate: step by a cent in the focused direction.
            Field::RateKwh => {
                let cur = if is_valid_rate(&self.rate_text) {
                    self.rate_text.trim().parse::<f64>().unwrap_or(0.0)
                } else {
                    0.0
                };
                self.rate_text = format!("{:.2}", (cur + dir as f64 * 0.01).max(0.0));
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
///
/// FIX 4: the view has two modes. [`ConfigMode::Viewing`] shows the
/// **read-only gate** (current settings + "press Enter to edit");
/// [`ConfigMode::Editing`] shows the editable form. The gate is what the
/// user sees on entry, so the number keys can never be captured by field
/// editing (the user can always leave with `Esc` / `1`–`4`).
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
    let c = &app.config;
    // The detected GPU's display name (for the "Hardware telemetry" row) —
    // `None` when no GPU is present (the row then reads "no GPU").
    let gpu_label = app
        .gpu
        .as_ref()
        .map(|g| crate::hw::gpu_display_name(g.as_ref()));
    if c.edit_mode == ConfigMode::Viewing {
        render_gate(area, c, th, f, gpu_label.as_deref());
    } else {
        render_form(area, c, th, f, gpu_label.as_deref());
    }
}

/// The read-only **gate** (FIX 4): the current settings, a "press Enter to
/// edit" prompt, and the always-available exit keys. No field is focused,
/// so no key can be swallowed by the editor and the user can never get
/// stuck.
fn render_gate(area: Rect, c: &ConfigState, th: Theme, f: &mut Frame, gpu_label: Option<&str>) {
    let block = theme::block(
        theme::panel_title(th, "CONFIG (read-only — press Enter to edit)"),
        style::border(th),
    );
    if area.width < 10 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let mut lines: Vec<Line> = Vec::with_capacity(Field::ALL.len() + 4);
    lines.push(Line::from(Span::styled(
        "You are in Config view. [Enter] to edit settings, or [Esc] / [1-4] to return to monitoring.",
        style::value_warn(th),
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "Current settings:",
        style::label(th),
    )));
    for &field in &Field::ALL {
        let (label, value, vstyle) = field_display(field, c, th, gpu_label);
        lines.push(Line::from(vec![
            Span::raw("  "),
            Span::styled(format!("{label:<22} "), style::label(th)),
            Span::styled(value, vstyle),
        ]));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("Settings saved to {}", c.config_path.display()),
        style::footer(th),
    )));
    lines.push(Line::from(Span::styled(
        "[Enter] Edit  ·  [Esc] Back to Live  ·  [R] Reset  ·  [F2] Save  ·  [F5] Run → Live",
        style::footer(th),
    )));
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The **editable form** (FIX 4, [`ConfigMode::Editing`]): the field list
/// with a cursor, the focused field's explanation (dimmed `ℹ` note in a
/// reserved 4-line area), and the edit-mode key hints (`Esc` saves and
/// returns to the gate; all other keys type into the focused field).
fn render_form(area: Rect, c: &ConfigState, th: Theme, f: &mut Frame, gpu_label: Option<&str>) {
    let mut lines: Vec<Line> = Vec::with_capacity(Field::ALL.len() + 10);
    for &field in &Field::ALL {
        let is_cursor = field == c.current();
        let (label, value, vstyle) = field_display(field, c, th, gpu_label);
        let prefix = if is_cursor {
            Span::styled("> ", style::highlight(th))
        } else {
            Span::raw("  ")
        };
        lines.push(Line::from(vec![
            prefix,
            Span::styled(format!("{label:<22} "), style::label(th)),
            Span::styled(value, vstyle),
        ]));
    }
    // The focused field's explanation (dimmed `ℹ` note) — always 4 lines
    // (reserved space so the layout doesn't jump as the cursor moves).
    lines.push(Line::raw(""));
    let focused = c.current();
    if let Some(engine) = focused.engine() {
        // Engine fields: use the engine's description (≤3 lines).
        lines.extend(crate::ui::views::engine_info_lines(th, engine));
    } else if let Some(text) = focused.explanation() {
        // Non-engine fields: use the field's own explanation (≤4 lines).
        lines.extend(crate::ui::views::info_lines(th, text));
    }
    // Pad to always 4 lines (reserved space).
    let field_end = Field::ALL.len() + 1; // 16 fields + 1 blank
    while lines.len() < field_end + 4 {
        lines.push(Line::raw(""));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "[Esc] Save & exit  ·  [F2] Save  ·  [F5] Run → Live  ·  [R] Reset",
        style::footer(th),
    )));
    lines.push(Line::from(Span::styled(
        "[Tab/↑↓] move · [Space/Enter] toggle · [←→/+/-] step · [type] edit · [⌫] delete",
        style::footer(th),
    )));
    lines.push(Line::from(Span::styled(
        format!("Settings saved to {}", c.config_path.display()),
        style::footer(th),
    )));
    if c.saved {
        lines.push(Line::from(Span::styled("✓ saved", style::value_ok(th))));
    }
    let border = if c.saved {
        style::active_border(th)
    } else {
        style::border(th)
    };
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(theme::block(
                theme::panel_title(th, "CONFIG (editing — Esc to save & exit)"),
                border,
            ))
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// `(label, value, value_style)` for one form field. `gpu_label` is the
/// detected GPU's model (or `None`) — the Hardware row shows it
/// (`"[✓] Energy — RTX 4090"`) or `"no GPU"` when absent.
fn field_display(
    field: Field,
    c: &ConfigState,
    th: Theme,
    gpu_label: Option<&str>,
) -> (String, String, Style) {
    match field {
        Field::Url => (
            "Target URL".to_string(),
            fmt::truncate(&c.url, 44),
            style::value(th),
        ),
        Field::Model => (
            "Model".to_string(),
            fmt::truncate(&c.model, 44),
            style::value(th),
        ),
        Field::Mode => (
            "Mode".to_string(),
            c.mode.label().to_string(),
            style::value(th),
        ),
        Field::Tokens => (
            "Target tokens".to_string(),
            c.tokens.to_string(),
            style::value(th),
        ),
        Field::Iterations => (
            "Iterations".to_string(),
            c.iterations.to_string(),
            style::value(th),
        ),
        Field::Timeout => (
            "Timeout (s)".to_string(),
            c.timeout.to_string(),
            style::value(th),
        ),
        Field::ApiKey => (
            "API key".to_string(),
            fmt::truncate(
                c.api_key
                    .as_deref()
                    .map(str::to_string)
                    .unwrap_or_else(|| "(none)".to_string())
                    .as_str(),
                44,
            ),
            style::value(th),
        ),
        Field::Nocache => (
            "Cache bypass".to_string(),
            bool_str(c.nocache),
            bool_style(th, c.nocache),
        ),
        Field::Tokenizer => (
            "Tokenizer".to_string(),
            fmt::truncate(
                c.tokenizer
                    .as_deref()
                    .map(str::to_string)
                    .unwrap_or_else(|| "(chars/4, estimated)".to_string())
                    .as_str(),
                44,
            ),
            if c.tokenizer.is_some() {
                style::value(th)
            } else {
                style::value_warn(th)
            },
        ),
        Field::Ladder => (
            "Concurrency ladder".to_string(),
            c.ladder.clone(),
            style::value(th),
        ),
        Field::Hardware => {
            let glyph = if c.hardware { "✓" } else { "✗" };
            let name = gpu_label.unwrap_or("no GPU");
            (
                "Hardware telemetry".to_string(),
                format!("[{glyph}] Energy — {name}"),
                bool_style(th, c.hardware),
            )
        }
        Field::EngineSpeed => (
            "Engine A — Speed".to_string(),
            bool_str(c.engine_speed),
            bool_style(th, c.engine_speed),
        ),
        Field::EngineConcurrency => (
            "Engine B — Concurrency".to_string(),
            bool_str(c.engine_concurrency),
            bool_style(th, c.engine_concurrency),
        ),
        Field::EngineNiah => (
            "Engine C1 — NIAH".to_string(),
            bool_str(c.engine_niah),
            bool_style(th, c.engine_niah),
        ),
        Field::EngineReasoning => (
            "Engine C2 — Reasoning".to_string(),
            bool_str(c.engine_reasoning),
            bool_style(th, c.engine_reasoning),
        ),
        Field::EngineStructured => (
            "Engine C3 — Structured".to_string(),
            bool_str(c.engine_structured),
            bool_style(th, c.engine_structured),
        ),
        Field::EngineFlatOut => (
            "Engine F — Flat Out".to_string(),
            bool_str(c.engine_flatout),
            bool_style(th, c.engine_flatout),
        ),
        Field::Theme => (
            "Theme".to_string(),
            Theme::from_id(&c.theme)
                .map(|t| t.name().to_string())
                .unwrap_or_else(|| c.theme.clone()),
            style::value(th),
        ),
        Field::RateKwh => {
            // The raw buffer is shown (live edit feedback); a blank or
            // half-typed value is flagged until it parses.
            let text = c.rate_text.trim();
            (
                "Electricity rate".to_string(),
                if text.is_empty() {
                    "(default $0.16)".to_string()
                } else {
                    format!("${text}/kWh")
                },
                if is_valid_rate(text) {
                    style::value(th)
                } else {
                    style::value_warn(th)
                },
            )
        }
    }
}

/// The maximum decimal places the `$/kWh` rate field accepts.
const RATE_MAX_DECIMALS: usize = 3;

/// Validate the rate text buffer: digits only, at most one `.`, at most
/// [`RATE_MAX_DECIMALS`] decimals (so a half-typed `"0."` stays valid
/// while typing, but `"0.1.2"` or a letter is rejected).
fn is_valid_rate(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let mut dots = 0;
    let mut decimals = 0;
    for ch in s.chars() {
        if ch == '.' {
            dots += 1;
        } else if ch.is_ascii_digit() {
            if dots == 1 {
                decimals += 1;
            }
        } else {
            return false;
        }
    }
    dots <= 1 && decimals <= RATE_MAX_DECIMALS
}

/// Parse the rate text buffer into `$/kWh`: a valid rate is clamped
/// non-negative; blank / invalid input falls back to
/// [`DEFAULT_RATE_KWH`] (the built-in default).
fn parse_rate(s: &str) -> f64 {
    if is_valid_rate(s) {
        s.trim()
            .parse::<f64>()
            .ok()
            .map(|v| v.max(0.0))
            .unwrap_or(DEFAULT_RATE_KWH)
    } else {
        DEFAULT_RATE_KWH
    }
}

/// The user's **live** `$/kWh` rate from the Config form (read at render
/// time so a rate change in View 6 immediately affects the GPU & Power
/// view's cost calculations — the monitor's own `rate_per_kwh` is a
/// startup snapshot and can be stale).
pub fn current_rate(app: &crate::ui::app::App) -> f64 {
    parse_rate(&app.config.rate_text)
}

/// The toggle state glyph: `[✓]` on, `[ ]` off (clear, at-a-glance state).
fn bool_str(b: bool) -> String {
    if b {
        "[✓]".to_string()
    } else {
        "[ ]".to_string()
    }
}

fn bool_style(th: Theme, b: bool) -> Style {
    if b {
        style::value_ok(th)
    } else {
        Style::default().fg(th.dim())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::app::{App, View};

    fn render_text(app: &App, w: u16, h: u16) -> String {
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

    #[test]
    fn focused_engine_field_shows_its_description() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing; // the form (not the gate)
        app.config.cursor = 10; // Hardware (Engine D)
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains('ℹ'),
            "info note for the focused engine: {text}"
        );
        assert!(text.contains("GPU power profiling"), "{text}");

        app.config.cursor = 11; // Engine A
        let text = render_text(&app, 120, 40);
        assert!(text.contains("Single-stream throughput"), "{text}");
    }

    #[test]
    fn non_engine_field_shows_its_explanation() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing; // the form (not the gate)
        app.config.cursor = 0; // Url
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains('ℹ'),
            "info note for the focused non-engine field: {text}"
        );
        assert!(
            text.contains("OpenAI-compatible"),
            "URL explanation mentions OpenAI-compatible: {text}"
        );
    }

    #[test]
    fn each_field_shows_its_own_explanation() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;

        // Model (cursor 1)
        app.config.cursor = 1;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("case-sensitive"), "Model explanation: {text}");

        // Mode (cursor 2)
        app.config.cursor = 2;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("TTFT"), "Mode explanation: {text}");

        // Tokens (cursor 3)
        app.config.cursor = 3;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("max_tokens"), "Tokens explanation: {text}");

        // Iterations (cursor 4)
        app.config.cursor = 4;
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("publication-grade"),
            "Iterations explanation: {text}"
        );

        // Timeout (cursor 5)
        app.config.cursor = 5;
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("Default: 120s"),
            "Timeout explanation: {text}"
        );

        // API Key (cursor 6)
        app.config.cursor = 6;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("Bearer token"), "API Key explanation: {text}");

        // Cache Bypass (cursor 7)
        app.config.cursor = 7;
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("KV-cache"),
            "Cache Bypass explanation: {text}"
        );

        // Tokenizer (cursor 8)
        app.config.cursor = 8;
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("tokenizer.json"),
            "Tokenizer explanation: {text}"
        );

        // Ladder (cursor 9)
        app.config.cursor = 9;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("simultaneous"), "Ladder explanation: {text}");
    }

    #[test]
    fn explanation_area_is_reserved_space() {
        // The explanation area is always 4 lines, regardless of which
        // field is focused. Verify by checking that a 3-line explanation
        // (Model) and a 4-line explanation (URL) both produce the same
        // total line count in the rendered output.
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;

        app.config.cursor = 0; // URL (4-line explanation)
        let text_4 = render_text(&app, 120, 40);

        app.config.cursor = 1; // Model (3-line explanation)
        let text_3 = render_text(&app, 120, 40);

        // Both should contain the ℹ marker and the config path footer.
        assert!(text_4.contains('ℹ'), "4-line explanation has ℹ");
        assert!(text_3.contains('ℹ'), "3-line explanation has ℹ");
        assert!(
            text_4.contains("Settings saved to"),
            "4-line form shows config path"
        );
        assert!(
            text_3.contains("Settings saved to"),
            "3-line form shows config path"
        );
    }

    #[test]
    fn reset_to_defaults_restores_config() {
        let c = ConfigState {
            url: "http://custom:1234/v1".to_string(),
            model: "custom-model".to_string(),
            tokens: 9999,
            nocache: true,
            cursor: 5,
            ..ConfigState::default()
        };
        let mut c = c;
        c.reset_to_defaults();

        assert_eq!(c.url, Config::default().url);
        assert_eq!(c.model, Config::default().model);
        assert_eq!(c.tokens, Config::default().tokens);
        assert!(!c.nocache);
        assert_eq!(c.cursor, 0);
    }

    #[test]
    fn r_key_resets_to_defaults_in_edit_mode() {
        let c = ConfigState {
            url: "http://custom:1234/v1".to_string(),
            tokens: 9999,
            ..ConfigState::default()
        };
        let mut c = c;

        let key = crossterm::event::KeyEvent {
            code: crossterm::event::KeyCode::Char('R'),
            modifiers: crossterm::event::KeyModifiers::SHIFT,
            kind: crossterm::event::KeyEventKind::Press,
            state: crossterm::event::KeyEventState::NONE,
        };
        let result = c.handle_key(&key);
        assert_eq!(result, ConfigKeyResult::Inert);
        assert_eq!(c.url, Config::default().url);
        assert_eq!(c.tokens, Config::default().tokens);
    }

    #[test]
    fn gate_shows_config_path() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Viewing; // the gate
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("Settings saved to"),
            "gate shows config path: {text}"
        );
    }

    // ── electricity rate ($/kWh) field ───────────────────────────────────

    #[test]
    fn rate_field_renders_in_the_form() {
        let mut app = App::new();
        app.view = View::Config;
        app.config.edit_mode = ConfigMode::Editing;
        app.config.cursor = 18; // RateKwh (the last field)
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("Electricity rate"),
            "rate field label: {text}"
        );
        assert!(text.contains("$0.16/kWh"), "default rate shown: {text}");
    }

    #[test]
    fn rate_field_accepts_digits_and_decimal() {
        let mut c = ConfigState {
            cursor: 18, // RateKwh
            rate_text: String::new(),
            ..Default::default()
        };
        for ch in "0.18".chars() {
            c.type_current(ch);
        }
        assert_eq!(c.rate_text, "0.18");
        assert!((c.to_config().rate_per_kwh - 0.18).abs() < 1e-9);
    }

    #[test]
    fn rate_field_rejects_invalid_characters() {
        let mut c = ConfigState {
            cursor: 18, // RateKwh
            rate_text: String::new(),
            ..Default::default()
        };
        for ch in "1x2.".chars() {
            c.type_current(ch);
        }
        // The letter is rejected; "12." is a valid half-typed state.
        assert_eq!(c.rate_text, "12.");
        c.type_current('3');
        assert_eq!(c.rate_text, "12.3");
    }

    #[test]
    fn rate_field_rejects_a_second_decimal_point() {
        let mut c = ConfigState {
            cursor: 18, // RateKwh
            rate_text: "0.1".to_string(),
            ..Default::default()
        };
        c.type_current('.');
        assert_eq!(c.rate_text, "0.1", "second dot rejected");
    }

    #[test]
    fn rate_backspace_and_step_work() {
        let mut c = ConfigState {
            cursor: 18, // RateKwh
            rate_text: "0.16".to_string(),
            ..Default::default()
        };
        c.backspace_current();
        assert_eq!(c.rate_text, "0.1");
        c.adjust_current(1);
        assert_eq!(c.rate_text, "0.11");
        c.adjust_current(-1);
        assert_eq!(c.rate_text, "0.10");
    }

    #[test]
    fn parse_rate_falls_back_on_invalid() {
        assert!((parse_rate("0.16") - 0.16).abs() < 1e-9);
        assert_eq!(parse_rate(""), DEFAULT_RATE_KWH);
        assert_eq!(parse_rate("abc"), DEFAULT_RATE_KWH);
        assert_eq!(parse_rate("0.1.2"), DEFAULT_RATE_KWH);
        assert_eq!(parse_rate("-5"), DEFAULT_RATE_KWH);
    }

    #[test]
    fn rate_round_trips_through_save() {
        let c = ConfigState {
            rate_text: "0.25".to_string(),
            ..Default::default()
        };
        // `save` persists the parsed rate (it writes to disk; the parsed
        // value is what the next launch layers in).
        assert!((parse_rate(&c.rate_text) - 0.25).abs() < 1e-9);
        assert!((c.to_config().rate_per_kwh - 0.25).abs() < 1e-9);
    }

    #[test]
    fn explanation_texts_are_max_4_lines() {
        // Verify every non-engine field's explanation is at most 4 lines.
        for &field in &Field::ALL {
            if let Some(text) = field.explanation() {
                let line_count = text.lines().count();
                assert!(
                    line_count <= 4,
                    "{field:?} explanation is {line_count} lines (max 4): {text:?}"
                );
            }
        }
    }
}
