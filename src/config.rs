//! Configuration: the single source of truth for every run (plan Chunk 7,
//! cross-cutting "Config as single source of truth").
//!
//! One [`Config`] — a **superset of both Python prototype CLIs**
//! (`llmspeedtest.py` + `llmspeedtest2.py`) — feeds the headless path
//! (this chunk), the TUI path (Chunk 8), and the export path (Chunk 13)
//! identically.
//!
//! Mode dispatch (Chunk 20): **the TUI is the default**. A run goes
//! headless only when `--headless` / `--json` is given (or stdout is not a
//! TTY); the `Config` fields [`Config::headless`] / [`Config::banner`]
//! carry those markers.
//!
//! Resolution order (first match wins, per field):
//!
//! 1. **CLI flag** (`--url`, `--model`, …) — [`Cli`] via `clap`;
//! 2. **environment variable** (`CRUCIBLE_URL`, `CRUCIBLE_MODEL`, …);
//! 3. **config file** (JSON; `~/.config/crucible/config.json` by default,
//!    overridable with `--config <path>` / `CRUCIBLE_CONFIG`);
//! 4. **built-in default** (mirroring `llmspeedtest.py`).
//!
//! The platform data dir (SQLite in Chunk 12, exports in Chunk 13) lives at
//! `dirs::data_dir()/crucible` — see [`data_dir`].

use std::env;
use std::path::{Path, PathBuf};

use clap::{ArgMatches, CommandFactory, FromArgMatches, Parser, ValueEnum};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::engines::concurrency::DEFAULT_LADDER;

/// Default endpoint (parity with `llmspeedtest.py`'s `DEFAULT_URL`).
pub const DEFAULT_URL: &str = "http://192.168.51.163:8080/v1/chat/completions";
/// Default model name (parity with `llmspeedtest.py`'s `DEFAULT_MODEL`).
pub const DEFAULT_MODEL: &str = "default";
/// Default connection/read timeout in seconds (parity with the prototype).
pub const DEFAULT_TIMEOUT_SECS: u64 = 120;
/// Default target prompt tokens for `long` mode (parity with the prototype).
pub const DEFAULT_TOKENS: u32 = 10000;
/// Default v0.1.1 2D matrix context axis (target prompt tokens):
/// `0` = the configured prompt as-is, plus the 8k and 32k context sizes.
pub const DEFAULT_MATRIX_CONTEXTS: [u32; 3] = [0, 8000, 32000];

/// The default config file name inside the config dir.
pub const CONFIG_FILE_NAME: &str = "config.json";

/// Environment variable names (layer 2 of the resolution order).
pub mod env_vars {
    pub const URL: &str = "CRUCIBLE_URL";
    pub const MODEL: &str = "CRUCIBLE_MODEL";
    pub const MODE: &str = "CRUCIBLE_MODE";
    pub const TOKENS: &str = "CRUCIBLE_TOKENS";
    pub const ITERATIONS: &str = "CRUCIBLE_ITERATIONS";
    pub const API_KEY: &str = "CRUCIBLE_API_KEY";
    pub const TIMEOUT: &str = "CRUCIBLE_TIMEOUT";
    pub const NOCACHE: &str = "CRUCIBLE_NOCACHE";
    pub const TOKENIZER: &str = "CRUCIBLE_TOKENIZER";
    pub const JSON: &str = "CRUCIBLE_JSON";
    pub const VERBOSE: &str = "CRUCIBLE_VERBOSE";
    pub const NO_COLOR: &str = "CRUCIBLE_NO_COLOR";
    pub const TUI: &str = "CRUCIBLE_TUI";
    /// `true`/`false` — run the classic headless one-shot benchmark
    /// instead of the TUI (Chunk 20; `--json` implies it too).
    pub const HEADLESS: &str = "CRUCIBLE_HEADLESS";
    pub const CONFIG: &str = "CRUCIBLE_CONFIG";
    /// Comma-separated concurrency ladder (Chunk 18).
    pub const LADDER: &str = "CRUCIBLE_LADDER";
    /// `true`/`false` — enable the hardware/energy telemetry poller
    /// (Chunk 18).
    pub const HARDWARE: &str = "CRUCIBLE_HARDWARE";
    /// Comma-separated engine selection (Chunk 18): `speed`,
    /// `concurrency`, `niah`, `reasoning`, `structured`, `hardware`.
    pub const ENGINE: &str = "CRUCIBLE_ENGINE";
    /// Directory for the run log (`latest.log` + `run-<timestamp>.log`
    /// archives). Default: `data_dir()/crucible/logs`.
    pub const LOG_DIR: &str = "CRUCIBLE_LOG_DIR";
    /// Comma-separated context sizes for the v0.1.1 2D matrix
    /// (e.g. `0,8k,32k`).
    pub const MATRIX_CONTEXT: &str = "CRUCIBLE_MATRIX_CONTEXT";
}

/// Prompt mode (`--mode`): `short` (~50 tok) or `long` (padded to
/// `--tokens`). `base` is accepted as an alias for `short`
/// (`llmspeedtest2.py`'s naming).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// The fixed ~50-token prompt (`llmspeedtest2.py` called it `base`).
    #[default]
    #[value(alias = "base")]
    Short,
    /// A prompt padded to `--tokens`.
    Long,
}

impl Mode {
    /// The CLI/JSON label (`"short"` / `"long"`).
    pub fn label(self) -> &'static str {
        match self {
            Mode::Short => "short",
            Mode::Long => "long",
        }
    }
}

impl std::str::FromStr for Mode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "short" | "base" => Ok(Mode::Short),
            "long" => Ok(Mode::Long),
            other => Err(format!(
                "expected one of short|long (or base), got {other:?}"
            )),
        }
    }
}

/// Export format for `--export` (wired to `storage/export.rs` in Chunk 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "lowercase")]
pub enum ExportFormat {
    /// Zero-alloc JSON (CI/CD regression gating).
    #[default]
    Json,
    /// GitHub-Flavored Markdown tables.
    Md,
    /// Raw CSV (per-packet timestamps + ITL).
    Csv,
}

impl ExportFormat {
    /// The CLI/JSON label.
    pub fn label(self) -> &'static str {
        match self {
            ExportFormat::Json => "json",
            ExportFormat::Md => "md",
            ExportFormat::Csv => "csv",
        }
    }
}

impl std::str::FromStr for ExportFormat {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "json" => Ok(ExportFormat::Json),
            "md" => Ok(ExportFormat::Md),
            "csv" => Ok(ExportFormat::Csv),
            other => Err(format!("expected one of json|md|csv, got {other:?}")),
        }
    }
}

/// Which of the four benchmark engines (blueprint §5) a single run
/// orchestrates (plan Chunk 18: "a single command/run can trigger any
/// subset of engines A–D").
///
/// * `speed` — Engine A (single-stream TTFT/PP/TG/MTP);
/// * `concurrency` — Engine B (the ladder sweep);
/// * `niah` / `reasoning` / `structured` — Engine C1/C2/C3;
/// * `hardware` — Engine D (the 100 ms power/VRAM poller).
///
/// The default run is **everything except Engine D**: `speed`,
/// `concurrency`, `niah`, `reasoning`, and `structured` are on;
/// `hardware` is off (it must run on the machine with the GPU — remote
/// users get N/A — so it is opt-in; the user can enable it in Setup /
/// View 5 when they are on the GPU box).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EngineSelection {
    /// Engine A — Speed & Latency.
    pub speed: bool,
    /// Engine B — Concurrency & Saturation sweep.
    pub concurrency: bool,
    /// Engine C1 — Needle-in-a-Haystack.
    pub niah: bool,
    /// Engine C2 — Deterministic reasoning / code verification.
    pub reasoning: bool,
    /// Engine C3 — Structured output / JSON-grammar compliance.
    pub structured: bool,
    /// Engine D — Hardware & Energy profiler.
    pub hardware: bool,
}

impl Default for EngineSelection {
    fn default() -> Self {
        Self {
            speed: true,
            concurrency: true,
            niah: true,
            reasoning: true,
            structured: true,
            // Engine D (energy) is off by default: it must run on the
            // machine with the GPU, and is opt-in for GPU-box users.
            hardware: false,
        }
    }
}

impl EngineSelection {
    /// Build a selection from a list of engine names (case-insensitive).
    ///
    /// Accepted names: `speed`/`a`, `concurrency`/`b`, `niah`/`needle`/`c1`,
    /// `reasoning`/`c2`, `structured`/`c3`, `hardware`/`energy`/`d`. The
    /// list is the **complete** selection: only the named engines are on
    /// (so `--engine niah` runs exactly NIAH, not the default five).
    /// Returns the selection plus any unknown names, so the caller can
    /// surface a precise error. (The built-in default — everything except
    /// D — applies only when no engines are named at all.)
    pub fn from_names(names: impl IntoIterator<Item = impl AsRef<str>>) -> (Self, Vec<String>) {
        let mut sel = Self {
            speed: false,
            concurrency: false,
            niah: false,
            reasoning: false,
            structured: false,
            hardware: false,
        };
        let mut unknown = Vec::new();
        for name in names {
            match name.as_ref().to_ascii_lowercase().as_str() {
                "speed" | "a" => sel.speed = true,
                "concurrency" | "b" => sel.concurrency = true,
                "niah" | "needle" | "c1" => sel.niah = true,
                "reasoning" | "c2" => sel.reasoning = true,
                "structured" | "c3" => sel.structured = true,
                "hardware" | "energy" | "d" => sel.hardware = true,
                other => unknown.push(other.to_string()),
            }
        }
        (sel, unknown)
    }

    /// `true` when no engine is selected.
    pub fn is_empty(&self) -> bool {
        !self.speed
            && !self.concurrency
            && !self.niah
            && !self.reasoning
            && !self.structured
            && !self.hardware
    }

    /// The number of selected engines.
    pub fn count(&self) -> usize {
        [
            self.speed,
            self.concurrency,
            self.niah,
            self.reasoning,
            self.structured,
            self.hardware,
        ]
        .iter()
        .filter(|&&b| b)
        .count()
    }

    /// The display labels of the enabled engines, in A/B/C1/C2/C3/D order
    /// (for logs and the headless "running selected engines" line).
    pub fn iter_labels(&self) -> impl Iterator<Item = &'static str> {
        [
            (self.speed, "A (speed)"),
            (self.concurrency, "B (concurrency)"),
            (self.niah, "C1 (niah)"),
            (self.reasoning, "C2 (reasoning)"),
            (self.structured, "C3 (structured)"),
            (self.hardware, "D (hardware)"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, label)| label)
    }
}

/// The `clap`-derived CLI surface: the superset of both Python prototype
/// CLIs plus `--tui` (Chunk 8), `--config`, and `--export` (Chunk 13).
#[derive(Debug, Clone, Parser)]
#[command(
    name = "crucible-llm",
    version,
    about = "High-performance terminal LLM benchmark & inference profiler"
)]
pub struct Cli {
    /// Endpoint URL (bare host, base URL, or full completions path).
    #[arg(long, default_value = DEFAULT_URL)]
    pub url: String,
    /// Model name.
    #[arg(long, default_value = DEFAULT_MODEL)]
    pub model: String,
    /// Prompt mode: `short` (~50 tok) or `long` (padded to `--tokens`).
    /// `base` is accepted as an alias for `short`.
    #[arg(long, default_value_t = Mode::Short, value_enum)]
    pub mode: Mode,
    /// Target prompt tokens for `long` mode.
    #[arg(long, default_value_t = DEFAULT_TOKENS)]
    pub tokens: u32,
    /// Number of runs (headless iterations).
    #[arg(long, default_value_t = 1)]
    pub iterations: u32,
    /// API key sent as a `Bearer` token.
    #[arg(long)]
    pub api_key: Option<String>,
    /// Connection/idle-read timeout in seconds.
    #[arg(long, default_value_t = DEFAULT_TIMEOUT_SECS)]
    pub timeout: u64,
    /// Prepend a unique random prefix to bypass the server KV cache.
    #[arg(long)]
    pub nocache: bool,
    /// Emit the result as JSON on stdout (prototype `--json` field set).
    /// Also forces headless mode (the TUI is the default, Chunk 20).
    #[arg(long)]
    pub json: bool,
    /// Show per-run chunk detail.
    #[arg(long)]
    pub verbose: bool,
    /// Force-disable ANSI colors (default: on only when stdout is a TTY).
    #[arg(long)]
    pub no_color: bool,
    /// Path to an HF `tokenizer.json` for exact prompt token counts
    /// (falls back to `chars/4`, flagged `estimated`, when absent).
    #[arg(long)]
    pub tokenizer: Option<PathBuf>,
    /// The interactive ratatui dashboard is the **default** mode (Chunk
    /// 20) — a bare `crucible-llm` launches it. This flag is kept for
    /// compatibility: it forces the TUI (e.g. on a non-TTY, where the
    /// run would otherwise fall back to headless).
    #[arg(long)]
    pub tui: bool,
    /// Run the classic headless one-shot benchmark instead of the TUI
    /// (Chunk 20): the prototype's result box(es) and/or `--json` on
    /// stdout.
    #[arg(long)]
    pub headless: bool,
    /// Print the startup banner and exit (Chunk 20: the bare-invocation
    /// banner path, demoted to an explicit flag).
    #[arg(long, hide = true)]
    pub banner: bool,
    /// Export format (`json` | `md` | `csv`); wired in Chunk 13.
    #[arg(long, value_enum)]
    pub export: Option<ExportFormat>,
    /// Optional export destination path (with `--export`).
    #[arg(long)]
    pub export_path: Option<PathBuf>,
    /// Config file path (default: `~/.config/crucible/config.json`).
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Concurrency ladder as a comma-separated list of stream counts
    /// (e.g. `--ladder 1,2,3,4,8,12,16,24,32`); Engine B (Chunk 18).
    #[arg(long, value_name = "CSV")]
    pub ladder: Option<String>,
    /// Select which engines a run orchestrates (repeatable; Chunk 18):
    /// `speed` (A), `concurrency` (B), `niah` (C1), `reasoning` (C2),
    /// `structured` (C3), `hardware` (D).
    #[arg(long, value_name = "ENGINE")]
    pub engine: Vec<String>,
    /// Disable the hardware/energy telemetry poller (Engine D; on by
    /// default — it degrades to N/A on a driverless host).
    #[arg(long)]
    pub no_hardware: bool,
    /// Directory for the run log (`latest.log` + `run-<timestamp>.log`
    /// archives). Default: `~/.local/share/crucible/logs`.
    #[arg(long)]
    pub log_dir: Option<PathBuf>,
    /// Context sizes for the v0.1.1 2D concurrency × context matrix, as a
    /// comma-separated list with optional `k`/`m` suffixes
    /// (e.g. `--matrix-context 0,8k,32k`). `0` means "the configured
    /// prompt as-is"; more than one value enables the matrix sweep.
    #[arg(long, value_name = "CSV")]
    pub matrix_context: Option<String>,
}

/// The resolved runtime configuration — what the headless, TUI, and export
/// paths all consume.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub url: String,
    pub model: String,
    pub mode: Mode,
    pub tokens: u32,
    pub iterations: u32,
    pub api_key: Option<String>,
    /// Seconds.
    pub timeout: u64,
    pub nocache: bool,
    pub json: bool,
    pub verbose: bool,
    pub no_color: bool,
    pub tokenizer: Option<PathBuf>,
    pub tui: bool,
    /// Headless mode (Chunk 20): the classic one-shot benchmark instead
    /// of the TUI. `--json` implies it at dispatch time; this field
    /// carries the explicit `--headless` marker (env/file layered).
    pub headless: bool,
    /// Print the startup banner and exit (Chunk 20; CLI-only, hidden
    /// flag — the demoted bare-invocation banner path).
    pub banner: bool,
    pub export: Option<ExportFormat>,
    pub export_path: Option<PathBuf>,
    /// The concurrency ladder Engine B sweeps (Chunk 18; default
    /// `1→2→3→4→8→12→16→24→32` — granular at the low end where home
    /// users operate).
    pub ladder: Vec<usize>,
    /// Enable the hardware/energy telemetry poller (Engine D, Chunk 17/18).
    /// On by default; a driverless host degrades to N/A, never a failure.
    pub hardware: bool,
    /// Which engines a single run orchestrates (Chunk 18).
    pub engines: EngineSelection,
    /// `true` when the URL was provided explicitly (CLI / env / config
    /// file) rather than falling back to the built-in default. Chunk 20:
    /// a bare `crucible-llm` with no target launches the TUI (the
    /// default mode) — it no longer prints the banner.
    #[serde(skip)]
    pub target_explicit: bool,
    /// The run-log directory (`latest.log` + `run-<timestamp>.log`
    /// archives). `None` → the default `data_dir()/crucible/logs`.
    pub log_dir: Option<PathBuf>,
    /// The v0.1.1 2D matrix context axis (target prompt tokens per
    /// sweep leg). Default `[0, 8000, 32000]`; `0` = the configured
    /// prompt as-is. More than one value makes Engine B run the full
    /// concurrency × context matrix.
    pub matrix_contexts: Vec<u32>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            url: DEFAULT_URL.to_string(),
            model: DEFAULT_MODEL.to_string(),
            mode: Mode::Short,
            tokens: DEFAULT_TOKENS,
            iterations: 1,
            api_key: None,
            timeout: DEFAULT_TIMEOUT_SECS,
            nocache: false,
            json: false,
            verbose: false,
            no_color: false,
            tokenizer: None,
            tui: false,
            headless: false,
            banner: false,
            export: None,
            export_path: None,
            ladder: DEFAULT_LADDER.to_vec(),
            hardware: true,
            engines: EngineSelection::default(),
            target_explicit: false,
            log_dir: None,
            matrix_contexts: DEFAULT_MATRIX_CONTEXTS.to_vec(),
        }
    }
}

impl Config {
    /// Parse `std::env::args()` and layer env + config file over the
    /// defaults.
    pub fn from_cli() -> Result<Self, ConfigError> {
        Self::from_args(&env::args().collect::<Vec<_>>())
    }

    /// Parse an explicit argument list (first element: program name).
    ///
    /// Layers, in order: CLI > environment > config file > defaults.
    pub fn from_args(args: &[String]) -> Result<Self, ConfigError> {
        let matches = Cli::command().try_get_matches_from(args)?;
        let cli = Cli::from_arg_matches(&matches)?;
        let env = env::vars().collect::<Vec<_>>();
        Self::resolve(&cli, &matches, &env)
    }

    /// Full layering over an explicit env slice (pure; used by tests).
    ///
    /// The config file path is `--config` > `CRUCIBLE_CONFIG` > the
    /// platform default; a missing file is fine (defaults apply).
    pub fn resolve(
        cli: &Cli,
        matches: &ArgMatches,
        env: &[(String, String)],
    ) -> Result<Self, ConfigError> {
        let cfg_path = cli
            .config
            .clone()
            .or_else(|| env_get(env, env_vars::CONFIG).map(PathBuf::from))
            .or_else(default_config_path);
        let file = match &cfg_path {
            Some(p) => load_config_file(p)?,
            None => None,
        };
        layer(cli, matches, env, file.as_ref())
    }
}

/// The JSON config file (layer 3). Every field is optional — only what the
/// file specifies overrides the defaults. `Serialize` supports the TUI
/// Config view's `F2` save (Chunk 18) writing the file back out.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ConfigFile {
    pub url: Option<String>,
    pub model: Option<String>,
    pub mode: Option<Mode>,
    pub tokens: Option<u32>,
    pub iterations: Option<u32>,
    pub api_key: Option<String>,
    pub timeout: Option<u64>,
    pub nocache: Option<bool>,
    pub json: Option<bool>,
    pub verbose: Option<bool>,
    pub no_color: Option<bool>,
    pub tokenizer: Option<PathBuf>,
    pub tui: Option<bool>,
    pub headless: Option<bool>,
    pub export: Option<ExportFormat>,
    pub export_path: Option<PathBuf>,
    /// The concurrency ladder (Chunk 18).
    pub ladder: Option<Vec<usize>>,
    /// Enable the hardware/energy telemetry poller (Chunk 18).
    pub hardware: Option<bool>,
    /// Which engines a run orchestrates (Chunk 18).
    pub engines: Option<EngineSelection>,
    /// The run-log directory (default `data_dir()/crucible/logs`).
    pub log_dir: Option<PathBuf>,
    /// The v0.1.1 2D matrix context axis (target prompt tokens).
    pub matrix_contexts: Option<Vec<u32>>,
}

impl ConfigFile {
    /// A [`Config`] representing this file's entries (absent fields keep
    /// the built-in defaults; a `url` entry makes the target explicit).
    pub fn to_config(&self) -> Config {
        let mut c = Config::default();
        if let Some(v) = &self.url {
            c.url = v.clone();
        }
        if let Some(v) = &self.model {
            c.model = v.clone();
        }
        if let Some(v) = self.mode {
            c.mode = v;
        }
        if let Some(v) = self.tokens {
            c.tokens = v;
        }
        if let Some(v) = self.iterations {
            c.iterations = v;
        }
        if let Some(v) = &self.api_key {
            c.api_key = Some(v.clone());
        }
        if let Some(v) = self.timeout {
            c.timeout = v;
        }
        if let Some(v) = self.nocache {
            c.nocache = v;
        }
        if let Some(v) = self.json {
            c.json = v;
        }
        if let Some(v) = self.verbose {
            c.verbose = v;
        }
        if let Some(v) = self.no_color {
            c.no_color = v;
        }
        if let Some(v) = &self.tokenizer {
            c.tokenizer = Some(v.clone());
        }
        if let Some(v) = self.tui {
            c.tui = v;
        }
        if let Some(v) = self.headless {
            c.headless = v;
        }
        if let Some(v) = self.export {
            c.export = Some(v);
        }
        if let Some(v) = &self.export_path {
            c.export_path = Some(v.clone());
        }
        if let Some(v) = &self.ladder {
            c.ladder = v.clone();
        }
        if let Some(v) = self.hardware {
            c.hardware = v;
        }
        if let Some(v) = self.engines {
            c.engines = v;
        }
        if let Some(v) = &self.log_dir {
            c.log_dir = Some(v.clone());
        }
        if let Some(v) = &self.matrix_contexts {
            c.matrix_contexts = v.clone();
        }
        c.target_explicit = self.url.is_some();
        c
    }
}

/// Load the JSON config file at `path`.
///
/// A missing file is `Ok(None)` (the file is optional); an unreadable or
/// unparseable file is an error.
pub fn load_config_file(path: &Path) -> Result<Option<ConfigFile>, FileError> {
    match std::fs::read_to_string(path) {
        Ok(raw) => serde_json::from_str(&raw)
            .map(Some)
            .map_err(|e| FileError::Parse {
                path: path.to_path_buf(),
                reason: e.to_string(),
            }),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(FileError::Io {
            path: path.to_path_buf(),
            source: e,
        }),
    }
}

/// Layer the three sources into one [`Config`] (pure; unit-testable).
///
/// Per field: explicit CLI value > env var > config file > built-in
/// default. `target_explicit` is set when the URL came from any explicit
/// source.
pub fn layer(
    cli: &Cli,
    matches: &ArgMatches,
    env: &[(String, String)],
    file: Option<&ConfigFile>,
) -> Result<Config, ConfigError> {
    let explicit =
        |id: &str| matches.value_source(id) == Some(clap::parser::ValueSource::CommandLine);
    let env_get = |key: &str| env_get(env, key);

    // ── url ──
    let url = if explicit("url") {
        Some(cli.url.clone())
    } else {
        env_get(env_vars::URL).or_else(|| file.and_then(|f| f.url.clone()))
    }
    .unwrap_or_else(|| cli.url.clone());
    let target_explicit = explicit("url")
        || env_get(env_vars::URL).is_some()
        || file.is_some_and(|f| f.url.is_some());

    // ── model ──
    let model = if explicit("model") {
        Some(cli.model.clone())
    } else {
        env_get(env_vars::MODEL).or_else(|| file.and_then(|f| f.model.clone()))
    }
    .unwrap_or_else(|| cli.model.clone());

    // ── mode ──
    let mode = if explicit("mode") {
        Some(cli.mode)
    } else {
        env_get(env_vars::MODE)
            .map(|s| s.parse::<Mode>())
            .transpose()
            .map_err(|e| ConfigError::InvalidEnv {
                var: env_vars::MODE.to_string(),
                value: env_get(env_vars::MODE).unwrap_or_default(),
                reason: e,
            })?
            .or_else(|| file.and_then(|f| f.mode))
    }
    .unwrap_or(cli.mode);

    // ── numeric fields ──
    let tokens = if explicit("tokens") {
        Some(cli.tokens)
    } else {
        env_get(env_vars::TOKENS)
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| file.and_then(|f| f.tokens))
    }
    .unwrap_or(cli.tokens);
    if let Some(raw) = env_get(env_vars::TOKENS) {
        if raw.parse::<u32>().is_err() {
            return Err(ConfigError::InvalidEnv {
                var: env_vars::TOKENS.to_string(),
                value: raw,
                reason: "not a valid u32".to_string(),
            });
        }
    }

    let iterations = if explicit("iterations") {
        Some(cli.iterations)
    } else {
        env_get(env_vars::ITERATIONS)
            .and_then(|s| s.parse::<u32>().ok())
            .or_else(|| file.and_then(|f| f.iterations))
    }
    .unwrap_or(cli.iterations);
    if let Some(raw) = env_get(env_vars::ITERATIONS) {
        if raw.parse::<u32>().is_err() {
            return Err(ConfigError::InvalidEnv {
                var: env_vars::ITERATIONS.to_string(),
                value: raw,
                reason: "not a valid u32".to_string(),
            });
        }
    }

    let timeout = if explicit("timeout") {
        Some(cli.timeout)
    } else {
        env_get(env_vars::TIMEOUT)
            .and_then(|s| s.parse::<u64>().ok())
            .or_else(|| file.and_then(|f| f.timeout))
    }
    .unwrap_or(cli.timeout);
    if let Some(raw) = env_get(env_vars::TIMEOUT) {
        if raw.parse::<u64>().is_err() {
            return Err(ConfigError::InvalidEnv {
                var: env_vars::TIMEOUT.to_string(),
                value: raw,
                reason: "not a valid u64".to_string(),
            });
        }
    }

    // ── optional scalars (clap argument ids are the *field* names) ──
    let api_key = if explicit("api_key") {
        cli.api_key.clone()
    } else {
        env_get(env_vars::API_KEY).or_else(|| file.and_then(|f| f.api_key.clone()))
    };

    let tokenizer = if explicit("tokenizer") {
        cli.tokenizer.clone()
    } else {
        env_get(env_vars::TOKENIZER)
            .map(PathBuf::from)
            .or_else(|| file.and_then(|f| f.tokenizer.clone()))
    };

    // ── bool flags ──
    let bool_layer =
        |cli_v: bool, id: &str, var: &str, file_v: Option<bool>| -> Result<bool, ConfigError> {
            if explicit(id) {
                Ok(cli_v)
            } else {
                env_get(var)
                    .map(|s| parse_bool(&s))
                    .transpose()
                    .map_err(|e| ConfigError::InvalidEnv {
                        var: var.to_string(),
                        value: env_get(var).unwrap_or_default(),
                        reason: e,
                    })?
                    .or(file_v)
                    .map(Ok)
                    .unwrap_or(Ok(cli_v))
            }
        };
    let nocache = bool_layer(
        cli.nocache,
        "nocache",
        env_vars::NOCACHE,
        file.and_then(|f| f.nocache),
    )?;
    let json = bool_layer(cli.json, "json", env_vars::JSON, file.and_then(|f| f.json))?;
    let verbose = bool_layer(
        cli.verbose,
        "verbose",
        env_vars::VERBOSE,
        file.and_then(|f| f.verbose),
    )?;
    let no_color = bool_layer(
        cli.no_color,
        "no_color",
        env_vars::NO_COLOR,
        file.and_then(|f| f.no_color),
    )?;
    let tui = bool_layer(cli.tui, "tui", env_vars::TUI, file.and_then(|f| f.tui))?;
    let headless = bool_layer(
        cli.headless,
        "headless",
        env_vars::HEADLESS,
        file.and_then(|f| f.headless),
    )?;
    // `--banner` is a CLI-only display marker (hidden flag): it never
    // layers from env/file.
    let banner = cli.banner;

    // ── export (Chunk 13) ──
    let export = if explicit("export") {
        cli.export
    } else {
        env_get("CRUCIBLE_EXPORT")
            .map(|s| s.parse::<ExportFormat>())
            .transpose()
            .map_err(|e| ConfigError::InvalidEnv {
                var: "CRUCIBLE_EXPORT".to_string(),
                value: env_get("CRUCIBLE_EXPORT").unwrap_or_default(),
                reason: e,
            })?
            .or_else(|| file.and_then(|f| f.export))
    };
    let export_path = if explicit("export_path") {
        cli.export_path.clone()
    } else {
        env_get("CRUCIBLE_EXPORT_PATH")
            .map(PathBuf::from)
            .or_else(|| file.and_then(|f| f.export_path.clone()))
    };

    // ── run-log directory ──
    let log_dir = if explicit("log_dir") {
        cli.log_dir.clone()
    } else {
        env_get(env_vars::LOG_DIR)
            .map(PathBuf::from)
            .or_else(|| file.and_then(|f| f.log_dir.clone()))
    };

    // ── v0.1.1 2D matrix context axis ──
    let matrix_contexts = if explicit("matrix_context") {
        cli.matrix_context
            .as_deref()
            .and_then(parse_context_list)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| DEFAULT_MATRIX_CONTEXTS.to_vec())
    } else {
        env_get(env_vars::MATRIX_CONTEXT)
            .and_then(|s| parse_context_list(&s))
            .or_else(|| file.and_then(|f| f.matrix_contexts.clone()))
            .unwrap_or_else(|| DEFAULT_MATRIX_CONTEXTS.to_vec())
    };

    // ── concurrency ladder (Chunk 18) ──
    let ladder = if explicit("ladder") {
        cli.ladder
            .as_deref()
            .and_then(parse_ladder)
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| DEFAULT_LADDER.to_vec())
    } else {
        env_get(env_vars::LADDER)
            .and_then(|s| parse_ladder(&s))
            .or_else(|| file.and_then(|f| f.ladder.clone()))
            .unwrap_or_else(|| DEFAULT_LADDER.to_vec())
    };

    // ── hardware / energy telemetry (Chunk 18) ──
    // `--no-hardware` is a flag (false by default), so its mere presence
    // disables the poller; otherwise env > file > the on-by-default value.
    let hardware = if cli.no_hardware {
        false
    } else {
        env_get(env_vars::HARDWARE)
            .map(|s| parse_bool(&s))
            .transpose()
            .map_err(|e| ConfigError::InvalidEnv {
                var: env_vars::HARDWARE.to_string(),
                value: env_get(env_vars::HARDWARE).unwrap_or_default(),
                reason: e,
            })?
            .or(file.and_then(|f| f.hardware))
            .unwrap_or(true)
    };

    // ── engine selection (Chunk 18) ──
    let engines = if !cli.engine.is_empty() {
        selection_from_names(&cli.engine)?
    } else if let Some(raw) = env_get(env_vars::ENGINE) {
        let names: Vec<String> = raw
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if names.is_empty() {
            file.and_then(|f| f.engines).unwrap_or_default()
        } else {
            selection_from_names(&names)?
        }
    } else {
        file.and_then(|f| f.engines).unwrap_or_default()
    };

    Ok(Config {
        url,
        model,
        mode,
        tokens,
        iterations,
        api_key,
        timeout,
        nocache,
        json,
        verbose,
        no_color,
        tokenizer,
        tui,
        headless,
        banner,
        export,
        export_path,
        ladder,
        hardware,
        engines,
        target_explicit,
        log_dir,
        matrix_contexts,
    })
}

/// Parse a comma-separated ladder (`"1,2,4,8"`) into stream counts.
/// `None` when any entry is not a valid `usize`.
pub fn parse_ladder(s: &str) -> Option<Vec<usize>> {
    let v: Vec<usize> = s
        .split(',')
        .map(|p| p.trim().parse::<usize>().ok())
        .collect::<Option<Vec<_>>>()?;
    Some(v)
}

/// Parse a comma-separated context-size list (v0.1.1 2D matrix) such as
/// `"0,8k,32k"` into target prompt token counts.
///
/// Each entry is a plain number (`0` = the configured prompt as-is) or a
/// number with a `k` (×1000) / `m` (×1,000,000) suffix (case-insensitive).
/// `None` when any entry is unparseable.
pub fn parse_context_list(s: &str) -> Option<Vec<u32>> {
    let v: Vec<u32> = s
        .split(',')
        .map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let (digits, mult) =
                if let Some(k) = p.strip_suffix(['k', 'K']).map(|d| d.parse::<u32>().ok()) {
                    (k?, 1_000u32)
                } else if let Some(m) = p.strip_suffix(['m', 'M']).map(|d| d.parse::<u32>().ok()) {
                    (m?, 1_000_000u32)
                } else {
                    (p.parse::<u32>().ok()?, 1u32)
                };
            digits.checked_mul(mult)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(v)
}

/// Build an [`EngineSelection`] from engine names, erroring on any unknown
/// name (so a typo in `--engine` is surfaced, not silently dropped).
fn selection_from_names(names: &[String]) -> Result<EngineSelection, ConfigError> {
    let (sel, unknown) = EngineSelection::from_names(names);
    if unknown.is_empty() {
        Ok(sel)
    } else {
        Err(ConfigError::InvalidEnv {
            var: "--engine".to_string(),
            value: unknown.join(","),
            reason: "unknown engine name(s)".to_string(),
        })
    }
}

/// `true`/`false` (case-insensitive), `1`/`0`, or `yes`/`no`.
fn parse_bool(s: &str) -> Result<bool, String> {
    match s.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" => Ok(true),
        "0" | "false" | "no" => Ok(false),
        other => Err(format!(
            "expected a boolean (true|false|1|0|yes|no), got {other:?}"
        )),
    }
}

/// Look up `key` in an env slice.
fn env_get(env: &[(String, String)], key: &str) -> Option<String> {
    env.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
}

/// The platform data dir: `dirs::data_dir()/crucible` (Linux
/// `~/.local/share/crucible`, Windows `%APPDATA%\crucible`).
pub fn data_dir() -> PathBuf {
    dirs::data_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("crucible")
}

/// The platform config dir: `dirs::config_dir()/crucible` (Linux
/// `~/.config/crucible`).
pub fn config_dir() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("crucible")
}

/// The default config file path: `config_dir()/config.json` (`None` when
/// the platform exposes no config dir).
pub fn default_config_path() -> Option<PathBuf> {
    dirs::config_dir().map(|d| d.join("crucible").join(CONFIG_FILE_NAME))
}

/// Configuration errors (CLI parsing, env layering, config file).
#[derive(Debug, Error)]
pub enum ConfigError {
    /// A CLI parse failure (bad flag, bad value).
    #[error("CLI error: {0}")]
    Clap(#[from] clap::Error),
    /// An environment variable was set but unparseable.
    #[error("invalid environment variable {var}={value:?}: {reason}")]
    InvalidEnv {
        var: String,
        value: String,
        reason: String,
    },
    /// The config file could not be read/parsed.
    #[error(transparent)]
    File(#[from] FileError),
}

/// Config-file errors (a missing file is *not* an error).
#[derive(Debug, Error)]
pub enum FileError {
    #[error("failed to read config file {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse config file {path}: {reason}")]
    Parse { path: PathBuf, reason: String },
}

#[cfg(test)]
mod tests {
    use super::*;

    fn matches_from(args: &[&str]) -> ArgMatches {
        Cli::command()
            .try_get_matches_from(args.iter().map(|s| s.to_string()))
            .unwrap()
    }

    fn cli_from(args: &[&str]) -> Cli {
        Cli::try_parse_from(args.iter().map(|s| s.to_string())).unwrap()
    }

    /// Layer with no env and no file.
    ///
    /// Goes through [`layer`] directly (not [`Config::resolve`]) so the
    /// platform-default config file on a developer's machine can never
    /// contaminate the assertions (a real `~/.config/crucible/config.json`
    /// with a `url`/`ladder` entry made these tests machine-dependent).
    fn resolve_bare(cli_args: &[&str]) -> Config {
        let cli = cli_from(cli_args);
        let matches = matches_from(cli_args);
        layer(&cli, &matches, &[], None).unwrap()
    }

    #[test]
    fn defaults_match_python_prototype() {
        let cfg = Config::default();
        assert_eq!(cfg.url, DEFAULT_URL);
        assert_eq!(cfg.model, "default");
        assert_eq!(cfg.mode, Mode::Short);
        assert_eq!(cfg.tokens, 10000);
        assert_eq!(cfg.iterations, 1);
        assert_eq!(cfg.timeout, 120);
        assert!(!cfg.nocache);
        assert!(!cfg.json);
        assert!(!cfg.verbose);
        assert!(!cfg.no_color);
        assert!(!cfg.tui);
        assert!(!cfg.headless);
        assert!(!cfg.banner);
        assert!(cfg.api_key.is_none());
        assert!(cfg.tokenizer.is_none());
        assert!(cfg.export.is_none());
        assert!(!cfg.target_explicit);
    }

    #[test]
    fn cli_parses_full_flag_set() {
        let cfg = resolve_bare(&[
            "crucible-llm",
            "--url",
            "http://localhost:8000/v1",
            "--model",
            "qwen3",
            "--mode",
            "long",
            "--tokens",
            "4000",
            "--iterations",
            "3",
            "--api-key",
            "sk-test",
            "--timeout",
            "30",
            "--nocache",
            "--json",
            "--verbose",
            "--no-color",
            "--tokenizer",
            "/tmp/tok.json",
            "--export",
            "csv",
            "--export-path",
            "/tmp/out.csv",
        ]);
        assert_eq!(cfg.url, "http://localhost:8000/v1");
        assert_eq!(cfg.model, "qwen3");
        assert_eq!(cfg.mode, Mode::Long);
        assert_eq!(cfg.tokens, 4000);
        assert_eq!(cfg.iterations, 3);
        assert_eq!(cfg.api_key.as_deref(), Some("sk-test"));
        assert_eq!(cfg.timeout, 30);
        assert!(cfg.nocache);
        assert!(cfg.json);
        assert!(cfg.verbose);
        assert!(cfg.no_color);
        assert_eq!(cfg.tokenizer.as_deref(), Some(Path::new("/tmp/tok.json")));
        assert_eq!(cfg.export, Some(ExportFormat::Csv));
        assert_eq!(cfg.export_path.as_deref(), Some(Path::new("/tmp/out.csv")));
        assert!(cfg.target_explicit);
    }

    #[test]
    fn mode_alias_base_maps_to_short() {
        let cfg = resolve_bare(&["crucible-llm", "--mode", "base"]);
        assert_eq!(cfg.mode, Mode::Short);
    }

    #[test]
    fn invalid_mode_is_rejected() {
        let r = Cli::try_parse_from(["crucible-llm", "--mode", "medium"]);
        assert!(r.is_err());
    }

    #[test]
    fn invalid_export_format_is_rejected() {
        let r = Cli::try_parse_from(["crucible-llm", "--export", "yaml"]);
        assert!(r.is_err());
    }

    #[test]
    fn target_explicit_only_when_url_is_given() {
        // Bare invocation: built-in default URL, banner path.
        assert!(!resolve_bare(&["crucible-llm"]).target_explicit);
        // Other flags alone do not make the target explicit.
        assert!(!resolve_bare(&["crucible-llm", "--model", "m"]).target_explicit);
        // An explicit URL does.
        assert!(resolve_bare(&["crucible-llm", "--url", "http://x"]).target_explicit);
    }

    fn file(json: &str) -> ConfigFile {
        serde_json::from_str(json).unwrap()
    }

    #[test]
    fn config_file_fills_unspecified_fields() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let f = file(r#"{"url": "http://file:1/v1", "model": "file-model", "tokens": 777}"#);
        let cfg = layer(&cli, &matches, &[], Some(&f)).unwrap();
        assert_eq!(cfg.url, "http://file:1/v1");
        assert_eq!(cfg.model, "file-model");
        assert_eq!(cfg.tokens, 777);
        assert!(cfg.target_explicit);
        // Untouched fields keep the defaults.
        assert_eq!(cfg.iterations, 1);
        assert_eq!(cfg.timeout, 120);
    }

    #[test]
    fn env_overrides_config_file() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let f = file(r#"{"url": "http://file:1/v1", "model": "file-model"}"#);
        let env = vec![
            (env_vars::URL.to_string(), "http://env:2/v1".to_string()),
            (env_vars::NOCACHE.to_string(), "1".to_string()),
            (env_vars::MODE.to_string(), "long".to_string()),
        ];
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.url, "http://env:2/v1");
        assert_eq!(cfg.model, "file-model");
        assert!(cfg.nocache);
        assert_eq!(cfg.mode, Mode::Long);
    }

    #[test]
    fn cli_overrides_env_and_file() {
        let cli = cli_from(&[
            "crucible-llm",
            "--url",
            "http://cli:3/v1",
            "--tokens",
            "999",
        ]);
        let matches = matches_from(&[
            "crucible-llm",
            "--url",
            "http://cli:3/v1",
            "--tokens",
            "999",
        ]);
        let f = file(r#"{"url": "http://file:1/v1", "tokens": 777}"#);
        let env = vec![(env_vars::URL.to_string(), "http://env:2/v1".to_string())];
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.url, "http://cli:3/v1");
        assert_eq!(cfg.tokens, 999);
    }

    #[test]
    fn env_numbers_and_bools_parse() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![
            (env_vars::TOKENS.to_string(), "1234".to_string()),
            (env_vars::ITERATIONS.to_string(), "5".to_string()),
            (env_vars::TIMEOUT.to_string(), "42".to_string()),
            (env_vars::JSON.to_string(), "true".to_string()),
            (env_vars::TUI.to_string(), "yes".to_string()),
        ];
        let cfg = layer(&cli, &matches, &env, None).unwrap();
        assert_eq!(cfg.tokens, 1234);
        assert_eq!(cfg.iterations, 5);
        assert_eq!(cfg.timeout, 42);
        assert!(cfg.json);
        assert!(cfg.tui);
    }

    // ── Chunk 20: headless marker ────────────────────────────────────────

    #[test]
    fn headless_flag_off_by_default() {
        assert!(!resolve_bare(&["crucible-llm"]).headless);
        // `--json` alone does not set the `headless` field — it implies
        // headless at dispatch time, not in the resolved config.
        let cfg = resolve_bare(&["crucible-llm", "--json"]);
        assert!(cfg.json);
        assert!(!cfg.headless);
    }

    #[test]
    fn headless_flag_layers_from_cli_and_env() {
        assert!(resolve_bare(&["crucible-llm", "--headless"]).headless);
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(env_vars::HEADLESS.to_string(), "1".to_string())];
        assert!(layer(&cli, &matches, &env, None).unwrap().headless);
    }

    #[test]
    fn invalid_env_value_is_an_error() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(env_vars::TOKENS.to_string(), "abc".to_string())];
        let e = layer(&cli, &matches, &env, None).unwrap_err();
        assert!(matches!(e, ConfigError::InvalidEnv { .. }));
        assert!(e.to_string().contains("CRUCIBLE_TOKENS"));
    }

    #[test]
    fn invalid_env_bool_is_an_error() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(env_vars::NOCACHE.to_string(), "maybe".to_string())];
        assert!(layer(&cli, &matches, &env, None).is_err());
    }

    #[test]
    fn config_file_missing_is_ok() {
        match load_config_file(Path::new("/nonexistent/crucible/config.json")) {
            Ok(None) => {}
            other => panic!("expected Ok(None), got {other:?}"),
        }
    }

    #[test]
    fn config_file_bad_json_is_an_error() {
        let p = std::env::temp_dir().join(format!("crucible-bad-{}.json", std::process::id()));
        std::fs::write(&p, "this is not json").unwrap();
        let r = load_config_file(&p);
        std::fs::remove_file(&p).ok();
        assert!(matches!(r, Err(FileError::Parse { .. })));
    }

    #[test]
    fn config_file_denies_unknown_fields() {
        assert!(serde_json::from_str::<ConfigFile>(r#"{"url": "http://x", "bogus": 1}"#).is_err());
    }

    #[test]
    fn config_file_round_trips_all_fields() {
        let p = std::env::temp_dir().join(format!("crucible-cfg-{}.json", std::process::id()));
        std::fs::write(
            &p,
            r#"{"url":"http://f:1/v1","model":"m","mode":"long","tokens":100,"iterations":2,
                "api_key":"k","timeout":9,"nocache":true,"json":true,"verbose":true,
                "no_color":true,"tokenizer":"/t.json","tui":false,"headless":false,
                "export":"md","export_path":"/o.md"}"#,
        )
        .unwrap();
        let f = load_config_file(&p).unwrap().unwrap();
        let c = f.to_config();
        std::fs::remove_file(&p).ok();
        assert_eq!(c.url, "http://f:1/v1");
        assert_eq!(c.model, "m");
        assert_eq!(c.mode, Mode::Long);
        assert_eq!(c.tokens, 100);
        assert_eq!(c.iterations, 2);
        assert_eq!(c.api_key.as_deref(), Some("k"));
        assert_eq!(c.timeout, 9);
        assert!(c.nocache);
        assert!(c.json);
        assert!(c.verbose);
        assert!(c.no_color);
        assert_eq!(c.tokenizer.as_deref(), Some(Path::new("/t.json")));
        assert!(!c.tui);
        assert!(!c.headless);
        assert_eq!(c.export, Some(ExportFormat::Md));
        assert_eq!(c.export_path.as_deref(), Some(Path::new("/o.md")));
        assert!(c.target_explicit);
    }

    #[test]
    fn data_and_config_dirs_end_in_crucible() {
        assert_eq!(
            data_dir().file_name(),
            Some(std::ffi::OsStr::new("crucible"))
        );
        assert_eq!(
            config_dir().file_name(),
            Some(std::ffi::OsStr::new("crucible"))
        );
    }

    #[test]
    fn help_text_lists_the_python_parity_flags() {
        let help = Cli::command().render_help().to_string();
        for flag in [
            "--url",
            "--model",
            "--mode",
            "--tokens",
            "--iterations",
            "--api-key",
            "--timeout",
            "--nocache",
            "--json",
            "--verbose",
            "--no-color",
            "--tokenizer",
            "--tui",
            "--headless",
            "--export",
            "--config",
            "--ladder",
            "--engine",
            "--no-hardware",
            "--matrix-context",
        ] {
            assert!(help.contains(flag), "help missing {flag}");
        }
    }

    // ── Chunk 18: engine selection, ladder, hardware ─────────────────────

    #[test]
    fn engine_selection_default_is_everything_except_hardware() {
        // FIX 4: the default run selects A, B, C1, C2, C3 — Energy (D)
        // is off (it must run on the GPU box and is opt-in).
        let e = EngineSelection::default();
        assert!(e.speed, "Engine A on by default");
        assert!(e.concurrency, "Engine B on by default");
        assert!(e.niah, "Engine C1 on by default");
        assert!(e.reasoning, "Engine C2 on by default");
        assert!(e.structured, "Engine C3 on by default");
        assert!(
            !e.hardware,
            "Engine D off by default (opt-in on the GPU box)"
        );
        assert_eq!(e.count(), 5);
        assert!(!e.is_empty());
    }

    #[test]
    fn engine_selection_parses_names_and_aliases() {
        let (e, unknown) =
            EngineSelection::from_names(["speed", "B", "niah", "C2", "structured", "D"]);
        assert!(unknown.is_empty(), "no unknown names: {unknown:?}");
        assert!(e.speed);
        assert!(e.concurrency); // "B"
        assert!(e.niah);
        assert!(e.reasoning); // "C2"
        assert!(e.structured);
        assert!(e.hardware); // "D"
        assert_eq!(e.count(), 6);
    }

    #[test]
    fn engine_selection_reports_unknown_names() {
        let (e, unknown) = EngineSelection::from_names(["speed", "bogus"]);
        assert_eq!(unknown, vec!["bogus".to_string()]);
        // The known name still applies.
        assert!(e.speed);
    }

    #[test]
    fn parse_ladder_splits_and_trims() {
        assert_eq!(parse_ladder("1,2,4,8"), Some(vec![1, 2, 4, 8]));
        assert_eq!(parse_ladder(" 16 , 32 ,64 "), Some(vec![16, 32, 64]));
        assert_eq!(parse_ladder("1,2,x"), None);
        // An empty string is not a valid ladder entry (the caller falls
        // back to the default ladder).
        assert_eq!(parse_ladder(""), None);
    }

    // ── v0.1.1: 2D matrix context axis ─────────────────────────────────

    #[test]
    fn parse_context_list_handles_k_and_m_suffixes() {
        assert_eq!(parse_context_list("0,8k,32k"), Some(vec![0, 8000, 32000]));
        assert_eq!(parse_context_list(" 4k , 128K "), Some(vec![4000, 128_000]));
        assert_eq!(parse_context_list("1m"), Some(vec![1_000_000]));
        assert_eq!(parse_context_list("512"), Some(vec![512]));
        assert_eq!(parse_context_list("0"), Some(vec![0]));
        // Invalid entries are rejected (the caller falls back to the
        // default axis).
        assert_eq!(parse_context_list("8x"), None);
        assert_eq!(parse_context_list("1,2,x"), None);
        assert_eq!(parse_context_list(""), None);
    }

    #[test]
    fn default_matrix_contexts_match_the_blueprint() {
        let cfg = resolve_bare(&["crucible-llm"]);
        assert_eq!(cfg.matrix_contexts, vec![0, 8000, 32000]);
    }

    #[test]
    fn cli_matrix_context_overrides_default() {
        let cfg = resolve_bare(&["crucible-llm", "--matrix-context", "0,4k,32k"]);
        assert_eq!(cfg.matrix_contexts, vec![0, 4000, 32000]);
    }

    #[test]
    fn env_and_file_layer_the_matrix_contexts() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        // env wins over file.
        let env = vec![(env_vars::MATRIX_CONTEXT.to_string(), "0,2k".to_string())];
        let f = file(r#"{"matrix_contexts": [0, 999]}"#);
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.matrix_contexts, vec![0, 2000]);
        // file only.
        let cfg = layer(&cli, &matches, &[], Some(&f)).unwrap();
        assert_eq!(cfg.matrix_contexts, vec![0, 999]);
    }

    #[test]
    fn cli_ladder_overrides_default() {
        let cfg = resolve_bare(&["crucible-llm", "--ladder", "1,4,16"]);
        assert_eq!(cfg.ladder, vec![1, 4, 16]);
    }

    #[test]
    fn default_ladder_matches_the_blueprint() {
        let cfg = resolve_bare(&["crucible-llm"]);
        // FIX 3: the new default ladder — granular at the low end.
        assert_eq!(cfg.ladder, vec![1, 2, 3, 4, 8, 12, 16, 24, 32]);
        assert!(cfg.hardware, "hardware on by default");
    }

    #[test]
    fn no_hardware_flag_disables_the_poller() {
        let cfg = resolve_bare(&["crucible-llm", "--no-hardware"]);
        assert!(!cfg.hardware);
    }

    #[test]
    fn cli_engine_selection_layers() {
        let cfg = resolve_bare(&[
            "crucible-llm",
            "--engine",
            "speed",
            "--engine",
            "concurrency",
            "--engine",
            "niah",
        ]);
        assert!(cfg.engines.speed);
        assert!(cfg.engines.concurrency);
        assert!(cfg.engines.niah);
        assert!(!cfg.engines.reasoning);
        assert!(!cfg.engines.structured);
    }

    #[test]
    fn unknown_cli_engine_is_an_error() {
        let r = Cli::try_parse_from(["crucible-llm", "--engine", "warp-drive"]);
        assert!(r.is_ok(), "clap accepts the raw value");
        // The layering step is where the unknown name is rejected.
        let cli = r.unwrap();
        let matches = matches_from(&["crucible-llm", "--engine", "warp-drive"]);
        let e = layer(&cli, &matches, &[], None).unwrap_err();
        assert!(matches!(e, ConfigError::InvalidEnv { .. }));
        assert!(e.to_string().contains("warp-drive"));
    }

    #[test]
    fn env_engine_selection_layers() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(
            env_vars::ENGINE.to_string(),
            "reasoning,structured".to_string(),
        )];
        let cfg = layer(&cli, &matches, &env, None).unwrap();
        assert!(cfg.engines.reasoning);
        assert!(cfg.engines.structured);
        // A named engine list is the *complete* selection: un-named
        // engines stay off (so the user can run exactly what they asked
        // for).
        assert!(!cfg.engines.speed, "un-named engines stay off");
        assert!(!cfg.engines.hardware, "un-named engines stay off");
    }

    #[test]
    fn config_file_carries_ladder_hardware_and_engines() {
        let p = std::env::temp_dir().join(format!("crucible-cfg18-{}.json", std::process::id()));
        std::fs::write(
            &p,
            r#"{"url":"http://f:1/v1","ladder":[1,8,32],"hardware":false,
                "engines":{"speed":true,"concurrency":true,"niah":true,
                           "reasoning":false,"structured":true,"hardware":false}}"#,
        )
        .unwrap();
        let f = load_config_file(&p).unwrap().unwrap();
        let c = f.to_config();
        std::fs::remove_file(&p).ok();
        assert_eq!(c.ladder, vec![1, 8, 32]);
        assert!(!c.hardware);
        assert!(c.engines.concurrency);
        assert!(c.engines.niah);
        assert!(c.engines.structured);
        assert!(!c.engines.reasoning);
        assert!(!c.engines.hardware);
    }
}
