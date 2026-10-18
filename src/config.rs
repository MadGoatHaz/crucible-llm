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

use crate::engines::capability::niah::{NIAH_DEPTHS, NIAH_MAX_GEN_TOKENS, NIAH_SIZES};
use crate::engines::capability::structured::STRUCTURED_MAX_GEN_TOKENS;
use crate::engines::concurrency::DEFAULT_LADDER;

/// Default endpoint.
pub const DEFAULT_URL: &str = "http://localhost:8000/v1";
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
    /// `true`/`false` — print the `[COMMUNITY-SUBMISSION]` matrix report
    /// on stdout after the run (auto-parsed by
    /// https://madgoathaz.github.io/llm-matrix/; implies headless).
    pub const MATRIX_REPORT: &str = "CRUCIBLE_MATRIX_REPORT";
    pub const CONFIG: &str = "CRUCIBLE_CONFIG";
    /// Comma-separated concurrency ladder (Chunk 18).
    pub const LADDER: &str = "CRUCIBLE_LADDER";
    /// `true`/`false` — enable the hardware/energy telemetry poller
    /// (Chunk 18).
    pub const HARDWARE: &str = "CRUCIBLE_HARDWARE";
    /// Comma-separated engine selection (Chunk 18): `speed`,
    /// `concurrency`, `niah`, `reasoning`, `structured`, `hardware`,
    /// `flatout`.
    pub const ENGINE: &str = "CRUCIBLE_ENGINE";
    /// Directory for the run log (`latest.log` + `run-<timestamp>.log`
    /// archives). Default: `data_dir()/crucible/logs`.
    pub const LOG_DIR: &str = "CRUCIBLE_LOG_DIR";
    /// Comma-separated context sizes for the v0.1.1 2D matrix
    /// (e.g. `0,8k,32k`).
    pub const MATRIX_CONTEXT: &str = "CRUCIBLE_MATRIX_CONTEXT";
    /// The TUI color theme (`cyberpunk` | `vampire` | `monochrome`).
    pub const THEME: &str = "CRUCIBLE_THEME";
    /// The electricity rate for the View 4 cost estimate ($/kWh).
    pub const RATE_KWH: &str = "CRUCIBLE_RATE_KWH";
    /// Comma-separated NIAH (C1) context sizes (e.g. `16k,64k,128k,256k,1M`).
    /// Overrides the default 2k→128k ladder (Issue #5).
    pub const NIAH_CONTEXTS: &str = "CRUCIBLE_NIAH_CONTEXTS";
    /// Comma-separated NIAH (C1) needle depths, percentages 0–100
    /// (e.g. `0,25,50,75,100`). Overrides the default 11-depth ladder
    /// (Issue #5).
    pub const NIAH_DEPTHS: &str = "CRUCIBLE_NIAH_DEPTHS";
    /// A single NIAH (C1) context size (e.g. `256k` / `1000000`) for the
    /// focused "count test" mode: run NIAH at exactly that size across the
    /// depth ladder (Issue #5 / RoPE·YaRN probing).
    pub const NIAH_CUSTOM: &str = "CRUCIBLE_NIAH_CUSTOM";
    /// The C1 (NIAH) per-cell generation budget in tokens (R3). Overrides
    /// the built-in [`NIAH_MAX_GEN_TOKENS`] hybrid-reasoning safety net.
    pub const NIAH_MAX_TOKENS: &str = "CRUCIBLE_NIAH_MAX_TOKENS";
    /// The C3 (structured) per-run generation budget in tokens (R3).
    /// Overrides the built-in [`STRUCTURED_MAX_GEN_TOKENS`] hybrid-reasoning
    /// safety net.
    pub const STRUCTURED_MAX_TOKENS: &str = "CRUCIBLE_STRUCTURED_MAX_TOKENS";
}

/// The default electricity rate for the View 4 cost estimate ($/kWh).
/// `0.16` is a representative US residential/commercial blended rate.
pub const DEFAULT_RATE_KWH: f64 = 0.16;

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

/// Which of the seven benchmark engines (blueprint §5) a single run
/// orchestrates (plan Chunk 18: "a single command/run can trigger any
/// subset of the engines").
///
/// * `speed` — Engine A (single-stream TTFT/PP/TG/MTP);
/// * `concurrency` — Engine B (the ladder sweep);
/// * `niah` / `reasoning` / `structured` — Engine C1/C2/C3;
/// * `hardware` — Engine D (the 100 ms power/VRAM poller);
/// * `flatout` — Engine F (the 60 s sustained max-speed finale).
///
/// The default run is **everything except Engine D (Energy)**: `speed`,
/// `concurrency`, `niah`, `reasoning`, `structured`, and `flatout` are
/// on; `hardware` is off (it must run on the machine with the GPU —
/// remote users get N/A — so it is opt-in; the user can enable it in
/// Setup / View 6 when they are on the GPU box).
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
    ///
    /// CF-39: this is the *selection* of the energy engine. The entry
    /// point keeps it consistent with the poller toggle
    /// ([`Config::hardware`]): when the operator did **not** pick engines
    /// explicitly, `hardware` is derived as `hardware && gpu_present` (see
    /// `main::resolve_engine_d`) — so `--no-hardware` and a driverless
    /// host both turn Engine D off. An explicit `--engine` selection is
    /// respected as-is.
    pub hardware: bool,
    /// Engine F — Flat Out (sustained max-speed, one continuous 60s
    /// stream). On by default (like A–C3); only Engine D (Energy) is
    /// opt-in.
    pub flatout: bool,
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
            // Engine F (Flat Out) is on by default (like A–C3): the
            // 60-second sustained maximum decode speed finale runs in
            // every default sequence. Only Engine D (Energy) is opt-in.
            flatout: true,
        }
    }
}

impl EngineSelection {
    /// Every engine on (A, B, C1, C2, C3, D, F). The `all` primitive:
    /// `--engine all`, `CRUCIBLE_ENGINE=all`, and the config-file string
    /// `"all"` all resolve to this.
    pub const fn all() -> Self {
        Self {
            speed: true,
            concurrency: true,
            niah: true,
            reasoning: true,
            structured: true,
            hardware: true,
            flatout: true,
        }
    }

    /// Build a selection from a list of engine names (case-insensitive).
    ///
    /// Accepted names: `all` (everything on), `speed`/`a`,
    /// `concurrency`/`b`, `niah`/`needle`/`c1`, `reasoning`/`c2`,
    /// `structured`/`c3`, `hardware`/`energy`/`d`, `flatout`/`f`. The
    /// list is the **complete** selection: only the named engines are on
    /// (so `--engine niah` runs exactly NIAH, not the default five);
    /// `all` turns everything on regardless of the other names.
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
            flatout: false,
        };
        let mut unknown = Vec::new();
        for name in names {
            match name.as_ref().to_ascii_lowercase().as_str() {
                // `all` is the select-everything primitive: it turns on all
                // seven engines (it composes with any other names, which
                // are then redundant).
                "all" => sel = Self::all(),
                "speed" | "a" => sel.speed = true,
                "concurrency" | "b" => sel.concurrency = true,
                "niah" | "needle" | "c1" => sel.niah = true,
                "reasoning" | "c2" => sel.reasoning = true,
                "structured" | "c3" => sel.structured = true,
                "hardware" | "energy" | "d" => sel.hardware = true,
                "flatout" | "flat" | "f" => sel.flatout = true,
                other => unknown.push(other.to_string()),
            }
        }
        (sel, unknown)
    }

    /// The canonical short names of the enabled engines, in A/B/C1/C2/C3/D/F
    /// order (`speed`, `concurrency`, `niah`, `reasoning`, `structured`,
    /// `hardware`, `flatout`). These are the exact tokens `from_names`
    /// accepts, so the list round-trips through `--engine` / the env var.
    pub fn selected_names(&self) -> Vec<&'static str> {
        [
            (self.speed, "speed"),
            (self.concurrency, "concurrency"),
            (self.niah, "niah"),
            (self.reasoning, "reasoning"),
            (self.structured, "structured"),
            (self.hardware, "hardware"),
            (self.flatout, "flatout"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, name)| name)
        .collect()
    }

    /// The JSON array of enabled engine names (e.g. `["speed","niah"]`),
    /// for the `benchmark_sessions.engines` traceability column.
    pub fn selected_names_json(&self) -> String {
        serde_json::to_string(&self.selected_names()).unwrap_or_else(|_| "[]".to_string())
    }

    /// `true` when no engine is selected.
    pub fn is_empty(&self) -> bool {
        !self.speed
            && !self.concurrency
            && !self.niah
            && !self.reasoning
            && !self.structured
            && !self.hardware
            && !self.flatout
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
            self.flatout,
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
            (self.flatout, "F (flat out)"),
        ]
        .into_iter()
        .filter(|(on, _)| *on)
        .map(|(_, label)| label)
    }
}

/// The `engines` entry as it appears in the JSON config file.
///
/// It accepts **two** shapes (the file is hand-editable, so both are
/// ergonomic):
///
/// * the **7-bool object** (the canonical form the TUI's `F2` save writes,
///   e.g. `{"speed": true, "hardware": false, …}`);
/// * the **string shorthand** `"all"` (every engine on) — and, as a bonus,
///   any single engine name (`"niah"`) resolves to just that engine.
///
/// A newtype with a hand-written `Deserialize` (via `serde_json::Value`)
/// is the only way to accept both shapes under the file's
/// `#[serde(deny_unknown_fields)]`: the field is always a *known* field
/// (`engines`), and the value is dispatched on its JSON type. `Serialize`
/// always emits the canonical 7-bool object, so a round-tripped file stays
/// in the explicit form the TUI expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineSelectionFile(EngineSelection);

impl EngineSelectionFile {
    /// Wrap an explicit [`EngineSelection`] (the TUI save path).
    #[must_use]
    pub fn explicit(sel: EngineSelection) -> Self {
        Self(sel)
    }

    /// The wrapped selection.
    #[must_use]
    pub fn into_selection(self) -> EngineSelection {
        self.0
    }
}

impl Serialize for EngineSelectionFile {
    // Always the canonical 7-bool object.
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.0.serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for EngineSelectionFile {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = serde_json::Value::deserialize(deserializer)?;
        match value {
            // String shorthand: `"all"` (or a single engine name).
            serde_json::Value::String(s) => {
                let (sel, unknown) = EngineSelection::from_names([s.as_str()]);
                if unknown.is_empty() {
                    Ok(Self(sel))
                } else {
                    Err(serde::de::Error::custom(format!(
                        "unknown engine selection {s:?} (use \"all\" or a 7-bool object)"
                    )))
                }
            }
            // The canonical 7-bool object.
            serde_json::Value::Object(_) => {
                let sel = serde_json::from_value(value).map_err(serde::de::Error::custom)?;
                Ok(Self(sel))
            }
            other => Err(serde::de::Error::custom(format!(
                "engines must be the string \"all\" or a 7-bool object, got {other:?}"
            ))),
        }
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
    /// Print the `[COMMUNITY-SUBMISSION]` matrix report on stdout after
    /// the run (auto-parsed by
    /// https://madgoathaz.github.io/llm-matrix/). Implies headless mode;
    /// the user fields (backend / quant / use case / notes) are left
    /// blank — the block is a copy-pasteable starting point. With
    /// `--json`, the JSON document prints first (the block's markers
    /// cannot appear inside it, so a pipe consumer slices on them).
    #[arg(long)]
    pub matrix_report: bool,
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
    /// `structured` (C3), `hardware` (D), `flatout` (F).
    #[arg(long, value_name = "ENGINE")]
    pub engine: Vec<String>,
    /// Disable the hardware/energy telemetry. CF-39: this turns off
    /// **both** the 100 ms poller *and* the Engine D (energy) selection
    /// (on by default — it degrades to N/A on a driverless host), so the
    /// two can never disagree.
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
    /// Context sizes for the NIAH (C1) matrix, as a comma-separated list
    /// with optional `k`/`M` suffixes (e.g. `--niah-contexts
    /// 16k,64k,128k,256k,500k,1M`). Overrides the default 2k→128k ladder
    /// (Issue #5). Every entry is a literal target token count (`0` is not
    /// allowed — unlike `--matrix-context`).
    #[arg(long, value_name = "CSV")]
    pub niah_contexts: Option<String>,
    /// Needle depths for the NIAH (C1) matrix, as a comma-separated list of
    /// percentages 0–100 (e.g. `--niah-depths 0,25,50,75,100`). Overrides
    /// the default 11-depth (0%…100% in 10% steps) ladder (Issue #5).
    #[arg(long, value_name = "CSV")]
    pub niah_depths: Option<String>,
    /// A single NIAH (C1) context size (e.g. `--niah-custom 256k` or
    /// `--niah-custom 1000000`) for the focused "count test" mode: run NIAH
    /// at exactly that size across the depth ladder — for probing a
    /// RoPE/YaRN-extended limit beyond the default 128k top (Issue #5).
    #[arg(long, value_name = "SIZE")]
    pub niah_custom: Option<String>,
    /// The C1 (NIAH) per-cell generation budget in tokens (R3). Overrides
    /// the built-in default (the hybrid-reasoning safety-net budget): a
    /// larger value gives a verbose reasoner room to finish its
    /// chain-of-thought *and* commit the answer; a tiny value force-tests
    /// the `truncated_by_reasoning` outcome.
    #[arg(long, value_name = "N")]
    pub niah_max_tokens: Option<u32>,
    /// The C3 (structured) per-run generation budget in tokens (R3).
    /// Overrides the built-in default (the hybrid-reasoning safety-net
    /// budget): a larger value gives a verbose reasoner room to finish its
    /// chain-of-thought *and* commit the JSON answer.
    #[arg(long, value_name = "N")]
    pub structured_max_tokens: Option<u32>,
    /// Disable the C1/C3 best-effort `disable_thinking` suppression
    /// (R3 calibration). The capability engines normally ask the server to
    /// suppress chain-of-thought (a speed optimization); a config-only
    /// server may silently drop that field, in which case the model thinks
    /// by default and the generation budget is the correctness guarantee.
    /// This flag makes the *client* stop requesting suppression, so the
    /// `truncated_by_reasoning` safety net can be force-tested live
    /// (tiny budget + thinking on → the model runs out mid-CoT).
    #[arg(long)]
    pub no_disable_thinking: bool,
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
    /// Print the `[COMMUNITY-SUBMISSION]` matrix report on stdout after
    /// the run (`--matrix-report` / `CRUCIBLE_MATRIX_REPORT` / config
    /// file). Implies headless mode at dispatch; the user fields are
    /// left blank (the block is a copy-pasteable starting point).
    pub matrix_report: bool,
    /// Print the startup banner and exit (Chunk 20; CLI-only, hidden
    /// flag — the demoted bare-invocation banner path).
    pub banner: bool,
    pub export: Option<ExportFormat>,
    pub export_path: Option<PathBuf>,
    /// The concurrency ladder Engine B sweeps (Chunk 18; default
    /// `1→2→3→4→8→12→16→24→32` — granular at the low end where home
    /// users operate).
    pub ladder: Vec<usize>,
    /// Enable the hardware/energy telemetry **poller** (the 100 ms sampler,
    /// Chunk 17/18). On by default; a driverless host degrades to N/A,
    /// never a failure.
    ///
    /// CF-39: this is the single master switch for hardware telemetry.
    /// `--no-hardware` disables **both** the poller *and* the Engine D
    /// (energy) selection: the entry point derives
    /// `engines.hardware = hardware && gpu_present` when the operator did
    /// not pick engines explicitly, so the two booleans can never disagree
    /// (a `--no-hardware` run can no longer select Engine D and starve it
    /// of data).
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
    /// The TUI color theme (`"cyberpunk"` | `"vampire"` |
    /// `"monochrome"`). Default `"cyberpunk"`.
    pub theme: String,
    /// `true` when the theme was provided explicitly (CLI / env / config
    /// file) rather than falling back to the built-in default. The TUI
    /// entry point shows the first-run theme picker only when this is
    /// `false` (no theme has ever been chosen).
    #[serde(skip)]
    pub theme_explicit: bool,
    /// `true` when the engine selection was provided explicitly (CLI
    /// `--engine` / env / config file) rather than the built-in default.
    /// The entry point applies the detection-based Engine D default (on
    /// when a GPU is present, off when not) *only* when the user did not
    /// pick engines themselves.
    #[serde(skip)]
    pub engines_explicit: bool,
    /// The electricity rate for the View 4 (GPU & Power) cost estimate,
    /// in $/kWh. Default [`DEFAULT_RATE_KWH`] ($0.16) — editable in the
    /// Config view (tab 6) and layered from env / config file.
    pub rate_per_kwh: f64,
    /// The NIAH (C1) context-size ladder (target prompt tokens per matrix
    /// row). Default [`NIAH_SIZES`] (2k→128k). Overridden by
    /// `--niah-contexts` / `CRUCIBLE_NIAH_CONTEXTS` / the config file
    /// (Issue #5). Sizes scale to 256k/500k/1M+ for RoPE/YaRN probing.
    pub niah_contexts: Vec<u32>,
    /// The NIAH (C1) needle-depth ladder (percentages 0–100, one per
    /// matrix column). Default [`NIAH_DEPTHS`] (0%…100% in 10% steps).
    /// Overridden by `--niah-depths` / `CRUCIBLE_NIAH_DEPTHS` / the config
    /// file (Issue #5).
    pub niah_depths: Vec<u8>,
    /// The NIAH (C1) focused "count test" mode: a single context size to run
    /// across the depth ladder (e.g. 256k for probing a RoPE/YaRN-extended
    /// limit). When `Some`, it takes precedence over `niah_contexts`
    /// (the matrix collapses to one row) (Issue #5).
    pub niah_custom: Option<u32>,
    /// The C1 (NIAH) per-cell generation budget in tokens (R3). Default
    /// [`NIAH_MAX_GEN_TOKENS`] — the hybrid-reasoning safety net (a
    /// verbose reasoner must be able to finish its chain-of-thought *and*
    /// commit the answer within the budget). Overridden by
    /// `--niah-max-tokens` / `CRUCIBLE_NIAH_MAX_TOKENS` / the config file.
    pub niah_max_tokens: u32,
    /// The C1/C3 best-effort `disable_thinking` suppression (R3). On by
    /// default (a speed optimization: the budget then rarely needs to
    /// carry a chain-of-thought). `--no-disable-thinking` turns it off so
    /// the model thinks by default — the force-test path for the
    /// `truncated_by_reasoning` safety net (a tiny budget then cuts the
    /// CoT off before any content is committed). CLI-only: this is a
    /// calibration tool, not a persisted preference.
    pub disable_thinking: bool,
    /// The C3 (structured) per-run generation budget in tokens (R3).
    /// Default [`STRUCTURED_MAX_GEN_TOKENS`] — the hybrid-reasoning safety
    /// net for the JSON answer. Overridden by `--structured-max-tokens` /
    /// `CRUCIBLE_STRUCTURED_MAX_TOKENS` / the config file.
    pub structured_max_tokens: u32,
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
            matrix_report: false,
            banner: false,
            export: None,
            export_path: None,
            ladder: DEFAULT_LADDER.to_vec(),
            hardware: true,
            engines: EngineSelection::default(),
            target_explicit: false,
            log_dir: None,
            matrix_contexts: DEFAULT_MATRIX_CONTEXTS.to_vec(),
            theme: "cyberpunk".to_string(),
            theme_explicit: false,
            engines_explicit: false,
            rate_per_kwh: DEFAULT_RATE_KWH,
            niah_contexts: NIAH_SIZES.to_vec(),
            niah_depths: NIAH_DEPTHS.to_vec(),
            niah_custom: None,
            niah_max_tokens: NIAH_MAX_GEN_TOKENS,
            structured_max_tokens: STRUCTURED_MAX_GEN_TOKENS,
            disable_thinking: true,
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
    pub matrix_report: Option<bool>,
    pub export: Option<ExportFormat>,
    pub export_path: Option<PathBuf>,
    /// The concurrency ladder (Chunk 18).
    pub ladder: Option<Vec<usize>>,
    /// Enable the hardware/energy telemetry poller (Chunk 18).
    pub hardware: Option<bool>,
    /// Which engines a run orchestrates (Chunk 18). Accepts the 7-bool
    /// object (the TUI's canonical form) or the string shorthand `"all"`.
    pub engines: Option<EngineSelectionFile>,
    /// The run-log directory (default `data_dir()/crucible/logs`).
    pub log_dir: Option<PathBuf>,
    /// The v0.1.1 2D matrix context axis (target prompt tokens).
    pub matrix_contexts: Option<Vec<u32>>,
    /// The TUI color theme (`"cyberpunk"` | `"vampire"` | `"monochrome"`).
    pub theme: Option<String>,
    /// The electricity rate for the View 4 cost estimate ($/kWh).
    pub rate_per_kwh: Option<f64>,
    /// The NIAH (C1) context-size ladder (Issue #5).
    pub niah_contexts: Option<Vec<u32>>,
    /// The NIAH (C1) needle-depth ladder (percentages 0–100, Issue #5).
    pub niah_depths: Option<Vec<u8>>,
    /// The NIAH (C1) focused single-size "count test" mode (Issue #5).
    pub niah_custom: Option<u32>,
    /// The C1 (NIAH) per-cell generation budget in tokens (R3).
    pub niah_max_tokens: Option<u32>,
    /// The C3 (structured) per-run generation budget in tokens (R3).
    pub structured_max_tokens: Option<u32>,
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
        if let Some(v) = self.matrix_report {
            c.matrix_report = v;
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
            c.engines = v.into_selection();
        }
        // A file-supplied engine selection is an explicit choice.
        c.engines_explicit = self.engines.is_some();
        if let Some(v) = &self.log_dir {
            c.log_dir = Some(v.clone());
        }
        if let Some(v) = &self.matrix_contexts {
            c.matrix_contexts = v.clone();
        }
        if let Some(v) = &self.theme {
            c.theme = v.clone();
            c.theme_explicit = true;
        }
        if let Some(v) = self.rate_per_kwh {
            c.rate_per_kwh = v.max(0.0);
        }
        // NIAH (C1) ladders (Issue #5): a file-supplied ladder overrides the
        // built-in default; a `niah_custom` of 0 is treated as absent.
        if let Some(v) = &self.niah_contexts {
            c.niah_contexts = v.clone();
        }
        if let Some(v) = &self.niah_depths {
            c.niah_depths = v.clone();
        }
        if let Some(v) = self.niah_custom.filter(|n| *n > 0) {
            c.niah_custom = Some(v);
        }
        // C1/C3 generation budgets (R3): a file value overrides the
        // built-in default; a zero value is treated as absent (the
        // resolution layer clamps any explicit zero to the 1-token
        // minimum).
        if let Some(v) = self.niah_max_tokens.filter(|n| *n > 0) {
            c.niah_max_tokens = v;
        }
        if let Some(v) = self.structured_max_tokens.filter(|n| *n > 0) {
            c.structured_max_tokens = v;
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
    let matrix_report = bool_layer(
        cli.matrix_report,
        "matrix_report",
        env_vars::MATRIX_REPORT,
        file.and_then(|f| f.matrix_report),
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

    // ── TUI theme ──
    // CLI has no `--theme` flag; layer env > file > the built-in default.
    // `theme_explicit` is set only when an explicit source named a theme
    // (a missing theme = first run → the TUI shows the theme picker).
    let (theme, theme_explicit) =
        match env_get(env_vars::THEME).or_else(|| file.and_then(|f| f.theme.clone())) {
            Some(raw) => (raw.to_ascii_lowercase(), true),
            None => ("cyberpunk".to_string(), false),
        };

    // ── electricity rate ($/kWh, the View 4 cost estimate) ──
    // env > file > the built-in default ($0.16). A negative rate is clamped
    // to zero (never a signed cost).
    let rate_per_kwh = env_get(env_vars::RATE_KWH)
        .and_then(|s| s.parse::<f64>().ok())
        .or_else(|| file.and_then(|f| f.rate_per_kwh))
        .unwrap_or(DEFAULT_RATE_KWH)
        .max(0.0);

    // ── NIAH (C1) ladders (Issue #5) ───────────────────────────────────
    // Per field: CLI > env > config file > built-in default. An invalid /
    // empty CLI or env list falls back to the next layer (then the
    // default), mirroring the `ladder` / `matrix_context` policy — a typo
    // never silently shrinks the matrix.
    let niah_contexts = if explicit("niah_contexts") {
        cli.niah_contexts
            .as_deref()
            .and_then(parse_niah_contexts)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| NIAH_SIZES.to_vec())
    } else {
        env_get(env_vars::NIAH_CONTEXTS)
            .and_then(|s| parse_niah_contexts(&s))
            .or_else(|| {
                file.and_then(|f| f.niah_contexts.clone())
                    .filter(|v| !v.is_empty())
            })
            .unwrap_or_else(|| NIAH_SIZES.to_vec())
    };

    let niah_depths = if explicit("niah_depths") {
        cli.niah_depths
            .as_deref()
            .and_then(parse_niah_depths)
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| NIAH_DEPTHS.to_vec())
    } else {
        env_get(env_vars::NIAH_DEPTHS)
            .and_then(|s| parse_niah_depths(&s))
            .or_else(|| {
                file.and_then(|f| f.niah_depths.clone())
                    .filter(|v| !v.is_empty())
            })
            .unwrap_or_else(|| NIAH_DEPTHS.to_vec())
    };

    // The focused "count test" single size: `Some` only when a valid
    // (non-zero) size is named in any layer.
    let niah_custom = if explicit("niah_custom") {
        cli.niah_custom
            .as_deref()
            .and_then(parse_size)
            .filter(|n| *n > 0)
    } else {
        env_get(env_vars::NIAH_CUSTOM)
            .and_then(|s| parse_size(&s))
            .or_else(|| file.and_then(|f| f.niah_custom))
            .filter(|n| *n > 0)
    };

    // ── C1 / C3 generation budgets (R3) ─────────────────────────────────
    // Per field: CLI > env > config file > the engine's built-in default
    // (the hybrid-reasoning safety-net budget). An invalid (non-numeric)
    // env value is a hard error (the `tokens` / `iterations` policy); a
    // zero value clamps to the 1-token minimum (the worker clamps too).
    let niah_max_tokens = if explicit("niah_max_tokens") {
        cli.niah_max_tokens
    } else {
        match env_get(env_vars::NIAH_MAX_TOKENS) {
            Some(raw) => Some(raw.parse::<u32>().map_err(|_| ConfigError::InvalidEnv {
                var: env_vars::NIAH_MAX_TOKENS.to_string(),
                value: raw,
                reason: "not a valid u32".to_string(),
            })?),
            None => file.and_then(|f| f.niah_max_tokens),
        }
    }
    .unwrap_or(NIAH_MAX_GEN_TOKENS)
    .max(1);

    let structured_max_tokens = if explicit("structured_max_tokens") {
        cli.structured_max_tokens
    } else {
        match env_get(env_vars::STRUCTURED_MAX_TOKENS) {
            Some(raw) => Some(raw.parse::<u32>().map_err(|_| ConfigError::InvalidEnv {
                var: env_vars::STRUCTURED_MAX_TOKENS.to_string(),
                value: raw,
                reason: "not a valid u32".to_string(),
            })?),
            None => file.and_then(|f| f.structured_max_tokens),
        }
    }
    .unwrap_or(STRUCTURED_MAX_GEN_TOKENS)
    .max(1);

    // ── C1/C3 thinking suppression (R3 calibration) ─────────────────────
    // `--no-disable-thinking` is a hard switch (like `--no-hardware`): the
    // flag's presence turns the best-effort suppression off, so the model
    // thinks by default and a tiny generation budget can force-test the
    // `truncated_by_reasoning` safety net. CLI-only (no env / file layer).
    let disable_thinking = !cli.no_disable_thinking;

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
    // `engines_explicit` records whether the selection came from an
    // explicit source (CLI / env / file) rather than the built-in
    // default — the entry point applies the detection-based Engine D
    // default only when the user did not pick engines themselves.
    let (engines, engines_explicit) = if !cli.engine.is_empty() {
        (selection_from_names(&cli.engine)?, true)
    } else if let Some(raw) = env_get(env_vars::ENGINE) {
        let names: Vec<String> = raw
            .split(',')
            .map(|p| p.trim().to_string())
            .filter(|p| !p.is_empty())
            .collect();
        if names.is_empty() {
            (
                file.and_then(|f| f.engines)
                    .map(|f| f.into_selection())
                    .unwrap_or_default(),
                file.and_then(|f| f.engines).is_some(),
            )
        } else {
            (selection_from_names(&names)?, true)
        }
    } else {
        (
            file.and_then(|f| f.engines)
                .map(|f| f.into_selection())
                .unwrap_or_default(),
            file.and_then(|f| f.engines).is_some(),
        )
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
        matrix_report,
        banner,
        export,
        export_path,
        ladder,
        hardware,
        engines,
        target_explicit,
        log_dir,
        matrix_contexts,
        theme,
        theme_explicit,
        engines_explicit,
        rate_per_kwh,
        niah_contexts,
        niah_depths,
        niah_custom,
        niah_max_tokens,
        structured_max_tokens,
        disable_thinking,
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

/// Parse a single token size such as `"8k"`, `"1M"`, or `"512"` into a
/// token count.
///
/// Accepts a plain number, or a number with a `k` (×1000) / `M`
/// (×1,000,000) suffix (case-insensitive). `None` when unparseable or when
/// the multiplication overflows `u32` (a size beyond ~4.29 B tokens is not
/// representable — far beyond any practical context window).
pub fn parse_size(s: &str) -> Option<u32> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    if let Some(digits) = s.strip_suffix(['k', 'K']) {
        digits.parse::<u32>().ok()?.checked_mul(1_000)
    } else if let Some(digits) = s.strip_suffix(['m', 'M']) {
        digits.parse::<u32>().ok()?.checked_mul(1_000_000)
    } else {
        s.parse::<u32>().ok()
    }
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
            parse_size(p)
        })
        .collect::<Option<Vec<_>>>()?;
    Some(v)
}

/// Parse a comma-separated NIAH (C1) context-size ladder such as
/// `"16k,64k,128k,256k,500k,1M"` into target token counts (Issue #5).
///
/// Each entry is a plain number or a number with a `k` / `M` suffix
/// (case-insensitive), parsed via [`parse_size`]. Unlike
/// [`parse_context_list`], a `0` entry is **rejected** — a NIAH haystack
/// must be a positive token count (there is no "the configured prompt"
/// special case for C1). `None` when any entry is unparseable or zero, so
/// the caller falls back to the default ladder.
pub fn parse_niah_contexts(s: &str) -> Option<Vec<u32>> {
    let v: Vec<u32> = s
        .split(',')
        .map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let n = parse_size(p)?;
            (n > 0).then_some(n)
        })
        .collect::<Option<Vec<_>>>()?;
    if v.is_empty() {
        return None;
    }
    Some(v)
}

/// Parse a comma-separated NIAH (C1) needle-depth ladder such as
/// `"0,25,50,75,100"` into percentages (Issue #5).
///
/// Each entry is an integer percentage in `0..=100`. `None` when any entry
/// is unparseable or out of range, so the caller falls back to the default
/// 11-depth ladder.
pub fn parse_niah_depths(s: &str) -> Option<Vec<u8>> {
    let v: Vec<u8> = s
        .split(',')
        .map(|p| {
            let p = p.trim();
            if p.is_empty() {
                return None;
            }
            let n: u8 = p.parse().ok()?;
            (n <= 100).then_some(n)
        })
        .collect::<Option<Vec<_>>>()?;
    if v.is_empty() {
        return None;
    }
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

    // ── hardware / energy telemetry (CF-39) ──────────────────────────────

    #[test]
    fn hardware_poller_defaults_to_on() {
        // The poller is on by default (a driverless host degrades to N/A).
        assert!(Config::default().hardware);
        let cfg = resolve_bare(&["crucible-llm"]);
        assert!(cfg.hardware, "no `--no-hardware` → the poller stays on");
    }

    #[test]
    fn no_hardware_flag_beats_env() {
        // The `--no-hardware` flag is a hard disable: it is checked first
        // in `layer`, so it wins over an env `true`.
        let env = [(env_vars::HARDWARE.to_string(), "true".to_string())];
        let cli = cli_from(&["crucible-llm", "--no-hardware"]);
        let matches = matches_from(&["crucible-llm", "--no-hardware"]);
        let cfg = layer(&cli, &matches, &env, None).unwrap();
        assert!(
            !cfg.hardware,
            "`--no-hardware` disables the poller even with env=true"
        );
    }

    #[test]
    fn engine_d_selection_defaults_to_off_and_is_independent_of_the_poller() {
        // The two booleans have distinct defaults: the poller is on, the
        // Engine D *selection* is off (opt-in). CF-39 unifies them at the
        // entry point (`engines.hardware = hardware && gpu_present` when
        // not explicit) — the config layer keeps them separate so an
        // explicit `--engine hardware` is respected.
        let cfg = Config::default();
        assert!(cfg.hardware, "poller on by default");
        assert!(
            !cfg.engines.hardware,
            "Engine D selection off by default (opt-in)"
        );
    }

    // ── TUI theme ────────────────────────────────────────────────────────

    #[test]
    fn theme_defaults_to_cyberpunk_and_is_not_explicit() {
        let cfg = Config::default();
        assert_eq!(cfg.theme, "cyberpunk");
        assert!(!cfg.theme_explicit, "a bare default is a first run");
    }

    #[test]
    fn a_config_file_theme_is_explicit() {
        let file = ConfigFile {
            theme: Some("vampire".to_string()),
            ..Default::default()
        };
        let cfg = file.to_config();
        assert_eq!(cfg.theme, "vampire");
        assert!(cfg.theme_explicit, "a file theme is an explicit choice");
    }

    #[test]
    fn a_missing_file_theme_is_a_first_run() {
        let file = ConfigFile {
            theme: None,
            ..Default::default()
        };
        let cfg = file.to_config();
        assert_eq!(cfg.theme, "cyberpunk");
        assert!(!cfg.theme_explicit);
    }

    #[test]
    fn env_theme_layer_wins_and_is_explicit() {
        let env = [(env_vars::THEME.to_string(), "monochrome".to_string())];
        let cfg = layer(
            &cli_from(&["crucible-llm"]),
            &matches_from(&["crucible-llm"]),
            &env,
            None,
        )
        .expect("layer");
        assert_eq!(cfg.theme, "monochrome");
        assert!(cfg.theme_explicit);
    }

    #[test]
    fn config_file_round_trips_the_theme() {
        let file = ConfigFile {
            theme: Some("vampire".to_string()),
            ..Default::default()
        };
        let json = serde_json::to_string(&file).expect("serialize");
        assert!(json.contains("\"theme\""), "theme persisted: {json}");
        let back: ConfigFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.theme, Some("vampire".to_string()));
    }

    // ── electricity rate ($/kWh, View 4 cost estimate) ───────────────────

    #[test]
    fn rate_per_kwh_defaults_to_sixteen_cents() {
        let cfg = Config::default();
        assert!((cfg.rate_per_kwh - DEFAULT_RATE_KWH).abs() < 1e-9);
        assert!((cfg.rate_per_kwh - 0.16).abs() < 1e-9);
    }

    #[test]
    fn rate_per_kwh_layers_from_env_and_file() {
        // env wins.
        let env = [(env_vars::RATE_KWH.to_string(), "0.30".to_string())];
        let cfg = layer(
            &cli_from(&["crucible-llm"]),
            &matches_from(&["crucible-llm"]),
            &env,
            None,
        )
        .unwrap();
        assert!((cfg.rate_per_kwh - 0.30).abs() < 1e-9);
        // a file value (no env) also layers in.
        let file = ConfigFile {
            rate_per_kwh: Some(0.12),
            ..Default::default()
        };
        let cfg = layer(
            &cli_from(&["crucible-llm"]),
            &matches_from(&["crucible-llm"]),
            &[],
            Some(&file),
        )
        .unwrap();
        assert!((cfg.rate_per_kwh - 0.12).abs() < 1e-9);
    }

    #[test]
    fn a_negative_rate_per_kwh_clamps_to_zero() {
        let file = ConfigFile {
            rate_per_kwh: Some(-5.0),
            ..Default::default()
        };
        let cfg = file.to_config();
        assert_eq!(cfg.rate_per_kwh, 0.0);
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
        assert!(!cfg.matrix_report);
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

    // ── Matrix Report marker (`--matrix-report`) ─────────────────────────

    #[test]
    fn matrix_report_flag_off_by_default() {
        assert!(!resolve_bare(&["crucible-llm"]).matrix_report);
        // The flag parses on the CLI surface.
        let cli = Cli::try_parse_from(["crucible-llm", "--matrix-report"]).unwrap();
        assert!(cli.matrix_report);
        assert!(resolve_bare(&["crucible-llm", "--matrix-report"]).matrix_report);
    }

    #[test]
    fn matrix_report_layering_cli_over_env_over_file() {
        let cli_from_args = |args: &[&str]| (cli_from(args), matches_from(args));
        let file_true = ConfigFile {
            matrix_report: Some(true),
            ..Default::default()
        };
        let file_false = ConfigFile {
            matrix_report: Some(false),
            ..Default::default()
        };
        let env_true = vec![(env_vars::MATRIX_REPORT.to_string(), "true".to_string())];
        let env_false = vec![(env_vars::MATRIX_REPORT.to_string(), "false".to_string())];
        // CLI beats env + file (both pointing the other way).
        let (cli, matches) = cli_from_args(&["crucible-llm", "--matrix-report"]);
        assert!(
            layer(&cli, &matches, &env_false, Some(&file_false))
                .unwrap()
                .matrix_report
        );
        // Env beats file (no CLI flag).
        let (cli, matches) = cli_from_args(&["crucible-llm"]);
        assert!(
            layer(&cli, &matches, &env_true, Some(&file_false))
                .unwrap()
                .matrix_report
        );
        // Env (false) beats file (true) — the file only fills what the
        // layers above leave unset.
        let (cli, matches) = cli_from_args(&["crucible-llm"]);
        assert!(
            !layer(&cli, &matches, &env_false, Some(&file_true))
                .unwrap()
                .matrix_report
        );
        // No env: the file beats the default.
        let (cli, matches) = cli_from_args(&["crucible-llm"]);
        assert!(
            layer(&cli, &matches, &[], Some(&file_true))
                .unwrap()
                .matrix_report
        );
        // The default is false.
        assert!(!Config::default().matrix_report);
    }

    #[test]
    fn config_file_matrix_report_roundtrip() {
        let f = ConfigFile {
            matrix_report: Some(true),
            ..Default::default()
        };
        assert!(f.to_config().matrix_report);
        // A file without the key keeps the default (false).
        let f = ConfigFile::default();
        assert!(!f.to_config().matrix_report);
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
            "--niah-contexts",
            "--niah-depths",
            "--niah-custom",
            "--niah-max-tokens",
            "--structured-max-tokens",
        ] {
            assert!(help.contains(flag), "help missing {flag}");
        }
    }

    // ── Chunk 18: engine selection, ladder, hardware ─────────────────────

    #[test]
    fn engine_selection_default_is_everything_except_hardware() {
        // The default run selects A, B, C1, C2, C3, F — everything
        // except Energy (D), which must run on the GPU box and is
        // opt-in.
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
        assert!(
            e.flatout,
            "Engine F on by default (sustained max-speed finale)"
        );
        assert_eq!(e.count(), 6);
        assert!(!e.is_empty());
    }

    #[test]
    fn engine_selection_parses_names_and_aliases() {
        let (e, unknown) =
            EngineSelection::from_names(["speed", "B", "niah", "C2", "structured", "D", "F"]);
        assert!(unknown.is_empty(), "no unknown names: {unknown:?}");
        assert!(e.speed);
        assert!(e.concurrency); // "B"
        assert!(e.niah);
        assert!(e.reasoning); // "C2"
        assert!(e.structured);
        assert!(e.hardware); // "D"
        assert!(e.flatout); // "F"
        assert_eq!(e.count(), 7);
    }

    #[test]
    fn engine_selection_reports_unknown_names() {
        let (e, unknown) = EngineSelection::from_names(["speed", "bogus"]);
        assert_eq!(unknown, vec!["bogus".to_string()]);
        // The known name still applies.
        assert!(e.speed);
    }

    // ── the `all` primitive ──────────────────────────────────────────────

    #[test]
    fn engine_selection_all_turns_everything_on() {
        let e = EngineSelection::all();
        assert!(e.speed);
        assert!(e.concurrency);
        assert!(e.niah);
        assert!(e.reasoning);
        assert!(e.structured);
        assert!(e.hardware);
        assert!(e.flatout);
        assert_eq!(e.count(), 7);
        assert!(!e.is_empty());
    }

    #[test]
    fn from_names_accepts_all() {
        let (e, unknown) = EngineSelection::from_names(["all"]);
        assert!(unknown.is_empty(), "all is a known name: {unknown:?}");
        assert_eq!(e, EngineSelection::all());
        assert_eq!(e.count(), 7);
    }

    #[test]
    fn from_names_all_composes_with_other_names() {
        // `all` plus a redundant name still yields all seven.
        let (e, unknown) = EngineSelection::from_names(["all", "niah"]);
        assert!(unknown.is_empty());
        assert_eq!(e, EngineSelection::all());
    }

    #[test]
    fn cli_engine_all_selects_everything() {
        // `--engine all` resolves to all seven engines.
        let cfg = resolve_bare(&["crucible-llm", "--engine", "all"]);
        assert_eq!(cfg.engines, EngineSelection::all());
        assert!(cfg.engines_explicit);
    }

    #[test]
    fn env_engine_all_selects_everything() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(env_vars::ENGINE.to_string(), "all".to_string())];
        let cfg = layer(&cli, &matches, &env, None).unwrap();
        assert_eq!(cfg.engines, EngineSelection::all());
    }

    #[test]
    fn config_file_accepts_engines_string_all() {
        // The string shorthand `"all"` in the config file selects every
        // engine (the object form is the canonical/round-tripped one).
        let f = file(r#"{"engines": "all"}"#);
        let c = f.to_config();
        assert_eq!(c.engines, EngineSelection::all());
        assert!(c.engines_explicit);
    }

    #[test]
    fn config_file_engines_object_still_accepted() {
        // The 7-bool object form (the TUI's canonical save shape) keeps
        // working alongside the string shorthand.
        let f = file(
            r#"{"engines":{"speed":true,"concurrency":false,"niah":true,
                "reasoning":false,"structured":false,"hardware":false,"flatout":false}}"#,
        );
        let c = f.to_config();
        assert!(c.engines.speed);
        assert!(!c.engines.concurrency);
        assert!(c.engines.niah);
        assert!(!c.engines.hardware);
        assert!(!c.engines.flatout);
    }

    #[test]
    fn config_file_engines_unknown_string_is_an_error() {
        // A string that is neither `all` nor a known engine name is
        // rejected at parse time (surfaced as a file parse error).
        assert!(
            serde_json::from_str::<ConfigFile>(r#"{"engines": "warp-drive"}"#).is_err(),
            "unknown engine string must be rejected"
        );
    }

    #[test]
    fn config_file_engines_round_trips_as_the_object() {
        // Saving a selection always serializes the canonical 7-bool object
        // (never the string), so a hand-`"all"` file becomes explicit on
        // the next `F2` save.
        let f = ConfigFile {
            engines: Some(EngineSelectionFile::explicit(EngineSelection::all())),
            ..Default::default()
        };
        let json = serde_json::to_string(&f).unwrap();
        assert!(json.contains("\"speed\":true"), "object form: {json}");
        assert!(
            !json.contains("\"engines\":\"all\""),
            "not the string: {json}"
        );
        let back: ConfigFile = serde_json::from_str(&json).unwrap();
        assert_eq!(
            back.engines,
            Some(EngineSelectionFile::explicit(EngineSelection::all()))
        );
    }

    #[test]
    fn selected_names_and_json_round_trip() {
        let e = EngineSelection {
            speed: true,
            concurrency: false,
            niah: true,
            reasoning: false,
            structured: false,
            hardware: false,
            flatout: true,
        };
        assert_eq!(e.selected_names(), vec!["speed", "niah", "flatout"]);
        assert_eq!(e.selected_names_json(), r#"["speed","niah","flatout"]"#);
        // The JSON list re-parses to the same selection.
        let back = EngineSelection::from_names(
            serde_json::from_str::<Vec<String>>(&e.selected_names_json()).unwrap(),
        )
        .0;
        assert_eq!(back, e);
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
                           "reasoning":false,"structured":true,"hardware":false,
                           "flatout":false}}"#,
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
        assert!(!c.engines.flatout);
    }

    // ── NIAH (C1) ladders (Issue #5) ─────────────────────────────────

    #[test]
    fn niah_ladders_default_to_the_builtin_2k_to_128k() {
        let cfg = resolve_bare(&["crucible-llm"]);
        assert_eq!(
            cfg.niah_contexts,
            vec![2000, 4000, 8000, 16000, 32000, 64000, 128000]
        );
        assert_eq!(
            cfg.niah_depths,
            vec![0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        );
        assert_eq!(cfg.niah_custom, None);
    }

    #[test]
    fn cli_niah_contexts_overrides_the_default_ladder() {
        // k and M suffixes, beyond the old 128k ceiling.
        let cfg = resolve_bare(&["crucible-llm", "--niah-contexts", "16k,64k,128k,256k,1M"]);
        assert_eq!(
            cfg.niah_contexts,
            vec![16000, 64000, 128000, 256000, 1_000_000]
        );
    }

    #[test]
    fn cli_niah_depths_overrides_the_default_ladder() {
        let cfg = resolve_bare(&["crucible-llm", "--niah-depths", "0,25,50,75,100"]);
        assert_eq!(cfg.niah_depths, vec![0, 25, 50, 75, 100]);
    }

    #[test]
    fn cli_niah_custom_is_a_single_size() {
        let cfg = resolve_bare(&["crucible-llm", "--niah-custom", "256k"]);
        assert_eq!(cfg.niah_custom, Some(256000));
        // A plain number (no suffix) works too.
        let cfg = resolve_bare(&["crucible-llm", "--niah-custom", "1000000"]);
        assert_eq!(cfg.niah_custom, Some(1_000_000));
    }

    #[test]
    fn niah_ladders_layer_from_env_and_file() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        // env wins over file for the two lists; the file supplies the custom.
        let env = vec![
            (env_vars::NIAH_CONTEXTS.to_string(), "32k,128k".to_string()),
            (env_vars::NIAH_DEPTHS.to_string(), "0,50,100".to_string()),
        ];
        let f = file(r#"{"niah_contexts":[8000],"niah_depths":[50],"niah_custom":4000}"#);
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.niah_contexts, vec![32000, 128000]);
        assert_eq!(cfg.niah_depths, vec![0, 50, 100]);
        assert_eq!(cfg.niah_custom, Some(4000));
    }

    #[test]
    fn parse_niah_contexts_rejects_zero_and_bad_entries() {
        assert_eq!(
            parse_niah_contexts("16k,64k,1M"),
            Some(vec![16000, 64000, 1_000_000])
        );
        // 0 has no "configured prompt" meaning for C1 — it is rejected (the
        // caller falls back to the default ladder).
        assert_eq!(parse_niah_contexts("0,8k"), None);
        assert_eq!(parse_niah_contexts("8x,16k"), None);
        assert_eq!(parse_niah_contexts(""), None);
    }

    #[test]
    fn parse_niah_depths_rejects_out_of_range() {
        assert_eq!(parse_niah_depths("0,25,50,100"), Some(vec![0, 25, 50, 100]));
        assert_eq!(parse_niah_depths("0,101"), None, "depths are 0..=100");
        assert_eq!(parse_niah_depths("a,b"), None);
    }

    #[test]
    fn parse_size_handles_k_m_and_plain() {
        assert_eq!(parse_size("8k"), Some(8000));
        assert_eq!(parse_size("1M"), Some(1_000_000));
        assert_eq!(parse_size("256K"), Some(256000));
        assert_eq!(parse_size("512"), Some(512));
        assert_eq!(parse_size("8x"), None);
        assert_eq!(parse_size(""), None);
    }

    // ── C1 / C3 generation budgets (R3) ───────────────────────────────

    #[test]
    fn niah_and_structured_budgets_default_to_the_builtin_values() {
        // The built-in safety-net budgets (the engine constants) are the
        // default when no layer names a value.
        let cfg = resolve_bare(&["crucible-llm"]);
        assert_eq!(cfg.niah_max_tokens, NIAH_MAX_GEN_TOKENS);
        assert_eq!(cfg.structured_max_tokens, STRUCTURED_MAX_GEN_TOKENS);
        // `Config::default()` agrees with the layering (no env / file).
        assert_eq!(Config::default().niah_max_tokens, NIAH_MAX_GEN_TOKENS);
        assert_eq!(
            Config::default().structured_max_tokens,
            STRUCTURED_MAX_GEN_TOKENS
        );
    }

    #[test]
    fn cli_niah_max_tokens_overrides_the_default() {
        let cfg = resolve_bare(&["crucible-llm", "--niah-max-tokens", "512"]);
        assert_eq!(cfg.niah_max_tokens, 512);
        // The C3 budget is untouched.
        assert_eq!(cfg.structured_max_tokens, STRUCTURED_MAX_GEN_TOKENS);
    }

    #[test]
    fn cli_structured_max_tokens_overrides_the_default() {
        let cfg = resolve_bare(&["crucible-llm", "--structured-max-tokens", "1024"]);
        assert_eq!(cfg.structured_max_tokens, 1024);
        // The C1 budget is untouched.
        assert_eq!(cfg.niah_max_tokens, NIAH_MAX_GEN_TOKENS);
    }

    #[test]
    fn budgets_layer_from_env_and_file() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        // env wins over file.
        let env = vec![
            (env_vars::NIAH_MAX_TOKENS.to_string(), "777".to_string()),
            (
                env_vars::STRUCTURED_MAX_TOKENS.to_string(),
                "888".to_string(),
            ),
        ];
        let f = file(r#"{"niah_max_tokens": 111, "structured_max_tokens": 222}"#);
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.niah_max_tokens, 777);
        assert_eq!(cfg.structured_max_tokens, 888);
        // file only (no env).
        let cfg = layer(&cli, &matches, &[], Some(&f)).unwrap();
        assert_eq!(cfg.niah_max_tokens, 111);
        assert_eq!(cfg.structured_max_tokens, 222);
    }

    #[test]
    fn cli_budget_beats_env_and_file() {
        // The layering order (CLI > env > file > default) for a budget.
        let cli = cli_from(&["crucible-llm", "--niah-max-tokens", "32"]);
        let matches = matches_from(&["crucible-llm", "--niah-max-tokens", "32"]);
        let env = vec![(env_vars::NIAH_MAX_TOKENS.to_string(), "555".to_string())];
        let f = file(r#"{"niah_max_tokens": 111}"#);
        let cfg = layer(&cli, &matches, &env, Some(&f)).unwrap();
        assert_eq!(cfg.niah_max_tokens, 32, "the CLI value wins");
    }

    #[test]
    fn invalid_budget_env_value_is_an_error() {
        let cli = cli_from(&["crucible-llm"]);
        let matches = matches_from(&["crucible-llm"]);
        let env = vec![(env_vars::NIAH_MAX_TOKENS.to_string(), "abc".to_string())];
        let e = layer(&cli, &matches, &env, None).unwrap_err();
        assert!(matches!(e, ConfigError::InvalidEnv { .. }));
        assert!(e.to_string().contains("CRUCIBLE_NIAH_MAX_TOKENS"));
    }

    #[test]
    fn a_zero_budget_clamps_to_one() {
        // A zero budget is meaningless (a 0-token generation); the layer
        // clamps it to the 1-token minimum (the worker clamps too —
        // `max_tokens.max(1)`).
        let cfg = resolve_bare(&["crucible-llm", "--niah-max-tokens", "0"]);
        assert_eq!(cfg.niah_max_tokens, 1);
        let cfg = resolve_bare(&["crucible-llm", "--structured-max-tokens", "0"]);
        assert_eq!(cfg.structured_max_tokens, 1);
    }

    #[test]
    fn disable_thinking_is_on_by_default_and_the_flag_turns_it_off() {
        // The C1/C3 best-effort suppression is on by default (a speed
        // optimization); `--no-disable-thinking` is a hard off switch (the
        // R3 force-test path for `truncated_by_reasoning`).
        let cfg = resolve_bare(&["crucible-llm"]);
        assert!(cfg.disable_thinking, "suppression on by default");
        let cfg = resolve_bare(&["crucible-llm", "--no-disable-thinking"]);
        assert!(!cfg.disable_thinking, "the flag turns it off");
        assert!(Config::default().disable_thinking);
    }

    #[test]
    fn config_file_round_trips_the_budgets() {
        let f = ConfigFile {
            niah_max_tokens: Some(1536),
            structured_max_tokens: Some(3072),
            ..Default::default()
        };
        let json = serde_json::to_string(&f).expect("serialize");
        assert!(json.contains("\"niah_max_tokens\":1536"), "{json}");
        assert!(json.contains("\"structured_max_tokens\":3072"), "{json}");
        let back: ConfigFile = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back.niah_max_tokens, Some(1536));
        assert_eq!(back.structured_max_tokens, Some(3072));
        let c = back.to_config();
        assert_eq!(c.niah_max_tokens, 1536);
        assert_eq!(c.structured_max_tokens, 3072);
    }
}
