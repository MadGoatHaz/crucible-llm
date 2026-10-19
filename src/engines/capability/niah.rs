//! Engine C1 — Needle-In-A-Haystack (NIAH) Context Retention (plan Chunk
//! 15, blueprint §5 Engine C1).
//!
//! A *needle* — a unique, cryptographically random key-value fact — is
//! inserted at a chosen depth (0%…100%) of a synthetic long-context
//! *haystack* document scaled to a set of context sizes (the default ladder
//! is 2k/4k/8k/16k/32k/64k/128k tokens, but the ladder and depths are
//! **runtime-configurable** via `--niah-contexts` / `--niah-depths` and can
//! scale to 256k/500k/1M+; a single size can be probed with
//! `--niah-custom` — Issue #5). The
//! retrieval query asks the model to state the needle's value; two things
//! are measured simultaneously (blueprint §5 C1):
//!
//! 1. **Accuracy** — does the model retrieve the needle cleanly? (strict
//!    substring check on the 96-bit random value: a hit is retrieval, a
//!    miss is a failure/hallucination);
//! 2. **Prefill degradation** — the TTFT / prefill-throughput decay curve
//!    as context scales. A cell is *throttled* when its TTFT exceeds
//!    `PREFILL_THROTTLE_FACTOR`× the *linear* expectation anchored on the
//!    smallest context size at the same depth.
//!
//! The result is an N×M grid ([`NiahResult`]) that View 3 renders with
//! ANSI color-coding (blueprint §6 View 3): green = accurate + nominal
//! prefill, yellow = accurate + throttled prefill, red = retrieval failed.
//!
//! The TUI consumes the result through a lock-free [`NiahSlot`] (the same
//! `ArcSwap` double-buffer pattern as the Chunk 6 metrics snapshot): the
//! `n` key spawns a background runner that *publishes* the result, and the
//! render path only ever *reads* — measurement isolation intact
//! (blueprint §4).
//!
//! Measurement-isolation note: the live runner reuses the Chunk 5
//! `StreamWorker` (quanta `T0..Tn` live in the worker); this module only
//! takes deltas of those records, exactly as Engines A/C2 do.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arc_swap::ArcSwap;
use rand::Rng;
use tokio::sync::mpsc;

use crate::client::{join_worker, spawn_worker, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::prompt::{count_tokens, Tokenizer, BASE_SENTENCES, FILLER};
use crate::sse::Chunk;
use crate::storage::models::NeedleEvaluation;
use crate::timing::StreamTimestamps;

/// Bounded worker→engine channel capacity (same as Engine A / C2).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per cell (the default; overridable at runtime via
/// `--niah-max-tokens` / `CRUCIBLE_NIAH_MAX_TOKENS` / the config file — R3).
///
/// **Hybrid reasoning (Option C):** the *budget* is the correctness
/// guarantee. `disable_thinking` is best-effort — a speed optimization a
/// config-only server may silently drop — so a reasoning model that ignores
/// it would spend a tiny budget entirely on chain-of-thought and emit no
/// answer (the v0.1.4 false failure, C1 ≈ 5/77).
///
/// **R3 calibration (live, 2026-10-11, qwen3.8-27b finetune @
/// 192.168.51.163):** the token-efficient finetune used **0** CoT tokens
/// per cell (the vLLM server honors `enable_thinking=false`); forced to
/// think (`--no-disable-thinking`) it spent 32+ CoT on the 2k NIAH query
/// (the 32-token force-test budget was exhausted mid-CoT) and ~90 on a
/// step-by-step math probe. 2048 leaves **large headroom** above that: a
/// verbose reasoner (several× the finetune's ~100, or a vanilla qwen3.8
/// that used 256 CoT on an 87-token prompt) can spend ~2000 tokens on CoT
/// and still commit the ~15-token needle answer. On a
/// suppression-honoring server the answer arrives in ~15 tokens and the
/// ceiling is rarely hit, so the matrix stays fast — the budget is a
/// ceiling, not a target.
pub const NIAH_MAX_GEN_TOKENS: u32 = 2048;

/// The context token sizes of the N×M matrix (blueprint §5 C1:
/// 2k/4k/8k/16k/32k/64k/128k).
pub const NIAH_SIZES: [u32; 7] = [2_000, 4_000, 8_000, 16_000, 32_000, 64_000, 128_000];

/// The needle depths: 0%…100% in 10% steps (blueprint §5 C1).
pub const NIAH_DEPTHS: [u8; 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/// A cell whose TTFT exceeds `PREFILL_THROTTLE_FACTOR`× the linear
/// prefill expectation (anchored on the smallest context size at the same
/// depth) is classified as *throttled* (yellow) instead of nominal
/// (green).
pub const PREFILL_THROTTLE_FACTOR: f64 = 2.0;

/// Format a token count as a compact human label: `2k`, `128k`, `256k`,
/// `1M`, … — the **shared** size formatter for the C1 grid labels, the
/// headless run log, and the per-cell worker tags (Issue #3: the headless
/// log used to render `128000k`; this renders `128k`).
///
/// Sizes ≥ 1 000 000 use `M`; sizes ≥ 1 000 use `k`; smaller counts are
/// shown as-is. A whole number is shown with no decimal (`128k`, `1M`);
/// a fractional one gets a single decimal (`1.5k`, `1.2M`).
///
/// `128000 → "128k"`, `1000000 → "1M"`, `256000 → "256k"`,
/// `500000 → "500k"`, `1500 → "1.5k"`, `999 → "999"`.
#[must_use]
pub fn human_size(tokens: u32) -> String {
    if tokens >= 1_000_000 {
        let m = tokens as f64 / 1_000_000.0;
        if m.fract().abs() < 1e-9 {
            format!("{m:.0}M")
        } else {
            format!("{m:.1}M")
        }
    } else if tokens >= 1000 {
        let k = tokens as f64 / 1000.0;
        if k.fract().abs() < 1e-9 {
            format!("{k:.0}k")
        } else {
            format!("{k:.1}k")
        }
    } else {
        format!("{tokens}")
    }
}

/// The per-cell worker tag (C3): `C1:{size}-d{depth}` with the shared
/// [`human_size`] formatter (`C1:128k-d50` / `C1:1M-d50`).
///
/// The `(approx.)` qualifier is deliberately **absent** — in the v0.1.7
/// log it repeated on every one of the per-cell lines (114 hits). It is
/// qualified once instead: the run-start `[CONFIG]` line (C1) and the C1
/// start line (`sequence.rs::run_niah`).
#[must_use]
pub fn cell_tag(size: u32, depth: u8) -> String {
    format!("C1:{}-d{depth}", human_size(size))
}

/// Render a raw context size with the optional `(approx.)` qualifier — the
/// free-function form of [`NiahCell::context_label`] for call sites that have
/// a size (not a cell) plus an estimate flag.
///
/// **C1 token-honesty (Issue #4 / CF-46):** with an exact `--tokenizer` the
/// haystack is padded to the target and the label reads plain (`128k`);
/// without one, the size comes from the `chars/4` heuristic (the low-entropy
/// repeated-sentence haystack), so the label reads `"{size} (approx.)"` — the
/// same qualifier the TUI grid renders. Surfacing it in the headless run log
/// / summary keeps the headless path consistent with the TUI.
#[must_use]
pub fn context_size_label(tokens: u32, approximate: bool) -> String {
    let size = human_size(tokens);
    if approximate {
        format!("{size} (approx.)")
    } else {
        size
    }
}

// ── The needle ────────────────────────────────────────────────────────────

/// A unique, cryptographically random key-value fact.
///
/// The value is drawn from OS entropy (`rand`'s thread RNG): 128 random
/// bits rendered as `{16 hex}-{16 hex}`, so a coincidental occurrence in
/// a model response is effectively impossible — a substring hit is a
/// genuine retrieval.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Needle {
    /// The fact's key (e.g. `the secret code`).
    pub key: String,
    /// The unique random value (e.g. `9F3A21BC44D08E51-7C21`).
    pub value: String,
}

impl Needle {
    /// A fresh, unique needle (blueprint §5 C1: "a unique,
    /// cryptographically random key-value fact").
    ///
    /// The value is 128 random bits (`{a:016X}-{b:016X}` = 33 chars) so a
    /// coincidental occurrence in a model response is effectively
    /// impossible — a substring hit is a genuine retrieval.
    pub fn random() -> Self {
        let mut rng = rand::rng();
        let a = rng.random::<u64>();
        let b = rng.random::<u64>();
        Self {
            key: "the secret code".to_string(),
            value: format!("{a:016X}-{b:016X}"),
        }
    }

    /// The sentence injected into the haystack at the depth position.
    pub fn sentence(&self) -> String {
        format!(
            "IMPORTANT: {} is {}. Remember the {} exactly as written.",
            self.key, self.value, self.key
        )
    }

    /// The retrieval query appended after the document.
    pub fn query(&self) -> String {
        format!("What is {}? Answer with the value only.", self.key)
    }

    /// `true` when `response` retrieves the needle: the unique random
    /// value appears in the model's output (strict substring).
    pub fn is_retrieved(&self, response: &str) -> bool {
        response.contains(&self.value)
    }
}

// ── Haystack document generation ──────────────────────────────────────────

/// One synthetic haystack document with the needle inserted at a depth.
#[derive(Debug, Clone)]
pub struct NiahDocument {
    /// The full prompt sent to the model (haystack + needle + query).
    pub text: String,
    /// The requested context size in tokens.
    pub target_tokens: u32,
    /// The measured (tokenizer) or estimated (chars/4) token count of
    /// `text`.
    pub token_count: u32,
    /// `true` when `token_count` came from the `chars/4` heuristic.
    pub estimated: bool,
    pub needle: Needle,
    pub depth_percent: u8,
}

/// Build one NIAH document: a synthetic haystack scaled to
/// `target_tokens`, with `needle` inserted at `depth_percent` (0 = front,
/// 100 = end) and the retrieval query appended.
///
/// With a tokenizer the document is padded with the token-stable
/// `" apple"` filler (Chunk 4) until the count reaches `target_tokens`,
/// distributed around the needle in proportion to the depth so the
/// insertion position stays correct. Without one, the sentence-rotation
/// filler's `chars/4` length is used and the count is flagged
/// `estimated` (graceful degradation — never panic).
///
/// **Scales to arbitrarily large targets (Issue #5):** both padding paths
/// are unbounded — the sentence rotation fills to `target_tokens * 4`
/// chars and the `" apple"` filler repeats to the exact count — so a
/// 256k / 500k / 1M+ context builds the same way a 2k one does. The needle
/// is placed at `depth_percent` of the *total* document, so depth placement
/// stays correct at any size (depth 0 = front, 100 = end, 50 = middle).
/// A 1M-token haystack is a ~4–6 MB prompt string; build time is linear in
/// the size (a few hundred ms at 1M) — memory and time are not a
/// practical constraint well past the old 128k ceiling.
pub fn build_document(
    target_tokens: u32,
    depth_percent: u8,
    needle: &Needle,
    tokenizer: Option<&Tokenizer>,
) -> NiahDocument {
    let target = target_tokens.max(1);
    let depth = depth_percent.min(100);

    // Split the sentence-rotation haystack (Chunk 4 bank) at the depth.
    let sentences = haystack_sentences(target);
    let n = sentences.len().max(1);
    let insert = (depth as usize * n) / 100; // 0..=n
    let mut before: Vec<String> = sentences[..insert].to_vec();
    let mut after: Vec<String> = sentences[insert..].to_vec();

    let (text, token_count, estimated) = match tokenizer {
        Some(tok) => pad_to_target(tok, &mut before, &mut after, needle, target, insert, n),
        None => {
            let text = assemble(&before, &after, needle);
            let c = count_tokens(&text, None);
            (text, c.tokens, true)
        }
    };

    NiahDocument {
        text,
        target_tokens: target,
        token_count,
        estimated,
        needle: needle.clone(),
        depth_percent: depth,
    }
}

/// Rotate the sentence bank until `target_tokens * 4` chars (the chars/4
/// notion of "target_tokens worth of text").
///
/// **Scales to arbitrarily large targets (Issue #5):** the loop is a plain
/// `chars/4` fill — no fixed ceiling — so a 256k / 500k / 1M+ context is
/// just more rotated sentences. Practical limits: the assembled prompt is
/// ~`4 bytes / token` (the sentence rotation), so a 1M-token haystack is a
/// ~4 MB string (~90k rotated sentences); 1M+ is memory-trivial, and the
/// build time is linear in the size (a few hundred ms at 1M). The `u64`
/// intermediate keeps `target * 4` overflow-safe for any `u32` target.
fn haystack_sentences(target_tokens: u32) -> Vec<String> {
    let target_chars = (target_tokens as u64 * 4) as usize;
    let mut out: Vec<String> = Vec::new();
    let mut total = 0usize;
    let mut i = 0usize;
    while total < target_chars {
        let s = BASE_SENTENCES[i % BASE_SENTENCES.len()].to_string();
        total += s.len() + 1;
        out.push(s);
        i += 1;
    }
    out
}

/// Pad the split haystack with `" apple"` filler until the tokenizer
/// count reaches `target`, distributing the deficit around the needle in
/// proportion to its depth (so the depth position is preserved).
fn pad_to_target(
    tok: &Tokenizer,
    before: &mut Vec<String>,
    after: &mut Vec<String>,
    needle: &Needle,
    target: u32,
    insert: usize,
    n: usize,
) -> (String, u32, bool) {
    let ratio = insert as f64 / n.max(1) as f64;
    for _ in 0..4 {
        let text = assemble(before, after, needle);
        let count = match tok.try_count(&text) {
            Some(c) => c,
            // The tokenizer cannot encode this (exotic vocabulary):
            // degrade to the chars/4 estimate, flagged.
            None => {
                let c = count_tokens(&text, None);
                return (text, c.tokens, true);
            }
        };
        if count >= target {
            return (text, count, false);
        }
        let deficit = target - count;
        let before_share = (deficit as f64 * ratio) as u32;
        append_filler(before, before_share);
        append_filler(after, deficit - before_share);
    }
    // Exotic case: the count did not converge (filler not ~1 token).
    // Return what we have — the count is best-effort.
    let text = assemble(before, after, needle);
    let count = tok
        .try_count(&text)
        .unwrap_or_else(|| (text.chars().count() / 4) as u32);
    (text, count, false)
}

/// Append `k` copies of the token-stable `" apple"` filler. `FILLER`
/// starts with a space, so concatenating onto the last segment (or
/// starting a new one) yields exactly one space per filler unit.
fn append_filler(segments: &mut Vec<String>, k: u32) {
    if k == 0 {
        return;
    }
    match segments.last_mut() {
        Some(last) => last.push_str(&FILLER.repeat(k as usize)),
        None => segments.push(FILLER.repeat(k as usize)),
    }
}

/// The full prompt: haystack segments *before* the depth, the needle
/// sentence, the segments *after*, and the query. Empty sides (depth 0 /
/// 100) are skipped without stray spaces.
fn assemble(before: &[String], after: &[String], needle: &Needle) -> String {
    let mut text = String::new();
    for s in before {
        if !text.is_empty() {
            text.push(' ');
        }
        text.push_str(s);
    }
    if !text.is_empty() {
        text.push(' ');
    }
    text.push_str(&needle.sentence());
    for s in after {
        text.push(' ');
        text.push_str(s);
    }
    text.push_str("\n\n");
    text.push_str(&needle.query());
    text
}

// ── Cell states & scoring ─────────────────────────────────────────────────

/// The per-cell state (View 3 color-coding, blueprint §6 View 3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NiahCellState {
    /// Accurate retrieval + nominal prefill (green).
    Nominal,
    /// Accurate retrieval + throttled prefill (yellow).
    Throttled,
    /// Retrieval failed / hallucinated / stream error (red).
    Failed,
}

/// Classify one cell's *prefill* state against the baseline.
///
/// * `!retrieved` (or no TTFT) → [`NiahCellState::Failed`];
/// * the baseline row itself (`tokens <= base_tokens`) →
///   [`NiahCellState::Nominal`] (nothing to compare — never a false throttle);
/// * a **missing / failed baseline** (`base_ttft_s <= 0.0`) on a larger cell →
///   [`NiahCellState::Nominal`] here, *but* [`NiahResult::compute_states`]
///   records that on the cell as `no_baseline = true` so it is **not** read as
///   a confirmed-healthy nominal (CF-45 — a failed smallest-size row used to
///   mask real prefill degradation by classifying every larger cell nominal);
/// * TTFT within `PREFILL_THROTTLE_FACTOR`× the linear expectation
///   (`base_ttft × tokens / base_tokens`) → [`NiahCellState::Nominal`];
/// * beyond that → [`NiahCellState::Throttled`].
///
/// The 3-state [`NiahCellState`] cannot express "no baseline" (the sibling
/// View-3 render matches it exhaustively), so that distinction is carried on
/// the cell itself, not the enum.
pub fn classify(
    retrieved: bool,
    ttft_s: f64,
    base_ttft_s: f64,
    base_tokens: u32,
    tokens: u32,
) -> NiahCellState {
    if !retrieved || ttft_s <= 0.0 {
        return NiahCellState::Failed;
    }
    if tokens <= base_tokens || base_ttft_s <= 0.0 {
        return NiahCellState::Nominal;
    }
    let expected_ttft = base_ttft_s * tokens as f64 / base_tokens as f64;
    if ttft_s > expected_ttft * PREFILL_THROTTLE_FACTOR {
        NiahCellState::Throttled
    } else {
        NiahCellState::Nominal
    }
}

/// The JSON string form of a cell's prefill state (the `--json` NIAH section).
#[must_use]
fn state_name(state: NiahCellState) -> &'static str {
    match state {
        NiahCellState::Nominal => "nominal",
        NiahCellState::Throttled => "throttled",
        NiahCellState::Failed => "failed",
    }
}

/// One size×depth cell of the matrix.
#[derive(Debug, Clone)]
pub struct NiahCell {
    pub target_tokens: u32,
    pub depth_percent: u8,
    /// `true` when the needle value was retrieved from the response.
    pub retrieved: bool,
    /// **Hybrid reasoning (Option C):** `true` when the stream completed OK
    /// but the model spent its entire generation budget on chain-of-thought —
    /// at least one `Chunk::Reasoning` frame and **zero** `Chunk::Content`
    /// frames. This is a distinct, **non-fail** outcome: an honest
    /// *inconclusive (budget)*. The model *was* generating, we just never saw
    /// it commit an answer. Content-only scoring is preserved — a value seen
    /// only in CoT is never credited as `retrieved`; this flag explains *why*
    /// a non-retrieved cell produced no answer (the over-thinker case). Set by
    /// [`NiahEngine::run_cell`] from
    /// [`StreamOutcome::is_truncated_by_reasoning`].
    pub truncated_by_reasoning: bool,
    /// The server-reported chain-of-thought tokens for this cell (R3
    /// calibration measurement): `usage.completion_tokens_details.
    /// reasoning_tokens`. `None` when the backend did not report a usage
    /// block (the honest "not reported" case); `Some(0)` when it reported
    /// zero CoT (thinking suppression worked). Set by
    /// [`NiahEngine::run_cell`] from [`StreamOutcome::usage_details`].
    pub reasoning_tokens: Option<u64>,
    /// The **prompt** (input) tokens for this cell (H2): the server-reported
    /// `usage.prompt_tokens`, falling back to the document's `token_count`
    /// (the `chars/4` estimate when no `--tokenizer`). Always a real value —
    /// the C1 J/token denominator's *input* side. Set by
    /// [`NiahEngine::run_cell`].
    pub prompt_tokens: u64,
    /// The **completion** (output) tokens for this cell (H2): the
    /// server-reported `usage.completion_tokens`, or `None` when the backend
    /// reported no usage block (the honest "not reported" case — mirrors
    /// [`reasoning_tokens`](Self::reasoning_tokens)). The C1 denominator's
    /// *output* side; `None` falls back to the labeled
    /// [`NIAH_MAX_GEN_TOKENS`] gen-cap estimate. Set by
    /// [`NiahEngine::run_cell`].
    pub completion_tokens: Option<u64>,
    /// Time-to-first-token, seconds (`T3 − T1`; `0.0` on stream failure).
    pub ttft_s: f64,
    /// Prefill throughput, tokens/s (prompt tokens ÷ TTFT).
    pub prefill_tps: f64,
    /// The ANSI class View 3 renders (assigned by [`NiahResult::compute_states`]).
    pub state: NiahCellState,
    /// `true` when this cell retrieved the needle but had **no valid baseline**
    /// to anchor the prefill comparison against — the smallest-size cell at this
    /// depth failed (`ttft_s == 0.0`), so the prefill-degradation verdict is
    /// *unknown*, not a confirmed-healthy nominal (CF-45). Set by
    /// [`NiahResult::compute_states`].
    pub no_baseline: bool,
    /// `true` when this cell's context size came from the `chars/4` estimate
    /// (no `--tokenizer` supplied) rather than an exact token count. Without a
    /// tokenizer the haystack is low-entropy repeated sentences sized by that
    /// heuristic, so the "128k" label is *approximate* — [`context_label`]
    /// surfaces that so the size is never presented as exact (CF-46). Set by
    /// [`NiahEngine::run_cell`] from the document's `estimated` flag.
    pub approximate: bool,
}

impl NiahCell {
    /// An unevaluated cell (the empty-grid placeholder).
    pub fn unrun() -> Self {
        Self {
            target_tokens: 0,
            depth_percent: 0,
            retrieved: false,
            truncated_by_reasoning: false,
            reasoning_tokens: None,
            prompt_tokens: 0,
            completion_tokens: None,
            ttft_s: 0.0,
            prefill_tps: 0.0,
            state: NiahCellState::Failed,
            no_baseline: false,
            approximate: false,
        }
    }

    /// The context-size label for this cell, rendered with the shared
    /// [`human_size`] formatter (`2k`, `128k`, `256k`, `1M`, …).
    ///
    /// **C1 token-honesty (Issue #4 / CF-46):** a NIAH context is built to a
    /// *target* token count. With an exact `--tokenizer` the haystack is
    /// padded to that count and the label reads plain (`128k`); without one,
    /// the size comes from the `chars/4` heuristic (the low-entropy
    /// repeated-sentence haystack), so the label reads `"{size} (approx.)"`
    /// (`128k (approx.)`, `1M (approx.)`) — the context length is never
    /// presented as exact. The `approximate` flag is set by
    /// [`NiahEngine::run_cell`] from the document's `estimated` flag.
    pub fn context_label(&self) -> String {
        context_size_label(self.target_tokens, self.approximate)
    }
}

/// The N×M matrix result (row-major: `sizes.len() × depths.len()` cells).
#[derive(Debug, Clone)]
pub struct NiahResult {
    pub sizes: Vec<u32>,
    pub depths: Vec<u8>,
    pub cells: Vec<NiahCell>,
}

impl NiahResult {
    /// An empty (unevaluated) grid.
    pub fn new(sizes: Vec<u32>, depths: Vec<u8>) -> Self {
        let cells = (0..sizes.len() * depths.len())
            .map(|_| NiahCell::unrun())
            .collect();
        Self {
            sizes,
            depths,
            cells,
        }
    }

    /// The cell at (context size, depth), `None` when absent.
    pub fn cell(&self, size: u32, depth: u8) -> Option<&NiahCell> {
        let si = self.sizes.iter().position(|s| *s == size)?;
        let di = self.depths.iter().position(|d| *d == depth)?;
        Some(&self.cells[si * self.depths.len() + di])
    }

    /// `(retrieved, run)` over the cells that actually *ran*.
    ///
    /// Unrun cells (the [`NiahCell::unrun`] placeholder, `target_tokens == 0`)
    /// are excluded from the denominator — exactly like
    /// [`size_pass_rate`]. This keeps the headline honest on a partial /
    /// interrupted matrix: it reports the pass rate over the cells evaluated,
    /// not one diluted by cells that never ran (CF-31 — the old version
    /// counted every grid slot, so an unrun cell read as a failure and the
    /// two "pass rate" numbers disagreed).
    pub fn accuracy(&self) -> (usize, usize) {
        let total = self.cells.iter().filter(|c| c.target_tokens > 0).count();
        let retrieved = self
            .cells
            .iter()
            .filter(|c| c.target_tokens > 0 && c.retrieved)
            .count();
        (retrieved, total)
    }

    /// The number of *run* cells that were `truncated_by_reasoning` (Option C):
    /// the stream completed OK but the model spent its whole generation budget
    /// on chain-of-thought (all-CoT / zero-content). These are **not**
    /// capability failures (the model was generating) and **not** verified
    /// passes (no answer was committed to grade) — an honest *inconclusive
    /// (budget)*. They are reported separately from the pass/total figure.
    #[must_use]
    pub fn truncated_count(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| c.target_tokens > 0 && c.truncated_by_reasoning)
            .count()
    }

    /// The number of *run* cells that are genuine retrieval **failures**: the
    /// stream completed, produced visible content, and the needle is absent —
    /// the model got it wrong (or hallucinated). Distinct from
    /// [`truncated_count`] (no content at all) and from a transport failure.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.cells
            .iter()
            .filter(|c| c.target_tokens > 0 && !c.retrieved && !c.truncated_by_reasoning)
            .count()
    }

    /// The headline accuracy figure, e.g. `71/77 retrieved (92.2%)`.
    ///
    /// Over a *partial* matrix (some cells never ran) the denominator is the
    /// run cells only, and the label makes that explicit — `30/42 of 77 run
    /// (71.4%)` — so an interrupted run is never read as a fleet of failures
    /// (CF-31, consistent with [`size_pass_rate`]).
    ///
    /// **Hybrid reasoning (Option C):** the headline is `pass/total` with the
    /// `truncated_by_reasoning` count **broken out** (and genuine fails, when
    /// any), e.g. `74/77 retrieved, 3 truncated-by-reasoning (96.1%)`. A
    /// truncated cell is *not* a capability failure, so it is reported
    /// separately rather than folded silently into the fail count; when there
    /// are none, the label reads exactly as before.
    pub fn accuracy_label(&self) -> String {
        let (r, t) = self.accuracy();
        let grid = self.cells.len();
        let pct = if t > 0 {
            r as f64 / t as f64 * 100.0
        } else {
            0.0
        };
        // The Option C breakdown, each clause omitted when its count is zero
        // so a clean run reads exactly as today.
        let truncated = self.truncated_count();
        let trunc_note = if truncated > 0 {
            format!(", {truncated} truncated-by-reasoning")
        } else {
            String::new()
        };
        let failed = self.failed_count();
        let fail_note = if failed > 0 {
            format!(", {failed} failed")
        } else {
            String::new()
        };
        if t == 0 {
            "No cells run yet".to_string()
        } else if t == grid {
            format!("{r}/{t} retrieved{trunc_note}{fail_note} ({pct:.1}%)")
        } else {
            format!("{r}/{t} of {grid} run{trunc_note}{fail_note} ({pct:.1}%)")
        }
    }

    /// The pass rate (0..=100) for one context size across all its depths —
    /// the per-row percentage View 3 shows in its `PASS %` column. `None`
    /// when the size has no *run* cells (an unrun row shows `--`).
    ///
    /// **Hybrid reasoning (Option C) — exact treatment of truncated cells:**
    /// a `truncated_by_reasoning` cell is *inconclusive (budget)*, **not** a
    /// retrieval failure. It is therefore **excluded from the capability
    /// denominator** — the pass rate is computed over the cells that produced
    /// a verifiable answer (ran and were *not* truncated). Counting a
    /// truncated cell as a non-pass would re-introduce the false negative this
    /// design eliminates (an over-thinker reading as a broken retriever).
    /// Consequently a row that is *entirely* truncated has **no** measurable
    /// pass rate (`None` → `--`), and a row's rate is unaffected by how many
    /// of its cells were truncated. This is the same split the headline
    /// [`accuracy_label`] makes explicit.
    pub fn size_pass_rate(&self, size: u32) -> Option<f64> {
        let si = self.sizes.iter().position(|s| *s == size)?;
        let start = si * self.depths.len();
        let cells = &self.cells[start..start + self.depths.len()];
        // Unrun cells are the `unrun()` placeholder (`target_tokens == 0`).
        let run = cells.iter().filter(|c| c.target_tokens > 0).count();
        if run == 0 {
            return None;
        }
        // Option C: truncate the denominator to the *measurable* cells (ran
        // and not truncated-by-reasoning). An all-truncated row → `None`.
        let measurable = run - cells.iter().filter(|c| c.truncated_by_reasoning).count();
        if measurable == 0 {
            return None;
        }
        let retrieved = cells.iter().filter(|c| c.retrieved).count();
        Some(retrieved as f64 / measurable as f64 * 100.0)
    }

    /// The largest context size whose pass rate is >= `min_pct` — the
    /// "reliable up to ~Xk" threshold for the practical interpretation line.
    /// `None` when no size clears the bar.
    ///
    /// **Option C:** because [`size_pass_rate`] excludes `truncated_by_reasoning`
    /// cells from the denominator, a size that is *entirely* truncated has no
    /// pass rate and can never be "reliable up to" — an unmeasurable size is
    /// not declared reliable. A size whose measurable cells clear the bar is
    /// reported normally, truncated or not.
    pub fn reliable_up_to(&self, min_pct: f64) -> Option<u32> {
        self.sizes
            .iter()
            .copied()
            .filter(|s| self.size_pass_rate(*s).is_some_and(|p| p >= min_pct))
            .max()
    }

    /// The plain-language interpretation of the matrix: how far the model
    /// reliably retrieves, and what that means for RAG / long-document use.
    /// Pure over the grid (unit-testable).
    ///
    /// F8: the size renders through the shared [`human_size`] formatter
    /// (`2k` / `128k` / `1M` / `1.5M`) — the old `size / 1000` + `k` format
    /// read a 1 000 000-token size as "~1000k".
    pub fn interpretation(&self) -> String {
        match self.reliable_up_to(80.0) {
            Some(size) => format!(
                "Reliable retrieval up to ~{} tokens. Beyond that, the model \
                 loses track of embedded information. For RAG: limit context \
                 windows to {} or use a model with better long-context training.",
                human_size(size),
                human_size(size)
            ),
            None => "No context size met the 80% reliability bar — this model is \
                     not dependable for retrieval in any tested length. Avoid RAG / \
                     long-document QA, or choose a better long-context model."
                .to_string(),
        }
    }

    /// Assign every cell's state: the *smallest* context size is the
    /// baseline for the linear prefill expectation (blueprint §5 C1:
    /// "the decay curve of prefill processing speed as context length
    /// scales up").
    pub fn compute_states(&mut self) {
        let base_size = *self.sizes.first().unwrap_or(&0);
        if base_size == 0 || self.depths.is_empty() {
            return;
        }
        let base_ttfts: Vec<f64> = self
            .depths
            .iter()
            .map(|d| self.cell(base_size, *d).map(|c| c.ttft_s).unwrap_or(0.0))
            .collect();
        for (si, size) in self.sizes.iter().enumerate() {
            for (di, _) in self.depths.iter().enumerate() {
                let cell = &mut self.cells[si * self.depths.len() + di];
                cell.state = classify(
                    cell.retrieved,
                    cell.ttft_s,
                    base_ttfts[di],
                    base_size,
                    *size,
                );
                // CF-45: an accurate *larger* cell whose baseline (same depth,
                // smallest size) failed has no valid anchor — its prefill
                // verdict is unknown, NOT a confirmed-healthy nominal. Record
                // that distinctly on the cell (the 3-state enum can't carry it,
                // and the sibling View-3 match is exhaustive over the 3).
                cell.no_baseline = cell.state == NiahCellState::Nominal
                    && *size > base_size
                    && base_ttfts[di] <= 0.0;
            }
        }
    }

    /// Map the matrix to `needle_evaluations` rows (storage, Chunk 12)
    /// for persistence by later integration.
    pub fn to_needle_rows(&self, session_id: &str) -> Vec<NeedleEvaluation> {
        self.cells
            .iter()
            .map(|c| NeedleEvaluation {
                eval_id: None,
                session_id: session_id.to_string(),
                context_length: Some(c.target_tokens as i64),
                depth_percent: Some(c.depth_percent as f64),
                retrieved_successfully: Some(c.retrieved),
                // Option C: persist the per-cell truncated-by-reasoning state
                // so the `needle_evaluations` row distinguishes an
                // inconclusive (budget) cell from a genuine retrieval fail.
                truncated_by_reasoning: Some(c.truncated_by_reasoning),
                latency_ms: Some(c.ttft_s * 1000.0),
            })
            .collect()
    }

    /// The `--json` document for this matrix (additive NIAH section, Issue #6).
    ///
    /// Carries the per-size recall, the `reliable_up_to` threshold, and the
    /// full per-cell detail — each cell labeled via [`NiahCell::context_label`],
    /// so the `(approx.)` qualifier is present when no tokenizer was supplied.
    /// This makes a headless `--json` run self-contained: the NIAH recall no
    /// longer lives only in the run log + DB.
    ///
    /// **Option C:** the top level carries the three-way split —
    /// `truncated_by_reasoning` (the aggregate count of inconclusive (budget)
    /// cells) and `failed` (genuine retrieval failures) — and each cell
    /// carries its own `truncated_by_reasoning` flag. Pure and testable.
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        let (retrieved, total) = self.accuracy();
        let sizes = self
            .sizes
            .iter()
            .enumerate()
            .map(|(si, s)| {
                let start = si * self.depths.len();
                let row = &self.cells[start..start + self.depths.len()];
                let run = row.iter().filter(|c| c.target_tokens > 0).count();
                let got = row.iter().filter(|c| c.retrieved).count();
                let approx = row.iter().any(|c| c.approximate);
                serde_json::json!({
                    "size": s,
                    "label": context_size_label(*s, approx),
                    "pass_rate": self.size_pass_rate(*s),
                    "retrieved": got,
                    "run": run,
                })
            })
            .collect::<Vec<_>>();
        let cells = self
            .cells
            .iter()
            .map(|c| {
                serde_json::json!({
                    "size": c.target_tokens,
                    "label": c.context_label(),
                    "depth": c.depth_percent,
                    "retrieved": c.retrieved,
                    "truncated_by_reasoning": c.truncated_by_reasoning,
                    // R3: the server-reported CoT tokens for the cell
                    // (`null` when the backend reported no usage block) —
                    // the calibration measurement of a model's reasoning
                    // usage.
                    "reasoning_tokens": c.reasoning_tokens,
                    // H2: the real per-cell prompt + completion token counts
                    // (the C1 J/token denominator). `prompt_tokens` is always
                    // present (usage, or the doc `chars/4` fallback);
                    // `completion_tokens` is `null` when the backend reported
                    // no usage block — additive, back-compatible.
                    "prompt_tokens": c.prompt_tokens,
                    "completion_tokens": c.completion_tokens,
                    "ttft_s": c.ttft_s,
                    "prefill_tps": c.prefill_tps,
                    "state": state_name(c.state),
                    "no_baseline": c.no_baseline,
                    "approximate": c.approximate,
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({
            "accuracy": self.accuracy_label(),
            "retrieved": retrieved,
            "total": total,
            // Option C: the three-way split, so a headless `--json` run is
            // self-contained about *why* cells didn't retrieve. `truncated_
            // by_reasoning` is an inconclusive (budget) outcome, reported
            // separately from genuine `failed` cells.
            "truncated_by_reasoning": self.truncated_count(),
            "failed": self.failed_count(),
            "reliable_up_to": self.reliable_up_to(80.0),
            "sizes": sizes,
            "depths": self.depths,
            "cells": cells,
        })
    }
}

// ── The live runner ───────────────────────────────────────────────────────

/// The Engine C1 runner: executes the size×depth matrix against an
/// endpoint.
///
/// Cells run *sequentially* (one stream at a time, like Engine C2) so
/// each cell measures the endpoint alone — no queueing cross-talk — and
/// the prefill-degradation curve is not polluted by concurrency.
#[derive(Debug)]
pub struct NiahEngine {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    timeout: u64,
    /// Optional HF tokenizer for exact haystack sizing (Chunk 4 policy:
    /// an explicit `--tokenizer` that fails to load is a hard error;
    /// without one, counts are `chars/4` estimates).
    tokenizer: Option<Arc<Tokenizer>>,
    /// The per-cell generation budget in tokens (R3). The resolved config
    /// value (`--niah-max-tokens` / `CRUCIBLE_NIAH_MAX_TOKENS` / the config
    /// file > the built-in [`NIAH_MAX_GEN_TOKENS`] hybrid-reasoning safety
    /// net). This is the budget a verbose reasoner must fit its
    /// chain-of-thought *and* the answer into.
    max_gen_tokens: u32,
    /// The best-effort `disable_thinking` suppression (R3). On by default
    /// (a speed optimization); `--no-disable-thinking` turns it off so the
    /// model thinks by default — the force-test path for the
    /// `truncated_by_reasoning` safety net.
    disable_thinking: bool,
    sizes: Vec<u32>,
    depths: Vec<u8>,
    /// Optional sequence seam: publish `Size …, Depth …` to the
    /// [`ProgressBus`] the Benchmark Sequence mirrors into the TUI.
    progress: Option<Arc<ProgressBus>>,
    /// Optional TUI seam: publish live single-stream
    /// [`crate::metrics::state::MetricsSnapshot`]s while a cell runs, so
    /// the Live Monitor shows the current cell's real stream data.
    metrics: Option<Arc<MetricsState>>,
    /// Optional `Space`-key pause gate: the matrix waits on it before
    /// spawning each cell's stream (in-flight cells complete).
    pause: Option<Arc<RunPause>>,
    /// Optional run logger (each cell's worker records its HTTP / SSE
    /// lifecycle; the engine records the matrix start / result).
    logger: Option<Arc<crate::log::RunLogger>>,
}

impl NiahEngine {
    /// Build from the resolved config (same client/tokenizer policy as
    /// Engines A/C2).
    ///
    /// **Issue #5 — runtime-configurable ladders:** the context-size and
    /// needle-depth ladders come from the resolved config
    /// (`--niah-contexts` / `--niah-depths`, layered CLI > env > file >
    /// default), not the compile-time [`NIAH_SIZES`] / [`NIAH_DEPTHS`]
    /// constants (those remain the defaults). The focused "count test" mode
    /// (`--niah-custom <size>`) takes precedence: it collapses the matrix to
    /// a **single size row** at exactly that context (across the depth
    /// ladder) — for probing a RoPE/YaRN-extended limit beyond 128k.
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        let tokenizer = match &cfg.tokenizer {
            Some(path) => Some(Arc::new(Tokenizer::from_file(path)?)),
            None => None,
        };
        let sizes = match cfg.niah_custom {
            Some(size) => vec![size],
            None => cfg.niah_contexts.clone(),
        };
        let depths = cfg.niah_depths.clone();
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            tokenizer,
            max_gen_tokens: cfg.niah_max_tokens,
            disable_thinking: cfg.disable_thinking,
            sizes,
            depths,
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
    }

    /// The configured per-cell generation budget in tokens (R3).
    #[must_use]
    pub fn max_gen_tokens(&self) -> u32 {
        self.max_gen_tokens
    }

    /// The configured `disable_thinking` suppression (R3; on by default).
    #[must_use]
    pub fn disable_thinking(&self) -> bool {
        self.disable_thinking
    }

    /// Override the context sizes (tests / custom matrices).
    pub fn sizes(mut self, sizes: Vec<u32>) -> Self {
        self.sizes = sizes;
        self
    }

    /// Override the needle depths (tests / custom matrices).
    pub fn depths(mut self, depths: Vec<u8>) -> Self {
        self.depths = depths;
        self
    }

    /// Attach a [`ProgressBus`] so the matrix run reports `Size …,
    /// Depth …/…` to the Benchmark Sequence (the TUI's progress bar).
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach a [`MetricsState`] publisher so each cell pushes live
    /// single-stream snapshots to the TUI (the Live Monitor's stream
    /// matrix / gauges show the current cell's real data).
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach the `Space`-key pause gate: each matrix cell waits on it
    /// before its stream is spawned (the headless path leaves it `None`).
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger (each cell's stream then logs its HTTP /
    /// SSE lifecycle).
    pub fn logger(mut self, logger: Arc<crate::log::RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// The configured context sizes.
    pub fn sizes_list(&self) -> &[u32] {
        &self.sizes
    }

    /// The configured needle depths.
    pub fn depths_list(&self) -> &[u8] {
        &self.depths
    }

    /// Run the full matrix (sequential), returning the scored grid
    /// (states computed against the smallest-size baseline).
    ///
    /// While a [`ProgressBus`] is attached, each cell publishes
    /// `Size …, Depth …/…` — the Benchmark Sequence mirrors it into the
    /// TUI's progress bar.
    pub async fn run(&self) -> NiahResult {
        let mut result = NiahResult::new(self.sizes.clone(), self.depths.clone());
        let total_cells = self.sizes.len() * self.depths.len();
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC1,
                format!(
                    "Engine C1 (NIAH) started — {total_cells} cells ({} sizes × {} depths)",
                    self.sizes.len(),
                    self.depths.len()
                ),
            );
        }
        for (si, size) in self.sizes.iter().enumerate() {
            for (di, depth) in self.depths.iter().enumerate() {
                let cell_idx = si * self.depths.len() + di;
                // The `Space`-key pause: hold before the next cell's
                // request goes out (an in-flight cell always completes).
                if let Some(gate) = &self.pause {
                    gate.wait_while_paused().await;
                }
                if let Some(bus) = &self.progress {
                    bus.publish(EngineProgress::Niah {
                        size: *size,
                        depth: di + 1,
                        total_depths: self.depths.len(),
                        cell: cell_idx + 1,
                        total_cells,
                    });
                }
                let cell = self.run_cell(*size, *depth).await;
                result.cells[cell_idx] = cell;
            }
        }
        result.compute_states();
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC1,
                format!("Engine C1 complete — {}", result.accuracy_label()),
            );
        }
        result
    }

    /// Run one cell: fresh unique needle → document → one
    /// [`StreamWorker`] stream → retrieval check + §7 timing deltas.
    async fn run_cell(&self, size: u32, depth: u8) -> NiahCell {
        let needle = Needle::random();
        let doc = build_document(size, depth, &needle, self.tokenizer.as_deref());

        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            &doc.text,
            self.max_gen_tokens,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        // Issue #3 + C3: the shared size formatter (was `C1:{size}k-…`,
        // which rendered a 128 000-token cell as `C1:128000k-d50` in the
        // headless log; now `C1:128k-d50` / `C1:1M-d50`, matching the TUI).
        // The `(approx.)` qualifier is **not** repeated here (it used to
        // ride every one of the per-cell lines): it is qualified once at
        // the `[CONFIG]` line and the C1 start line (`sequence.rs`)
        // instead.
        .tag(cell_tag(size, depth));
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }
        // CF-01 (best-effort, Option C hybrid): suppress the model's
        // chain-of-thought so the generation budget goes to the visible
        // answer. This is a *speed* optimization, **not** a correctness
        // dependency: a config-only server may silently drop the per-request
        // switch (the raised budget is then the safety net), and Qwen3.5+
        // `/no_think` is not a reliable switch. Non-reasoning models and
        // backends without a switch are unaffected; and when suppression is
        // ignored (or off via `--no-disable-thinking`, the R3 force-test
        // path), an all-CoT / zero-content cell is flagged
        // `truncated_by_reasoning` (below) instead of a false capability
        // failure (Strawberry Issue 1).
        worker = worker.disable_thinking(self.disable_thinking);

        // Spawn without awaiting: drain the channel concurrently to avoid
        // the >capacity deadlock (server-agnostic fix).
        let handle = spawn_worker(worker, tx);
        // CF-01: the visible answer only. Reasoning deltas are deliberately
        // NOT accumulated into the retrieval check (see the loop below).
        let mut content_text = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    // Content-only matching (CF-01 / CF-15): the retrieval
                    // verdict is the model's *visible answer*, not its
                    // chain-of-thought. Judging on the content span means a
                    // model that thinks then answers is graded on its answer,
                    // and a model that merely *mentions* the needle value in
                    // its CoT cannot false-pass a cell it never answered.
                    // (Reasoning deltas are still forwarded for the live
                    // snapshot / metrics, just not accumulated here.)
                    if let Chunk::Content(text) = &frame.chunk {
                        content_text.push_str(text);
                    }
                }
            }
            // Publish a live snapshot every 8 events, and immediately on
            // a terminal event, so the TUI's Live Monitor shows the
            // current cell's real stream data (measurement-isolation
            // invariant: the render loop only reads the snapshot).
            let is_terminal = matches!(
                &event,
                StreamEvent::Complete { .. } | StreamEvent::Failed { .. }
            );
            events.push(event);
            batch += 1;
            if let Some(state) = &self.metrics {
                if batch >= 8 || is_terminal {
                    state.update(SpeedEngine::single_stream_snapshot(
                        &self.url,
                        &self.model,
                        "NIAH",
                        &events,
                        self.max_gen_tokens,
                        self.tokenizer.as_deref(),
                    ));
                    batch = 0;
                }
            }
        }
        // Final publish (the terminal event may have been batched out).
        if let Some(state) = &self.metrics {
            state.update(SpeedEngine::single_stream_snapshot(
                &self.url,
                &self.model,
                "NIAH",
                &events,
                self.max_gen_tokens,
                self.tokenizer.as_deref(),
            ));
        }

        // Channel closed: collect the outcome.
        let outcome = join_worker(handle).await;

        // R3: the server-reported CoT token count for this cell (the
        // calibration measurement). `None` when the backend reported no
        // usage block (the honest "not reported" case).
        let reasoning_tokens = outcome.usage_details.map(|d| d.reasoning_tokens);

        // **Hybrid reasoning (Option C):** a stream that completed OK but spent
        // its whole budget on chain-of-thought (all-CoT / zero-content) is a
        // distinct, non-fail outcome — *inconclusive (budget)*. The verdict is
        // frame-based (the wire truth the SSE parser already classified), so it
        // works on every server with no `usage` dependency.
        let truncated_by_reasoning = outcome.is_truncated_by_reasoning(&events);

        // Content-only verdict (CF-01): the needle is retrieved iff it appears
        // in the model's visible answer, not in its chain-of-thought. A value
        // seen only in CoT is never credited — content-only scoring is
        // preserved unchanged by the hybrid.
        let retrieved = needle.is_retrieved(&content_text);
        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let ttft = ts.ttft_nanos().map(|n| n as f64 / 1e9).unwrap_or(0.0);
        let prompt_tokens = outcome
            .usage
            .map(|u| u.prompt_tokens)
            .unwrap_or(doc.token_count as u64);
        // H2: the server-reported completion (output) tokens for the cell —
        // `None` when the backend reported no usage block (the honest
        // "not reported" case). The C1 J/token denominator's output side.
        let completion_tokens = outcome.usage.map(|u| u.completion_tokens);
        let prefill_tps = if ttft > 0.0 {
            prompt_tokens as f64 / ttft
        } else {
            0.0
        };
        NiahCell {
            target_tokens: size,
            depth_percent: depth,
            retrieved,
            truncated_by_reasoning,
            reasoning_tokens,
            prompt_tokens,
            completion_tokens,
            ttft_s: ttft,
            prefill_tps,
            // Assigned by `compute_states` once the whole matrix is in.
            state: NiahCellState::Failed,
            no_baseline: false,
            // CF-46: propagate the document's estimate flag so the cell's
            // context size is labeled approximate (chars/4) when no tokenizer
            // was supplied, never presented as exact.
            approximate: doc.estimated,
        }
    }
}

// ── The lock-free result slot (TUI seam) ──────────────────────────────────

/// A lock-free holder for the latest NIAH result.
///
/// The same `ArcSwap` double-buffer pattern as the Chunk 6 metrics
/// snapshot: the background runner (spawned from the `n` key) *publishes*
/// via [`store`], and View 3 *reads* via [`load`] — the render loop never
/// blocks, never takes a mutex, and never touches the timing path
/// (measurement-isolation invariant, blueprint §4).
#[derive(Debug)]
pub struct NiahSlot {
    result: ArcSwap<Option<NiahResult>>,
    running: AtomicBool,
}

impl NiahSlot {
    /// An empty slot (no result yet, nothing running).
    pub fn new() -> Self {
        Self {
            result: ArcSwap::from_pointee(None),
            running: AtomicBool::new(false),
        }
    }

    /// Lock-free read of the current result (`None` until the first run
    /// completes).
    pub fn load(&self) -> Arc<Option<NiahResult>> {
        self.result.load_full()
    }

    /// Publish a completed matrix.
    pub fn store(&self, result: NiahResult) {
        self.result.store(Arc::new(Some(result)));
    }

    /// Mark a run in progress (key path) / finished (runner path).
    pub fn set_running(&self, running: bool) {
        self.running.store(running, Ordering::Relaxed);
    }

    /// `true` while a matrix run is in progress.
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Clear the slot back to its "not run" state (B1): store `None` and
    /// clear the running flag. Called at the start of every sequence run so
    /// a prior run's matrix never leaks into the next — the "None = not run"
    /// invariant the Engine D denominator and `persist_results` rely on.
    pub fn reset(&self) {
        self.result.store(Arc::new(None));
        self.running.store(false, Ordering::Relaxed);
    }
}

impl Default for NiahSlot {
    fn default() -> Self {
        Self::new()
    }
}

// ── TUI runner config ─────────────────────────────────────────────────────

/// The data the `n` key needs to spawn a background NIAH run (the
/// `Config` fields the engine consumes, cloned out of the CLI config).
#[derive(Debug, Clone)]
pub struct NiahEngineConfig {
    pub url: String,
    pub model: String,
    pub api_key: Option<String>,
    pub timeout: u64,
    pub tokenizer: Option<std::path::PathBuf>,
    /// The NIAH context-size ladder (Issue #5; default 2k→128k).
    pub niah_contexts: Vec<u32>,
    /// The NIAH needle-depth ladder (Issue #5; default 0%…100% step 10).
    pub niah_depths: Vec<u8>,
    /// The focused single-size "count test" mode (Issue #5). When `Some`,
    /// it takes precedence over `niah_contexts`.
    pub niah_custom: Option<u32>,
    /// The per-cell generation budget in tokens (R3; default
    /// [`NIAH_MAX_GEN_TOKENS`]).
    pub max_gen_tokens: u32,
    /// The best-effort `disable_thinking` suppression (R3; on by default).
    pub disable_thinking: bool,
}

impl NiahEngineConfig {
    /// Snapshot the NIAH-relevant fields from the resolved config.
    pub fn from_config(cfg: &Config) -> Self {
        Self {
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            tokenizer: cfg.tokenizer.clone(),
            niah_contexts: cfg.niah_contexts.clone(),
            niah_depths: cfg.niah_depths.clone(),
            niah_custom: cfg.niah_custom,
            max_gen_tokens: cfg.niah_max_tokens,
            disable_thinking: cfg.disable_thinking,
        }
    }

    /// Build a runner from this config (Issue #5: the resolved ladders plus
    /// the focused single-size "count test" mode).
    pub fn engine(&self) -> Result<NiahEngine, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(self.timeout.max(1)))
            .build()?;
        let tokenizer = match &self.tokenizer {
            Some(path) => Some(Arc::new(Tokenizer::from_file(path)?)),
            None => None,
        };
        let sizes = match self.niah_custom {
            Some(size) => vec![size],
            None => self.niah_contexts.clone(),
        };
        let depths = self.niah_depths.clone();
        Ok(NiahEngine {
            client,
            url: self.url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            timeout: self.timeout,
            tokenizer,
            max_gen_tokens: self.max_gen_tokens,
            disable_thinking: self.disable_thinking,
            sizes,
            depths,
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_tokenizer() -> Tokenizer {
        Tokenizer::from_json(include_str!("../../../tests/fixtures/mini_tokenizer.json")).unwrap()
    }

    // ── Needle ────────────────────────────────────────────────────────────

    #[test]
    fn random_needles_are_unique_and_well_formed() {
        let a = Needle::random();
        let b = Needle::random();
        assert_ne!(a.value, b.value, "needles must be unique per cell");
        // 16 hex + '-' + 16 hex = 33 chars (128 random bits).
        assert_eq!(a.value.len(), 33);
        assert!(a.value.chars().all(|c| c.is_ascii_hexdigit() || c == '-'));
        assert_eq!(a.value.split('-').count(), 2);
    }

    #[test]
    fn retrieval_check_is_strict() {
        let n = Needle {
            key: "the secret code".into(),
            value: "ABCDEF0123456789-1234ABCD".into(),
        };
        assert!(n.is_retrieved("The value is ABCDEF0123456789-1234ABCD."));
        assert!(n.is_retrieved("sure! ABCDEF0123456789-1234ABCD was the code"));
        // One differing hex char is a failure (no fuzzy matching).
        assert!(!n.is_retrieved("The value is ABCDEF0123456789-1234ABCE."));
        assert!(!n.is_retrieved("I don't know."));
        assert!(!n.is_retrieved(""));
    }

    // ── Document generation ───────────────────────────────────────────────

    #[test]
    fn document_depth_zero_puts_needle_at_the_front() {
        let needle = Needle::random();
        let doc = build_document(500, 0, &needle, None);
        // Depth 0: the needle sentence opens the document, the query
        // closes it.
        assert!(
            doc.text.starts_with(&needle.sentence()),
            "doc: {}",
            &doc.text[..80]
        );
        assert!(doc.text.ends_with(&needle.query()));
        assert!(doc.text.contains(&needle.value));
        assert_eq!(doc.depth_percent, 0);
    }

    #[test]
    fn document_depth_100_puts_needle_at_the_end() {
        let needle = Needle::random();
        let doc = build_document(500, 100, &needle, None);
        // Depth 100: the needle comes after the last haystack sentence
        // and before the trailing query.
        let haystack_tail = BASE_SENTENCES
            .iter()
            .cloned()
            .cycle()
            .take(3)
            .collect::<Vec<_>>()
            .join(" ");
        let needle_pos = doc.text.find(&needle.sentence()).unwrap();
        let last_haystack = doc.text.find(&haystack_tail).unwrap();
        assert!(
            needle_pos > last_haystack,
            "needle must come after the haystack tail"
        );
        assert!(doc.text.ends_with(&needle.query()));
    }

    #[test]
    fn document_depth_50_splits_the_haystack() {
        let needle = Needle::random();
        let doc = build_document(500, 50, &needle, None);
        let pos = doc.text.find(&needle.sentence()).unwrap();
        // Haystack on both sides of the needle.
        assert!(pos > 100, "there must be haystack before the needle");
        assert!(
            doc.text[pos + needle.sentence().len()..].contains(BASE_SENTENCES[0]),
            "there must be haystack after the needle"
        );
    }

    #[test]
    fn document_with_tokenizer_lands_on_target() {
        let tok = test_tokenizer();
        let needle = Needle::random();
        let doc = build_document(2000, 50, &needle, Some(&tok));
        assert!(!doc.estimated, "tokenizer counts are not estimates");
        // The `" apple"` filler is exactly 1 token for the word-level
        // fixture, so the pad loop lands on (or within one sentence of)
        // the target.
        assert!(
            (2000..=2015).contains(&doc.token_count),
            "token_count = {}",
            doc.token_count
        );
        assert!(doc.text.contains(&needle.value));
    }

    #[test]
    fn document_without_tokenizer_is_estimated() {
        let needle = Needle::random();
        let doc = build_document(2000, 30, &needle, None);
        assert!(doc.estimated, "chars/4 counts are flagged estimated");
        assert!(
            (doc.token_count as i64 - 2000).abs() <= 200,
            "token_count = {} (target 2000, ±10%)",
            doc.token_count
        );
        assert!(doc.text.contains(&needle.value));
    }

    // ── Classification ────────────────────────────────────────────────────

    #[test]
    fn classify_failed_nominal_and_throttled() {
        // Failure dominates (missed retrieval / no TTFT).
        assert_eq!(classify(false, 0.1, 0.1, 2000, 4000), NiahCellState::Failed);
        assert_eq!(classify(true, 0.0, 0.1, 2000, 4000), NiahCellState::Failed);
        // The baseline row itself is nominal (nothing to compare).
        assert_eq!(classify(true, 0.2, 0.2, 2000, 2000), NiahCellState::Nominal);
        // Linear scaling (2k: 0.1 → 4k: 0.2) is nominal.
        assert_eq!(classify(true, 0.2, 0.1, 2000, 4000), NiahCellState::Nominal);
        // 1.5× the linear expectation is still nominal (≤ 2×).
        assert_eq!(classify(true, 0.3, 0.1, 2000, 4000), NiahCellState::Nominal);
        // 2.5× the linear expectation is throttled.
        assert_eq!(
            classify(true, 0.5, 0.1, 2000, 4000),
            NiahCellState::Throttled
        );
        // A failed baseline (0.0) must not false-positive as throttled.
        assert_eq!(classify(true, 0.5, 0.0, 2000, 4000), NiahCellState::Nominal);
    }

    // ── Result grid ───────────────────────────────────────────────────────

    fn cell(size: u32, depth: u8, retrieved: bool, ttft: f64) -> NiahCell {
        NiahCell {
            target_tokens: size,
            depth_percent: depth,
            retrieved,
            truncated_by_reasoning: false,
            reasoning_tokens: None,
            // H2: the test fixture reports no usage block (the honest
            // "not reported" case) — the real-sum path is covered by the
            // dedicated all_engines_tokens test.
            prompt_tokens: size as u64,
            completion_tokens: None,
            ttft_s: ttft,
            prefill_tps: if ttft > 0.0 { size as f64 / ttft } else { 0.0 },
            state: NiahCellState::Failed,
            no_baseline: false,
            approximate: false,
        }
    }

    /// A cell that was *truncated by reasoning* (Option C): the stream
    /// completed OK but the model spent its whole budget on chain-of-thought —
    /// no content was committed, so the needle is not `retrieved`, but the cell
    /// is **not** a capability failure. It has a real TTFT (the first CoT
    /// frame arrived) so the prefill timing is measurable.
    fn truncated_cell(size: u32, depth: u8, ttft: f64) -> NiahCell {
        let mut c = cell(size, depth, false, ttft);
        c.truncated_by_reasoning = true;
        c
    }

    #[test]
    fn result_cell_lookup_and_accuracy() {
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 50, 100]);
        assert_eq!(r.cells.len(), 6);
        assert!(r.cell(8000, 0).is_none(), "absent size → None");
        assert!(r.cell(2000, 70).is_none(), "absent depth → None");

        // Mark every slot as *run* (target_tokens set) and retrieved, then
        // make 4000 @ 50 (row 1, col 1) miss.
        for (i, c) in r.cells.iter_mut().enumerate() {
            c.target_tokens = r.sizes[i / r.depths.len()];
            c.retrieved = true;
            c.ttft_s = 0.1;
        }
        r.cells[4].retrieved = false;

        // All 6 slots ran, so the headline is the classic full-matrix form.
        // The one non-retrieved cell is a genuine fail (not truncated), so the
        // Option C breakdown reports it separately.
        assert_eq!(r.accuracy(), (5, 6));
        assert_eq!(r.accuracy_label(), "5/6 retrieved, 1 failed (83.3%)");
        assert!(!r.cell(4000, 50).unwrap().retrieved);
        assert!(r.cell(2000, 100).unwrap().retrieved);
    }

    #[test]
    fn accuracy_counts_only_run_cells_cf31() {
        // 2 sizes × 3 depths = 6 grid slots.
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 50, 100]);
        // 2k row: 2 of 3 ran (2k @ 100 never ran); 1 of the 2 run retrieved.
        r.cells[0] = cell(2000, 0, true, 0.1);
        r.cells[1] = cell(2000, 50, false, 0.1);
        // r.cells[2] (2k @ 100) stays an unrun placeholder (target_tokens 0).
        // 4k row: nothing ran.

        // CF-31: the denominator is the 2 run cells, not all 6 — so the
        // headline now agrees with size_pass_rate(2000) (both 1/2 = 50%).
        assert_eq!(r.accuracy(), (1, 2));
        assert!((r.size_pass_rate(2000).unwrap() - 50.0).abs() < 0.1);
        assert_eq!(r.size_pass_rate(4000), None);
        // The label is honest about the partial run: 1 of the 2 that ran,
        // out of the 6-cell grid (50.0%). The other run cell is a genuine
        // fail (not truncated), reported separately (Option C).
        assert_eq!(r.accuracy_label(), "1/2 of 6 run, 1 failed (50.0%)");
    }

    #[test]
    fn compute_states_uses_smallest_size_as_baseline() {
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        r.cells[0] = cell(2000, 0, true, 0.1);
        r.cells[1] = cell(2000, 100, true, 0.1);
        // 4k @ 0: 0.3 s vs the linear expectation 0.2 s (1.5×) → nominal.
        r.cells[2] = cell(4000, 0, true, 0.3);
        // 4k @ 100: retrieval failed → failed.
        r.cells[3] = cell(4000, 100, false, 0.0);

        r.compute_states();
        assert_eq!(r.cells[0].state, NiahCellState::Nominal);
        assert_eq!(r.cells[1].state, NiahCellState::Nominal);
        assert_eq!(r.cells[2].state, NiahCellState::Nominal);
        assert_eq!(r.cells[3].state, NiahCellState::Failed);

        // Push the 4k @ 0 cell to 2.5× the expectation → throttled.
        r.cells[2].ttft_s = 0.5;
        r.compute_states();
        assert_eq!(r.cells[2].state, NiahCellState::Throttled);
    }

    #[test]
    fn compute_states_flags_missing_baseline_distinctly_cf45() {
        // 2k is the baseline row. The 2k @ 0 cell FAILED (no retrieval, no
        // TTFT), so there is no valid anchor for the 4k @ 0 cell at that depth.
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        // Baseline row (2k): @0 failed, @100 fine.
        r.cells[0] = cell(2000, 0, false, 0.0);
        r.cells[1] = cell(2000, 100, true, 0.1);
        // Larger row (4k): @0 retrieved+fast, @100 fine.
        r.cells[2] = cell(4000, 0, true, 0.2);
        r.cells[3] = cell(4000, 100, true, 0.2);

        r.compute_states();

        // CF-45: the 4k @ 0 cell retrieved the needle, but its baseline (2k @ 0)
        // failed → its prefill verdict is *unknown*, flagged distinctly (not a
        // silent, confirmed-healthy nominal).
        assert!(r.cells[2].no_baseline, "4k@0 has no valid baseline (CF-45)");
        assert_eq!(r.cells[2].state, NiahCellState::Nominal);
        // The 4k @ 100 cell HAS a valid baseline (2k @ 100 = 0.1) → not flagged.
        assert!(!r.cells[3].no_baseline, "4k@100 has a valid baseline");
        assert_eq!(r.cells[3].state, NiahCellState::Nominal);
        // The baseline row itself is never flagged (nothing to compare).
        assert!(
            !r.cells[0].no_baseline,
            "baseline row is never 'no baseline'"
        );
        assert!(
            !r.cells[1].no_baseline,
            "baseline row is never 'no baseline'"
        );
    }

    #[test]
    fn context_label_marks_approximate_haystack_cf46() {
        // A size measured with an exact tokenizer reads plain (not approximate).
        let measured = cell(128000, 50, true, 0.5);
        assert!(!measured.approximate);
        assert_eq!(measured.context_label(), "128k");
        // A size that came from the `chars/4` estimate is labeled approximate —
        // the context length is not presented as exact (CF-46).
        let mut estimated = cell(128000, 50, true, 0.5);
        estimated.approximate = true;
        assert_eq!(estimated.context_label(), "128k (approx.)");
    }

    #[test]
    fn to_needle_rows_maps_all_fields() {
        let mut r = NiahResult::new(vec![2000], vec![50]);
        r.cells[0] = cell(2000, 50, true, 0.25);
        let rows = r.to_needle_rows("sess-1");
        assert_eq!(rows.len(), 1);
        let row = &rows[0];
        assert_eq!(row.eval_id, None);
        assert_eq!(row.session_id, "sess-1");
        assert_eq!(row.context_length, Some(2000));
        assert_eq!(row.depth_percent, Some(50.0));
        assert_eq!(row.retrieved_successfully, Some(true));
        // Option C: the per-cell truncated state is persisted (false for a
        // clean retrieval).
        assert_eq!(row.truncated_by_reasoning, Some(false));
        assert_eq!(row.latency_ms, Some(250.0));
    }

    #[test]
    fn to_dict_includes_niah_results_for_json() {
        // Issue #6: the `--json` NIAH section is self-contained — per-size
        // recall, `reliable_up_to`, and the per-cell detail (each labeled via
        // `context_label`, so the `(approx.)` qualifier is present when no
        // tokenizer was supplied).
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 50, 100]);
        // 2k: all 3 retrieved. 4k: 2 of 3 retrieved.
        r.cells[0] = cell(2000, 0, true, 0.1);
        r.cells[1] = cell(2000, 50, true, 0.1);
        r.cells[2] = cell(2000, 100, true, 0.1);
        r.cells[3] = cell(4000, 0, true, 0.2);
        r.cells[4] = cell(4000, 50, true, 0.2);
        r.cells[5] = cell(4000, 100, false, 0.2);
        // No tokenizer → the sizes are the `chars/4` estimate (approximate).
        for c in r.cells.iter_mut() {
            c.approximate = true;
        }
        let v = r.to_dict();
        // Headline + raw counts.
        assert!(v["accuracy"].is_string(), "accuracy label present");
        assert_eq!(v["retrieved"], 5);
        assert_eq!(v["total"], 6);
        // Option C three-way split: cell 5 is a genuine fail (content present,
        // needle absent — NOT truncated), and no cell is truncated-by-reasoning.
        assert_eq!(v["truncated_by_reasoning"], 0);
        assert_eq!(v["failed"], 1);
        // Only 2k (100%) clears the 80% bar → reliable up to 2000.
        assert_eq!(v["reliable_up_to"], 2000);
        // Per-size recall, with the approx label.
        let sizes = v["sizes"].as_array().expect("sizes array");
        assert_eq!(sizes.len(), 2);
        assert_eq!(sizes[0]["size"], 2000);
        assert_eq!(sizes[0]["label"], "2k (approx.)");
        assert_eq!(sizes[0]["pass_rate"], 100.0);
        assert_eq!(sizes[1]["label"], "4k (approx.)");
        assert_eq!(sizes[1]["retrieved"], 2);
        assert_eq!(sizes[1]["run"], 3);
        // Per-cell detail (the custom-size result lives here too).
        let cells = v["cells"].as_array().expect("cells array");
        assert_eq!(cells.len(), 6);
        assert_eq!(cells[0]["label"], "2k (approx.)");
        assert_eq!(cells[0]["retrieved"], true);
        assert_eq!(cells[5]["retrieved"], false);
        // The per-cell `truncated_by_reasoning` flag is present on every cell
        // (false here — none of these cells are the over-thinker case).
        assert_eq!(cells[0]["truncated_by_reasoning"], false);
        assert_eq!(cells[5]["truncated_by_reasoning"], false);
    }

    // ── Per-size pass rate / reliable threshold ───────────────────────────

    #[test]
    fn size_pass_rate_and_reliable_up_to() {
        let mut r = NiahResult::new(vec![2000, 4000, 8000], vec![0, 50, 100]);
        // 2k: all 3 retrieved.
        r.cells[0] = cell(2000, 0, true, 0.1);
        r.cells[1] = cell(2000, 50, true, 0.1);
        r.cells[2] = cell(2000, 100, true, 0.1);
        // 4k: 2 of 3 retrieved.
        r.cells[3] = cell(4000, 0, true, 0.2);
        r.cells[4] = cell(4000, 50, true, 0.2);
        r.cells[5] = cell(4000, 100, false, 0.2);
        // 8k: 1 of 3 retrieved.
        r.cells[6] = cell(8000, 0, true, 0.4);
        r.cells[7] = cell(8000, 50, false, 0.4);
        r.cells[8] = cell(8000, 100, false, 0.4);

        assert_eq!(r.size_pass_rate(2000), Some(100.0));
        assert!((r.size_pass_rate(4000).unwrap() - 66.666).abs() < 0.1);
        assert!((r.size_pass_rate(8000).unwrap() - 33.333).abs() < 0.1);
        // Only 2k (100%) clears 80%; 4k (66.7%) clears 60%.
        assert_eq!(r.reliable_up_to(80.0), Some(2000));
        assert_eq!(r.reliable_up_to(60.0), Some(4000));
        assert_eq!(r.reliable_up_to(101.0), None);
    }

    #[test]
    fn unrun_size_has_no_pass_rate() {
        let r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        // Nothing run (all `unrun()` placeholders → target_tokens == 0).
        assert_eq!(r.size_pass_rate(2000), None);
        assert_eq!(r.size_pass_rate(4000), None);
        assert_eq!(r.reliable_up_to(80.0), None);
        assert!(r.interpretation().contains("No context size met"));
    }

    #[test]
    fn interpretation_names_the_reliable_threshold() {
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        r.cells[0] = cell(2000, 0, true, 0.1);
        r.cells[1] = cell(2000, 100, true, 0.1);
        r.cells[2] = cell(4000, 0, true, 0.2);
        r.cells[3] = cell(4000, 100, true, 0.2);
        let interp = r.interpretation();
        assert!(interp.contains("up to ~4k"), "{interp}");
        assert!(interp.contains("For RAG"), "{interp}");
    }

    // ── F8: the interpretation size label uses the shared human_size ────

    #[test]
    fn f8_interpretation_renders_a_million_size_as_1m() {
        // A 1 000 000-token reliable size renders as "1M" — the old
        // `size / 1000` + `k` format read it as "~1000k".
        let mut r = NiahResult::new(vec![1_000_000], vec![0, 100]);
        for c in &mut r.cells {
            c.target_tokens = 1_000_000;
            c.retrieved = true;
            c.ttft_s = 0.5;
            c.state = NiahCellState::Nominal;
        }
        let interp = r.interpretation();
        assert!(interp.contains("up to ~1M tokens"), "{interp}");
        assert!(interp.contains("windows to 1M"), "{interp}");
        assert!(!interp.contains("1000k"), "a 1M size must not read 1000k: {interp}");

        // A fractional million renders with a single decimal.
        let mut r = NiahResult::new(vec![1_500_000], vec![0, 100]);
        for c in &mut r.cells {
            c.target_tokens = 1_500_000;
            c.retrieved = true;
            c.ttft_s = 0.5;
            c.state = NiahCellState::Nominal;
        }
        let interp = r.interpretation();
        assert!(interp.contains("up to ~1.5M tokens"), "{interp}");
    }

    // ── Option C: three-way classification (pass / truncated / fail) ─────

    #[test]
    fn truncated_cell_is_inconclusive_not_a_failure() {
        // A cell whose stream completed OK but was all-CoT / zero-content is
        // `truncated_by_reasoning`: not a clean pass, and NOT a capability
        // failure (the model was generating; we just never saw an answer).
        let mut r = NiahResult::new(vec![2000], vec![0]);
        r.cells[0] = truncated_cell(2000, 0, 0.1);

        assert!(!r.cells[0].retrieved, "no content → not retrieved");
        assert!(r.cells[0].truncated_by_reasoning);

        // Counted as truncated, not as a fail.
        assert_eq!(r.truncated_count(), 1);
        assert_eq!(r.failed_count(), 0, "a truncated cell is not a failure");
        // The headline breaks it out separately from the pass/total figure.
        assert_eq!(
            r.accuracy_label(),
            "0/1 retrieved, 1 truncated-by-reasoning (0.0%)"
        );
        // The capability pass rate excludes it (an all-truncated row → None,
        // not a false 0%).
        assert_eq!(r.size_pass_rate(2000), None);
        // …and an unmeasurable size is never "reliable up to".
        assert_eq!(r.reliable_up_to(80.0), None);
    }

    #[test]
    fn content_but_no_needle_is_a_genuine_fail() {
        // A cell with visible content but the needle absent is a genuine
        // retrieval failure — distinct from a truncated (zero-content) cell.
        let mut r = NiahResult::new(vec![2000], vec![0]);
        r.cells[0] = cell(2000, 0, false, 0.1); // retrieved=false, truncated=false

        assert_eq!(r.failed_count(), 1);
        assert_eq!(
            r.truncated_count(),
            0,
            "a content-present fail is not truncated"
        );
        assert_eq!(r.accuracy_label(), "0/1 retrieved, 1 failed (0.0%)");
        // The capability pass rate counts it as a non-pass (denominator 1).
        assert_eq!(r.size_pass_rate(2000), Some(0.0));
    }

    #[test]
    fn needle_in_content_is_a_clean_pass() {
        // A cell with the needle in the content is a pass — and a clean run's
        // label reads exactly as before the Option C change (no clauses).
        let mut r = NiahResult::new(vec![2000], vec![0]);
        r.cells[0] = cell(2000, 0, true, 0.1); // retrieved=true

        assert_eq!(r.accuracy(), (1, 1));
        assert_eq!(r.failed_count(), 0);
        assert_eq!(r.truncated_count(), 0);
        assert_eq!(r.accuracy_label(), "1/1 retrieved (100.0%)");
        assert_eq!(r.size_pass_rate(2000), Some(100.0));
    }

    #[test]
    fn aggregate_label_reads_74_of_77_with_3_truncated() {
        // The headline format from REASONING_DESIGN.md: `74/77 pass, 3
        // truncated_by_reasoning` (pass/total, truncated broken out; no
        // genuine fails in this case).
        let sizes = vec![2000, 4000, 8000, 16000, 32000, 64000, 128000];
        let depths: Vec<u8> = (0..=100).step_by(10).collect(); // 11 depths
        let mut r = NiahResult::new(sizes, depths);
        assert_eq!(r.cells.len(), 77, "7 sizes × 11 depths");
        // 74 pass, 3 truncated, 0 fail.
        for (i, c) in r.cells.iter_mut().enumerate() {
            c.target_tokens = r.sizes[i / r.depths.len()];
            c.ttft_s = 0.1;
            if i < 74 {
                c.retrieved = true;
            } else {
                c.truncated_by_reasoning = true;
            }
        }
        assert_eq!(r.accuracy(), (74, 77));
        assert_eq!(r.truncated_count(), 3);
        assert_eq!(r.failed_count(), 0);
        assert_eq!(
            r.accuracy_label(),
            "74/77 retrieved, 3 truncated-by-reasoning (96.1%)"
        );
    }

    #[test]
    fn to_dict_carries_truncated_flag_per_cell_and_aggregate() {
        // Option C: the `--json` niah section carries the per-cell flag and the
        // aggregate three-way split.
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        r.cells[0] = cell(2000, 0, true, 0.1); // pass
        r.cells[1] = truncated_cell(2000, 100, 0.1); // truncated
        r.cells[2] = cell(4000, 0, true, 0.2); // pass
        r.cells[3] = cell(4000, 100, false, 0.2); // genuine fail
        let v = r.to_dict();
        // Aggregate split.
        assert_eq!(v["retrieved"], 2);
        assert_eq!(v["total"], 4);
        assert_eq!(v["truncated_by_reasoning"], 1);
        assert_eq!(v["failed"], 1);
        // Per-cell flags.
        let cells = v["cells"].as_array().expect("cells array");
        assert_eq!(cells[0]["truncated_by_reasoning"], false);
        assert_eq!(cells[1]["truncated_by_reasoning"], true);
        assert_eq!(cells[2]["truncated_by_reasoning"], false);
        assert_eq!(cells[3]["truncated_by_reasoning"], false);
    }

    // ── Slot ──────────────────────────────────────────────────────────────

    #[test]
    fn slot_starts_empty_and_publishes() {
        let slot = NiahSlot::new();
        assert!(slot.load().is_none());
        assert!(!slot.is_running());

        let mut r = NiahResult::new(vec![2000], vec![0]);
        r.cells[0] = cell(2000, 0, true, 0.1);
        slot.set_running(true);
        assert!(slot.is_running());
        slot.store(r);
        slot.set_running(false);

        assert!(!slot.is_running());
        let loaded = slot.load();
        assert!(loaded.is_some());
        assert_eq!(loaded.as_ref().as_ref().unwrap().accuracy(), (1, 1));
    }

    // ── human_size shared formatter (Issue #3) ────────────────────────

    #[test]
    fn human_size_formats_k_and_m() {
        assert_eq!(human_size(2000), "2k");
        assert_eq!(human_size(128000), "128k");
        assert_eq!(human_size(256000), "256k");
        assert_eq!(human_size(500000), "500k");
        assert_eq!(human_size(1_000_000), "1M");
        assert_eq!(human_size(2_000_000), "2M");
        // Fractional sizes get a single decimal.
        assert_eq!(human_size(1500), "1.5k");
        assert_eq!(human_size(1_500_000), "1.5M");
        // Sub-1k counts are shown as-is.
        assert_eq!(human_size(999), "999");
    }

    #[test]
    fn context_label_uses_human_size_and_marks_approx() {
        // Exact (tokenizer) at 1M reads plain "1M" (was "1000k" before #4).
        assert_eq!(cell(1_000_000, 50, true, 0.5).context_label(), "1M");
        // The chars/4 estimate at 256k is labeled approximate (C1, #4).
        let mut approx = cell(256000, 50, true, 0.5);
        approx.approximate = true;
        assert_eq!(approx.context_label(), "256k (approx.)");
        // 128k exact (the old CF-46 case, now rendered via human_size).
        assert_eq!(cell(128000, 50, true, 0.5).context_label(), "128k");
    }

    #[test]
    fn context_size_label_marks_approx() {
        // Exact (tokenizer) reads plain.
        assert_eq!(context_size_label(128_000, false), "128k");
        assert_eq!(context_size_label(1_000_000, false), "1M");
        // The `chars/4` estimate carries the qualifier (Issue #4).
        assert_eq!(context_size_label(128_000, true), "128k (approx.)");
        assert_eq!(context_size_label(1_000_000, true), "1M (approx.)");
    }

    // C3 (qualify once): the per-cell worker tag no longer repeats
    // `(approx.)` (the v0.1.7 log had 114 hits) — it is a plain
    // `C1:{size}-d{depth}`; the one-time qualification lives in the
    // `[CONFIG]` line + the C1 start line.
    #[test]
    fn cell_tag_is_plain_and_uses_human_size() {
        assert_eq!(cell_tag(128_000, 50), "C1:128k-d50");
        assert_eq!(cell_tag(1_000_000, 0), "C1:1M-d0");
        assert_eq!(cell_tag(2_000, 100), "C1:2k-d100");
        // No `(approx.)` on the tag (the estimate case is the same).
        assert!(!cell_tag(128_000, 50).contains("approx"));
    }

    // ── large haystack scaling (Issue #5) ─────────────────────────────

    #[test]
    fn build_document_scales_to_a_large_size_with_approx_label() {
        // 200k tokens (beyond the old 128k ceiling), no tokenizer → the
        // chars/4 estimate is flagged `estimated` (the "approx" label).
        let needle = Needle::random();
        let doc = build_document(200_000, 50, &needle, None);
        assert!(
            doc.estimated,
            "no tokenizer → the size is a chars/4 estimate"
        );
        assert_eq!(doc.target_tokens, 200_000);
        // The achieved count lands within ±10% of the 200k target.
        assert!(
            (doc.token_count as i64 - 200_000).abs() <= 20_000,
            "token_count = {} (target 200k, ±10%)",
            doc.token_count
        );
        // Depth 50: the needle sits at the middle of the document.
        let pos = doc.text.find(&needle.sentence()).expect("needle present");
        let frac = pos as f64 / doc.text.len() as f64;
        assert!(
            (frac - 0.5).abs() < 0.1,
            "needle at {:.1}% of the document (expected ~50%)",
            frac * 100.0
        );
        // A 200k-token haystack is ~800k chars (≈4 bytes/token).
        assert!(
            doc.text.len() > 700_000,
            "haystack ~800k chars, got {}",
            doc.text.len()
        );
    }

    // ── matrix runner honors the config ladders (Issue #5) ────────────

    #[test]
    fn engine_honors_custom_context_and_depth_ladders() {
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            niah_contexts: vec![16_000, 256_000, 1_000_000],
            niah_depths: vec![0, 50, 100],
            niah_custom: None,
            ..Config::default()
        };
        let engine = NiahEngine::new(&cfg).unwrap();
        assert_eq!(engine.sizes_list(), &[16_000, 256_000, 1_000_000]);
        assert_eq!(engine.depths_list(), &[0, 50, 100]);
    }

    #[test]
    fn engine_custom_single_size_collapses_to_one_row() {
        // The focused "count test": `niah_custom` takes precedence over
        // `niah_contexts` and the matrix is a single size row.
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            niah_contexts: vec![2_000, 4_000],
            niah_depths: vec![0, 50, 100],
            niah_custom: Some(500_000),
            ..Config::default()
        };
        let engine = NiahEngine::new(&cfg).unwrap();
        assert_eq!(engine.sizes_list(), &[500_000]);
        assert_eq!(engine.depths_list(), &[0, 50, 100]);
    }

    #[test]
    fn engine_defaults_to_the_builtin_ladders_when_unconfigured() {
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            ..Config::default()
        };
        let engine = NiahEngine::new(&cfg).unwrap();
        assert_eq!(engine.sizes_list(), NIAH_SIZES.as_slice());
        assert_eq!(engine.depths_list(), NIAH_DEPTHS.as_slice());
    }

    #[test]
    fn niah_engine_config_carries_and_applies_the_ladders() {
        // The TUI standalone `n`-key path builds the engine via
        // NiahEngineConfig::from_config + engine().
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            niah_contexts: vec![64_000, 256_000],
            niah_depths: vec![10, 90],
            niah_custom: Some(1_000_000),
            ..Config::default()
        };
        let nec = NiahEngineConfig::from_config(&cfg);
        assert_eq!(nec.niah_contexts, vec![64_000, 256_000]);
        assert_eq!(nec.niah_depths, vec![10, 90]);
        assert_eq!(nec.niah_custom, Some(1_000_000));
        let engine = nec.engine().unwrap();
        assert_eq!(engine.sizes_list(), &[1_000_000]);
        assert_eq!(engine.depths_list(), &[10, 90]);
    }

    // ── R3: the configurable generation budget ─────────────────────────

    #[test]
    fn engine_honors_the_configured_generation_budget() {
        // The resolved config budget (R3) reaches the engine and is used
        // for the per-cell worker / snapshots (not the built-in default).
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            niah_max_tokens: 512,
            ..Config::default()
        };
        let engine = NiahEngine::new(&cfg).unwrap();
        assert_eq!(engine.max_gen_tokens(), 512);
        // The default config carries the built-in safety-net budget.
        let engine = NiahEngine::new(&Config::default()).unwrap();
        assert_eq!(engine.max_gen_tokens(), NIAH_MAX_GEN_TOKENS);
    }

    #[test]
    fn niah_engine_config_carries_the_generation_budget() {
        // The TUI `n`-key path carries the budget through
        // `NiahEngineConfig` too.
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            niah_max_tokens: 1024,
            ..Config::default()
        };
        let nec = NiahEngineConfig::from_config(&cfg);
        assert_eq!(nec.max_gen_tokens, 1024);
        let engine = nec.engine().unwrap();
        assert_eq!(engine.max_gen_tokens(), 1024);
    }

    #[test]
    fn engine_honors_the_disable_thinking_switch() {
        // R3: the resolved suppression switch reaches the engine (on by
        // default; `--no-disable-thinking` → off, the force-test path).
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            disable_thinking: false,
            ..Config::default()
        };
        let engine = NiahEngine::new(&cfg).unwrap();
        assert!(!engine.disable_thinking());
        // The default config keeps suppression on.
        let engine = NiahEngine::new(&Config::default()).unwrap();
        assert!(engine.disable_thinking());
    }

    #[test]
    fn to_dict_carries_the_per_cell_reasoning_tokens() {
        // R3: the `--json` niah section carries each cell's
        // server-reported CoT tokens (`null` when the backend reported no
        // usage block) — the calibration measurement.
        let mut r = NiahResult::new(vec![2000], vec![0, 100]);
        r.cells[0] = cell(2000, 0, true, 0.1);
        let mut c = cell(2000, 100, false, 0.1);
        c.reasoning_tokens = Some(37);
        c.truncated_by_reasoning = true;
        r.cells[1] = c;
        let v = r.to_dict();
        let cells = v["cells"].as_array().expect("cells array");
        assert_eq!(cells[0]["reasoning_tokens"], serde_json::Value::Null);
        assert_eq!(cells[1]["reasoning_tokens"], 37);
    }
}
