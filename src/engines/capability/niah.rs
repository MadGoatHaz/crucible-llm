//! Engine C1 — Needle-In-A-Haystack (NIAH) Context Retention (plan Chunk
//! 15, blueprint §5 Engine C1).
//!
//! A *needle* — a unique, cryptographically random key-value fact — is
//! inserted at a chosen depth (0%…100%) of a synthetic long-context
//! *haystack* document scaled to 2k/4k/8k/16k/32k/64k/128k tokens. The
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

use crate::client::{StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::prompt::{count_tokens, Tokenizer, BASE_SENTENCES, FILLER};
use crate::sse::Chunk;
use crate::storage::models::NeedleEvaluation;
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Bounded worker→engine channel capacity (same as Engine A / C2).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per cell: the answer is the value only, so a
/// short budget keeps the matrix fast.
pub const NIAH_MAX_GEN_TOKENS: u32 = 64;

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
fn haystack_sentences(target_tokens: u32) -> Vec<String> {
    let target_chars = (target_tokens * 4) as usize;
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

/// Classify one cell.
///
/// * `!retrieved` (or no TTFT) → [`NiahCellState::Failed`];
/// * the baseline row itself (`tokens <= base_tokens`), or a missing
///   baseline (the smallest-size cell failed) → [`NiahCellState::Nominal`]
///   (nothing to compare — never a false throttle);
/// * TTFT within `PREFILL_THROTTLE_FACTOR`× the linear expectation
///   (`base_ttft × tokens / base_tokens`) → [`NiahCellState::Nominal`];
/// * beyond that → [`NiahCellState::Throttled`].
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

/// One size×depth cell of the matrix.
#[derive(Debug, Clone)]
pub struct NiahCell {
    pub target_tokens: u32,
    pub depth_percent: u8,
    /// `true` when the needle value was retrieved from the response.
    pub retrieved: bool,
    /// Time-to-first-token, seconds (`T3 − T1`; `0.0` on stream failure).
    pub ttft_s: f64,
    /// Prefill throughput, tokens/s (prompt tokens ÷ TTFT).
    pub prefill_tps: f64,
    /// The ANSI class View 3 renders (assigned by [`NiahResult::compute_states`]).
    pub state: NiahCellState,
}

impl NiahCell {
    /// An unevaluated cell (the empty-grid placeholder).
    pub fn unrun() -> Self {
        Self {
            target_tokens: 0,
            depth_percent: 0,
            retrieved: false,
            ttft_s: 0.0,
            prefill_tps: 0.0,
            state: NiahCellState::Failed,
        }
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

    /// `(retrieved, total)` over all cells.
    pub fn accuracy(&self) -> (usize, usize) {
        let total = self.cells.len();
        let retrieved = self.cells.iter().filter(|c| c.retrieved).count();
        (retrieved, total)
    }

    /// `71/77 retrieved (92.2%)` — the headline accuracy figure.
    pub fn accuracy_label(&self) -> String {
        let (r, t) = self.accuracy();
        let pct = if t > 0 {
            r as f64 / t as f64 * 100.0
        } else {
            0.0
        };
        format!("{r}/{t} retrieved ({pct:.1}%)")
    }

    /// The pass rate (0..=100) for one context size across all its depths —
    /// the per-row percentage View 3 shows in its `PASS %` column. `None`
    /// when the size has no *run* cells (an unrun row shows `--`).
    pub fn size_pass_rate(&self, size: u32) -> Option<f64> {
        let si = self.sizes.iter().position(|s| *s == size)?;
        let start = si * self.depths.len();
        let cells = &self.cells[start..start + self.depths.len()];
        // Unrun cells are the `unrun()` placeholder (`target_tokens == 0`).
        let run = cells.iter().filter(|c| c.target_tokens > 0).count();
        if run == 0 {
            return None;
        }
        let retrieved = cells.iter().filter(|c| c.retrieved).count();
        Some(retrieved as f64 / run as f64 * 100.0)
    }

    /// The largest context size whose pass rate is >= `min_pct` — the
    /// "reliable up to ~Xk" threshold for the practical interpretation line.
    /// `None` when no size clears the bar.
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
    pub fn interpretation(&self) -> String {
        match self.reliable_up_to(80.0) {
            Some(size) => format!(
                "Reliable retrieval up to ~{}k tokens. Beyond that, the model \
                 loses track of embedded information. For RAG: limit context \
                 windows to {}k or use a model with better long-context training.",
                size / 1000,
                size / 1000
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
                latency_ms: Some(c.ttft_s * 1000.0),
            })
            .collect()
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
    /// Engines A/C2) with the default 7×11 matrix.
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        let tokenizer = match &cfg.tokenizer {
            Some(path) => Some(Arc::new(Tokenizer::from_file(path)?)),
            None => None,
        };
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            tokenizer,
            sizes: NIAH_SIZES.to_vec(),
            depths: NIAH_DEPTHS.to_vec(),
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
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
            NIAH_MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        .tag(format!("C1:{size}k-d{depth}"));
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }

        let start = MonotonicInstant::now();
        let outcome = tokio::spawn(worker.run(tx))
            .await
            .expect("niah worker task panicked");
        let mut response = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    if let Chunk::Content(text) | Chunk::Reasoning(text) = &frame.chunk {
                        response.push_str(text);
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
                        &start,
                        NIAH_MAX_GEN_TOKENS,
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
                &start,
                NIAH_MAX_GEN_TOKENS,
            ));
        }

        let retrieved = needle.is_retrieved(&response);
        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let ttft = ts.ttft_nanos().map(|n| n as f64 / 1e9).unwrap_or(0.0);
        let prompt_tokens = outcome
            .usage
            .map(|u| u.prompt_tokens)
            .unwrap_or(doc.token_count as u64);
        let prefill_tps = if ttft > 0.0 {
            prompt_tokens as f64 / ttft
        } else {
            0.0
        };
        NiahCell {
            target_tokens: size,
            depth_percent: depth,
            retrieved,
            ttft_s: ttft,
            prefill_tps,
            // Assigned by `compute_states` once the whole matrix is in.
            state: NiahCellState::Failed,
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
        }
    }

    /// Build a default-matrix runner from this config.
    pub fn engine(&self) -> Result<NiahEngine, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(self.timeout.max(1)))
            .build()?;
        let tokenizer = match &self.tokenizer {
            Some(path) => Some(Arc::new(Tokenizer::from_file(path)?)),
            None => None,
        };
        Ok(NiahEngine {
            client,
            url: self.url.clone(),
            model: self.model.clone(),
            api_key: self.api_key.clone(),
            timeout: self.timeout,
            tokenizer,
            sizes: NIAH_SIZES.to_vec(),
            depths: NIAH_DEPTHS.to_vec(),
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
            ttft_s: ttft,
            prefill_tps: if ttft > 0.0 { size as f64 / ttft } else { 0.0 },
            state: NiahCellState::Failed,
        }
    }

    #[test]
    fn result_cell_lookup_and_accuracy() {
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 50, 100]);
        assert_eq!(r.cells.len(), 6);
        assert!(r.cell(8000, 0).is_none(), "absent size → None");
        assert!(r.cell(2000, 70).is_none(), "absent depth → None");

        for c in r.cells.iter_mut() {
            c.retrieved = true;
            c.ttft_s = 0.1;
        }
        // 4000 @ 50 (row 1, col 1) misses.
        r.cells[4].retrieved = false;

        assert_eq!(r.accuracy(), (5, 6));
        assert_eq!(r.accuracy_label(), "5/6 retrieved (83.3%)");
        assert!(!r.cell(4000, 50).unwrap().retrieved);
        assert!(r.cell(2000, 100).unwrap().retrieved);
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
        assert_eq!(row.latency_ms, Some(250.0));
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
}
