//! Engine A — Speed & Latency: single-stream TTFT / PP / TG / MTP
//! orchestration (plan Chunk 7).
//!
//! This is the **single-stream CLI parity** engine: it drives one
//! [`StreamWorker`] (Chunk 5) per iteration against the configured
//! endpoint, classifies every parsed frame (Chunk 3), and computes the
//! blueprint §7 formulas —
//!
//! * `TTFT = T3 − T1` (first token arrival − request dispatched),
//! * `TG speed = completion_tokens / (stream_time − TTFT)`,
//! * `MTP η = completion_tokens / total_token_frames` (Reasoning + Content),
//!
//! with the same result-box layout, `--json` field set, and exit-code
//! semantics as `llmspeedtest.py` (`print_result_box` / `output_json` /
//! "exit 1 if all runs failed").
//!
//! **CF-05 prefill/latency split.** The old single "PP speed"
//! (`prompt_tokens / TTFT`) conflated two distinct metrics on a ~45–87-token
//! short prompt — a *latency floor* wearing a *bandwidth* label. Engine A now
//! reports them separately, each with the right instrument:
//!
//! * **first-token latency (ms)** — the **short** prompt, both the
//!   client-perceived (`T3−T0`, [`perceived_ttft_seconds`]) and the
//!   server-side (`T3−T1`, [`SpeedResult::ttft`]) values. A *latency* number.
//! * **pre-fill throughput (tok/s)** — a **long** prompt
//!   ([`SpeedEngine::run_prefill`], `prompt_tokens / server-side TTFT`). The
//!   *bandwidth* number that scales with context.
//!
//! The headless report ([`format_split_report`] / [`json_report_split`]) shows
//! both, clearly labeled, plus a pointer to C1's 2k→128k prefill-degradation
//! curve for the full scaling picture. Decode / ITL / MTP are intact.
//!
//! **CF-12 — the canonical MTP η (the contract siblings consume).**
//! `mtp_eta = completion_tokens / total_token_frames`, where
//! `total_token_frames` is the count of **all** token frames observed in the
//! stream (Reasoning + Content combined). This is the single definition
//! Engine A applies ([`mtp_eta`], [`SpeedResult::mtp_efficiency`], the live
//! snapshot) and the one Engine B (P2-C) mirrors and the methodology block
//! (P2-F) documents verbatim. **`η > 1` is legitimate** — it arises from
//! multi-token / speculative batching (vLLM MTP, EAGLE), where one SSE frame
//! carries several accepted tokens, not only from true MTP. The old
//! content-chunks-only denominator (which read `0` on a pure-CoT stream and
//! undercounted batched servers) is gone.
//!
//! Measurement-isolation note: all timing is captured in the worker
//! (quanta, `T0..Tn`); this module only takes deltas of those records.
//! Nothing here touches the TUI or the timing path.

use std::sync::Arc;
use std::time::Duration;

use serde_json::json;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::client::{
    authoritative_tokens, join_worker, spawn_worker, stream_text, token_window, StreamError,
    StreamEvent, StreamOutcome, StreamWorker,
};
use crate::config::{Config, Mode};
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::histogram::LatencyHistogram;
use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamMetric, StreamStatus};
use crate::prompt::{count_tokens, GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::{Chunk, Usage, UsageDetails};
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Max generation tokens per run (parity with the prototype's
/// `DEFAULT_MAX_GEN_TOKENS`).
pub const MAX_GEN_TOKENS: u32 = 256;

/// Bounded worker→engine channel capacity.
const CHANNEL_CAPACITY: usize = 256;

/// Metrics from a single speed-test run (parity with the prototype's
/// `TestResult`).
#[derive(Debug, Clone)]
pub struct SpeedResult {
    /// Time-to-first-token, seconds (`T3 − T1`; falls back to the total
    /// stream time when no token frame arrived).
    pub ttft: f64,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    /// Prefill bandwidth, tokens/s (`prompt_tokens / TTFT`).
    pub pp_speed: f64,
    /// Decode speed, tokens/s. CF-26: the **answer-phase** rate (content
    /// tokens over the content window) when the server reports
    /// `reasoning_tokens`; the full-window blended rate otherwise.
    pub tg_speed: f64,
    /// MTP acceptance ratio (`η`, CF-12 canonical):
    /// `completion_tokens / total_token_frames` (Reasoning + Content).
    /// `> 1` is legitimate (multi-token / speculative batching).
    pub mtp_efficiency: f64,
    /// Full stream duration, seconds (`T_n − T0`).
    pub stream_time: f64,
    /// All non-terminator frames (the `[DONE]` marker is not a chunk).
    pub total_chunks: u64,
    pub content_chunks: u64,
    pub reasoning_chunks: u64,
    pub other_chunks: u64,
    /// `true` when any token count came from the `chars/4` estimate.
    pub estimated: bool,
    pub model: String,
    pub mode: String,
    /// `Some` on a hard failure, or as a warning note (e.g. skipped
    /// malformed frames) on an otherwise successful run.
    pub error: Option<String>,
    /// The decode-loop guard (v0.1.1) flagged this run: its throughput
    /// numbers are zeroed (excluded) and it is kept out of summaries.
    pub looping: bool,
    /// CF-10 (P2-A surfacing): the server's `usage.completion_tokens`
    /// exceeded ~4× the token frames observed on the wire (a
    /// max_tokens-inflating backend) — the decode-rate layer corrected the
    /// count toward the wire truth.
    pub usage_discrepancy: bool,
    /// CF-44 (P2-A surfacing): a non-streaming (plain-JSON) result — its
    /// "TTFT" is total response latency and it is excluded from ITL/decode.
    pub e2e_only: bool,
    /// CF-24 (P2-A surfacing): ITL samples clamped at the histogram's
    /// 60 s bound — the ITL percentiles are capped, not exact.
    pub itl_clamped: u64,
}

impl SpeedResult {
    /// The prototype's `all_failed` condition: an error with zero
    /// completion tokens.
    pub fn is_failed(&self) -> bool {
        self.error.is_some() && self.completion_tokens == 0
    }

    /// CF-25: the run is **degraded** when no token frame was ever received
    /// (a pure control / usage stream, or a failure) — its TTFT is the
    /// `0.0` N/A sentinel, not a measurement. Derived from the stored chunk
    /// counts (no new field) so the `SpeedResult` field set stays stable for
    /// the sibling construction sites.
    #[must_use]
    pub fn is_degraded(&self) -> bool {
        self.content_chunks + self.reasoning_chunks == 0
    }

    /// The `--json` object for this run — **exactly** the key set of the
    /// prototype's `TestResult.to_dict` (note: `other_chunks` is
    /// deliberately absent, matching the prototype).
    pub fn to_dict(&self) -> serde_json::Value {
        json!({
            "ttft_s": round(self.ttft, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "pp_speed_tok_s": round(self.pp_speed, 2),
            "tg_speed_tok_s": round(self.tg_speed, 2),
            "mtp_efficiency": round(self.mtp_efficiency, 4),
            "stream_time_s": round(self.stream_time, 4),
            "total_chunks": self.total_chunks,
            "content_chunks": self.content_chunks,
            "reasoning_chunks": self.reasoning_chunks,
            "estimated": self.estimated,
            "model": self.model,
            "mode": self.mode,
            "error": self.error,
            "looping": self.looping,
            "degraded": self.is_degraded(),
            // P2-A surfacing (CF-10 / CF-24 / CF-44): the stream-integrity
            // flags the worker computed for this run.
            "usage_discrepancy": self.usage_discrepancy,
            "e2e_only": self.e2e_only,
            "itl_clamped": self.itl_clamped,
        })
    }
}

/// The CF-05 **pre-fill throughput** metric — the *bandwidth* half of the
/// split, distinct from first-token *latency*.
///
/// Measured on a **long** prompt (a real context size, `cfg.tokens`) so the
/// numerator is large enough that prefill *compute* dominates the TTFT
/// denominator: `tok_s = prompt_tokens / server-side TTFT (T3−T1)`. This is
/// the number that scales with context — the honest replacement for the old
/// single "PP speed", which divided a ~45–87-token short prompt by an
/// overhead-dominated TTFT (a latency floor wearing a bandwidth label).
#[derive(Debug, Clone)]
pub struct PrefillMetric {
    /// Pre-fill throughput, tokens/s (`prompt_tokens / server-side TTFT`),
    /// measured on the long prompt.
    pub tok_s: f64,
    /// The long prompt's token count (the numerator).
    pub prompt_tokens: u64,
    /// The long prompt's server-side TTFT, seconds (T3−T1, the denominator).
    pub server_ttft_s: f64,
    /// `true` when the token count came from the `chars/4` estimate.
    pub estimated: bool,
}

impl PrefillMetric {
    /// `true` when the measurement is usable (a real prefill window).
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.server_ttft_s > 0.0 && self.prompt_tokens > 0
    }
}

/// Round to `decimals` places (the prototype's `round(x, n)` in
/// `to_dict` / `output_json`).
fn round(v: f64, decimals: u32) -> f64 {
    let f = 10f64.powi(decimals as i32);
    (v * f).round() / f
}

/// The mean of `values`, `0.0` when empty (the CF-05 split report averages).
fn mean_of(values: &[f64]) -> f64 {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<f64>() / values.len() as f64
    }
}

/// The **canonical MTP η** (CF-12) — the speculative / MTP acceptance ratio.
///
/// `mtp_eta = completion_tokens / total_token_frames`, where
/// `total_token_frames` is the count of **all** token frames observed in the
/// stream (Reasoning + Content combined).
///
/// **`η > 1` is legitimate**: it arises from multi-token / speculative
/// batching (vLLM MTP, EAGLE, …), where a single SSE frame carries several
/// accepted tokens — not only from true MTP. A server that emits one token
/// per frame yields `η ≈ 1`; a batching server yields `η > 1`.
///
/// This is the single definition Engine A applies ([`SpeedResult::mtp_efficiency`]
/// and the live snapshot) and the one Engine B (P2-C) mirrors and the
/// methodology block (P2-F) documents verbatim. `0.0` (N/A) when no token
/// frame was observed — never a division by zero.
#[must_use]
pub fn mtp_eta(completion_tokens: u64, total_token_frames: u64) -> f64 {
    if total_token_frames == 0 {
        0.0
    } else {
        completion_tokens as f64 / total_token_frames as f64
    }
}

/// CF-26: the **cache-hit** flag — `true` when the server reports
/// `UsageDetails.cached_tokens > 0` (a warm KV / prefix-cache hit). `false`
/// when the detail is absent (`None`) or reports zero cached tokens.
///
/// The CF-02 SSE parser already extracts `cached_tokens`; this is the
/// canonical derivation. Wiring the live value into a stored
/// [`crate::storage::models::StreamMetricRow::cache_hit`] needs the parser's
/// `UsageDetails` to reach the engine — a `client/stream.rs` propagation
/// seam owned by P2-A (see the module docs and `work/scratch/p2b_report.md`).
#[must_use]
pub fn cache_hit(details: Option<&UsageDetails>) -> bool {
    details.is_some_and(|d| d.cached_tokens > 0)
}

/// The (first, last) **content** token-arrival instants — the CF-26
/// **answer-phase decode window** (split at the reasoning→content
/// transition). It excludes the thinking phase before the first content
/// frame. `(None, None)` when no content frame arrived (a pure-CoT stream).
#[must_use]
fn content_window(events: &[StreamEvent]) -> (Option<MonotonicInstant>, Option<MonotonicInstant>) {
    let mut first: Option<MonotonicInstant> = None;
    let mut last: Option<MonotonicInstant> = None;
    for e in events {
        if let StreamEvent::Frame { frame, at, .. } = e {
            if matches!(frame.chunk, Chunk::Content(_)) {
                if first.is_none() {
                    first = Some(*at);
                }
                last = Some(*at);
            }
        }
    }
    (first, last)
}

/// The run's **inter-token latency histogram** (CF-09/CF-24): one sample
/// per token-frame gap, recorded from the worker's per-frame arrival
/// stamps (CF-09: a batched MTP frame no longer collapses to a zero
/// delta). Shared by the live snapshot and [`aggregate`], which reads
/// [`LatencyHistogram::clamp_count`] for the CF-24 surfacing.
#[must_use]
fn itl_histogram(events: &[StreamEvent]) -> LatencyHistogram {
    let mut itl = LatencyHistogram::default();
    let mut last: Option<MonotonicInstant> = None;
    for event in events {
        if let StreamEvent::Frame { frame, at, .. } = event {
            if frame.chunk.is_token() {
                if let Some(prev) = last {
                    itl.record(prev.delta_nanos(at));
                }
                last = Some(*at);
            }
        }
    }
    itl
}

/// Engine construction errors.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The shared `reqwest` client could not be built.
    #[error("failed to build HTTP client: {0}")]
    Client(#[from] reqwest::Error),
    /// An explicit `--tokenizer` file failed to load/parse.
    #[error(transparent)]
    Tokenizer(#[from] TokenizerError),
}

/// One benchmark run (N sequential single-stream iterations).
#[derive(Debug)]
pub struct SpeedEngine {
    cfg: Config,
    client: reqwest::Client,
    generator: PromptGenerator,
    /// Optional TUI seam: publish live [`MetricsSnapshot`]s while the
    /// engine runs (the render loop reads them lock-free).
    metrics: Option<Arc<MetricsState>>,
    /// Optional sequence seam: publish [`EngineProgress`] (iteration /
    /// tokens) to the [`ProgressBus`] the Benchmark Sequence mirrors into
    /// the TUI's sequence header (the render loop reads it lock-free).
    progress: Option<Arc<ProgressBus>>,
    /// Optional `Space`-key pause gate: the run loop waits on it before
    /// spawning each new iteration (in-flight streams complete).
    pause: Option<Arc<RunPause>>,
    /// Optional run logger: the engine records per-iteration start /
    /// outcome (the worker itself logs the HTTP / SSE detail).
    logger: Option<Arc<RunLogger>>,
}

impl SpeedEngine {
    /// Build the engine from the resolved config.
    ///
    /// An explicit `--tokenizer` file that fails to load is a hard error
    /// (the user asked for exact counts); without the flag, counts fall
    /// back to `chars/4` and are flagged `estimated` (graceful
    /// degradation, blueprint §3).
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        let generator = match &cfg.tokenizer {
            Some(path) => PromptGenerator::new(Some(Tokenizer::from_file(path)?)),
            None => PromptGenerator::new(None),
        };
        Ok(Self {
            cfg: cfg.clone(),
            client,
            generator,
            metrics: None,
            progress: None,
            pause: None,
            logger: None,
        })
    }

    /// Attach a [`MetricsState`] publisher so the engine pushes live
    /// snapshots to the TUI while running (measurement-isolation invariant,
    /// blueprint §4). The TUI's `start_run` wires this; the headless path
    /// leaves it `None`.
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach a [`ProgressBus`] so the engine reports `Iteration
    /// {current}/{total}` + tokens generated to the Benchmark Sequence
    /// (the TUI's progress bar). The headless path leaves it `None`.
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach the `Space`-key pause gate: each iteration waits on it
    /// before the worker is spawned (the headless path leaves it `None`).
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger: each iteration then logs its start and
    /// outcome (the worker logs the HTTP / SSE lifecycle detail).
    pub fn logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// The prompt for this run (generated once and reused across
    /// iterations — parity with the prototype; `--nocache` prepends a
    /// unique uuid prefix that busts the server KV cache).
    pub fn generate_prompt(&self) -> GeneratedPrompt {
        match (self.cfg.mode, self.cfg.nocache) {
            (Mode::Short, false) => self.generator.short(),
            (Mode::Short, true) => self.generator.nocache_short(),
            (Mode::Long, false) => self.generator.long(self.cfg.tokens),
            (Mode::Long, true) => self.generator.nocache_long(self.cfg.tokens),
        }
    }

    /// One full iteration: run the worker, drain its bounded channel, and
    /// synthesize the §7 metrics.
    pub async fn run_iteration(&self, prompt: &GeneratedPrompt) -> SpeedResult {
        self.run_iteration_events(prompt).await.0
    }

    /// [`run_iteration`] plus the raw worker events (Chunk 13): the
    /// per-packet arrival timestamps the CSV export dumps. The events are
    /// drained and returned as-is; the metrics are synthesized exactly as
    /// in [`run_iteration`].
    pub async fn run_iteration_events(
        &self,
        prompt: &GeneratedPrompt,
    ) -> (SpeedResult, Vec<StreamEvent>) {
        self.run_iteration_events_tagged(prompt, "A").await
    }

    /// [`run_iteration_events`] with an explicit log tag prefix (the
    /// sequence executor tags iterations `A:1`, `A:2`, …).
    pub async fn run_iteration_events_tagged(
        &self,
        prompt: &GeneratedPrompt,
        tag_prefix: &str,
    ) -> (SpeedResult, Vec<StreamEvent>) {
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)))
        .tag(tag_prefix);
        if let Some(key) = &self.cfg.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }

        // Spawn the worker without awaiting it: the channel must be
        // drained concurrently, or a >capacity frame count deadlocks
        // the worker's `tx.send()` (server-agnostic fix: llama.cpp,
        // LM Studio, Unsloth emit one SSE frame per token including
        // reasoning_content, easily exceeding the 256-slot channel).
        let handle = spawn_worker(worker, tx);
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            events.push(event);
            batch += 1;
            // Publish a live snapshot every 8 events, and immediately on
            // a terminal event, so the TUI's Live Monitor updates in real
            // time (measurement-isolation invariant: the render loop only
            // reads the snapshot; it never touches this timing path).
            if let Some(state) = &self.metrics {
                let is_terminal = matches!(
                    events.last(),
                    Some(StreamEvent::Complete { .. }) | Some(StreamEvent::Failed { .. })
                );
                if batch >= 8 || is_terminal {
                    state.update(self.live_snapshot(&events));
                    batch = 0;
                }
            }
        }
        // Final publish (the terminal event may have been batched out).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&events));
        }
        // Channel is closed (worker done): collect the outcome. A task
        // panic becomes a failed outcome, never a crash.
        let outcome = join_worker(handle).await;
        (
            aggregate(
                &self.cfg,
                prompt,
                &outcome,
                &events,
                self.generator.tokenizer().as_deref(),
                // CF-26: the SSE parser (CF-02) captures `UsageDetails`
                // (reasoning / cached tokens), but the worker
                // (`client/stream.rs`) does not yet propagate them into the
                // `StreamOutcome` — a P2-A seam. Until that lands, Engine A
                // passes `None`: the decode split falls back to the
                // full-window blended rate and `cache_hit` stays unset.
                None,
            ),
            events,
        )
    }

    /// Build a live [`MetricsSnapshot`] from the events collected so far
    /// (single-stream Engine A → the TUI's Live Monitor).
    fn live_snapshot(&self, events: &[StreamEvent]) -> MetricsSnapshot {
        Self::single_stream_snapshot(
            &self.cfg.url,
            &self.cfg.model,
            self.cfg.mode.label(),
            events,
            MAX_GEN_TOKENS,
            self.generator.tokenizer().as_deref(),
        )
    }

    /// Build a live single-stream [`MetricsSnapshot`] from the events
    /// collected so far — the shared TUI seam for the one-stream engines
    /// (A, C1, C2, C3): the Live Monitor's gauges / ITL distribution /
    /// stream matrix / token counter all read it lock-free while the
    /// current engine runs (measurement-isolation invariant, blueprint §4).
    ///
    /// The **single decode-rate formula** used throughout:
    /// `tokens_received / (T_last_token − T_first_token)`.
    ///
    /// * `tokens_received` — the server's `usage.completion_tokens` once it
    ///   arrives (the **primary** source); the re-tokenized text (exact
    ///   tokenizer, else `chars/4`) while the stream is open. Never the raw
    ///   frame tally — batched vLLM frames undercount by 30–40 %.
    /// * `T_first_token` / `T_last_token` — the decode window (prefill /
    ///   TTFT and the trailing gap to `[DONE]` are excluded).
    ///
    /// This is the same formula the Overall Metrics panel uses via
    /// `gen_tps` and the `DecodeRateTracker` uses for the rolling series —
    /// one code path from token reception to display.
    pub fn single_stream_snapshot(
        endpoint: &str,
        model: &str,
        mode: &str,
        events: &[StreamEvent],
        max_tokens: u32,
        tokenizer: Option<&Tokenizer>,
    ) -> MetricsSnapshot {
        let mut reasoning_frames = 0u64;
        let mut content_frames = 0u64;
        let mut first_token_at: Option<MonotonicInstant> = None;
        let mut last_token_at: Option<MonotonicInstant> = None;
        let mut ttft_ns: Option<u64> = None;
        let mut usage: Option<Usage> = None;
        let mut state = StreamStatus::Waiting;
        let mut t0: Option<MonotonicInstant> = None;
        let mut t_end: Option<MonotonicInstant> = None;
        // v0.1.1 decode-loop guard verdict (carried by the terminal event).
        let mut looping = false;
        let mut loop_excluded_tokens = 0u64;

        for event in events {
            match event {
                StreamEvent::Frame {
                    frame,
                    at,
                    timestamps,
                } => {
                    if frame.chunk.is_token() {
                        match &frame.chunk {
                            Chunk::Reasoning(_) => reasoning_frames += 1,
                            _ => content_frames += 1,
                        }
                        if first_token_at.is_none() {
                            first_token_at = Some(*at);
                        }
                        last_token_at = Some(*at);
                        if state == StreamStatus::Waiting {
                            state = StreamStatus::Streaming;
                        }
                    }
                    if t0.is_none() {
                        t0 = timestamps.t0;
                    }
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
                StreamEvent::Complete {
                    timestamps,
                    usage: u,
                    looping: l,
                    loop_excluded_tokens: e,
                    ..
                } => {
                    usage = *u;
                    t_end = timestamps.t_end;
                    state = StreamStatus::Done;
                    looping = *l;
                    loop_excluded_tokens = *e;
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
                StreamEvent::Failed {
                    timestamps,
                    looping: l,
                    loop_excluded_tokens: e,
                    ..
                } => {
                    t_end = timestamps.t_end;
                    state = StreamStatus::Error;
                    looping = *l;
                    loop_excluded_tokens = *e;
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
            }
        }

        // CF-24/CF-09: the ITL histogram from the worker's per-frame
        // stamps (shared with `aggregate`).
        let itl = itl_histogram(events);

        // The authoritative token count: the server's
        // `usage.completion_tokens` once it arrives (the primary source),
        // else the re-tokenized text (exact tokenizer, else `chars/4`).
        // Never the raw frame tally — batched vLLM frames undercount.
        let (tokens_received, _est) = authoritative_tokens(events, usage, tokenizer);
        let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
        let ttft_s = ttft_ns.map(|ns| ns as f64 / 1_000_000_000.0);

        // The decode rate: `tokens_received / (T_last − T_first)`.
        // One formula, used for the per-stream detail, the labeled layer,
        // and the aggregate.
        let decode_rate = match (first_token_at, last_token_at) {
            (Some(first), Some(last)) => {
                let span_s = first.delta_nanos(&last) as f64 / 1e9;
                if span_s > 0.0 && tokens_received > 0 {
                    tokens_received as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };

        // v0.1.1 labeled metric layers — three separate numbers, never
        // blended (blueprint §v0.1.1-B):
        //   prefill = prompt_tokens / TTFT
        //   decode  = tokens_received / (T_last − T_first)
        //   e2e     = tokens_received / total wall time (T0 → Tn)
        // A looping stream is excluded from all three.
        let (prefill_tps, e2e_tps) = if looping {
            (0.0, 0.0)
        } else {
            let prefill = match (prompt_tokens, ttft_ns) {
                (p, Some(ns)) if p > 0 && ns > 0 => p as f64 / (ns as f64 / 1e9),
                _ => 0.0,
            };
            let e2e = match (t0, t_end) {
                (Some(t0), Some(end)) => {
                    let span_s = t0.delta_nanos(&end) as f64 / 1e9;
                    if span_s > 0.0 && tokens_received > 0 {
                        tokens_received as f64 / span_s
                    } else {
                        0.0
                    }
                }
                _ => 0.0,
            };
            (prefill, e2e)
        };
        // CF-12: the canonical MTP η over ALL token frames (Reasoning +
        // Content), not content-only.
        let total_token_frames = reasoning_frames + content_frames;
        let mtp = (total_token_frames > 0).then(|| mtp_eta(tokens_received, total_token_frames));
        let progress = if state == StreamStatus::Done {
            1.0
        } else {
            (tokens_received as f64 / max_tokens.max(1) as f64).min(1.0)
        };
        let kind = if reasoning_frames > content_frames {
            "Reasoning"
        } else if content_frames > 0 {
            "Content"
        } else {
            "Control"
        };

        MetricsSnapshot {
            endpoint: endpoint.to_string(),
            backend: String::new(),
            model: model.to_string(),
            mode: mode.to_string(),
            aggregate_tps: decode_rate,
            prefill_throughput: prefill_tps,
            decode_throughput: decode_rate,
            e2e_throughput: e2e_tps,
            active_streams: (state == StreamStatus::Streaming && !looping) as usize,
            total_streams: 1,
            itl_p50_ns: itl.p50() as u64,
            itl_p90_ns: itl.p90() as u64,
            itl_p99_ns: itl.p99() as u64,
            itl_p999_ns: itl.p999() as u64,
            prompt_tokens,
            completion_tokens: tokens_received,
            // The decode-rate numerator: the authoritative token count
            // (usage → text estimate), matching `completion_tokens`.
            observed_frames: tokens_received,
            status: state,
            loop_excluded_streams: looping as usize,
            loop_excluded_tokens,
            streams: vec![StreamMetric {
                id: 0,
                kind: kind.to_string(),
                state,
                pp_tokens: (prompt_tokens > 0).then_some(prompt_tokens),
                tg_tokens: (tokens_received > 0).then_some(tokens_received),
                ttft_s,
                gen_tps: (decode_rate > 0.0).then_some(decode_rate),
                mtp,
                progress,
                looping,
            }],
            ..Default::default()
        }
    }

    /// All `iterations` runs (sequential — single-stream engine).
    ///
    /// While a [`ProgressBus`] is attached, each iteration publishes
    /// `Iteration {i}/{total}` plus the running token count — the
    /// Benchmark Sequence mirrors it into the TUI's progress bar.
    pub async fn run(&self) -> (GeneratedPrompt, Vec<SpeedResult>) {
        let prompt = self.generate_prompt();
        let iterations = self.cfg.iterations.max(1) as usize;
        let mut results = Vec::with_capacity(iterations);
        let mut tokens = 0u64;
        for i in 0..iterations {
            // The `Space`-key pause: hold before the next request goes out
            // (an in-flight iteration always completes).
            if let Some(gate) = &self.pause {
                gate.wait_while_paused().await;
            }
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::Speed {
                    iteration: i + 1,
                    total: iterations,
                    tokens,
                });
            }
            let r = self
                .run_iteration_events_tagged(&prompt, &format!("A:{}", i + 1))
                .await
                .0;
            tokens += r.completion_tokens;
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::Speed {
                    iteration: i + 1,
                    total: iterations,
                    tokens,
                });
            }
            if let Some(l) = &self.logger {
                if r.is_failed() {
                    l.error(
                        Context::EngineA,
                        format!(
                            "Iteration {}/{} failed: {}",
                            i + 1,
                            iterations,
                            r.error.as_deref().unwrap_or("unknown")
                        ),
                    );
                } else {
                    l.info(
                        Context::EngineA,
                        format!(
                            "Iteration {}/{} complete: {} tok, {:.1} t/s decode, TTFT {:.0} ms",
                            i + 1,
                            iterations,
                            r.completion_tokens,
                            r.tg_speed,
                            r.ttft * 1000.0
                        ),
                    );
                }
            }
            results.push(r);
        }
        (prompt, results)
    }

    /// The CF-05 **pre-fill throughput** measurement: run the **long** prompt
    /// (padded to `cfg.tokens`) and report `prompt_tokens / server-side TTFT`
    /// (T3−T1) — the *bandwidth* number that scales with context. This is
    /// distinct from the short-prompt first-token *latency* (the
    /// [`SpeedResult::ttft`] server value + [`perceived_ttft_seconds`] client
    /// value).
    ///
    /// A small fixed number of clean prefill runs are averaged; a run that
    /// fails, loops, or has no usable window is excluded. `cfg.nocache` busts
    /// the server KV cache so the prefill is genuinely recomputed (a cached
    /// prefill reads near-zero TTFT and would inflate the number).
    pub async fn run_prefill(&self) -> PrefillMetric {
        let prompt = if self.cfg.nocache {
            self.generator.nocache_long(self.cfg.tokens)
        } else {
            self.generator.long(self.cfg.tokens)
        };
        const PREFILL_RUNS: usize = 3;
        let mut tok_s_sum = 0.0f64;
        let mut ttft_sum = 0.0f64;
        let mut valid = 0usize;
        let mut estimated = false;
        for i in 0..PREFILL_RUNS {
            let (r, _events) = self
                .run_iteration_events_tagged(&prompt, &format!("A:prefill:{}", i + 1))
                .await;
            estimated |= r.estimated;
            // A usable prefill sample: not failed, not looping, real window.
            if !r.is_failed() && !r.looping && r.ttft > 0.0 && r.prompt_tokens > 0 {
                tok_s_sum += r.pp_speed; // = prompt_tokens / server TTFT
                ttft_sum += r.ttft;
                valid += 1;
            }
        }
        PrefillMetric {
            tok_s: if valid > 0 {
                tok_s_sum / valid as f64
            } else {
                0.0
            },
            prompt_tokens: prompt.token_count as u64,
            server_ttft_s: if valid > 0 {
                ttft_sum / valid as f64
            } else {
                0.0
            },
            estimated,
        }
    }
}

/// Synthesize one [`SpeedResult`] from a finished worker run plus its
/// channel events (blueprint §7 formulas; prototype semantics for
/// token-count fallbacks, chunk counting, and the error surface).
///
/// `usage_details` is the CF-02 `UsageDetails` (reasoning / cached tokens)
/// for the run. It drives the CF-26 decode-window split (answer vs. thinking
/// phase) and the CF-26 `cache_hit` derivation. `None` keeps the historical
/// full-window blended decode rate — the current production path, since the
/// SSE parser captures the detail but `client/stream.rs` (P2-A) does not yet
/// propagate it into the [`StreamOutcome`].
#[must_use]
pub fn aggregate(
    cfg: &Config,
    prompt: &GeneratedPrompt,
    outcome: &StreamOutcome,
    events: &[StreamEvent],
    tokenizer: Option<&Tokenizer>,
    usage_details: Option<UsageDetails>,
) -> SpeedResult {
    let ts: StreamTimestamps = outcome.timestamps;
    let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
    // §7.1: TTFT = T3 − T1.
    //
    // CF-25 (honest fallback): when **no token frame** ever arrived (a
    // usage-only stream, a failure, …), TTFT is the `0.0` N/A sentinel —
    // NOT the total stream time (the old fallback mislabeled total latency
    // as TTFT and fed `pp_speed` unflagged). The run is then marked
    // `degraded` ([`SpeedResult::is_degraded`]).
    let ttft = ts.ttft_nanos().map(|n| n as f64 / 1e9).unwrap_or(0.0);

    // Chunk counting: the `[DONE]` terminator is not a chunk; `Usage`
    // frames count as "other" (prototype `__usage__` handling).
    let mut content_chunks = 0u64;
    let mut reasoning_chunks = 0u64;
    let mut other_chunks = 0u64;
    for event in events {
        let StreamEvent::Frame { frame, .. } = event else {
            continue;
        };
        if frame.done {
            continue;
        }
        match &frame.chunk {
            Chunk::Content(_) => content_chunks += 1,
            Chunk::Reasoning(_) => reasoning_chunks += 1,
            Chunk::Usage(_) | Chunk::Control => other_chunks += 1,
        }
    }
    let total_chunks = content_chunks + reasoning_chunks + other_chunks;

    // Token counts — the research's **Authoritative Counting Rule**:
    //
    // 1. **PRIMARY** — the server's own `usage.completion_tokens` (the
    //    terminal usage chunk; every compliant backend populates it under
    //    `stream_options.include_usage`). This is the count the server
    //    reports, so matching it is the goal.
    // 2. **FALLBACK** (a non-compliant proxy that omits `usage`) —
    //    re-tokenize the full reasoning + content text with the model's
    //    exact tokenizer. Counting SSE chunks is fundamentally inaccurate
    //    (multi-token frames, speculative batching, keepalives).
    // 3. **LAST RESORT** (no usage AND no tokenizer) — the observed frame
    //    count, flagged `estimated` (never the primary method).
    let (prompt_tokens, completion_tokens, estimated) = {
        let mut est = false;
        let completion = match outcome
            .usage
            .map(|u| u.completion_tokens)
            .filter(|&n| n > 0)
        {
            Some(n) => n,
            None => match tokenizer
                .and_then(|t| t.try_count(&stream_text(events)))
                .filter(|&n| n > 0)
            {
                Some(n) => n as u64,
                None => {
                    // No usage and no tokenizer: estimate from the full
                    // reasoning + content text (`chars/4`) — counting raw
                    // frames undercounts servers that batch several tokens
                    // per frame (vLLM MTP / speculative decoding).
                    let c = count_tokens(&stream_text(events), tokenizer);
                    est = c.estimated;
                    c.tokens as u64
                }
            },
        };
        let prompt = outcome
            .usage
            .map(|u| u.prompt_tokens)
            .filter(|&n| n > 0)
            .unwrap_or_else(|| {
                est = true;
                prompt.token_count.max(1) as u64
            });
        (prompt, completion, est)
    };

    // The v0.1.1 decode-loop guard: a looping stream's repeating output is
    // excluded from the throughput numbers (the layers are zeroed, the
    // run is flagged, and summaries skip it).
    let looping = outcome.looping;
    let pp_speed = if looping {
        0.0
    } else if ttft > 0.0 {
        prompt_tokens as f64 / ttft
    } else {
        0.0
    };
    // §7.2 — the research's **Timing Boundary Rule**: the decode window is
    // the first token to the last (`T_first` → `T_last`), which excludes
    // prefill/TTFT and the trailing gap to `[DONE]`.
    //
    // CF-26 (decode-window split): when the server reports
    // `reasoning_tokens` (the CF-02 `UsageDetails`), the window is split at
    // the **reasoning→content transition** so `tg_speed` reflects the
    // *answer* phase (content tokens / content window), not the *thinking*
    // phase. Without that detail (the current production path — the parser
    // captures it, `client/stream.rs` does not yet propagate it), we keep the
    // historical full-window blended rate.
    let (t_first, t_last) = token_window(events);
    let (t_first_content, t_last_content) = content_window(events);
    let reasoning_tokens = usage_details.map(|d| d.reasoning_tokens).unwrap_or(0);
    let content_tokens = completion_tokens.saturating_sub(reasoning_tokens);
    let tg_speed = if looping {
        0.0
    } else if usage_details.is_some() {
        // CF-26 answer phase: content tokens over the content window.
        match (t_first_content, t_last_content) {
            (Some(fc), Some(lc)) if content_tokens > 0 => {
                let w = (fc.delta_nanos(&lc) as f64 / 1e9).max(0.001);
                content_tokens as f64 / w
            }
            // Pure CoT (no content frames) or zero content tokens: no answer phase.
            _ => 0.0,
        }
    } else {
        // Fallback (no reasoning-token detail): the full-window blended rate.
        let w = match (t_first, t_last) {
            (Some(f), Some(l)) => (f.delta_nanos(&l) as f64 / 1e9).max(0.001),
            _ => (stream_time - ttft).max(0.001),
        };
        completion_tokens as f64 / w
    };
    // §7.4 — CF-12 canonical: MTP η = completion_tokens / ALL token frames
    // (Reasoning + Content), not content-only.
    let total_token_frames = content_chunks + reasoning_chunks;
    let mtp_efficiency = mtp_eta(completion_tokens, total_token_frames);

    // CF-24 (P2-A surfacing): ITL samples clamped at the histogram's
    // 60 s bound — the percentile figures are capped, not exact.
    let itl_clamped = itl_histogram(events).clamp_count();

    // Error surface (prototype wording).
    let error = match &outcome.error {
        Some(e) => Some(match e {
            StreamError::Connection(_) => "Connection refused — is the server running?".to_string(),
            StreamError::Http { status, body } => format!("HTTP {status}: {body}"),
            StreamError::Timeout(d) => format!("Timed out after {d:?}"),
            StreamError::WorkerTimeout(d) => format!("Worker killed after {d:?} (timeout)"),
            StreamError::Read(msg) => format!("Network error: {msg}"),
            StreamError::InvalidJson(msg) => format!("Invalid JSON response: {msg}"),
        }),
        None if total_chunks == 0 && outcome.usage.is_none() => {
            Some("No data received from server".to_string())
        }
        None if outcome.malformed_frames > 0 => Some(format!(
            "{} malformed chunk(s) skipped",
            outcome.malformed_frames
        )),
        None => None,
    };

    SpeedResult {
        ttft,
        prompt_tokens,
        completion_tokens,
        pp_speed,
        tg_speed,
        mtp_efficiency,
        stream_time,
        total_chunks,
        content_chunks,
        reasoning_chunks,
        other_chunks,
        estimated,
        model: cfg.model.clone(),
        mode: cfg.mode.label().to_string(),
        error,
        looping,
        // P2-A surfacing (CF-10/CF-24/CF-44): the outcome's computed
        // stream-integrity flags, published in the JSON + result lines.
        usage_discrepancy: outcome.has_usage_discrepancy(),
        e2e_only: outcome.is_e2e_only(),
        itl_clamped,
    }
}

/// The `--json` document (parity with the prototype's `output_json`):
/// top-level `url` / `model` / `mode` / `iterations` / `results`, plus a
/// `summary` when more than one run is valid.
///
/// v0.1.1 additions (measurement credibility): a `timing` block (ns
/// resolution + measured overhead), a `methodology` block (the formula
/// behind every number), and a `loop_guard` block when any run was
/// excluded by the decode-loop guard.
#[must_use]
pub fn json_report(cfg: &Config, results: &[SpeedResult]) -> serde_json::Value {
    let mut output = json!({
        "url": cfg.url,
        "model": cfg.model,
        "mode": cfg.mode.label(),
        "iterations": results.len(),
        "results": results.iter().map(|r| r.to_dict()).collect::<Vec<_>>(),
    });
    let valid: Vec<&SpeedResult> = results.iter().filter(|r| !r.is_failed()).collect();
    if valid.len() > 1 {
        let n = valid.len() as f64;
        output["summary"] = json!({
            "avg_ttft": round(valid.iter().map(|r| r.ttft).sum::<f64>() / n, 4),
            "avg_pp_speed": round(valid.iter().map(|r| r.pp_speed).sum::<f64>() / n, 2),
            "avg_tg_speed": round(valid.iter().map(|r| r.tg_speed).sum::<f64>() / n, 2),
            "avg_mtp": round(valid.iter().map(|r| r.mtp_efficiency).sum::<f64>() / n, 4),
        });
    }
    let overhead = crate::timing::measure_timestamp_overhead();
    output["timing"] = crate::metrics::methodology::timing_block(overhead);
    output["methodology"] = crate::metrics::methodology::methodology_block(overhead);
    let detected = results.iter().filter(|r| r.looping).count();
    if detected > 0 {
        let excluded: u64 = results
            .iter()
            .filter(|r| r.looping)
            .map(|r| r.completion_tokens)
            .sum();
        output["loop_guard"] = json!({
            "detected_streams": detected,
            "excluded_tokens": excluded,
        });
    }
    output
}

/// Exit-code rule (prototype `main`): `true` when **every** run failed
/// → the process must exit 1.
#[must_use]
pub fn all_failed(results: &[SpeedResult]) -> bool {
    !results.is_empty() && results.iter().all(SpeedResult::is_failed)
}

// ── CF-05: the prefill/latency split ──────────────────────────────────────

/// The client-perceived first-token latency (T3−T0) in seconds for one run's
/// events — the *full* client round-trip (request-send network leg + server
/// queue/prefill/first-decode). `0.0` when no token frame / milestone is
/// available.
///
/// This is the **perceived** half of the CF-05 first-token *latency* metric;
/// the **server-side** half (T3−T1) is [`SpeedResult::ttft`].
#[must_use]
pub fn perceived_ttft_seconds(events: &[StreamEvent]) -> f64 {
    for event in events.iter().rev() {
        if let StreamEvent::Complete { timestamps, .. } = event {
            if let Some(ns) = timestamps.ttft_perceived_nanos() {
                return ns as f64 / 1e9;
            }
        }
    }
    0.0
}

/// The CF-05 terminal report: Engine A's two **distinct, labeled** metrics —
/// first-token *latency* (short prompt, a latency number) and pre-fill
/// *throughput* (long prompt, a bandwidth number) — plus the intact decode /
/// MTP numbers and a pointer to C1's 2k→128k prefill-degradation curve for
/// the full scaling picture.
///
/// `results` are the short-prompt runs (latency + decode); `perceived` is the
/// per-run client-perceived TTFT (T3−T0, seconds); `prefill` is the long-
/// prompt throughput measurement.
#[must_use]
pub fn format_split_report(
    results: &[SpeedResult],
    perceived: &[f64],
    prefill: &PrefillMetric,
    color: bool,
) -> String {
    // The valid (non-failed, non-looping) short runs drive the latency +
    // decode numbers.
    let valid: Vec<&SpeedResult> = results
        .iter()
        .filter(|r| !r.is_failed() && !r.looping)
        .collect();
    let est = if valid.iter().any(|r| r.estimated) || prefill.estimated {
        " [ESTIMATED]"
    } else {
        ""
    };

    let server_ms = mean_of(&valid.iter().map(|r| r.ttft * 1000.0).collect::<Vec<_>>());
    let perceived_ms = mean_of(
        &perceived
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .map(|v| v * 1000.0)
            .collect::<Vec<_>>(),
    );
    let lat_tokens = valid.first().map(|r| r.prompt_tokens).unwrap_or(0);
    let decode_tps = mean_of(&valid.iter().map(|r| r.tg_speed).collect::<Vec<_>>());
    let mtp = mean_of(&valid.iter().map(|r| r.mtp_efficiency).collect::<Vec<_>>());

    // Build the interior lines; a `None` entry is a section separator.
    let mut items: Vec<Option<String>> = Vec::new();
    items.push(Some(
        "Engine A — Speed & Latency  (CF-05 prefill/latency split)".to_string(),
    ));
    items.push(None);
    items.push(Some(
        "■ FIRST-TOKEN LATENCY — short prompt (a latency)".to_string(),
    ));
    items.push(Some(format!(
        "  perceived (T3−T0, incl. network): {perceived_ms:>9.0} ms{est}"
    )));
    items.push(Some(format!(
        "  server-side (T3−T1):            {server_ms:>9.0} ms{est}"
    )));
    items.push(Some(format!(
        "  prompt size:                    {lat_tokens:>9} tok"
    )));
    items.push(None);
    items.push(Some(
        "■ PRE-FILL THROUGHPUT — long prompt (bandwidth)".to_string(),
    ));
    if prefill.is_valid() {
        items.push(Some(format!(
            "  {} tok / server TTFT {:.0} ms: {tok:>8.1} tok/s{est}",
            prefill.prompt_tokens,
            prefill.server_ttft_s * 1000.0,
            tok = prefill.tok_s
        )));
    } else {
        items.push(Some(
            "  (no valid long-prompt prefill measurement)".to_string(),
        ));
    }
    items.push(None);
    items.push(Some("■ DECODE — intact (short prompt)".to_string()));
    items.push(Some(format!("  TG speed:  {decode_tps:.1} tok/s{est}")));
    items.push(Some(format!("  MTP eff:   {mtp:.2} tok/chunk")));
    items.push(None);
    items.push(Some(
        "Scaling: the full prefill-degradation curve (the".to_string(),
    ));
    items.push(Some(
        "compute→memory-bandwidth crossover) is measured by".to_string(),
    ));
    items.push(Some(
        "Engine C1's 2k→128k NIAH matrix (per-cell prefill_tps).".to_string(),
    ));

    // Render the box at the widest interior line (min 50, prototype width).
    let w = items
        .iter()
        .filter_map(|i| i.as_ref())
        .map(|s| s.chars().count())
        .max()
        .unwrap_or(50)
        .max(50);
    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("╔{}╗", "═".repeat(w)));
    for item in &items {
        match item {
            Some(text) => lines.push(format!("║{:<width$}║", text, width = w)),
            None => lines.push(format!("╠{}╣", "─".repeat(w))),
        }
    }
    lines.push(format!("╚{}╝", "═".repeat(w)));

    let mut out = String::new();
    for line in &lines {
        if color {
            out.push_str(CYAN);
            out.push_str(line);
            out.push_str(RESET);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// The CF-05 `--json` document: Engine A's two **distinct, labeled** metrics
/// (first-token *latency* from the short prompt; pre-fill *throughput* from
/// the long prompt) plus the intact decode numbers, the per-iteration
/// short-run records, and the standard `timing` / `methodology` blocks.
#[must_use]
pub fn json_report_split(
    cfg: &Config,
    results: &[SpeedResult],
    perceived: &[f64],
    prefill: &PrefillMetric,
) -> serde_json::Value {
    let valid: Vec<&SpeedResult> = results
        .iter()
        .filter(|r| !r.is_failed() && !r.looping)
        .collect();
    let server_ms = round(
        mean_of(&valid.iter().map(|r| r.ttft * 1000.0).collect::<Vec<_>>()),
        1,
    );
    let perceived_ms = round(
        mean_of(
            &perceived
                .iter()
                .copied()
                .filter(|v| *v > 0.0)
                .map(|v| v * 1000.0)
                .collect::<Vec<_>>(),
        ),
        1,
    );
    let decode_tps = round(
        mean_of(&valid.iter().map(|r| r.tg_speed).collect::<Vec<_>>()),
        2,
    );
    let mtp = round(
        mean_of(&valid.iter().map(|r| r.mtp_efficiency).collect::<Vec<_>>()),
        4,
    );

    let mut output = json!({
        "url": cfg.url,
        "model": cfg.model,
        "engine": "A (Speed & Latency)",
        "first_token_latency": {
            "perceived_ms": perceived_ms,
            "server_ms": server_ms,
            "prompt_tokens": valid.first().map(|r| r.prompt_tokens).unwrap_or(0),
            "note": "Short-prompt first-token LATENCY. perceived = T3−T0 (client round-trip, network-included); server = T3−T1 (server queue + prefill + first-decode). A latency number, not bandwidth."
        },
        "prefill_throughput": {
            "tok_s": round(prefill.tok_s, 1),
            "prompt_tokens": prefill.prompt_tokens,
            "server_ttft_ms": round(prefill.server_ttft_s * 1000.0, 1),
            "estimated": prefill.estimated,
            "note": "Long-prompt pre-fill BANDWIDTH = prompt_tokens / server-side TTFT (T3−T1). The numerator is a real context size, so this is the number that scales with context."
        },
        "decode": {
            "tg_speed_tok_s": decode_tps,
            "mtp_efficiency": mtp,
        },
        "scaling_note": "prefill_throughput above is one point at the long-prompt size. The full prefill-degradation curve (the compute→memory-bandwidth crossover as context grows) is measured by Engine C1's 2k→128k NIAH matrix (per-cell prefill_tps = prompt_tokens / TTFT).",
        "iterations": results.len(),
        "results": results.iter().map(|r| r.to_dict()).collect::<Vec<_>>(),
    });
    // The prototype's per-run `summary` block (avg over the valid short runs)
    // is preserved for backward-compatible JSON consumers / exports.
    if valid.len() > 1 {
        let n = valid.len() as f64;
        output["summary"] = json!({
            "avg_ttft": round(valid.iter().map(|r| r.ttft).sum::<f64>() / n, 4),
            "avg_pp_speed": round(valid.iter().map(|r| r.pp_speed).sum::<f64>() / n, 2),
            "avg_tg_speed": round(valid.iter().map(|r| r.tg_speed).sum::<f64>() / n, 2),
            "avg_mtp": round(valid.iter().map(|r| r.mtp_efficiency).sum::<f64>() / n, 4),
        });
    }
    let overhead = crate::timing::measure_timestamp_overhead();
    output["timing"] = crate::metrics::methodology::timing_block(overhead);
    output["methodology"] = crate::metrics::methodology::methodology_block(overhead);
    output
}

// ── Terminal formatting (prototype parity) ───────────────────────────────

const CYAN: &str = "\u{1b}[36m";
const RESET: &str = "\u{1b}[0m";
const BOLD: &str = "\u{1b}[1m";

/// The result box (parity with the prototype's `print_result_box`),
/// including its width quirk (52-char borders/title, 54-char label rows).
/// `None` for a hard failure — the caller prints the `FAILED` line.
#[must_use]
pub fn format_result_box(
    result: &SpeedResult,
    iteration: usize,
    total: usize,
    color: bool,
) -> Option<String> {
    if result.is_failed() {
        return None;
    }
    let est_tag = if result.estimated { " [ESTIMATED]" } else { "" };

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!("╔{}╗", "═".repeat(50)));
    let title = if total > 1 {
        format!(" LLM SpeedTest  [{iteration}/{total}]")
    } else {
        " LLM SpeedTest".to_string()
    };
    lines.push(format!("║{title:<50}║"));
    lines.push(format!("╠{}╣", "═".repeat(50)));

    row(&mut lines, "Model:", &result.model);
    row(
        &mut lines,
        "Prompt tokens:",
        &format!("{}{est_tag}", result.prompt_tokens),
    );
    row(
        &mut lines,
        "Gen tokens:",
        &result.completion_tokens.to_string(),
    );
    lines.push(format!("╠{}╣", "─".repeat(50)));
    // CF-25: a degraded run (no token frame) shows TTFT as N/A, not the
    // 0.0000s sentinel.
    let ttft_display = if result.is_degraded() {
        "N/A".to_string()
    } else {
        format!("{:.4}s", result.ttft)
    };
    row(&mut lines, "TTFT:", &ttft_display);
    row(
        &mut lines,
        "PP speed:",
        &format!("{:.1} tok/s{est_tag}", result.pp_speed),
    );
    row(
        &mut lines,
        "TG speed:",
        &format!("{:.1} tok/s{est_tag}", result.tg_speed),
    );
    row(
        &mut lines,
        "MTP eff:",
        &format!("{:.2} tok/chunk", result.mtp_efficiency),
    );
    lines.push(format!("╠{}╣", "─".repeat(50)));
    row(
        &mut lines,
        "Stream time:",
        &format!("{:.2}s", result.stream_time),
    );
    row(
        &mut lines,
        "Chunks:",
        &format!(
            "{} ({} content)",
            result.total_chunks, result.content_chunks
        ),
    );
    if result.reasoning_chunks > 0 {
        row(
            &mut lines,
            "  reasoning:",
            &result.reasoning_chunks.to_string(),
        );
    }
    // P2-A surfacing: the stream-integrity flags (CF-44 / CF-10 / CF-24).
    if result.e2e_only {
        row(&mut lines, "  e2e-only:", "non-streaming (total latency)");
    }
    if result.usage_discrepancy {
        row(&mut lines, "  usage check:", "⚠ > 4× frames (corrected)");
    }
    if result.itl_clamped > 0 {
        row(
            &mut lines,
            "  ITL clamped:",
            &format!("{} sample(s) at the 60 s cap", result.itl_clamped),
        );
    }
    if result.looping {
        row(&mut lines, "  loop guard:", "EXCLUDED (repeating pattern)");
    }
    lines.push(format!("╚{}╝", "═".repeat(50)));

    let mut out = String::new();
    for line in &lines {
        if color {
            out.push_str(CYAN);
            out.push_str(line);
            out.push_str(RESET);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    Some(out)
}

/// One `║ label  value  ║` row of the result box (the prototype's `row`
/// helper: 18-char label, 31-char value, 54-char line).
fn row(lines: &mut Vec<String>, label: &str, value: &str) {
    lines.push(format!("║ {label:<18} {value:<31} ║"));
}

/// The multi-iteration summary (parity with the prototype's
/// `print_summary`): `None` when fewer than two runs are valid.
#[must_use]
pub fn format_summary(results: &[SpeedResult], color: bool) -> Option<String> {
    // Looping runs (v0.1.1) are excluded from the averaged summary — their
    // throughput is not a measurement of the server.
    let valid: Vec<&SpeedResult> = results
        .iter()
        .filter(|r| !r.is_failed() && !r.looping)
        .collect();
    if valid.len() < 2 {
        return None;
    }
    let header = format!("── Summary (averaged) {}", "─".repeat(30));
    let mut out = String::new();
    out.push('\n');
    if color {
        out.push_str(BOLD);
        out.push_str(&header);
        out.push_str(RESET);
    } else {
        out.push_str(&header);
    }
    out.push('\n');
    out.push_str(&format!(
        "  TTFT:       {}\n",
        stat(&valid, |r| r.ttft, 4, "s")
    ));
    out.push_str(&format!(
        "  PP speed:   {}\n",
        stat(&valid, |r| r.pp_speed, 1, " tok/s")
    ));
    out.push_str(&format!(
        "  TG speed:   {}\n",
        stat(&valid, |r| r.tg_speed, 1, " tok/s")
    ));
    out.push_str(&format!(
        "  MTP eff:    {}\n",
        stat(&valid, |r| r.mtp_efficiency, 2, " tok/chunk")
    ));
    out.push_str(&format!(
        "  Stream:     {}\n",
        stat(&valid, |r| r.stream_time, 2, "s")
    ));
    Some(out)
}

/// `avg=… min=… max=…` over the positive values (the prototype's `stat`
/// helper), `N/A` when none.
fn stat(
    valid: &[&SpeedResult],
    get: impl Fn(&SpeedResult) -> f64,
    decimals: u32,
    unit: &str,
) -> String {
    let vals: Vec<f64> = valid.iter().map(|r| get(r)).filter(|v| *v > 0.0).collect();
    if vals.is_empty() {
        return "N/A".to_string();
    }
    let avg = vals.iter().sum::<f64>() / vals.len() as f64;
    let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
    let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    format!(
        "avg={avg:.prec$}{unit} min={min:.prec$}{unit} max={max:.prec$}{unit}",
        prec = decimals as usize
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::{ParsedFrame, Usage, UsageDetails};
    use crate::timing::MonotonicInstant;

    fn cfg() -> Config {
        Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "test-model".to_string(),
            ..Config::default()
        }
    }

    fn prompt() -> GeneratedPrompt {
        // ~100 chars → 25 chars/4 tokens via the fallback.
        GeneratedPrompt {
            text: "x".repeat(100),
            token_count: 25,
            estimated: true,
            nocache: false,
        }
    }

    /// A synthetic completed run on an exact mock clock: T0 (0 ms) → T1
    /// (5 ms) → first token (10 ms) → last token (30 ms) → Tn (40 ms), with
    /// 1 reasoning frame + 2 content frames (the 3 token frames spread over
    /// 10→30 ms), a usage frame, and the `[DONE]` terminator.
    fn completed_run() -> (StreamOutcome, Vec<StreamEvent>) {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(5_000_000);
            let t1 = MonotonicInstant::now(); // 5 ms
            mock.increment(5_000_000);
            let t3 = MonotonicInstant::now(); // 10 ms (first token)
            mock.increment(10_000_000);
            let t_mid = MonotonicInstant::now(); // 20 ms
            mock.increment(10_000_000);
            let t_last = MonotonicInstant::now(); // 30 ms (last token)
            mock.increment(10_000_000);
            let t_end = MonotonicInstant::now(); // 40 ms

            let usage = Usage {
                prompt_tokens: 128,
                completion_tokens: 34,
            };
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let frames: Vec<(Chunk, MonotonicInstant, bool)> = vec![
                (Chunk::Reasoning("think".into()), t3, false),
                (Chunk::Content("Hello, ".into()), t_mid, false),
                (Chunk::Content("world!".into()), t_last, false),
                (Chunk::Usage(usage), t_end, false),
                (Chunk::Control, t_end, true),
            ];
            let events: Vec<StreamEvent> = frames
                .iter()
                .map(|(chunk, at, done)| StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: chunk.clone(),
                        t_nanos: 0,
                        done: *done,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 3,
            };
            (outcome, events)
        })
    }

    #[test]
    fn aggregate_computes_section7_formulas() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);

        // Chunk counting: `[DONE]` excluded; usage → "other".
        assert_eq!(r.content_chunks, 2);
        assert_eq!(r.reasoning_chunks, 1);
        assert_eq!(r.other_chunks, 1);
        assert_eq!(r.total_chunks, 4);

        // Server usage is the authoritative count.
        assert_eq!(r.prompt_tokens, 128);
        assert_eq!(r.completion_tokens, 34);
        assert!(!r.estimated);
        assert!(r.error.is_none());
        assert!(!r.is_failed());

        // TTFT = 5 ms (T3 − T1); stream time = 40 ms (Tn − T0).
        let ttft_ms = r.ttft * 1000.0;
        assert!((4.9..5.1).contains(&ttft_ms), "ttft {ttft_ms}ms");
        let stream_ms = r.stream_time * 1000.0;
        assert!((39.9..40.1).contains(&stream_ms), "stream {stream_ms}ms");

        // PP = prompt / TTFT; TG = completion / decode window (first→last
        // token = 10→30 ms = 20 ms; `usage_details=None` → the full-window
        // blended rate); MTP η (CF-12) = 34 / 3 total token frames (1
        // reasoning + 2 content) = 11.333 — not the old content-only
        // 34 / 2 = 17.
        assert!((r.pp_speed - 128.0 / r.ttft).abs() < 1e-9);
        let gen = 0.02;
        assert!((r.tg_speed - 34.0 / gen).abs() < 1e-9, "tg {}", r.tg_speed);
        assert!(
            (r.mtp_efficiency - 34.0 / 3.0).abs() < 1e-9,
            "mtp {}",
            r.mtp_efficiency
        );

        assert_eq!(r.model, "test-model");
        assert_eq!(r.mode, "short");
    }

    #[test]
    fn missing_usage_falls_back_to_text_estimate_and_flags_estimated() {
        let (mut outcome, events) = completed_run();
        outcome.usage = None;
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        // Prompt fallback: the generated prompt's own (estimated) count.
        assert_eq!(r.prompt_tokens, 25);
        // Completion fallback (no usage, no tokenizer): the re-tokenized
        // full text (`chars/4`) — 18 chars in `completed_run` ("think" +
        // "Hello, " + "world!") → 4 tokens. Counting raw frames (3)
        // undercounts batched servers, so it is no longer used. Flagged
        // `estimated`.
        assert_eq!(r.completion_tokens, 4);
        assert!(r.estimated);
        assert!(r.error.is_none());
    }

    #[test]
    fn no_data_at_all_is_a_failure() {
        let t0 = MonotonicInstant::now();
        let t_end = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t_end: Some(t_end),
            ..StreamTimestamps::default()
        };
        let outcome = StreamOutcome {
            timestamps: ts,
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: None,

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None, None);
        assert_eq!(r.error.as_deref(), Some("No data received from server"));
        assert_eq!(r.completion_tokens, 0);
        assert!(r.is_failed());
        assert!(all_failed(std::slice::from_ref(&r)));
    }

    #[test]
    fn http_error_maps_to_prototype_wording_and_fails() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            ..StreamTimestamps::default()
        };
        let outcome = StreamOutcome {
            timestamps: ts,
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Http {
                status: 500,
                body: "boom".into(),
            }),

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None, None);
        assert_eq!(r.error.as_deref(), Some("HTTP 500: boom"));
        assert!(r.is_failed());
    }

    #[test]
    fn connection_refused_uses_the_prototype_message() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            ..StreamTimestamps::default()
        };
        let outcome = StreamOutcome {
            timestamps: ts,
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Connection("refused".into())),

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None, None);
        assert_eq!(
            r.error.as_deref(),
            Some("Connection refused — is the server running?")
        );
        assert!(r.is_failed());
    }

    #[test]
    fn malformed_frames_are_a_note_not_a_failure() {
        let (mut outcome, events) = completed_run();
        outcome.malformed_frames = 2;
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        assert_eq!(r.error.as_deref(), Some("2 malformed chunk(s) skipped"));
        // Tokens were produced → not a failed run (prototype semantics).
        assert!(!r.is_failed());
        assert!(!all_failed(&[r]));
    }

    #[test]
    fn json_report_matches_prototype_shape() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);

        // One valid run: no summary (the prototype emits one only for
        // >1 valid runs). v0.1.1: the `timing` + `methodology` blocks are
        // always present; `loop_guard` only when a run was excluded.
        let v = json_report(&cfg(), std::slice::from_ref(&r));
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "url",
                "model",
                "mode",
                "iterations",
                "results",
                "timing",
                "methodology"
            ]
        );
        assert!(v.get("summary").is_none());
        assert!(v.get("loop_guard").is_none(), "no looping runs → no block");

        // Result keys: the prototype `to_dict` set + v0.1.1 `looping`
        // + CF-25 `degraded` + the P2-A integrity flags (CF-10/24/44)
        // (no `other_chunks`).
        let rj = v["results"][0].as_object().unwrap();
        assert_eq!(rj.len(), 19);
        for key in [
            "ttft_s",
            "prompt_tokens",
            "completion_tokens",
            "pp_speed_tok_s",
            "tg_speed_tok_s",
            "mtp_efficiency",
            "stream_time_s",
            "total_chunks",
            "content_chunks",
            "reasoning_chunks",
            "estimated",
            "model",
            "mode",
            "error",
            "looping",
            "degraded",
            "usage_discrepancy",
            "e2e_only",
            "itl_clamped",
        ] {
            assert!(rj.contains_key(key), "missing {key}");
        }
        assert!(!rj.contains_key("other_chunks"));
        assert_eq!(rj["looping"], false);
        // P2-A surfacing: `completed_run` reports 34 usage tokens over 3
        // observed frames (34 > 3×4) → the CF-10 guard trips; it is a
        // streaming result (not E2E-only) with no clamped ITL samples.
        assert_eq!(rj["usage_discrepancy"], true);
        assert_eq!(rj["e2e_only"], false);
        assert_eq!(rj["itl_clamped"], 0);

        // The v0.1.1 blocks carry the published methodology.
        assert_eq!(v["timing"]["resolution"], "nanosecond");
        assert!(v["timing"]["overhead_ns"].is_u64());
        assert_eq!(v["methodology"]["version"], env!("CARGO_PKG_VERSION"));
        assert!(v["methodology"]["prefill_throughput"].is_string());

        // Two valid runs: summary with the prototype's four keys.
        let v2 = json_report(&cfg(), &[r.clone(), r]);
        let s = v2["summary"].as_object().unwrap();
        let skeys: Vec<&str> = s.keys().map(|k| k.as_str()).collect();
        assert_eq!(
            skeys,
            vec!["avg_ttft", "avg_pp_speed", "avg_tg_speed", "avg_mtp"]
        );
        assert_eq!(v2["iterations"], 2);
    }

    // ── v0.1.1: labeled layers, loop guard, timing publication ─────────

    #[test]
    fn aggregate_zeroes_throughput_for_a_looping_run() {
        let (mut outcome, events) = completed_run();
        outcome.looping = true;
        outcome.loop_excluded_tokens = outcome.usage.map(|u| u.completion_tokens).unwrap_or(0);
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        assert!(r.looping, "the flag survives into the result");
        assert_eq!(r.pp_speed, 0.0, "prefill excluded");
        assert_eq!(r.tg_speed, 0.0, "decode excluded");
        // The observed token count is preserved (it is the exclusion
        // numerator), only the rates are zeroed.
        assert_eq!(r.completion_tokens, 34);
    }

    #[test]
    fn single_stream_snapshot_computes_the_three_labeled_layers() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(10_000_000);
            let t1 = MonotonicInstant::now(); // 10 ms
            mock.increment(20_000_000);
            let t3 = MonotonicInstant::now(); // 30 ms (first token)
                                              // Ten content frames spread 30, 40, …, 120 ms (last token at
                                              // 120 ms), then the stream closes at 130 ms.
            let frame_times: Vec<MonotonicInstant> = (0..10)
                .map(|_| {
                    let at = MonotonicInstant::now();
                    mock.increment(10_000_000);
                    at
                })
                .collect();
            let t_end = MonotonicInstant::now(); // 130 ms
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 60,
                completion_tokens: 10,
            };
            let events: Vec<StreamEvent> = frame_times
                .iter()
                .map(|at| StreamEvent::Frame {
                    frame: crate::sse::ParsedFrame {
                        chunk: Chunk::Content("a".into()),
                        t_nanos: 0,
                        done: false,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let snap =
                SpeedEngine::single_stream_snapshot("http://x", "m", "short", &events, 256, None);
            // prefill = 60 prompt tokens / 20 ms TTFT = 3000 t/s.
            assert!(
                (snap.prefill_throughput - 3000.0).abs() < 1e-6,
                "{}",
                snap.prefill_throughput
            );
            // decode = 10 tokens / (first→last token = 120−30 ms = 90 ms).
            assert!(
                (snap.decode_throughput - 10.0 / 0.09).abs() < 1e-6,
                "{}",
                snap.decode_throughput
            );
            // e2e = 10 tokens / 130 ms ≈ 76.9 t/s.
            assert!(
                (snap.e2e_throughput - 10.0 / 0.13).abs() < 1e-6,
                "{}",
                snap.e2e_throughput
            );
            // The three layers are distinct (never blended).
            assert_ne!(snap.prefill_throughput, snap.decode_throughput);
            assert_ne!(snap.decode_throughput, snap.e2e_throughput);
            assert_eq!(snap.loop_excluded_streams, 0);
        });
    }

    #[test]
    fn single_stream_snapshot_excludes_a_looping_stream() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(10_000_000);
            let t3 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t_end = MonotonicInstant::now();
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t0),
                t2: Some(t3),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 60,
                completion_tokens: 10,
            };
            let events = vec![
                StreamEvent::Frame {
                    frame: crate::sse::ParsedFrame {
                        chunk: Chunk::Content("a".into()),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t3,
                    timestamps: ts,
                },
                StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: true,
                    loop_excluded_tokens: 10,
                },
            ];
            let snap =
                SpeedEngine::single_stream_snapshot("http://x", "m", "short", &events, 256, None);
            assert_eq!(snap.prefill_throughput, 0.0);
            assert_eq!(snap.decode_throughput, 0.0);
            assert_eq!(snap.e2e_throughput, 0.0);
            assert_eq!(snap.loop_excluded_streams, 1);
            assert_eq!(snap.loop_excluded_tokens, 10);
            assert!(snap.streams[0].looping);
        });
    }

    #[test]
    fn measured_timestamp_overhead_is_positive_and_small() {
        let ns = crate::timing::measure_timestamp_overhead();
        // A real clock measures a positive overhead; quanta's TSC path is
        // well under a microsecond per call (the "< 100 ns" marketing
        // claim is for the TSC case; allow generous headroom here).
        assert!(ns > 0, "overhead must be positive");
        assert!(ns < 10_000, "a single timestamp call is not 10 µs: {ns}");
    }

    #[test]
    fn all_failed_only_when_every_run_failed() {
        let (outcome, events) = completed_run();
        let ok = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let t0 = MonotonicInstant::now();
        let bad = StreamOutcome {
            timestamps: StreamTimestamps {
                t0: Some(t0),
                ..StreamTimestamps::default()
            },
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Connection("refused".into())),

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let failed = aggregate(&cfg(), &prompt(), &bad, &[], None, None);
        assert!(!all_failed(&[ok.clone(), failed.clone()]));
        assert!(all_failed(&[failed]));
        assert!(!all_failed(&[]));
    }

    #[test]
    fn result_box_mirrors_prototype_layout() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let box_ = format_result_box(&r, 1, 3, false).unwrap();
        let lines: Vec<&str> = box_.lines().collect();

        // Prototype quirk: 52-char borders *and title line*, 54-char label
        // rows (char counts; box-drawing glyphs are multi-byte in UTF-8).
        assert_eq!(lines[0], &format!("╔{}╗", "═".repeat(50)));
        assert_eq!(lines[0].chars().count(), 52);
        assert!(lines[1].starts_with("║ LLM SpeedTest  [1/3]"));
        assert_eq!(lines[1].chars().count(), 52);
        assert!(lines[1].ends_with('║'));
        assert_eq!(lines[3].chars().count(), 54);
        assert!(lines[3].starts_with("║ Model:"));
        assert!(box_.contains("Model:"));
        // Label column is padded to 18 chars before the value column.
        assert!(box_.contains("Prompt tokens:"));
        assert!(box_.contains("Gen tokens:"));
        assert!(box_.contains("128"));
        assert!(box_.contains("34"));
        assert!(box_.contains("MTP eff:"));
        assert!(box_.contains("tok/chunk"));
        assert!(box_.contains("Chunks:"));
        assert!(box_.contains("4 (2 content)"));
        // The indented reasoning row appears only when reasoning chunks > 0.
        assert!(box_.contains("reasoning:"));
        assert_eq!(lines.last().unwrap(), &format!("╚{}╝", "═".repeat(50)));

        // A hard failure yields no box.
        let t0 = MonotonicInstant::now();
        let failed = StreamOutcome {
            timestamps: StreamTimestamps {
                t0: Some(t0),
                ..StreamTimestamps::default()
            },
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Connection("refused".into())),

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let fr = aggregate(&cfg(), &prompt(), &failed, &[], None, None);
        assert!(format_result_box(&fr, 1, 1, false).is_none());
    }

    #[test]
    fn summary_renders_for_two_valid_runs() {
        let (outcome, events) = completed_run();
        let a = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let b = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let s = format_summary(&[a.clone(), b], false).unwrap();
        assert!(s.contains("── Summary (averaged) "));
        assert!(s.contains("TTFT:"));
        assert!(s.contains("avg="));
        assert!(s.contains("min="));
        assert!(s.contains("max="));
        // Fewer than two valid runs: no summary.
        let t0 = MonotonicInstant::now();
        let bad = StreamOutcome {
            timestamps: StreamTimestamps {
                t0: Some(t0),
                ..StreamTimestamps::default()
            },
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Connection("x".into())),

            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
        };
        let failed = aggregate(&cfg(), &prompt(), &bad, &[], None, None);
        assert!(format_summary(&[a.clone(), failed], false).is_none());
    }

    // ── CF-05: the prefill/latency split ─────────────────────────────────

    /// A long-prompt prefill metric (the bandwidth half): 10 000 tok over a
    /// 2.0 s server-side TTFT → 5 000 tok/s.
    fn long_prefill() -> PrefillMetric {
        PrefillMetric {
            tok_s: 5000.0,
            prompt_tokens: 10_000,
            server_ttft_s: 2.0,
            estimated: false,
        }
    }

    #[test]
    fn perceived_ttft_is_t3_minus_t0_distinct_from_server() {
        let (outcome, events) = completed_run();
        // completed_run: T0=0 ms, T1=5 ms, T3=10 ms.
        let perceived = perceived_ttft_seconds(&events);
        assert!((perceived - 0.010).abs() < 1e-6, "perceived {perceived}");
        // The server-side value (SpeedResult.ttft = T3−T1 = 5 ms) is a
        // distinct, smaller number.
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        assert!((r.ttft - 0.005).abs() < 1e-6, "server {}", r.ttft);
        assert!((perceived - r.ttft).abs() > 0.004, "perceived != server");
    }

    #[test]
    fn split_report_labels_latency_and_prefill_as_distinct_metrics() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let perceived = vec![perceived_ttft_seconds(&events)];
        let prefill = long_prefill();
        let report = format_split_report(std::slice::from_ref(&r), &perceived, &prefill, false);

        // Two distinct, clearly-labeled sections.
        assert!(report.contains("FIRST-TOKEN LATENCY"), "{report}");
        assert!(report.contains("PRE-FILL THROUGHPUT"), "{report}");
        // Latency from the SHORT prompt: perceived (T3−T0) + server (T3−T1).
        assert!(report.contains("perceived (T3−T0"), "{report}");
        assert!(report.contains("server-side (T3−T1)"), "{report}");
        // Prefill from the LONG prompt: 10 000 tok → 5 000 tok/s.
        assert!(report.contains("10000 tok"), "{report}");
        assert!(report.contains("5000.0 tok/s"), "{report}");
        // The scaling pointer to C1's 2k→128k curve.
        assert!(report.contains("2k→128k"), "{report}");
    }

    #[test]
    fn json_report_split_computes_metrics_from_the_right_inputs() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        let perceived = vec![perceived_ttft_seconds(&events)];
        let prefill = long_prefill();
        let v = json_report_split(&cfg(), std::slice::from_ref(&r), &perceived, &prefill);

        // Latency from the SHORT prompt: perceived (T3−T0 = 10 ms) + server
        // (T3−T1 = 5 ms).
        assert!((v["first_token_latency"]["perceived_ms"].as_f64().unwrap() - 10.0).abs() < 0.1);
        assert!((v["first_token_latency"]["server_ms"].as_f64().unwrap() - 5.0).abs() < 0.1);
        // Throughput from the LONG prompt: 10 000 tok / 2.0 s = 5 000 tok/s.
        assert!((v["prefill_throughput"]["tok_s"].as_f64().unwrap() - 5000.0).abs() < 0.1);
        assert_eq!(
            v["prefill_throughput"]["prompt_tokens"].as_u64().unwrap(),
            10_000
        );
        // Decode is intact and present.
        assert!(v["decode"]["tg_speed_tok_s"].is_number());
        assert!(v["decode"]["mtp_efficiency"].is_number());
        // The scaling pointer references C1.
        assert!(v["scaling_note"].as_str().unwrap().contains("C1"));
    }

    #[test]
    fn short_prompt_run_no_longer_emits_a_single_prefill_bandwidth() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        // The short run's own pp_speed (128 tok / 5 ms server TTFT = 25 600)
        // is the old misleading "bandwidth" — it must NOT be presented as the
        // prefill throughput. The prefill metric comes from the LONG prompt.
        let short_pp = r.pp_speed;
        assert!(
            (short_pp - 25_600.0).abs() < 1.0,
            "short pp_speed {short_pp}"
        );
        let prefill = long_prefill();
        let perceived = vec![perceived_ttft_seconds(&events)];

        let report = format_split_report(std::slice::from_ref(&r), &perceived, &prefill, false);
        // The report's pre-fill throughput is the LONG number (5 000), not the
        // short run's pp_speed (25 600).
        assert!(report.contains("5000.0 tok/s"), "{report}");
        assert!(
            !report.contains(&format!("{short_pp:.1} tok/s")),
            "short pp_speed {short_pp} must not be the prefill bandwidth: {report}"
        );

        // In JSON, prefill_throughput.tok_s is the long value, distinct from
        // the per-run pp_speed_tok_s.
        let v = json_report_split(&cfg(), std::slice::from_ref(&r), &perceived, &prefill);
        let prefill_tps = v["prefill_throughput"]["tok_s"].as_f64().unwrap();
        let run_pp = v["results"][0]["pp_speed_tok_s"].as_f64().unwrap();
        assert!((prefill_tps - 5000.0).abs() < 0.1);
        assert!((run_pp - short_pp).abs() < 1.0);
        assert_ne!(prefill_tps, run_pp, "the two metrics are distinct");
    }

    // ── CF-12 / CF-25 / CF-26: Phase 2 metric integrity ─────────────────

    /// CF-12: the canonical MTP η is completion tokens over ALL token frames.
    #[test]
    fn mtp_eta_canonical_is_completion_over_all_frames() {
        // 34 completion tokens over 3 token frames (1 reasoning + 2 content)
        // = 11.333 — the old content-only definition gave 34 / 2 = 17.
        assert!((mtp_eta(34, 3) - 34.0 / 3.0).abs() < 1e-9);
        // No token frames → 0.0 (N/A), never a division by zero.
        assert_eq!(mtp_eta(10, 0), 0.0);
    }

    /// CF-12: η > 1 is legitimate — multi-token / speculative batching.
    #[test]
    fn mtp_eta_above_one_is_legitimate_batching() {
        // 100 completion tokens delivered in 40 frames = 2.5 tokens/frame:
        // a batching server (vLLM MTP / EAGLE) emits several accepted tokens
        // per SSE frame, so η > 1 is expected, not an error.
        let eta = mtp_eta(100, 40);
        assert!((eta - 2.5).abs() < 1e-9);
        assert!(eta > 1.0, "η > 1 is legitimate: {eta}");
    }

    /// CF-12: the live snapshot uses the same canonical η (all frames).
    #[test]
    fn snapshot_mtp_uses_total_token_frames() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(10_000_000);
            let t1 = MonotonicInstant::now(); // 10 ms
            mock.increment(10_000_000);
            // 2 reasoning + 3 content frames = 5 total token frames.
            let frame_times: Vec<MonotonicInstant> = (0..5)
                .map(|_| {
                    let at = MonotonicInstant::now();
                    mock.increment(10_000_000);
                    at
                })
                .collect();
            let t3 = frame_times[0]; // first token
            let t_end = MonotonicInstant::now();
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 50,
                completion_tokens: 10,
            };
            let events: Vec<StreamEvent> = frame_times
                .iter()
                .enumerate()
                .map(|(i, at)| StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: if i < 2 {
                            Chunk::Reasoning("r".into())
                        } else {
                            Chunk::Content("c".into())
                        },
                        t_nanos: 0,
                        done: false,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let snap =
                SpeedEngine::single_stream_snapshot("http://x", "m", "short", &events, 256, None);
            // η = 10 completion / 5 total frames = 2.0 (not 10/3 content-only).
            assert!((snap.streams[0].mtp.unwrap() - 2.0).abs() < 1e-9);
        });
    }

    /// CF-25: a no-token-frame run keeps TTFT = 0.0 (N/A), is `degraded`,
    /// and does NOT fall back to the total stream time.
    #[test]
    fn no_token_frame_yields_na_ttft_and_degraded() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(5_000_000);
            let t1 = MonotonicInstant::now(); // 5 ms
            mock.increment(50_000_000);
            let t_end = MonotonicInstant::now(); // 55 ms
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: None, // no token frame
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 100,
                completion_tokens: 0,
            };
            let events = vec![
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(usage),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t_end,
                    timestamps: ts,
                },
                StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                },
            ];
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 0,
            };
            let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
            // CF-25: TTFT is the 0.0 N/A sentinel — NOT the 55 ms stream time.
            assert_eq!(r.ttft, 0.0, "no token frame → TTFT N/A (0.0)");
            assert!(r.is_degraded(), "no token frame → degraded");
            assert_eq!(r.pp_speed, 0.0, "no TTFT → no prefill bandwidth");
            assert!(!r.is_failed(), "usage-only, no error → not failed");
        });
    }

    /// CF-26: with `reasoning_tokens`, the decode window splits at the
    /// reasoning→content transition (answer phase, not thinking phase).
    #[test]
    fn decode_window_excludes_the_reasoning_span() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(5_000_000);
            let t1 = MonotonicInstant::now(); // 5 ms
                                              // 3 reasoning frames: 10, 15, 20 ms.
            mock.increment(5_000_000);
            let t_r1 = MonotonicInstant::now(); // 10 ms
            mock.increment(5_000_000);
            let t_r2 = MonotonicInstant::now(); // 15 ms
            mock.increment(5_000_000);
            let t_r3 = MonotonicInstant::now(); // 20 ms
                                                // 2 content frames: 30, 40 ms.
            mock.increment(10_000_000);
            let t_c1 = MonotonicInstant::now(); // 30 ms
            mock.increment(10_000_000);
            let t_c2 = MonotonicInstant::now(); // 40 ms
            mock.increment(10_000_000);
            let t_end = MonotonicInstant::now(); // 50 ms
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t_r1),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 100,
                completion_tokens: 5,
            };
            let frames = [
                (Chunk::Reasoning("a".into()), t_r1),
                (Chunk::Reasoning("b".into()), t_r2),
                (Chunk::Reasoning("c".into()), t_r3),
                (Chunk::Content("d".into()), t_c1),
                (Chunk::Content("e".into()), t_c2),
            ];
            let events: Vec<StreamEvent> = frames
                .iter()
                .map(|(chunk, at)| StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: chunk.clone(),
                        t_nanos: 0,
                        done: false,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 5,
            };
            // CF-26 (details present): answer phase = 2 content tokens over
            // the 30→40 ms content window = 200 t/s. The full-window blended
            // rate (5 / 30 ms ≈ 166.7) is NOT what we report.
            let details = UsageDetails {
                reasoning_tokens: 3,
                cached_tokens: 0,
            };
            let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, Some(details));
            assert!(
                (r.tg_speed - 200.0).abs() < 1e-6,
                "answer-phase {}",
                r.tg_speed
            );
            // Without details: the historical full-window blended rate.
            let r2 = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
            assert!(
                (r2.tg_speed - 5.0 / 0.03).abs() < 1e-6,
                "blended {}",
                r2.tg_speed
            );
            assert!((r.tg_speed - r2.tg_speed).abs() > 1.0, "split != blended");
        });
    }

    /// CF-26: a pure-CoT run (no content frames) has no answer phase → 0.
    #[test]
    fn pure_cot_run_has_no_answer_phase() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(10_000_000);
            let t1 = MonotonicInstant::now(); // 10 ms
                                              // 3 reasoning frames, no content.
            let frame_times: Vec<MonotonicInstant> = (0..3)
                .map(|_| {
                    let at = MonotonicInstant::now();
                    mock.increment(10_000_000);
                    at
                })
                .collect();
            let t3 = frame_times[0];
            let t_end = MonotonicInstant::now();
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 100,
                completion_tokens: 3,
            };
            let events: Vec<StreamEvent> = frame_times
                .iter()
                .map(|at| StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Reasoning("r".into()),
                        t_nanos: 0,
                        done: false,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 3,
            };
            // All 3 tokens are reasoning → content_tokens = 0 → no answer phase.
            let details = UsageDetails {
                reasoning_tokens: 3,
                cached_tokens: 0,
            };
            let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, Some(details));
            assert_eq!(r.tg_speed, 0.0, "pure CoT → no answer-phase decode");
        });
    }

    /// CF-26: the `cache_hit` flag is `true` iff `cached_tokens > 0`.
    #[test]
    fn cache_hit_set_when_cached_tokens_positive() {
        assert!(cache_hit(Some(&UsageDetails {
            reasoning_tokens: 0,
            cached_tokens: 5,
        })));
        // Zero cached tokens (a cold cache) → no hit.
        assert!(!cache_hit(Some(&UsageDetails::default())));
        // No detail at all (backend doesn't report) → no hit.
        assert!(!cache_hit(None));
    }

    // ── P2-A surfacing: CF-10 / CF-24 / CF-44 ─────────────────────────────

    /// CF-44: a plain-JSON (non-streaming) result — no token frames with a
    /// usage present and no error — is labeled E2E-only in the result, the
    /// JSON, and the result box.
    #[test]
    fn e2e_only_result_is_labeled_in_the_result_and_json() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(5_000_000);
            let t1 = MonotonicInstant::now();
            mock.increment(45_000_000);
            let t_end = MonotonicInstant::now();
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: None, // no token frame — the plain-JSON signature
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 100,
                completion_tokens: 12,
            };
            let events = vec![StreamEvent::Complete {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                looping: false,
                loop_excluded_tokens: 0,
            }];
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 0,
            };
            let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
            assert!(r.e2e_only, "frames==0 + usage + no error → E2E-only");
            assert!(!r.is_failed());
            // The label is published in the JSON and the result box.
            let v = json_report(&cfg(), std::slice::from_ref(&r));
            assert_eq!(v["results"][0]["e2e_only"], true);
            let box_ = format_result_box(&r, 1, 1, false).unwrap();
            assert!(box_.contains("e2e-only:"), "{box_}");
            assert!(box_.contains("non-streaming (total latency)"), "{box_}");
        });
    }

    /// CF-10: a usage count exceeding ~4× the observed token frames is
    /// flagged in the result and the JSON; a plausible usage is not.
    #[test]
    fn usage_discrepancy_is_flagged_in_the_result_and_json() {
        let (outcome, events) = completed_run();
        // 34 usage tokens over 3 frames (34 > 3×4) → the guard trips.
        assert!(outcome.has_usage_discrepancy());
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
        assert!(r.usage_discrepancy);
        let v = json_report(&cfg(), std::slice::from_ref(&r));
        assert_eq!(v["results"][0]["usage_discrepancy"], true);
        let box_ = format_result_box(&r, 1, 1, false).unwrap();
        assert!(box_.contains("usage check:"), "{box_}");
        // A plausible usage (12 tokens over 12 frames) is not flagged.
        let plausible = StreamOutcome {
            frames: 12,
            ..outcome
        };
        assert!(!plausible.has_usage_discrepancy());
    }

    /// CF-24: an inter-token gap beyond the 60 s histogram bound is clamped
    /// and counted; the count is published in the result and the JSON.
    #[test]
    fn clamped_itl_samples_are_counted_in_the_result() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(5_000_000);
            let t1 = MonotonicInstant::now();
            mock.increment(5_000_000);
            let t3 = MonotonicInstant::now(); // first token
            mock.increment(90_000_000_000); // 90 s gap → clamped at 60 s
            let t_last = MonotonicInstant::now();
            mock.increment(10_000_000);
            let t_end = MonotonicInstant::now();
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            };
            let usage = Usage {
                prompt_tokens: 100,
                completion_tokens: 2,
            };
            let frames = [
                (Chunk::Content("a".into()), t3),
                (Chunk::Content("b".into()), t_last),
            ];
            let events: Vec<StreamEvent> = frames
                .iter()
                .map(|(chunk, at)| StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: chunk.clone(),
                        t_nanos: 0,
                        done: false,
                    },
                    at: *at,
                    timestamps: ts,
                })
                .chain(std::iter::once(StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(usage),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                }))
                .collect();
            let outcome = StreamOutcome {
                timestamps: ts,
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
                error: None,
                looping: false,
                loop_excluded_tokens: 0,
                frames: 2,
            };
            let r = aggregate(&cfg(), &prompt(), &outcome, &events, None, None);
            assert_eq!(r.itl_clamped, 1, "the 90 s gap clamps and counts");
            let v = json_report(&cfg(), std::slice::from_ref(&r));
            assert_eq!(v["results"][0]["itl_clamped"], 1);
        });
    }
}
