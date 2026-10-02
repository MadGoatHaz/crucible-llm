//! Engine A — Speed & Latency: single-stream TTFT / PP / TG / MTP
//! orchestration (plan Chunk 7).
//!
//! This is the **single-stream CLI parity** engine: it drives one
//! [`StreamWorker`] (Chunk 5) per iteration against the configured
//! endpoint, classifies every parsed frame (Chunk 3), and computes the
//! blueprint §7 formulas —
//!
//! * `TTFT = T3 − T1` (first token arrival − request dispatched),
//! * `PP speed = prompt_tokens / TTFT`,
//! * `TG speed = completion_tokens / (stream_time − TTFT)`,
//! * `MTP η = completion_tokens / content_chunks`,
//!
//! with the same result-box layout, `--json` field set, and exit-code
//! semantics as `llmspeedtest.py` (`print_result_box` / `output_json` /
//! "exit 1 if all runs failed").
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
use crate::sse::{Chunk, Usage};
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
    /// Decode speed, tokens/s.
    pub tg_speed: f64,
    /// MTP multiplier, tokens per content chunk (`η`).
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
}

impl SpeedResult {
    /// The prototype's `all_failed` condition: an error with zero
    /// completion tokens.
    pub fn is_failed(&self) -> bool {
        self.error.is_some() && self.completion_tokens == 0
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
        })
    }
}

/// Round to `decimals` places (the prototype's `round(x, n)` in
/// `to_dict` / `output_json`).
fn round(v: f64, decimals: u32) -> f64 {
    let f = 10f64.powi(decimals as i32);
    (v * f).round() / f
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

        let start = MonotonicInstant::now();
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
                    state.update(self.live_snapshot(&events, &start));
                    batch = 0;
                }
            }
        }
        // Final publish (the terminal event may have been batched out).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&events, &start));
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
            ),
            events,
        )
    }

    /// Build a live [`MetricsSnapshot`] from the events collected so far
    /// (single-stream Engine A → the TUI's Live Monitor).
    fn live_snapshot(&self, events: &[StreamEvent], start: &MonotonicInstant) -> MetricsSnapshot {
        Self::single_stream_snapshot(
            &self.cfg.url,
            &self.cfg.model,
            self.cfg.mode.label(),
            events,
            start,
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
        start: &MonotonicInstant,
        max_tokens: u32,
        tokenizer: Option<&Tokenizer>,
    ) -> MetricsSnapshot {
        let _elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let mut itl = LatencyHistogram::default();
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
                        if let Some(prev) = last_token_at {
                            itl.record(prev.delta_nanos(at));
                        }
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
        let mtp = (content_frames > 0).then(|| tokens_received as f64 / content_frames as f64);
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
}

/// Synthesize one [`SpeedResult`] from a finished worker run plus its
/// channel events (blueprint §7 formulas; prototype semantics for
/// token-count fallbacks, chunk counting, and the error surface).
pub fn aggregate(
    cfg: &Config,
    prompt: &GeneratedPrompt,
    outcome: &StreamOutcome,
    events: &[StreamEvent],
    tokenizer: Option<&Tokenizer>,
) -> SpeedResult {
    let ts: StreamTimestamps = outcome.timestamps;
    let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
    // §7.1: TTFT = T3 − T1. No token frame (e.g. a usage-only stream) →
    // fall back to the total time, as the prototype does.
    let ttft = match ts.ttft_nanos() {
        Some(n) => n as f64 / 1e9,
        None => stream_time,
    };

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
    // the first content/reasoning token to the last (`T_first` → `T_last`),
    // which excludes prefill/TTFT and the trailing gap to `[DONE]`. This is
    // exactly the window vLLM / llama.cpp / SGLang / LM Studio / Ollama /
    // TGI use for their own decode rate, so `completion_tokens / window`
    // matches the server's displayed t/s.
    let (t_first, t_last) = token_window(events);
    let generation_time = match (t_first, t_last) {
        (Some(f), Some(l)) => (f.delta_nanos(&l) as f64 / 1e9).max(0.001),
        _ => (stream_time - ttft).max(0.001), // no token window: legacy fallback
    };
    let tg_speed = if looping {
        0.0
    } else {
        completion_tokens as f64 / generation_time
    };
    // §7.4: MTP η = tokens / content packets.
    let mtp_efficiency = if content_chunks > 0 {
        completion_tokens as f64 / content_chunks as f64
    } else {
        0.0
    };

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
pub fn all_failed(results: &[SpeedResult]) -> bool {
    !results.is_empty() && results.iter().all(SpeedResult::is_failed)
}

// ── Terminal formatting (prototype parity) ───────────────────────────────

const CYAN: &str = "\u{1b}[36m";
const RESET: &str = "\u{1b}[0m";
const BOLD: &str = "\u{1b}[1m";

/// The result box (parity with the prototype's `print_result_box`),
/// including its width quirk (52-char borders/title, 54-char label rows).
/// `None` for a hard failure — the caller prints the `FAILED` line.
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
    row(&mut lines, "TTFT:", &format!("{:.4}s", result.ttft));
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
    use crate::sse::{ParsedFrame, Usage};
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);

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
        // token = 10→30 ms = 20 ms); MTP = 34 / 2 content chunks.
        assert!((r.pp_speed - 128.0 / r.ttft).abs() < 1e-9);
        let gen = 0.02;
        assert!((r.tg_speed - 34.0 / gen).abs() < 1e-9, "tg {}", r.tg_speed);
        assert!((r.mtp_efficiency - 17.0).abs() < 1e-9);

        assert_eq!(r.model, "test-model");
        assert_eq!(r.mode, "short");
    }

    #[test]
    fn missing_usage_falls_back_to_text_estimate_and_flags_estimated() {
        let (mut outcome, events) = completed_run();
        outcome.usage = None;
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None);
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None);
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &[], None);
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);
        assert_eq!(r.error.as_deref(), Some("2 malformed chunk(s) skipped"));
        // Tokens were produced → not a failed run (prototype semantics).
        assert!(!r.is_failed());
        assert!(!all_failed(&[r]));
    }

    #[test]
    fn json_report_matches_prototype_shape() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);

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
        // (no `other_chunks`).
        let rj = v["results"][0].as_object().unwrap();
        assert_eq!(rj.len(), 15);
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
        ] {
            assert!(rj.contains_key(key), "missing {key}");
        }
        assert!(!rj.contains_key("other_chunks"));
        assert_eq!(rj["looping"], false);

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
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);
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
            let snap = SpeedEngine::single_stream_snapshot(
                "http://x", "m", "short", &events, &t0, 256, None,
            );
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
            let snap = SpeedEngine::single_stream_snapshot(
                "http://x", "m", "short", &events, &t0, 256, None,
            );
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
        let ok = aggregate(&cfg(), &prompt(), &outcome, &events, None);
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
        let failed = aggregate(&cfg(), &prompt(), &bad, &[], None);
        assert!(!all_failed(&[ok.clone(), failed.clone()]));
        assert!(all_failed(&[failed]));
        assert!(!all_failed(&[]));
    }

    #[test]
    fn result_box_mirrors_prototype_layout() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events, None);
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
        let fr = aggregate(&cfg(), &prompt(), &failed, &[], None);
        assert!(format_result_box(&fr, 1, 1, false).is_none());
    }

    #[test]
    fn summary_renders_for_two_valid_runs() {
        let (outcome, events) = completed_run();
        let a = aggregate(&cfg(), &prompt(), &outcome, &events, None);
        let b = aggregate(&cfg(), &prompt(), &outcome, &events, None);
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
        let failed = aggregate(&cfg(), &prompt(), &bad, &[], None);
        assert!(format_summary(&[a.clone(), failed], false).is_none());
    }
}
