//! Engine F — Flat Out: real-world maximum throughput at the optimal
//! user count.
//!
//! The goal is to measure what a production deployment actually serves:
//! the server's **total** tokens/sec when it is loaded at its
//! **sweet-spot concurrency** — the number of users Engine B's sweep
//! recommends (the highest level where every user still gets ≥ 29.4
//! t/s: 30 t/s ideal with a 2% margin). That aggregate number, with the
//! per-stream number confirming each user stays comfortable, is the
//! figure to quote when comparing servers or configurations.
//!
//! Method:
//!
//! * **`n` concurrent streams** (the sweet spot, or
//!   [`DEFAULT_STREAM_COUNT`] when Engine B did not run) — all spawned
//!   together via the shared [`WorkerPool`];
//! * **one minimal (~15-token) open-ended prompt per stream** — prefill
//!   takes <100 ms, negligible against the 60-second decode window;
//! * **`max_tokens = 100,000` + `ignore_eos`** — effectively unlimited,
//!   so the *only* stop condition is the 60-second window
//!   ([`WINDOW_SECS`]); `ignore_eos` keeps llama.cpp servers from letting
//!   the model end a stream on its own end-token (other backends ignore
//!   the field);
//! * **all streams start together and are all aborted at the 60 s mark**
//!   (aborting the supervisor drops every worker task, closing its
//!   socket so the server stops generating);
//! * **authoritative token counting** — each stream's count is the
//!   server's `usage.completion_tokens` when it arrives, else the
//!   re-tokenized text ([`authoritative_tokens`]) — never the raw frame
//!   tally, which undercounts batched servers (vLLM MTP) by 30–40 %.
//!
//! The result records the **aggregate t/s** (the headline), the
//! **per-stream t/s** (aggregate ÷ streams — ≈ 30 confirms the sweet
//! spot), the total tokens, and the pooled TTFT / ITL evidence.
//!
//! Measurement-isolation note: all timing is captured in the workers
//! (quanta, `T0..Tn`); this module only takes deltas of those records and
//! counts the tokens that arrive. Nothing here touches the TUI render
//! path.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::client::pool::{PoolEvent, WorkerPool};
use crate::client::{authoritative_tokens, StreamEvent};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::histogram::LatencyHistogram;
use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamMetric, StreamStatus};
use crate::prompt::tokenizer::count_tokens;
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::{Chunk, Usage};
use crate::timing::MonotonicInstant;

/// The continuous run window (seconds). All streams start together and
/// are all aborted when it elapses — the only real stop condition of the
/// test.
pub const WINDOW_SECS: f64 = 60.0;

/// The run window in nanoseconds (the same domain as
/// [`MonotonicInstant::delta_nanos`]) for the exact window comparison.
/// Kept in sync with [`WINDOW_SECS`].
const WINDOW_NS: u64 = 60 * 1_000_000_000;

/// The `max_tokens` requested per stream: 100,000 — effectively
/// unlimited. The 60-second window, not this cap, ends the test.
pub const MAX_TOKENS: u32 = 100_000;

/// The stream count when Engine B did not run (there is no sweet spot to
/// read): a small, always-valid load.
pub const DEFAULT_STREAM_COUNT: usize = 3;

/// The minimal open-ended prompt (~15 tokens) every stream runs.
///
/// Short enough that prefill takes <100 ms (negligible against the 60 s
/// window), open-ended enough that the model keeps generating without
/// stopping, and not a question (questions get short answers).
pub const MINIMAL_PROMPT: &str =
    "Continue this story: The old lighthouse keeper walked down the spiral stairs and";

/// Where the run's stream count came from (the JSON export's
/// `stream_count_source` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCountSource {
    /// The practical sweet spot from Engine B's sweep (the recommended
    /// production concurrency).
    ConcurrencySweetSpot,
    /// The built-in default ([`DEFAULT_STREAM_COUNT`]) — Engine B did not
    /// run, so there was no sweet spot to read.
    Default,
}

impl StreamCountSource {
    /// The JSON label (`"concurrency_sweet_spot"` / `"default"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConcurrencySweetSpot => "concurrency_sweet_spot",
            Self::Default => "default",
        }
    }
}

/// The complete Flat Out result: `n` concurrent streams (the sweet spot)
/// running for one [`WINDOW_SECS`]-second window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOutResult {
    /// The number of concurrent streams (the sweet spot, or the default).
    pub stream_count: usize,
    /// Where the stream count came from.
    pub stream_count_source: StreamCountSource,
    /// Total tokens generated across **all** streams in the window.
    pub total_tokens: u64,
    /// How long the run took (seconds; ~[`WINDOW_SECS`], or less when
    /// every stream ended on its own).
    pub duration_secs: f64,
    /// The headline number: `total_tokens / duration_secs` — the
    /// server's real-world maximum throughput at full sweet-spot load.
    pub aggregate_tps: f64,
    /// `aggregate_tps / stream_count` — what each user gets at the
    /// recommended load (≈ 30 t/s confirms the sweet spot).
    pub per_stream_tps: f64,
    /// Mean time to first token across the streams (milliseconds).
    pub ttft_avg_ms: f64,
    /// Inter-token latency p50, pooled across all streams (milliseconds).
    pub itl_p50_ms: f64,
    /// Inter-token latency p99, pooled across all streams (milliseconds).
    pub itl_p99_ms: f64,
}

impl FlatOutResult {
    /// Build the result from the events observed per stream.
    ///
    /// Each stream's token count follows the **Authoritative Counting
    /// Rule** ([`authoritative_tokens`]): the server's
    /// `usage.completion_tokens` when it arrived (the rare path — the
    /// 60 s window normally aborts the streams first), else the
    /// re-tokenized reasoning + content text (the exact tokenizer when
    /// one is supplied, else `chars/4`). Counting SSE frames is never
    /// used: batched servers (vLLM MTP) pack multiple tokens into one
    /// frame, which is the 30–40 % undercount this engine used to show.
    ///
    /// The **aggregate t/s** is the total over the *wall* window
    /// (`total_tokens / duration_secs`) — prefill and any trailing gap
    /// included, exactly what a full-load deployment sustains. The ITL
    /// percentiles are the gaps between consecutive token frames, pooled
    /// across all streams.
    #[must_use]
    pub fn from_events(
        per_stream: &[Vec<StreamEvent>],
        duration_secs: f64,
        tokenizer: Option<&Tokenizer>,
        source: StreamCountSource,
    ) -> Self {
        let stream_count = per_stream.len().max(1);
        let mut total_tokens = 0u64;
        let mut ttfts_ms: Vec<f64> = Vec::new();
        let mut itl = LatencyHistogram::default();
        for events in per_stream {
            // The stream's authoritative count (usage → text estimate).
            let (tokens, _) = authoritative_tokens(events, stream_usage(events), tokenizer);
            total_tokens += tokens;
            // Pooled ITL: the gaps between this stream's consecutive
            // token frames.
            let mut prev: Option<MonotonicInstant> = None;
            for e in events {
                if let StreamEvent::Frame { frame, at, .. } = e {
                    if frame.chunk.is_token() {
                        if let Some(p) = prev {
                            itl.record(p.delta_nanos(at));
                        }
                        prev = Some(*at);
                    }
                }
            }
            // TTFT of the stream's first token frame.
            if let Some(ts) = events.iter().find_map(|e| match e {
                StreamEvent::Frame {
                    frame, timestamps, ..
                } if frame.chunk.is_token() => Some(*timestamps),
                _ => None,
            }) {
                if let Some(ns) = ts.ttft_nanos() {
                    ttfts_ms.push(ns as f64 / 1e6);
                }
            }
        }
        let aggregate_tps = if duration_secs > 0.0 {
            total_tokens as f64 / duration_secs
        } else {
            0.0
        };
        let (itl_p50_ms, itl_p99_ms) = itl_percentiles_ms(&itl);
        let ttft_avg_ms = if ttfts_ms.is_empty() {
            0.0
        } else {
            ttfts_ms.iter().sum::<f64>() / ttfts_ms.len() as f64
        };
        Self {
            stream_count,
            stream_count_source: source,
            total_tokens,
            duration_secs,
            aggregate_tps,
            per_stream_tps: aggregate_tps / stream_count as f64,
            ttft_avg_ms,
            itl_p50_ms,
            itl_p99_ms,
        }
    }

    /// The one-line summary for the sequence header / headless report.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "{agg:.1} t/s aggregate ({n} streams) · {ps:.1} t/s per stream · {tok} tok in {d:.0}s",
            agg = self.aggregate_tps,
            n = self.stream_count,
            ps = self.per_stream_tps,
            tok = self.total_tokens,
            d = self.duration_secs
        )
    }

    /// The `--json` object for this result.
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "stream_count": self.stream_count,
            "stream_count_source": self.stream_count_source,
            "total_tokens": self.total_tokens,
            "duration_secs": self.duration_secs,
            "aggregate_tps": self.aggregate_tps,
            "per_stream_tps": self.per_stream_tps,
            "ttft_avg_ms": self.ttft_avg_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p99_ms": self.itl_p99_ms,
        })
    }
}

/// `(p50 ms, p99 ms)` of a pooled inter-token-latency histogram (the
/// same histogram Engine A uses for its ITL distribution).
fn itl_percentiles_ms(h: &LatencyHistogram) -> (f64, f64) {
    (h.p50() / 1e6, h.p99() / 1e6)
}

/// A stream's authoritative `usage` (the last usage frame, or the
/// terminal `Complete` event's usage — they mirror each other).
fn stream_usage(events: &[StreamEvent]) -> Option<Usage> {
    let mut usage = None;
    for e in events {
        match e {
            StreamEvent::Frame { frame, .. } => {
                if let Chunk::Usage(u) = &frame.chunk {
                    usage = Some(*u);
                }
            }
            StreamEvent::Complete { usage: u, .. } => usage = *u,
            StreamEvent::Failed { .. } => {}
        }
    }
    usage
}

/// Convert one pooled (stream-tagged) event back to the untagged
/// [`StreamEvent`] of the stream it came from (the per-stream event
/// lists the result / snapshot builders consume).
fn to_stream_event(e: &PoolEvent) -> StreamEvent {
    match e {
        PoolEvent::Frame {
            frame,
            at,
            timestamps,
            ..
        } => StreamEvent::Frame {
            frame: frame.clone(),
            at: *at,
            timestamps: *timestamps,
        },
        PoolEvent::Complete {
            timestamps,
            usage,
            premature,
            malformed_frames,
            looping,
            loop_excluded_tokens,
            ..
        } => StreamEvent::Complete {
            timestamps: *timestamps,
            usage: *usage,
            premature: *premature,
            malformed_frames: *malformed_frames,
            looping: *looping,
            loop_excluded_tokens: *loop_excluded_tokens,
        },
        PoolEvent::Failed {
            timestamps,
            error,
            looping,
            loop_excluded_tokens,
            ..
        } => StreamEvent::Failed {
            timestamps: *timestamps,
            error: error.clone(),
            looping: *looping,
            loop_excluded_tokens: *loop_excluded_tokens,
        },
    }
}

/// Engine construction errors.
#[derive(Debug, Error)]
pub enum FlatOutError {
    /// The shared `reqwest` client could not be built.
    #[error("failed to build HTTP client: {0}")]
    Client(#[from] reqwest::Error),
    /// An explicit `--tokenizer` file failed to load/parse.
    #[error("transparent")]
    Tokenizer(#[from] TokenizerError),
}

/// The Flat Out engine: `n` concurrent streams (the sweet spot) with the
/// minimal prompt, an effectively-unlimited `max_tokens` cap, and one
/// [`WINDOW_SECS`]-second window that ends the test.
#[derive(Debug)]
pub struct FlatOutEngine {
    cfg: Config,
    client: reqwest::Client,
    generator: PromptGenerator,
    /// The number of concurrent streams (the sweet spot, or the default).
    stream_count: usize,
    /// Where the stream count came from (the result's
    /// `stream_count_source`).
    stream_count_source: StreamCountSource,
    /// Optional TUI seam: publish live [`MetricsSnapshot`]s.
    metrics: Option<Arc<MetricsState>>,
    /// Optional sequence seam: publish [`EngineProgress`].
    progress: Option<Arc<ProgressBus>>,
    /// Optional `Space`-key pause gate.
    pause: Option<Arc<RunPause>>,
    /// Optional run logger.
    logger: Option<Arc<RunLogger>>,
}

impl FlatOutEngine {
    /// Build the engine from the resolved config (the stream count
    /// defaults to [`DEFAULT_STREAM_COUNT`] — use
    /// [`stream_count`](Self::stream_count) to drive it from Engine B's
    /// sweet spot).
    pub fn new(cfg: &Config) -> Result<Self, FlatOutError> {
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
            stream_count: DEFAULT_STREAM_COUNT,
            stream_count_source: StreamCountSource::Default,
            metrics: None,
            progress: None,
            pause: None,
            logger: None,
        })
    }

    /// Set the stream count (e.g. Engine B's sweet spot) and its source.
    pub fn stream_count(mut self, n: usize, source: StreamCountSource) -> Self {
        self.stream_count = n.max(1);
        self.stream_count_source = source;
        self
    }

    /// Attach a [`MetricsState`] publisher.
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach a [`ProgressBus`].
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach the `Space`-key pause gate.
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger.
    pub fn logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// Run the `n`-stream window and return the result.
    ///
    /// All streams are spawned together (one [`WorkerPool`] level),
    /// drained until the [`WINDOW_SECS`] window elapses (or every stream
    /// ends on its own, whichever comes first), then aborted: the
    /// supervisor's task tree is dropped, every socket closes, and the
    /// server stops generating.
    pub async fn run(&self) -> FlatOutResult {
        let prompt = self.generate_prompt();
        let n = self.stream_count;

        // The `Space`-key pause: hold before the streams go out.
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }

        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: 0.0,
                streams: n,
                tokens: 0,
                tps: 0.0,
            });
        }

        if self.stream_count_source == StreamCountSource::Default {
            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!("No concurrency data — using default {n} streams"),
                );
            }
        } else {
            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!(
                        "Real-world max throughput: {n} streams (concurrency sweet spot) for {WINDOW_SECS:.0}s, max_tokens {MAX_TOKENS}"
                    ),
                );
            }
        }

        let pool = WorkerPool::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            MAX_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)))
        // The 60 s window — not the model's own end-token — is the only
        // stop: llama.cpp servers honor `ignore_eos`, other backends
        // ignore the field (graceful degradation).
        .ignore_eos(true);
        let pool = match &self.cfg.api_key {
            Some(key) => pool.api_key(key.clone()),
            None => pool,
        };
        let pool = match &self.logger {
            Some(l) => pool.logger(l.clone()),
            None => pool,
        };

        // Spawn all `n` streams together; drain the aggregate channel
        // until the window elapses (then abort) or every stream ends on
        // its own. `tokio::time::timeout` bounds each `recv` to the time
        // remaining in the window, so a slow server is cut at 60 s.
        let (mut rx, supervisor) = pool.spawn_tagged(n, "F");
        let start = MonotonicInstant::now();
        let mut per_stream: Vec<Vec<StreamEvent>> = vec![Vec::new(); n];
        let mut terminals = 0usize;
        let mut batch = 0u32;
        loop {
            let elapsed_ns = start.delta_nanos(&MonotonicInstant::now());
            if elapsed_ns >= WINDOW_NS {
                break;
            }
            let remaining = Duration::from_nanos(WINDOW_NS - elapsed_ns);
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    if let Some(bucket) = per_stream.get_mut(event.stream() as usize) {
                        bucket.push(to_stream_event(&event));
                    }
                    if event.is_terminal() {
                        terminals += 1;
                    }
                    batch += 1;
                    if batch >= 16 || terminals == n {
                        // Live TUI snapshot + sequence progress bus (the
                        // 10 Hz ticker mirrors the bus into the state
                        // slot). While the window is open the run is live
                        // (`finished = false`).
                        if let Some(state) = &self.metrics {
                            state.update(self.live_snapshot(&per_stream, &start, false));
                        }
                        if let Some(bus) = &self.progress {
                            let (tokens, tps) =
                                Self::observed(&per_stream, self.generator.tokenizer().as_deref());
                            bus.publish(EngineProgress::FlatOut {
                                elapsed_secs: start.elapsed().as_secs_f64(),
                                streams: n,
                                tokens,
                                tps,
                            });
                        }
                        batch = 0;
                    }
                    if terminals == n {
                        break; // every stream ended before the window
                    }
                }
                Ok(None) => break, // channel closed (all workers done)
                Err(_) => break,   // 60 s window elapsed
            }
        }

        // Stop the run: abort the supervisor task — it owns every worker
        // task (and its forwarder), so they are aborted with it and their
        // sockets close (the server stops generating). A no-op when all
        // streams already ended on their own.
        supervisor.abort();
        drop(rx);

        let elapsed = start.elapsed().as_secs_f64();
        let result = FlatOutResult::from_events(
            &per_stream,
            elapsed,
            self.generator.tokenizer().as_deref(),
            self.stream_count_source,
        );

        // Final publish (the last batch may not have flushed). The 60 s
        // window is over and the workers are aborted: every stream is a
        // *completed* benchmark stream, reported `Done` (the
        // accumulator counts each one's decode rate — the last engine
        // must appear in OVERALL METRICS).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&per_stream, &start, true));
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: elapsed,
                streams: n,
                tokens: result.total_tokens,
                tps: result.aggregate_tps,
            });
        }
        if let Some(l) = &self.logger {
            l.info(
                Context::EngineF,
                format!(
                    "Flat Out: {n} streams, {} tok in {:.1}s — {:.1} t/s aggregate, {:.1} t/s per stream (TTFT avg {:.0} ms, ITL p50 {:.1} / p99 {:.1} ms)",
                    result.total_tokens,
                    result.duration_secs,
                    result.aggregate_tps,
                    result.per_stream_tps,
                    result.ttft_avg_ms,
                    result.itl_p50_ms,
                    result.itl_p99_ms
                ),
            );
        }

        result
    }

    /// The prompt for this run: the minimal (~15-token) open-ended
    /// [`MINIMAL_PROMPT`], so prefill is <100 ms — negligible against the
    /// 60-second decode window.
    pub fn generate_prompt(&self) -> GeneratedPrompt {
        let c = count_tokens(MINIMAL_PROMPT, self.generator.tokenizer().as_deref());
        GeneratedPrompt {
            text: MINIMAL_PROMPT.to_string(),
            token_count: c.tokens,
            estimated: c.estimated,
            nocache: false,
        }
    }

    /// The live progress figures: the authoritative token count so far
    /// (summed across all streams) + its decode rate over the global
    /// first→last token window (prefill/TTFT excluded — the research's
    /// Timing Boundary Rule).
    fn observed(per_stream: &[Vec<StreamEvent>], tokenizer: Option<&Tokenizer>) -> (u64, f64) {
        let mut total = 0u64;
        let mut first: Option<MonotonicInstant> = None;
        let mut last: Option<MonotonicInstant> = None;
        for events in per_stream {
            total += authoritative_tokens(events, stream_usage(events), tokenizer).0;
            for e in events {
                if let StreamEvent::Frame { frame, at, .. } = e {
                    if frame.chunk.is_token() {
                        if first.is_none() {
                            first = Some(*at);
                        }
                        last = Some(*at);
                    }
                }
            }
        }
        let tps = match (first, last) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 {
                    total as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        (total, tps)
    }

    /// Build a live [`MetricsSnapshot`] from the per-stream events
    /// collected so far in the run.
    ///
    /// The decode rate uses the same formula as Engine A and the Overall
    /// Metrics panel: `tokens_received / (T_last − T_first)` — over all
    /// streams combined (the live aggregate).
    ///
    /// `finished` marks the **final** publish (the 60 s window elapsed
    /// and the workers were aborted): every stream is a *completed*
    /// benchmark stream and is reported `Done`. The abort means no
    /// terminal `Complete` event ever arrives for the in-flight streams,
    /// so without this flag they would stay `Streaming` forever and the
    /// `OverallAccumulator`'s non-terminal → terminal detector would
    /// never count this engine's decode rate — the last engine would
    /// silently vanish from the OVERALL METRICS panel.
    fn live_snapshot(
        &self,
        per_stream: &[Vec<StreamEvent>],
        start: &MonotonicInstant,
        finished: bool,
    ) -> MetricsSnapshot {
        let n = per_stream.len().max(1);
        let elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let (tokens_received, decode_rate) =
            Self::observed(per_stream, self.generator.tokenizer().as_deref());

        // Pooled ITL + one stream row per stream.
        let mut itl = LatencyHistogram::default();
        let mut active = 0usize;
        let mut streams: Vec<StreamMetric> = Vec::with_capacity(n);
        for (id, events) in per_stream.iter().enumerate() {
            let mut prev: Option<MonotonicInstant> = None;
            let mut first_token: Option<MonotonicInstant> = None;
            let mut last_token: Option<MonotonicInstant> = None;
            let mut ttft_s: Option<f64> = None;
            let mut complete = false;
            let mut failed = false;
            for e in events {
                match e {
                    StreamEvent::Frame {
                        frame,
                        at,
                        timestamps,
                    } => {
                        if frame.chunk.is_token() {
                            if let Some(p) = prev {
                                itl.record(p.delta_nanos(at));
                            }
                            prev = Some(*at);
                            if first_token.is_none() {
                                first_token = Some(*at);
                                if let Some(ns) = timestamps.ttft_nanos() {
                                    ttft_s = Some(ns as f64 / 1e9);
                                }
                            }
                            last_token = Some(*at);
                        }
                    }
                    StreamEvent::Complete { .. } => complete = true,
                    StreamEvent::Failed { .. } => failed = true,
                }
            }
            let (tokens, gen_tps) = Self::stream_rate(
                events,
                first_token,
                last_token,
                self.generator.tokenizer().as_deref(),
            );
            let state = if finished {
                StreamStatus::Done
            } else if failed {
                StreamStatus::Error
            } else if complete {
                StreamStatus::Done
            } else if tokens > 0 {
                StreamStatus::Streaming
            } else {
                StreamStatus::Waiting
            };
            if state == StreamStatus::Streaming {
                active += 1;
            }
            streams.push(StreamMetric {
                id: id as u32,
                kind: "Content".to_string(),
                state,
                tg_tokens: (tokens > 0).then_some(tokens),
                gen_tps: (gen_tps > 0.0).then_some(gen_tps),
                ttft_s,
                progress: (elapsed_ns as f64 / 1e9 / WINDOW_SECS).min(1.0),
                ..Default::default()
            });
        }

        MetricsSnapshot {
            endpoint: self.cfg.url.clone(),
            backend: String::new(),
            model: self.cfg.model.clone(),
            mode: "FlatOut".to_string(),
            aggregate_tps: decode_rate,
            active_streams: active,
            total_streams: n,
            itl_p50_ns: itl.p50() as u64,
            itl_p90_ns: itl.p90() as u64,
            itl_p99_ns: itl.p99() as u64,
            itl_p999_ns: itl.p999() as u64,
            completion_tokens: tokens_received,
            // The decode-rate numerator: the authoritative token count
            // (usage → text estimate), summed across all streams.
            observed_frames: tokens_received,
            prompt_tokens: 0,
            status: if finished {
                StreamStatus::Done
            } else {
                StreamStatus::Streaming
            },
            streams,
            ..Default::default()
        }
    }

    /// One stream's authoritative token count + its decode rate over its
    /// own first→last token window (`tokens / span` — the same formula
    /// the Overall Metrics panel records per stream).
    fn stream_rate(
        events: &[StreamEvent],
        first_token: Option<MonotonicInstant>,
        last_token: Option<MonotonicInstant>,
        tokenizer: Option<&Tokenizer>,
    ) -> (u64, f64) {
        let (tokens, _) = authoritative_tokens(events, stream_usage(events), tokenizer);
        let gen_tps = match (first_token, last_token) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 && tokens > 0 {
                    tokens as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        (tokens, gen_tps)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sse::{Chunk, ParsedFrame, Usage};
    use crate::timing::StreamTimestamps;

    /// A single content frame at `at` with the given timestamps.
    fn content_frame(at: MonotonicInstant, ts: StreamTimestamps) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content("a".into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: ts,
        }
    }

    /// A content frame carrying `text` (for the text-estimate paths).
    fn content_frame_text(text: &str, at: MonotonicInstant, ts: StreamTimestamps) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content(text.into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: ts,
        }
    }

    fn ts_at(t: MonotonicInstant) -> StreamTimestamps {
        StreamTimestamps {
            t0: Some(t),
            t1: Some(t),
            t2: Some(t),
            t3: Some(t),
            t_end: Some(t),
        }
    }

    /// A timestamp record with `t_end` unset — the shape of every frame
    /// captured from an *aborted* stream (the worker is dropped before it
    /// can send the terminal event, so `Tn` is never recorded).
    fn ts_open(t: MonotonicInstant) -> StreamTimestamps {
        StreamTimestamps {
            t0: Some(t),
            t1: Some(t),
            t2: Some(t),
            t3: Some(t),
            t_end: None,
        }
    }

    #[test]
    fn window_is_60_seconds() {
        assert!((WINDOW_SECS - 60.0).abs() < 1e-9);
    }

    #[test]
    fn max_tokens_is_effectively_unlimited() {
        assert_eq!(MAX_TOKENS, 100_000);
    }

    #[test]
    fn minimal_prompt_is_short() {
        // ~15 tokens; the chars/4 estimate keeps this a soft bound.
        let c = count_tokens(MINIMAL_PROMPT, None);
        assert!(c.tokens <= 30, "prompt should be ~15 tokens: {}", c.tokens);
    }

    #[test]
    fn default_stream_count_is_three() {
        assert_eq!(DEFAULT_STREAM_COUNT, 3);
    }

    #[test]
    fn stream_count_source_labels() {
        assert_eq!(
            StreamCountSource::ConcurrencySweetSpot.as_str(),
            "concurrency_sweet_spot"
        );
        assert_eq!(StreamCountSource::Default.as_str(), "default");
    }

    #[test]
    fn engine_defaults_to_three_streams() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(engine.stream_count, DEFAULT_STREAM_COUNT);
        assert_eq!(engine.stream_count_source, StreamCountSource::Default);
    }

    #[test]
    fn engine_stream_count_builder_sets_count_and_source() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        let engine = engine.stream_count(8, StreamCountSource::ConcurrencySweetSpot);
        assert_eq!(engine.stream_count, 8);
        assert_eq!(
            engine.stream_count_source,
            StreamCountSource::ConcurrencySweetSpot
        );
        // The builder floors at one stream (never zero).
        let engine = FlatOutEngine::new(&Config::default())
            .unwrap()
            .stream_count(0, StreamCountSource::Default);
        assert_eq!(engine.stream_count, 1);
    }

    #[test]
    fn from_events_prefers_server_usage_over_the_text_estimate() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // 100 ms (first token)
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // 200 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 300 ms (last token)
            let ts = ts_at(t0);
            // Three content frames over 100→300 ms, plus a usage frame
            // claiming 100,000 tokens.
            let events = vec![
                content_frame(t1, ts),
                content_frame(t2, ts),
                content_frame(t3, ts),
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(Usage {
                            prompt_tokens: 15,
                            completion_tokens: 100_000,
                        }),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t3,
                    timestamps: ts,
                },
            ];
            // The server's usage is the authoritative count (primary
            // method); the aggregate is total / wall window (10 s).
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.stream_count, 1);
            assert_eq!(r.total_tokens, 100_000, "server usage is authoritative");
            assert!(
                (r.aggregate_tps - 10_000.0).abs() < 1.0,
                "aggregate: {}",
                r.aggregate_tps
            );
            assert!(
                (r.per_stream_tps - 10_000.0).abs() < 1.0,
                "per-stream: {}",
                r.per_stream_tps
            );
            assert!((r.duration_secs - 10.0).abs() < 1e-9);
        });
    }

    #[test]
    fn from_events_sums_streams_and_pools_itl() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // 100 ms
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // 200 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 300 ms
            let ts = ts_at(t0);
            // Two streams: 4-char content frames (chars/4 → 1 token each).
            // Stream 0: two tokens (100→200 ms, a 100 ms gap); stream 1:
            // one token (300 ms). Total 3 tokens over the 10 s window.
            let s0 = vec![
                content_frame_text("abcd", t1, ts),
                content_frame_text("efgh", t2, ts),
            ];
            let s1 = vec![content_frame_text("ijkl", t3, ts)];
            let r = FlatOutResult::from_events(&[s0, s1], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.stream_count, 2);
            assert_eq!(r.total_tokens, 3, "1 + 1 + 1 token per 4-char frame");
            assert!(
                (r.aggregate_tps - 0.3).abs() < 1e-9,
                "aggregate: {}",
                r.aggregate_tps
            );
            assert!(
                (r.per_stream_tps - 0.15).abs() < 1e-9,
                "per-stream: {}",
                r.per_stream_tps
            );
            // The pooled ITL has one 100 ms sample.
            assert!((r.itl_p50_ms - 100.0).abs() < 0.5, "p50: {}", r.itl_p50_ms);
            assert!((r.itl_p99_ms - 100.0).abs() < 0.5, "p99: {}", r.itl_p99_ms);
        });
    }

    #[test]
    fn from_events_no_usage_estimates_from_the_text() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // first token
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // last token
            let ts = ts_at(t0);
            // No usage frame (the aborted-stream case), no tokenizer: the
            // count is estimated from the full text (`chars/4`), NOT the
            // raw frame tally — counting frames undercounts batched
            // servers. Two frames of 4 chars each → 8 chars → 2 tokens.
            let events = vec![
                content_frame_text("abcd", t1, ts),
                content_frame_text("efgh", t2, ts),
            ];
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(
                r.total_tokens, 2,
                "no usage + no tokenizer → chars/4 estimate"
            );
            assert!(
                (r.aggregate_tps - 0.2).abs() < 1e-9,
                "aggregate: {}",
                r.aggregate_tps
            );
        });
    }

    #[test]
    fn from_events_no_usage_retokenizes_with_the_tokenizer() {
        let tok =
            Tokenizer::from_json(include_str!("../../tests/fixtures/mini_tokenizer.json")).unwrap();
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(50_000_000);
            let t1 = MonotonicInstant::now(); // first token
            mock.increment(50_000_000);
            let t2 = MonotonicInstant::now(); // last token
            let ts = ts_at(t0);
            // No usage frame; two content frames "apple " + "apple" → the
            // word-level tokenizer counts "apple apple" as exactly 2
            // tokens.
            let events = vec![
                content_frame_text("apple ", t1, ts),
                content_frame_text("apple", t2, ts),
            ];
            let r =
                FlatOutResult::from_events(&[events], 10.0, Some(&tok), StreamCountSource::Default);
            assert_eq!(r.total_tokens, 2, "re-tokenized count (apple apple)");
        });
    }

    #[test]
    fn from_events_zero_duration_gives_zero_tps() {
        let t0 = MonotonicInstant::now();
        let events = vec![content_frame_text("abcd", t0, ts_at(t0))];
        let r = FlatOutResult::from_events(&[events], 0.0, None, StreamCountSource::Default);
        assert_eq!(r.total_tokens, 1); // 4 chars → chars/4 = 1 token
        assert_eq!(r.aggregate_tps, 0.0);
        assert_eq!(r.per_stream_tps, 0.0);
    }

    #[test]
    fn from_events_ignores_control_and_counts_reasoning() {
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let mut reasoning = content_frame(t0, ts);
        if let StreamEvent::Frame { frame, .. } = &mut reasoning {
            frame.chunk = Chunk::Reasoning("think".into());
        }
        let control = StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Control,
                t_nanos: 0,
                done: true,
            },
            at: t0,
            timestamps: ts,
        };
        let events = vec![reasoning, control];
        let r = FlatOutResult::from_events(&[events], 5.0, None, StreamCountSource::Default);
        // One reasoning token frame; the `[DONE]` control frame is not a
        // token.
        assert_eq!(r.total_tokens, 1);
    }

    #[test]
    fn from_events_averages_ttft_across_streams() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            // Stream 0: request at t0, first token at 100 ms (TTFT 100 ms);
            // stream 1: first token at 300 ms (TTFT 300 ms) → mean 200 ms.
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            mock.increment(200_000_000);
            let t2 = MonotonicInstant::now();
            // TTFT = T3 − T1: the request went out at t0, the first token
            // frame decoded at its arrival instant.
            let ts1 = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t0),
                t2: Some(t1),
                t3: Some(t1),
                t_end: None,
            };
            let ts2 = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t0),
                t2: Some(t2),
                t3: Some(t2),
                t_end: None,
            };
            let s0 = content_frame_text("abcd", t1, ts1);
            let s1 = content_frame_text("efgh", t2, ts2);
            let r = FlatOutResult::from_events(
                &[vec![s0], vec![s1]],
                5.0,
                None,
                StreamCountSource::Default,
            );
            assert!(
                (r.ttft_avg_ms - 200.0).abs() < 1.0,
                "ttft avg: {}",
                r.ttft_avg_ms
            );
        });
    }

    #[test]
    fn to_stream_event_round_trips_all_variants() {
        let (clock, _mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            let ts = ts_at(t0);
            let frame = PoolEvent::from_event(
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Content("tok".into()),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t0,
                    timestamps: ts,
                },
                3,
            );
            assert!(matches!(to_stream_event(&frame), StreamEvent::Frame { .. }));
            let complete = PoolEvent::from_event(
                StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: 20,
                    }),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                },
                1,
            );
            assert!(matches!(
                to_stream_event(&complete),
                StreamEvent::Complete {
                    usage: Some(u),
                    ..
                } if u.completion_tokens == 20
            ));
            let failed = PoolEvent::from_event(
                StreamEvent::Failed {
                    timestamps: ts,
                    error: crate::client::StreamError::Connection("refused".into()),
                    looping: false,
                    loop_excluded_tokens: 0,
                },
                2,
            );
            assert!(matches!(
                to_stream_event(&failed),
                StreamEvent::Failed { .. }
            ));
        });
    }

    #[test]
    fn live_snapshot_reports_per_stream_states() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        let t0 = MonotonicInstant::now();
        // Two streams: one streaming (tokens, no terminal), one failed.
        let s0 = vec![content_frame_text("abcd", t0, ts_open(t0))];
        let s1 = vec![StreamEvent::Failed {
            timestamps: ts_at(t0),
            error: crate::client::StreamError::Connection("refused".into()),
            looping: false,
            loop_excluded_tokens: 0,
        }];
        // Live publish: stream 0 Streaming, stream 1 Error.
        let live = engine.live_snapshot(&[s0.clone(), s1.clone()], &t0, false);
        assert_eq!(live.total_streams, 2);
        assert_eq!(live.active_streams, 1);
        assert_eq!(live.streams[0].state, StreamStatus::Streaming);
        assert_eq!(live.streams[1].state, StreamStatus::Error);
        // Final publish (window elapsed, workers aborted): every stream
        // is a completed benchmark stream, reported Done (the accumulator
        // counts each one's decode rate).
        let final_snap = engine.live_snapshot(&[s0, s1], &t0, true);
        assert_eq!(final_snap.status, StreamStatus::Done);
        assert_eq!(final_snap.active_streams, 0);
        assert_eq!(final_snap.streams[0].state, StreamStatus::Done);
        assert_eq!(final_snap.streams[1].state, StreamStatus::Done);
    }

    #[test]
    fn summary_line_carries_the_headline_numbers() {
        let r = FlatOutResult {
            stream_count: 8,
            stream_count_source: StreamCountSource::ConcurrencySweetSpot,
            total_tokens: 17_124,
            duration_secs: 60.0,
            aggregate_tps: 285.4,
            per_stream_tps: 35.7,
            ttft_avg_ms: 312.5,
            itl_p50_ms: 28.3,
            itl_p99_ms: 65.1,
        };
        let line = r.summary_line();
        assert!(line.contains("285.4 t/s aggregate (8 streams)"), "{line}");
        assert!(line.contains("35.7 t/s per stream"), "{line}");
        assert!(line.contains("17124 tok in 60s"), "{line}");
    }

    #[test]
    fn to_dict_is_the_full_export_object() {
        let r = FlatOutResult {
            stream_count: 8,
            stream_count_source: StreamCountSource::ConcurrencySweetSpot,
            total_tokens: 17_124,
            duration_secs: 60.0,
            aggregate_tps: 285.4,
            per_stream_tps: 35.7,
            ttft_avg_ms: 312.5,
            itl_p50_ms: 28.3,
            itl_p99_ms: 65.1,
        };
        let v = r.to_dict();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "stream_count",
                "stream_count_source",
                "total_tokens",
                "duration_secs",
                "aggregate_tps",
                "per_stream_tps",
                "ttft_avg_ms",
                "itl_p50_ms",
                "itl_p99_ms"
            ]
        );
        assert_eq!(v["stream_count"], 8);
        assert_eq!(v["stream_count_source"], "concurrency_sweet_spot");
        assert_eq!(v["total_tokens"], 17_124);
        assert!((v["aggregate_tps"].as_f64().unwrap() - 285.4).abs() < 1e-9);
        assert!((v["per_stream_tps"].as_f64().unwrap() - 35.7).abs() < 1e-9);
    }
}
