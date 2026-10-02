//! Engine F — Flat Out: sustained maximum decode speed.
//!
//! The goal is to measure the server's **true maximum sustained decode
//! speed** with zero prefill interference. One minimal (~15-token)
//! open-ended prompt, ONE continuous stream, let it run for exactly
//! **60 seconds**, count every token frame observed on the wire, and
//! divide: `tps = total_tokens / 60.0`. That single number is the
//! server's peak sustained tokens-per-second.
//!
//! Best practice for measuring peak throughput:
//!
//! * **one stream** — no concurrency cross-talk, the classic single-user
//!   decode path;
//! * **minimal prompt** (~15 tokens) — prefill takes <100 ms, negligible
//!   against the 60-second decode window;
//! * **`max_tokens = 100,000` + `ignore_eos`** — effectively unlimited,
//!   so the *only* stop condition is the 60-second window (aborted by
//!   dropping the worker task, which closes the HTTP connection so the
//!   server stops generating). `ignore_eos` keeps llama.cpp servers from
//!   letting the model end the stream on its own end-token (a model that
//!   "finishes" the 15-token story in 300 tokens would otherwise end the
//!   60-second test in 5 seconds); other backends ignore the field;
//! * **count what arrives** — tokens are the frame count observed on the
//!   wire (not the server's self-reported `usage.completion_tokens`,
//!   which some servers — e.g. Unsloth / llama.cpp GGUF — inflate to the
//!   requested `max_tokens` target, overstating the rate).
//!
//! The result also records TTFT and the ITL p50/p99 of the run, so the
//! one headline number is backed by latency evidence.
//!
//! Measurement-isolation note: all timing is captured in the worker
//! (quanta, `T0..Tn`); this module only takes deltas of those records and
//! counts the frames that arrive. Nothing here touches the TUI render path.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::client::{spawn_worker, stream_text, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::histogram::LatencyHistogram;
use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamStatus};
use crate::prompt::tokenizer::count_tokens;
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::{Chunk, Usage};
use crate::timing::MonotonicInstant;

/// The continuous run window (seconds). The stream is aborted when it
/// elapses — the only real stop condition of the test.
pub const WINDOW_SECS: f64 = 60.0;

/// The run window in nanoseconds (the same domain as
/// [`MonotonicInstant::delta_nanos`]) for the exact window comparison.
/// Kept in sync with [`WINDOW_SECS`].
const WINDOW_NS: u64 = 60 * 1_000_000_000;

/// The `max_tokens` requested for the single stream: 100,000 —
/// effectively unlimited. The 60-second window, not this cap, ends the
/// test.
pub const MAX_TOKENS: u32 = 100_000;

/// Bounded worker→engine channel capacity.
const CHANNEL_CAPACITY: usize = 256;

/// The minimal open-ended prompt (~15 tokens).
///
/// Short enough that prefill takes <100 ms (negligible against the 60s
/// window), open-ended enough that the model keeps generating without
/// stopping, and not a question (questions get short answers).
pub const MINIMAL_PROMPT: &str =
    "Continue this story: The old lighthouse keeper walked down the spiral stairs and";

/// The complete Flat Out result: one continuous 60-second stream.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOutResult {
    /// All token frames **observed on the wire** during the window (what
    /// the server actually produced).
    pub total_tokens: u64,
    /// How long the run took (seconds; ~[`WINDOW_SECS`], or less when the
    /// stream ended on its own).
    pub duration_secs: f64,
    /// The headline number: `total_tokens / duration_secs` — the server's
    /// sustained maximum decode speed.
    pub tps: f64,
    /// Time to first token (milliseconds).
    pub ttft_ms: f64,
    /// Inter-token latency p50 for the run (milliseconds).
    pub itl_p50_ms: f64,
    /// Inter-token latency p99 for the run (milliseconds).
    pub itl_p99_ms: f64,
}

impl FlatOutResult {
    /// Build the result from the events observed during the window.
    ///
    /// The token count follows the research's **Authoritative Counting
    /// Rule**:
    ///
    /// 1. **PRIMARY** — the server's `usage.completion_tokens`, present only
    ///    when the stream ended *naturally* before the 60 s window (Engine F
    ///    normally aborts it, so this is the rare path);
    /// 2. **FALLBACK** — re-tokenize the full reasoning + content text with
    ///    the model's exact tokenizer (the normal path for an aborted
    ///    stream: counting SSE chunks is fundamentally inaccurate when
    ///    speculative batching packs multiple tokens into one frame);
    /// 3. **LAST RESORT** (no usage, no tokenizer) — the observed token
    ///    frame count.
    ///
    /// The t/s is that count over the **decode window** — the first
    /// content/reasoning token to the last (`T_first` → `T_last`), which
    /// excludes prefill/TTFT and the trailing gap to abort — the research's
    /// **Timing Boundary Rule** and exactly how each backend measures its own
    /// decode rate. The ITL percentiles are the gaps between consecutive
    /// token frames.
    #[must_use]
    pub fn from_events(
        events: &[StreamEvent],
        duration_secs: f64,
        tokenizer: Option<&Tokenizer>,
    ) -> Self {
        let mut deltas_ns: Vec<u64> = Vec::new();
        let mut token_frames = 0u64;
        let mut ttft_ms = 0.0f64;
        let mut first_token_at: Option<MonotonicInstant> = None;
        let mut last_token_at: Option<MonotonicInstant> = None;
        let mut usage: Option<Usage> = None;
        for event in events {
            match event {
                StreamEvent::Frame {
                    frame,
                    at,
                    timestamps,
                } => {
                    // The usage chunk carries the server's authoritative
                    // count (the frame where it physically appears in the
                    // stream); a `Complete` event mirrors it too.
                    if let Chunk::Usage(u) = &frame.chunk {
                        usage = Some(*u);
                    }
                    if frame.chunk.is_token() {
                        if let Some(prev) = last_token_at {
                            deltas_ns.push(prev.delta_nanos(at));
                        }
                        token_frames += 1;
                        if ttft_ms == 0.0 {
                            ttft_ms = timestamps
                                .ttft_nanos()
                                .map(|ns| ns as f64 / 1e6)
                                .unwrap_or(0.0);
                        }
                        if first_token_at.is_none() {
                            first_token_at = Some(*at);
                        }
                        last_token_at = Some(*at);
                    }
                }
                StreamEvent::Complete { usage: u, .. } => {
                    usage = *u;
                }
                StreamEvent::Failed { .. } => {}
            }
        }
        let (itl_p50_ms, itl_p99_ms) = itl_percentiles_ms(&deltas_ns);

        // Authoritative count (usage → re-tokenize → observed frames).
        let total_tokens = match usage.map(|u| u.completion_tokens).filter(|&n| n > 0) {
            Some(n) => n,
            None => match tokenizer
                .and_then(|t| t.try_count(&stream_text(events)))
                .filter(|&n| n > 0)
            {
                Some(n) => n as u64,
                None => token_frames,
            },
        };

        // Decode window: first → last content/reasoning token.
        let tps = match (first_token_at, last_token_at) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 {
                    total_tokens as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        Self {
            total_tokens,
            duration_secs,
            tps,
            ttft_ms,
            itl_p50_ms,
            itl_p99_ms,
        }
    }

    /// The one-line summary for the sequence header / headless report.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "{:.1} t/s sustained · {} tok in {:.0}s · TTFT {:.0} ms",
            self.tps, self.total_tokens, self.duration_secs, self.ttft_ms
        )
    }

    /// The `--json` object for this result (a single object — there are
    /// no segments).
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "total_tokens": self.total_tokens,
            "duration_secs": self.duration_secs,
            "tps": self.tps,
            "ttft_ms": self.ttft_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p99_ms": self.itl_p99_ms,
        })
    }
}

/// `(p50 ms, p99 ms)` of inter-token deltas recorded into a
/// [`LatencyHistogram`] (the same histogram Engine A uses for its ITL
/// distribution). `(0.0, 0.0)` for an empty list.
fn itl_percentiles_ms(deltas_ns: &[u64]) -> (f64, f64) {
    let mut h = LatencyHistogram::default();
    for d in deltas_ns {
        h.record(*d);
    }
    (h.p50() / 1e6, h.p99() / 1e6)
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

/// The Flat Out engine: one continuous [`WINDOW_SECS`]-second stream with
/// a minimal prompt and an effectively-unlimited `max_tokens` cap.
#[derive(Debug)]
pub struct FlatOutEngine {
    cfg: Config,
    client: reqwest::Client,
    generator: PromptGenerator,
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
    /// Build the engine from the resolved config.
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
            metrics: None,
            progress: None,
            pause: None,
            logger: None,
        })
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

    /// Run the single 60-second stream and return the result.
    pub async fn run(&self) -> FlatOutResult {
        let prompt = self.generate_prompt();

        // The `Space`-key pause: hold before the stream goes out.
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }

        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: 0.0,
                tokens: 0,
                tps: 0.0,
            });
        }

        if let Some(l) = &self.logger {
            l.info(
                Context::EngineF,
                format!("Flat Out: one {WINDOW_SECS:.0}s stream, max_tokens {MAX_TOKENS}"),
            );
        }

        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            MAX_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)))
        // The 60s window — not the model's own end-token — is the only
        // stop: llama.cpp servers honor `ignore_eos`, other backends
        // ignore the field (graceful degradation).
        .ignore_eos(true)
        .tag("F");
        if let Some(key) = &self.cfg.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }

        // Spawn the worker without awaiting it: the channel must be drained
        // concurrently (the same pattern as Engine A's server-agnostic fix).
        let handle = spawn_worker(worker, tx);
        let start = MonotonicInstant::now();

        let mut events = Vec::new();
        let mut batch = 0u32;

        // Drain until the 60s window elapses (then abort) or the stream
        // ends on its own. `tokio::time::timeout` bounds each `recv` to the
        // time remaining in the window, so a slow server is cut at 60s.
        loop {
            let elapsed_ns = start.delta_nanos(&MonotonicInstant::now());
            if elapsed_ns >= WINDOW_NS {
                break;
            }
            let remaining = Duration::from_nanos(WINDOW_NS - elapsed_ns);
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    let is_terminal = matches!(
                        event,
                        StreamEvent::Complete { .. } | StreamEvent::Failed { .. }
                    );
                    events.push(event);
                    batch += 1;
                    if batch >= 8 || is_terminal {
                        // Live TUI snapshot + sequence progress bus (the
                        // 10 Hz ticker mirrors the bus into the state slot).
                        if let Some(state) = &self.metrics {
                            state.update(self.live_snapshot(&events, &start));
                        }
                        if let Some(bus) = &self.progress {
                            let (tokens, tps) = Self::observed(&events);
                            bus.publish(EngineProgress::FlatOut {
                                elapsed_secs: start.elapsed().as_secs_f64(),
                                tokens,
                                tps,
                            });
                        }
                        batch = 0;
                    }
                    if is_terminal {
                        break;
                    }
                }
                Ok(None) => break, // channel closed (worker done)
                Err(_) => break,   // 60s window elapsed
            }
        }

        // Stop the stream: abort the worker task (drops the HTTP response →
        // the connection closes → the server stops generating). A no-op if
        // the stream already ended on its own before the window.
        handle.abort();
        drop(rx);

        let elapsed = start.elapsed().as_secs_f64();
        let result =
            FlatOutResult::from_events(&events, elapsed, self.generator.tokenizer().as_deref());

        // Final publish (the last batch may not have flushed).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&events, &start));
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: elapsed,
                tokens: result.total_tokens,
                tps: result.tps,
            });
        }
        if let Some(l) = &self.logger {
            l.info(
                Context::EngineF,
                format!(
                    "Flat Out: {} tok in {:.1}s ({:.1} t/s, TTFT {:.0} ms, ITL p50 {:.1} / p99 {:.1} ms)",
                    result.total_tokens,
                    result.duration_secs,
                    result.tps,
                    result.ttft_ms,
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

    /// The token frames observed so far + their decode rate over the
    /// first→last token window (the live progress figures; prefill/TTFT
    /// excluded, the research's Timing Boundary Rule).
    fn observed(events: &[StreamEvent]) -> (u64, f64) {
        let mut first: Option<MonotonicInstant> = None;
        let mut last: Option<MonotonicInstant> = None;
        let mut tokens = 0u64;
        for e in events {
            if let StreamEvent::Frame { frame, at, .. } = e {
                if frame.chunk.is_token() {
                    if first.is_none() {
                        first = Some(*at);
                    }
                    last = Some(*at);
                    tokens += 1;
                }
            }
        }
        let tps = match (first, last) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 {
                    tokens as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        (tokens, tps)
    }

    /// Build a live [`MetricsSnapshot`] from the events collected so far
    /// in the run.
    ///
    /// The decode rate uses the same single formula as Engine A and the
    /// Overall Metrics panel: `tokens_received / (T_last − T_first)`.
    fn live_snapshot(&self, events: &[StreamEvent], start: &MonotonicInstant) -> MetricsSnapshot {
        let elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let (frame_tokens, decode_rate) = Self::observed(events);
        let mut itl = LatencyHistogram::default();
        let mut last_token_at: Option<MonotonicInstant> = None;
        let mut usage: Option<Usage> = None;
        let mut t_end: Option<MonotonicInstant> = None;

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
                        last_token_at = Some(*at);
                    }
                    if t_end.is_none() {
                        t_end = timestamps.t_end;
                    }
                }
                StreamEvent::Complete {
                    usage: u,
                    timestamps,
                    ..
                } => {
                    usage = *u;
                    t_end = timestamps.t_end;
                }
                StreamEvent::Failed { timestamps, .. } => {
                    t_end = timestamps.t_end;
                }
            }
        }

        // The token count: the server's `usage.completion_tokens` when the
        // stream ended naturally; while the 60 s stream is still open (the
        // normal case — we abort it), the observed token frames.
        let tokens_received = usage
            .map(|u| u.completion_tokens)
            .filter(|&n| n > 0)
            .unwrap_or(frame_tokens);
        let status = if t_end.is_some() {
            StreamStatus::Done
        } else {
            StreamStatus::Streaming
        };

        MetricsSnapshot {
            endpoint: self.cfg.url.clone(),
            backend: String::new(),
            model: self.cfg.model.clone(),
            mode: "FlatOut".to_string(),
            aggregate_tps: decode_rate,
            active_streams: (t_end.is_none()) as usize,
            total_streams: 1,
            itl_p50_ns: itl.p50() as u64,
            itl_p90_ns: itl.p90() as u64,
            itl_p99_ns: itl.p99() as u64,
            itl_p999_ns: itl.p999() as u64,
            completion_tokens: tokens_received,
            // The wire-truth numerator: the token frames actually observed
            // on the wire (never the server's self-reported `usage`).
            observed_frames: frame_tokens,
            prompt_tokens: usage.map(|u| u.prompt_tokens).unwrap_or(0),
            status,
            streams: vec![crate::metrics::state::StreamMetric {
                id: 0,
                kind: "Content".to_string(),
                state: status,
                tg_tokens: (tokens_received > 0).then_some(tokens_received),
                gen_tps: (decode_rate > 0.0).then_some(decode_rate),
                progress: (elapsed_ns as f64 / 1e9 / WINDOW_SECS).min(1.0),
                ..Default::default()
            }],
            ..Default::default()
        }
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

    fn ts_at(t: MonotonicInstant) -> StreamTimestamps {
        StreamTimestamps {
            t0: Some(t),
            t1: Some(t),
            t2: Some(t),
            t3: Some(t),
            t_end: Some(t),
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
    fn from_events_prefers_server_usage_over_the_token_window() {
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
            // The server's usage is the authoritative count (primary method);
            // the rate is over the first→last token window (100→300 ms).
            let r = FlatOutResult::from_events(&events, 10.0, None);
            assert_eq!(r.total_tokens, 100_000, "server usage is authoritative");
            assert!((r.tps - 100_000.0 / 0.2).abs() < 1.0, "tps: {}", r.tps);
            assert!((r.duration_secs - 10.0).abs() < 1e-9);
        });
    }

    #[test]
    fn from_events_no_usage_falls_back_to_frames() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // first token
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // last token
            let ts = ts_at(t0);
            // No usage frame (the aborted-stream case), no tokenizer: the
            // observed token frames are the last-resort count.
            let events = vec![content_frame(t1, ts), content_frame(t2, ts)];
            let r = FlatOutResult::from_events(&events, 10.0, None);
            assert_eq!(r.total_tokens, 2, "no usage + no tokenizer → frames");
            assert!((r.tps - 2.0 / 0.1).abs() < 1e-9, "tps: {}", r.tps); // 2 / 100 ms
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
            // word-level tokenizer counts "apple apple" as exactly 2 tokens.
            let mut e1 = content_frame(t1, ts);
            if let StreamEvent::Frame { frame, .. } = &mut e1 {
                frame.chunk = Chunk::Content("apple ".into());
            }
            let mut e2 = content_frame(t2, ts);
            if let StreamEvent::Frame { frame, .. } = &mut e2 {
                frame.chunk = Chunk::Content("apple".into());
            }
            let events = vec![e1, e2];
            let r = FlatOutResult::from_events(&events, 10.0, Some(&tok));
            assert_eq!(r.total_tokens, 2, "re-tokenized count (apple apple)");
            assert!((r.tps - 2.0 / 0.05).abs() < 1e-9, "tps: {}", r.tps); // 2 / 50 ms
        });
    }

    #[test]
    fn from_events_zero_duration_gives_zero_tps() {
        let t0 = MonotonicInstant::now();
        let events = vec![content_frame(t0, ts_at(t0))];
        let r = FlatOutResult::from_events(&events, 0.0, None);
        assert_eq!(r.total_tokens, 1);
        assert_eq!(r.tps, 0.0); // a single frame → a zero decode window
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
        let r = FlatOutResult::from_events(&events, 5.0, None);
        // One reasoning token frame; the `[DONE]` control frame is not a
        // token. A single frame gives a zero decode window → 0 t/s.
        assert_eq!(r.total_tokens, 1);
        assert_eq!(r.tps, 0.0);
    }

    #[test]
    fn from_events_records_ttft_from_the_first_token() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t0),
            t_end: None,
        };
        let events = vec![content_frame(t0, ts)];
        let r = FlatOutResult::from_events(&events, 60.0, None);
        // All milestones equal → TTFT is 0 (recorded, not missing).
        assert_eq!(r.ttft_ms, 0.0);
    }

    #[test]
    fn itl_percentiles_track_the_gap_distribution() {
        // 1 ms and 2 ms gaps: p50 is the first value covering half the
        // count (the 1 ms sample), p99 the 2 ms sample (hdrhistogram
        // cumulative-count semantics; buckets carry a tiny rounding
        // slack).
        let (p50, p99) = itl_percentiles_ms(&[1_000_000, 2_000_000]);
        assert!((p50 - 1.0).abs() < 0.01, "p50: {p50}");
        assert!((p99 - 2.0).abs() < 0.05, "p99: {p99}");
    }

    #[test]
    fn itl_percentiles_empty_is_zero() {
        assert_eq!(itl_percentiles_ms(&[]), (0.0, 0.0));
    }

    #[test]
    fn summary_line_is_the_single_number() {
        let r = FlatOutResult {
            total_tokens: 3462,
            duration_secs: 60.0,
            tps: 57.7,
            ttft_ms: 42.0,
            itl_p50_ms: 17.0,
            itl_p99_ms: 40.0,
        };
        let line = r.summary_line();
        assert!(line.starts_with("57.7 t/s sustained"), "{line}");
        assert!(line.contains("3462 tok in 60s"), "{line}");
    }

    #[test]
    fn to_dict_is_a_single_flat_object() {
        let r = FlatOutResult {
            total_tokens: 3462,
            duration_secs: 60.0,
            tps: 57.7,
            ttft_ms: 42.0,
            itl_p50_ms: 17.0,
            itl_p99_ms: 40.0,
        };
        let v = r.to_dict();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "total_tokens",
                "duration_secs",
                "tps",
                "ttft_ms",
                "itl_p50_ms",
                "itl_p99_ms"
            ]
        );
        assert_eq!(v["total_tokens"], 3462);
        assert!((v["tps"].as_f64().unwrap() - 57.7).abs() < 1e-9);
    }
}
