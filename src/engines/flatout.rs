//! Engine F — Flat Out: sustained max-speed test.
//!
//! The goal is to find the absolute best-case speed the server can produce
//! and give the user a satisfying "big number" to end on.
//!
//! The test runs for up to 60 seconds total, divided into 6 segments of a
//! hard **10-second time window** each, with DECREASING `max_tokens` caps:
//!
//! | Segment | `max_tokens` cap | Purpose                        |
//! |---------|------------------|--------------------------------|
//! | 1       | 10,000           | Sustained throughput under load|
//! | 2       | 8,000            | Sustained throughput           |
//! | 3       | 6,000            | Sustained throughput           |
//! | 4       | 4,000            | Moderate load                  |
//! | 5       | 2,000            | Burst speed                    |
//! | 6       | 1,000            | Best-case single-response speed|
//!
//! **The segments are 10-second time windows, not token targets.** Each
//! segment lets the server run at full capacity for up to 10 seconds and
//! counts how many tokens actually arrive in that window:
//!
//! * If the server is fast enough to hit the `max_tokens` cap *before* 10s,
//!   the stream ends on its own and the segment finishes early.
//! * If the server is slower, the 10s window elapses and the stream is
//!   **aborted** (the worker task is dropped, closing the HTTP connection so
//!   the server stops generating). The segment records whatever tokens did
//!   arrive in those 10s.
//!
//! Either way, every segment is at most 10 seconds, so the whole test is at
//! most ~60 seconds. This is what made the old design wrong: it waited for
//! the *full* token target to complete, so a ~29 t/s server took 147s for
//! segment 1 alone (8+ minutes total).
//!
//! The segment's t/s is measured from the tokens we **observed on the wire**
//! (not the server's self-reported `usage.completion_tokens`, which some
//! servers — e.g. Unsloth / llama.cpp GGUF — inflate to the requested
//! `max_tokens` target, overstating the rate).
//!
//! Measurement-isolation note: all timing is captured in the worker
//! (quanta, `T0..Tn`); this module only takes deltas of those records and
//! counts the frames that arrive. Nothing here touches the TUI render path.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::client::{spawn_worker, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::Usage;
use crate::timing::MonotonicInstant;

/// The 6 per-segment `max_tokens` caps (decreasing): 10k, 8k, 6k, 4k, 2k,
/// 1k. A segment is cut off at the 10s window before this cap is reached on
/// a slow server; a fast server that hits the cap first ends early.
pub const SEGMENT_TARGETS: [u32; 6] = [10_000, 8_000, 6_000, 4_000, 2_000, 1_000];

/// Bounded worker→engine channel capacity.
const CHANNEL_CAPACITY: usize = 256;

/// The per-segment hard time window (seconds). A segment never runs longer
/// than this: when it elapses the stream is aborted and the next segment
/// starts. This is what keeps the whole test to ~60s.
pub const SEGMENT_WINDOW_SECS: f64 = 10.0;

/// The per-segment hard time window in nanoseconds, as a `u64` (the same
/// domain as [`MonotonicInstant::delta_nanos`]) for the exact window
/// comparison. Kept in sync with [`SEGMENT_WINDOW_SECS`].
const SEGMENT_WINDOW_NS: u64 = 10 * 1_000_000_000;

/// Total nominal duration (6 × 10s).
pub const TOTAL_DURATION_SECS: f64 = 60.0;

/// Results from one segment of the Flat Out test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentResult {
    /// 0-based segment index (0–5).
    pub index: usize,
    /// The `max_tokens` cap requested for this segment (the decreasing
    /// 10k→1k ladder). Not a hard target — the segment is cut at the 10s
    /// window if the server is slower than the cap.
    pub target_tokens: u32,
    /// The token frames **observed on the wire** during the segment (what the
    /// server actually produced in the window).
    pub actual_tokens: u64,
    /// How long the segment took (seconds; at most [`SEGMENT_WINDOW_SECS`]).
    pub duration_secs: f64,
    /// Tokens/sec for this segment (`actual_tokens / duration_secs`),
    /// measured from the observed frames.
    pub tps: f64,
    /// Time to first token (milliseconds).
    pub ttft_ms: f64,
}

impl SegmentResult {
    /// Build a segment result from the events observed during the window.
    ///
    /// The token count is the number of token-bearing frames that arrived
    /// (a [`Chunk::Reasoning`] or [`Chunk::Content`] delta); a
    /// [`Chunk::Usage`] / [`Chunk::Control`] frame is not a token. The t/s is
    /// that count over the segment's wall time. This is the honest
    /// wire-measured rate, independent of the server's self-reported usage.
    #[must_use]
    pub fn from_events(
        index: usize,
        target_tokens: u32,
        events: &[StreamEvent],
        duration_secs: f64,
    ) -> Self {
        let mut token_frames = 0u64;
        let mut ttft_ms = 0.0f64;
        for event in events {
            if let StreamEvent::Frame {
                frame, timestamps, ..
            } = event
            {
                if frame.chunk.is_token() {
                    token_frames += 1;
                    if ttft_ms == 0.0 {
                        ttft_ms = timestamps
                            .ttft_nanos()
                            .map(|ns| ns as f64 / 1e6)
                            .unwrap_or(0.0);
                    }
                }
            }
        }
        let tps = if duration_secs > 0.0 {
            token_frames as f64 / duration_secs
        } else {
            0.0
        };
        Self {
            index,
            target_tokens,
            actual_tokens: token_frames,
            duration_secs,
            tps,
            ttft_ms,
        }
    }
}

/// The complete Flat Out result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOutResult {
    /// The per-segment results (6 entries).
    pub segments: Vec<SegmentResult>,
    /// The highest t/s across all segments.
    pub best_tps: f64,
    /// Which segment (0-based) produced the best t/s.
    pub best_segment: usize,
    /// Sum of all tokens observed across the segments.
    pub total_tokens: u64,
    /// Actual total time (seconds; at most ~[`TOTAL_DURATION_SECS`]).
    pub total_duration: f64,
    /// Average t/s (`total_tokens / total_duration`).
    pub avg_tps: f64,
}

impl FlatOutResult {
    /// The one-line summary for the sequence header.
    #[must_use]
    pub fn summary_line(&self) -> String {
        format!(
            "BEST: {:.1} t/s (segment {})",
            self.best_tps,
            self.best_segment + 1
        )
    }

    /// The `--json` object for this result.
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        serde_json::json!({
            "segments": self.segments.iter().map(|s| serde_json::json!({
                "index": s.index,
                "target_tokens": s.target_tokens,
                "actual_tokens": s.actual_tokens,
                "duration_secs": s.duration_secs,
                "tps": s.tps,
                "ttft_ms": s.ttft_ms,
            })).collect::<Vec<_>>(),
            "best_tps": self.best_tps,
            "best_segment": self.best_segment,
            "total_tokens": self.total_tokens,
            "total_duration": self.total_duration,
            "avg_tps": self.avg_tps,
        })
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

/// The Flat Out engine: 6 sequential single-stream 10-second time windows
/// with decreasing `max_tokens` caps.
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

    /// Run all 6 segments sequentially, each capped at a 10-second window.
    pub async fn run(&self) -> FlatOutResult {
        let prompt = self.generate_prompt();
        let mut segments = Vec::with_capacity(6);
        let total_start = MonotonicInstant::now();

        for (i, &target) in SEGMENT_TARGETS.iter().enumerate() {
            // The `Space`-key pause: hold before the next segment goes out.
            if let Some(gate) = &self.pause {
                gate.wait_while_paused().await;
            }

            // Publish progress at segment start.
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::FlatOut {
                    segment: i + 1,
                    total_segments: 6,
                    target_tokens: target,
                    tps: 0.0,
                });
            }

            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!(
                        "Flat Out segment {}/6: 10s window, max_tokens cap {}",
                        i + 1,
                        target
                    ),
                );
            }

            // Run one 10-second time-window segment.
            let seg = self.run_segment(i, &prompt, target).await;

            // Publish progress at segment completion.
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::FlatOut {
                    segment: i + 1,
                    total_segments: 6,
                    target_tokens: target,
                    tps: seg.tps,
                });
            }

            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!(
                        "Flat Out segment {}/6: {} tok in {:.1}s ({:.1} t/s, TTFT {:.0} ms)",
                        i + 1,
                        seg.actual_tokens,
                        seg.duration_secs,
                        seg.tps,
                        seg.ttft_ms
                    ),
                );
            }

            segments.push(seg);
        }

        let total_duration = total_start.elapsed().as_secs_f64();
        let total_tokens: u64 = segments.iter().map(|s| s.actual_tokens).sum();
        let best_idx = segments
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.tps.total_cmp(&b.1.tps))
            .map(|(i, _)| i)
            .unwrap_or(0);
        let best_tps = segments[best_idx].tps;
        let avg_tps = if total_duration > 0.0 {
            total_tokens as f64 / total_duration
        } else {
            0.0
        };

        FlatOutResult {
            best_tps,
            best_segment: best_idx,
            total_tokens,
            total_duration,
            avg_tps,
            segments,
        }
    }

    /// The prompt for this run (generated once and reused across segments).
    ///
    /// Flat Out always uses a short prompt to minimize prefill time and
    /// maximize decode speed. The per-segment `max_tokens` cap is controlled
    /// by [`SEGMENT_TARGETS`]; the 10s window is the real limit.
    pub fn generate_prompt(&self) -> GeneratedPrompt {
        self.generator.short()
    }

    /// Run one segment within a hard [`SEGMENT_WINDOW_SECS`] time window.
    ///
    /// Spawns a single stream with the segment's `max_tokens` cap, drains its
    /// channel counting the token frames that arrive, and stops the segment
    /// as soon as either (a) the stream ends on its own (the server hit the
    /// cap before 10s) or (b) the 10s window elapses (the stream is aborted,
    /// closing the HTTP connection so the server stops generating). The
    /// returned [`SegmentResult`] counts what was actually observed.
    async fn run_segment(
        &self,
        i: usize,
        prompt: &GeneratedPrompt,
        target_tokens: u32,
    ) -> SegmentResult {
        let seg_start = MonotonicInstant::now();

        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            target_tokens,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)))
        .tag(format!("F:{}", target_tokens));
        if let Some(key) = &self.cfg.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }

        // Spawn the worker without awaiting it: the channel must be drained
        // concurrently (the same pattern as Engine A's server-agnostic fix).
        let handle = spawn_worker(worker, tx);

        let mut events = Vec::new();
        let mut batch = 0u32;

        // Drain until the 10s window elapses (then abort) or the stream ends
        // on its own. `tokio::time::timeout` bounds each `recv` to the time
        // remaining in the window, so a slow server is cut at 10s.
        loop {
            let elapsed_ns = seg_start.delta_nanos(&MonotonicInstant::now());
            if elapsed_ns >= SEGMENT_WINDOW_NS {
                break;
            }
            let remaining = Duration::from_nanos(SEGMENT_WINDOW_NS - elapsed_ns);
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    let is_terminal = matches!(
                        event,
                        StreamEvent::Complete { .. } | StreamEvent::Failed { .. }
                    );
                    events.push(event);
                    batch += 1;
                    if let Some(state) = &self.metrics {
                        if batch >= 8 || is_terminal {
                            state.update(self.live_snapshot(&events, &seg_start, target_tokens));
                            batch = 0;
                        }
                    }
                    if is_terminal {
                        break;
                    }
                }
                Ok(None) => break, // channel closed (worker done)
                Err(_) => break,   // 10s window elapsed
            }
        }

        // Stop the stream: abort the worker task (drops the HTTP response →
        // the connection closes → the server stops generating). A no-op if
        // the stream already ended on its own before the window.
        handle.abort();
        drop(rx);

        let elapsed = seg_start.elapsed().as_secs_f64();
        let seg = SegmentResult::from_events(i, target_tokens, &events, elapsed);

        // Final publish (the terminal event may have been batched out).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&events, &seg_start, target_tokens));
        }

        seg
    }

    /// Build a live [`MetricsSnapshot`] from the events collected so far in
    /// the current segment.
    fn live_snapshot(
        &self,
        events: &[StreamEvent],
        start: &MonotonicInstant,
        max_tokens: u32,
    ) -> MetricsSnapshot {
        let elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let mut token_frames = 0u64;
        let mut usage: Option<Usage> = None;
        let mut t3: Option<MonotonicInstant> = None;
        let mut t_end: Option<MonotonicInstant> = None;

        for event in events {
            match event {
                StreamEvent::Frame {
                    frame,
                    at,
                    timestamps,
                } => {
                    if frame.chunk.is_token() {
                        token_frames += 1;
                        if t3.is_none() {
                            t3 = Some(*at);
                        }
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

        // Measure the rate from the frames we OBSERVED on the wire, not the
        // server's self-reported `usage.completion_tokens` (some servers
        // inflate it to the `max_tokens` target). Fall back to the server
        // figure only when no token frames arrived (a usage-only stream).
        let tokens = if token_frames > 0 {
            token_frames
        } else {
            usage.map(|u| u.completion_tokens).unwrap_or(0)
        };
        let gen_tps = match (t3, t_end) {
            (Some(t3), Some(end)) => {
                let span_s = t3.delta_nanos(&end) as f64 / 1_000_000_000.0;
                if span_s > 0.0 && tokens > 0 {
                    Some(tokens as f64 / span_s)
                } else {
                    None
                }
            }
            _ => None,
        };

        MetricsSnapshot {
            endpoint: self.cfg.url.clone(),
            // The backend is unknown (vLLM, llama.cpp, LM Studio, …) —
            // never assume one.
            backend: String::new(),
            model: self.cfg.model.clone(),
            mode: "FlatOut".to_string(),
            aggregate_tps: tokens as f64 / (elapsed_ns as f64 / 1_000_000_000.0),
            active_streams: if t3.is_some() && t_end.is_none() {
                1
            } else {
                0
            },
            total_streams: 1,
            completion_tokens: tokens,
            prompt_tokens: usage.map(|u| u.prompt_tokens).unwrap_or(0),
            status: if t_end.is_some() {
                crate::metrics::state::StreamStatus::Done
            } else {
                crate::metrics::state::StreamStatus::Streaming
            },
            streams: vec![crate::metrics::state::StreamMetric {
                id: 0,
                kind: "Content".to_string(),
                state: if t_end.is_some() {
                    crate::metrics::state::StreamStatus::Done
                } else {
                    crate::metrics::state::StreamStatus::Streaming
                },
                tg_tokens: (tokens > 0).then_some(tokens),
                gen_tps,
                progress: (tokens as f64 / max_tokens.max(1) as f64).min(1.0),
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

    #[test]
    fn segment_targets_are_decreasing() {
        for w in SEGMENT_TARGETS.windows(2) {
            assert!(w[0] > w[1], "targets must be decreasing: {w:?}");
        }
    }

    #[test]
    fn segment_targets_sum_to_31000() {
        let sum: u32 = SEGMENT_TARGETS.iter().sum();
        assert_eq!(sum, 31_000);
    }

    #[test]
    fn segment_from_events_counts_observed_frames_not_server_usage() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t0),
            t_end: Some(t0),
        };
        // Three content frames, plus a usage frame claiming 10,000 tokens
        // (the misreporting-server case). Only the 3 token frames count.
        let events = vec![
            content_frame(t0, ts),
            content_frame(t0, ts),
            content_frame(t0, ts),
            StreamEvent::Frame {
                frame: ParsedFrame {
                    chunk: Chunk::Usage(Usage {
                        prompt_tokens: 5,
                        completion_tokens: 10_000,
                    }),
                    t_nanos: 0,
                    done: false,
                },
                at: t0,
                timestamps: ts,
            },
        ];
        let seg = SegmentResult::from_events(0, 10_000, &events, 10.0);
        assert_eq!(seg.index, 0);
        assert_eq!(seg.target_tokens, 10_000);
        assert_eq!(seg.actual_tokens, 3, "observed frames, not the 10k claim");
        assert!((seg.tps - 0.3).abs() < 1e-9, "tps: {}", seg.tps); // 3 / 10s
        assert!((seg.duration_secs - 10.0).abs() < 1e-9);
    }

    #[test]
    fn segment_from_events_zero_duration_gives_zero_tps() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t0),
            t_end: Some(t0),
        };
        let events = vec![content_frame(t0, ts)];
        let seg = SegmentResult::from_events(0, 1_000, &events, 0.0);
        assert_eq!(seg.actual_tokens, 1);
        assert_eq!(seg.tps, 0.0);
    }

    #[test]
    fn segment_from_events_ignores_control_and_counts_reasoning() {
        let t0 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t0),
            t_end: Some(t0),
        };
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
        let seg = SegmentResult::from_events(2, 6_000, &events, 5.0);
        // One reasoning token frame; the `[DONE]` control frame is not a
        // token.
        assert_eq!(seg.actual_tokens, 1);
        assert!((seg.tps - 0.2).abs() < 1e-9);
    }

    #[test]
    fn flat_out_result_summary_line() {
        let result = FlatOutResult {
            segments: vec![
                SegmentResult {
                    index: 0,
                    target_tokens: 10_000,
                    actual_tokens: 289,
                    duration_secs: 10.0,
                    tps: 28.9,
                    ttft_ms: 50.0,
                },
                SegmentResult {
                    index: 5,
                    target_tokens: 1_000,
                    actual_tokens: 300,
                    duration_secs: 10.0,
                    tps: 30.0,
                    ttft_ms: 10.0,
                },
            ],
            best_tps: 30.0,
            best_segment: 1,
            total_tokens: 589,
            total_duration: 20.0,
            avg_tps: 29.45,
        };
        let line = result.summary_line();
        assert!(line.contains("BEST: 30.0 t/s"), "{line}");
        assert!(line.contains("segment 2"), "{line}");
    }

    #[test]
    fn flat_out_result_to_dict() {
        let result = FlatOutResult {
            segments: vec![SegmentResult {
                index: 0,
                target_tokens: 10_000,
                actual_tokens: 289,
                duration_secs: 10.0,
                tps: 28.9,
                ttft_ms: 85.0,
            }],
            best_tps: 28.9,
            best_segment: 0,
            total_tokens: 289,
            total_duration: 10.0,
            avg_tps: 28.9,
        };
        let v = result.to_dict();
        assert_eq!(v["best_tps"], 28.9);
        assert_eq!(v["best_segment"], 0);
        assert_eq!(v["total_tokens"], 289);
        assert_eq!(v["segments"][0]["target_tokens"], 10_000);
        assert_eq!(v["segments"][0]["actual_tokens"], 289);
    }

    #[test]
    fn total_duration_is_60_seconds() {
        assert!((TOTAL_DURATION_SECS - 60.0).abs() < 1e-9);
    }

    #[test]
    fn segment_window_is_10_seconds() {
        assert!((SEGMENT_WINDOW_SECS - 10.0).abs() < 1e-9);
    }

    #[test]
    fn six_segments() {
        assert_eq!(SEGMENT_TARGETS.len(), 6);
    }
}
