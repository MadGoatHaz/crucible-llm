//! Engine F — Flat Out: sustained max-speed test with decreasing token
//! targets.
//!
//! The goal is to find the absolute best-case speed the server can produce
//! and give the user a satisfying "big number" to end on.
//!
//! Runs for up to 60 seconds total, divided into 6 segments of 10 seconds
//! each, with DECREASING token targets:
//!
//! | Segment | Target | Purpose                        |
//! |---------|--------|--------------------------------|
//! | 1       | 10,000 | Sustained throughput under load|
//! | 2       | 8,000  | Sustained throughput           |
//! | 3       | 6,000  | Sustained throughput           |
//! | 4       | 4,000  | Moderate load                  |
//! | 5       | 2,000  | Burst speed                    |
//! | 6       | 1,000  | Best-case single-response speed|
//!
//! The FINAL number (segment 6, 1k tokens) is typically the HIGHEST t/s
//! because: shorter prompt = faster prefill, less KV cache pressure, server
//! is warm. This is the "big number" the user sees.
//!
//! Each segment:
//! - Spawns a single stream (concurrency=1) requesting the target token count
//! - If the stream completes before 10s, immediately start the next segment
//! - If the stream is still going at 10s, let it finish (don't kill it) and
//!   THEN start the next segment
//! - Records: actual tokens produced, time taken, t/s for that segment
//!
//! Measurement-isolation note: all timing is captured in the worker
//! (quanta, `T0..Tn`); this module only takes deltas of those records.
//! Nothing here touches the TUI or the timing path.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::mpsc;

use crate::client::{join_worker, spawn_worker, StreamEvent, StreamOutcome, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::Usage;
use crate::timing::MonotonicInstant;

/// The 6 segment targets (decreasing): 10k, 8k, 6k, 4k, 2k, 1k.
pub const SEGMENT_TARGETS: [u32; 6] = [10_000, 8_000, 6_000, 4_000, 2_000, 1_000];

/// Bounded worker→engine channel capacity.
const CHANNEL_CAPACITY: usize = 256;

/// The per-segment time window (seconds). The stream is not killed at this
/// boundary — it is allowed to finish — but the window defines the segment's
/// nominal duration for progress reporting.
pub const SEGMENT_WINDOW_SECS: f64 = 10.0;

/// Total nominal duration (6 × 10s).
pub const TOTAL_DURATION_SECS: f64 = 60.0;

/// Results from one segment of the Flat Out test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SegmentResult {
    /// 0-based segment index (0–5).
    pub index: usize,
    /// The target token count we asked for.
    pub target_tokens: u32,
    /// The actual tokens produced by the server.
    pub actual_tokens: u64,
    /// How long the segment took (seconds).
    pub duration_secs: f64,
    /// Tokens/sec for this segment (`actual_tokens / duration_secs`).
    pub tps: f64,
    /// Time to first token (seconds).
    pub ttft_ms: f64,
}

impl SegmentResult {
    /// Build from the stream outcome and timing.
    pub fn from_outcome(
        index: usize,
        target_tokens: u32,
        outcome: &StreamOutcome,
        duration_secs: f64,
    ) -> Self {
        let actual_tokens = outcome
            .usage
            .map(|u| u.completion_tokens)
            .unwrap_or(0);
        let ttft_ms = outcome
            .timestamps
            .ttft_nanos()
            .map(|ns| ns as f64 / 1e6)
            .unwrap_or(0.0);
        let tps = if duration_secs > 0.0 {
            actual_tokens as f64 / duration_secs
        } else {
            0.0
        };
        Self {
            index,
            target_tokens,
            actual_tokens,
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
    /// Sum of all tokens produced.
    pub total_tokens: u64,
    /// Actual total time (seconds).
    pub total_duration: f64,
    /// Average t/s (`total_tokens / total_duration`).
    pub avg_tps: f64,
}

impl FlatOutResult {
    /// The one-line summary for the sequence header.
    pub fn summary_line(&self) -> String {
        format!(
            "BEST: {:.1} t/s (segment {}, {}k tokens)",
            self.best_tps,
            self.best_segment + 1,
            self.segments
                .get(self.best_segment)
                .map(|s| s.target_tokens / 1000)
                .unwrap_or(0)
        )
    }

    /// The `--json` object for this result.
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

/// The Flat Out engine: 6 sequential single-stream segments with decreasing
/// token targets.
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

    /// Run all 6 segments sequentially.
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
                    format!("Flat Out segment {}/6: targeting {} tokens", i + 1, target),
                );
            }

            // Run one stream for this segment.
            let seg_start = MonotonicInstant::now();
            let outcome = self.run_segment(&prompt, target).await;
            let duration = seg_start.elapsed().as_secs_f64();

            let seg = SegmentResult::from_outcome(i, target, &outcome, duration);

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
                        "Flat Out segment {}/6 complete: {} tok in {:.1}s ({:.1} t/s, TTFT {:.0} ms)",
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
    pub fn generate_prompt(&self) -> GeneratedPrompt {
        // Flat Out always uses a short prompt to minimize prefill time and
        // maximize decode speed. The token target is controlled per-segment
        // via `max_tokens`.
        self.generator.short()
    }

    /// Run one segment: spawn a single stream with the target token count,
    /// drain its channel, and return the outcome.
    async fn run_segment(&self, prompt: &GeneratedPrompt, target_tokens: u32) -> StreamOutcome {
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

        let start = MonotonicInstant::now();
        // Spawn the worker without awaiting it: the channel must be
        // drained concurrently (same pattern as Engine A's fix).
        let handle = spawn_worker(worker, tx);
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            events.push(event);
            batch += 1;
            if let Some(state) = &self.metrics {
                let is_terminal = matches!(
                    events.last(),
                    Some(StreamEvent::Complete { .. }) | Some(StreamEvent::Failed { .. })
                );
                if batch >= 8 || is_terminal {
                    state.update(self.live_snapshot(&events, &start, target_tokens));
                    batch = 0;
                }
            }
        }
        // Final publish.
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&events, &start, target_tokens));
        }
        // Channel is closed (worker done): collect the outcome.
        let outcome = join_worker(handle).await;
        outcome
    }

    /// Build a live [`MetricsSnapshot`] from the events collected so far.
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
                    let is_tok = matches!(
                        &frame.chunk,
                        crate::sse::Chunk::Reasoning(_) | crate::sse::Chunk::Content(_)
                    );
                    if is_tok {
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

        let tokens = usage.map(|u| u.completion_tokens).unwrap_or(token_frames);
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
            backend: "vLLM".to_string(),
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
    use crate::timing::StreamTimestamps;

    /// A synthetic stream outcome for testing.
    fn mock_outcome(tokens: u64, _ttft_ns: u64, _total_ns: u64) -> StreamOutcome {
        let t0 = MonotonicInstant::now();
        // We can't easily set absolute timestamps, so we use a simple
        // synthetic outcome for unit tests of the result struct.
        StreamOutcome {
            timestamps: StreamTimestamps {
                t0: Some(t0),
                t_end: Some(t0),
                ..StreamTimestamps::default()
            },
            usage: Some(Usage {
                prompt_tokens: 100,
                completion_tokens: tokens,
            }),
            premature: false,
            malformed_frames: 0,
            error: None,
            looping: false,
            loop_excluded_tokens: 0,
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
    fn segment_result_from_outcome_computes_tps() {
        let outcome = mock_outcome(1000, 0, 0);
        let seg = SegmentResult::from_outcome(0, 10_000, &outcome, 5.0);
        assert_eq!(seg.index, 0);
        assert_eq!(seg.target_tokens, 10_000);
        assert_eq!(seg.actual_tokens, 1000);
        assert!((seg.tps - 200.0).abs() < 1e-9, "tps: {}", seg.tps);
        assert!((seg.duration_secs - 5.0).abs() < 1e-9);
    }

    #[test]
    fn segment_result_zero_duration_gives_zero_tps() {
        let outcome = mock_outcome(100, 0, 0);
        let seg = SegmentResult::from_outcome(0, 1000, &outcome, 0.0);
        assert_eq!(seg.tps, 0.0);
    }

    #[test]
    fn flat_out_result_summary_line() {
        let result = FlatOutResult {
            segments: vec![
                SegmentResult {
                    index: 0,
                    target_tokens: 10_000,
                    actual_tokens: 10_000,
                    duration_secs: 10.0,
                    tps: 1000.0,
                    ttft_ms: 50.0,
                },
                SegmentResult {
                    index: 5,
                    target_tokens: 1_000,
                    actual_tokens: 1_000,
                    duration_secs: 2.0,
                    tps: 500.0,
                    ttft_ms: 10.0,
                },
            ],
            best_tps: 1000.0,
            best_segment: 0,
            total_tokens: 11_000,
            total_duration: 12.0,
            avg_tps: 916.7,
        };
        let line = result.summary_line();
        assert!(line.contains("BEST: 1000.0 t/s"), "{line}");
        assert!(line.contains("segment 1"), "{line}");
        assert!(line.contains("10k tokens"), "{line}");
    }

    #[test]
    fn flat_out_result_to_dict() {
        let result = FlatOutResult {
            segments: vec![SegmentResult {
                index: 0,
                target_tokens: 10_000,
                actual_tokens: 9_500,
                duration_secs: 12.5,
                tps: 760.0,
                ttft_ms: 85.0,
            }],
            best_tps: 760.0,
            best_segment: 0,
            total_tokens: 9_500,
            total_duration: 12.5,
            avg_tps: 760.0,
        };
        let v = result.to_dict();
        assert_eq!(v["best_tps"], 760.0);
        assert_eq!(v["best_segment"], 0);
        assert_eq!(v["total_tokens"], 9_500);
        assert_eq!(v["segments"][0]["target_tokens"], 10_000);
        assert_eq!(v["segments"][0]["actual_tokens"], 9_500);
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
