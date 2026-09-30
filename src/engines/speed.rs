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

use crate::client::{StreamError, StreamEvent, StreamOutcome, StreamWorker};
use crate::config::{Config, Mode};
use crate::engines::sequence::{EngineProgress, ProgressBus};
use crate::metrics::histogram::LatencyHistogram;
use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamMetric, StreamStatus};
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
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
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)));
        if let Some(key) = &self.cfg.api_key {
            worker = worker.api_key(key);
        }

        let start = MonotonicInstant::now();
        let outcome = tokio::spawn(worker.run(tx))
            .await
            .expect("stream worker task panicked");
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
        (aggregate(&self.cfg, prompt, &outcome, &events), events)
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
        )
    }

    /// Build a live single-stream [`MetricsSnapshot`] from the events
    /// collected so far — the shared TUI seam for the one-stream engines
    /// (A, C1, C2, C3): the Live Monitor's gauges / ITL distribution /
    /// stream matrix / token counter all read it lock-free while the
    /// current engine runs (measurement-isolation invariant, blueprint §4).
    pub fn single_stream_snapshot(
        endpoint: &str,
        model: &str,
        mode: &str,
        events: &[StreamEvent],
        start: &MonotonicInstant,
        max_tokens: u32,
    ) -> MetricsSnapshot {
        let elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let mut itl = LatencyHistogram::default();
        let mut token_frames = 0u64;
        let mut reasoning_frames = 0u64;
        let mut content_frames = 0u64;
        let mut last_token_at: Option<MonotonicInstant> = None;
        let mut ttft_ns: Option<u64> = None;
        let mut usage: Option<Usage> = None;
        let mut state = StreamStatus::Waiting;
        let mut t3: Option<MonotonicInstant> = None;
        let mut t_end: Option<MonotonicInstant> = None;

        for event in events {
            match event {
                StreamEvent::Frame {
                    frame,
                    at,
                    timestamps,
                } => {
                    let is_tok = matches!(&frame.chunk, Chunk::Reasoning(_) | Chunk::Content(_));
                    if is_tok {
                        if let Some(prev) = last_token_at {
                            itl.record(prev.delta_nanos(at));
                        }
                        token_frames += 1;
                        match &frame.chunk {
                            Chunk::Reasoning(_) => reasoning_frames += 1,
                            _ => content_frames += 1,
                        }
                        last_token_at = Some(*at);
                        if t3.is_none() {
                            t3 = Some(*at);
                        }
                        if state == StreamStatus::Waiting {
                            state = StreamStatus::Streaming;
                        }
                    }
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
                StreamEvent::Complete {
                    timestamps,
                    usage: u,
                    ..
                } => {
                    usage = *u;
                    t_end = timestamps.t_end;
                    state = StreamStatus::Done;
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
                StreamEvent::Failed { timestamps, .. } => {
                    t_end = timestamps.t_end;
                    state = StreamStatus::Error;
                    if ttft_ns.is_none() {
                        ttft_ns = timestamps.ttft_nanos();
                    }
                }
            }
        }

        let tokens = usage.map(|u| u.completion_tokens).unwrap_or(token_frames);
        let prompt_tokens = usage.map(|u| u.prompt_tokens).unwrap_or(0);
        let ttft_s = ttft_ns.map(|ns| ns as f64 / 1_000_000_000.0);
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
        let mtp = if content_frames > 0 {
            Some(tokens as f64 / content_frames as f64)
        } else {
            None
        };
        let progress = if state == StreamStatus::Done {
            1.0
        } else {
            (tokens as f64 / max_tokens.max(1) as f64).min(1.0)
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
            backend: "vLLM".to_string(),
            model: model.to_string(),
            mode: mode.to_string(),
            aggregate_tps: tokens as f64 / (elapsed_ns as f64 / 1_000_000_000.0),
            active_streams: if state == StreamStatus::Streaming {
                1
            } else {
                0
            },
            total_streams: 1,
            itl_p50_ns: itl.p50() as u64,
            itl_p90_ns: itl.p90() as u64,
            itl_p99_ns: itl.p99() as u64,
            itl_p999_ns: itl.p999() as u64,
            prompt_tokens,
            completion_tokens: tokens,
            status: state,
            streams: vec![StreamMetric {
                id: 0,
                kind: kind.to_string(),
                state,
                pp_tokens: (prompt_tokens > 0).then_some(prompt_tokens),
                tg_tokens: (tokens > 0).then_some(tokens),
                ttft_s,
                gen_tps,
                mtp,
                progress,
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
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::Speed {
                    iteration: i + 1,
                    total: iterations,
                    tokens,
                });
            }
            let r = self.run_iteration(&prompt).await;
            tokens += r.completion_tokens;
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::Speed {
                    iteration: i + 1,
                    total: iterations,
                    tokens,
                });
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
    let mut content_chars = 0usize;
    for event in events {
        let StreamEvent::Frame { frame, .. } = event else {
            continue;
        };
        if frame.done {
            continue;
        }
        match &frame.chunk {
            Chunk::Content(text) => {
                content_chunks += 1;
                content_chars += text.len();
            }
            Chunk::Reasoning(_) => reasoning_chunks += 1,
            Chunk::Usage(_) | Chunk::Control => other_chunks += 1,
        }
    }
    let total_chunks = content_chunks + reasoning_chunks + other_chunks;

    // Token counts: server `usage` first; the prototype's fallbacks
    // otherwise (prompt → `chars/4` of the prompt; completion →
    // `chars/4` of the content, only when some chunk arrived), flagging
    // `estimated`.
    let (prompt_tokens, completion_tokens, estimated);
    match outcome.usage {
        Some(Usage {
            prompt_tokens: p,
            completion_tokens: c,
        }) if p > 0 && c > 0 => {
            (prompt_tokens, completion_tokens, estimated) = (p, c, false);
        }
        _ => {
            let mut est = false;
            prompt_tokens = outcome
                .usage
                .map(|u| u.prompt_tokens)
                .filter(|&n| n > 0)
                .unwrap_or_else(|| {
                    est = true;
                    prompt.token_count.max(1) as u64
                });
            completion_tokens = match outcome
                .usage
                .map(|u| u.completion_tokens)
                .filter(|&n| n > 0)
            {
                Some(n) => n,
                None if total_chunks > 0 => {
                    est = true;
                    (content_chars / 4).max(1) as u64
                }
                None => 0,
            };
            estimated = est;
        }
    }

    let pp_speed = if ttft > 0.0 {
        prompt_tokens as f64 / ttft
    } else {
        0.0
    };
    // §7.2: isolate the decode window from the prefill.
    let generation_time = (stream_time - ttft).max(0.001);
    let tg_speed = completion_tokens as f64 / generation_time;
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
    }
}

/// The `--json` document (parity with the prototype's `output_json`):
/// top-level `url` / `model` / `mode` / `iterations` / `results`, plus a
/// `summary` when more than one run is valid.
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
    let valid: Vec<&SpeedResult> = results.iter().filter(|r| !r.is_failed()).collect();
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

    /// A synthetic completed run: T0 → T1 (immediate) → T3 (+1ms) →
    /// Tn (+2ms), with 2 content frames, 1 reasoning frame, 1 usage
    /// frame, and the `[DONE]` terminator.
    fn completed_run() -> (StreamOutcome, Vec<StreamEvent>) {
        let t0 = MonotonicInstant::now();
        let t1 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(1));
        let t3 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(1));
        let t_end = MonotonicInstant::now();

        let usage = Usage {
            prompt_tokens: 128,
            completion_tokens: 34,
        };
        let frames = [
            ParsedFrame {
                chunk: Chunk::Reasoning("think".into()),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Content("Hello, ".into()),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Content("world!".into()),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Usage(usage),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Control,
                t_nanos: 0,
                done: true,
            },
        ];
        let events: Vec<StreamEvent> = frames
            .iter()
            .map(|f| StreamEvent::Frame {
                frame: f.clone(),
                at: t3,
                timestamps: StreamTimestamps {
                    t0: Some(t0),
                    t1: Some(t1),
                    t2: Some(t1),
                    t3: Some(t3),
                    t_end: Some(t_end),
                },
            })
            .chain(std::iter::once(StreamEvent::Complete {
                timestamps: StreamTimestamps {
                    t0: Some(t0),
                    t1: Some(t1),
                    t2: Some(t1),
                    t3: Some(t3),
                    t_end: Some(t_end),
                },
                usage: Some(usage),
                premature: false,
                malformed_frames: 0,
            }))
            .collect();
        let outcome = StreamOutcome {
            timestamps: StreamTimestamps {
                t0: Some(t0),
                t1: Some(t1),
                t2: Some(t1),
                t3: Some(t3),
                t_end: Some(t_end),
            },
            usage: Some(usage),
            premature: false,
            malformed_frames: 0,
            error: None,
        };
        (outcome, events)
    }

    #[test]
    fn aggregate_computes_section7_formulas() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events);

        // Chunk counting: `[DONE]` excluded; usage + … → "other".
        assert_eq!(r.content_chunks, 2);
        assert_eq!(r.reasoning_chunks, 1);
        assert_eq!(r.other_chunks, 1);
        assert_eq!(r.total_chunks, 4);

        // Server usage wins over the (estimated) prompt count.
        assert_eq!(r.prompt_tokens, 128);
        assert_eq!(r.completion_tokens, 34);
        assert!(!r.estimated);
        assert!(r.error.is_none());
        assert!(!r.is_failed());

        // TTFT ≈ 1ms (T3 − T1), stream time ≈ 2ms (Tn − T0).
        let ttft_ms = r.ttft * 1000.0;
        assert!(
            (0.5..2.5).contains(&ttft_ms),
            "ttft {ttft_ms}ms outside 0.5..2.5"
        );
        let stream_ms = r.stream_time * 1000.0;
        assert!(
            (1.5..4.0).contains(&stream_ms),
            "stream {stream_ms}ms outside 1.5..4.0"
        );

        // PP = tokens / TTFT; TG = tokens / decode window; MTP = 34/2.
        assert!((r.pp_speed - 128.0 / r.ttft).abs() < 1e-9);
        let gen = (r.stream_time - r.ttft).max(0.001);
        assert!((r.tg_speed - 34.0 / gen).abs() < 1e-9);
        assert!((r.mtp_efficiency - 17.0).abs() < 1e-9);

        assert_eq!(r.model, "test-model");
        assert_eq!(r.mode, "short");
    }

    #[test]
    fn missing_usage_falls_back_to_chars_over_4_and_flags_estimated() {
        let (mut outcome, events) = completed_run();
        outcome.usage = None;
        let r = aggregate(&cfg(), &prompt(), &outcome, &events);
        // Prompt fallback: the generated prompt's own (estimated) count.
        assert_eq!(r.prompt_tokens, 25);
        // Completion fallback: content chars (14) / 4 = 3.
        assert_eq!(r.completion_tokens, 3);
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
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[]);
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
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[]);
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
        };
        let r = aggregate(&cfg(), &prompt(), &outcome, &[]);
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
        let r = aggregate(&cfg(), &prompt(), &outcome, &events);
        assert_eq!(r.error.as_deref(), Some("2 malformed chunk(s) skipped"));
        // Tokens were produced → not a failed run (prototype semantics).
        assert!(!r.is_failed());
        assert!(!all_failed(&[r]));
    }

    #[test]
    fn json_report_matches_prototype_shape() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events);

        // One valid run: no summary (the prototype emits one only for
        // >1 valid runs).
        let v = json_report(&cfg(), std::slice::from_ref(&r));
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(keys, vec!["url", "model", "mode", "iterations", "results"]);
        assert!(v.get("summary").is_none());

        // Result keys: exactly the prototype `to_dict` set (no
        // `other_chunks`).
        let rj = v["results"][0].as_object().unwrap();
        assert_eq!(rj.len(), 14);
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
        ] {
            assert!(rj.contains_key(key), "missing {key}");
        }
        assert!(!rj.contains_key("other_chunks"));

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

    #[test]
    fn all_failed_only_when_every_run_failed() {
        let (outcome, events) = completed_run();
        let ok = aggregate(&cfg(), &prompt(), &outcome, &events);
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
        };
        let failed = aggregate(&cfg(), &prompt(), &bad, &[]);
        assert!(!all_failed(&[ok.clone(), failed.clone()]));
        assert!(all_failed(&[failed]));
        assert!(!all_failed(&[]));
    }

    #[test]
    fn result_box_mirrors_prototype_layout() {
        let (outcome, events) = completed_run();
        let r = aggregate(&cfg(), &prompt(), &outcome, &events);
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
        };
        let fr = aggregate(&cfg(), &prompt(), &failed, &[]);
        assert!(format_result_box(&fr, 1, 1, false).is_none());
    }

    #[test]
    fn summary_renders_for_two_valid_runs() {
        let (outcome, events) = completed_run();
        let a = aggregate(&cfg(), &prompt(), &outcome, &events);
        let b = aggregate(&cfg(), &prompt(), &outcome, &events);
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
        };
        let failed = aggregate(&cfg(), &prompt(), &bad, &[]);
        assert!(format_summary(&[a.clone(), failed], false).is_none());
    }
}
