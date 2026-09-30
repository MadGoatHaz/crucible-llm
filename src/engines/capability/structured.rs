//! Engine C3 — Structured Output & JSON Grammar Compliance (plan Chunk 16,
//! blueprint §5 Engine C3).
//!
//! Runs the same generation task **twice** against the endpoint:
//!
//! 1. **free-form** — unconstrained decoding (the Chunk 5 worker as-is);
//! 2. **constrained** — the OpenAI-compatible
//!    `response_format: { "type": "json_object" }` directive, which
//!    server-side constrained-decoding engines (vLLM XGrammar, SGLang,
//!    llama.cpp grammar support, Outlines) honor by restricting the
//!    sampling engine to JSON-grammar tokens.
//!
//! It then quantifies the **speed penalty of grammar validation** —
//! `penalty% = (free_tps − constrained_tps) / free_tps × 100` (positive
//! = the grammar constraint costs throughput) — alongside **compliance**:
//! the constrained output must parse as a JSON object carrying the
//! required fields.
//!
//! Measurement-isolation note: both runs reuse the Chunk 5
//! `StreamWorker` (quanta `T0..Tn` live in the worker); this module only
//! takes deltas of those records.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::client::{StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::sse::Chunk;
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Bounded worker→engine channel capacity (same as Engine A).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per structured run (a small JSON object).
pub const STRUCTURED_MAX_GEN_TOKENS: u32 = 256;

/// The generation task both runs share (identical prompt — only the
/// decoding constraint differs, isolating the grammar cost).
pub const STRUCTURED_TASK: &str = "Generate a JSON object with exactly these fields: \
\"name\" (a string), \"age\" (an integer), \"city\" (a string). \
Respond with the JSON object only.";

/// The required fields the constrained output must carry.
pub const REQUIRED_FIELDS: [&str; 3] = ["name", "age", "city"];

/// The result of one structured-output evaluation.
#[derive(Debug, Clone)]
pub struct StructuredResult {
    /// Free-form (unconstrained) decode speed, tokens/s (§7.2).
    pub free_tps: f64,
    /// Constrained (grammar-validated) decode speed, tokens/s.
    pub constrained_tps: f64,
    /// `(free − constrained) / free × 100` — positive = the grammar
    /// constraint costs throughput (the quantified penalty).
    pub penalty_pct: f64,
    pub free_ttft: f64,
    pub constrained_ttft: f64,
    /// The constrained output parses as a JSON object with every
    /// required field present (the compliance check).
    pub compliant: bool,
    /// The concatenated constrained response text.
    pub constrained_body: String,
    /// The concatenated free-form response text.
    pub free_body: String,
}

/// The live runner: two single-stream runs (free-form, then constrained).
#[derive(Debug)]
pub struct StructuredEngine {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    timeout: u64,
    /// Optional sequence seam: publish `Run {n}/{total}` to the
    /// [`ProgressBus`] the Benchmark Sequence mirrors into the TUI.
    progress: Option<Arc<ProgressBus>>,
    /// Optional TUI seam: publish live single-stream
    /// [`MetricsState`] snapshots while a run, so the Live Monitor shows
    /// the current run's real stream data.
    metrics: Option<Arc<MetricsState>>,
    /// Optional `Space`-key pause gate: each of the two runs waits on it
    /// before its stream is spawned (in-flight runs complete).
    pause: Option<Arc<RunPause>>,
    /// Optional run logger (each run's worker records its HTTP / SSE
    /// lifecycle; the engine records the evaluation start / result).
    logger: Option<Arc<crate::log::RunLogger>>,
}

impl StructuredEngine {
    /// Build from the resolved config (same client policy as Engine A).
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
    }

    /// Attach a [`ProgressBus`] so the two runs report `Run {n}/{total}`
    /// to the Benchmark Sequence (the TUI's progress bar).
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach a [`MetricsState`] publisher so each run pushes live
    /// single-stream snapshots to the TUI (the Live Monitor's stream
    /// matrix / gauges show the current run's real data).
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach the `Space`-key pause gate: each run waits on it before
    /// its stream is spawned (the headless path leaves it `None`).
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger (each run's stream then logs its HTTP /
    /// SSE lifecycle).
    pub fn logger(mut self, logger: Arc<crate::log::RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// One run: spawn a worker (optionally with the JSON `response_format`
    /// constraint), drain its channel, and synthesize the §7 metrics.
    async fn run_once(&self, constrained: bool) -> (String, f64, f64) {
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            STRUCTURED_TASK,
            STRUCTURED_MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        .tag(if constrained {
            "C3:constrained"
        } else {
            "C3:free"
        });
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }
        if constrained {
            worker = worker.response_format("json_object");
        }

        let start = MonotonicInstant::now();
        let outcome = tokio::spawn(worker.run(tx))
            .await
            .expect("structured worker task panicked");
        let mut body = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    if let Chunk::Content(text) = &frame.chunk {
                        body.push_str(text);
                    }
                }
            }
            // Publish a live snapshot every 8 events, and immediately on
            // a terminal event, so the TUI's Live Monitor shows the
            // current run's real stream data (measurement-isolation
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
                        "Structured",
                        &events,
                        &start,
                        STRUCTURED_MAX_GEN_TOKENS,
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
                "Structured",
                &events,
                &start,
                STRUCTURED_MAX_GEN_TOKENS,
            ));
        }

        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
        let ttft = match ts.ttft_nanos() {
            Some(n) => n as f64 / 1e9,
            None => stream_time,
        };
        let completion = outcome
            .usage
            .map(|u| u.completion_tokens)
            .unwrap_or_else(|| (body.chars().count() / 4).max(1) as u64);
        let generation_time = (stream_time - ttft).max(0.001);
        let tps = completion as f64 / generation_time;
        (body, ttft, tps)
    }

    /// The full evaluation: free-form baseline, then the grammar-
    /// constrained run, plus the compliance check and penalty.
    ///
    /// While a [`ProgressBus`] is attached, each run publishes
    /// `Run {n}/2` — the Benchmark Sequence mirrors it into the TUI's
    /// progress bar.
    pub async fn run(&self) -> StructuredResult {
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC3,
                "Engine C3 (Structured) started — free-form vs constrained",
            );
        }
        // The `Space`-key pause: hold before each run's request goes out
        // (an in-flight run always completes).
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured { run: 1, total: 2 });
        }
        let (free_body, free_ttft, free_tps) = self.run_once(false).await;
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured { run: 2, total: 2 });
        }
        let (constrained_body, constrained_ttft, constrained_tps) = self.run_once(true).await;

        let penalty_pct = if free_tps > 0.0 {
            (free_tps - constrained_tps) / free_tps * 100.0
        } else {
            0.0
        };

        StructuredResult {
            free_tps,
            constrained_tps,
            penalty_pct,
            free_ttft,
            constrained_ttft,
            compliant: is_json_compliant(&constrained_body, &REQUIRED_FIELDS),
            constrained_body,
            free_body,
        }
    }
}

/// `true` when `body` is a JSON object carrying every `required_fields`
/// key.
///
/// Normalizes the common model habit of wrapping the object in a
/// ```json fenced block before parsing.
pub fn is_json_compliant(body: &str, required_fields: &[&str]) -> bool {
    let normalized = strip_json_fence(body);
    serde_json::from_str::<serde_json::Value>(normalized)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .map(|o| required_fields.iter().all(|f| o.contains_key(*f)))
        .unwrap_or(false)
}

/// Strip a single leading/trailing ``` (json) fence, if present.
fn strip_json_fence(body: &str) -> &str {
    let trimmed = body.trim();
    if let Some(inner) = trimmed.strip_prefix("```") {
        // Skip the optional language tag on the fence line.
        let after_lang = inner.split_once('\n').map(|(l, r)| {
            let lang = l.trim();
            if lang.is_empty() || lang.eq_ignore_ascii_case("json") {
                r
            } else {
                inner
            }
        });
        if let Some(r) = after_lang {
            if let Some(closing) = r.rsplit_once("```") {
                return closing.0.trim_end();
            }
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compliant_object_passes() {
        let body = r#"{"name": "Ada", "age": 36, "city": "London"}"#;
        assert!(is_json_compliant(body, &REQUIRED_FIELDS));
    }

    #[test]
    fn missing_field_fails() {
        let body = r#"{"name": "Ada", "age": 36}"#;
        assert!(!is_json_compliant(body, &REQUIRED_FIELDS));
    }

    #[test]
    fn non_json_fails() {
        assert!(!is_json_compliant(
            "Just some prose, no JSON here.",
            &REQUIRED_FIELDS
        ));
        assert!(!is_json_compliant("", &REQUIRED_FIELDS));
        // A JSON array is not a JSON object.
        assert!(!is_json_compliant(r#"[{"name": "Ada"}]"#, &REQUIRED_FIELDS));
    }

    #[test]
    fn fenced_json_is_normalized() {
        let body = "```json\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &REQUIRED_FIELDS));
        // Bare fence without a language tag too.
        let body = "```\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &REQUIRED_FIELDS));
    }

    #[test]
    fn strip_json_fence_leaves_plain_text_alone() {
        assert_eq!(strip_json_fence("  hello  "), "hello");
    }

    #[test]
    fn penalty_is_positive_when_constrained_is_slower() {
        // Mirrors the production formula.
        let free: f64 = 100.0;
        let constrained: f64 = 70.0;
        let penalty: f64 = (free - constrained) / free * 100.0;
        assert!((penalty - 30.0).abs() < 1e-9);
    }
}
