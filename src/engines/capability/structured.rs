//! Engine C3 — Structured Output & JSON Grammar Compliance (plan Chunk 16,
//! blueprint §5 Engine C3).
//!
//! C3 answers two questions for **API / agent** use:
//!
//! 1. **How well does the model follow a JSON schema?** A *ladder* of three
//!    cases of increasing complexity (Simple flat object → Medium
//!    fixed-length array → Complex nested object) is each run under the
//!    OpenAI-compatible `response_format: { "type": "json_object" }`
//!    directive (server-side constrained decoding — vLLM XGrammar, SGLang,
//!    llama.cpp grammar, Outlines) and scored on a small set of checks
//!    (valid JSON, schema shape, all fields, correct types). Each case is
//!    classified [`CaseVerdict`]:
//!    * **Compliant** — every check passes;
//!    * **Partial** — valid JSON, but one or more checks fail;
//!    * **Failed** — the output is not valid JSON (cannot be evaluated).
//! 2. **What does grammar validation cost in speed?** One free-form
//!    baseline run (case 1, unconstrained) vs the constrained run (case 1)
//!    measures `penalty% = (free_tps − constrained_tps) / free_tps × 100`
//!    (positive = the grammar constraint costs throughput).
//!
//! The [`StructuredResult`] carries the per-case checks, the overall score
//! (N/3 compliant / partial / failed), the speed penalty, and an
//! auto-generated practical **verdict** — the TUI renders the full
//! compliance detail in the Live capability panel and a summary line in the
//! sequence header / headless output.
//!
//! Measurement-isolation note: every run reuses the Chunk 5 `StreamWorker`
//! (quanta `T0..Tn` live in the worker); this module only takes deltas of
//! those records.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::mpsc;

use crate::client::{join_worker, spawn_worker, token_window, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::sse::Chunk;
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Bounded worker→engine channel capacity (same as Engine A).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per structured run (a small JSON value).
pub const STRUCTURED_MAX_GEN_TOKENS: u32 = 256;

/// The complexity ladder: three cases of increasing schema depth. Each
/// prompt asks for a *specific* JSON shape so the constrained output can be
/// checked field-by-field.
#[derive(Debug, Clone, Copy)]
pub struct StructuredCase {
    /// The display name (`Simple` / `Medium` / `Complex`).
    pub name: &'static str,
    /// The generation prompt (the *only* thing that differs between runs is
    /// the `response_format` constraint — the prompt is identical for the
    /// free-form baseline and its constrained pair).
    pub prompt: &'static str,
}

/// The three C3 cases (blueprint §5 C3: simple → complex schema adherence).
pub const STRUCTURED_CASES: [StructuredCase; 3] = [
    StructuredCase {
        name: "Simple",
        prompt: "Return a JSON object with fields: \"name\" (a string), \
                 \"age\" (an integer). Respond with the JSON object only.",
    },
    StructuredCase {
        name: "Medium",
        prompt: "Return a JSON array of exactly 3 objects, each with: \
                 \"id\" (an integer), \"label\" (a string), \"active\" (a \
                 boolean). Respond with the JSON array only.",
    },
    StructuredCase {
        name: "Complex",
        prompt: "Return a JSON object matching this schema: \
                 {{\"user\": {{\"name\": string, \"email\": string}}, \
                 \"orders\": [{{\"id\": int, \"total\": number, \
                 \"items\": [string]}}]}}. Respond with the JSON object only.",
    },
];

/// The per-case compliance verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CaseVerdict {
    /// Every check passed.
    Compliant,
    /// Valid JSON, but one or more checks failed.
    Partial,
    /// Not valid JSON — cannot be evaluated.
    Failed,
}

/// One compliance check within a case (the `✓`/`✗` items in the UI).
#[derive(Debug, Clone)]
pub struct CaseCheck {
    /// The check label (`Valid JSON`, `Schema match`, `All fields`,
    /// `Types`, `Array length`).
    pub label: String,
    pub passed: bool,
}

/// The scored result of one case (its checks + verdict + raw output).
#[derive(Debug, Clone)]
pub struct StructuredCaseResult {
    /// The case name (`Simple` / `Medium` / `Complex`).
    pub name: String,
    /// The per-check results, in display order.
    pub checks: Vec<CaseCheck>,
    /// The classified outcome.
    pub verdict: CaseVerdict,
    /// The raw constrained output (shown, truncated, for non-compliant cases).
    pub output: String,
}

impl StructuredCaseResult {
    /// `true` when the `Valid JSON` check (the first check) passed.
    pub fn is_valid_json(&self) -> bool {
        self.checks
            .first()
            .is_some_and(|c| c.label == "Valid JSON" && c.passed)
    }

    /// `true` when the case is fully compliant.
    pub fn is_compliant(&self) -> bool {
        self.verdict == CaseVerdict::Compliant
    }
}

/// The full structured-output evaluation result.
#[derive(Debug, Clone)]
pub struct StructuredResult {
    /// Free-form (unconstrained) decode speed, tokens/s (case 1 baseline).
    pub free_tps: f64,
    /// Constrained (grammar-validated) decode speed, tokens/s (case 1).
    pub constrained_tps: f64,
    /// `(free − constrained) / free × 100` — positive = the grammar
    /// constraint costs throughput (the quantified penalty).
    pub penalty_pct: f64,
    pub free_ttft: f64,
    pub constrained_ttft: f64,
    /// The three scored cases (Simple → Complex).
    pub cases: Vec<StructuredCaseResult>,
    /// The constrained case-1 output (kept for headless / export detail).
    pub constrained_body: String,
    /// The free-form case-1 output.
    pub free_body: String,
}

impl StructuredResult {
    /// `(compliant, partial, failed)` counts over the cases.
    pub fn score(&self) -> (usize, usize, usize) {
        let (mut c, mut p, mut f) = (0, 0, 0);
        for case in &self.cases {
            match case.verdict {
                CaseVerdict::Compliant => c += 1,
                CaseVerdict::Partial => p += 1,
                CaseVerdict::Failed => f += 1,
            }
        }
        (c, p, f)
    }

    /// `true` when *every* case is fully compliant (the old boolean flag).
    pub fn fully_compliant(&self) -> bool {
        self.cases
            .iter()
            .all(|c| c.verdict == CaseVerdict::Compliant)
    }

    /// A compact score label (`2/3 compliant, 1 partial, 0 failed`).
    pub fn score_label(&self) -> String {
        let (c, p, f) = self.score();
        let total = self.cases.len();
        if p == 0 && f == 0 {
            format!("{c}/{total} compliant")
        } else if f == 0 {
            format!("{c}/{total} compliant, {p} partial")
        } else {
            format!("{c}/{total} compliant, {p} partial, {f} failed")
        }
    }

    /// The auto-generated practical verdict (based on the compliant count):
    /// 3/3 → API/agent-ready; 2/3 → simple-only; 1/3 → trivial only;
    /// 0/3 → not suitable.
    pub fn verdict_line(&self) -> String {
        let (c, _, _) = self.score();
        match c {
            3 => "✓ Fully suitable for API/agent use".to_string(),
            2 => "⚠ Suitable for simple structures, unreliable for complex schemas".to_string(),
            1 => "⚠ Limited use — only trivial key-value extraction".to_string(),
            _ => "✗ Not suitable for structured output. Requires output parsing/fallback."
                .to_string(),
        }
    }

    /// One-line summary for the sequence header / headless output
    /// (`+30.0% penalty · 2/3 compliant`).
    pub fn summary_line(&self) -> String {
        format!("{:+.1}% penalty · {}", self.penalty_pct, self.score_label())
    }
}

/// The live runner: a free-form baseline plus the three constrained cases.
#[derive(Debug)]
pub struct StructuredEngine {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    timeout: u64,
    /// Optional HF tokenizer for re-tokenizing output when a proxy omits
    /// `usage` (the research's fallback counting method).
    tokenizer: Option<Arc<crate::prompt::Tokenizer>>,
    /// Optional sequence seam: publish `Run {n}/{total}` to the
    /// [`ProgressBus`] the Benchmark Sequence mirrors into the TUI.
    progress: Option<Arc<ProgressBus>>,
    /// Optional TUI seam: publish live single-stream
    /// [`MetricsState`] snapshots while a run, so the Live Monitor shows
    /// the current run's real stream data.
    metrics: Option<Arc<MetricsState>>,
    /// Optional `Space`-key pause gate: each run waits on it before its
    /// stream is spawned (in-flight runs complete).
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
        let tokenizer = match &cfg.tokenizer {
            Some(path) => Some(Arc::new(crate::prompt::Tokenizer::from_file(path)?)),
            None => None,
        };
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            tokenizer,
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
    }

    /// Attach a [`ProgressBus`] so the runs report `Run {n}/{total}` to the
    /// Benchmark Sequence (the TUI's progress bar).
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
    async fn run_once(&self, prompt: &str, constrained: bool) -> (String, f64, f64) {
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            prompt,
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
        // Spawn without awaiting: drain the channel concurrently to avoid
        // the >capacity deadlock (server-agnostic fix).
        let handle = spawn_worker(worker, tx);
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

        // Channel closed: collect the outcome.
        let outcome = join_worker(handle).await;

        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
        let ttft = match ts.ttft_nanos() {
            Some(n) => n as f64 / 1e9,
            None => stream_time,
        };
        // Authoritative count: the server's `usage.completion_tokens`
        // first; re-tokenize the full output with the model's exact
        // tokenizer when a proxy omits usage; the observed frame count as
        // a last-resort estimate.
        let completion = match outcome
            .usage
            .map(|u| u.completion_tokens)
            .filter(|&n| n > 0)
        {
            Some(n) => n,
            None => match self
                .tokenizer
                .as_deref()
                .and_then(|t| t.try_count(&body))
                .filter(|&n| n > 0)
            {
                Some(n) => n as u64,
                None => outcome.frames.max(1),
            },
        };
        // Decode window: first → last content/reasoning token (prefill /
        // TTFT excluded) — the research's Timing Boundary Rule.
        let (t_first, t_last) = token_window(&events);
        let generation_time = match (t_first, t_last) {
            (Some(f), Some(l)) => (f.delta_nanos(&l) as f64 / 1e9).max(0.001),
            _ => (stream_time - ttft).max(0.001),
        };
        let tps = completion as f64 / generation_time;
        (body, ttft, tps)
    }

    /// The full evaluation: a free-form baseline (case 1), then the three
    /// constrained cases. The speed penalty compares case 1 free-form vs
    /// constrained; each case is scored on its own schema.
    ///
    /// While a [`ProgressBus`] is attached, each run publishes
    /// `Run {n}/4` — the Benchmark Sequence mirrors it into the TUI's
    /// progress bar.
    pub async fn run(&self) -> StructuredResult {
        const TOTAL_RUNS: usize = 4;
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC3,
                "Engine C3 (Structured) started — 3 schema cases (Simple → Complex) + free-form baseline",
            );
        }

        // Run 1 — free-form baseline (case 1 prompt, *no* constraint).
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured {
                run: 1,
                total: TOTAL_RUNS,
            });
        }
        let (free_body, free_ttft, free_tps) =
            self.run_once(STRUCTURED_CASES[0].prompt, false).await;

        // Run 2 — constrained case 1 (the grammar-speed pair).
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured {
                run: 2,
                total: TOTAL_RUNS,
            });
        }
        let (c1_body, c1_ttft, c1_tps) = self.run_once(STRUCTURED_CASES[0].prompt, true).await;

        // Run 3 — constrained case 2.
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured {
                run: 3,
                total: TOTAL_RUNS,
            });
        }
        let (c2_body, _, _) = self.run_once(STRUCTURED_CASES[1].prompt, true).await;

        // Run 4 — constrained case 3.
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured {
                run: 4,
                total: TOTAL_RUNS,
            });
        }
        let (c3_body, _, _) = self.run_once(STRUCTURED_CASES[2].prompt, true).await;

        let constrained_tps = c1_tps;
        let constrained_ttft = c1_ttft;
        let penalty_pct = if free_tps > 0.0 {
            (free_tps - constrained_tps) / free_tps * 100.0
        } else {
            0.0
        };

        let cases = vec![
            evaluate_case(STRUCTURED_CASES[0].name, &c1_body),
            evaluate_case(STRUCTURED_CASES[1].name, &c2_body),
            evaluate_case(STRUCTURED_CASES[2].name, &c3_body),
        ];

        let result = StructuredResult {
            free_tps,
            constrained_tps,
            penalty_pct,
            free_ttft,
            constrained_ttft,
            cases,
            constrained_body: c1_body,
            free_body,
        };
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC3,
                format!("Engine C3 complete — {}", result.summary_line()),
            );
        }
        result
    }
}

// ── Per-case evaluation ────────────────────────────────────────────────────

/// `true` when `v` is a JSON integer (not a float).
fn is_int(v: &serde_json::Value) -> bool {
    v.as_i64().is_some()
}

/// Score one case's constrained output against its checks and classify it.
///
/// Pure over its inputs (unit-testable, no network): the output is parsed
/// (a ` ```json ` fence is stripped first), and each case checks its own
/// schema shape, fields, and types.
pub fn evaluate_case(name: &str, body: &str) -> StructuredCaseResult {
    let normalized = strip_json_fence(body);
    let value = serde_json::from_str::<serde_json::Value>(normalized).ok();
    let valid = value.is_some();

    let mut checks: Vec<CaseCheck> = vec![CaseCheck {
        label: "Valid JSON".to_string(),
        passed: valid,
    }];

    // Only the non-JSON checks run when the body parsed; otherwise the case
    // is `Failed` and there is nothing further to evaluate.
    if let Some(v) = &value {
        match name {
            "Simple" => {
                let schema = v.is_object();
                let (fields, types) = if let Some(o) = v.as_object() {
                    (
                        o.contains_key("name") && o.contains_key("age"),
                        o.get("name").map(|x| x.is_string()).unwrap_or(false)
                            && o.get("age").map(is_int).unwrap_or(false),
                    )
                } else {
                    (false, false)
                };
                checks.push(CaseCheck {
                    label: "Schema match".into(),
                    passed: schema,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
            }
            "Medium" => {
                let is_array = v.is_array();
                let (len_ok, fields, types) = if let Some(a) = v.as_array() {
                    (
                        a.len() == 3,
                        a.iter().all(|o| {
                            o.is_object()
                                && o.get("id").is_some()
                                && o.get("label").is_some()
                                && o.get("active").is_some()
                        }),
                        a.iter().all(|o| {
                            o.get("id").map(is_int).unwrap_or(false)
                                && o.get("label").map(|x| x.is_string()).unwrap_or(false)
                                && o.get("active").map(|x| x.is_boolean()).unwrap_or(false)
                        }),
                    )
                } else {
                    (false, false, false)
                };
                checks.push(CaseCheck {
                    label: "Array length".into(),
                    passed: len_ok,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
                // The "Schema match" for the array case is carried by the
                // Array-length check; report shape separately.
                checks.insert(
                    1,
                    CaseCheck {
                        label: "Schema match".into(),
                        passed: is_array,
                    },
                );
            }
            "Complex" => {
                let schema = v.is_object();
                let (fields, types) = if let Some(o) = v.as_object() {
                    let user_fields = o
                        .get("user")
                        .map(|u| {
                            u.is_object() && u.get("name").is_some() && u.get("email").is_some()
                        })
                        .unwrap_or(false);
                    let orders_fields = o
                        .get("orders")
                        .and_then(|ord| ord.as_array())
                        .map(|items| {
                            items.iter().all(|it| {
                                it.is_object()
                                    && it.get("id").is_some()
                                    && it.get("total").is_some()
                                    && it.get("items").is_some()
                            })
                        })
                        .unwrap_or(false);
                    let user_types = o
                        .get("user")
                        .map(|u| {
                            u.get("name").map(|x| x.is_string()).unwrap_or(false)
                                && u.get("email").map(|x| x.is_string()).unwrap_or(false)
                        })
                        .unwrap_or(false);
                    let orders_types = o
                        .get("orders")
                        .and_then(|ord| ord.as_array())
                        .map(|items| {
                            items.iter().all(|it| {
                                it.get("id").map(is_int).unwrap_or(false)
                                    && it.get("total").map(|x| x.is_number()).unwrap_or(false)
                                    && it
                                        .get("items")
                                        .and_then(|i| i.as_array())
                                        .map(|arr| arr.iter().all(|s| s.is_string()))
                                        .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    (user_fields && orders_fields, user_types && orders_types)
                } else {
                    (false, false)
                };
                checks.push(CaseCheck {
                    label: "Schema match".into(),
                    passed: schema,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
            }
            // Unknown case name: only the Valid-JSON check was added.
            _ => {}
        }
    }

    let verdict = if !valid {
        CaseVerdict::Failed
    } else if checks.iter().all(|c| c.passed) {
        CaseVerdict::Compliant
    } else {
        CaseVerdict::Partial
    };

    StructuredCaseResult {
        name: name.to_string(),
        checks,
        verdict,
        output: body.to_string(),
    }
}

/// `true` when `body` is a JSON object carrying every `required_fields` key.
///
/// A general-purpose compliance helper (kept public for the headless /
/// export paths and the tests). Normalizes the common model habit of
/// wrapping the object in a ` ```json ` fenced block before parsing.
pub fn is_json_compliant(body: &str, required_fields: &[&str]) -> bool {
    let normalized = strip_json_fence(body);
    serde_json::from_str::<serde_json::Value>(normalized)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .map(|o| required_fields.iter().all(|f| o.contains_key(*f)))
        .unwrap_or(false)
}

/// Strip a single leading/trailing ` ``` ` (json) fence, if present.
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

    // ── is_json_compliant (general helper) ───────────────────────────────

    #[test]
    fn compliant_object_passes() {
        let body = r#"{"name": "Ada", "age": 36, "city": "London"}"#;
        assert!(is_json_compliant(body, &["name", "age", "city"]));
    }

    #[test]
    fn missing_field_fails() {
        let body = r#"{"name": "Ada", "age": 36}"#;
        assert!(!is_json_compliant(body, &["name", "age", "city"]));
    }

    #[test]
    fn non_json_fails() {
        assert!(!is_json_compliant(
            "Just some prose, no JSON here.",
            &["name", "age", "city"]
        ));
        assert!(!is_json_compliant("", &["name", "age", "city"]));
        // A JSON array is not a JSON object.
        assert!(!is_json_compliant(
            r#"[{"name": "Ada"}]"#,
            &["name", "age", "city"]
        ));
    }

    #[test]
    fn fenced_json_is_normalized() {
        let body = "```json\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &["name", "age", "city"]));
        // Bare fence without a language tag too.
        let body = "```\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &["name", "age", "city"]));
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

    // ── evaluate_case (the per-case schema checks) ───────────────────────

    #[test]
    fn simple_case_compliant() {
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#);
        assert_eq!(r.verdict, CaseVerdict::Compliant);
        assert!(r.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn simple_case_partial_on_bad_type() {
        // `age` is a string, not an integer → Types fails, rest pass.
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": "36"}"#);
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(!r.checks.iter().find(|c| c.label == "Types").unwrap().passed);
    }

    #[test]
    fn medium_case_compliant_array_of_three() {
        let r = evaluate_case(
            "Medium",
            r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}, {"id": 3, "label": "three", "active": true}]"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Compliant);
    }

    #[test]
    fn medium_case_partial_on_wrong_length() {
        // Two objects instead of three → Array length fails.
        let r = evaluate_case(
            "Medium",
            r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}]"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(
            !r.checks
                .iter()
                .find(|c| c.label == "Array length")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn complex_case_compliant_nested() {
        let r = evaluate_case(
            "Complex",
            r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": [{"id": 1, "total": 50.5, "items": ["a", "b"]}, {"id": 2, "total": 10, "items": []}]}"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Compliant);
        assert!(r.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn complex_case_partial_when_orders_is_text() {
        // `orders` is free text, not an array of objects → schema/types fail.
        let r = evaluate_case(
            "Complex",
            r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": "I have 2 orders totaling $5000"}"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(!r.checks.iter().find(|c| c.label == "Types").unwrap().passed);
    }

    #[test]
    fn non_json_is_failed() {
        let r = evaluate_case("Simple", "this is definitely not json");
        assert_eq!(r.verdict, CaseVerdict::Failed);
        assert!(!r.is_valid_json());
        // Only the Valid-JSON check is present (the rest are unevaluable).
        assert_eq!(r.checks.len(), 1);
    }

    #[test]
    fn fenced_output_is_stripped_before_parsing() {
        let r = evaluate_case("Simple", "```json\n{\"name\": \"Ada\", \"age\": 36}\n```");
        assert_eq!(r.verdict, CaseVerdict::Compliant);
    }

    // ── StructuredResult score / verdict ─────────────────────────────────

    fn result_with(c1: &str, c2: &str, c3: &str) -> StructuredResult {
        StructuredResult {
            free_tps: 65.2,
            constrained_tps: 61.8,
            penalty_pct: 5.2,
            free_ttft: 0.2,
            constrained_ttft: 0.21,
            cases: vec![
                evaluate_case("Simple", c1),
                evaluate_case("Medium", c2),
                evaluate_case("Complex", c3),
            ],
            constrained_body: c1.to_string(),
            free_body: "prose".to_string(),
        }
    }

    #[test]
    fn score_and_verdict_track_compliant_count() {
        // 3/3 compliant.
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        assert_eq!(r.score(), (3, 0, 0));
        assert!(r.fully_compliant());
        assert_eq!(r.verdict_line(), "✓ Fully suitable for API/agent use");

        // 1/3 compliant (simple passes; medium wrong length; complex text).
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true}]"#,
            r#"{"user":{"name":"Ada"},"orders":"free text"}"#,
        );
        assert_eq!(r.score(), (1, 2, 0));
        assert_eq!(
            r.verdict_line(),
            "⚠ Limited use — only trivial key-value extraction"
        );

        // 0/3 (all invalid JSON).
        let r = result_with("nope", "nope", "nope");
        assert_eq!(r.score(), (0, 0, 3));
        assert!(!r.fully_compliant());
        assert!(r.verdict_line().starts_with("✗ Not suitable"));
    }

    #[test]
    fn summary_line_combines_penalty_and_score() {
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        // 2 compliant (simple, complex), 1 partial (medium: 2 not 3).
        assert_eq!(r.score(), (2, 1, 0));
        assert!(r.summary_line().contains("+5.2% penalty"));
        assert!(r.summary_line().contains("2/3 compliant"));
    }
}
