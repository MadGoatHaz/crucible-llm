//! Engine C2 — Deterministic Reasoning & Code Verification (plan Chunk 16,
//! blueprint §5 Engine C2).
//!
//! A standardized bank of self-contained logic / math / code-generation
//! challenges. Every challenge is validated by a **strict, deterministic
//! checker** — no LLM-as-judge, no fuzzy scoring — so the resulting
//! pass/fail accuracy (`N/M solved`) is reproducible run-to-run and
//! comparable across models, quantizations, and server versions:
//!
//! * **Math** — the *answer-position* (first) numeric value in the
//!   committed span must equal the exact expected value; the expected
//!   value is also accepted when it is present whole anywhere in the span
//!   (CF-33 — a chatty-but-correct answer that restates other numbers
//!   after the answer is not false-failed).
//! * **Logic** — the *answer-position* yes/no verdict (case-insensitive,
//!   whole-word) in the committed span must match the expected verdict
//!   (CF-34 — the committed answer, not trailing prose).
//! * **Code** — the response must contain a fenced code block; the block
//!   must have balanced, properly nested delimiters (`()` `[]` `{}`) and
//!   contain every required structural marker (the function signature,
//!   key identifiers).
//!
//! **The judged span (CF-16).** Every checker runs on the model's
//! *committed content span* only. The request suppresses the
//! chain-of-thought (per-backend, graceful — `StreamWorker::disable_thinking`,
//! CF-01), and the runner accumulates `Chunk::Content` deltas only, never
//! `Chunk::Reasoning`: a value that appears *only* in the CoT is latent
//! reasoning, not a demonstrated answer (judging the combined stream would
//! false-pass it), and a CoT truncated mid-thought leaves an intermediate
//! value where the answer belongs (judging the combined stream would
//! false-fail it). The CoT is still forwarded to the live snapshot /
//! metrics — just not to the checkers.
//!
//! **Dependency note:** all checkers are pure `std` (no `regex` or
//! parser crates) to preserve the §3.1 dependency gate. The code checker
//! is a *structural* (lexical) validator; full AST parsing is a
//! post-v1 extension that does not change the pass/fail contract.
//!
//! Measurement-isolation note: the live runner reuses the Chunk 5
//! `StreamWorker` (quanta `T0..Tn` live in the worker); this module only
//! takes deltas of those records, exactly as Engine A does.

use std::time::Duration;

use tokio::sync::mpsc;

use std::sync::Arc;

use crate::client::{join_worker, spawn_worker, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::sse::{Chunk, Usage, UsageDetails};
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Bounded worker→engine channel capacity (same as Engine A).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per challenge (code answers can be long).
pub const REASONING_MAX_GEN_TOKENS: u32 = 512;

/// The strict, deterministic checker for one challenge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Checker {
    /// The answer-position (first) number must equal `value`, or `value`
    /// must be present whole in the committed span (CF-33).
    ExactNumber(f64),
    /// The answer-position whole-word yes/no verdict must be
    /// `expected_yes` (CF-34).
    Verdict { expected_yes: bool },
    /// A fenced code block with balanced delimiters containing every
    /// required structural marker.
    Code { required: &'static [&'static str] },
}

/// One self-contained challenge from the bank.
#[derive(Debug, Clone, Copy)]
pub struct Challenge {
    /// Stable identifier (e.g. `math-mul`).
    pub id: &'static str,
    /// `math` | `logic` | `code`.
    pub category: &'static str,
    /// The prompt sent to the model.
    pub prompt: &'static str,
    /// The strict checker.
    pub checker: Checker,
    /// A known-correct response (the canonical answer) — used by tests to
    /// prove the checker accepts the truth and rejects corruptions.
    pub canonical_answer: &'static str,
    /// A unique ASCII anchor present in `prompt` — the mock test
    /// server dispatches on it (the body carries the prompt, not this
    /// field).
    pub test_key: &'static str,
}

impl Challenge {
    /// `true` when `response` solves this challenge (strict checker).
    pub fn is_solved(&self, response: &str) -> bool {
        match self.checker {
            // CF-33: the answer-position (first) number matches, OR the
            // expected value is present whole anywhere in the committed
            // span — a chatty-but-correct answer that restates other
            // numbers is not false-failed.
            Checker::ExactNumber(v) => {
                extract_answer_number(response).is_some_and(|got| (got - v).abs() < 1e-6)
                    || contains_number(response, v)
            }
            // CF-34: the committed content span's answer-position verdict.
            Checker::Verdict { expected_yes } => extract_verdict(response) == Some(expected_yes),
            Checker::Code { required } => extract_code_fence(response).is_some_and(|code| {
                delimiters_balanced(&code) && required.iter().all(|m| contains_marker(&code, m))
            }),
        }
    }
}

/// The standardized challenge bank (13 challenges: 5 math, 5 logic,
/// 3 code). Every entry has a hand-verified canonical answer.
pub const REASONING_BANK: &[Challenge] = &[
    Challenge {
        id: "math-mul",
        category: "math",
        prompt: "What is 27 * 43? Answer with the number only.",
        checker: Checker::ExactNumber(1161.0),
        canonical_answer: "1161",
        test_key: "27 * 43",
    },
    Challenge {
        id: "math-frac",
        category: "math",
        prompt: "What is 3/4 of 240? Answer with the number only.",
        checker: Checker::ExactNumber(180.0),
        canonical_answer: "180",
        test_key: "3/4 of 240",
    },
    Challenge {
        id: "math-seq",
        category: "math",
        prompt: "What is the next number in the sequence 2, 4, 8, 16, 32? Answer with the number only.",
        checker: Checker::ExactNumber(64.0),
        canonical_answer: "64",
        test_key: "2, 4, 8, 16, 32",
    },
    Challenge {
        id: "math-pct",
        category: "math",
        prompt: "A price of $80 is increased by 15%. What is the new price in dollars? Answer with the number only.",
        checker: Checker::ExactNumber(92.0),
        canonical_answer: "92",
        test_key: "increased by 15%",
    },
    Challenge {
        id: "math-sum",
        category: "math",
        prompt: "What is the sum of the integers from 1 to 100? Answer with the number only.",
        checker: Checker::ExactNumber(5050.0),
        canonical_answer: "5050",
        test_key: "from 1 to 100",
    },
    Challenge {
        id: "logic-syllogism",
        category: "logic",
        prompt: "All engineers are programmers. Some programmers are designers. Therefore, all engineers are designers. Is this conclusion valid? Answer yes or no only.",
        checker: Checker::Verdict {
            expected_yes: false,
        },
        canonical_answer: "No",
        test_key: "All engineers are programmers",
    },
    Challenge {
        id: "logic-parity",
        category: "logic",
        prompt: "Is the product 17 * 19 an even number? Answer yes or no only.",
        checker: Checker::Verdict {
            expected_yes: false,
        },
        canonical_answer: "No",
        test_key: "17 * 19",
    },
    Challenge {
        id: "logic-knights",
        category: "logic",
        prompt: "On an island where knights always tell the truth and knaves always lie, A says \"B is a knave\" and B says \"A is a knave\". Can both A and B be knights? Answer yes or no only.",
        checker: Checker::Verdict {
            expected_yes: false,
        },
        canonical_answer: "No",
        test_key: "Can both A and B be knights",
    },
    Challenge {
        id: "logic-eggs",
        category: "logic",
        prompt: "If it takes 5 minutes to boil one egg in a pot of boiling water, how many minutes does it take to boil four eggs in the same pot? Answer with the number only.",
        checker: Checker::ExactNumber(5.0),
        canonical_answer: "5",
        test_key: "boil four eggs",
    },
    Challenge {
        id: "logic-premise",
        category: "logic",
        prompt: "Assume for this question that all birds can fly and that penguins are birds. Under these assumptions, can penguins fly? Answer yes or no only.",
        checker: Checker::Verdict {
            expected_yes: true,
        },
        canonical_answer: "Yes",
        test_key: "can penguins fly",
    },
    Challenge {
        id: "code-fib",
        category: "code",
        prompt: "Write a Rust function `fib(n: u32) -> u32` that returns the nth Fibonacci number (fib(0) = 0, fib(1) = 1). Answer with the code in a fenced code block only.",
        checker: Checker::Code {
            required: &["fn fib", "n: u32", "-> u32"],
        },
        canonical_answer: "```rust\nfn fib(n: u32) -> u32 {\n    if n < 2 { return n; }\n    let (mut a, mut b) = (0u32, 1u32);\n    for _ in 2..=n {\n        let c = a + b;\n        a = b;\n        b = c;\n    }\n    b\n}\n```",
        test_key: "nth Fibonacci number",
    },
    Challenge {
        id: "code-reverse",
        category: "code",
        prompt: "Write a Rust function `reverse_string(s: &str) -> String` that returns `s` with its characters in reverse order. Answer with the code in a fenced code block only.",
        checker: Checker::Code {
            required: &["fn reverse_string", "s: &str", "-> String"],
        },
        canonical_answer: "```rust\nfn reverse_string(s: &str) -> String {\n    s.chars().rev().collect()\n}\n```",
        test_key: "reverse order",
    },
    Challenge {
        id: "code-gcd",
        category: "code",
        prompt: "Write a Rust function `gcd(a: u64, b: u64) -> u64` that computes the greatest common divisor using the Euclidean algorithm. Answer with the code in a fenced code block only.",
        checker: Checker::Code {
            required: &["fn gcd", "a: u64", "b: u64", "-> u64"],
        },
        canonical_answer: "```rust\nfn gcd(mut a: u64, mut b: u64) -> u64 {\n    while b != 0 {\n        let t = a % b;\n        a = b;\n        b = t;\n    }\n    a\n}\n```",
        test_key: "Euclidean algorithm",
    },
];

/// The deterministic accuracy score over the committed content spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReasoningScore {
    pub total: usize,
    pub solved: usize,
    /// `(solved, total)` per category, in the order math / logic / code.
    pub by_category: [(usize, usize); 3],
}

impl ReasoningScore {
    /// `N/M solved (P%)` — the headline accuracy figure.
    pub fn label(&self) -> String {
        let pct = if self.total > 0 {
            self.solved as f64 / self.total as f64 * 100.0
        } else {
            0.0
        };
        format!("{}/{} solved ({pct:.1}%)", self.solved, self.total)
    }
}

/// Score `responses` (index-aligned with [`REASONING_BANK`]) against the
/// strict checkers. Pure — no network, no timing.
///
/// **CF-48:** the denominator (`total`) is the number of challenges
/// *actually scored* — `min(bank length, responses provided)` — not the
/// full bank length. A partial / interrupted run (fewer responses than
/// bank entries) reports an honest `N/M` instead of silently counting the
/// un-run challenges against the model.
pub fn score_responses(responses: &[String]) -> ReasoningScore {
    let total = responses.len().min(REASONING_BANK.len());
    let mut score = ReasoningScore {
        total,
        solved: 0,
        by_category: [(0, 0); 3],
    };
    for (challenge, response) in REASONING_BANK.iter().zip(responses) {
        let cat = match challenge.category {
            "math" => 0,
            "logic" => 1,
            _ => 2,
        };
        score.by_category[cat].1 += 1;
        if challenge.is_solved(response) {
            score.solved += 1;
            score.by_category[cat].0 += 1;
        }
    }
    score
}

/// The result of running the whole bank against a live endpoint: the
/// per-challenge responses, the §7 timing metrics alongside the
/// pass/fail score (blueprint: "a deterministic pass/fail accuracy
/// score alongside speed").
#[derive(Debug, Clone)]
pub struct ReasoningResult {
    /// One committed content span per bank entry (index-aligned; the
    /// chain-of-thought is excluded from the judged text — CF-16).
    pub responses: Vec<String>,
    /// Per-challenge TTFT, seconds (`T3 − T1`; the `0.0` N/A sentinel —
    /// CF-25 — when no token frame arrived).
    pub ttfts: Vec<f64>,
    /// Per-challenge **content-phase** decode speed, tokens/s (CF-49: the
    /// committed answer over the content window, not the blended
    /// think+write rate). **F2:** the numerator is the authoritative
    /// answer-phase count — `usage.completion_tokens − reasoning_tokens`
    /// when the server reports a `usage` block (consistent with Engine A's
    /// CF-26 split); the observed **content frame** tally only when no
    /// `usage` arrives (flagged by [`estimated`](Self::estimated)) — the
    /// frame count understates batched servers (vLLM MTP / speculative
    /// decoding) and is never presented as exact.
    pub tg_speeds: Vec<f64>,
    pub score: ReasoningScore,
    /// **F2:** `true` when any challenge's decode-speed count fell back to
    /// the content frame tally (the server reported no `usage` block) — an
    /// *estimated* number (batched frames undercount), never a measured one.
    pub estimated: bool,
}

impl ReasoningResult {
    /// Mean content-phase decode speed across challenges that produced a
    /// measurable committed answer (CF-49).
    ///
    /// H1: the `> 0.0` filter drops the CF-25 `0.0` N/A sentinels — a
    /// sub-10 ms content window (a 1–2-token answer in one MTP batch) is
    /// stored as `0.0` by [`answer_phase_speed`] and excluded here, so
    /// the average is a **real decode rate**, never a clamped-sliver
    /// phantom (the v0.1.8 log's indefensible ~799 t/s).
    pub fn avg_tg_speed(&self) -> f64 {
        let vals: Vec<f64> = self
            .tg_speeds
            .iter()
            .copied()
            .filter(|v| *v > 0.0)
            .collect();
        if vals.is_empty() {
            0.0
        } else {
            vals.iter().sum::<f64>() / vals.len() as f64
        }
    }
}

/// The live runner: one [`StreamWorker`] stream per bank challenge.
#[derive(Debug)]
pub struct ReasoningEngine {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    timeout: u64,
    /// Optional HF tokenizer for re-tokenizing a response when a proxy
    /// omits `usage` (the research's fallback counting method).
    tokenizer: Option<Arc<crate::prompt::Tokenizer>>,
    /// Optional sequence seam: publish `Challenge {n}/{total}` to the
    /// [`ProgressBus`] the Benchmark Sequence mirrors into the TUI.
    progress: Option<Arc<ProgressBus>>,
    /// Optional TUI seam: publish live single-stream
    /// [`MetricsState`] snapshots while a challenge runs, so the Live
    /// Monitor shows the current challenge's real stream data.
    metrics: Option<Arc<MetricsState>>,
    /// Optional `Space`-key pause gate: the bank waits on it before
    /// spawning each challenge's stream (in-flight challenges complete).
    pause: Option<Arc<RunPause>>,
    /// Optional run logger (each challenge's worker records its HTTP /
    /// SSE lifecycle; the engine records the bank start / score).
    logger: Option<Arc<crate::log::RunLogger>>,
}

impl ReasoningEngine {
    /// Build from the resolved config (same client/tokenizer policy as
    /// Engine A: an explicit `--tokenizer` that fails to load is a hard
    /// error; without it, counts fall back to `chars/4`).
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        // An explicit `--tokenizer` that fails to load is a hard error; it is
        // then stored for re-tokenizing responses when a proxy omits `usage`
        // (the research's fallback counting method).
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

    /// Attach a [`ProgressBus`] so the bank run reports `Challenge
    /// {n}/{total}` to the Benchmark Sequence (the TUI's progress bar).
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach a [`MetricsState`] publisher so each challenge pushes live
    /// single-stream snapshots to the TUI (the Live Monitor's stream
    /// matrix / gauges show the current challenge's real data).
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach the `Space`-key pause gate: each challenge waits on it
    /// before its stream is spawned (the headless path leaves it `None`).
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger (each challenge's stream then logs its
    /// HTTP / SSE lifecycle).
    pub fn logger(mut self, logger: Arc<crate::log::RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// Run one challenge: spawn a worker, drain its channel, and return
    /// the committed content span (the chain-of-thought is excluded —
    /// CF-16) plus the §7 timing deltas and the F2 `estimated` flag
    /// (`true` when the decode-speed count fell back to the content frame
    /// tally because the server reported no `usage` block).
    async fn run_challenge(&self, challenge: &Challenge) -> (String, f64, f64, bool) {
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            challenge.prompt,
            REASONING_MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        .tag(format!("C2:{}", challenge.id))
        // H2: the worker's `stream complete` line counts with the same
        // authoritative method (and tokenizer) as this engine's result.
        .tokenizer(self.tokenizer.clone());
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }
        // CF-16: suppress the model's chain-of-thought so the 512-token
        // budget goes to the committed answer. Without this, a reasoning
        // model spends the *entire* budget on CoT (`content = 0`,
        // `finish_reason = length`) and the checkers read a truncated
        // intermediate — a false failure of a reasoning capability the
        // model actually has (the C2 face of the C1 root cause, CF-01).
        // The suppression is per-backend and graceful (see
        // `StreamWorker::disable_thinking`), so non-reasoning models and
        // backends without a switch are unaffected.
        worker = worker.disable_thinking(true);

        // Spawn without awaiting: drain the channel concurrently to avoid
        // the >capacity deadlock (server-agnostic fix).
        let handle = spawn_worker(worker, tx);
        // CF-16: the committed answer only. The checkers judge the
        // model's *visible content span*, not its chain-of-thought:
        // reasoning deltas are deliberately NOT accumulated into
        // `response` (a value only in the CoT must not be credited, and a
        // truncated CoT must not supply the "last" number/verdict). The
        // CoT is still forwarded to the live snapshot / metrics.
        let mut response = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    if let Chunk::Content(text) = &frame.chunk {
                        response.push_str(text);
                    }
                }
            }
            // Publish a live snapshot every 8 events, and immediately on
            // a terminal event, so the TUI's Live Monitor shows the
            // current challenge's real stream data (measurement-isolation
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
                        "Reasoning",
                        &events,
                        REASONING_MAX_GEN_TOKENS,
                        self.tokenizer.as_deref(),
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
                "Reasoning",
                &events,
                REASONING_MAX_GEN_TOKENS,
                self.tokenizer.as_deref(),
            ));
        }

        // Channel closed: collect the outcome.
        let outcome = join_worker(handle).await;

        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
        let ttft = challenge_ttft(&ts);
        // CF-49 / F2: the decode speed is the **content (answer) phase**
        // rate — the committed answer's token count over the content window
        // (first → last content frame), NOT the blended think+write rate.
        // The request suppresses the chain-of-thought (CF-16), so in the
        // common case the content phase is the whole generation; when a
        // model emits CoT despite suppression, the thinking phase is
        // excluded from both the count and the window. The count is the
        // authoritative `completion − reasoning` when the server reports a
        // `usage` block (Engine A's CF-26 split); the content frame tally
        // — flagged `estimated` — only when it does not.
        let (tg_speed, estimated) =
            answer_phase_speed(&events, stream_time, ttft, outcome.usage, outcome.usage_details);
        (response, ttft, tg_speed, estimated)
    }

    /// Run the whole bank (sequential — one stream at a time, so each
    /// challenge measures the endpoint alone, no queueing cross-talk).
    ///
    /// While a [`ProgressBus`] is attached, each challenge publishes
    /// `Challenge {n}/{total}` — the Benchmark Sequence mirrors it into
    /// the TUI's progress bar.
    pub async fn run(&self) -> ReasoningResult {
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC2,
                format!(
                    "Engine C2 (Reasoning) started — {} challenges",
                    REASONING_BANK.len()
                ),
            );
        }
        let mut responses = Vec::with_capacity(REASONING_BANK.len());
        let mut ttfts = Vec::with_capacity(REASONING_BANK.len());
        let mut tg_speeds = Vec::with_capacity(REASONING_BANK.len());
        let mut estimated = false;
        for (i, challenge) in REASONING_BANK.iter().enumerate() {
            // The `Space`-key pause: hold before the next challenge's
            // request goes out (an in-flight challenge always completes).
            if let Some(gate) = &self.pause {
                gate.wait_while_paused().await;
            }
            if let Some(bus) = &self.progress {
                bus.publish(EngineProgress::Reasoning {
                    challenge: i + 1,
                    total: REASONING_BANK.len(),
                });
            }
            let (response, ttft, tg_speed, est) = self.run_challenge(challenge).await;
            responses.push(response);
            ttfts.push(ttft);
            tg_speeds.push(tg_speed);
            estimated |= est;
        }
        let result = ReasoningResult {
            score: score_responses(&responses),
            responses,
            ttfts,
            tg_speeds,
            estimated,
        };
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC2,
                format!(
                    "Engine C2 complete — {} (avg {:.1} t/s)",
                    result.score.label(),
                    result.avg_tg_speed()
                ),
            );
        }
        result
    }
}

// ── Extraction helpers (pure, deterministic) ─────────────────────────────

/// Iterator over every maximal numeric token in `text` (integers and
/// decimals, optionally signed), in order of appearance. A token is a run
/// of digits with an optional leading `-` and a single optional `.`-decimal
/// part; malformed fragments (a lone `-`, a trailing `.` with no digits)
/// are skipped.
struct Numbers<'a> {
    text: &'a str,
    bytes: &'a [u8],
    i: usize,
}

impl<'a> Numbers<'a> {
    fn new(text: &'a str) -> Self {
        Self {
            text,
            bytes: text.as_bytes(),
            i: 0,
        }
    }
}

impl<'a> Iterator for Numbers<'a> {
    type Item = f64;
    fn next(&mut self) -> Option<f64> {
        let n = self.bytes.len();
        loop {
            // Advance to the next number start (a digit, or a `-`
            // immediately followed by a digit).
            while self.i < n {
                let b = self.bytes[self.i];
                let is_start = b.is_ascii_digit()
                    || (b == b'-' && self.i + 1 < n && self.bytes[self.i + 1].is_ascii_digit());
                if is_start {
                    break;
                }
                self.i += 1;
            }
            if self.i >= n {
                return None;
            }
            let start = self.i;
            if self.bytes[self.i] == b'-' {
                self.i += 1;
            }
            while self.i < n && self.bytes[self.i].is_ascii_digit() {
                self.i += 1;
            }
            if self.i < n && self.bytes[self.i] == b'.' {
                self.i += 1;
                while self.i < n && self.bytes[self.i].is_ascii_digit() {
                    self.i += 1;
                }
            }
            if let Ok(v) = self.text[start..self.i].parse::<f64>() {
                return Some(v);
            }
            // A malformed fragment (e.g. "3." with no following digit) is
            // skipped: the loop keeps scanning from `self.i`.
        }
    }
}

/// The **answer-position** number in `text` — the *first* numeric token
/// (CF-33). The bank's math prompts ask for "the number only", so the
/// model's answer is the first number it commits; a trailing explanation
/// that restates *other* numbers ("1161, which is 27 × 43") must not
/// override it. `None` when the text contains no number.
pub fn extract_answer_number(text: &str) -> Option<f64> {
    Numbers::new(text).next()
}

/// `true` when `value` occurs in `text` as a whole number — i.e. some
/// maximal numeric token parses to `value` (within tolerance). CF-33:
/// accepts the expected value even when it is not the first token, so a
/// chatty-but-correct answer ("…the result is 1161") is not false-failed.
pub fn contains_number(text: &str, value: f64) -> bool {
    Numbers::new(text).any(|n| (n - value).abs() < 1e-6)
}
/// The **answer-position** whole-word yes/no verdict in `text` (CF-34) —
/// the *first* whole-word "yes" (`true`) or "no" (`false`),
/// case-insensitive. The bank's logic prompts ask for "yes or no only", so
/// the model's committed verdict is the first one it states; a trailing
/// elaboration that contradicts it ("No, but actually yes") must not
/// override the answer. `None` when neither occurs as a whole word (so
/// "none" / "normal" / "yeses" never match).
///
/// The runner feeds this the committed **content span** only (CF-16), so a
/// verdict that appears only in the chain-of-thought is out of scope by
/// construction — this reads the answer, not scattered reasoning.
pub fn extract_verdict(text: &str) -> Option<bool> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    while i < n {
        match bytes[i] {
            b'y' if lower[i..].starts_with("yes") && word_boundary(bytes, i, i + 3) => {
                return Some(true);
            }
            b'n' if lower[i..].starts_with("no") && word_boundary(bytes, i, i + 2) => {
                return Some(false);
            }
            _ => {}
        }
        i += 1;
    }
    None
}

/// `true` when the span `bytes[start..end]` is a whole word: the
/// preceding and following bytes (if any) are not ASCII letters or
/// digits (non-ASCII bytes count as boundaries).
fn word_boundary(bytes: &[u8], start: usize, end: usize) -> bool {
    let after_ok = end >= bytes.len() || !bytes[end].is_ascii_alphanumeric();
    let before_ok = start == 0 || !bytes[start - 1].is_ascii_alphanumeric();
    after_ok && before_ok
}

/// `true` when `marker` occurs in `code` as a whole word (bounded by
/// non-alphanumeric characters) — so `fn fib` matches `fn fib(n: u32)`
/// but **not** `fn fibonacci`.
pub fn contains_marker(code: &str, marker: &str) -> bool {
    let bytes = code.as_bytes();
    let mut search_from = 0usize;
    while let Some(rel) = code[search_from..].find(marker) {
        let start = search_from + rel;
        let end = start + marker.len();
        if word_boundary(bytes, start, end) {
            return true;
        }
        search_from = start + 1;
    }
    false
}
/// The content of the last fenced code block in `text`
/// (``` ... ```), `None` when no complete fence pair exists.
pub fn extract_code_fence(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.lines().collect();
    let mut fence_lines: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("```") {
            fence_lines.push(i);
        }
    }
    if fence_lines.len() < 2 {
        return None;
    }
    let open = fence_lines[fence_lines.len() - 2];
    let close = fence_lines[fence_lines.len() - 1];
    if close <= open + 1 {
        return None;
    }
    Some(lines[open + 1..close].join("\n"))
}

/// `true` when every `()` / `[]` / `{}` in `code` is balanced and
/// properly nested.
///
/// **CF-47:** string literals (`"…"` / `'…'`) and comments (`// …` /
/// `/* … */`) are *skipped*, so a delimiter inside a string or comment
/// (e.g. `let s = "unbalanced )";` or `// fix ( this`) no longer
/// false-fails a correct answer. The check remains structural (lexical) —
/// full AST parsing is a post-v1 extension; raw strings (`r"…"`) with
/// embedded quotes are a documented limitation.
pub fn delimiters_balanced(code: &str) -> bool {
    let bytes = code.as_bytes();
    let n = bytes.len();
    let mut stack: Vec<char> = Vec::new();
    let mut i = 0usize;
    while i < n {
        let b = bytes[i];
        match b {
            // `//` line comment → skip to (not including) the newline.
            b'/' if i + 1 < n && bytes[i + 1] == b'/' => {
                while i < n && bytes[i] != b'\n' {
                    i += 1;
                }
            }
            // `/*` block comment → skip to (and past) the closing `*/`.
            b'/' if i + 1 < n && bytes[i + 1] == b'*' => {
                i += 2;
                while i + 1 < n && !(bytes[i] == b'*' && bytes[i + 1] == b'/') {
                    i += 1;
                }
                i += 2;
            }
            // `"…"` string literal → skip to the unescaped closing quote.
            b'"' => {
                i += 1;
                while i < n {
                    if bytes[i] == b'\\' {
                        i += 2;
                    } else if bytes[i] == b'"' {
                        i += 1;
                        break;
                    } else {
                        i += 1;
                    }
                }
            }
            // `'…'` char literal (or a lifetime) → skip a real literal.
            b'\'' => {
                if i + 1 < n && bytes[i + 1] == b'\\' {
                    // escape form `'\n'` / `'\''` … → closing `'` at i+3
                    if i + 3 < n && bytes[i + 3] == b'\'' {
                        i += 4;
                    } else {
                        i += 1;
                    }
                } else if i + 2 < n && bytes[i + 2] == b'\'' {
                    // single-char form `'a'` → closing `'` at i+2
                    i += 3;
                } else {
                    // not a char literal → a lifetime / normal code
                    i += 1;
                }
            }
            b'(' | b'[' | b'{' => {
                stack.push(b as char);
                i += 1;
            }
            b')' => {
                if stack.pop() != Some('(') {
                    return false;
                }
                i += 1;
            }
            b']' => {
                if stack.pop() != Some('[') {
                    return false;
                }
                i += 1;
            }
            b'}' => {
                if stack.pop() != Some('{') {
                    return false;
                }
                i += 1;
            }
            _ => {
                i += 1;
            }
        }
    }
    stack.is_empty()
}

/// CF-49: the **(first, last) content token-arrival instants** and the
/// **content frame count** across a stream's events — the *answer-phase*
/// decode window and token count. It excludes the thinking phase (the
/// `Chunk::Reasoning` frames before the first content frame), so the
/// resulting rate is a clean *write* speed, not a blended think+write
/// rate. `(None, None, 0)` when no content frame arrived (a pure-CoT
/// stream, a failure, or an empty answer).
fn content_phase(
    events: &[StreamEvent],
) -> (Option<MonotonicInstant>, Option<MonotonicInstant>, u64) {
    let mut first: Option<MonotonicInstant> = None;
    let mut last: Option<MonotonicInstant> = None;
    let mut count = 0u64;
    for e in events {
        if let StreamEvent::Frame { frame, at, .. } = e {
            if matches!(frame.chunk, Chunk::Content(_)) {
                if first.is_none() {
                    first = Some(*at);
                }
                last = Some(*at);
                count += 1;
            }
        }
    }
    (first, last, count)
}

/// F6 (CF-25): one challenge's TTFT in seconds — `T3 − T1`, or the
/// **`0.0` N/A sentinel** when no token frame ever arrived (a usage-only
/// stream, a failure). Engine A's convention: the total stream time is
/// NOT a TTFT (the old fallback mislabeled total latency as first-token
/// time), and summary consumers filter `ttft > 0.0`.
fn challenge_ttft(ts: &StreamTimestamps) -> f64 {
    ts.ttft_nanos().map(|n| n as f64 / 1e9).unwrap_or(0.0)
}

/// F2: the **answer-phase token count** for one C2 challenge, with an
/// `estimated` flag.
///
/// The **Authoritative Counting Rule** (the C2 face — the same rule the
/// live snapshot and Engine A apply, `client/stream.rs`):
///
/// 1. **PRIMARY** — the server's `usage` block: `completion_tokens −
///    reasoning_tokens` (the CF-02 detail; `0` when the backend does not
///    report it). `completion_tokens` *includes* the chain-of-thought
///    tokens, so the subtraction is what makes this a *content*
///    (answer-phase) count — consistent with Engine A's CF-26 split.
/// 2. **FALLBACK** (no `usage` — a non-compliant proxy, an aborted stream)
///    — the observed **content frame** tally, flagged `estimated`: a
///    batched server (vLLM MTP / speculative decoding) carries several
///    tokens per frame, so the frame count understates the true answer
///    length (~2.4× on a vLLM-MTP server) and is never presented as
///    exact.
///
/// `estimated` is `true` only when the fallback produced a non-zero count
/// (an empty stream has no number to estimate).
fn content_count(
    usage: Option<Usage>,
    usage_details: Option<UsageDetails>,
    content_frames: u64,
) -> (u64, bool) {
    match usage
        .map(|u| u.completion_tokens)
        .filter(|&n| n > 0)
    {
        Some(completion) => {
            let reasoning = usage_details.map(|d| d.reasoning_tokens).unwrap_or(0);
            (completion.saturating_sub(reasoning), false)
        }
        None => (content_frames, content_frames > 0),
    }
}

/// F2: the published C2 **answer-phase decode speed** for one finished
/// challenge — content tokens over the content window (CF-49), where the
/// count is the authoritative [`content_count`]:
///
/// * `usage` reported → `completion − reasoning` (the same count the live
///   snapshot publishes via `SpeedEngine::single_stream_snapshot`, so the
///   Live Monitor and the published result agree — on a usage-reporting
///   server with thinking suppressed, the common C2 case, both read
///   `completion / (T_last_content − T_first_content)`);
/// * no `usage` → the content frame tally, flagged `estimated`.
///
/// `(0.0, false)` when there is no committed answer (pure CoT / failure /
/// empty content) — no meaningful content decode rate.
///
/// **H1 (the 10 ms floor):** a content window below
/// [`MIN_MEASURABLE_WINDOW_SECS`] is *not a measurable decode rate* — a
/// 1–2-token C2 answer committed in one MTP / speculative-decoding batch
/// arrives in a sub-millisecond burst (`first ≈ last`), and the old
/// `.max(0.001)` clamp turned that sliver into phantom 10³–10⁴ t/s
/// figures (the indefensible ~799 t/s "average" in the v0.1.8 log). Such
/// a challenge reports the CF-25 `0.0` N/A sentinel instead (its
/// `estimated` flag preserved), and [`ReasoningResult::avg_tg_speed`]
/// excludes it — the published average is a real decode rate, or nothing.
fn answer_phase_speed(
    events: &[StreamEvent],
    stream_time: f64,
    ttft: f64,
    usage: Option<Usage>,
    usage_details: Option<UsageDetails>,
) -> (f64, bool) {
    use crate::client::stream::MIN_MEASURABLE_WINDOW_SECS;
    let (c_first, c_last, content_frames) = content_phase(events);
    let (content_tokens, estimated) = content_count(usage, usage_details, content_frames);
    if content_tokens == 0 {
        return (0.0, false);
    }
    let content_time = match (c_first, c_last) {
        (Some(f), Some(l)) => f.delta_nanos(&l) as f64 / 1e9,
        // A single content frame (no span) → attribute the full
        // post-TTFT generation time to it.
        _ => stream_time - ttft,
    };
    // H1: a sub-floor window is not measurable — the N/A sentinel
    // stands (no clamped-divisor phantom).
    if content_time < MIN_MEASURABLE_WINDOW_SECS {
        return (0.0, estimated);
    }
    (content_tokens as f64 / content_time, estimated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::{StreamError, StreamEvent};
    use crate::sse::ParsedFrame;
    use crate::timing::{MonotonicInstant, StreamTimestamps};

    #[test]
    fn extract_answer_number_finds_the_answer() {
        // CF-33: the *first* (answer-position) number is extracted.
        assert_eq!(extract_answer_number("The answer is 1161."), Some(1161.0));
        assert_eq!(extract_answer_number("3.5"), Some(3.5));
        assert_eq!(extract_answer_number("no numbers here"), None);
        // The FIRST number is the answer position — a trailing restatement
        // of *other* numbers does not override it.
        assert_eq!(extract_answer_number("1161 then 999"), Some(1161.0));
        assert_eq!(extract_answer_number("1161 (27 × 43)"), Some(1161.0));
        assert_eq!(
            extract_answer_number("64. The sequence doubles each time."),
            Some(64.0)
        );
    }

    #[test]
    fn contains_number_accepts_the_expected_value_present_whole() {
        // CF-33: the expected value is accepted when present whole anywhere
        // in the span, not only at the answer position.
        assert!(contains_number("The result is 1161.", 1161.0));
        assert!(contains_number("1161, which is 27 × 43", 1161.0));
        // A value that is only a *substring* of a larger number does not
        // count (1161 is not in 21161 or 11610).
        assert!(!contains_number("The total is 21161.", 1161.0));
        assert!(!contains_number("11610 tokens", 1161.0));
        // A different value is not present.
        assert!(!contains_number("The answer is 999.", 1161.0));
    }

    #[test]
    fn extract_verdict_answer_position_wins() {
        // CF-34: the *first* (answer-position) whole-word verdict is
        // extracted.
        assert_eq!(extract_verdict("No, that's not valid."), Some(false));
        assert_eq!(extract_verdict("Yes"), Some(true));
        // The FIRST whole-word verdict is the committed answer — a trailing
        // flip in prose does not override it.
        assert_eq!(extract_verdict("yes... no"), Some(true));
        assert_eq!(extract_verdict("no, but yes"), Some(false));
        // Whole-word only: none / normal / yeses do not count.
        assert_eq!(extract_verdict("none of the above"), None);
        assert_eq!(extract_verdict("a normal day"), None);
    }

    #[test]
    fn delimiters_balance_check() {
        assert!(delimiters_balanced("fn f() { let x = (1 + 2) * [3]; }"));
        assert!(!delimiters_balanced("fn f() { let x = (1 + 2; }"));
        assert!(!delimiters_balanced("fn f() { let x = [1); }"));
        assert!(!delimiters_balanced("fn f() {"));
        assert!(delimiters_balanced("no delimiters at all"));
    }

    // ── CF-47: the balance check skips strings and comments ──────────────

    #[test]
    fn cf47_delimiters_skip_strings_and_comments() {
        // A `}` / `)` inside a string literal must not break the balance
        // check (the old lexical scan counted it and false-failed).
        let string = "fn f() -> String {\n    let s = \"unbalanced ) { here\";\n    s\n}";
        assert!(
            delimiters_balanced(string),
            "a delimiter inside a \"string\" literal is ignored"
        );
        // A `(` in a line comment must not break the balance check.
        let line_comment = "fn g() {\n    // fix ( this later\n}";
        assert!(
            delimiters_balanced(line_comment),
            "a ( inside a // comment is ignored"
        );
        // A `{` in a block comment must not break the balance check.
        let block_comment = "fn h() {\n    /* a { block comment */\n}";
        assert!(
            delimiters_balanced(block_comment),
            "a {{ inside a /* */ comment is ignored"
        );
        // A `}` in a char literal must not break the balance check.
        let char_literal = "fn c() {\n    let ch = '}';\n}";
        assert!(
            delimiters_balanced(char_literal),
            "a }} inside a 'char' literal is ignored"
        );
        // The canonical code answers must keep passing the balance check.
        for c in REASONING_BANK.iter().filter(|c| c.category == "code") {
            let code = extract_code_fence(c.canonical_answer).unwrap();
            assert!(
                delimiters_balanced(&code),
                "canonical {} must stay balanced",
                c.id
            );
        }
        // Genuine imbalance in real code (outside strings/comments) fails.
        assert!(!delimiters_balanced("fn f() { let x = (1 + 2; }"));
    }

    #[test]
    fn contains_marker_requires_word_boundaries() {
        assert!(contains_marker("fn fib(n: u32) -> u32 { n }", "fn fib"));
        // `fibonacci` must NOT satisfy the `fn fib` marker.
        assert!(!contains_marker(
            "fn fibonacci(n: u32) -> u32 { n }",
            "fn fib"
        ));
        assert!(contains_marker("x = 5;", "5"));
        assert!(!contains_marker("x = 50;", "5"));
    }

    #[test]
    fn extract_code_fence_last_complete_block() {
        let text =
            "Here you go:\n```rust\nfn a() {}\n```\nAnd again:\n```rust\nfn b() {}\n```\ndone";
        assert_eq!(extract_code_fence(text).as_deref(), Some("fn b() {}"));
        assert_eq!(extract_code_fence("no fence here"), None);
        assert_eq!(extract_code_fence("```rust\n```"), None);
    }

    #[test]
    fn canonical_answers_pass_their_own_checkers() {
        for c in REASONING_BANK {
            assert!(
                c.is_solved(c.canonical_answer),
                "canonical answer must solve {}",
                c.id
            );
        }
    }

    #[test]
    fn corrupted_answers_fail_their_checkers() {
        // Math: a wrong number fails.
        for c in REASONING_BANK.iter().filter(|c| c.category == "math") {
            let wrong = "The answer is 999999.".to_string();
            assert!(!c.is_solved(&wrong), "wrong number must fail {}", c.id);
        }
        // Logic: the opposite verdict fails.
        for c in REASONING_BANK
            .iter()
            .filter(|c| matches!(c.checker, Checker::Verdict { .. }))
        {
            let expected = match c.checker {
                Checker::Verdict { expected_yes } => expected_yes,
                _ => unreachable!(),
            };
            let wrong = if expected { "No" } else { "Yes" };
            assert!(!c.is_solved(wrong), "flipped verdict must fail {}", c.id);
        }
        // Code: unbalanced delimiters fail.
        let fib = REASONING_BANK.iter().find(|c| c.id == "code-fib").unwrap();
        let broken = "```rust\nfn fib(n: u32) -> u32 {\n    n\n```\n";
        assert!(!fib.is_solved(broken), "unbalanced block must fail");
        // Code: a missing required marker fails.
        let missing = "```rust\nfn fibonacci(n: u32) -> u32 {\n    n\n}\n```";
        assert!(!fib.is_solved(missing), "missing marker must fail");
        // No fence at all fails.
        assert!(!fib.is_solved("fn fib(n: u32) -> u32 { n }"));
    }

    // ── CF-16: the checkers judge the committed content span ────────────

    #[test]
    fn content_span_passes_where_the_combined_span_fails() {
        // Math (math-mul: 27 * 43 = 1161). A CoT that ends on a digit
        // fuses with the answer's first digit in the old reasoning+content
        // concatenation — "…running total 7" + "1161" reads as "71161", so
        // the *last* number is a corrupted intermediate (false-fail). The
        // committed content span alone is clean.
        let math = REASONING_BANK.iter().find(|c| c.id == "math-mul").unwrap();
        let cot = "Let me work it out step by step, keeping track of the running total 7";
        let content = "1161";
        assert!(
            math.is_solved(content),
            "the committed content span solves it"
        );
        assert!(
            !math.is_solved(&format!("{cot}{content}")),
            "the old concatenated span must fail (digit-boundary fusion)"
        );

        // Logic (logic-syllogism: expected "No"). A CoT that ends on the
        // word "no" fuses with the answer's "No" — "noNo" contains no
        // *whole-word* verdict, so the old last-verdict read yields
        // `None` (false-fail). The content span alone is a clean "No".
        let logic = REASONING_BANK
            .iter()
            .find(|c| c.id == "logic-syllogism")
            .unwrap();
        let cot2 = "All engineers are programmers, some programmers are designers, so no";
        let content2 = "No";
        assert!(
            logic.is_solved(content2),
            "the committed content span solves it"
        );
        assert!(
            !logic.is_solved(&format!("{cot2}{content2}")),
            "the old concatenated span must fail (word-boundary fusion)"
        );
    }

    #[test]
    fn cot_only_is_not_credited() {
        // The old false-pass: the model computes the answer inside its
        // chain-of-thought but commits nothing visible. When the CoT eats
        // the whole budget the content span is empty — judging it must NOT
        // credit the latent value.
        let math = REASONING_BANK.iter().find(|c| c.id == "math-mul").unwrap();
        assert!(
            !math.is_solved(""),
            "an empty committed span (CoT ate the budget) is not solved"
        );
        // And a CoT that states the value but commits no answer span:
        // the runner feeds the checkers the content span (empty here), so
        // the latent "1161" in the thought is out of scope by construction.
        assert!(
            extract_answer_number("I am fairly sure the answer is 1161, let me stop thinking")
                .is_some(),
            "the value is present in the CoT text itself"
        );
        assert!(!math.is_solved(""));
    }

    #[test]
    fn score_responses_counts_per_category() {
        // All canonical → 13/13.
        let all = REASONING_BANK
            .iter()
            .map(|c| c.canonical_answer.to_string())
            .collect::<Vec<_>>();
        let s = score_responses(&all);
        assert_eq!(s.total, 13);
        assert_eq!(s.solved, 13);
        assert_eq!(s.by_category, [(5, 5), (5, 5), (3, 3)]);
        assert_eq!(s.label(), "13/13 solved (100.0%)");

        // All garbage → 0/13.
        let none = vec!["I don't know.".to_string(); 13];
        let s = score_responses(&none);
        assert_eq!(s.solved, 0);
        assert_eq!(s.label(), "0/13 solved (0.0%)");
    }

    // ── CF-33: answer-position number + whole-value acceptance ───────────

    #[test]
    fn cf33_expected_number_early_and_restated_is_solved() {
        // CF-33: the expected number appears at the answer position
        // (first) AND is restated in a trailing explanation. The old
        // "last number wins" read the restated value and false-failed; the
        // answer-position read (plus whole-value acceptance) solves it.
        let math = REASONING_BANK.iter().find(|c| c.id == "math-mul").unwrap(); // 1161
        assert!(
            math.is_solved("1161 (27 × 43)"),
            "the answer-position number is the answer"
        );
        assert!(
            math.is_solved("The answer is 1161. To verify: 27 × 43 = 1161."),
            "the expected value present whole is accepted"
        );
        // A genuinely wrong answer (no 1161 anywhere) still fails.
        assert!(
            !math.is_solved("1160 (27 × 43)"),
            "a wrong number is not solved"
        );
        assert!(!math.is_solved("I don't know."), "no number is not solved");
    }

    // ── CF-34: the verdict is read at the answer position in content ─────

    #[test]
    fn cf34_verdict_reads_the_answer_position_in_content() {
        // CF-34: the committed verdict is the *first* whole-word yes/no in
        // the content span. A trailing elaboration that contradicts it must
        // not override the answer. (The runner feeds the checkers the
        // content span only — CF-16 — so scattered reasoning is excluded.)
        let syllogism = REASONING_BANK
            .iter()
            .find(|c| c.id == "logic-syllogism")
            .unwrap(); // expected: No
        let content = "No. (Although one could argue yes in some readings.)";
        assert_eq!(
            extract_verdict(content),
            Some(false),
            "the answer-position verdict is No"
        );
        assert!(
            syllogism.is_solved(content),
            "the committed No is judged, not the trailing yes"
        );
        assert_eq!(
            extract_verdict("No, I think. (Though one might argue yes.)"),
            Some(false),
            "a trailing flip does not override the committed verdict"
        );
    }

    #[test]
    fn cf34_content_verdict_wins_over_reasoning() {
        // CF-34: a verdict that appears only in the chain-of-thought must
        // not be judged — the runner feeds the checkers the committed
        // content span. Here the CoT says "yes" but the model commits "No".
        let knights = REASONING_BANK
            .iter()
            .find(|c| c.id == "logic-knights")
            .unwrap(); // expected: No
        let content = "No"; // the committed answer span
        let cot = "Hmm, can both be knights? Yes, if they were both lying…";
        assert_eq!(extract_verdict(content), Some(false));
        assert!(
            knights.is_solved(content),
            "the committed content verdict (No) is judged"
        );
        // Had the old combined span (CoT + content) been judged, the
        // scattered CoT verdict would be in play — the content span differs:
        assert_ne!(
            extract_verdict(&format!("{cot}{content}")),
            extract_verdict(content),
            "the CoT carries a different verdict than the committed content"
        );
    }

    // ── CF-48: the denominator is the number actually scored ─────────────

    #[test]
    fn cf48_partial_run_reports_an_honest_denominator() {
        // CF-48: a partial / interrupted run provides fewer responses than
        // the bank has challenges. The denominator is the number actually
        // scored, not the full bank length.
        let partial: Vec<String> = REASONING_BANK
            .iter()
            .take(5)
            .map(|c| c.canonical_answer.to_string())
            .collect();
        assert_eq!(partial.len(), 5);
        let s = score_responses(&partial);
        assert_eq!(s.total, 5, "total is the number actually scored");
        assert_eq!(s.solved, 5, "the 5 provided canonical answers all solve");
        assert_eq!(s.by_category, [(5, 5), (0, 0), (0, 0)]);
        assert_eq!(s.label(), "5/5 solved (100.0%)");

        // A partial run with a mix of solved / unsolved.
        let mixed: Vec<String> = vec![
            "1161".to_string(),          // math-mul: solved
            "I don't know.".to_string(), // math-frac: unsolved
        ];
        let s = score_responses(&mixed);
        assert_eq!(s.total, 2, "total is the 2 responses scored");
        assert_eq!(s.solved, 1);
        assert_eq!(s.label(), "1/2 solved (50.0%)");

        // An empty run reports 0/0 (not 0/13).
        let s = score_responses(&[]);
        assert_eq!(s.total, 0);
        assert_eq!(s.solved, 0);
        assert_eq!(s.label(), "0/0 solved (0.0%)");
    }

    // ── CF-49: the decode speed is the content (answer) phase only ───────

    #[test]
    fn cf49_content_phase_excludes_the_thinking_phase() {
        // CF-49: `content_phase` counts and windows the *content* frames
        // only — the reasoning (CoT) frames are excluded, so the decode
        // speed is a clean write rate, not a blended think+write rate.
        let now = MonotonicInstant::now();
        let frame = |chunk: Chunk| StreamEvent::Frame {
            frame: ParsedFrame {
                chunk,
                t_nanos: 0,
                done: false,
            },
            at: now,
            timestamps: StreamTimestamps::default(),
        };
        // 1 control + 2 reasoning (CoT) + 3 content (answer) frames.
        let events = vec![
            frame(Chunk::Control),
            frame(Chunk::Reasoning("thinking ".into())),
            frame(Chunk::Reasoning("more ".into())),
            frame(Chunk::Content("answ".into())),
            frame(Chunk::Content("ing ".into())),
            frame(Chunk::Content("now".into())),
        ];
        let (first, last, count) = content_phase(&events);
        // The 2 CoT frames are excluded: only the 3 content frames count.
        assert_eq!(count, 3, "content frames only — the CoT is not counted");
        // The window is anchored on a content frame (all frames share
        // `now` here, so the returned instants equal it).
        assert_eq!(first, Some(now), "the window starts on a content frame");
        assert_eq!(last, Some(now), "the window ends on a content frame");

        // A pure-CoT stream (no content frames) yields an empty window and
        // a zero count → no content decode speed (the 0.0 guard).
        let cot_only = vec![
            frame(Chunk::Control),
            frame(Chunk::Reasoning("still thinking ".into())),
        ];
        let (f, l, c) = content_phase(&cot_only);
        assert_eq!(c, 0, "no content frames → no content phase");
        assert!(f.is_none() && l.is_none());
    }

    #[test]
    fn stream_error_mapping_is_unused_here_but_worker_is_shared() {
        // Documentation test: the live runner shares Chunk 5's worker, so
        // the error surface stays identical across engines.
        let e = StreamError::Connection("refused".into());
        assert!(e.is_retriable());
    }

    // ── F2: the decode speed is the authoritative token count ────────────

    /// A content-only stream (3 frames, the first two sharing one instant,
    /// the third a short sleep later → a real, non-clamped content window)
    /// plus a terminal `Complete` carrying the optional `usage` block.
    fn f2_events(usage: Option<Usage>) -> Vec<StreamEvent> {
        let first = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(20));
        let last = MonotonicInstant::now();
        let frame = |at: MonotonicInstant, text: &str| StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content(text.into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: StreamTimestamps::default(),
        };
        let mut events = vec![frame(first, "a"), frame(first, "b"), frame(last, "c")];
        events.push(StreamEvent::Complete {
            timestamps: StreamTimestamps::default(),
            usage,
            premature: false,
            malformed_frames: 0,
            looping: false,
            loop_excluded_tokens: 0,
        });
        events
    }

    #[test]
    fn f2_content_count_is_authoritative_not_the_frame_tally() {
        // A batched server (vLLM MTP / speculative decoding): 3 content
        // frames carry 100 completion tokens, 20 of them chain-of-thought —
        // the answer phase is 80 tokens, not 3 frames.
        let (n, est) = content_count(
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 100,
            }),
            Some(UsageDetails {
                reasoning_tokens: 20,
                cached_tokens: 0,
            }),
            3,
        );
        assert_eq!((n, est), (80, false), "completion − reasoning, not estimated");

        // A `usage` block without the CF-02 detail field: the whole
        // completion count is the answer (thinking suppressed / a backend
        // that does not report the detail).
        let (n, est) = content_count(
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 100,
            }),
            None,
            3,
        );
        assert_eq!((n, est), (100, false));

        // A reported `0` completion is treated as absent (Engine A's
        // filter) → the frame tally, flagged estimated.
        let (n, est) = content_count(
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 0,
            }),
            None,
            3,
        );
        assert_eq!((n, est), (3, true));

        // No `usage` at all → the frame tally, flagged estimated; a stream
        // with no content frames has no number to estimate.
        assert_eq!(content_count(None, None, 3), (3, true));
        assert_eq!(content_count(None, None, 0), (0, false));
    }

    #[test]
    fn f2_published_speed_uses_the_authoritative_count() {
        let usage = Usage {
            prompt_tokens: 42,
            completion_tokens: 100,
        };
        let events = f2_events(Some(usage));
        let (speed, estimated) = answer_phase_speed(
            &events,
            0.0,
            0.0,
            Some(usage),
            Some(UsageDetails {
                reasoning_tokens: 20,
                cached_tokens: 0,
            }),
        );
        assert!(!estimated, "a usage-reported count is not an estimate");
        // 80 answer tokens over the content window — never the 3-frame
        // tally (which would read 3 / window, ~27× lower here).
        let (c_first, c_last, _) = content_phase(&events);
        let window = c_first.unwrap().delta_nanos(&c_last.unwrap()) as f64 / 1e9;
        assert!(
            (speed - 80.0 / window).abs() < 1e-9,
            "80 authoritative tokens over the content window: {speed}"
        );
    }

    #[test]
    fn f2_live_snapshot_and_published_speed_agree_on_a_usage_server() {
        // The common C2 case: a usage-reporting server with thinking
        // suppressed (`reasoning_tokens == 0`, no CoT frames on the wire).
        // The live panel (`single_stream_snapshot`) and the published
        // result must read the same number — before F2 the published
        // result was the raw frame tally and the two disagreed.
        let usage = Usage {
            prompt_tokens: 42,
            completion_tokens: 100,
        };
        let events = f2_events(Some(usage));
        let (published, estimated) = answer_phase_speed(
            &events,
            0.0,
            0.0,
            Some(usage),
            Some(UsageDetails {
                reasoning_tokens: 0,
                cached_tokens: 0,
            }),
        );
        assert!(!estimated);
        let live = SpeedEngine::single_stream_snapshot(
            "http://test",
            "test-model",
            "Reasoning",
            &events,
            REASONING_MAX_GEN_TOKENS,
            None,
        );
        assert!(
            (live.decode_throughput - published).abs() < 1e-9,
            "live {:.4} vs published {:.4} t/s must agree",
            live.decode_throughput,
            published
        );
    }

    #[test]
    fn f2_no_usage_falls_back_to_content_frames_flagged_estimated() {
        // A non-compliant proxy: no `usage` block → the count is the
        // observed content frame tally, flagged estimated (never exact).
        let events = f2_events(None);
        let (speed, estimated) = answer_phase_speed(&events, 0.0, 0.0, None, None);
        assert!(estimated, "the frame fallback must be flagged estimated");
        let (c_first, c_last, content_frames) = content_phase(&events);
        let window = c_first.unwrap().delta_nanos(&c_last.unwrap()) as f64 / 1e9;
        assert_eq!(content_frames, 3);
        assert!(
            (speed - 3.0 / window).abs() < 1e-9,
            "the frame tally over the content window: {speed}"
        );
    }

    // ── F6: the no-frame TTFT is the N/A sentinel, not total time ────────

    #[test]
    fn f6_no_frame_stream_ttft_is_the_na_sentinel() {
        // A stream that never delivered a token frame (a usage-only /
        // failed stream): TTFT is the `0.0` N/A sentinel — NOT the total
        // stream time (the CF-25 anti-pattern Engine A explicitly fixed).
        let t0 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(5));
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: None,
            t3: None,
            t_end: Some(MonotonicInstant::now()),
        };
        assert!(
            ts.total_nanos().unwrap() > 0,
            "the stream did take real total time"
        );
        assert_eq!(
            challenge_ttft(&ts),
            0.0,
            "no token frame → the 0.0 N/A sentinel, not the total"
        );
        // A summary that filters `ttft > 0.0` (Engine A's convention)
        // excludes it.
        let ttfts = [challenge_ttft(&ts), 0.3];
        let filtered: Vec<f64> = ttfts.iter().copied().filter(|t| *t > 0.0).collect();
        assert_eq!(filtered, vec![0.3], "the N/A sentinel is filtered out");
    }

    #[test]
    fn f6_token_frame_stream_ttft_is_t3_minus_t1() {
        // The measured case is unchanged: a stream with token frames
        // reports `T3 − T1`.
        let t0 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(10));
        let t3 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t3),
            t_end: Some(MonotonicInstant::now()),
        };
        let expected = t0.delta_nanos(&t3) as f64 / 1e9;
        let got = challenge_ttft(&ts);
        assert!((got - expected).abs() < 1e-9, "{got} vs {expected}");
        assert!(got > 0.0);
    }

    // ── H1: sub-floor content windows are N/A, never phantoms ─────────

    /// A content frame at `at` carrying `text`.
    fn h1_content_frame(at: MonotonicInstant, text: &str) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content(text.into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: StreamTimestamps::default(),
        }
    }

    #[test]
    fn h1_two_token_sub_ms_burst_is_na_not_a_phantom() {
        // The v0.1.8 fault: 2-token C2 answers arriving in one
        // sub-millisecond MTP batch — the old `.max(0.001)` clamp minted
        // 2 / 0.001 = 2000 t/s each (→ the indefensible ~799 t/s avg).
        // Now: the CF-25 `0.0` N/A sentinel (a sub-10 ms window is not a
        // measurable decode rate).
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let first = MonotonicInstant::now();
            mock.increment(1_000_000); // 1 ms later: second frame
            let last = MonotonicInstant::now();
            let events = vec![h1_content_frame(first, "No"), h1_content_frame(last, ".")];
            let (speed, estimated) = answer_phase_speed(
                &events,
                0.0,
                0.0,
                Some(Usage {
                    prompt_tokens: 42,
                    completion_tokens: 2,
                }),
                Some(UsageDetails {
                    reasoning_tokens: 0,
                    cached_tokens: 0,
                }),
            );
            assert_eq!(
                speed, 0.0,
                "a 1 ms content window is below the 10 ms floor → N/A, not 2000 t/s"
            );
            assert!(!estimated, "a usage-reported count is not an estimate");
        });
    }

    #[test]
    fn h1_single_frame_sub_floor_post_ttft_is_na() {
        // A single content frame (no span) attributes the post-TTFT
        // generation time to it; below the floor that is N/A too.
        let (clock, _mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let at = MonotonicInstant::now();
            let events = vec![h1_content_frame(at, "42")];
            // stream_time − ttft = 5 ms − 2 ms = 3 ms < 10 ms → N/A.
            let (speed, _) = answer_phase_speed(&events, 0.005, 0.002, None, None);
            assert_eq!(speed, 0.0, "a 3 ms post-TTFT window is below the floor → N/A");
        });
    }

    #[test]
    fn h1_normal_content_window_still_measures() {
        // A genuine 200 ms content window reads its real rate — the floor
        // must not mask a real (fast) decode.
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let first = MonotonicInstant::now();
            mock.increment(200_000_000); // 200 ms
            let last = MonotonicInstant::now();
            let events = vec![h1_content_frame(first, "a"), h1_content_frame(last, "b")];
            let (speed, estimated) = answer_phase_speed(
                &events,
                0.0,
                0.0,
                Some(Usage {
                    prompt_tokens: 42,
                    completion_tokens: 10,
                }),
                Some(UsageDetails {
                    reasoning_tokens: 0,
                    cached_tokens: 0,
                }),
            );
            assert!(!estimated);
            assert!(
                (speed - 10.0 / 0.2).abs() < 1e-9,
                "10 tokens over 200 ms = 50 t/s, got {speed}"
            );
        });
    }

    #[test]
    fn h1_avg_excludes_sub_floor_phantoms() {
        // The published average: [0.0 (sub-floor N/A), 40.0] == 40.0 —
        // never the old (2000 + 40) / 2 ≈ 1020 phantom blend. An all-N/A
        // bank reports 0.0 (nothing measurable), not a fake rate.
        let with_phantom = ReasoningResult {
            responses: vec![],
            ttfts: vec![],
            tg_speeds: vec![0.0, 40.0],
            score: ReasoningScore {
                total: 0,
                solved: 0,
                by_category: [(0, 0); 3],
            },
            estimated: false,
        };
        assert_eq!(with_phantom.avg_tg_speed(), 40.0);
        let all_na = ReasoningResult {
            responses: vec![],
            ttfts: vec![],
            tg_speeds: vec![0.0, 0.0],
            score: ReasoningScore {
                total: 0,
                solved: 0,
                by_category: [(0, 0); 3],
            },
            estimated: false,
        };
        assert_eq!(all_na.avg_tg_speed(), 0.0);
    }
}
