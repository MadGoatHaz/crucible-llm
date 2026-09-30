//! Engine C2 — Deterministic Reasoning & Code Verification (plan Chunk 16,
//! blueprint §5 Engine C2).
//!
//! A standardized bank of self-contained logic / math / code-generation
//! challenges. Every challenge is validated by a **strict, deterministic
//! checker** — no LLM-as-judge, no fuzzy scoring — so the resulting
//! pass/fail accuracy (`N/M solved`) is reproducible run-to-run and
//! comparable across models, quantizations, and server versions:
//!
//! * **Math** — the *last* numeric value in the response must equal the
//!   exact expected value (models are prompted to answer with just the
//!   number, so the last number is the answer).
//! * **Logic** — the *last* yes/no verdict (case-insensitive, whole-word)
//!   must match the expected verdict.
//! * **Code** — the response must contain a fenced code block; the block
//!   must have balanced, properly nested delimiters (`()` `[]` `{}`) and
//!   contain every required structural marker (the function signature,
//!   key identifiers).
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

use crate::client::{StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::prompt::PromptGenerator;
use crate::sse::Chunk;
use crate::timing::{MonotonicInstant, StreamTimestamps};

/// Bounded worker→engine channel capacity (same as Engine A).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per challenge (code answers can be long).
pub const REASONING_MAX_GEN_TOKENS: u32 = 512;

/// The strict, deterministic checker for one challenge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Checker {
    /// The last number in the response must equal `value`.
    ExactNumber(f64),
    /// The last whole-word yes/no verdict must be `expected_yes`.
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
            Checker::ExactNumber(v) => {
                extract_last_number(response).is_some_and(|got| (got - v).abs() < 1e-6)
            }
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

/// The deterministic accuracy score over the bank.
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
pub fn score_responses(responses: &[String]) -> ReasoningScore {
    let mut score = ReasoningScore {
        total: REASONING_BANK.len(),
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
    /// One response per bank entry (index-aligned).
    pub responses: Vec<String>,
    /// Per-challenge TTFT, seconds (`T3 − T1`).
    pub ttfts: Vec<f64>,
    /// Per-challenge decode speed, tokens/s (§7.2 formula).
    pub tg_speeds: Vec<f64>,
    pub score: ReasoningScore,
}

impl ReasoningResult {
    /// Mean decode speed across challenges that produced tokens.
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
    #[allow(dead_code)]
    generator: PromptGenerator,
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
        let generator = match &cfg.tokenizer {
            Some(path) => PromptGenerator::new(Some(crate::prompt::Tokenizer::from_file(path)?)),
            None => PromptGenerator::new(None),
        };
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            generator,
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
    /// the concatenated response text plus the §7 timing deltas.
    async fn run_challenge(&self, challenge: &Challenge) -> (String, f64, f64) {
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            challenge.prompt,
            REASONING_MAX_GEN_TOKENS,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        .tag(format!("C2:{}", challenge.id));
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }

        let start = MonotonicInstant::now();
        let outcome = tokio::spawn(worker.run(tx))
            .await
            .expect("reasoning worker task panicked");
        let mut response = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    if let Chunk::Content(text) | Chunk::Reasoning(text) = &frame.chunk {
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
                        &start,
                        REASONING_MAX_GEN_TOKENS,
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
                &start,
                REASONING_MAX_GEN_TOKENS,
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
            .unwrap_or_else(|| (response.chars().count() / 4).max(1) as u64);
        let generation_time = (stream_time - ttft).max(0.001);
        let tg_speed = completion as f64 / generation_time;
        (response, ttft, tg_speed)
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
            let (response, ttft, tg_speed) = self.run_challenge(challenge).await;
            responses.push(response);
            ttfts.push(ttft);
            tg_speeds.push(tg_speed);
        }
        let result = ReasoningResult {
            score: score_responses(&responses),
            responses,
            ttfts,
            tg_speeds,
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

/// The last numeric value in `text` (integers and decimals), `None` when
/// the text contains no number.
///
/// The bank's math prompts ask for "the number only", so the *last* number
/// in the response is the model's answer (prose like "the answer is N"
/// yields N; a trailing explanation containing numbers would be a
/// genuine model failure — strict by design).
pub fn extract_last_number(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let mut last: Option<f64> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        let is_number_start = bytes[i].is_ascii_digit()
            || (bytes[i] == b'-' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit());
        if !is_number_start {
            i += 1;
            continue;
        }
        let start = i;
        if bytes[i] == b'-' {
            i += 1;
        }
        while i < bytes.len() && bytes[i].is_ascii_digit() {
            i += 1;
        }
        if i < bytes.len() && bytes[i] == b'.' {
            i += 1;
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                i += 1;
            }
        }
        if let Ok(v) = text[start..i].parse::<f64>() {
            last = Some(v);
        }
    }
    last
}
/// The last whole-word yes/no verdict in `text` (case-insensitive):
/// `true` for "yes", `false` for "no", `None` when neither occurs as a
/// whole word (so "none" / "normal" / "yeses" never match).
pub fn extract_verdict(text: &str) -> Option<bool> {
    let lower = text.to_ascii_lowercase();
    let bytes = lower.as_bytes();
    let mut last: Option<bool> = None;
    let mut i = 0usize;
    while i < bytes.len() {
        match bytes[i] {
            b'y' if lower[i..].starts_with("yes") && word_boundary(bytes, i, i + 3) => {
                last = Some(true);
                i += 3;
                continue;
            }
            b'n' if lower[i..].starts_with("no") && word_boundary(bytes, i, i + 2) => {
                last = Some(false);
                i += 2;
                continue;
            }
            _ => {}
        }
        i += 1;
    }
    last
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
/// Structural (lexical) check — string/comment contents are not
/// excluded (a documented approximation; the required-signature markers
/// plus balanced delimiters are what gate a code answer).
pub fn delimiters_balanced(code: &str) -> bool {
    let mut stack: Vec<char> = Vec::new();
    for c in code.chars() {
        match c {
            '(' | '[' | '{' => stack.push(c),
            ')' if stack.pop() != Some('(') => return false,
            ']' if stack.pop() != Some('[') => return false,
            '}' if stack.pop() != Some('{') => return false,
            _ => {}
        }
    }
    stack.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::client::StreamError;

    #[test]
    fn extract_last_number_finds_the_answer() {
        assert_eq!(extract_last_number("The answer is 1161."), Some(1161.0));
        assert_eq!(extract_last_number("2, 4, 8, 16, 32, 64"), Some(64.0));
        assert_eq!(extract_last_number("3.5"), Some(3.5));
        assert_eq!(extract_last_number("no numbers here"), None);
        // The LAST number wins (a trailing "999" corrupts the answer).
        assert_eq!(extract_last_number("1161 then 999"), Some(999.0));
    }

    #[test]
    fn extract_verdict_last_whole_word_wins() {
        assert_eq!(extract_verdict("No, that's not valid."), Some(false));
        assert_eq!(extract_verdict("Yes"), Some(true));
        // Last occurrence wins.
        assert_eq!(extract_verdict("yes... no"), Some(false));
        assert_eq!(extract_verdict("no, but yes"), Some(true));
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

    #[test]
    fn stream_error_mapping_is_unused_here_but_worker_is_shared() {
        // Documentation test: the live runner shares Chunk 5's worker, so
        // the error surface stays identical across engines.
        let e = StreamError::Connection("refused".into());
        assert!(e.is_retriable());
    }
}
