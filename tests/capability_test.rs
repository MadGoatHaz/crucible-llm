//! Chunk 16 — Engine C2 & C3 acceptance tests.
//!
//! Acceptance (plan Chunk 16):
//!
//! * the reasoning bank yields a **pass/fail accuracy score** (N/M
//!   solved), and a **known-correct response passes while a known-wrong
//!   one fails** — verified for every challenge in the bank, both through
//!   the pure checkers and through a live run against an in-process mock
//!   server (the same `tokio` harness the other engines use, fully
//!   offline);
//! * the structured runner reports **t/s under the grammar constraint vs
//!   the free-form baseline (penalty %)** and flags non-compliant
//!   output — again against the mock (valid JSON under
//!   `response_format`, malformed JSON in the second mode).

use std::sync::Arc;
use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpStream;
use tokio::sync::Mutex;

use crucible_llm::config::Config;
use crucible_llm::engines::{
    is_json_compliant, score_responses, ReasoningEngine, StructuredEngine, REASONING_BANK,
};

mod common;
use common::mock::{read_request, write_all, write_chunk, SSE_HEADERS};

// ── Mock server ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum MockMode {
    /// Every reasoning challenge answered with its canonical (correct)
    /// answer, dispatched on the challenge's `test_key` in the body.
    ReasoningClever,
    /// Every reasoning challenge answered with "I don't know."
    /// (the bank must score 0/13).
    ReasoningDumb,
    /// C2 reasoning model (CF-16): each challenge answered with a
    /// `reasoning_content` delta (crafted so the *old* reasoning+content
    /// concatenation false-fails: a trailing digit fuses with the answer
    /// for number checkers, a trailing "no"/"yes" fuses with the verdict
    /// for logic) followed by the canonical answer as `content` deltas.
    /// Content-only judging must score 13/13.
    ReasoningThenAnswers,
    /// Structured: a valid JSON object when the body carries
    /// `response_format`, free-form prose otherwise.
    Structured,
    /// Structured: malformed (non-JSON) output under the constraint.
    StructuredBad,
    /// C3 reasoning model (CF-18): a `reasoning_content` delta followed
    /// by the valid JSON answer as `content` deltas (or prose for the
    /// free-form run). Content-only accumulation must stay compliant —
    /// the CoT never leaks into the judged body.
    StructuredReasons,
}

/// An SSE stream whose content is split into ~12-char deltas (like real
/// token frames), with a usage block and the `[DONE]` terminator.
async fn respond_sse(sock: &mut TcpStream, content: &str, prompt_tokens: u64) {
    write_all(sock, SSE_HEADERS).await;
    // Every frame needs the blank-line separator: without it the parser
    // joins this `data:` line with the next frame's (multi-line data
    // join) and drops the merged blob as malformed JSON.
    write_chunk(
        sock,
        b"data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    )
    .await;
    let chars: Vec<char> = content.chars().collect();
    let mut completion: u64 = 0;
    for chunk in chars.chunks(12) {
        let delta: String = chunk.iter().collect();
        completion += delta.chars().count() as u64;
        let frame = format!(
            "data: {{\"id\":\"cmpl-1\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(&delta).unwrap()
        );
        write_chunk(sock, frame.as_bytes()).await;
    }
    let usage = format!(
        "data: {{\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion}}}}}\n\n"
    );
    write_chunk(sock, usage.as_bytes()).await;
    write_chunk(sock, b"data: [DONE]\n\n").await;
    write_chunk(sock, b"").await; // terminating chunk
    let _ = sock.shutdown().await;
}

/// An SSE stream whose `reasoning_content` deltas (the chain-of-thought)
/// come first, then whose `content` deltas carry the visible answer, with
/// a usage block and the `[DONE]` terminator. Mirrors the reasoning-model
/// wire shape the C1 tests use (`tests/niah_test.rs` `Mock::Reasons`).
async fn respond_sse_reasoning(
    sock: &mut TcpStream,
    reasoning: &str,
    content: &str,
    prompt_tokens: u64,
) {
    write_all(sock, SSE_HEADERS).await;
    write_chunk(
        sock,
        b"data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
    )
    .await;
    let mut completion: u64 = 0;
    let reason_chars: Vec<char> = reasoning.chars().collect();
    for chunk in reason_chars.chunks(12) {
        let delta: String = chunk.iter().collect();
        completion += delta.chars().count() as u64;
        let frame = format!(
            "data: {{\"id\":\"cmpl-1\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":{}}}}}]}}\n\n",
            serde_json::to_string(&delta).unwrap()
        );
        write_chunk(sock, frame.as_bytes()).await;
    }
    let content_chars: Vec<char> = content.chars().collect();
    for chunk in content_chars.chunks(12) {
        let delta: String = chunk.iter().collect();
        completion += delta.chars().count() as u64;
        let frame = format!(
            "data: {{\"id\":\"cmpl-1\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(&delta).unwrap()
        );
        write_chunk(sock, frame.as_bytes()).await;
    }
    let usage = format!(
        "data: {{\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion}}}}}\n\n"
    );
    write_chunk(sock, usage.as_bytes()).await;
    write_chunk(sock, b"data: [DONE]\n\n").await;
    write_chunk(sock, b"").await; // terminating chunk
    let _ = sock.shutdown().await;
}

/// Start the mock; returns the URL plus a log of every request body it
/// received (the CF-16/CF-18 tests assert the suppression fields on them).
async fn start_mock(mode: MockMode) -> (String, Arc<Mutex<Vec<String>>>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let bodies: Arc<Mutex<Vec<String>>> = Arc::new(Mutex::new(Vec::new()));
    let log = bodies.clone();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let (_, body) = read_request(&mut sock).await;
            log.lock().await.push(body.clone());
            match mode {
                MockMode::ReasoningClever => {
                    let answer = REASONING_BANK
                        .iter()
                        .find(|c| body.contains(c.test_key))
                        .map(|c| c.canonical_answer)
                        .unwrap_or("I don't know.");
                    respond_sse(&mut sock, answer, 50).await;
                }
                MockMode::ReasoningDumb => {
                    respond_sse(&mut sock, "I don't know.", 50).await;
                }
                MockMode::ReasoningThenAnswers => {
                    // CF-16: think first, then answer. The CoT is crafted
                    // so the *old* reasoning+content concatenation
                    // false-fails: for number checkers its trailing digit
                    // fuses with the answer's first digit ("…7" + "1161"
                    // → "71161"); for verdict checkers its trailing
                    // "no"/"yes" fuses with the answer word ("noNo"
                    // contains no whole-word verdict). The committed
                    // content span (the canonical answer) is clean —
                    // content-only judging must score 13/13.
                    if let Some(c) = REASONING_BANK.iter().find(|c| body.contains(c.test_key)) {
                        let cot = match (c.category, c.checker) {
                            ("math", _) => {
                                "Let me work it out step by step, keeping track of the running total 7"
                            }
                            (
                                "logic",
                                crucible_llm::engines::Checker::ExactNumber(_),
                            ) => "All four eggs share one pot of boiling water, so 5",
                            (
                                "logic",
                                crucible_llm::engines::Checker::Verdict {
                                    expected_yes: false,
                                },
                            ) => "Weighing the premises carefully, the conclusion does not follow, so no",
                            (
                                "logic",
                                crucible_llm::engines::Checker::Verdict {
                                    expected_yes: true,
                                },
                            ) => "Under the stated assumptions the statement holds, so yes",
                            ("code", _) => {
                                "Let me draft the signature and walk through the algorithm before writing it"
                            }
                            _ => unreachable!("bank categories are math / logic / code"),
                        };
                        respond_sse_reasoning(&mut sock, cot, c.canonical_answer, 50).await;
                    } else {
                        respond_sse(&mut sock, "I don't know.", 50).await;
                    }
                }
                MockMode::Structured => {
                    if body.contains("response_format") {
                        // Dispatch on which of the three C3 cases the prompt
                        // asks for, returning a *valid* answer for each shape.
                        let answer = if body.contains("exactly 3 objects") {
                            r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}, {"id": 3, "label": "three", "active": true}]"#
                        } else if body.contains("orders") {
                            r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": [{"id": 1, "total": 50.5, "items": ["a", "b"]}]}"#
                        } else {
                            r#"{"name": "Ada", "age": 36}"#
                        };
                        respond_sse(&mut sock, answer, 40).await;
                    } else {
                        respond_sse(
                            &mut sock,
                            "Sure! Here is the information you asked for: Ada, 36 years old, lives in London.",
                            40,
                        )
                        .await;
                    }
                }
                MockMode::StructuredBad => {
                    if body.contains("response_format") {
                        respond_sse(&mut sock, "this is definitely not json", 40).await;
                    } else {
                        respond_sse(
                            &mut sock,
                            "Sure! Here is the information you asked for: Ada, 36 years old, lives in London.",
                            40,
                        )
                        .await;
                    }
                }
                MockMode::StructuredReasons => {
                    // CF-18: think first, then emit the JSON in the
                    // content span. The content-only accumulation must
                    // parse it (the CoT never leaks into the judged body).
                    if body.contains("response_format") {
                        let (cot, answer) = if body.contains("exactly 3 objects") {
                            (
                                "Let me list exactly three objects with the required fields.",
                                r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}, {"id": 3, "label": "three", "active": true}]"#,
                            )
                        } else if body.contains("orders") {
                            (
                                "Let me build the nested object with the user and their orders.",
                                r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": [{"id": 1, "total": 50.5, "items": ["a", "b"]}]}"#,
                            )
                        } else {
                            (
                                "Let me recall the person's name and age.",
                                r#"{"name": "Ada", "age": 36}"#,
                            )
                        };
                        respond_sse_reasoning(&mut sock, cot, answer, 40).await;
                    } else {
                        respond_sse_reasoning(
                            &mut sock,
                            "Let me think about what to say before answering.",
                            "Sure! Here is the information you asked for: Ada, 36 years old, lives in London.",
                            40,
                        )
                        .await;
                    }
                }
            }
        }
    });
    // `StreamWorker` normalizes a bare host to `/v1/chat/completions`.
    (format!("http://{addr}"), bodies)
}

fn test_config(url: &str) -> Config {
    Config {
        url: url.to_string(),
        model: "mock-model".to_string(),
        timeout: 10,
        ..Config::default()
    }
}

/// Assert a recorded chat-completions body carries all three per-backend
/// thinking-suppression mechanisms (CF-01 / CF-16 / CF-18):
/// `chat_template_kwargs.enable_thinking=false` (vLLM / SGLang / LM
/// Studio), `think=false` (Ollama), and the trailing `/no_think` prompt
/// suffix (llama.cpp).
fn assert_body_suppresses_thinking(body: &str) {
    let v: serde_json::Value = serde_json::from_str(body).expect("mock bodies are JSON");
    assert_eq!(
        v["chat_template_kwargs"]["enable_thinking"], false,
        "vLLM / SGLang / LM Studio switch must be present"
    );
    assert_eq!(v["think"], false, "Ollama switch must be present");
    let prompt = v["messages"][0]["content"]
        .as_str()
        .expect("the user message content is a string");
    assert!(
        prompt.ends_with("/no_think"),
        "llama.cpp suffix must be appended: {prompt:?}"
    );
}

// ── C2: deterministic reasoning bank ──────────────────────────────────────

#[test]
fn bank_shape_is_stable() {
    assert_eq!(REASONING_BANK.len(), 13);
    let ids: Vec<&str> = REASONING_BANK.iter().map(|c| c.id).collect();
    assert_eq!(
        ids.len(),
        ids.iter().collect::<std::collections::HashSet<_>>().len()
    );
    let math = REASONING_BANK
        .iter()
        .filter(|c| c.category == "math")
        .count();
    let logic = REASONING_BANK
        .iter()
        .filter(|c| c.category == "logic")
        .count();
    let code = REASONING_BANK
        .iter()
        .filter(|c| c.category == "code")
        .count();
    assert_eq!((math, logic, code), (5, 5, 3));
}

#[test]
fn known_correct_passes_and_known_wrong_fails_for_every_challenge() {
    for c in REASONING_BANK {
        // The canonical answer must always pass its own strict checker.
        assert!(
            c.is_solved(c.canonical_answer),
            "canonical answer must solve {}",
            c.id
        );
        // A known-wrong answer must always fail.
        let wrong = match c.checker {
            crucible_llm::engines::Checker::ExactNumber(_) => "The answer is 999999.",
            crucible_llm::engines::Checker::Verdict { expected_yes } => {
                if expected_yes {
                    "No"
                } else {
                    "Yes"
                }
            }
            crucible_llm::engines::Checker::Code { .. } => {
                // Unbalanced delimiters inside a fence.
                "```rust\nfn fib(n: u32) -> u32 {\n    n\n```"
            }
        };
        assert!(!c.is_solved(wrong), "known-wrong answer must fail {}", c.id);
    }
}

#[test]
fn score_responses_reports_n_over_m_accuracy() {
    let all = REASONING_BANK
        .iter()
        .map(|c| c.canonical_answer.to_string())
        .collect::<Vec<_>>();
    let s = score_responses(&all);
    assert_eq!(s.total, 13);
    assert_eq!(s.solved, 13);
    assert_eq!(s.by_category, [(5, 5), (5, 5), (3, 3)]);
    assert_eq!(s.label(), "13/13 solved (100.0%)");

    let none = vec!["I don't know.".to_string(); 13];
    let s = score_responses(&none);
    assert_eq!(s.solved, 0);
    assert_eq!(s.label(), "0/13 solved (0.0%)");
}

// ── C2: live bank run against the mock (offline) ──────────────────────────

#[tokio::test]
async fn live_bank_run_against_clever_mock_scores_13_of_13() {
    let (url, _bodies) = start_mock(MockMode::ReasoningClever).await;
    let engine = ReasoningEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert_eq!(
        result.score.solved, 13,
        "clever mock must solve the whole bank"
    );
    assert_eq!(result.score.by_category, [(5, 5), (5, 5), (3, 3)]);
    // Speed alongside accuracy: every challenge produced tokens with a
    // measured TTFT and decode speed.
    assert_eq!(result.ttfts.len(), 13);
    assert!(result.ttfts.iter().all(|t| *t >= 0.0));
    assert!(result.tg_speeds.iter().all(|t| *t > 0.0));
    assert!(result.avg_tg_speed() > 0.0);
}

#[tokio::test]
async fn live_bank_run_against_dumb_mock_scores_0_of_13() {
    let (url, _bodies) = start_mock(MockMode::ReasoningDumb).await;
    let engine = ReasoningEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert_eq!(
        result.score.solved, 0,
        "unhelpful answers must fail every challenge"
    );
    assert_eq!(result.score.label(), "0/13 solved (0.0%)");
}

// ── C2: CF-16 reasoning-model suppression + content-only judging ────────

#[tokio::test]
async fn live_bank_run_against_reasoning_mock_scores_13_of_13() {
    // CF-16: the mock emits `reasoning` deltas (crafted to corrupt the
    // *old* reasoning+content concatenation — a trailing digit/verdict
    // fusing with the answer) and then `content` deltas carrying the
    // canonical answers. Content-only judging must credit the committed
    // answers: 13/13. Under the pre-CF-16 accumulation the fusing items
    // false-failed (see `reasoning.rs`
    // `content_span_passes_where_the_combined_span_fails`).
    let (url, _bodies) = start_mock(MockMode::ReasoningThenAnswers).await;
    let engine = ReasoningEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert_eq!(
        result.score.solved, 13,
        "reasoning-then-content must be judged on the content span (13/13)"
    );
    assert_eq!(result.score.by_category, [(5, 5), (5, 5), (3, 3)]);
    // The stored spans are the committed content (CoT excluded) and each
    // one solves its own challenge.
    for (c, r) in REASONING_BANK.iter().zip(&result.responses) {
        assert!(c.is_solved(r), "content span must solve {}", c.id);
    }
}

#[tokio::test]
async fn c2_requests_carry_thinking_suppression() {
    // CF-16: every C2 request carries the per-backend suppression fields
    // (the same `disable_thinking` seam C1 uses).
    let (url, bodies) = start_mock(MockMode::ReasoningThenAnswers).await;
    let engine = ReasoningEngine::new(&test_config(&url)).unwrap();
    let _ = engine.run().await;

    let bodies = bodies.lock().await;
    assert_eq!(bodies.len(), 13, "one request per bank challenge");
    for b in bodies.iter() {
        assert_body_suppresses_thinking(b);
    }
}

// ── C3: structured output (constrained vs free-form) ──────────────────────

#[tokio::test]
async fn structured_run_reports_tps_penalty_and_compliance() {
    let (url, _bodies) = start_mock(MockMode::Structured).await;
    let engine = StructuredEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    // Both runs measured (tokens produced, §7.2 decode speed).
    assert!(result.free_tps > 0.0, "free-form run must produce speed");
    assert!(
        result.constrained_tps > 0.0,
        "constrained run must produce speed"
    );
    // The penalty is a finite percentage (sign depends on the mock's
    // relative timing; it must be *computed*, not NaN/inf).
    assert!(result.penalty_pct.is_finite());
    // Compliance: every case's constrained output is valid JSON matching its
    // schema → all three compliant.
    assert!(
        result.fully_compliant(),
        "valid JSON for every case must be compliant"
    );
    assert_eq!(result.score(), (3, 0, 0));
    assert!(result.constrained_body.contains("Ada"));
    // The free-form run (no constraint) is prose, not JSON.
    assert!(!is_json_compliant(&result.free_body, &["name", "age"]));
}

#[tokio::test]
async fn structured_run_flags_non_compliant_output() {
    let (url, _bodies) = start_mock(MockMode::StructuredBad).await;
    let engine = StructuredEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert!(result.constrained_tps > 0.0);
    assert!(
        !result.fully_compliant(),
        "non-JSON constrained output must fail compliance"
    );
    assert_eq!(result.score(), (0, 0, 3));
}

// ── C3: CF-18 reasoning-model suppression ───────────────────────────────

#[tokio::test]
async fn structured_run_with_reasoning_deltas_stays_compliant() {
    // CF-18: the mock emits a `reasoning` delta (CoT) and then the JSON
    // answer in the `content` span. The content-only accumulation must
    // parse the JSON — the CoT never leaks into the judged body, so a
    // reasoning model that thinks then answers stays fully compliant
    // (pre-CF-18 it spent the 256-token budget on CoT and emitted empty
    // content → "not valid JSON").
    let (url, _bodies) = start_mock(MockMode::StructuredReasons).await;
    let engine = StructuredEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert!(
        result.fully_compliant(),
        "content-only accumulation must stay compliant with reasoning deltas present"
    );
    assert_eq!(result.score(), (3, 0, 0));
}

#[tokio::test]
async fn c3_requests_carry_thinking_suppression() {
    // CF-18: every C3 request (free-form baseline + 3 constrained cases)
    // carries the per-backend suppression fields.
    let (url, bodies) = start_mock(MockMode::StructuredReasons).await;
    let engine = StructuredEngine::new(&test_config(&url)).unwrap();
    let _ = engine.run().await;

    let bodies = bodies.lock().await;
    assert_eq!(bodies.len(), 4, "free-form baseline + 3 constrained cases");
    for b in bodies.iter() {
        assert_body_suppresses_thinking(b);
    }
}

#[tokio::test]
async fn read_request_captures_full_body() {
    use std::io::Write as _;
    use std::net::TcpStream as StdTcpStream;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let accept = tokio::spawn(async move {
        let (mut sock, _) = listener.accept().await.unwrap();
        read_request(&mut sock).await
    });
    let mut client = StdTcpStream::connect(addr).unwrap();
    client
        .write_all(b"POST /v1 HTTP/1.1\r\nHost: x\r\nContent-Length: 11\r\n\r\nhello world")
        .unwrap();
    let body = accept.await.unwrap().1;
    assert_eq!(body, "hello world");
}

// ── keep the harness imports honest ───────────────────────────────────────

#[tokio::test]
async fn mock_server_uses_the_same_worker_path() {
    // The mock only exercises the Chunk 5 worker via the engines above;
    // this test pins that a bare host normalizes to the completions path
    // (the assumption `start_mock` relies on).
    use crucible_llm::client::normalize_endpoint;
    assert_eq!(
        normalize_endpoint("http://127.0.0.1:1234"),
        "http://127.0.0.1:1234/v1/chat/completions"
    );
    // The timeout field is honored (10s in test_config).
    assert_eq!(test_config("http://x").timeout, 10);
    assert!(Duration::from_secs(10) > Duration::from_secs(1));
}
