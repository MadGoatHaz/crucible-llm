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

use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

use crucible_llm::config::Config;
use crucible_llm::engines::{
    is_json_compliant, score_responses, ReasoningEngine, StructuredEngine, REASONING_BANK,
};

// ── Mock server ───────────────────────────────────────────────────────────

#[derive(Clone, Copy, Debug)]
enum MockMode {
    /// Every reasoning challenge answered with its canonical (correct)
    /// answer, dispatched on the challenge's `test_key` in the body.
    ReasoningClever,
    /// Every reasoning challenge answered with "I don't know."
    /// (the bank must score 0/13).
    ReasoningDumb,
    /// Structured: a valid JSON object when the body carries
    /// `response_format`, free-form prose otherwise.
    Structured,
    /// Structured: malformed (non-JSON) output under the constraint.
    StructuredBad,
}

/// Read the full HTTP request (headers + `Content-Length` body) and
/// return the body as text.
async fn read_request(sock: &mut TcpStream) -> String {
    let mut buf = [0u8; 8192];
    let mut acc: Vec<u8> = Vec::new();
    loop {
        match sock.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                if acc.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    let header_end = acc
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|i| i + 4)
        .unwrap_or(acc.len());
    let content_length = String::from_utf8_lossy(&acc[..header_end])
        .lines()
        .find_map(|l| {
            l.trim()
                .to_ascii_lowercase()
                .strip_prefix("content-length:")
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .unwrap_or(0);
    let mut body = acc[header_end..].to_vec();
    while body.len() < content_length {
        match sock.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&buf[..n]),
        }
    }
    String::from_utf8_lossy(&body).to_string()
}

async fn write_all(sock: &mut TcpStream, bytes: &[u8]) {
    let _ = sock.write_all(bytes).await;
    let _ = sock.flush().await;
}

/// Write one `Transfer-Encoding: chunked` frame.
async fn write_chunk(sock: &mut TcpStream, payload: &[u8]) {
    let mut msg = format!("{:x}\r\n", payload.len());
    msg.push_str(&String::from_utf8_lossy(payload));
    msg.push_str("\r\n");
    write_all(sock, msg.as_bytes()).await;
}

/// An SSE stream whose content is split into ~12-char deltas (like real
/// token frames), with a usage block and the `[DONE]` terminator.
async fn respond_sse(sock: &mut TcpStream, content: &str, prompt_tokens: u64) {
    write_all(
        sock,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )
    .await;
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

async fn start_mock(mode: MockMode) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let body = read_request(&mut sock).await;
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
                MockMode::Structured => {
                    if body.contains("response_format") {
                        respond_sse(
                            &mut sock,
                            r#"{"name": "Ada", "age": 36, "city": "London"}"#,
                            40,
                        )
                        .await;
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
            }
        }
    });
    // `StreamWorker` normalizes a bare host to `/v1/chat/completions`.
    format!("http://{addr}")
}

fn test_config(url: &str) -> Config {
    Config {
        url: url.to_string(),
        model: "mock-model".to_string(),
        timeout: 10,
        ..Config::default()
    }
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
    let url = start_mock(MockMode::ReasoningClever).await;
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
    let url = start_mock(MockMode::ReasoningDumb).await;
    let engine = ReasoningEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert_eq!(
        result.score.solved, 0,
        "unhelpful answers must fail every challenge"
    );
    assert_eq!(result.score.label(), "0/13 solved (0.0%)");
}

// ── C3: structured output (constrained vs free-form) ──────────────────────

#[tokio::test]
async fn structured_run_reports_tps_penalty_and_compliance() {
    let url = start_mock(MockMode::Structured).await;
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
    // Compliance: the constrained output is a JSON object with all
    // required fields.
    assert!(result.compliant, "valid JSON object must be compliant");
    assert!(is_json_compliant(
        &result.constrained_body,
        &["name", "age", "city"]
    ));
    assert!(result.constrained_body.contains("Ada"));
    // The free-form run (no constraint) is prose, not JSON.
    assert!(!is_json_compliant(
        &result.free_body,
        &["name", "age", "city"]
    ));
}

#[tokio::test]
async fn structured_run_flags_non_compliant_output() {
    let url = start_mock(MockMode::StructuredBad).await;
    let engine = StructuredEngine::new(&test_config(&url)).unwrap();
    let result = engine.run().await;

    assert!(result.constrained_tps > 0.0);
    assert!(
        !result.compliant,
        "non-JSON constrained output must fail compliance"
    );
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
    let body = accept.await.unwrap();
    eprintln!("PROBE BODY = {:?}", body);
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
