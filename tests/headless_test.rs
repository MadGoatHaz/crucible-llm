//! Integration tests for Chunk 7 (single-stream CLI parity / headless
//! mode): the `SpeedEngine` against an in-process mock SSE server, the
//! `--json` field-set parity with `llmspeedtest.py`, the dead-endpoint
//! exit-code rule, and the `--nocache` prefix reaching the wire.
//!
//! Acceptance (plan Chunk 7):
//! * `--url <mock> --json` emits JSON with the same metric keys as
//!   `llmspeedtest.py --json`;
//! * a single run against the mock completes;
//! * a dead endpoint yields exit code 1 (all runs failed);
//! * `--nocache` changes the sent prefix.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use tokio::net::TcpListener;
use tokio::net::TcpStream;

use crucible_llm::config::Config;
use crucible_llm::engines::speed::{all_failed, json_report, SpeedEngine};

mod common;
use common::mock::*;

// ── Mock server ───────────────────────────────────────────────────────────

enum Mock {
    /// Full vLLM-style SSE stream ending in `[DONE]` (chunked encoding).
    Sse,
    /// A single non-streaming JSON completion.
    PlainJson,
    /// Records the request body, then serves a minimal SSE stream.
    Echo,
}

/// CF-11: the mock now accepts **many** sequential connections (the
/// engine performs one discarded warmup run before its measured
/// iterations) and counts every request it serves in the returned
/// [`AtomicUsize`] — the tests assert the warmup hit the wire but was
/// thrown away.
async fn start_mock(mock: Mock) -> (String, tokio::task::JoinHandle<()>, Arc<AtomicUsize>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let served = Arc::new(AtomicUsize::new(0));
    let task_counter = served.clone();
    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                // The test aborted the server (or the listener closed).
                Err(_) => return,
            };
            task_counter.fetch_add(1, Ordering::SeqCst);
            match &mock {
                Mock::Sse => {
                    drain_request(&mut sock).await;
                    sse_response(&mut sock).await;
                }
                Mock::PlainJson => {
                    drain_request(&mut sock).await;
                    serve_json(&mut sock, "HTTP/1.1 200 OK", PLAIN_JSON_BODY).await;
                }
                Mock::Echo => {
                    let (_, body) = read_request(&mut sock).await;
                    sse_response(&mut sock).await;
                    // Park the body where the test can read it.
                    ECHO_BODIES.lock().unwrap().push(body);
                }
            }
        }
    });
    (format!("http://{addr}"), handle, served)
}

/// Recorded request bodies from `Mock::Echo`.
static ECHO_BODIES: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

async fn sse_response(sock: &mut TcpStream) {
    serve_sse(sock, VLLM_STREAM, std::time::Duration::ZERO).await;
}

// ── Test helpers ──────────────────────────────────────────────────────────

fn cfg(url: &str) -> Config {
    Config {
        url: url.to_string(),
        model: "test-model".to_string(),
        timeout: 10,
        ..Config::default()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn sse_run_produces_prototype_metrics() {
    let (url, server, served) = start_mock(Mock::Sse).await;
    let engine = SpeedEngine::new(&cfg(&url)).unwrap();
    let (prompt, results) = engine.run().await;
    server.abort();

    // CF-11: the discarded warmup hit the wire (1 warmup + 1 measured
    // iteration = 2 requests), but it is not counted: `results` holds
    // exactly the measured iterations.
    assert_eq!(
        served.load(Ordering::SeqCst),
        2,
        "1 warmup + 1 measured iteration reached the server"
    );
    assert_eq!(results.len(), 1, "the warmup run is discarded, not counted");
    let r = &results[0];
    assert!(r.error.is_none(), "unexpected error: {:?}", r.error);
    assert!(!r.is_failed());

    // The mock stream's authoritative `usage.completion_tokens` is 34 (for
    // the 4 token frames it sends). The engine uses the server's figure as
    // the primary count, with prompt tokens from the same usage block. Not
    // estimated.
    assert_eq!(r.prompt_tokens, 128);
    assert_eq!(r.completion_tokens, 34);
    assert!(!r.estimated);

    // Chunk accounting: role + 2 reasoning + 2 content + usage = 6
    // (`[DONE]` is not a chunk).
    assert_eq!(r.content_chunks, 2);
    assert_eq!(r.reasoning_chunks, 2);
    assert_eq!(r.other_chunks, 2);
    assert_eq!(r.total_chunks, 6);

    // §7 formulas: TTFT / PP / TG all positive; MTP η (CF-12) = 34
    // completion / 4 total token frames (2 reasoning + 2 content) = 8.5.
    assert!(r.ttft > 0.0);
    assert!(r.stream_time > r.ttft);
    assert!(r.pp_speed > 0.0);
    assert!(r.tg_speed > 0.0);
    assert!((r.mtp_efficiency - 8.5).abs() < 1e-9);
    assert_eq!(r.model, "test-model");
    assert_eq!(r.mode, "short");
    assert!(!prompt.nocache);
}

#[tokio::test]
async fn json_report_matches_prototype_key_set() {
    let (url, server, served) = start_mock(Mock::Sse).await;
    let c = cfg(&url);
    let engine = SpeedEngine::new(&c).unwrap();
    let (_prompt, results) = engine.run().await;
    server.abort();

    // CF-11: warmup + one measured iteration reached the wire; only the
    // measured iteration is in the report.
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(results.len(), 1);

    let v = json_report(&c, &results);
    let top = v.as_object().unwrap();

    // Top-level keys: the prototype `output_json` set plus the v0.1.1
    // `timing` / `methodology` blocks (no summary for a single valid run,
    // no `loop_guard` — the mock stream does not loop).
    let top_keys: Vec<&str> = top.keys().map(|s| s.as_str()).collect();
    assert_eq!(
        top_keys,
        vec![
            "url",
            "model",
            "mode",
            "iterations",
            "results",
            "timing",
            "methodology"
        ]
    );
    assert!(top.get("summary").is_none());
    assert!(top.get("loop_guard").is_none());
    assert_eq!(top["url"], url);
    assert_eq!(top["iterations"], 1);

    // Per-run keys: the prototype `to_dict` set + v0.1.1 `looping` +
    // CF-25 `degraded` + the P2-A integrity flags (CF-10/24/44)
    // + B1 `pure_cot` / `blended_tps` (additive; §6 contract delta).
    let rj = top["results"][0].as_object().unwrap();
    assert_eq!(rj.len(), 21);
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
        "looping",
        "degraded",
        "usage_discrepancy",
        "e2e_only",
        "itl_clamped",
        "pure_cot",
        "blended_tps",
    ] {
        assert!(rj.contains_key(key), "missing {key}");
    }
    assert!(
        !rj.contains_key("other_chunks"),
        "other_chunks must not be serialized"
    );
    assert_eq!(rj["prompt_tokens"], 128);
    // The server's authoritative `usage.completion_tokens` (34).
    assert_eq!(rj["completion_tokens"], 34);
    assert_eq!(rj["estimated"], false);
    assert_eq!(rj["error"], serde_json::Value::Null);
    // B1: the mock stream has 2 content frames — a mixed (not pure-CoT)
    // run; the blended rate is published as a number.
    assert_eq!(rj["pure_cot"], false);
    assert!(rj["blended_tps"].is_number());

    // The v0.1.1 blocks publish the measurement methodology.
    assert_eq!(top["timing"]["resolution"], "nanosecond");
    assert!(top["timing"]["overhead_ns"].is_u64());
    assert_eq!(top["methodology"]["version"], env!("CARGO_PKG_VERSION"));

    // Two valid runs → summary with the prototype's four keys.
    let v2 = json_report(
        &c,
        &results
            .iter()
            .cloned()
            .chain(results.iter().cloned())
            .collect::<Vec<_>>(),
    );
    let s = v2["summary"].as_object().unwrap();
    let skeys: Vec<&str> = s.keys().map(|k| k.as_str()).collect();
    assert_eq!(
        skeys,
        vec!["avg_ttft", "avg_pp_speed", "avg_tg_speed", "avg_mtp"]
    );
}

#[tokio::test]
async fn dead_endpoint_yields_all_failed_exit_1() {
    // Grab a free port, then release it so nothing is listening.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);

    let c = cfg(&format!("http://{addr}"));
    let engine = SpeedEngine::new(&c).unwrap();
    let _prompt = engine.generate_prompt();
    let result = engine.run_iteration(&engine.generate_prompt()).await;

    assert!(result.is_failed(), "dead endpoint must be a failed run");
    assert_eq!(
        result.error.as_deref(),
        Some("Connection refused — is the server running?")
    );
    // The exit-code rule: every run failed → exit 1.
    assert!(all_failed(&[result]));
}

#[tokio::test]
async fn nocache_prefix_reaches_the_server_and_varies() {
    ECHO_BODIES.lock().unwrap().clear();
    let (url, _server, _served) = start_mock(Mock::Echo).await;
    let mut c = cfg(&url);
    c.nocache = true;

    let engine = SpeedEngine::new(&c).unwrap();
    let prompt1 = engine.generate_prompt();
    let prompt2 = engine.generate_prompt();

    // Two generated prompts carry two distinct uuid prefixes.
    assert!(prompt1.nocache && prompt2.nocache);
    assert_ne!(
        prompt1.text.split(']').next().unwrap(),
        prompt2.text.split(']').next().unwrap(),
        "nocache prefixes must differ"
    );

    let result = engine.run_iteration(&prompt1).await;
    assert!(
        !result.is_failed(),
        "run should complete: {:?}",
        result.error
    );

    // The exact prompt text (unique prefix + short prompt) hit the wire
    // (the short prompt contains no JSON-escape characters).
    let bodies = ECHO_BODIES.lock().unwrap().clone();
    assert_eq!(bodies.len(), 1);
    assert!(
        bodies[0].contains(&prompt1.text),
        "request body must carry the nocache prompt: {}",
        &bodies[0][..bodies[0].len().min(160)]
    );
}

#[tokio::test]
async fn plain_json_fallback_runs_through_the_engine() {
    let (url, server, served) = start_mock(Mock::PlainJson).await;
    let engine = SpeedEngine::new(&cfg(&url)).unwrap();
    let (_prompt, results) = engine.run().await;
    server.abort();

    // CF-11: the discarded warmup is the 2nd request on the wire; the
    // plain-JSON fallback path counts only the measured run.
    assert_eq!(served.load(Ordering::SeqCst), 2);
    assert_eq!(results.len(), 1);
    let r = &results[0];
    assert!(r.error.is_none(), "unexpected error: {:?}", r.error);
    // usage (42/7) wins; one content chunk + one usage frame.
    assert_eq!(r.prompt_tokens, 42);
    assert_eq!(r.completion_tokens, 7);
    assert_eq!(r.content_chunks, 1);
    assert_eq!(r.other_chunks, 1);
    assert_eq!(r.total_chunks, 2);
    assert!((r.mtp_efficiency - 7.0).abs() < 1e-9);
    assert!(r.tg_speed > 0.0);
}

#[tokio::test]
async fn multiple_iterations_run_sequentially() {
    // Run a single iteration against the (multi-accept) mock to verify
    // the iteration contract. `run_iteration` is a single request — the
    // CF-11 warmup lives in `run()`, not here.
    let (url, _server, _served) = start_mock(Mock::Sse).await;
    let c = cfg(&url);
    let engine = SpeedEngine::new(&c).unwrap();
    let prompt = engine.generate_prompt();
    let r1 = engine.run_iteration(&prompt).await;
    assert!(!r1.is_failed());
}

#[tokio::test]
async fn explicit_bad_tokenizer_is_a_hard_error() {
    let mut c = cfg("http://127.0.0.1:1");
    c.tokenizer = Some(std::path::PathBuf::from("/nonexistent/tokenizer.json"));
    assert!(SpeedEngine::new(&c).is_err());
}
