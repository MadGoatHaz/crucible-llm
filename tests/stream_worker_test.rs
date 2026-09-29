//! Integration tests for the Chunk 5 `StreamWorker` against an in-process
//! mock SSE server (plan testing strategy: a tiny `tokio` test harness so
//! the suite runs offline).
//!
//! Acceptance (plan Chunk 5): the worker records T0/T2/T3, delivers
//! correctly-classified frames on the channel, captures `usage`, and shuts
//! down cleanly on `[DONE]` and on a simulated early close.

use std::time::Duration;

use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::sync::mpsc;

use crucible_llm::client::{StreamError, StreamEvent, StreamWorker};
use crucible_llm::sse::{Chunk, Usage};

// ── Mock SSE payloads (vLLM-style) ────────────────────────────────────────

const FRAME_ROLE: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n";
const FRAME_REASON_1: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Let me think.\"}}]}\n\n";
const FRAME_REASON_2: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Step by step.\"}}]}\n\n";
const FRAME_CONTENT_1: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer \"}}]}\n\n";
const FRAME_CONTENT_2: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is 42.\"}}]}\n\n";
const FRAME_USAGE: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":34}}\n\n";
const FRAME_DONE: &str = "data: [DONE]\n\n";

const PLAIN_JSON_BODY: &str = r#"{
    "id": "cmpl-pj",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": { "role": "assistant", "content": "Hello, world!" },
        "finish_reason": "stop"
    }],
    "usage": { "prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49 }
}"#;

// ── Mock server ───────────────────────────────────────────────────────────

/// What the mock server does for the connection(s) it accepts.
#[derive(Clone, Copy, PartialEq)]
enum Mock {
    /// Full vLLM-style SSE stream ending in `[DONE]` (chunked encoding).
    Sse,
    /// A few SSE frames, then a hard close with no `[DONE]` and no
    /// terminating chunk (broken `chunked` close).
    EarlyClose,
    /// A single non-streaming JSON completion.
    PlainJson,
    /// HTTP 500 with a JSON error body.
    Http500,
    /// One SSE chunk, then silence (stalls the stream).
    Stalled,
    /// Drop the first `n` connections immediately, then serve a full SSE
    /// stream (simulates a flaky endpoint coming up).
    Flaky(u32),
}

async fn start_mock(mock: Mock) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let mut dropped = 0u32;
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            drain_request(&mut sock).await;
            let serve = match mock {
                Mock::Flaky(n) if dropped < n => {
                    dropped += 1;
                    false
                }
                _ => true,
            };
            if !serve {
                let _ = sock.shutdown().await;
                continue;
            }
            respond(&mut sock, mock).await;
            return;
        }
    });
    (format!("http://{addr}"), handle)
}

/// Read the request until the header block is complete (the body is not
/// needed by the mock).
async fn drain_request(sock: &mut tokio::net::TcpStream) {
    let mut buf = [0u8; 8192];
    let mut acc = Vec::new();
    use tokio::io::AsyncReadExt;
    loop {
        match sock.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                if acc.windows(4).any(|w| w == b"\r\n\r\n") || acc.len() > 65536 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

async fn respond(sock: &mut tokio::net::TcpStream, mock: Mock) {
    match mock {
        Mock::Sse | Mock::Flaky(_) => {
            write_all(
                sock,
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .await;
            for frame in [
                FRAME_ROLE,
                FRAME_REASON_1,
                FRAME_REASON_2,
                FRAME_CONTENT_1,
                FRAME_CONTENT_2,
                FRAME_USAGE,
                FRAME_DONE,
            ] {
                write_chunk(sock, frame.as_bytes()).await;
            }
            write_chunk(sock, b"").await; // terminating chunk
            let _ = sock.shutdown().await;
        }
        Mock::EarlyClose => {
            write_all(
                sock,
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .await;
            write_chunk(sock, FRAME_ROLE.as_bytes()).await;
            write_chunk(sock, FRAME_REASON_1.as_bytes()).await;
            // A third frame with no terminating blank line, then a hard
            // close — no `0\r\n\r\n` terminator (broken chunked close).
            write_chunk(
                sock,
                b"data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"orphan\"}}]}",
            )
            .await;
            let _ = sock.shutdown().await;
        }
        Mock::PlainJson => {
            let header = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                PLAIN_JSON_BODY.len()
            );
            write_all(sock, header.as_bytes()).await;
            write_all(sock, PLAIN_JSON_BODY.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
        Mock::Http500 => {
            let body = "{\"error\":{\"message\":\"boom\",\"type\":\"server_error\"}}";
            let header = format!(
                "HTTP/1.1 500 Internal Server Error\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            write_all(sock, header.as_bytes()).await;
            write_all(sock, body.as_bytes()).await;
            let _ = sock.shutdown().await;
        }
        Mock::Stalled => {
            write_all(
                sock,
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
            )
            .await;
            write_chunk(sock, FRAME_ROLE.as_bytes()).await;
            // Then silence: the worker's idle read timeout must fire.
            tokio::time::sleep(Duration::from_secs(30)).await;
        }
    }
}

async fn write_all(sock: &mut tokio::net::TcpStream, bytes: &[u8]) {
    let _ = sock.write_all(bytes).await;
    let _ = sock.flush().await;
}

/// Write one `Transfer-Encoding: chunked` frame.
async fn write_chunk(sock: &mut tokio::net::TcpStream, payload: &[u8]) {
    let mut msg = format!("{:x}\r\n", payload.len());
    msg.push_str(&String::from_utf8_lossy(payload));
    msg.push_str("\r\n");
    write_all(sock, msg.as_bytes()).await;
}

// ── Test harness ──────────────────────────────────────────────────────────

fn test_client() -> reqwest::Client {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .build()
        .unwrap()
}

/// Run the worker against `url` and collect every channel event.
async fn run_worker(
    url: &str,
    read_timeout: Duration,
    retries: u32,
) -> (crucible_llm::client::StreamOutcome, Vec<StreamEvent>) {
    let (tx, mut rx) = mpsc::channel(256);
    let worker = StreamWorker::new(test_client(), url, "test-model", "Hello there", 16)
        .read_timeout(read_timeout)
        .retries(retries);
    let outcome = tokio::spawn(worker.run(tx)).await.unwrap();
    let mut events = Vec::new();
    while let Some(event) = rx.recv().await {
        events.push(event);
    }
    (outcome, events)
}

fn frames(events: &[StreamEvent]) -> Vec<crucible_llm::sse::ParsedFrame> {
    events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Frame { frame, .. } => Some(frame.clone()),
            _ => None,
        })
        .collect()
}

const IDLE: Duration = Duration::from_secs(5);

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn sse_stream_records_milestones_and_emits_classified_frames() {
    let (url, _server) = start_mock(Mock::Sse).await;
    let (outcome, events) = run_worker(&url, IDLE, 0).await;

    // Clean completion on [DONE].
    assert!(outcome.is_ok(), "outcome: {:?}", outcome.error);
    assert!(!outcome.premature);
    assert_eq!(outcome.malformed_frames, 0);

    // T0..Tn all recorded, in order.
    let ts = outcome.timestamps;
    assert!(ts.t0.is_some(), "T0");
    assert!(ts.t1.is_some(), "T1");
    assert!(ts.t2.is_some(), "T2");
    assert!(ts.t3.is_some(), "T3");
    assert!(ts.is_complete(), "Tn");
    // Milestone ordering is structural (u64 deltas are non-negative); the
    // meaningful checks are that the full span and TTFT are strictly
    // positive — both span real loopback round trips.
    assert!(ts.total_nanos().unwrap() > 0);
    assert!(ts.ttft_nanos().unwrap() > 0);
    assert!(ts.first_byte_nanos().unwrap() > 0);

    // Seven frames: role + 2 reasoning + 2 content + usage + [DONE].
    let fr = frames(&events);
    assert_eq!(fr.len(), 7);
    assert_eq!(
        fr.iter()
            .filter(|f| matches!(f.chunk, Chunk::Reasoning(_)))
            .count(),
        2
    );
    assert_eq!(
        fr.iter()
            .filter(|f| matches!(f.chunk, Chunk::Content(_)))
            .count(),
        2
    );
    assert_eq!(
        fr.iter()
            .filter(|f| matches!(f.chunk, Chunk::Usage(_)))
            .count(),
        1
    );
    assert_eq!(
        fr.iter()
            .filter(|f| matches!(f.chunk, Chunk::Control))
            .count(),
        2
    );
    assert!(fr.last().unwrap().done, "[DONE] is the last frame");

    // Arrival stamps are present and non-decreasing.
    let stamps: Vec<u64> = fr.iter().map(|f| f.t_nanos).collect();
    assert!(stamps.iter().all(|&n| n > 0));
    assert!(stamps.windows(2).all(|w| w[0] <= w[1]));

    // Usage captured in the frame, the terminal event, and the outcome.
    let usage = Usage {
        prompt_tokens: 128,
        completion_tokens: 34,
    };
    assert_eq!(
        fr.iter().find_map(|f| match f.chunk {
            Chunk::Usage(u) => Some(u),
            _ => None,
        }),
        Some(usage)
    );
    match events.last().unwrap() {
        StreamEvent::Complete {
            usage: u,
            premature,
            ..
        } => {
            assert_eq!(*u, Some(usage));
            assert!(!*premature);
        }
        other => panic!("terminal event must be Complete, got {other:?}"),
    }
    assert_eq!(outcome.usage, Some(usage));
}

#[tokio::test]
async fn early_close_is_marked_premature_and_flushes_pending_frame() {
    let (url, _server) = start_mock(Mock::EarlyClose).await;
    let (outcome, events) = run_worker(&url, IDLE, 0).await;

    // Not an error: partial data is preserved (plan: mark premature_exit).
    assert!(outcome.is_ok(), "outcome: {:?}", outcome.error);
    assert!(outcome.premature, "stream ended without [DONE]");
    assert_eq!(outcome.usage, None);

    let ts = outcome.timestamps;
    assert!(ts.t0.is_some() && ts.t1.is_some() && ts.t2.is_some());
    assert!(
        ts.t3.is_some(),
        "a token frame was decoded before the close"
    );
    assert!(ts.is_complete(), "Tn recorded at the premature close");

    // Role + reasoning delivered live, orphan content flushed by finish().
    let fr = frames(&events);
    assert_eq!(fr.len(), 3);
    assert!(matches!(fr[0].chunk, Chunk::Control));
    assert!(matches!(fr[1].chunk, Chunk::Reasoning(_)));
    assert_eq!(fr[2].chunk, Chunk::Content("orphan".to_string()));

    match events.last().unwrap() {
        StreamEvent::Complete { premature, .. } => assert!(*premature),
        other => panic!("terminal event must be Complete, got {other:?}"),
    }
}

#[tokio::test]
async fn plain_json_fallback_yields_single_content_and_usage() {
    let (url, _server) = start_mock(Mock::PlainJson).await;
    let (outcome, events) = run_worker(&url, IDLE, 0).await;

    assert!(outcome.is_ok(), "outcome: {:?}", outcome.error);
    assert!(
        !outcome.premature,
        "a complete JSON response is not premature"
    );

    let fr = frames(&events);
    assert_eq!(fr.len(), 2);
    assert_eq!(fr[0].chunk, Chunk::Content("Hello, world!".to_string()));
    assert_eq!(
        fr[1].chunk,
        Chunk::Usage(Usage {
            prompt_tokens: 42,
            completion_tokens: 7
        })
    );

    // T2 (first byte) and T3 (the content frame) are recorded.
    let ts = outcome.timestamps;
    assert!(ts.t2.is_some() && ts.t3.is_some() && ts.is_complete());

    assert_eq!(
        outcome.usage,
        Some(Usage {
            prompt_tokens: 42,
            completion_tokens: 7
        })
    );
}

#[tokio::test]
async fn http_500_is_a_failed_outcome_with_status_and_body() {
    let (url, _server) = start_mock(Mock::Http500).await;
    let (outcome, events) = run_worker(&url, IDLE, 0).await;

    assert!(!outcome.is_ok());
    match &outcome.error {
        Some(StreamError::Http { status, body }) => {
            assert_eq!(*status, 500);
            assert!(body.contains("boom"), "body: {body}");
        }
        other => panic!("expected Http error, got {other:?}"),
    }
    // The request was sent (T1) but no body byte arrived (T2).
    assert!(outcome.timestamps.t0.is_some());
    assert!(outcome.timestamps.t1.is_some());
    assert!(outcome.timestamps.t2.is_none());

    // No frames, one terminal Failed event.
    assert!(frames(&events).is_empty());
    assert!(matches!(events.last().unwrap(), StreamEvent::Failed { .. }));
}

#[tokio::test]
async fn connection_refused_is_a_failed_outcome_at_t0() {
    // Grab a free port, then release it so nothing is listening.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    drop(listener);
    let url = format!("http://{addr}");

    let (outcome, events) = run_worker(&url, IDLE, 0).await;

    assert!(!outcome.is_ok());
    assert!(
        matches!(&outcome.error, Some(StreamError::Connection(_))),
        "expected Connection error, got {:?}",
        outcome.error
    );
    // Never got past the socket start.
    assert!(outcome.timestamps.t0.is_some());
    assert!(outcome.timestamps.t1.is_none());
    assert!(frames(&events).is_empty());
    assert!(matches!(events.last().unwrap(), StreamEvent::Failed { .. }));
}

#[tokio::test]
async fn stalled_stream_hits_the_read_timeout() {
    let (url, _server) = start_mock(Mock::Stalled).await;
    let read_timeout = Duration::from_millis(300);
    let (outcome, events) = run_worker(&url, read_timeout, 0).await;

    assert!(!outcome.is_ok());
    assert_eq!(
        outcome.error,
        Some(StreamError::Timeout(read_timeout)),
        "expected a stall timeout"
    );
    // One chunk arrived (T2) before the silence. The role-only opener is a
    // real (Control) SSE packet and is forwarded — the MTP packet count
    // needs it — but it is not a token frame, so T3 stays unset.
    assert!(outcome.timestamps.t2.is_some());
    assert!(
        outcome.timestamps.t3.is_none(),
        "no token frame before the stall"
    );
    let fr = frames(&events);
    assert_eq!(fr.len(), 1);
    assert!(matches!(fr[0].chunk, Chunk::Control));
    assert!(matches!(events.last().unwrap(), StreamEvent::Failed { .. }));
}

#[tokio::test]
async fn connect_failure_is_retried_then_succeeds() {
    let (url, _server) = start_mock(Mock::Flaky(1)).await;
    let (outcome, events) = run_worker(&url, IDLE, 1).await;

    assert!(
        outcome.is_ok(),
        "second attempt must succeed: {:?}",
        outcome.error
    );
    assert!(!outcome.premature);
    assert_eq!(
        outcome.usage,
        Some(Usage {
            prompt_tokens: 128,
            completion_tokens: 34
        })
    );

    // Exactly one Failed (the dropped first connection), then a clean
    // stream: 7 frames + Complete.
    let failed = events
        .iter()
        .filter(|e| matches!(e, StreamEvent::Failed { .. }))
        .count();
    assert_eq!(failed, 1);
    assert!(matches!(
        events.last().unwrap(),
        StreamEvent::Complete { .. }
    ));
    assert_eq!(frames(&events).len(), 7);
}

#[tokio::test]
async fn non_retriable_failure_is_not_retried() {
    let (url, _server) = start_mock(Mock::Http500).await;
    let (tx, mut rx) = mpsc::channel(256);
    let worker = StreamWorker::new(test_client(), &url, "test-model", "Hi", 16).retries(3);
    let outcome = tokio::spawn(worker.run(tx)).await.unwrap();
    let mut event_count = 0usize;
    while rx.recv().await.is_some() {
        event_count += 1;
    }

    assert!(!outcome.is_ok());
    assert!(matches!(
        outcome.error.as_ref().unwrap(),
        StreamError::Http { .. }
    ));
    // One attempt only: a single Failed event, no retry traffic.
    assert_eq!(event_count, 1);
}
