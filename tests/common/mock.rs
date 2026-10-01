//! In-process mock HTTP/SSE server primitives (the test suite runs fully
//! offline — no external network).
//!
//! These are the pieces every mock server shares: chunked-encoding frame
//! writers, request readers (headers-only or headers + `Content-Length`
//! body), and the representative vLLM-style SSE payloads. Each test file
//! keeps its own *behavior* (what the mock answers per mode) on top of
//! these primitives.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

// ── vLLM-style SSE payloads ─────────────────────────────────────────────

/// Role-only opener (not a token frame).
pub const FRAME_ROLE: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n";
/// Two `reasoning_content` deltas…
pub const FRAME_REASON_1: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Let me think.\"}}]}\n\n";
pub const FRAME_REASON_2: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Step by step.\"}}]}\n\n";
/// …two `content` deltas…
pub const FRAME_CONTENT_1: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer \"}}]}\n\n";
pub const FRAME_CONTENT_2: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is 42.\"}}]}\n\n";
/// …a `usage` block, and the `[DONE]` terminator.
pub const FRAME_USAGE: &str =
    "data: {\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":34}}\n\n";
pub const FRAME_DONE: &str = "data: [DONE]\n\n";

/// A non-streaming plain-JSON completion (the `stream: true` fallback).
pub const PLAIN_JSON_BODY: &str = r#"{
    "id": "cmpl-pj",
    "object": "chat.completion",
    "choices": [{
        "index": 0,
        "message": { "role": "assistant", "content": "Hello, world!" },
        "finish_reason": "stop"
    }],
    "usage": { "prompt_tokens": 42, "completion_tokens": 7, "total_tokens": 49 }
}"#;

/// The full well-behaved vLLM stream: role + 2 reasoning + 2 content +
/// usage + `[DONE]` (7 frames).
pub const VLLM_STREAM: &[&str] = &[
    FRAME_ROLE,
    FRAME_REASON_1,
    FRAME_REASON_2,
    FRAME_CONTENT_1,
    FRAME_CONTENT_2,
    FRAME_USAGE,
    FRAME_DONE,
];

// ── writers ─────────────────────────────────────────────────────────────

/// Write `bytes` to the socket and flush (errors ignored — a mock server
/// dropping a test client is not a failure worth reporting).
pub async fn write_all(sock: &mut TcpStream, bytes: &[u8]) {
    let _ = sock.write_all(bytes).await;
    let _ = sock.flush().await;
}

/// Write one `Transfer-Encoding: chunked` frame.
pub async fn write_chunk(sock: &mut TcpStream, payload: &[u8]) {
    let mut msg = format!("{:x}\r\n", payload.len());
    msg.push_str(&String::from_utf8_lossy(payload));
    msg.push_str("\r\n");
    write_all(sock, msg.as_bytes()).await;
}

// ── readers ─────────────────────────────────────────────────────────────

/// Read until the header block (`\r\n\r\n`) is complete — the body is not
/// needed by mocks that only shape the response.
pub async fn drain_request(sock: &mut TcpStream) {
    read_headers(sock).await;
}

/// Read the header block; returns it lowercased (an empty string when the
/// connection closes first).
pub async fn read_headers(sock: &mut TcpStream) -> String {
    let mut buf = [0u8; 8192];
    let mut acc = Vec::new();
    loop {
        match sock.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                if acc.windows(4).any(|w| w == b"\r\n\r\n") || acc.len() > 65536 {
                    break;
                }
            }
        }
    }
    let header_end = acc
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(acc.len());
    String::from_utf8_lossy(&acc[..header_end]).to_ascii_lowercase()
}

/// Read the full request (headers + `Content-Length` body) and return
/// `(lowercased header block, body text)`.
pub async fn read_request(sock: &mut TcpStream) -> (String, String) {
    let mut buf = [0u8; 8192];
    let mut acc: Vec<u8> = Vec::new();
    // Phase 1: the header block.
    let mut header_end: Option<usize> = None;
    while header_end.is_none() {
        match sock.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                header_end = acc.windows(4).position(|w| w == b"\r\n\r\n").map(|p| p + 4);
            }
        }
    }
    let he = header_end.unwrap_or(acc.len());
    let headers = String::from_utf8_lossy(&acc[..he]).to_ascii_lowercase();
    // Phase 2: the `Content-Length` body (it may arrive across reads).
    let len = content_length(&headers);
    while acc.len() < he + len {
        match sock.read(&mut buf).await {
            Ok(0) | Err(_) => break,
            Ok(n) => acc.extend_from_slice(&buf[..n]),
        }
    }
    let end = (he + len).min(acc.len());
    (headers, String::from_utf8_lossy(&acc[he..end]).into_owned())
}

/// The `Content-Length` header value (`0` when absent or malformed).
pub fn content_length(headers: &str) -> usize {
    headers
        .lines()
        .find_map(|l| {
            l.trim()
                .strip_prefix("content-length:")
                .and_then(|v| v.trim().parse::<usize>().ok())
        })
        .unwrap_or(0)
}

/// Extract `messages[0].content` from a chat-completions JSON body
/// (`None` when the body is not the expected shape).
pub fn prompt_from_body(body: &str) -> Option<String> {
    serde_json::from_str::<serde_json::Value>(body)
        .ok()?
        .get("messages")?
        .get(0)?
        .get("content")?
        .as_str()
        .map(str::to_owned)
}

// ── response shapers ────────────────────────────────────────────────────

/// The standard vLLM-style `200` SSE headers.
pub const SSE_HEADERS: &[u8] = b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n";

/// Serve a vLLM-style SSE stream: the given frames (each already carrying
/// its trailing blank line), spaced by `frame_delay` (when non-zero),
/// then the terminating chunk.
pub async fn serve_sse<I, S>(sock: &mut TcpStream, frames: I, frame_delay: std::time::Duration)
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    write_all(sock, SSE_HEADERS).await;
    for frame in frames {
        if frame_delay > std::time::Duration::ZERO {
            tokio::time::sleep(frame_delay).await;
        }
        write_chunk(sock, frame.as_ref().as_bytes()).await;
    }
    write_chunk(sock, b"").await; // terminating chunk
    let _ = sock.shutdown().await;
}

/// Serve a `Content-Length` JSON response with the given status line.
pub async fn serve_json(sock: &mut TcpStream, status_line: &str, body: &str) {
    let header = format!(
        "{status_line}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    write_all(sock, header.as_bytes()).await;
    write_all(sock, body.as_bytes()).await;
    let _ = sock.shutdown().await;
}
