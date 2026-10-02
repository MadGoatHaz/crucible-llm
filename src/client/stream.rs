//! `StreamWorker`: one HTTP/2 SSE stream against `/v1/chat/completions`.
//!
//! The worker is the inference-engine adapter (plan Chunk 5). Given an
//! already-generated prompt (Chunk 4 `PromptGenerator`), it:
//!
//! 1. POSTs a single-message chat body — `stream: true`,
//!    `stream_options.include_usage: true`, `temperature: 0`, `max_tokens` —
//!    through a shared, connection-pooled `reqwest` client (HTTP/2 over TLS,
//!    HTTP/1.1 over plain HTTP, which is what local vLLM / llama.cpp
//!    endpoints speak);
//! 2. records the `T0..Tn` lifecycle milestones (blueprint §4.1 / §7) on the
//!    `quanta` monotonic clock:
//!    * `T0` — just before the request is sent (socket connection start);
//!    * `T1` — request write complete (`send()` resolves on response
//!      headers);
//!    * `T2` — first body byte received;
//!    * `T3` — first token frame decoded (the first `Reasoning` /
//!      `Content` chunk; a role-only opener is *not* a token);
//!    * `Tn` — stream close;
//! 3. streams response bytes into the Chunk 3 `SseParser`, stamping each
//!    emitted `ParsedFrame` with its arrival time (`t_nanos`, nanoseconds
//!    since `T0`) — the parser's documented contract;
//! 4. emits [`StreamEvent`]s over a bounded `tokio::sync::mpsc` channel for
//!    the Engine Core (Chunk 6): one `Frame { frame, at, timestamps }` per
//!    parsed frame, plus a terminal `Complete` (clean `[DONE]`, or
//!    `premature: true` for an early close) / `Failed` event.
//!
//! Error handling (plan Chunk 5):
//!
//! * **non-200 HTTP** → `Failed` with [`StreamError::Http`] (body
//!   truncated for logging);
//! * **connection refused / DNS / connect timeout / reset** →
//!   [`StreamError::Connection`] — the only *retriable* class; [`run`]
//!   re-attempts up to `retries` times (a failed connect emits no frames,
//!   so the channel stays clean across attempts);
//! * **read timeout** (no bytes for `read_timeout`) →
//!   [`StreamError::Timeout`];
//! * **premature close** — the stream ends (clean EOF, or a broken
//!   `chunked`-encoding close / connection reset) without `[DONE]` →
//!   `Complete` with `premature: true`; any pending parser frame is
//!   flushed and all captured timestamps are preserved;
//! * **non-SSE plain-JSON fallback** — a server that ignores `stream: true`
//!   and returns a single completion object → the whole body is parsed as
//!   one completion: a single `Content` frame plus the `Usage` frame.
//!
//! Measurement-isolation note: all timestamps are captured in the worker
//! only (blueprint §4). Nothing in this module touches the UI or the
//! storage rings.

use std::sync::Arc;
use std::time::Duration;

use futures_util::StreamExt;
use thiserror::Error;
use tokio::sync::mpsc;

use crate::log::{Context, Level, RunLogger};
use crate::sse::{Chunk, ParsedFrame, SseParser, Usage};
use crate::timing::{MonotonicInstant, StreamTimestamps};

use super::loop_guard::LoopGuard;
use super::truncate_body;

/// Completions path appended to a bare host / base URL.
const COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// A base URL that already ends in `/v1` only needs the rest of the path.
const V1_SUFFIX: &str = "/v1";

/// UTF-8 byte-order mark (some servers prefix the stream with one).
const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Minimum prefix length needed to recognize an SSE `data:` frame.
const SSE_PREFIX_LEN: usize = b"data:".len();

/// Default idle (read) timeout: a stream that produces no bytes for this
/// long is declared stalled.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// One item on the worker's bounded channel, consumed by the Engine Core.
#[derive(Debug, Clone)]
pub enum StreamEvent {
    /// A parsed, timestamped frame.
    ///
    /// `frame.t_nanos` is the arrival time in nanoseconds since `T0`;
    /// `at` is the raw monotonic instant (the Engine Core takes deltas of
    /// consecutive `at` values for ITL); `timestamps` is the running
    /// `T0..Tn` record at the moment the frame was emitted.
    Frame {
        frame: ParsedFrame,
        at: MonotonicInstant,
        timestamps: StreamTimestamps,
    },
    /// The stream ended: cleanly on `[DONE]` (`premature: false`), or
    /// early without a terminator (`premature: true`).
    ///
    /// `looping` / `loop_excluded_tokens` (v0.1.1 decode-loop guard):
    /// `looping` is `true` when the guard flagged a repeating 32-token
    /// pattern ×3; `loop_excluded_tokens` is the stream's total token
    /// count when looping (its output is excluded from throughput
    /// calculations), else `0`.
    Complete {
        timestamps: StreamTimestamps,
        usage: Option<Usage>,
        premature: bool,
        malformed_frames: u64,
        looping: bool,
        loop_excluded_tokens: u64,
    },
    /// The stream failed (HTTP error, connection refused, timeout,
    /// read error, unparseable plain-JSON fallback).
    Failed {
        timestamps: StreamTimestamps,
        error: StreamError,
        looping: bool,
        loop_excluded_tokens: u64,
    },
}

/// Failure modes for a single stream attempt.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StreamError {
    /// The server answered with a non-2xx status.
    #[error("HTTP {status}: {body}")]
    Http {
        status: u16,
        /// The response body, truncated for logging.
        body: String,
    },
    /// The connection could not be established (refused, DNS failure,
    /// connect timeout, reset before a response).
    #[error("connection failed: {0}")]
    Connection(String),
    /// No bytes arrived for the configured read timeout.
    #[error("stream stalled: no data for {0:?}")]
    Timeout(Duration),
    /// The worker exceeded its max lifetime and was killed by the
    /// concurrency sweep (per-worker timeout). Any tokens received before
    /// the kill were relayed as partial results.
    #[error("worker timed out after {0:?} (killed)")]
    WorkerTimeout(Duration),
    /// A byte-stream read error before any data was received.
    #[error("stream read error: {0}")]
    Read(String),
    /// A non-streaming response body that is not valid JSON.
    #[error("non-streaming response is not valid JSON: {0}")]
    InvalidJson(String),
}

impl StreamError {
    /// Whether a fresh attempt is likely to succeed.
    ///
    /// Only connect-phase failures are retriable: a definitive HTTP answer
    /// or an accepted-then-stalled request is a *measurement* of the
    /// endpoint, not a transient fault — retrying those would hide exactly
    /// the condition the benchmark exists to expose.
    pub fn is_retriable(&self) -> bool {
        matches!(self, Self::Connection(_))
    }
}

/// Final state of a worker run (also summarized in the terminal channel
/// event).
#[derive(Debug, Clone)]
pub struct StreamOutcome {
    /// The full `T0..Tn` record.
    pub timestamps: StreamTimestamps,
    /// Server-reported usage, if the stream provided it.
    pub usage: Option<Usage>,
    /// `true` when the stream ended without a `[DONE]` terminator.
    pub premature: bool,
    /// Malformed-JSON frames the parser skipped.
    pub malformed_frames: u64,
    /// `Some` when the run failed.
    pub error: Option<StreamError>,
    /// The decode-loop guard flagged this stream (v0.1.1): its repeating
    /// output is excluded from throughput calculations.
    pub looping: bool,
    /// The stream's total token count when `looping` (excluded from
    /// throughput), else `0`.
    pub loop_excluded_tokens: u64,
}

impl StreamOutcome {
    /// `true` when the run completed without an error (including
    /// `premature` completions, which preserve partial data).
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }
}

/// A single-stream benchmark worker (one independent HTTP/2 stream).
///
/// Build one with [`StreamWorker::new`] (sharing a `reqwest::Client` across
/// workers gives connection pooling, per blueprint §4.1), then hand it to
/// [`run`] with the sending half of a bounded `tokio::sync::mpsc` channel.
#[derive(Debug, Clone)]
pub struct StreamWorker {
    client: reqwest::Client,
    endpoint: String,
    model: String,
    api_key: Option<String>,
    prompt: String,
    max_tokens: u32,
    read_timeout: Duration,
    retries: u32,
    /// The OpenAI-compatible `response_format` type (e.g.
    /// `"json_object"`) — Engine C3's grammar-constrained runs. `None`
    /// (the default) sends the body unchanged.
    response_format: Option<String>,
    /// Optional run-log writer: when present, the worker logs its HTTP /
    /// SSE lifecycle (request, TTFB, first token, every 50th token,
    /// completion / failure) to the file-based [`RunLogger`].
    logger: Option<Arc<RunLogger>>,
    /// A short identifier for this worker's log lines (e.g. `A:2` for
    /// Engine A iteration 2, `B5:3` for Engine B step 5 worker 3).
    tag: String,
}

impl StreamWorker {
    /// Create a worker for `endpoint` (a bare host, a base URL with or
    /// without `/v1` — see [`normalize_endpoint`]), `model`, and the
    /// fully-generated `prompt` text (Chunk 4).
    pub fn new(
        client: reqwest::Client,
        endpoint: &str,
        model: &str,
        prompt: &str,
        max_tokens: u32,
    ) -> Self {
        Self {
            client,
            endpoint: normalize_endpoint(endpoint),
            model: model.to_string(),
            api_key: None,
            prompt: prompt.to_string(),
            max_tokens: max_tokens.max(1),
            read_timeout: DEFAULT_READ_TIMEOUT,
            retries: 0,
            response_format: None,
            logger: None,
            tag: "stream".to_string(),
        }
    }

    /// Set an `Authorization: Bearer` header.
    pub fn api_key(mut self, key: impl Into<String>) -> Self {
        self.api_key = Some(key.into());
        self
    }

    /// Idle read timeout (default [`DEFAULT_READ_TIMEOUT`]).
    pub fn read_timeout(mut self, d: Duration) -> Self {
        self.read_timeout = d;
        self
    }

    /// Retries for connect-phase failures only
    /// ([`StreamError::is_retriable`]).
    pub fn retries(mut self, n: u32) -> Self {
        self.retries = n;
        self
    }

    /// Request constrained (structured) output via the OpenAI-compatible
    /// `response_format` directive (Engine C3, Chunk 16): the body gains
    /// `"response_format": { "type": <format> }` (e.g. `"json_object"`),
    /// which server-side constrained-decoding engines honor.
    pub fn response_format(mut self, format: &str) -> Self {
        self.response_format = Some(format.to_string());
        self
    }

    /// Attach the shared run logger (file-based, non-blocking): the
    /// worker then records its HTTP / SSE lifecycle lines.
    pub fn logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// The short identifier for this worker's log lines.
    pub fn tag(mut self, tag: impl Into<String>) -> Self {
        self.tag = tag.into();
        self
    }

    /// Log one line (a no-op when no logger is attached).
    fn logl(&self, level: Level, ctx: Context, msg: &str) {
        let Some(l) = &self.logger else {
            return;
        };
        match level {
            Level::Debug => l.debug(ctx, msg),
            Level::Info => l.info(ctx, msg),
            Level::Warn => l.warn(ctx, msg),
            Level::Error => l.error(ctx, msg),
        }
    }

    /// Run the full stream lifecycle, emitting events on `tx`.
    ///
    /// Returns the final [`StreamOutcome`]. Connect-phase failures are
    /// re-attempted up to `retries` times; every other outcome (success,
    /// premature completion, HTTP/timeout/read failure) is final.
    pub async fn run(self, tx: mpsc::Sender<StreamEvent>) -> StreamOutcome {
        let mut attempt = 0u32;
        loop {
            attempt += 1;
            let outcome = self.clone().run_once(&mut tx.clone()).await;
            let retriable = outcome
                .error
                .as_ref()
                .is_some_and(StreamError::is_retriable);
            if retriable && attempt <= self.retries {
                // The failed attempt emitted no frames (it never reached
                // T2), so the channel only carries its `Failed` event and
                // the next attempt starts a clean lifecycle.
                continue;
            }
            return outcome;
        }
    }

    /// One full attempt: send → status check → stream (or fallback) →
    /// terminal event.
    async fn run_once(&self, tx: &mut mpsc::Sender<StreamEvent>) -> StreamOutcome {
        // T0 — socket connection start.
        let mut ts = StreamTimestamps {
            t0: Some(MonotonicInstant::now()),
            ..StreamTimestamps::default()
        };

        self.logl(
            Level::Info,
            Context::Http,
            &format!(
                "{} → POST {} (stream=true, model={}, tokens_target={})",
                self.tag, self.endpoint, self.model, self.max_tokens
            ),
        );

        let resp = match self.send_request().await {
            Ok(r) => r,
            Err(e) => {
                ts.t_end = Some(MonotonicInstant::now());
                return self
                    .finish_failed(tx, ts, StreamError::Connection(e.to_string()), false, 0)
                    .await;
            }
        };
        // T1 — request write complete (`send` resolved on response headers).
        ts.t1 = Some(MonotonicInstant::now());

        if !resp.status().is_success() {
            let status = resp.status().as_u16();
            let body = truncate_body(&resp.text().await.unwrap_or_default(), 512);
            self.logl(
                Level::Error,
                Context::Http,
                &format!("{} ← {status} {body}", self.tag),
            );
            ts.t_end = Some(MonotonicInstant::now());
            return self
                .finish_failed(tx, ts, StreamError::Http { status, body }, false, 0)
                .await;
        }

        let mut stream = resp.bytes_stream();
        let mut parser = SseParser::new();

        // The first bytes decide the response shape. A tiny first segment
        // cannot be classified yet, so accumulate until the `data:` prefix
        // (or lack of one) is visible.
        let mut head: Vec<u8> = match read_chunk(&mut stream, self.read_timeout).await {
            ChunkRead::Bytes(b) => b.to_vec(),
            ChunkRead::Eof => {
                // Empty 200 body: nothing to measure.
                ts.t_end = Some(MonotonicInstant::now());
                return self
                    .finish_complete(tx, &ts, None, false, parser.malformed_frames(), (false, 0))
                    .await;
            }
            ChunkRead::Timeout => {
                ts.t_end = Some(MonotonicInstant::now());
                return self
                    .finish_failed(tx, ts, StreamError::Timeout(self.read_timeout), false, 0)
                    .await;
            }
            ChunkRead::Error(msg) => {
                ts.t_end = Some(MonotonicInstant::now());
                return self
                    .finish_failed(tx, ts, StreamError::Read(msg), false, 0)
                    .await;
            }
        };
        while head.len() < SSE_PREFIX_LEN {
            match read_chunk(&mut stream, self.read_timeout).await {
                ChunkRead::Bytes(b) => head.extend_from_slice(b.as_ref()),
                ChunkRead::Eof => break,
                ChunkRead::Timeout => {
                    ts.t_end = Some(MonotonicInstant::now());
                    return self
                        .finish_failed(tx, ts, StreamError::Timeout(self.read_timeout), false, 0)
                        .await;
                }
                ChunkRead::Error(msg) => {
                    ts.t_end = Some(MonotonicInstant::now());
                    return self
                        .finish_failed(tx, ts, StreamError::Read(msg), false, 0)
                        .await;
                }
            }
        }
        let head_at = MonotonicInstant::now();
        if ts.t2.is_none() {
            ts.t2 = Some(head_at);
            // TTFB = first body byte − request write complete (T2 − T1).
            let status = 200u16; // success was checked above
            let ttfb_ms = ts
                .t1
                .map(|t1| t1.delta_nanos(&head_at))
                .map(|ns| ns as f64 / 1e6)
                .unwrap_or(0.0);
            self.logl(
                Level::Info,
                Context::Http,
                &format!("{} ← {status} OK (TTFB: {ttfb_ms:.0}ms)", self.tag),
            );
        }

        if looks_like_sse(head.as_ref()) {
            self.run_sse(head, head_at, &mut parser, &mut ts, &mut stream, tx)
                .await
        } else {
            self.run_plain_json(head, &mut ts, &mut stream, tx).await
        }
    }

    /// The SSE path: feed bytes into the parser, stamp and forward frames,
    /// until `[DONE]`, clean EOF, premature close, timeout, or read error.
    ///
    /// The per-stream [`LoopGuard`] (v0.1.1 decode-loop guard) sees every
    /// decoded token here; a flagged stream's terminal event carries
    /// `looping: true` so the engines exclude its tokens from throughput.
    async fn run_sse(
        &self,
        head: impl AsRef<[u8]>,
        head_at: MonotonicInstant,
        parser: &mut SseParser,
        ts: &mut StreamTimestamps,
        stream: &mut (impl StreamExt<Item = reqwest::Result<impl AsRef<[u8]>>> + Unpin),
        tx: &mut mpsc::Sender<StreamEvent>,
    ) -> StreamOutcome {
        let mut guard = LoopGuard::new();
        let loop_state = |guard: &LoopGuard, usage: Option<Usage>, frames: u64| {
            let looping = guard.is_detected();
            (
                looping,
                if looping {
                    usage.map(|u| u.completion_tokens).unwrap_or(frames)
                } else {
                    0
                },
            )
        };
        let mut tokens = 0u64;
        let mut done = match self
            .emit_frames(parser.feed(head.as_ref()), &head_at, ts, tx, &mut guard)
            .await
        {
            Ok((d, tc)) => {
                tokens += tc;
                self.log_token_milestones(ts, tokens);
                d
            }
            Err(_) => {
                // Receiver gone: stop measuring.
                ts.t_end = Some(MonotonicInstant::now());
                let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                return self
                    .finish_complete(
                        tx,
                        ts,
                        parser.usage(),
                        true,
                        parser.malformed_frames(),
                        (looping, excluded),
                    )
                    .await;
            }
        };

        while !done {
            let read = read_chunk(stream, self.read_timeout).await;
            let at = MonotonicInstant::now();
            match read {
                ChunkRead::Timeout => {
                    ts.t_end = Some(at);
                    return self
                        .finish_failed(tx, *ts, StreamError::Timeout(self.read_timeout), false, 0)
                        .await;
                }
                ChunkRead::Eof => {
                    // Clean EOF without `[DONE]`: premature close.
                    if let Ok((_, tc)) = self
                        .emit_frames(parser.finish(), &at, ts, tx, &mut guard)
                        .await
                    {
                        tokens += tc;
                    }
                    ts.t_end = Some(at);
                    self.log_stream_end(ts, tokens, parser.usage(), true);
                    let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                    return self
                        .finish_complete(
                            tx,
                            ts,
                            parser.usage(),
                            true,
                            parser.malformed_frames(),
                            (looping, excluded),
                        )
                        .await;
                }
                ChunkRead::Error(msg) => {
                    if ts.t2.is_some() {
                        // Premature close mid-stream: a broken `chunked`
                        // close or connection reset. Flush the pending
                        // frame, keep every captured timestamp.
                        if let Ok((_, tc)) = self
                            .emit_frames(parser.finish(), &at, ts, tx, &mut guard)
                            .await
                        {
                            tokens += tc;
                        }
                        ts.t_end = Some(at);
                        self.log_stream_end(ts, tokens, parser.usage(), true);
                        let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                        return self
                            .finish_complete(
                                tx,
                                ts,
                                parser.usage(),
                                true,
                                parser.malformed_frames(),
                                (looping, excluded),
                            )
                            .await;
                    }
                    ts.t_end = Some(at);
                    return self
                        .finish_failed(tx, *ts, StreamError::Read(msg), false, 0)
                        .await;
                }
                ChunkRead::Bytes(b) => {
                    if ts.t2.is_none() {
                        ts.t2 = Some(at);
                    }
                    match self
                        .emit_frames(parser.feed(b.as_ref()), &at, ts, tx, &mut guard)
                        .await
                    {
                        Ok((d, tc)) => {
                            done = d;
                            tokens += tc;
                            self.log_token_milestones(ts, tokens);
                        }
                        Err(_) => {
                            ts.t_end = Some(at);
                            let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                            return self
                                .finish_complete(
                                    tx,
                                    ts,
                                    parser.usage(),
                                    !done,
                                    parser.malformed_frames(),
                                    (looping, excluded),
                                )
                                .await;
                        }
                    }
                }
            }
        }

        // `[DONE]` received: Tn — stream close.
        ts.t_end = Some(MonotonicInstant::now());
        self.log_stream_end(ts, tokens, parser.usage(), false);
        let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
        self.finish_complete(
            tx,
            ts,
            parser.usage(),
            false,
            parser.malformed_frames(),
            (looping, excluded),
        )
        .await
    }

    /// Log the first token and every 50th token of the stream (the
    /// per-token hot path stays allocation-free: at most one formatted
    /// line per 50 tokens per worker).
    fn log_token_milestones(&self, ts: &StreamTimestamps, tokens: u64) {
        if tokens == 1 {
            let elapsed_ms = ts
                .t0
                .map(|t0| t0.elapsed().as_secs_f64() * 1000.0)
                .unwrap_or(0.0);
            self.logl(
                Level::Info,
                Context::Sse,
                &format!(
                    "{} token #1 received (elapsed: {elapsed_ms:.0}ms)",
                    self.tag
                ),
            );
        } else if tokens.is_multiple_of(50) {
            let now = MonotonicInstant::now();
            let elapsed_s = ts
                .t0
                .map(|t0| t0.delta_nanos(&now) as f64 / 1e9)
                .unwrap_or(0.0);
            let rate = ts
                .t3
                .map(|t3| {
                    let span = t3.delta_nanos(&now).max(1) as f64 / 1e9;
                    tokens as f64 / span
                })
                .unwrap_or(0.0);
            self.logl(
                Level::Info,
                Context::Sse,
                &format!(
                    "{} token #{tokens} received (elapsed: {elapsed_s:.1}s, rate: {rate:.1} t/s)",
                    self.tag
                ),
            );
        }
    }

    /// Log the stream's completion. `frames` is the count of SSE chunks
    /// that carried tokens — an *estimate*, not the server's token count.
    /// When the server reported `usage.completion_tokens`, that is the
    /// authoritative count and the line reads
    /// `… N tokens (M frames) in Xs (Y t/s)`; otherwise it reads
    /// `… M frames in Xs (~Y t/s est)`. The rate is over the T3→Tn
    /// window, and a `premature` marker is added when the stream closed
    /// without `[DONE]`.
    fn log_stream_end(
        &self,
        ts: &StreamTimestamps,
        frames: u64,
        usage: Option<Usage>,
        premature: bool,
    ) {
        let tokens = usage.map(|u| u.completion_tokens).unwrap_or(frames);
        let (total_s, rate) = match (ts.t0, ts.t_end, ts.t3) {
            (Some(t0), Some(end), Some(t3)) => {
                let total = t0.delta_nanos(&end) as f64 / 1e9;
                let span = t3.delta_nanos(&end).max(1) as f64 / 1e9;
                (total, (tokens as f64 / span).max(0.0))
            }
            (Some(t0), Some(end), None) => (t0.delta_nanos(&end) as f64 / 1e9, 0.0),
            _ => (0.0, 0.0),
        };
        let tag = if premature { " (premature close)" } else { "" };
        let msg = match usage {
            Some(u) => format!(
                "{} stream complete: {} tokens ({} frames) in {total_s:.1}s ({:.1} t/s){tag}",
                self.tag, u.completion_tokens, frames, rate
            ),
            None => format!(
                "{} stream complete: {frames} frames in {total_s:.1}s (~{rate:.1} t/s est){tag}",
                self.tag
            ),
        };
        self.logl(Level::Info, Context::Sse, &msg);
    }

    /// The non-SSE plain-JSON fallback: a server that ignored `stream: true`
    /// and returned a single completion object. Extract usage + one content
    /// chunk (plan Chunk 5).
    async fn run_plain_json(
        &self,
        head: impl AsRef<[u8]>,
        ts: &mut StreamTimestamps,
        stream: &mut (impl StreamExt<Item = reqwest::Result<impl AsRef<[u8]>>> + Unpin),
        tx: &mut mpsc::Sender<StreamEvent>,
    ) -> StreamOutcome {
        let mut body = Vec::with_capacity(head.as_ref().len());
        body.extend_from_slice(head.as_ref());
        loop {
            match read_chunk(stream, self.read_timeout).await {
                ChunkRead::Bytes(b) => body.extend_from_slice(b.as_ref()),
                ChunkRead::Eof => break,
                ChunkRead::Timeout => {
                    ts.t_end = Some(MonotonicInstant::now());
                    return self
                        .finish_failed(tx, *ts, StreamError::Timeout(self.read_timeout), false, 0)
                        .await;
                }
                ChunkRead::Error(msg) => {
                    ts.t_end = Some(MonotonicInstant::now());
                    return self
                        .finish_failed(tx, *ts, StreamError::Read(msg), false, 0)
                        .await;
                }
            }
        }

        // A single non-streaming completion cannot loop (at most one token
        // frame), but the guard is fed for a uniform `emit_frames` contract.
        let mut guard = LoopGuard::new();
        let parsed_at = MonotonicInstant::now();
        match parse_plain_completion(&body) {
            Ok((content, usage)) => {
                let mut frames = Vec::with_capacity(2);
                if let Some(text) = content {
                    frames.push(ParsedFrame {
                        chunk: Chunk::Content(text),
                        t_nanos: 0,
                        done: false,
                    });
                }
                if let Some(u) = usage {
                    frames.push(ParsedFrame {
                        chunk: Chunk::Usage(u),
                        t_nanos: 0,
                        done: false,
                    });
                }
                // `emit_frames` stamps `t_nanos` and sets T3 on the first
                // token (content) frame.
                let tokens = match self
                    .emit_frames(&frames, &parsed_at, ts, tx, &mut guard)
                    .await
                {
                    Ok((_, tc)) => tc,
                    Err(_) => {
                        ts.t_end = Some(parsed_at);
                        return self
                            .finish_complete(tx, ts, usage, true, 0, (false, 0))
                            .await;
                    }
                };
                ts.t_end = Some(MonotonicInstant::now());
                self.log_stream_end(ts, tokens, usage, false);
                // A complete plain-JSON response is a normal completion, not
                // a premature one (there is no `[DONE]` to expect).
                self.finish_complete(tx, ts, usage, false, 0, (false, 0))
                    .await
            }
            Err(msg) => {
                ts.t_end = Some(parsed_at);
                self.finish_failed(tx, *ts, StreamError::InvalidJson(msg), false, 0)
                    .await
            }
        }
    }

    /// The chat-completion request body. Gains a `response_format`
    /// object when [`response_format`](Self::response_format) is set
    /// (Engine C3); otherwise byte-identical to the Engine A body.
    fn request_body(&self) -> serde_json::Value {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": [{ "role": "user", "content": self.prompt }],
            "stream": true,
            "stream_options": { "include_usage": true },
            "temperature": 0,
            "max_tokens": self.max_tokens,
        });
        if let Some(format) = &self.response_format {
            body["response_format"] = serde_json::json!({ "type": format });
        }
        body
    }

    /// Build and send the chat-completion request.
    async fn send_request(&self) -> Result<reqwest::Response, reqwest::Error> {
        let body = self.request_body();
        let mut req = self.client.post(&self.endpoint).json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }
        req.send().await
    }

    /// Emit the terminal `Complete` event and build the outcome.
    ///
    /// `looping` / `loop_excluded_tokens` (v0.1.1) carry the decode-loop
    /// guard's verdict so the engines can exclude a looping stream's
    /// tokens from throughput.
    async fn finish_complete(
        &self,
        tx: &mut mpsc::Sender<StreamEvent>,
        ts: &StreamTimestamps,
        usage: Option<Usage>,
        premature: bool,
        malformed_frames: u64,
        loop_state: (bool, u64),
    ) -> StreamOutcome {
        let (looping, loop_excluded_tokens) = loop_state;
        let event = StreamEvent::Complete {
            timestamps: *ts,
            usage,
            premature,
            malformed_frames,
            looping,
            loop_excluded_tokens,
        };
        let _ = tx.send(event).await;
        StreamOutcome {
            timestamps: *ts,
            usage,
            premature,
            malformed_frames,
            error: None,
            looping,
            loop_excluded_tokens,
        }
    }

    /// Emit the terminal `Failed` event and build the outcome.
    async fn finish_failed(
        &self,
        tx: &mut mpsc::Sender<StreamEvent>,
        ts: StreamTimestamps,
        error: StreamError,
        looping: bool,
        loop_excluded_tokens: u64,
    ) -> StreamOutcome {
        self.logl(
            Level::Error,
            Context::Error,
            &format!("{} failed: {error}", self.tag),
        );
        let event = StreamEvent::Failed {
            timestamps: ts,
            error: error.clone(),
            looping,
            loop_excluded_tokens,
        };
        let _ = tx.send(event).await;
        StreamOutcome {
            timestamps: ts,
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(error),
            looping,
            loop_excluded_tokens,
        }
    }
}

/// Spawn a [`StreamWorker`] on its own task and await its
/// [`StreamOutcome`] — the shared seam every engine uses (A, C1, C2, C3).
///
/// A task panic (an internal bug, not an endpoint fault) is converted
/// into a [`StreamError::Read`] failure outcome instead of unwinding the
/// benchmark — the N/A-never-fail rule: one bad stream is recorded, the
/// run continues, and the process never crashes mid-measurement.
///
/// **Deadlock warning:** this function awaits the worker to completion
/// before the caller can drain the `tx` channel. If the worker produces
/// more frames than the channel capacity, `tx.send()` blocks and the
/// worker never finishes → deadlock. Engines that drain the channel
/// concurrently should use [`spawn_worker`] instead.
pub async fn run_worker(worker: StreamWorker, tx: mpsc::Sender<StreamEvent>) -> StreamOutcome {
    match tokio::spawn(worker.run(tx)).await {
        Ok(outcome) => outcome,
        Err(join) => StreamOutcome {
            timestamps: StreamTimestamps::default(),
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Read(format!("worker task failed: {join}"))),
            looping: false,
            loop_excluded_tokens: 0,
        },
    }
}

/// Spawn a [`StreamWorker`] on its own task and return the
/// [`tokio::task::JoinHandle`] immediately (non-blocking).
///
/// The caller must drain the `tx` channel concurrently with the worker
/// running, then await the handle for the final [`StreamOutcome`]. This
/// avoids the deadlock that `run_worker` causes when the worker produces
/// more frames than the bounded channel capacity.
///
/// A task panic (an internal bug, not an endpoint fault) is converted
/// into a [`StreamError::Read`] failure outcome when the handle is
/// awaited — the N/A-never-fail rule.
pub fn spawn_worker(
    worker: StreamWorker,
    tx: mpsc::Sender<StreamEvent>,
) -> tokio::task::JoinHandle<StreamOutcome> {
    tokio::spawn(worker.run(tx))
}

/// Await a [`spawn_worker`] handle, converting a task panic into a
/// [`StreamError::Read`] failure outcome (N/A-never-fail).
pub async fn join_worker(handle: tokio::task::JoinHandle<StreamOutcome>) -> StreamOutcome {
    match handle.await {
        Ok(outcome) => outcome,
        Err(join) => StreamOutcome {
            timestamps: StreamTimestamps::default(),
            usage: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Read(format!("worker task failed: {join}"))),
            looping: false,
            loop_excluded_tokens: 0,
        },
    }
}

/// The result of one `bytes_stream()` poll with an idle timeout.
enum ChunkRead<B> {
    Bytes(B),
    Eof,
    Timeout,
    Error(String),
}

/// Poll the next body chunk, bounded by `read_timeout` (the idle/stall
/// timeout — it resets on every chunk, so long generations are fine).
async fn read_chunk<S, B>(stream: &mut S, read_timeout: Duration) -> ChunkRead<B>
where
    S: StreamExt<Item = Result<B, reqwest::Error>> + Unpin,
    B: AsRef<[u8]>,
{
    match tokio::time::timeout(read_timeout, stream.next()).await {
        Err(_) => ChunkRead::Timeout,
        Ok(None) => ChunkRead::Eof,
        Ok(Some(Ok(b))) => ChunkRead::Bytes(b),
        Ok(Some(Err(e))) => ChunkRead::Error(e.to_string()),
    }
}

impl StreamWorker {
    /// Stamp and forward the frames produced by one parser `feed` /
    /// `finish`.
    ///
    /// `frame.t_nanos` is set to the arrival time in nanoseconds since
    /// `T0` (the parser leaves it `0` by contract); `T3` is latched on the
    /// first token frame. Each decoded token string is also fed to the
    /// per-stream [`LoopGuard`] (v0.1.1) — the first detection logs a
    /// `[LOOP_GUARD]` warning line. Returns `(done, token_frames)`: `true`
    /// when a terminal `[DONE]` frame was among them, plus the number of
    /// token frames emitted in this call (the worker's log-milestone
    /// counter).
    async fn emit_frames(
        &self,
        frames: &[ParsedFrame],
        at: &MonotonicInstant,
        ts: &mut StreamTimestamps,
        tx: &mut mpsc::Sender<StreamEvent>,
        guard: &mut LoopGuard,
    ) -> Result<(bool, u64), ()> {
        let mut done_seen = false;
        let mut token_frames = 0u64;
        for frame in frames {
            let mut frame = frame.clone();
            frame.t_nanos = ts.t0.map_or(0, |t0| t0.delta_nanos(at));
            if frame.chunk.is_token() {
                token_frames += 1;
                if let Some(text) = frame.chunk.token_text() {
                    if guard.push(text) {
                        self.logl(
                            Level::Warn,
                            Context::Sse,
                            &format!(
                                "[LOOP_GUARD] Stream #{}: repeating pattern detected (32 tokens × 3) — excluded",
                                self.tag
                            ),
                        );
                    }
                }
                if ts.t3.is_none() {
                    ts.t3 = Some(*at);
                }
            }
            if frame.done {
                done_seen = true;
            }
            tx.send(StreamEvent::Frame {
                frame,
                at: *at,
                timestamps: *ts,
            })
            .await
            .map_err(|_| ())?;
        }
        Ok((done_seen, token_frames))
    }
}

/// True when `bytes` begin an SSE `data:` frame (BOM tolerated).
fn looks_like_sse(bytes: &[u8]) -> bool {
    strip_bom(bytes).starts_with(b"data:")
}

/// Strip a leading UTF-8 BOM, if present.
fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&BOM).unwrap_or(bytes)
}

/// Parse a non-streaming chat-completion object: extract the single content
/// string and the top-level `usage` block.
fn parse_plain_completion(body: &[u8]) -> Result<(Option<String>, Option<Usage>), String> {
    let value: serde_json::Value =
        serde_json::from_slice(strip_bom(body)).map_err(|e| e.to_string())?;
    let content = value
        .get("choices")
        .and_then(|c| c.get(0))
        .and_then(|c| c.get("message"))
        .and_then(|m| m.get("content"))
        .and_then(|c| c.as_str())
        .filter(|s| !s.is_empty())
        .map(str::to_owned);
    Ok((content, Usage::from_value(&value)))
}

/// Resolve a user-supplied endpoint to the full chat-completions URL:
///
/// * `http://host:8000` → `http://host:8000/v1/chat/completions`
/// * `http://host:8000/` → same (trailing slashes ignored)
/// * `http://host:8000/v1` → `http://host:8000/v1/chat/completions`
/// * `http://host:8000/v1/chat/completions` → unchanged
pub fn normalize_endpoint(base: &str) -> String {
    let base = base.trim_end_matches('/');
    if base.ends_with(COMPLETIONS_PATH) {
        base.to_string()
    } else if base.ends_with(V1_SUFFIX) {
        format!("{base}/chat/completions")
    } else {
        format!("{base}{COMPLETIONS_PATH}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_bare_host_gets_full_path() {
        assert_eq!(
            normalize_endpoint("http://localhost:8000"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_strips_trailing_slash() {
        assert_eq!(
            normalize_endpoint("http://localhost:8000/"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_v1_base_gets_remaining_path() {
        assert_eq!(
            normalize_endpoint("https://api.example.com/v1"),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn normalize_full_path_unchanged() {
        assert_eq!(
            normalize_endpoint("http://localhost:8000/v1/chat/completions"),
            "http://localhost:8000/v1/chat/completions"
        );
    }

    #[test]
    fn plain_json_extracts_content_and_usage() {
        let body = br#"{
            "id": "cmpl-pj",
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": "Hello, world!" },
                "finish_reason": "stop"
            }],
            "usage": { "prompt_tokens": 42, "completion_tokens": 7 }
        }"#;
        let (content, usage) = parse_plain_completion(body).unwrap();
        assert_eq!(content.as_deref(), Some("Hello, world!"));
        assert_eq!(
            usage,
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 7
            })
        );
    }

    #[test]
    fn plain_json_without_content_or_usage_yields_none() {
        let (content, usage) = parse_plain_completion(br#"{"choices": []}"#).unwrap();
        assert_eq!(content, None);
        assert_eq!(usage, None);
    }

    #[test]
    fn plain_json_invalid_body_is_an_error() {
        assert!(parse_plain_completion(b"this is not json").is_err());
    }

    #[test]
    fn sse_detection_with_and_without_bom() {
        assert!(looks_like_sse(b"data: {\"a\":1}\n\n"));
        assert!(looks_like_sse(&[
            0xEF, 0xBB, 0xBF, b'd', b'a', b't', b'a', b':'
        ]));
        assert!(!looks_like_sse(b"{\"choices\":[]}"));
        assert!(!looks_like_sse(b"ev"));
    }

    #[test]
    fn only_connection_errors_are_retriable() {
        assert!(StreamError::Connection("refused".into()).is_retriable());
        assert!(!StreamError::Http {
            status: 500,
            body: "boom".into()
        }
        .is_retriable());
        assert!(!StreamError::Timeout(Duration::from_secs(1)).is_retriable());
        assert!(!StreamError::WorkerTimeout(Duration::from_secs(120)).is_retriable());
        assert!(!StreamError::Read("reset".into()).is_retriable());
        assert!(!StreamError::InvalidJson("nope".into()).is_retriable());
    }

    #[test]
    fn truncate_body_respects_char_boundaries() {
        let s = "héllo wörld"; // multi-byte chars
        let t = truncate_body(s, 5);
        // 5 bytes + the 3-byte ellipsis, never splitting a char.
        assert!(t.len() <= 8);
        assert!(t.ends_with('…'));
        // A cut landing mid-character backs up to the previous boundary.
        let mid = truncate_body(s, 2); // 2 is inside the 2-byte `é`
        assert!(mid.len() <= 5);
        assert_eq!(truncate_body("short", 100), "short");
    }

    #[test]
    fn chunk_is_token_covers_reasoning_and_content_only() {
        assert!(Chunk::Reasoning("x".into()).is_token());
        assert!(Chunk::Content("x".into()).is_token());
        assert!(!Chunk::Control.is_token());
        assert!(!Chunk::Usage(Usage::default()).is_token());
    }

    #[test]
    fn request_body_without_response_format_is_the_engine_a_shape() {
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        );
        let body = w.request_body();
        assert!(body.get("response_format").is_none());
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["max_tokens"], 10);
        assert_eq!(body["stream_options"]["include_usage"], true);
    }

    #[test]
    fn request_body_with_response_format_adds_the_directive() {
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .response_format("json_object");
        let body = w.request_body();
        assert_eq!(body["response_format"]["type"], "json_object");
        // The rest of the body is untouched.
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 10);
    }

    #[test]
    fn stream_end_log_labels_frames_and_tokens_correctly() {
        let dir = std::env::temp_dir().join(format!("crucible-streamend-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .logger(logger.clone())
        .tag("test");

        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 100 ms
            mock.increment(100_000_000);
            let t_end = MonotonicInstant::now(); // 200 ms

            // Server reports usage: the authoritative token count (256)
            // is logged alongside the 119 counted frames; the rate is
            // 256 / 0.1 s over the T3→Tn window.
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t3),
                    t_end: Some(t_end),
                },
                119,
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 256,
                }),
                false,
            );
            // No usage: the frame count is only an estimate.
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t0),
                    t_end: Some(t_end),
                },
                119,
                None,
                false,
            );
            // A premature close keeps the marker.
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t3),
                    t_end: Some(t_end),
                },
                119,
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 256,
                }),
                true,
            );
        });
        logger.finish();
        // The writer flushes on every newline; read after finish.
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        assert!(
            latest.contains("test stream complete: 256 tokens (119 frames) in 0.2s (2560.0 t/s)"),
            "usage line missing: {latest}"
        );
        assert!(
            latest.contains("test stream complete: 119 frames in 0.2s (~595.0 t/s est)"),
            "estimate line missing: {latest}"
        );
        assert!(
            latest.contains("256 tokens (119 frames) in 0.2s (2560.0 t/s) (premature close)"),
            "premature marker missing: {latest}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }
}
