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
//!    * `T1` — response headers received (`send()` resolves when the
//!      server's headers arrive — the request is accepted; this is NOT the
//!      moment the last request byte left the socket, so the T1→T3 window
//!      excludes the request-send network leg);
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
use crate::prompt::{count_tokens, Tokenizer};
use crate::sse::{Chunk, ParsedFrame, SseParser, Usage, UsageDetails};
use crate::timing::{MonotonicInstant, StreamTimestamps};

use super::loop_guard::LoopGuard;
use super::truncate_body;

/// Completions path appended to a bare host / base URL.
const COMPLETIONS_PATH: &str = "/v1/chat/completions";

/// A base URL that already ends in `/v1` only needs the rest of the path.
const V1_SUFFIX: &str = "/v1";

/// UTF-8 byte-order mark (some servers prefix the stream with one).
const BOM: [u8; 3] = [0xEF, 0xBB, 0xBF];

/// Cap on the response body retained on the SSE path for end-of-stream
/// inspection (the JSON rescue / [`StreamError::InvalidTransport`] check).
/// A normal stream confirms itself with its first valid `data:` frame, so the
/// buffer only grows large for degenerate (zero-valid-frame) bodies — at most
/// a JSON completion (a few KB) or garbage. 1 MiB bounds the worst case
/// without touching normal-stream memory.
const PENDING_BODY_CAP: usize = 1 << 20;

/// Default idle (read) timeout: a stream that produces no bytes for this
/// long is declared stalled.
pub const DEFAULT_READ_TIMEOUT: Duration = Duration::from_secs(30);

/// The minimum **measurable** decode window (10 ms) — the H1/H1b/M1
/// floor shared by every decode-rate computation in the tool.
///
/// A token window shorter than this — a 1–2-token answer committed in a
/// single MTP / speculative-decoding batch, where `first ≈ last` — is not
/// a measurable decode cadence. Dividing by a clamped sliver (the old
/// `.max(0.001)`) mints phantom 10³–10⁴ t/s figures; rates over such
/// windows are reported as the CF-25 `0.0` N/A sentinel (log lines:
/// `n/a`), never as a clamped-divisor number. 10 ms is well below any
/// real inter-token gap, so a genuinely fast decode is never masked.
pub const MIN_MEASURABLE_WINDOW_SECS: f64 = 0.010;

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
    /// The response body was **neither a valid SSE stream nor valid JSON**.
    ///
    /// This is a genuine transport/protocol fault, not a measurement: the
    /// worker asked for `stream: true`, the server answered `200`, but the
    /// body carried no parseable `data:` frame (so it is not usable SSE) and
    /// does not decode as a JSON completion (so it is not the non-streaming
    /// fallback). The canonical cause it guards against is a stream whose
    /// preamble (keep-alive `:comment` lines, `event:`/`id:`/`retry:` fields)
    /// is followed by nothing but garbage — which the old first-byte sniff
    /// would have misrouted to the JSON path and reported as a spurious
    /// "retrieval miss". Surfacing it as a real error keeps a broken
    /// endpoint out of the throughput/accuracy results.
    #[error("response body is neither valid SSE nor valid JSON: {0}")]
    InvalidTransport(String),
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
    /// The CF-02 `usage` detail fields (reasoning / cached tokens) when the
    /// stream reported a `usage` block, else `None` (a backend that does not
    /// report the details — the honest "not reported" case). This is the seam
    /// that lets Engine A's `aggregate()` split the decode window (CF-26) and
    /// let persistence populate `stream_metrics.reasoning_tokens`.
    pub usage_details: Option<UsageDetails>,
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
    /// The number of token-bearing SSE frames the client **observed** on the
    /// wire (a [`Chunk::Reasoning`] or [`Chunk::Content`] delta per frame).
    ///
    /// This is the ground-truth output count for a decode-rate measurement:
    /// it is what actually arrived, independent of the server's self-reported
    /// [`Usage::completion_tokens`]. Some servers (notably Unsloth /
    /// llama.cpp GGUF builds) report `completion_tokens` as the requested
    /// `max_tokens` target rather than the tokens they generated, which
    /// inflates a `completion_tokens / decode-window` rate by a wide margin.
    /// `0` when no token frames were observed (a failure, an empty body, or a
    /// non-streaming plain-JSON response that is counted separately).
    pub frames: u64,
}

impl StreamOutcome {
    /// `true` when the run completed without an error (including
    /// `premature` completions, which preserve partial data).
    pub fn is_ok(&self) -> bool {
        self.error.is_none()
    }

    /// **Hybrid-reasoning detection (R1) — `truncated_by_reasoning`.**
    ///
    /// `true` when the stream **completed OK** ([`is_ok`](Self::is_ok) — not a
    /// transport / HTTP / read / parse failure) but the model spent its entire
    /// generation budget on chain-of-thought: it emitted at least one
    /// [`Chunk::Reasoning`] frame and **zero** [`Chunk::Content`] frames.
    ///
    /// This is the "budget exhausted in CoT, no answer produced" condition.
    /// The frame split is read from the stream's `events` — the wire truth the
    /// SSE parser already classifies into [`Chunk::Reasoning`] /
    /// [`Chunk::Content`] — via [`count_reasoning_content_frames`]. It works on
    /// **every** server with no `usage` dependency, unlike
    /// `UsageDetails::is_reasoning_dominated` (which needs the server to report
    /// `reasoning_tokens`; config-only servers may omit it).
    ///
    /// The [`is_ok`](Self::is_ok) gate is essential: a genuine transport
    /// failure is a *fail*, not a `truncated_by_reasoning`. Only a *completed*
    /// stream that produced all-CoT / zero-content is the distinct, non-fail
    /// outcome the C1/C3 capability engines report (Option C,
    /// `work/plans/REASONING_DESIGN.md`) — an honest "inconclusive (budget)"
    /// instead of a false capability failure.
    ///
    /// `events` is the run's drained `Vec<StreamEvent>` (the same collection
    /// the engines already keep alongside this outcome — see
    /// `speed.rs::aggregate`, which takes both).
    #[must_use]
    pub fn is_truncated_by_reasoning(&self, events: &[StreamEvent]) -> bool {
        if !self.is_ok() {
            return false;
        }
        let (reasoning, content) = count_reasoning_content_frames(events);
        reasoning > 0 && content == 0
    }

    /// **CF-10 — usage-vs-frames plausibility guard.**
    ///
    /// `true` when the server's self-reported `usage.completion_tokens`
    /// exceeds ~4× the token frames actually observed on the wire. Some
    /// backends (notably certain Unsloth / llama.cpp GGUF builds) report
    /// `completion_tokens` as the *requested* `max_tokens` target rather than
    /// the tokens they generated — trusting that figure inflates a
    /// `completion_tokens / decode-window` rate by up to ~400×. When the guard
    /// trips, [`effective_completion`](Self::effective_completion) and
    /// [`authoritative_tokens`] prefer the observed frame count (the wire
    /// truth) over the inflated usage.
    ///
    /// `false` when no usage arrived, when the usage is plausible relative to
    /// the frames, or when no token frames were observed (nothing to compare
    /// against).
    #[must_use]
    pub fn has_usage_discrepancy(&self) -> bool {
        let Some(u) = &self.usage else {
            return false;
        };
        self.frames > 0 && u.completion_tokens > self.frames.saturating_mul(4)
    }

    /// **CF-44 — non-streaming (plain-JSON) results are E2E-only.**
    ///
    /// `true` for a result that came from the non-streaming fallback
    /// ([`run_plain_json`]): a server that ignored `stream: true` and returned
    /// a single completion object. The tell is `frames == 0` (a plain-JSON
    /// blob is one frame, not a token count) with a `usage` present and no
    /// error. Such a result's "TTFT" (`T3 − T1`) is really the *total*
    /// response latency — `T3` is stamped only after the whole body has been
    /// read — so it must be labeled E2E-only and kept out of the ITL / decode
    /// distributions (a single blob frame produces no inter-token deltas and
    /// a zero decode span).
    #[must_use]
    pub fn is_e2e_only(&self) -> bool {
        self.frames == 0 && self.usage.is_some() && self.error.is_none()
    }

    /// The authoritative completion-token count for a decode-rate
    /// measurement: the server-reported [`Usage::completion_tokens`] when it
    /// arrived and is plausible (the **primary** source — the server's own
    /// count), else the frames observed on the wire as a last-resort
    /// estimate.
    ///
    /// The server's `usage` is authoritative because backends that batch
    /// several tokens into one SSE frame (vLLM MTP / speculative decoding)
    /// make the raw frame count understate the true output by 30–40 %. The
    /// frame count is used when no `usage` chunk arrived at all (an aborted
    /// stream from a non-compliant proxy) **or** when the usage fails the
    /// CF-10 plausibility guard (it exceeds ~4× the observed frames — a
    /// max_tokens-inflating backend).
    #[must_use]
    pub fn effective_completion(&self) -> u64 {
        match self.usage.map(|u| u.completion_tokens).filter(|&n| n > 0) {
            Some(n) if !self.has_usage_discrepancy() => n,
            _ => self.frames,
        }
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
    /// The OpenAI-compatible `response_format` object — Engine C3's
    /// constrained-decoding runs. Two forms are supported:
    /// * a simple type hint, `{"type": "json_object"}` (set via
    ///   [`response_format`](Self::response_format));
    /// * a full `json_schema` object, `{"type": "json_schema",
    ///   "json_schema": {name, schema, strict}}` (set via
    ///   [`response_format_value`](Self::response_format_value)) — the form
    ///   that engages server-side constrained decoding (XGrammar / Outlines /
    ///   GBNF) on vLLM / SGLang.
    ///
    /// `None` (the default) sends the body unchanged. Backends that do not
    /// support `response_format` (llama.cpp, TGI) drop the field per the
    /// OpenAI spec — a graceful no-op that never breaks the request.
    response_format: Option<serde_json::Value>,
    /// When `true`, the body gains `"ignore_eos": true` (a llama.cpp
    /// server parameter): the model is not allowed to stop on its own
    /// end-token, so generation runs until `max_tokens` — the engine's
    /// own stop condition (Engine F's 60-second window). Servers that do
    /// not know the field (vLLM, LM Studio, …) ignore it per the OpenAI
    /// spec, so the request degrades to the model's natural behavior.
    ignore_eos: bool,
    /// When `true`, the request asks a reasoning model to answer without a
    /// chain-of-thought monologue (CF-01 — the reasoning-model blind spot).
    ///
    /// **This is a *best-effort speed optimization*, NOT a correctness
    /// dependency.** The mechanism is **per-backend and graceful** (each
    /// backend honors the one it knows and ignores the rest per the OpenAI
    /// spec, so a backend with no switch degrades to the model's natural
    /// behavior):
    ///
    /// * `chat_template_kwargs: {"enable_thinking": false}` — vLLM / SGLang /
    ///   LM Studio (the Jinja chat template turns off CoT);
    /// * `think: false` — Ollama's extension field;
    /// * a trailing `/no_think` in the user message — llama.cpp (Qwen3's
    ///   native template checks for it).
    ///
    /// Where a server honors one of these, generation skips CoT and the
    /// visible answer fits a small budget — a pure speed/cost win. Where a
    /// server ignores them (config-only inference servers, and Qwen3.5+ where
    /// `/no_think` is not a guaranteed switch), the request degrades
    /// gracefully to the model's natural (thinking-on) behavior — harmless
    /// per the OpenAI spec's drop-unknown-fields rule (the same pattern as
    /// `response_format` / `ignore_eos`).
    ///
    /// **Correctness therefore rests on the generation budget (`max_tokens`),
    /// which every OpenAI-compatible server honors — not on this switch.** A
    /// raised budget lets a verbose model finish its CoT and *still* emit
    /// content; if it over-thinks and commits nothing, the stream is reported
    /// `truncated_by_reasoning` (see
    /// [`StreamOutcome::is_truncated_by_reasoning`] /
    /// [`count_reasoning_content_frames`]) — an honest, non-fail outcome
    /// rather than the false capability failure this once caused (Engine C1's
    /// 64-token NIAH probe, Strawberry Issue 1).
    disable_thinking: bool,
    /// Optional run-log writer: when present, the worker logs its HTTP /
    /// SSE lifecycle (request, TTFB, first token, every 50th token,
    /// completion / failure) to the file-based [`RunLogger`].
    logger: Option<Arc<RunLogger>>,
    /// A short identifier for this worker's log lines (e.g. `A:2` for
    /// Engine A iteration 2, `B5:3` for Engine B step 5 worker 3).
    tag: String,
    /// H2 — the engine's optional tokenizer, threaded into the worker so
    /// the per-stream `stream complete` line counts with the **same
    /// method** as the engine's summary ([`authoritative_count`]: the
    /// server's `usage` first, else a re-tokenization of the stream's
    /// text — this tokenizer when attached, else `chars/4`, flagged
    /// `≈`). Without it, a configured `--tokenizer` made the summary and
    /// the per-stream line read different numbers on the same stream.
    tokenizer: Option<Arc<Tokenizer>>,
    /// N11 — the `[CONTEXT]` a *failure* line carries (the worker's
    /// engine): `finish_failed` logs `Level::Error` with **this**
    /// context instead of `Context::Error` — the old `[ERROR] [ERROR]`
    /// double tag lost the engine attribution. `Context::Sse` (the
    /// worker's own subsystem) stands when the engine is unknown; the
    /// message's tag (`C2:…` / `B5-3` / …) disambiguates.
    context: Context,
    /// P2/N14 — an optional label for a *deliberate* early close,
    /// overriding the generic `(premature close)` tag on the
    /// completion line. Engine F sets `"60s window close"`: Flat Out
    /// aborts every stream at the 60-second window (an EOF without
    /// `[DONE]` is the designed stop, not a fault). A genuine
    /// mid-stream drop (no note) still reads `(premature close)`.
    close_note: Option<String>,
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
            ignore_eos: false,
            disable_thinking: false,
            logger: None,
            tag: "stream".to_string(),
            tokenizer: None,
            context: Context::Sse,
            close_note: None,
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
    ///
    /// This is the simple *type-hint* form. For the full `json_schema`
    /// object (which engages real grammar-constrained decoding), use
    /// [`response_format_value`](Self::response_format_value).
    pub fn response_format(mut self, format: &str) -> Self {
        self.response_format = Some(serde_json::json!({ "type": format }));
        self
    }

    /// Set a full OpenAI-compatible `response_format` **object** (Engine C3's
    /// `json_schema` constrained decoding, CF-08): the body's
    /// `response_format` field is set to `value` verbatim, e.g.
    /// `{"type": "json_schema", "json_schema": {name, schema, strict}}`.
    ///
    /// This is the wire encoding that lets a schema-constrained run be
    /// transmitted end-to-end — the `{"type": <string>}` form of
    /// [`response_format`](Self::response_format) cannot carry the
    /// `json_schema` sub-object. Backends that do not support
    /// `response_format` drop the field per the OpenAI spec — a graceful
    /// no-op that never breaks the request.
    pub fn response_format_value(mut self, value: impl Into<serde_json::Value>) -> Self {
        self.response_format = Some(value.into());
        self
    }

    /// Disallow the model's own end-token from stopping generation
    /// (`"ignore_eos": true`, honored by llama.cpp servers; unknown to —
    /// and ignored by — other backends). Engine F sets this so the
    /// 60-second window, not the model, is the only stop.
    pub fn ignore_eos(mut self, v: bool) -> Self {
        self.ignore_eos = v;
        self
    }

    /// Ask a reasoning model to suppress its chain-of-thought (CF-01) — a
    /// **best-effort speed optimization**, not a correctness requirement.
    ///
    /// Injects the per-backend switches (see the `disable_thinking` field
    /// doc): `chat_template_kwargs.enable_thinking=false` (vLLM / SGLang /
    /// LM Studio), `think=false` (Ollama), and a trailing `/no_think` in the
    /// user message (llama.cpp). Where a server honors one, CoT is skipped
    /// and the visible answer fits a small budget (fast + cheap). Backends
    /// without a switch ignore the unknown fields per the OpenAI spec, so the
    /// request degrades gracefully to the model's natural behavior (the same
    /// pattern as [`ignore_eos`](Self::ignore_eos)) — non-reasoning models
    /// and config-only servers are unaffected.
    ///
    /// **Correctness does not depend on this:** it rests on the generation
    /// budget (`max_tokens`), which every server honors. When suppression is
    /// ignored and a verbose model over-thinks past the budget, the result is
    /// reported `truncated_by_reasoning` (see
    /// [`StreamOutcome::is_truncated_by_reasoning`]), not a false failure.
    /// Engine C1 sets this on its NIAH probe.
    pub fn disable_thinking(mut self, v: bool) -> Self {
        self.disable_thinking = v;
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

    /// H2 — attach the engine's optional tokenizer: the completion
    /// line's no-`usage` count re-tokenizes the stream's text with it
    /// (the same source as the engine's summary, so the two reconcile;
    /// `None` keeps the `chars/4` fallback, flagged `≈`).
    pub fn tokenizer(mut self, tokenizer: Option<Arc<Tokenizer>>) -> Self {
        self.tokenizer = tokenizer;
        self
    }

    /// N11 — the engine's log context for this worker's failure lines
    /// (`finish_failed` uses it instead of `Context::Error`, so a failed
    /// stream reads `[ERROR] [ENGINE_C2] C2:… failed: …` — engine
    /// attribution restored, the `[ERROR] [ERROR]` double tag gone).
    pub fn context(mut self, ctx: Context) -> Self {
        self.context = ctx;
        self
    }

    /// P2/N14 — label a deliberate early close (e.g. Engine F's 60 s
    /// window): the completion line prints `(<note>)` instead of the
    /// fault-reading `(premature close)`.
    pub fn close_note(mut self, note: Option<String>) -> Self {
        self.close_note = note;
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
        // T1 — response headers received. `reqwest`'s `send()` resolves when
        // the server's response headers arrive (the request is accepted and
        // the server begins responding); it is NOT stamped when the last
        // request byte is written to the socket. The T1→T3 window therefore
        // excludes the request-send network leg and approximates the server's
        // queue/schedule + prefill + first-decode time (the "server-side"
        // TTFT, [`StreamTimestamps::ttft_server_nanos`]); T0→T3 is the
        // "perceived" TTFT (the full client-visible round-trip,
        // [`StreamTimestamps::ttft_perceived_nanos`]).
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

        // The response shape is decided by the `Content-Type` header and our
        // request intent — NOT by sniffing the first data byte (which broke on
        // SSE keep-alive `:comment` preambles; see `route_is_sse`, Issue #1).
        let content_type = resp
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let mut stream = resp.bytes_stream();
        let mut parser = SseParser::new();

        // Read the first body chunk: it stamps T2 (first byte) and surfaces
        // the empty-body / timeout / read-error terminal cases. The routing
        // decision no longer depends on these bytes.
        let head: Vec<u8> = match read_chunk(&mut stream, self.read_timeout).await {
            ChunkRead::Bytes(b) => b.to_vec(),
            ChunkRead::Eof => {
                // Empty 200 body: nothing to measure.
                ts.t_end = Some(MonotonicInstant::now());
                return self
                    .finish_complete(
                        tx,
                        &ts,
                        None,
                        None,
                        false,
                        parser.malformed_frames(),
                        (false, 0),
                        0,
                    )
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

        if route_is_sse(&content_type) {
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
        // C3: latch the `token #1` milestone (a keep-alive batch keeps the
        // cumulative count at 1 — the arm must fire at most once).
        let mut first_logged = false;
        // C3 residual: latch each every-50 milestone (the index `tokens / 50`
        // must strictly increase) so a whitespace-only batch that keeps the
        // count at a 50-multiple cannot re-fire it.
        let mut last_logged_50 = 0u64;
        let mut last_token_at: Option<MonotonicInstant> = None;
        // H2: the stream's text (reasoning + content, arrival order) —
        // the completion line's no-`usage` count re-tokenizes it (the
        // same source the engine's summary uses).
        let mut stream_text = String::new();
        // Retain the body (bounded) so a stream that ends with *zero* valid
        // `data:` frames can be inspected: rescued as a non-streaming JSON
        // completion, or surfaced as a real [`StreamError::InvalidTransport`]
        // (Issue #1 — never a silent retrieval miss).
        let mut pending_body: Vec<u8> = Vec::new();
        let mut body_dropped = false;
        extend_pending_body(&mut pending_body, &mut body_dropped, head.as_ref());
        let mut done = match self
            .emit_frames(
                parser.feed(head.as_ref()),
                &head_at,
                ts,
                tx,
                &mut guard,
                &mut last_token_at,
                &mut stream_text,
            )
            .await
        {
            Ok((d, tc)) => {
                tokens += tc;
                self.log_token_milestones(ts, tokens, &mut first_logged, &mut last_logged_50);
                d
            }
            Err(_) => {
                // Receiver gone: stop measuring.
                ts.t_end = Some(MonotonicInstant::now());
                let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                return self
                    .resolve_sse_end(
                        tx,
                        ts,
                        true,
                        parser,
                        &pending_body,
                        body_dropped,
                        (looping, excluded),
                        tokens,
                        &last_token_at,
                        &stream_text,
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
                        .emit_frames(
                            parser.finish(),
                            &at,
                            ts,
                            tx,
                            &mut guard,
                            &mut last_token_at,
                            &mut stream_text,
                        )
                        .await
                    {
                        tokens += tc;
                    }
                    ts.t_end = Some(at);
                    let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                    return self
                        .resolve_sse_end(
                            tx,
                            ts,
                            true,
                            parser,
                            &pending_body,
                            body_dropped,
                            (looping, excluded),
                            tokens,
                            &last_token_at,
                            &stream_text,
                        )
                        .await;
                }
                ChunkRead::Error(msg) => {
                    if ts.t2.is_some() {
                        // Premature close mid-stream: a broken `chunked`
                        // close or connection reset. Flush the pending
                        // frame, keep every captured timestamp.
                        if let Ok((_, tc)) = self
                            .emit_frames(
                                parser.finish(),
                                &at,
                                ts,
                                tx,
                                &mut guard,
                                &mut last_token_at,
                                &mut stream_text,
                            )
                            .await
                        {
                            tokens += tc;
                        }
                        ts.t_end = Some(at);
                        let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                        return self
                            .resolve_sse_end(
                                tx,
                                ts,
                                true,
                                parser,
                                &pending_body,
                                body_dropped,
                                (looping, excluded),
                                tokens,
                                &last_token_at,
                                &stream_text,
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
                    extend_pending_body(&mut pending_body, &mut body_dropped, b.as_ref());
                    match self
                        .emit_frames(
                            parser.feed(b.as_ref()),
                            &at,
                            ts,
                            tx,
                            &mut guard,
                            &mut last_token_at,
                            &mut stream_text,
                        )
                        .await
                    {
                        Ok((d, tc)) => {
                            done = d;
                            tokens += tc;
                            self.log_token_milestones(
                                ts,
                                tokens,
                                &mut first_logged,
                                &mut last_logged_50,
                            );
                        }
                        Err(_) => {
                            ts.t_end = Some(at);
                            let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
                            return self
                                .resolve_sse_end(
                                    tx,
                                    ts,
                                    !done,
                                    parser,
                                    &pending_body,
                                    body_dropped,
                                    (looping, excluded),
                                    tokens,
                                    &last_token_at,
                                    &stream_text,
                                )
                                .await;
                        }
                    }
                }
            }
        }

        // `[DONE]` received: Tn — stream close.
        ts.t_end = Some(MonotonicInstant::now());
        let (looping, excluded) = loop_state(&guard, parser.usage(), tokens);
        self.resolve_sse_end(
            tx,
            ts,
            false,
            parser,
            &pending_body,
            body_dropped,
            (looping, excluded),
            tokens,
            &last_token_at,
            &stream_text,
        )
        .await
    }

    /// Log the first token and every 50th token of the stream (the
    /// per-token hot path stays allocation-free: at most one formatted
    /// line per 50 tokens per worker).
    ///
    /// C3 (noise cleanup):
    /// * `tokens == 0` returns early — a keep-alive batch (`tc == 0`) used
    ///   to hit `0.is_multiple_of(50)` and log a meaningless
    ///   `token #0 received (… rate: 0.0 t/s)` line;
    /// * `first_logged` latches the `token #1` milestone —
    ///   whitespace-only delta frames keep the cumulative count at 1, and
    ///   the old `tokens == 1` arm re-fired on every such batch (3× per
    ///   C2 logic challenge in the v0.1.7 log). The latch is a `&mut bool`
    ///   parameter so [`run_sse`] stays field-free;
    /// * `last_logged_50` latches each every-50 milestone the same way —
    ///   the milestone index (`tokens / 50`) must strictly increase before
    ///   it logs, so a whitespace-only batch that keeps the count at a
    ///   50-multiple (the residual `#100` re-fire seen in the v0.1.7 live
    ///   verify) is a no-op, while genuine advancement (50 → 100 → 150)
    ///   logs each new milestone once.
    fn log_token_milestones(
        &self,
        ts: &StreamTimestamps,
        tokens: u64,
        first_logged: &mut bool,
        last_logged_50: &mut u64,
    ) {
        if tokens == 0 {
            return;
        }
        if tokens == 1 {
            if *first_logged {
                return;
            }
            *first_logged = true;
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
            // C3 residual: latch each 50-multiple so a whitespace-only delta
            // batch (which keeps the cumulative count at the same 50-multiple)
            // cannot re-fire it. The milestone index `tokens / 50` must
            // strictly increase before it logs.
            let milestone = tokens / 50;
            if milestone <= *last_logged_50 {
                return;
            }
            *last_logged_50 = milestone;
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

    /// Log the stream's completion.
    ///
    /// The count is the **authoritative** one ([`authoritative_count`] —
    /// the single source of truth every engine summary uses, so the
    /// per-stream line and the summary always read the same number, H2):
    /// the server's `usage.completion_tokens` when it arrived (the
    /// primary method — the server's own count), else a re-tokenization
    /// of the stream's reasoning + content text (the worker's attached
    /// tokenizer when present, else the `chars/4` estimate — flagged
    /// `≈`), which only happens when a non-compliant proxy omits
    /// `usage` entirely.
    ///
    /// The line is **self-consistent** (H5): the wall span (`total_s`,
    /// `T0` → last token — includes TTFB/prefill) and the **decode
    /// window** (first token `T3` → last token — the window the decode
    /// rate is measured over, excluding prefill/TTFT and the trailing
    /// gap to `[DONE]`, matching how every backend measures its own
    /// decode speed) are disclosed separately:
    ///
    /// `{tag} stream complete: {count} tokens in {total}s
    /// (decode {rate} t/s; TTFB {ttfb}s)`
    ///
    /// with `TTFB = total − decode window` (`T0` → `T3`), so a reader
    /// can verify the figure straight off the line:
    /// `count / (total − TTFB) ≈ decode rate`.
    ///
    /// A decode window below [`MIN_MEASURABLE_WINDOW_SECS`] (a
    /// 1–2-token burst committed in one MTP / speculative-decoding
    /// batch — the span is sub-millisecond, not a real decode cadence)
    /// reports `decode n/a` (the CF-25 N/A convention, M1) — never a
    /// clamped-divisor phantom rate, and never a misleading bare
    /// `0.0 t/s`.
    fn log_stream_end(
        &self,
        ts: &StreamTimestamps,
        usage: Option<Usage>,
        last_token_at: Option<MonotonicInstant>,
        premature: bool,
        text: &str,
    ) {
        // H2: one counting method for the per-stream line and the
        // engine's summary (usage first; else re-tokenize the stream's
        // text — exact tokenizer when attached, else `chars/4`).
        let (count, estimated) = authoritative_count(usage, text, self.tokenizer.as_deref());
        // H5: the three windows, disclosed separately. `total` = T0 →
        // last (the wall span, includes TTFB/prefill); `decode` = T3 →
        // last (the decode window the rate is measured over); `ttfb` =
        // total − decode (T0 → T3), so the printed rate is derivable:
        // count / (total − TTFB) = rate.
        let (total_s, decode_s, ttfb_s) = match (ts.t0, ts.t3, last_token_at) {
            (Some(t0), Some(t3), Some(last)) => {
                let total = t0.delta_nanos(&last) as f64 / 1e9;
                let decode = t3.delta_nanos(&last) as f64 / 1e9;
                (total, decode, (total - decode).max(0.0))
            }
            (None, Some(t3), Some(last)) => {
                // T0 unrecorded: the wall span IS the decode window.
                let decode = t3.delta_nanos(&last) as f64 / 1e9;
                (decode, decode, 0.0)
            }
            _ => (0.0, 0.0, 0.0),
        };
        // M1: a sub-floor decode window (single token / sub-ms MTP
        // batch) is not a measurable rate — `n/a`, never a clamped
        // phantom, never a misleading `0.0 t/s`.
        let rate = (decode_s >= MIN_MEASURABLE_WINDOW_SECS).then(|| count as f64 / decode_s);
        // N23: one approximation glyph (`≈`) for every estimated figure.
        let approx = if estimated { "≈" } else { "" };
        let rate_str = match rate {
            Some(r) => format!("{approx}{r:.1} t/s"),
            None => "n/a".to_string(),
        };
        let est_note = if estimated { " (est, no usage)" } else { "" };
        // P2/N14: a deliberate early close (Engine F's 60 s window)
        // carries its own label; a genuine mid-stream drop keeps the
        // generic fault marker.
        let tag = if premature {
            self.close_note
                .as_deref()
                .map(|n| format!(" ({n})"))
                .unwrap_or_else(|| " (premature close)".to_string())
        } else {
            String::new()
        };
        let msg = format!(
            "{} stream complete: {approx}{count} tokens{est_note} in {total_s:.1}s (decode {rate_str}; TTFB {ttfb_s:.1}s){tag}",
            self.tag
        );
        self.logl(Level::Info, Context::Sse, &msg);
    }

    /// The non-SSE plain-JSON fallback: a server that ignored `stream: true`
    /// and returned a single completion object. Extract usage + one content
    /// chunk (plan Chunk 5).
    ///
    /// **CF-44 — these results are E2E-only, not streaming measurements.**
    /// The whole body is read *before* it is parsed, so `T3` (the "first
    /// token") is stamped at `parsed_at` — after the entire response has
    /// arrived. The resulting "TTFT" (`T3 − T1`) is therefore the *total*
    /// response latency, not a first-token latency. The outcome is labeled
    /// E2E-only via [`StreamOutcome::is_e2e_only`] (a plain-JSON blob reports
    /// `frames == 0` with a `usage` and no error). It is kept out of the ITL /
    /// decode distributions automatically: a single blob frame produces no
    /// inter-token deltas (ITL needs ≥2 token frames) and a zero decode span
    /// (`first == last`), so there is nothing to pool into the ITL histogram
    /// or the decode rate.
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
        let parsed_at = MonotonicInstant::now();
        match parse_plain_completion(&body) {
            Ok((content, usage)) => {
                self.emit_completion_frames(content, usage, parsed_at, ts, tx)
                    .await
            }
            Err(msg) => {
                ts.t_end = Some(parsed_at);
                self.finish_failed(tx, *ts, StreamError::InvalidJson(msg), false, 0)
                    .await
            }
        }
    }

    /// Emit the frames of a complete non-streaming JSON completion and return
    /// the terminal [`StreamOutcome`].
    ///
    /// Shared by [`run_plain_json`] (the content-type-routed path) and the SSE
    /// path's end-of-stream rescue (a JSON body that reached [`run_sse`]
    /// because the content type was absent/ambiguous).
    ///
    /// **CF-44 — these results are E2E-only, not streaming measurements.**
    /// The whole body is parsed *after* it has fully arrived, so `T3` (the
    /// "first token") is stamped at `parsed_at` — the *total* response
    /// latency, not a first-token latency. A single blob frame yields no ITL
    /// deltas and a zero decode span, so the result is labeled E2E-only via
    /// [`StreamOutcome::is_e2e_only`] and kept out of the ITL / decode
    /// distributions. The plain-JSON path does not parse the CF-02 detail
    /// fields, so `usage_details` is `None` (honest "not reported").
    async fn emit_completion_frames(
        &self,
        content: Option<String>,
        usage: Option<Usage>,
        parsed_at: MonotonicInstant,
        ts: &mut StreamTimestamps,
        tx: &mut mpsc::Sender<StreamEvent>,
    ) -> StreamOutcome {
        // A single non-streaming completion cannot loop (at most one token
        // frame), but the guard is fed for a uniform `emit_frames` contract.
        let mut guard = LoopGuard::new();
        let mut last_token_at: Option<MonotonicInstant> = None;
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
        // `emit_frames` stamps `t_nanos` and sets T3 on the first token
        // (content) frame. A non-streamed blob is a single frame, so its frame
        // count is not a meaningful token count — the server's `usage` is.
        // Report `0` observed frames so the downstream rate falls back to
        // `usage.completion_tokens`.
        // H2: the blob's text is accumulated as the (single) content frame is
        // emitted — the same source the no-`usage` count re-tokenizes.
        let mut stream_text = String::new();
        let tokens = match self
            .emit_frames(
                &frames,
                &parsed_at,
                ts,
                tx,
                &mut guard,
                &mut last_token_at,
                &mut stream_text,
            )
            .await
        {
            Ok(_) => 0,
            Err(_) => {
                // Receiver gone before the token count is known: no frames are
                // confirmed, so report `0`.
                ts.t_end = Some(parsed_at);
                return self
                    .finish_complete(tx, ts, usage, None, true, 0, (false, 0), 0)
                    .await;
            }
        };
        ts.t_end = Some(MonotonicInstant::now());
        self.log_stream_end(ts, usage, last_token_at, false, &stream_text);
        // A complete plain-JSON response is a normal completion, not a
        // premature one (there is no `[DONE]` to expect).
        self.finish_complete(tx, ts, usage, None, false, 0, (false, 0), tokens)
            .await
    }

    /// Resolve the terminal outcome when an SSE stream ends (clean `[DONE]`,
    /// EOF, premature close, or the receiver going away).
    ///
    /// A stream with at least one valid `data:` frame is a normal completion —
    /// `premature` reflects whether `[DONE]` was seen. A stream with *zero*
    /// valid frames is not usable SSE, so we first try to read the retained
    /// body as a non-streaming JSON completion (a server that ignored
    /// `stream: true` while we defaulted to SSE because the content type was
    /// absent/ambiguous). Only if that also fails do we surface a real
    /// [`StreamError::InvalidTransport`] — never a silent retrieval miss
    /// (Issue #1).
    #[allow(clippy::too_many_arguments)]
    async fn resolve_sse_end(
        &self,
        tx: &mut mpsc::Sender<StreamEvent>,
        ts: &mut StreamTimestamps,
        premature: bool,
        parser: &SseParser,
        pending_body: &[u8],
        body_dropped: bool,
        loop_state: (bool, u64),
        tokens: u64,
        last_token_at: &Option<MonotonicInstant>,
        stream_text: &str,
    ) -> StreamOutcome {
        if parser.has_valid_frame() {
            self.log_stream_end(
                ts,
                parser.usage(),
                *last_token_at,
                premature,
                stream_text,
            );
            return self
                .finish_complete(
                    tx,
                    ts,
                    parser.usage(),
                    parser.usage().map(|_| parser.details()),
                    premature,
                    parser.malformed_frames(),
                    loop_state,
                    tokens,
                )
                .await;
        }

        // Zero valid `data:` frames — not usable SSE. Try the JSON rescue:
        // the body may be a non-streaming completion that reached `run_sse`
        // because the content type was absent/ambiguous.
        if !body_dropped && !pending_body.is_empty() {
            if let Ok((content, usage)) = parse_plain_completion(pending_body) {
                let parsed_at = MonotonicInstant::now();
                return self
                    .emit_completion_frames(content, usage, parsed_at, ts, tx)
                    .await;
            }
        }

        // Genuinely neither valid SSE nor valid JSON: a real transport fault.
        let detail = if body_dropped {
            "body exceeded the inspection cap and no valid SSE frame parsed".to_string()
        } else {
            format!(
                "no valid data: frame and the body is not JSON ({} bytes)",
                pending_body.len()
            )
        };
        self.finish_failed(tx, *ts, StreamError::InvalidTransport(detail), false, 0)
            .await
    }

    /// The chat-completion request body. Gains a `response_format`
    /// object when [`response_format`](Self::response_format) /
    /// [`response_format_value`](Self::response_format_value) is set
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
            // Set verbatim: the simple `{"type": <string>}` hint (from
            // `response_format`) or the full `json_schema` object (from
            // `response_format_value`) both pass through unchanged.
            body["response_format"] = format.clone();
        }
        if self.ignore_eos {
            body["ignore_eos"] = serde_json::json!(true);
        }
        if self.disable_thinking {
            // CF-01 — per-backend thinking suppression, sent as a
            // **best-effort speed optimization** (NOT a correctness
            // dependency). Each backend honors the mechanism it knows and
            // ignores the rest (unknown top-level fields are dropped per the
            // OpenAI spec), so where no switch exists the request degrades
            // gracefully to the model's natural behavior:
            //   * `chat_template_kwargs.enable_thinking=false` → vLLM / SGLang /
            //     LM Studio (the Jinja chat template turns off CoT) — fast;
            //   * `think=false` → Ollama's extension field;
            //   * a trailing `/no_think` in the user message → llama.cpp
            //     (Qwen3's native template checks for it).
            // Where honored, a reasoning model skips CoT and the visible
            // answer fits a small budget (a pure speed/cost win). Where
            // ignored (config-only servers, Qwen3.5+), correctness rests on
            // the generation budget (`max_tokens`), which every server
            // honors; an over-thinker that commits nothing is reported
            // `truncated_by_reasoning` — not a false failure (Strawberry
            // Issue 1).
            body["chat_template_kwargs"] = serde_json::json!({ "enable_thinking": false });
            body["think"] = serde_json::json!(false);
            let base = self.prompt.trim_end_matches(['\n', '\r']);
            let prompt = if base.is_empty() {
                "/no_think".to_string()
            } else {
                format!("{base}\n/no_think")
            };
            body["messages"] = serde_json::json!([{ "role": "user", "content": prompt }]);
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
    /// tokens from throughput. `frames` is the observed token-frame count
    /// (the wire truth for the decode rate).
    #[allow(clippy::too_many_arguments)]
    async fn finish_complete(
        &self,
        tx: &mut mpsc::Sender<StreamEvent>,
        ts: &StreamTimestamps,
        usage: Option<Usage>,
        usage_details: Option<UsageDetails>,
        premature: bool,
        malformed_frames: u64,
        loop_state: (bool, u64),
        frames: u64,
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
            usage_details,
            premature,
            malformed_frames,
            error: None,
            looping,
            loop_excluded_tokens,
            frames,
        }
    }

    /// Emit the terminal `Failed` event and build the outcome.
    ///
    /// N11: the failure line carries the **worker's engine context**
    /// ([`context`](Self::context) — `Context::Sse` when the engine is
    /// unknown) with `Level::Error`, not `Context::Error`: the old
    /// `[ERROR] [ERROR]` double tag lost the engine attribution (the
    /// message's tag — `C2:…`, `F-1`, … — already disambiguates).
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
            self.context,
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
            usage_details: None,
            premature: false,
            malformed_frames: 0,
            error: Some(error),
            looping,
            loop_excluded_tokens,
            frames: 0,
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
            usage_details: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Read(format!("worker task failed: {join}"))),
            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
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
            usage_details: None,
            premature: false,
            malformed_frames: 0,
            error: Some(StreamError::Read(format!("worker task failed: {join}"))),
            looping: false,
            loop_excluded_tokens: 0,
            frames: 0,
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
    /// first token frame and `*last_token_at` is advanced to the arrival of
    /// every token frame (the decode window's end). Each decoded token
    /// string is also fed to the per-stream [`LoopGuard`] (v0.1.1) — the
    /// first detection logs a `[LOOP_GUARD]` warning line. Returns
    /// `(done, token_frames)`: `true` when a terminal `[DONE]` frame was
    /// among them, plus the number of token frames emitted in this call
    /// (the worker's log-milestone counter).
    ///
    /// **CF-09 — one `now()` per parsed frame (MTP-batch de-pollution).**
    /// Each frame is stamped with a fresh [`MonotonicInstant::now`] as it is
    /// emitted, NOT a single shared `at` for the whole `read_chunk` batch. A
    /// batch of N MTP / speculative-decoding tokens arriving in one read would
    /// otherwise all carry the same arrival instant, producing N−1 *zero*
    /// inter-token deltas that clamp to ~1 ns and drive the ITL p50 toward ~0
    /// exactly when speculative decoding is active. Per-frame `now()` gives
    /// each frame its own monotonic stamp, so the intra-batch deltas are the
    /// real (sub-µs) client processing cadence — not a clamped-to-zero
    /// artifact. The `at` parameter (the batch arrival instant) is retained in
    /// the signature as the reference the callers already compute; the
    /// per-frame stamp is what is recorded on each [`StreamEvent::Frame`].
    #[allow(clippy::too_many_arguments)]
    async fn emit_frames(
        &self,
        frames: &[ParsedFrame],
        at: &MonotonicInstant,
        ts: &mut StreamTimestamps,
        tx: &mut mpsc::Sender<StreamEvent>,
        guard: &mut LoopGuard,
        last_token_at: &mut Option<MonotonicInstant>,
        stream_text: &mut String,
    ) -> Result<(bool, u64), ()> {
        let mut done_seen = false;
        let mut token_frames = 0u64;
        let mut first = true;
        for frame in frames {
            // CF-09: one `now()` per parsed frame. The first frame of the
            // batch uses the read's arrival instant `at` (the true moment the
            // bytes landed); each subsequent frame stamps a fresh
            // `MonotonicInstant::now()`. A batch of N MTP / speculative
            // tokens arriving in a single read would otherwise all share one
            // arrival instant, producing N−1 *zero* inter-token deltas that
            // clamp to ~1 ns and drag the ITL p50 toward ~0 exactly when
            // speculative decoding is active. Per-frame stamps keep those
            // intra-batch deltas a real (sub-µs) positive cadence.
            let frame_at = if first {
                first = false;
                *at
            } else {
                MonotonicInstant::now()
            };
            let mut frame = frame.clone();
            frame.t_nanos = ts.t0.map_or(0, |t0| t0.delta_nanos(&frame_at));
            if frame.chunk.is_token() {
                token_frames += 1;
                if let Some(tok) = frame.chunk.token_text() {
                    // H2: accumulate the stream's text (reasoning +
                    // content, arrival order) so the completion line's
                    // no-`usage` count re-tokenizes the SAME source the
                    // engine's summary does (`authoritative_count`).
                    stream_text.push_str(tok);
                    if guard.push(tok) {
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
                    ts.t3 = Some(frame_at);
                }
                *last_token_at = Some(frame_at);
            }
            if frame.done {
                done_seen = true;
            }
            tx.send(StreamEvent::Frame {
                frame,
                at: frame_at,
                timestamps: *ts,
            })
            .await
            .map_err(|_| ())?;
        }
        Ok((done_seen, token_frames))
    }
}

/// Decide the response shape from the `Content-Type` header and our request
/// intent — NOT from sniffing the first data byte.
///
/// Sniffing the first byte broke on SSE keep-alive `:comment` preambles: a
/// stream that opens with `:keep-alive\n\n` before its first `data:` frame
/// starts with `:`, not `data:`, so it was misrouted to the non-streaming JSON
/// path and a valid retrieval looked like a "retrieval miss" (Issue #1). The
/// `Content-Type` header is the authoritative signal:
///
/// * `text/event-stream` → SSE (the spec's mandatory content type for SSE).
/// * `application/json` → the non-streaming completion object.
/// * absent / unrecognized → we always request `stream: true`, so default to
///   SSE. The SSE path then resolves a misrouted JSON body (the end-of-stream
///   rescue in [`run_sse`]) and surfaces a real
///   [`StreamError::InvalidTransport`] when the body is genuinely neither.
fn route_is_sse(content_type: &str) -> bool {
    let ct = content_type.to_ascii_lowercase();
    if ct.contains("text/event-stream") {
        return true;
    }
    if ct.contains("application/json") {
        return false;
    }
    // Absent or an unrecognized content type: we asked for `stream: true`, so
    // treat the response as SSE. This is what keeps a keep-alive-prefixed
    // stream (first byte `:`, not `data:`) from being misrouted to JSON.
    true
}

/// Strip a leading UTF-8 BOM, if present.
fn strip_bom(bytes: &[u8]) -> &[u8] {
    bytes.strip_prefix(&BOM).unwrap_or(bytes)
}

/// Append `chunk` to the retained SSE-path body until [`PENDING_BODY_CAP`];
/// once over the cap, drop the buffer and set `*dropped` so the caller knows
/// the body can no longer be inspected (a body that large cannot be a JSON
/// completion, so a zero-valid-frame stream is a genuine transport fault).
fn extend_pending_body(pending: &mut Vec<u8>, dropped: &mut bool, chunk: &[u8]) {
    if *dropped {
        return;
    }
    pending.extend_from_slice(chunk);
    if pending.len() > PENDING_BODY_CAP {
        *dropped = true;
        pending.clear();
    }
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

/// The **(first, last) content/reasoning token arrival** instants across a
/// stream's events — the research's **decode window**
/// (`T_first` → `T_last`): the moment the first generated token lands to the
/// moment the last one lands.
///
/// This is the window every throughput calculation must use: it excludes
/// prefill/TTFT (before the first token) and the trailing gap to `[DONE]`
/// (after the last token) — exactly how vLLM, llama.cpp, SGLang, LM Studio,
/// Ollama, and TGI each measure their own decode rate. `(None, None)` when no
/// token frame arrived (a pure control/usage stream, a failure, or a
/// non-streaming plain-JSON blob counted separately).
#[must_use]
pub fn token_window(
    events: &[StreamEvent],
) -> (Option<MonotonicInstant>, Option<MonotonicInstant>) {
    let mut first: Option<MonotonicInstant> = None;
    let mut last: Option<MonotonicInstant> = None;
    for e in events {
        if let StreamEvent::Frame { frame, at, .. } = e {
            if frame.chunk.is_token() {
                if first.is_none() {
                    first = Some(*at);
                }
                last = Some(*at);
            }
        }
    }
    (first, last)
}

/// The **concatenated reasoning + content text** of a stream, in arrival
/// order — the research's re-tokenization input. A non-compliant proxy that
/// omits `usage` is counted by re-encoding this text with the model's exact
/// tokenizer (counting SSE chunks directly is fundamentally inaccurate).
#[must_use]
pub fn stream_text(events: &[StreamEvent]) -> String {
    let mut text = String::new();
    for e in events {
        if let StreamEvent::Frame { frame, .. } = e {
            if let Some(t) = frame.chunk.token_text() {
                text.push_str(t);
            }
        }
    }
    text
}

/// The authoritative completion-token count for a stream's events, with an
/// `estimated` flag. This is the single source of truth every engine and the
/// live decode-rate tracker use, so the count is never the raw SSE frame
/// tally.
///
/// The **Authoritative Counting Rule**:
///
/// 1. **PRIMARY** — the server's own `usage.completion_tokens` (the terminal
///    usage chunk; every compliant backend populates it under
///    `stream_options.include_usage`). This is the count the server reports,
///    so matching it is the goal.
/// 2. **FALLBACK** (no `usage` — an aborted stream, or a non-compliant proxy)
///    — re-tokenize the full reasoning + content text: the exact tokenizer
///    when one is supplied, else the `chars/4` estimate (flagged
///    `estimated`). Counting raw SSE frames is *not* a valid count here:
///    servers that batch several tokens into one frame (vLLM MTP /
///    speculative decoding) make the frame count understate the true output
///    by 30–40 %.
///
/// **CF-10 note:** the *plausibility* of a server's `usage` relative to the
/// frames it actually sent is a *decode-rate* concern, handled one layer up by
/// [`StreamOutcome::has_usage_discrepancy`] /
/// [`StreamOutcome::effective_completion`] — not here. This counting
/// primitive keeps its "usage is primary" contract intact (the engines and
/// their fixtures rely on it); the CF-10 override lives where the count is
/// used as a decode-rate numerator, so it cannot silently re-scale a shared
/// counting function.
///
/// Returns `(tokens, estimated)`.
///
/// H2: the events are reduced to their [`stream_text`] and the count is
/// delegated to [`authoritative_count`] — the single seam the engine
/// summaries **and** the worker's per-stream `stream complete` line both
/// drive from, so the two can never use different methods (the v0.1.8
/// 3.5× Flat Out summary-vs-per-stream gap).
#[must_use]
pub fn authoritative_tokens(
    events: &[StreamEvent],
    usage: Option<Usage>,
    tokenizer: Option<&Tokenizer>,
) -> (u64, bool) {
    authoritative_count(usage, &stream_text(events), tokenizer)
}

/// The **Authoritative Counting Rule core** over already-collected
/// `text` (H2): the single source of truth for a stream's completion
/// token count, shared by every engine summary and the worker's
/// per-stream `stream complete` line, so the two always reconcile:
///
/// 1. **PRIMARY** — the server's own `usage.completion_tokens` (positive
///    when present — the count the server reports; matching it is the
///    goal).
/// 2. **FALLBACK** (no `usage` — an aborted stream, a non-compliant
///    proxy) — re-tokenize `text`: the exact `tokenizer` when supplied,
///    else the `chars/4` estimate (flagged `estimated`). Counting raw SSE
///    frames is *not* a valid count: batched servers (vLLM MTP /
///    speculative decoding) understate the true output by 30–40 %.
///
/// Returns `(tokens, estimated)`.
#[must_use]
pub fn authoritative_count(
    usage: Option<Usage>,
    text: &str,
    tokenizer: Option<&Tokenizer>,
) -> (u64, bool) {
    if let Some(u) = usage {
        if u.completion_tokens > 0 {
            return (u.completion_tokens, false);
        }
    }
    let c = count_tokens(text, tokenizer);
    (c.tokens as u64, c.estimated)
}

/// Count the [`Chunk::Reasoning`] and [`Chunk::Content`] frames in a
/// stream's events — the **wire-truth** frame split.
///
/// This is the primary signal for hybrid-reasoning
/// [`truncated_by_reasoning`](StreamOutcome::is_truncated_by_reasoning)
/// detection (Option C, `work/plans/REASONING_DESIGN.md`). The C1/C3
/// capability engines drain a run's events into a `Vec<StreamEvent>` and call
/// this to apply the rule
///
/// ```text
/// truncated_by_reasoning =
///     outcome.is_ok()          // a completed stream, not a transport fault
///     && reasoning_frames > 0   // the model was thinking
///     && content_frames == 0     // …and committed nothing visible
/// ```
///
/// It counts frames by the SSE classifier's [`Chunk`] kind, matching the
/// [`Chunk::is_token`] convention: a `Control` / `Usage` frame is neither a
/// reasoning nor a content token, so it is counted in neither. It works on
/// **every** server with no `usage` dependency — unlike
/// `UsageDetails::is_reasoning_dominated`, which needs the server to report
/// `reasoning_tokens` (config-only servers may omit it).
///
/// Returns `(reasoning_frames, content_frames)`.
#[must_use]
pub fn count_reasoning_content_frames(events: &[StreamEvent]) -> (u64, u64) {
    let mut reasoning = 0u64;
    let mut content = 0u64;
    for e in events {
        if let StreamEvent::Frame { frame, .. } = e {
            match &frame.chunk {
                Chunk::Reasoning(_) => reasoning += 1,
                Chunk::Content(_) => content += 1,
                _ => {}
            }
        }
    }
    (reasoning, content)
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
    fn route_is_sse_by_content_type_and_request_intent() {
        // The authoritative signal: `text/event-stream` → SSE (case-insensitive,
        // parameters tolerated).
        assert!(route_is_sse("text/event-stream"));
        assert!(route_is_sse("Text/Event-Stream; charset=utf-8"));
        assert!(route_is_sse("text/event-stream;charset=UTF-8"));
        // An explicit JSON content type → the non-streaming path.
        assert!(!route_is_sse("application/json"));
        assert!(!route_is_sse("Application/JSON; charset=utf-8"));
        // Absent / unrecognized content type: we requested `stream: true`, so
        // default to SSE. This is the Issue #1 guarantee — a keep-alive-prefixed
        // stream (first byte `:`, not `data:`) must NOT fall through to JSON.
        assert!(route_is_sse(""));
        assert!(route_is_sse("text/plain"));
        assert!(route_is_sse("application/octet-stream"));
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
    fn request_body_with_json_schema_response_format_is_encoded_verbatim() {
        // CF-08 handoff: C3's `json_schema` constrained-decoding object is
        // transmitted end-to-end. The full object (not just `{"type": s}`)
        // is set on the body verbatim, matching the exact shape WS3's
        // `json_schema_response_format()` builds.
        let wf = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .response_format_value(serde_json::json!({
            "type": "json_schema",
            "json_schema": {
                "name": "crucible_case",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": { "code": { "type": "string" } },
                    "required": ["code"]
                }
            }
        }));
        let body = wf.request_body();
        // The `json_schema` sub-object is present and intact.
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(
            body["response_format"]["json_schema"]["name"],
            "crucible_case"
        );
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["properties"]["code"]["type"],
            "string"
        );
        assert_eq!(
            body["response_format"]["json_schema"]["schema"]["required"],
            serde_json::json!(["code"])
        );
        // The rest of the body is untouched.
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 10);
    }

    #[test]
    fn request_body_response_format_is_a_graceful_no_op_when_unset() {
        // Graceful: a worker that never sets `response_format` (FreeForm
        // mode, or a backend that does not support it) sends the plain
        // Engine A body — no `response_format` field, nothing broken.
        let wf = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        );
        let body = wf.request_body();
        assert!(body.get("response_format").is_none());
        // The body is still complete and valid (all required fields present).
        assert_eq!(body["model"], "m");
        assert_eq!(body["stream"], true);
        assert_eq!(body["temperature"], 0);
        assert_eq!(body["max_tokens"], 10);
        assert_eq!(body["stream_options"]["include_usage"], true);
        assert!(body["messages"].is_array());
    }

    #[test]
    fn request_body_ignore_eos_is_opt_in() {
        // Default: the field is absent (Engine A / B / C bodies are
        // byte-identical to before).
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        );
        assert!(w.request_body().get("ignore_eos").is_none());

        // Opt-in (Engine F): the field is present, the rest untouched.
        let w = w.ignore_eos(true);
        let body = w.request_body();
        assert_eq!(body["ignore_eos"], true);
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 10);
    }

    #[test]
    fn request_body_disable_thinking_is_opt_in_and_per_backend() {
        // Default: no suppression fields (Engine A / B / C2 / C3 bodies are
        // byte-identical to before) — a no-op for every backend.
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "Answer with the value only.",
            64,
        );
        let body = w.request_body();
        assert!(body.get("chat_template_kwargs").is_none());
        assert!(body.get("think").is_none());
        assert_eq!(
            body["messages"][0]["content"],
            "Answer with the value only."
        );

        // Opt-in (Engine C1): all three per-backend mechanisms are present,
        // and the `/no_think` prompt suffix is appended (llama.cpp).
        let w = w.disable_thinking(true);
        let body = w.request_body();
        assert_eq!(
            body["chat_template_kwargs"]["enable_thinking"], false,
            "vLLM / SGLang / LM Studio"
        );
        assert_eq!(body["think"], false, "Ollama");
        assert_eq!(
            body["messages"][0]["content"], "Answer with the value only.\n/no_think",
            "llama.cpp /no_think suffix"
        );
        // The rest of the body is untouched.
        assert_eq!(body["stream"], true);
        assert_eq!(body["max_tokens"], 64);
        assert_eq!(body["model"], "m");
    }

    #[test]
    fn stream_end_log_is_self_consistent_and_counts_authoritatively() {
        // H2 + H5 + M1 + N23: the completion line counts with the
        // authoritative method (usage first, else the re-tokenized stream
        // text — never the raw frame tally) and discloses the wall span,
        // the decode window, and the TTFB separately — so the printed
        // decode rate is derivable straight off the line:
        // `count / (total − TTFB) ≈ decode`.
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

            // Usage arrived (256): the authoritative count; wall span
            // 0.2 s, decode window T3 (100 ms) → last (200 ms) = 0.1 s →
            // 256 / 0.1 = 2560 t/s; TTFB = 0.2 − 0.1 = 0.1 s.
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t3),
                    t_end: Some(t_end),
                },
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 256,
                }),
                Some(t_end),
                false,
                "unused-when-usage-present",
            );
            // No usage: the count re-tokenizes the stream's text (4 chars →
            // chars/4 = 1, flagged ≈ — N23's single approximation glyph);
            // wall span 0.2 s, decode window T3 (= T0) → last = 0.2 s →
            // 1 / 0.2 = 5 t/s; TTFB 0.0 s.
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t0),
                    t_end: Some(t_end),
                },
                None,
                Some(t_end),
                false,
                "abcd",
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
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 256,
                }),
                Some(t_end),
                true,
                "unused-when-usage-present",
            );
        });
        logger.finish();
        // The writer flushes on every newline; read after finish.
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        // H5: the decode rate is derivable: 256 / (0.2 − 0.1) = 2560.
        assert!(
            latest
                .contains("test stream complete: 256 tokens in 0.2s (decode 2560.0 t/s; TTFB 0.1s)"),
            "self-consistent usage line missing: {latest}"
        );
        // H2 + N23: no usage → the text is re-counted (≈); 1 / (0.2 − 0.0)
        // = 5, and the old frame-tally count is gone.
        assert!(
            latest.contains(
                "test stream complete: ≈1 tokens (est, no usage) in 0.2s (decode ≈5.0 t/s; TTFB 0.0s)"
            ),
            "authoritative text-count line missing: {latest}"
        );
        // N23: no `~` approximation marker remains on the lines.
        assert!(!latest.contains("stream complete: ~"), "no `~` estimate marker: {latest}");
        // The premature marker is intact.
        assert!(
            latest.contains("256 tokens in 0.2s (decode 2560.0 t/s; TTFB 0.1s) (premature close)"),
            "premature marker missing: {latest}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stream_end_sub_floor_decode_window_is_na() {
        // M1: a stream whose decode window is below the 10 ms floor
        // (a single token frame — `first == last` — or a sub-ms MTP
        // batch) has no measurable decode rate: the line reads
        // `decode n/a`, never a misleading `0.0 t/s` and never a
        // clamped-sliver phantom.
        let dir = std::env::temp_dir().join(format!("crucible-streamend-na-{}", std::process::id()));
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
            // Zero span: the only token frame IS the first (200 ms).
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(200_000_000);
            let t3 = MonotonicInstant::now(); // 200 ms
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t3),
                    t_end: Some(t3),
                },
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 1,
                }),
                Some(t3),
                false,
                "one",
            );
            // Sub-ms batch: two token frames 4 ms apart (< 10 ms floor).
            mock.increment(5_000_000);
            let t_first = MonotonicInstant::now(); // 205 ms
            mock.increment(4_000_000);
            let t_last = MonotonicInstant::now(); // 209 ms
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t_first),
                    t_end: Some(t_last),
                },
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 2,
                }),
                Some(t_last),
                false,
                "one two",
            );
            // A real window (100 ms) still measures: the floor must not
            // mask a genuine decode.
            mock.increment(100_000_000);
            let t_real = MonotonicInstant::now(); // 309 ms
            w.log_stream_end(
                &StreamTimestamps {
                    t0: Some(t0),
                    t1: None,
                    t2: None,
                    t3: Some(t_last),
                    t_end: Some(t_real),
                },
                Some(Usage {
                    prompt_tokens: 10,
                    completion_tokens: 10,
                }),
                Some(t_real),
                false,
                "text",
            );
        });
        logger.finish();
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        // Zero span: n/a (wall span 0.2 s, TTFB 0.2 s, no decode window).
        assert!(
            latest.contains("1 tokens in 0.2s (decode n/a; TTFB 0.2s)"),
            "zero-span line must read `decode n/a`: {latest}"
        );
        // Sub-ms batch: n/a (wall span 0.209 s, TTFB 0.205 s).
        assert!(
            latest.contains("2 tokens in 0.2s (decode n/a; TTFB 0.2s)"),
            "sub-floor line must read `decode n/a`: {latest}"
        );
        // Real window: 10 / 0.1 = 100 t/s, derivable (0.309 − 0.209 = 0.1).
        assert!(
            latest.contains("10 tokens in 0.3s (decode 100.0 t/s; TTFB 0.2s)"),
            "a real decode window must still measure: {latest}"
        );
        // No misleading `decode 0.0 t/s` (a zero rate reads as "the
        // server produced nothing" — false).
        assert!(
            !latest.contains("decode 0.0 t/s"),
            "no bare decode 0.0 t/s: {latest}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn authoritative_count_matches_authoritative_tokens() {
        // H2: the per-stream line's count core and the engine summary's
        // event-based count are the SAME method — equal on identical
        // input, with and without a `usage` block.
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
        let events = vec![
            frame(Chunk::Content("abcd".into())),
            frame(Chunk::Content("efgh".into())),
        ];
        // No usage: both methods re-tokenize the same text (chars/4 → 2).
        assert_eq!(
            authoritative_count(None, &stream_text(&events), None),
            authoritative_tokens(&events, None, None),
            "no-usage counts must agree"
        );
        // Usage present: both methods take the server's figure.
        let usage = Usage {
            prompt_tokens: 5,
            completion_tokens: 42,
        };
        assert_eq!(
            authoritative_count(Some(usage), &stream_text(&events), None),
            authoritative_tokens(&events, Some(usage), None),
            "usage counts must agree"
        );
    }

    // ── P2/N14: a deliberate early close carries its own label ─────────

    #[test]
    fn stream_end_premature_uses_the_close_note() {
        // P2/N14: a *deliberate* early close (Engine F's 60 s window —
        // EOF without `[DONE]` is the designed stop) is labeled as such,
        // not the fault-reading `(premature close)`; a genuine
        // mid-stream drop (no note) keeps the old marker.
        let dir =
            std::env::temp_dir().join(format!("crucible-close-note-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        let noted = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .logger(logger.clone())
        .tag("F-1")
        .close_note(Some("60s window close".to_string()));
        let bare = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .logger(logger.clone())
        .tag("B1-1");

        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 100 ms
            mock.increment(100_000_000);
            let last = MonotonicInstant::now(); // 200 ms
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: None,
                t2: None,
                t3: Some(t3),
                t_end: Some(last),
            };
            let usage = Some(Usage {
                prompt_tokens: 10,
                completion_tokens: 256,
            });
            noted.log_stream_end(&ts, usage, Some(last), true, "x");
            bare.log_stream_end(&ts, usage, Some(last), true, "x");
        });
        logger.finish();
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        assert!(
            latest.contains("F-1 stream complete: 256 tokens in 0.2s (decode 2560.0 t/s; TTFB 0.1s) (60s window close)"),
            "the deliberate close carries its own label: {latest}"
        );
        assert!(
            latest.contains("B1-1 stream complete: 256 tokens in 0.2s (decode 2560.0 t/s; TTFB 0.1s) (premature close)"),
            "a genuine drop keeps the fault marker: {latest}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── N11: a worker failure carries the engine's context ─────────────

    #[tokio::test]
    async fn finish_failed_uses_the_engine_context() {
        // N11: a worker failure logs `Level::Error` with the **engine's**
        // context — `[ERROR] [ENGINE_C2] C2:… failed: …` — restoring
        // engine attribution; the old `[ERROR] [ERROR]` double tag is
        // gone. A worker with no engine set keeps `Context::Sse`.
        let dir = std::env::temp_dir().join(format!("crucible-fail-ctx-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let logger = RunLogger::start(Some(&dir));
        let c2 = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .logger(logger.clone())
        .tag("C2:math-mul")
        .context(Context::EngineC2);
        let sse = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        )
        .logger(logger.clone())
        .tag("pool-0");

        let (mut tx, _rx) = mpsc::channel(16);
        c2.finish_failed(
            &mut tx,
            StreamTimestamps::default(),
            StreamError::Connection("refused".into()),
            false,
            0,
        )
        .await;
        drop(tx);
        let (mut tx, _rx) = mpsc::channel(16);
        sse.finish_failed(
            &mut tx,
            StreamTimestamps::default(),
            StreamError::Read("reset".into()),
            false,
            0,
        )
        .await;
        drop(tx);
        logger.finish();
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        assert!(
            latest.contains("[ERROR] [ENGINE_C2] C2:math-mul failed:"),
            "the failure line carries the engine's context: {latest}"
        );
        assert!(
            latest.contains("[ERROR] [SSE] pool-0 failed:"),
            "no engine set → the worker's own subsystem: {latest}"
        );
        assert!(
            !latest.contains("[ERROR] [ERROR]"),
            "the double tag is gone: {latest}"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // C3 (noise cleanup): the `token #1` milestone latches (a keep-alive /
    // whitespace-only batch keeps the cumulative count at 1 — it must not
    // re-fire), and `token #0` never logs (no more meaningless
    // `rate: 0.0 t/s` lines).
    #[test]
    fn token_milestones_latch_first_and_skip_zero() {
        let dir = std::env::temp_dir().join(format!("crucible-miles-{}", std::process::id()));
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
            mock.increment(10_000_000);
            let t3 = MonotonicInstant::now(); // 10 ms (first token)
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: None,
                t2: None,
                t3: Some(t3),
                t_end: None,
            };
            let mut first_logged = false;
            let mut last_logged_50 = 0u64;
            // Keep-alive batch (tc == 0): the count stays 0 — no line.
            w.log_token_milestones(&ts, 0, &mut first_logged, &mut last_logged_50);
            // The first token: the milestone fires.
            w.log_token_milestones(&ts, 1, &mut first_logged, &mut last_logged_50);
            // Whitespace-only batches (the count stays 1): the latch holds.
            w.log_token_milestones(&ts, 1, &mut first_logged, &mut last_logged_50);
            w.log_token_milestones(&ts, 1, &mut first_logged, &mut last_logged_50);
            // The every-50 milestone (t3 → now = 50 ms → 1000 t/s). The
            // count is cumulative (each multiple fires once per stream),
            // so the realistic sequence advances it.
            mock.increment(50_000_000);
            w.log_token_milestones(&ts, 50, &mut first_logged, &mut last_logged_50);
            // The every-100 milestone (t3 → now = 100 ms → 1000 t/s).
            mock.increment(50_000_000);
            w.log_token_milestones(&ts, 100, &mut first_logged, &mut last_logged_50);
        });
        logger.finish();
        // The writer flushes on every newline; read after finish.
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        // Exactly one `token #1`, zero `token #0`, one each of #50 / #100.
        assert_eq!(latest.matches("token #1 received").count(), 1, "{latest}");
        assert_eq!(latest.matches("token #0").count(), 0, "{latest}");
        assert_eq!(latest.matches("token #50 received").count(), 1, "{latest}");
        assert_eq!(latest.matches("token #100 received").count(), 1, "{latest}");
        // No meaningless `rate: 0.0` line (the keep-alive batch is gone).
        assert!(!latest.contains("rate: 0.0"), "{latest}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // C3 residual: the every-50 milestone latches per distinct 50-multiple.
    // A whitespace-only delta batch keeps the cumulative count at the same
    // 50-multiple — it must NOT re-fire (the residual `#100` re-fire the
    // v0.1.7 live verify saw), while genuine advancement (50 → 100 → 150)
    // logs each new multiple exactly once.
    #[test]
    fn token_milestones_latch_each_50_multiple_once() {
        let dir = std::env::temp_dir().join(format!("crucible-miles-50-{}", std::process::id()));
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
            mock.increment(10_000_000);
            let t3 = MonotonicInstant::now(); // 10 ms (first token)
            let ts = StreamTimestamps {
                t0: Some(t0),
                t1: None,
                t2: None,
                t3: Some(t3),
                t_end: None,
            };
            let mut first_logged = false;
            let mut last_logged_50 = 0u64;
            // The #50 milestone fires.
            mock.increment(40_000_000);
            w.log_token_milestones(&ts, 50, &mut first_logged, &mut last_logged_50);
            // Whitespace-only batches keep the count at 50 — the latch holds
            // (no re-fire).
            w.log_token_milestones(&ts, 50, &mut first_logged, &mut last_logged_50);
            w.log_token_milestones(&ts, 50, &mut first_logged, &mut last_logged_50);
            // Genuine advancement to #100 fires the next multiple.
            mock.increment(50_000_000);
            w.log_token_milestones(&ts, 100, &mut first_logged, &mut last_logged_50);
            // Whitespace-only batches keep the count at 100 — no re-fire.
            w.log_token_milestones(&ts, 100, &mut first_logged, &mut last_logged_50);
            w.log_token_milestones(&ts, 100, &mut first_logged, &mut last_logged_50);
            // Genuine advancement to #150 fires the next multiple.
            mock.increment(50_000_000);
            w.log_token_milestones(&ts, 150, &mut first_logged, &mut last_logged_50);
            // Whitespace-only batch keeps the count at 150 — no re-fire.
            w.log_token_milestones(&ts, 150, &mut first_logged, &mut last_logged_50);
        });
        logger.finish();
        // The writer flushes on every newline; read after finish.
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        // Each 50-multiple fires exactly once (no re-fire on the
        // whitespace-only batches); #50 / #100 / #150 all present.
        assert_eq!(latest.matches("token #50 received").count(), 1, "{latest}");
        assert_eq!(latest.matches("token #100 received").count(), 1, "{latest}");
        assert_eq!(latest.matches("token #150 received").count(), 1, "{latest}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- P2-A: CF-09 (ITL de-pollution) + CF-10 (usage plausibility) +
    //      CF-44 (E2E-only) tests ----

    fn test_outcome(
        frames: u64,
        usage: Option<Usage>,
        error: Option<StreamError>,
    ) -> StreamOutcome {
        StreamOutcome {
            timestamps: StreamTimestamps::default(),
            usage,
            usage_details: None,
            premature: false,
            malformed_frames: 0,
            error,
            looping: false,
            loop_excluded_tokens: 0,
            frames,
        }
    }

    /// CF-09: a batch of token frames that arrive in a single read must each
    /// get its own monotonic stamp, not one shared arrival instant. A shared
    /// instant yields zero inter-token deltas (clamped to ~1 ns) that drag the
    /// ITL p50 toward ~0 exactly when speculative decoding is active; a
    /// per-frame `now()` keeps the intra-batch span a real positive value.
    #[tokio::test]
    async fn emit_frames_stamps_one_now_per_batched_frame() {
        let w = StreamWorker::new(
            reqwest::Client::new(),
            "http://localhost:8000",
            "m",
            "p",
            10,
        );
        let (mut tx, mut rx) = mpsc::channel(16);
        let mut ts = StreamTimestamps {
            t0: Some(MonotonicInstant::now()),
            ..StreamTimestamps::default()
        };
        // Three token frames arriving in ONE read (the same batch `at`).
        let at = MonotonicInstant::now();
        let frames = vec![
            ParsedFrame {
                chunk: Chunk::Content("a".into()),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Content("b".into()),
                t_nanos: 0,
                done: false,
            },
            ParsedFrame {
                chunk: Chunk::Content("c".into()),
                t_nanos: 0,
                done: false,
            },
        ];
        let mut guard = LoopGuard::new();
        let mut last = None;
        let mut stream_text = String::new();
        w.emit_frames(
            &frames,
            &at,
            &mut ts,
            &mut tx,
            &mut guard,
            &mut last,
            &mut stream_text,
        )
        .await
        .unwrap();
        // H2: the batched token text is accumulated for the completion
        // line's no-`usage` count (the same source the summary uses).
        assert_eq!(stream_text, "abc", "reasoning + content text, in order");
        drop(tx);
        let mut ats = Vec::new();
        while let Some(StreamEvent::Frame { at, .. }) = rx.recv().await {
            ats.push(at);
        }
        assert_eq!(ats.len(), 3, "all three batched frames are emitted");
        // The span from the first to the last frame is positive: each frame
        // got its own stamp, so the intra-batch deltas are not the zero
        // (→1 ns) artifacts that pollute the ITL p50.
        let span = ats.first().unwrap().delta_nanos(ats.last().unwrap());
        assert!(
            span > 0,
            "batched frames must have a positive span, got {span}"
        );
    }

    /// CF-10: a server-reported `completion_tokens` that exceeds ~4× the
    /// observed token frames marks a max_tokens-inflating backend. The flag
    /// trips and the effective count falls back to the wire truth.
    #[test]
    fn usage_exceeding_4x_frames_is_flagged_and_ignored() {
        let o = test_outcome(
            10,
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 500,
            }),
            None,
        );
        assert!(o.has_usage_discrepancy(), "500 > 4×10 must be flagged");
        // The effective count is the observed frames, not the inflated usage.
        assert_eq!(o.effective_completion(), 10);
    }

    /// CF-10: a plausible usage (within ~4× the frames) is trusted.
    #[test]
    fn plausible_usage_is_trusted() {
        let o = test_outcome(
            10,
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 12,
            }),
            None,
        );
        assert!(!o.has_usage_discrepancy(), "12 ≤ 4×10 must not be flagged");
        assert_eq!(o.effective_completion(), 12);
    }

    /// CF-10: exactly 4× is *not* a discrepancy ("exceeds" is strict).
    #[test]
    fn exactly_4x_frames_is_not_a_discrepancy() {
        let o = test_outcome(
            10,
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 40,
            }),
            None,
        );
        assert!(!o.has_usage_discrepancy(), "40 == 4×10 is not *exceeding*");
        assert_eq!(o.effective_completion(), 40);
    }

    /// CF-10: no observed frames (nothing to compare) → no discrepancy, the
    /// usage is trusted.
    #[test]
    fn no_frames_means_no_discrepancy() {
        let o = test_outcome(
            0,
            Some(Usage {
                prompt_tokens: 5,
                completion_tokens: 500,
            }),
            None,
        );
        assert!(!o.has_usage_discrepancy(), "no frames → nothing to compare");
        assert_eq!(o.effective_completion(), 500);
    }

    /// CF-44: a plain-JSON (non-streaming) result — `frames == 0` with a usage
    /// and no error — is labeled E2E-only.
    #[test]
    fn plain_json_result_is_e2e_only() {
        let o = test_outcome(
            0,
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 7,
            }),
            None,
        );
        assert!(o.is_e2e_only(), "frames==0 + usage + no error = E2E-only");
    }

    /// CF-44: a normal streaming result (observed token frames) is NOT
    /// E2E-only.
    #[test]
    fn streaming_result_is_not_e2e_only() {
        let o = test_outcome(
            10,
            Some(Usage {
                prompt_tokens: 42,
                completion_tokens: 10,
            }),
            None,
        );
        assert!(!o.is_e2e_only(), "frames>0 = a real streaming result");
    }

    /// CF-44: a failed result (no usage) is not E2E-only either.
    #[test]
    fn failed_result_is_not_e2e_only() {
        let o = test_outcome(0, None, Some(StreamError::Connection("refused".into())));
        assert!(!o.is_e2e_only(), "an error is not a plain-JSON completion");
    }

    // ---- R1: hybrid-reasoning `truncated_by_reasoning` detection ----

    /// Build a `StreamEvent::Frame` carrying one `chunk` — the fixture for
    /// the frame-count / truncation-detection tests below.
    fn frame_event(chunk: Chunk) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk,
                t_nanos: 0,
                done: false,
            },
            at: MonotonicInstant::now(),
            timestamps: StreamTimestamps::default(),
        }
    }

    /// `count_reasoning_content_frames` splits the wire frames by the SSE
    /// classifier's `Chunk` kind (matching `is_token`: `Control` / `Usage`
    /// are counted in neither).
    #[test]
    fn count_frames_splits_reasoning_and_content() {
        let events = vec![
            frame_event(Chunk::Reasoning("a".into())),
            frame_event(Chunk::Content("b".into())),
            frame_event(Chunk::Reasoning("c".into())),
            frame_event(Chunk::Control),
            frame_event(Chunk::Usage(Usage::default())),
        ];
        let (reasoning, content) = count_reasoning_content_frames(&events);
        assert_eq!(reasoning, 2, "two reasoning frames");
        assert_eq!(content, 1, "one content frame");
    }

    /// `count_reasoning_content_frames` is `(0, 0)` for a list with no token
    /// frames (or an empty list).
    #[test]
    fn count_frames_is_zero_for_no_token_events() {
        let events = vec![
            frame_event(Chunk::Control),
            frame_event(Chunk::Usage(Usage::default())),
        ];
        assert_eq!(count_reasoning_content_frames(&events), (0, 0));
        assert_eq!(count_reasoning_content_frames(&[]), (0, 0));
    }

    /// (a) A **completed** stream (no error) that emitted reasoning frames
    /// but **zero** content frames is `truncated_by_reasoning` — the budget
    /// was exhausted in CoT and no answer was produced.
    #[test]
    fn reasoning_only_completed_stream_is_truncated() {
        let events = vec![
            frame_event(Chunk::Reasoning("thinking…".into())),
            frame_event(Chunk::Reasoning("more thought…".into())),
        ];
        let outcome = test_outcome(2, None, None); // is_ok: no error
        assert!(outcome.is_ok());
        assert!(
            outcome.is_truncated_by_reasoning(&events),
            "reasoning>0 && content==0 && is_ok ⇒ truncated"
        );
    }

    /// (b) A completed stream that emitted **any** content frame is NOT
    /// `truncated_by_reasoning` — the model committed a visible answer.
    #[test]
    fn content_present_stream_is_not_truncated() {
        let events = vec![
            frame_event(Chunk::Reasoning("thinking…".into())),
            frame_event(Chunk::Content("the answer".into())),
        ];
        let outcome = test_outcome(2, None, None);
        assert!(
            !outcome.is_truncated_by_reasoning(&events),
            "content>0 ⇒ not truncated"
        );
    }

    /// A completed stream with **neither** reasoning nor content (a
    /// control/usage-only stream) is NOT `truncated_by_reasoning` — the model
    /// was not "thinking".
    #[test]
    fn no_token_frames_is_not_truncated() {
        let events = vec![
            frame_event(Chunk::Control),
            frame_event(Chunk::Usage(Usage::default())),
        ];
        let outcome = test_outcome(0, None, None);
        assert!(!outcome.is_truncated_by_reasoning(&events));
    }

    /// (c) A **transport-error** stream is NOT `truncated_by_reasoning` — it
    /// is a genuine failure, not a budget exhaustion. The `is_ok()` gate
    /// makes a failed all-CoT stream read as a fail, not a truncation.
    #[test]
    fn transport_error_stream_is_not_truncated() {
        let events = vec![frame_event(Chunk::Reasoning("thinking…".into()))];
        let outcome = test_outcome(1, None, Some(StreamError::Connection("refused".into())));
        assert!(!outcome.is_ok());
        assert!(
            !outcome.is_truncated_by_reasoning(&events),
            "an error is a fail, not a truncation"
        );
    }
}
