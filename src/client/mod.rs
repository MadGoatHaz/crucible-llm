//! HTTP/2 SSE streaming client: a single-stream `StreamWorker`, the
//! multi-stream worker `pool`, and the `models` discovery client.
//!
//! * [`stream`] — `StreamWorker` (Chunk 5): one POST to
//!   `/v1/chat/completions` (`stream: true`), `T0..Tn` lifecycle timestamps
//!   on the `quanta` clock, response bytes fed into the Chunk 3 `SseParser`,
//!   and timestamped [`StreamEvent`]s over a bounded `tokio::sync::mpsc`
//!   channel for the Engine Core.
//! * [`pool`] — (Chunk 10) spawn/manage N workers with independent HTTP/2
//!   streams, fanning their channels into one bounded channel.
//! * [`models`] — (Chunk 20) `GET {base}/models` against an
//!   OpenAI-compatible endpoint: [`list_models`] returns the served
//!   [`ModelInfo`] list (deduped, id-sorted) for the TUI's model picker.

pub mod loop_guard;
pub mod models;
pub mod pool;
pub mod stream;

pub use loop_guard::{LoopGuard, LOOP_BUFFER, LOOP_REPETITIONS, LOOP_WINDOW};
pub use models::{list_models, normalize_base_url, ModelError, ModelInfo};
pub use stream::{
    join_worker, normalize_endpoint, run_worker, spawn_worker, stream_text, token_window,
    StreamError, StreamEvent, StreamOutcome, StreamWorker, DEFAULT_READ_TIMEOUT,
};

/// Truncate `s` to at most `max` bytes on a character boundary, appending
/// a `…` when a cut was made. Shared by the streaming client
/// ([`stream::StreamError::Http`] bodies) and the discovery client
/// ([`models::ModelError::Http`] bodies) — response bodies are untrusted
/// and must never blow up a log line.
#[must_use]
pub(crate) fn truncate_body(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}
