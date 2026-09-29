//! HTTP/2 SSE streaming client: a single-stream `StreamWorker` and the
//! multi-stream worker `pool`.
//!
//! * [`stream`] — `StreamWorker` (Chunk 5): one POST to
//!   `/v1/chat/completions` (`stream: true`), `T0..Tn` lifecycle timestamps
//!   on the `quanta` clock, response bytes fed into the Chunk 3 `SseParser`,
//!   and timestamped [`StreamEvent`]s over a bounded `tokio::sync::mpsc`
//!   channel for the Engine Core.
//! * [`pool`] — (Chunk 10) spawn/manage N workers with independent HTTP/2
//!   streams, fanning their channels into one bounded channel.

pub mod pool;
pub mod stream;

pub use stream::{
    normalize_endpoint, StreamError, StreamEvent, StreamOutcome, StreamWorker, DEFAULT_READ_TIMEOUT,
};
