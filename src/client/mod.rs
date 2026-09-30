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

pub mod models;
pub mod pool;
pub mod stream;

pub use models::{list_models, normalize_base_url, ModelError, ModelInfo};
pub use stream::{
    normalize_endpoint, StreamError, StreamEvent, StreamOutcome, StreamWorker, DEFAULT_READ_TIMEOUT,
};
