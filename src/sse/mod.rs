//! Server-Sent Events (SSE) parsing: incremental low-allocation frame parser
//! and the `Chunk` model (Reasoning / Content / Control / Usage).
//!
//! * [`chunk`] — the `Chunk` enum plus `classify()`, which categorizes one
//!   parsed JSON frame (reasoning deltas first, then content, then usage).
//! * [`parser`] — `SseParser`: feed it raw bytes from a streaming endpoint,
//!   get back classified [`ParsedFrame`]s with `[DONE]` tracking and
//!   malformed-JSON tolerance.

pub mod chunk;
pub mod parser;

pub use chunk::{classify, Chunk, Usage};
pub use parser::{ParsedFrame, SseParser};
