//! `StreamWorker`: one HTTP/2 SSE stream against `/v1/chat/completions`,
//! recording `T0..Tn` timestamps and emitting parsed frames over a bounded
//! `tokio::sync::mpsc` channel.
//!
//! Filled in by Chunk 5 (stream worker).
