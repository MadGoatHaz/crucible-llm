//! Worker `pool`: spawn/manage N `StreamWorker`s with independent HTTP/2
//! streams, fanning their channels into a single bounded `tokio::mpsc`.
//!
//! Filled in by Chunk 10 (multi-stream pool + concurrency ladder sweep).
