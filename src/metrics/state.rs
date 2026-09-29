//! `ArcSwap<MetricsSnapshot>` double buffer: the engine pushes a unified
//! snapshot per batch and the UI reads it lock-free.
//!
//! Filled in by Chunk 6 (engine core + metric synthesis).
