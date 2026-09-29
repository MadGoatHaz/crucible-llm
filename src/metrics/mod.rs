//! Metrics synthesis: the `EngineCore` aggregator, the `LatencyHistogram`
//! wrapper, and the `ArcSwap` double-buffered snapshot for the UI.

pub mod engine;
pub mod histogram;
pub mod state;

pub use histogram::LatencyHistogram;
