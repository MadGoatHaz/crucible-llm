//! Metrics synthesis: the `LatencyHistogram` wrapper and the `ArcSwap`
//! double-buffered snapshot the UI reads lock-free.

pub mod histogram;
pub mod methodology;
pub mod state;

pub use histogram::LatencyHistogram;
pub use state::{LoopGuardSummary, MetricsSnapshot, MetricsState, StreamMetric, StreamStatus};
