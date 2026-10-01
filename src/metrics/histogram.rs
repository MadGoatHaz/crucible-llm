//! `LatencyHistogram`: a thin wrapper over `hdrhistogram` exposing
//! `record(nanos)` and `percentile(p) -> f64` for p50/p90/p99/p99.9.

use hdrhistogram::Histogram;

/// Default upper bound for a single latency sample: 1 second in nanoseconds.
pub const DEFAULT_MAX_NANOS: u64 = 1_000_000_000;

/// A high-dynamic-range histogram of latency samples (in nanoseconds),
/// wrapping `hdrhistogram` for p50/p90/p99/p99.9 queries.
///
/// The recording path is non-panicking: samples are clamped into
/// `[1, max_ns]` before recording, so a stray out-of-range value can never
/// trip the underlying `record` error.
pub struct LatencyHistogram {
    inner: Histogram<u64>,
    max_ns: u64,
}

impl LatencyHistogram {
    /// Create a histogram tracking samples in `[1, max_ns]` nanoseconds with
    /// 3 significant digits of precision (max ~0.1% quantization error).
    pub fn new(max_ns: u64) -> Self {
        let max_ns = max_ns.max(1);
        let inner = Histogram::new_with_bounds(1, max_ns, 3)
            .expect("histogram bounds [1, max_ns] are always valid");
        Self { inner, max_ns }
    }

    /// Record a single latency sample in nanoseconds (clamped into range).
    #[inline]
    pub fn record(&mut self, ns: u64) {
        let ns = ns.clamp(1, self.max_ns);
        let _ = self.inner.record(ns);
    }

    /// Value at the given percentile `p` in `0.0..=100.0`, in nanoseconds.
    ///
    /// `p` is clamped to `[0, 100]`; an empty histogram returns `0.0`.
    #[must_use]
    pub fn percentile(&self, p: f64) -> f64 {
        self.inner.value_at_percentile(p.clamp(0.0, 100.0)) as f64
    }

    /// p50 in nanoseconds.
    #[inline]
    #[must_use]
    pub fn p50(&self) -> f64 {
        self.percentile(50.0)
    }

    /// p90 in nanoseconds.
    #[inline]
    #[must_use]
    pub fn p90(&self) -> f64 {
        self.percentile(90.0)
    }

    /// p99 in nanoseconds.
    #[inline]
    #[must_use]
    pub fn p99(&self) -> f64 {
        self.percentile(99.0)
    }

    /// p99.9 in nanoseconds.
    #[inline]
    #[must_use]
    pub fn p999(&self) -> f64 {
        self.percentile(99.9)
    }

    /// Mean of the recorded samples, in nanoseconds.
    #[must_use]
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    /// Maximum recorded sample, in nanoseconds.
    #[must_use]
    pub fn max(&self) -> f64 {
        self.inner.max() as f64
    }

    /// Number of recorded samples.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len() as usize
    }

    /// True if no samples have been recorded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.len() == 0
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_NANOS)
    }
}
