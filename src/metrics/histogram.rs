//! `LatencyHistogram`: a thin wrapper over `hdrhistogram` exposing
//! `record(nanos)` and `percentile(p) -> f64` for p50/p90/p99/p99.9.
//!
//! The recording path is non-panicking: a sample **above** the upper bound
//! (a saturated-server gap beyond `max_ns`) is **clamped** into range and
//! counted in [`clamp_count`](Self::clamp_count), so a saturated histogram is
//! never silently presented as an accurate distribution (CF-24). The low end
//! clamps to 1 ns (the sub-quantum floor), preserving the generic
//! `record(0) → 1` contract other callers rely on.
//!
//! De-polluting the inter-token distribution from MTP / speculative-decoding
//! *batched* frames (CF-09) is handled at the **stamping** site, not here:
//! [`StreamWorker::emit_frames`](crate::client::stream) stamps one `now()` per
//! frame, so batched frames never collapse to a 0 (→1 ns) delta. Keeping that
//! logic out of `record()` is what lets the generic clamp contract above stay
//! intact for every other consumer.

use hdrhistogram::Histogram;

/// Default upper bound for a single latency sample: 60 seconds in
/// nanoseconds (CF-24).
///
/// Raised from 1 s so that genuinely long inter-token gaps (a saturated /
/// heavily loaded server stalling between tokens) are captured as real
/// measurements instead of being clamped to a 1 s ceiling — the old cap made
/// a saturated server's p99/p99.9 read exactly "1000 ms", a cap, not a value.
/// Any gap still beyond 60 s is clamped and counted (`clamp_count`) so a
/// saturated histogram is never silently presented as accurate.
pub const DEFAULT_MAX_NANOS: u64 = 60_000_000_000;

/// A high-dynamic-range histogram of latency samples (in nanoseconds),
/// wrapping `hdrhistogram` for p50/p90/p99/p99.9 queries.
///
/// The recording path is non-panicking: a sample above `max_ns` is clamped
/// and counted (CF-24), and the low end clamps to 1 ns — so a stray
/// out-of-range value can never trip the underlying `record` error.
pub struct LatencyHistogram {
    inner: Histogram<u64>,
    max_ns: u64,
    /// CF-24: samples clamped to `max_ns` because they exceeded it.
    clamped: u64,
}

impl LatencyHistogram {
    /// Create a histogram tracking samples in `[1, max_ns]` nanoseconds with
    /// 3 significant digits of precision (max ~0.1% quantization error).
    pub fn new(max_ns: u64) -> Self {
        let max_ns = max_ns.max(1);
        let inner = Histogram::new_with_bounds(1, max_ns, 3)
            .expect("histogram bounds [1, max_ns] are always valid");
        Self {
            inner,
            max_ns,
            clamped: 0,
        }
    }

    /// Record a single latency sample in nanoseconds.
    ///
    /// * `ns > max_ns` (a saturated-server gap beyond the bound) is
    ///   **clamped** to `max_ns` and counted in `clamped` (CF-24), so a
    ///   saturated histogram is never silently presented as accurate.
    /// * the low end clamps to 1 ns (a 0 sample is the sub-quantum floor —
    ///   the generic contract other callers rely on).
    /// * everything in `[1, max_ns]` is recorded as-is.
    #[inline]
    pub fn record(&mut self, ns: u64) {
        let ns = if ns > self.max_ns {
            self.clamped += 1;
            self.max_ns
        } else {
            ns.clamp(1, self.max_ns)
        };
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

    /// Number of samples **clamped** to the upper bound because they
    /// exceeded `max_ns` (CF-24) — a saturated-server gap beyond the
    /// histogram's range. A non-zero count means the percentile figures are
    /// capped (a ceiling), not exact measurements.
    #[must_use]
    pub fn clamp_count(&self) -> u64 {
        self.clamped
    }
}

impl Default for LatencyHistogram {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_NANOS)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // CF-09 (de-polluting the ITL from MTP / speculative-decoding *batched*
    // frames) is exercised at the *stamping* site —
    // `StreamWorker::emit_frames` (see the `stream` module tests), which
    // stamps one `now()` per frame so batched frames never collapse to a 0
    // (->1 ns) delta. It is deliberately *not* done here: `record(0)` keeps
    // its generic clamp-to-1 contract that other callers (and
    // `tests/metrics_test.rs`) rely on. This module's CF-24 work — the cap +
    // clamp count — is tested below.

    /// CF-24: the default cap is 60 s, so long inter-token gaps (well under
    /// 60 s) are captured as real measurements rather than clamped to a 1 s
    /// ceiling.
    #[test]
    fn default_cap_is_60_seconds() {
        assert_eq!(DEFAULT_MAX_NANOS, 60_000_000_000);
        let mut h = LatencyHistogram::default();
        // A 5 s gap (within 60 s) is recorded as-is (within 3-sig-fig
        // quantization), not clamped to the old 1 s ceiling.
        h.record(5_000_000_000);
        assert_eq!(h.clamp_count(), 0);
        assert!(
            (h.max() - 5_000_000_000.0).abs() < 5_000_000_000.0 * 0.005,
            "max {} should be ~5 s, not clamped",
            h.max()
        );
    }

    /// CF-24: a gap beyond the 60 s cap is clamped into range AND counted, so
    /// a saturated histogram is never silently presented as accurate.
    #[test]
    fn a_gap_beyond_the_cap_is_clamped_and_counted() {
        let mut h = LatencyHistogram::default();
        h.record(90_000_000_000); // 90 s > 60 s cap
        assert_eq!(h.clamp_count(), 1, "the overflow must be counted");
        // The recorded value is the cap (within 3-sig-fig quantization), not
        // the raw 90 s.
        assert!(
            (h.max() - DEFAULT_MAX_NANOS as f64).abs() < (DEFAULT_MAX_NANOS as f64) * 0.005,
            "max {} should be clamped to ~60 s, not 90 s",
            h.max()
        );
    }
}
