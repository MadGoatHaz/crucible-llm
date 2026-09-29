//! Integration tests for Chunk 2: the `MonotonicInstant` timing core and the
//! `LatencyHistogram` percentile wrapper.

use std::time::Duration;

use crucible_llm::metrics::LatencyHistogram;
use crucible_llm::timing::{MonotonicInstant, StreamTimestamps};

/// Assert `actual` is within `tol_pct` percent of `expected`.
fn within_pct(actual: f64, expected: f64, tol_pct: f64) -> bool {
    let tol = expected * tol_pct / 100.0;
    (actual - expected).abs() <= tol
}

#[test]
fn histogram_recovers_percentiles_within_2pct() {
    let mut h = LatencyHistogram::new(1_000_000);
    // Uniform distribution over [1, 10_000] ns: the true p50/p90/p99 are
    // 5_000 / 9_000 / 9_900 ns respectively.
    for v in 1..=10_000 {
        h.record(v);
    }

    assert_eq!(h.len(), 10_000);
    assert!(!h.is_empty());

    assert!(within_pct(h.p50(), 5_000.0, 2.0), "p50={}", h.p50());
    assert!(within_pct(h.p90(), 9_000.0, 2.0), "p90={}", h.p90());
    assert!(within_pct(h.p99(), 9_900.0, 2.0), "p99={}", h.p99());

    // Percentiles must be monotonically non-decreasing.
    assert!(h.p50() <= h.p90());
    assert!(h.p90() <= h.p99());
    assert!(h.p99() <= h.p999());
}

#[test]
fn histogram_empty_returns_zero() {
    let h = LatencyHistogram::default();
    assert!(h.is_empty());
    assert_eq!(h.len(), 0);
    assert_eq!(h.percentile(50.0), 0.0);
    assert_eq!(h.p99(), 0.0);
}

#[test]
fn histogram_clamps_out_of_range_samples() {
    let mut h = LatencyHistogram::new(1_000);
    // 0 and values above the max must clamp into [1, 1000] without panicking.
    h.record(0);
    h.record(1_000_000);
    h.record(500);
    assert_eq!(h.len(), 3);
    assert_eq!(h.max(), 1_000.0);
}

#[test]
fn instant_delta_is_monotonic_and_non_negative() {
    // An instant measured against itself is exactly zero.
    let a = MonotonicInstant::now();
    assert_eq!(a.delta_nanos(&a), 0);

    // Monotonic: for three time-ordered instants t0 < t1 < t2, the longer
    // span is never shorter than the shorter one. The `u64` return type
    // guarantees every delta is non-negative by construction.
    let t0 = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(20));
    let t1 = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(40));
    let t2 = MonotonicInstant::now();
    assert!(t0.delta_nanos(&t2) >= t0.delta_nanos(&t1));

    // A strictly later instant (well above any clock granularity) yields a
    // positive delta in the forward direction...
    let earlier = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(100));
    let later = MonotonicInstant::now();
    assert!(earlier.delta_nanos(&later) > 0);

    // ...and clamps to exactly 0 in the reverse direction (never negative,
    // never a wrapped large value).
    assert_eq!(later.delta_nanos(&earlier), 0);
}

#[test]
fn stream_timestamps_ttft_and_total() {
    let mut ts = StreamTimestamps::new();
    assert!(!ts.is_complete());

    ts.t0 = Some(MonotonicInstant::now());
    std::thread::sleep(Duration::from_millis(1));
    ts.t1 = Some(MonotonicInstant::now());
    std::thread::sleep(Duration::from_millis(10));
    ts.t3 = Some(MonotonicInstant::now());
    std::thread::sleep(Duration::from_millis(10));
    ts.t_end = Some(MonotonicInstant::now());

    assert!(ts.is_complete());

    // TTFT = T3 - T1 ~= 10 ms = 10_000_000 ns (generous tolerance for the
    // synthetic sleep).
    let ttft = ts.ttft_nanos().expect("t1 and t3 both recorded");
    assert!(ttft > 5_000_000 && ttft < 50_000_000, "ttft={ttft}");

    // Total = Tn - T0 ~= 21 ms, always greater than TTFT.
    let total = ts.total_nanos().expect("t0 and t_end both recorded");
    assert!(total > ttft, "total must exceed ttft");
}

#[test]
fn stream_timestamps_missing_milestones_yield_none() {
    let ts = StreamTimestamps::new();
    assert_eq!(ts.ttft_nanos(), None);
    assert_eq!(ts.total_nanos(), None);
    assert_eq!(ts.first_byte_nanos(), None);
}
