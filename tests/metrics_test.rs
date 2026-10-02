//! Integration tests for the metrics pipeline:
//! - Chunk 2: the `MonotonicInstant` timing core and the `LatencyHistogram`
//!   percentile wrapper.
//! - Chunk 6: the `ArcSwap<MetricsSnapshot>` double-buffered snapshot the TUI
//!   reads lock-free.

use std::sync::Arc;
use std::time::Duration;

use crucible_llm::metrics::{
    LatencyHistogram, MetricsSnapshot, MetricsState, StreamMetric, StreamStatus,
};
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

// ---------------------------------------------------------------------------
// Chunk 6: ArcSwap metrics snapshot pipeline
// ---------------------------------------------------------------------------

/// The default snapshot is fully zeroed / empty.
#[test]
fn snapshot_default_is_zeroed() {
    let s = MetricsSnapshot::default();
    assert_eq!(s.aggregate_tps, 0.0);
    assert_eq!(s.prompt_throughput, 0.0);
    assert_eq!(s.active_streams, 0);
    assert_eq!(s.total_streams, 0);
    assert_eq!(s.prompt_tokens, 0);
    assert_eq!(s.completion_tokens, 0);
    assert_eq!(s.reasoning_tokens, 0);
    assert_eq!(s.itl_p50_ns, 0);
    assert_eq!(s.itl_p99_ns, 0);
    assert!(s.itl_bins.is_empty());
    assert!(s.streams.is_empty());
    assert!(s.throughput_series.is_empty());
    assert_eq!(s.status, StreamStatus::Waiting);
}

/// `update` then `load` round-trips every field exactly.
#[test]
fn snapshot_update_load_roundtrip() {
    let state = MetricsState::new();

    let snap = MetricsSnapshot {
        endpoint: "http://127.0.0.1:8000/v1".into(),
        backend: "vLLM".into(),
        model: "test-model".into(),
        mode: "Speed".into(),
        aggregate_tps: 123.45,
        active_streams: 3,
        total_streams: 4,
        vram_used_gb: 10.5,
        vram_total_gb: 24.0,
        power_w: 250.0,
        joules_per_token: 0.2,
        itl_p50_ns: 10_000_000,
        itl_p90_ns: 20_000_000,
        itl_p99_ns: 30_000_000,
        itl_p999_ns: 40_000_000,
        prompt_tokens: 512,
        completion_tokens: 256,
        reasoning_tokens: 128,
        status: StreamStatus::Streaming,
        streams: vec![StreamMetric {
            id: 7,
            kind: "Reasoning".into(),
            state: StreamStatus::Streaming,
            pp_tokens: Some(512),
            tg_tokens: Some(256),
            ttft_s: Some(0.12),
            gen_tps: Some(60.0),
            mtp: Some(1.5),
            progress: 0.5,
            looping: false,
        }],
        throughput_series: vec![1.0, 2.0, 3.0],
        ..Default::default()
    };

    state.update(snap);

    let loaded = state.load();
    assert_eq!(loaded.endpoint, "http://127.0.0.1:8000/v1");
    assert_eq!(loaded.backend, "vLLM");
    assert_eq!(loaded.model, "test-model");
    assert_eq!(loaded.mode, "Speed");
    assert!((loaded.aggregate_tps - 123.45).abs() < f64::EPSILON);
    assert_eq!(loaded.active_streams, 3);
    assert_eq!(loaded.total_streams, 4);
    assert!((loaded.vram_used_gb - 10.5).abs() < f64::EPSILON);
    assert!((loaded.power_w - 250.0).abs() < f64::EPSILON);
    assert_eq!(loaded.itl_p50_ns, 10_000_000);
    assert_eq!(loaded.itl_p999_ns, 40_000_000);
    assert_eq!(loaded.prompt_tokens, 512);
    assert_eq!(loaded.completion_tokens, 256);
    assert_eq!(loaded.reasoning_tokens, 128);
    assert_eq!(loaded.status, StreamStatus::Streaming);
    assert_eq!(loaded.streams.len(), 1);
    assert_eq!(loaded.streams[0].id, 7);
    assert_eq!(loaded.streams[0].kind, "Reasoning");
    // A writer-seeded window is adopted on the first publish.
    assert_eq!(loaded.throughput_series, vec![1.0, 2.0, 3.0]);
    // Prompt throughput is derived: 512 prompt tokens / 0.12 s TTFT.
    assert!((loaded.prompt_throughput - 512.0 / 0.12).abs() < 1e-9);
}

/// The rolling window persists across `update()` calls (the engines
/// publish fresh `..Default::default()` snapshots, which would otherwise
/// wipe the series on every batch) and samples at most once per second.
///
/// The series samples the **cumulative decode rate**
/// (`completion_tokens / elapsed_since_first_token`), so the exact values
/// depend on wall-clock timing. The test verifies the structural
/// behavior: one sample per period, persistence across updates, and
/// positive rates when tokens are present.
#[test]
fn update_maintains_the_rolling_throughput_series() {
    let state = MetricsState::new();
    // The first update latches the first-token origin; with ~0 elapsed the
    // tracker emits no sample (a divide-by-near-zero would spike and
    // dominate the chart's auto-scaled y-axis).
    state.update(MetricsSnapshot {
        aggregate_tps: 100.0,
        completion_tokens: 100,
        observed_frames: 100,
        ..Default::default()
    });
    assert_eq!(state.load().throughput_series.len(), 0);

    // A fresh default snapshot (as every engine batch publishes) does not
    // add a sample (no tokens) and does not wipe the (empty) window.
    state.update(MetricsSnapshot::default());
    assert_eq!(state.load().throughput_series.len(), 0);

    // After a real elapsed (1.1 s) the first sample is recorded — a bounded
    // cumulative rate (200 tokens / ~1.1 s), never a spike.
    std::thread::sleep(Duration::from_millis(1100));
    state.update(MetricsSnapshot {
        aggregate_tps: 200.0,
        completion_tokens: 200,
        observed_frames: 200,
        ..Default::default()
    });
    let series = state.load().throughput_series.clone();
    assert_eq!(
        series.len(),
        1,
        "one sample after a real elapsed: {series:?}"
    );
    assert!(series[0] > 0.0, "sample must be positive: {}", series[0]);
    assert!(
        series[0] < 1000.0,
        "sample must be bounded, not a divide-by-floor spike: {}",
        series[0]
    );
}

/// A later `update` atomically replaces the pointee; `load` sees the latest.
#[test]
fn snapshot_update_replaces_previous() {
    let state = MetricsState::new();
    state.update(MetricsSnapshot {
        aggregate_tps: 842.3,
        ..Default::default()
    });
    assert_eq!(state.load().aggregate_tps, 842.3);

    let s = MetricsSnapshot {
        aggregate_tps: 1.0,
        ..Default::default()
    };
    state.update(s);
    assert_eq!(state.load().aggregate_tps, 1.0);
}

/// Many concurrent lock-free readers never block while a writer publishes.
#[test]
fn snapshot_lockfree_reads_concurrent_with_updates() {
    let state = Arc::new(MetricsState::new());
    let mut handles = vec![];

    // 8 reader threads: tight loop of `load()` + field read. `load()` is a
    // lock-free atomic read, so none of these can block on the writer.
    for _ in 0..8 {
        let s = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            let mut reads = 0u64;
            let start = std::time::Instant::now();
            while start.elapsed() < Duration::from_millis(50) {
                let snap = s.load();
                let _ = snap.aggregate_tps; // plain immutable read
                reads += 1;
            }
            reads
        }));
    }

    // One writer thread publishing a fresh snapshot each iteration.
    {
        let s = Arc::clone(&state);
        handles.push(std::thread::spawn(move || {
            let mut v = 0u64;
            let start = std::time::Instant::now();
            while start.elapsed() < Duration::from_millis(50) {
                v += 1;
                let snap = MetricsSnapshot {
                    aggregate_tps: v as f64,
                    status: StreamStatus::Streaming,
                    ..Default::default()
                };
                s.update(snap);
            }
            v
        }));
    }

    let total: u64 = handles
        .into_iter()
        .map(|h| h.join().expect("reader/writer thread panicked (deadlock?)"))
        .sum();

    assert!(total > 0, "no thread made progress");
    // The writer published at least one snapshot with a positive t/s.
    let final_snap = state.load();
    assert!(final_snap.aggregate_tps > 0.0, "writer never published");
    assert_eq!(final_snap.status, StreamStatus::Streaming);
}

/// Nanosecond percentiles convert to milliseconds for display.
#[test]
fn snapshot_ns_to_ms_conversion() {
    let s = MetricsSnapshot {
        itl_p50_ns: 12_100_000,  // 12.1 ms
        itl_p90_ns: 16_400_000,  // 16.4 ms
        itl_p99_ns: 41_200_000,  // 41.2 ms
        itl_p999_ns: 55_000_000, // 55.0 ms
        ..Default::default()
    };

    assert!((s.itl_p50_ms() - 12.1).abs() < 1e-6);
    assert!((s.itl_p90_ms() - 16.4).abs() < 1e-6);
    assert!((s.itl_p99_ms() - 41.2).abs() < 1e-6);
    assert!((s.itl_p999_ms() - 55.0).abs() < 1e-6);
}

/// The snapshot pulls ITL percentiles directly from the Chunk 2 histogram.
#[test]
fn snapshot_from_histogram() {
    let mut h = LatencyHistogram::new(1_000_000);
    for v in 1..=10_000 {
        h.record(v);
    }
    let s = MetricsSnapshot::default().with_itl_percentiles(&h);
    // Uniform [1, 10_000] ns → p50 ≈ 5_000, p90 ≈ 9_000, p99 ≈ 9_900.
    assert!(
        within_pct(s.itl_p50_ns as f64, 5_000.0, 2.0),
        "p50={}",
        s.itl_p50_ns
    );
    assert!(
        within_pct(s.itl_p90_ns as f64, 9_000.0, 2.0),
        "p90={}",
        s.itl_p90_ns
    );
    assert!(
        within_pct(s.itl_p99_ns as f64, 9_900.0, 2.0),
        "p99={}",
        s.itl_p99_ns
    );
}

/// Every `StreamStatus` maps to its `STATE` column label.
#[test]
fn stream_status_labels() {
    assert_eq!(StreamStatus::Waiting.label(), "Waiting");
    assert_eq!(StreamStatus::Streaming.label(), "Streaming");
    assert_eq!(StreamStatus::Done.label(), "Done");
    assert_eq!(StreamStatus::Error.label(), "Error");
}
