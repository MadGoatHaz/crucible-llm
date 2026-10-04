//! Chunk 17 — Engine D (Hardware & Energy Efficiency Profiler) integration
//! tests.
//!
//! Covers the plan acceptance criteria that are verifiable offline:
//!
//! * `Joules/Token = ∫P(t)dt / Total_Generated_Tokens` (blueprint §5D)
//!   over synthetic power traces (trapezoidal integration, window
//!   slicing);
//! * the N/A rule — a machine without a GPU/driver reports all hardware
//!   fields as N/A (`None` / `0.0` sentinels) and the app keeps working
//!   (never panics);
//! * the VRAM fragmentation warning fires as occupancy approaches the
//!   threshold;
//! * the 100 ms poller merges into the lock-free `ArcSwap<MetricsSnapshot>`
//!   pipeline (blueprint §4.2) without touching the timing path.
//!
//! The NVML-dependent path is exercised (feature-gated) without
//! requiring a GPU: `NvmlBackend::try_init()` must return `None` — never
//! panic — on a driver-less host.

use std::time::Duration;

use crucible_llm::engines::hardware::{
    fragmentation_warning, integrate_joules, joules_per_token, profile,
    VRAM_FRAGMENTATION_THRESHOLD,
};
use crucible_llm::hw::{merge_hw, HwPoller, HwSample};
use crucible_llm::metrics::state::{MetricsSnapshot, MetricsState};
use crucible_llm::timing::MonotonicInstant;

/// A sample with an explicit power reading (and optional VRAM pair).
fn sample(t: MonotonicInstant, power_mw: Option<u64>, vram: Option<(u64, u64)>) -> HwSample {
    HwSample {
        t,
        power_mw,
        vram_used_bytes: vram.as_ref().map(|(used, _)| *used),
        vram_total_bytes: vram.as_ref().map(|(_, total)| *total),
        ..HwSample::default()
    }
}

#[test]
fn constant_power_integrates_exactly() {
    // 50 samples at 100 W (100_000 mW) taken back-to-back: the
    // trapezoidal rule of a constant telescopes exactly —
    // ∫ = P × (t_last − t_first), independent of the actual spacing
    // (including zero-width gaps when the cycle clock doesn't advance).
    let samples: Vec<HwSample> = (0..50)
        .map(|_| sample(MonotonicInstant::now(), Some(100_000), None))
        .collect();
    let first = samples.first().unwrap().t;
    let last = samples.last().unwrap().t;
    let span_s = first.delta_nanos(&last) as f64 / 1e9;

    let (joules, has_power) = integrate_joules(&samples, None);
    assert!(has_power);
    assert!(
        (joules - 100.0 * span_s).abs() < 1e-9,
        "joules={joules} span={span_s}"
    );
}

#[test]
fn window_slicing_integrates_only_the_active_window() {
    let a = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(5));
    let b = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(5));
    let c = MonotonicInstant::now();
    let samples = vec![
        sample(a, Some(1_000), None),
        sample(b, Some(1_000), None),
        sample(c, Some(1_000), None),
    ];
    // Whole trace: 1 W across (a → c).
    let (full, _) = integrate_joules(&samples, None);
    let full_span = a.delta_nanos(&c) as f64 / 1e9;
    assert!((full - full_span).abs() < 1e-9, "full={full}");
    // The [b, c] window (the active token output window): only (b → c).
    let (win, has) = integrate_joules(&samples, Some((b, c)));
    assert!(has);
    let win_span = b.delta_nanos(&c) as f64 / 1e9;
    assert!((win - win_span).abs() < 1e-9, "win={win}");
    assert!(win < full);
}

#[test]
fn joules_per_token_matches_the_blueprint_formula() {
    // 200 W constant; the run window covers the whole trace; 1000
    // generated tokens → J/token = ∫P dt / 1000 (blueprint §5D).
    let samples: Vec<HwSample> = (0..20)
        .map(|_| sample(MonotonicInstant::now(), Some(200_000), None))
        .collect();
    let window = (samples.first().unwrap().t, samples.last().unwrap().t);

    let jpt = joules_per_token(&samples, Some(window), 1000).expect("power present");
    let expected = 200.0 * (window.0.delta_nanos(&window.1) as f64 / 1e9) / 1000.0;
    assert!(
        (jpt - expected).abs() < 1e-9,
        "jpt={jpt} expected={expected}"
    );

    // Zero tokens → N/A, never a spurious 0.0.
    assert_eq!(joules_per_token(&samples, Some(window), 0), None);
}

#[test]
fn missing_power_is_na_not_zero() {
    let a = MonotonicInstant::now();
    let b = MonotonicInstant::now();
    let samples = vec![
        sample(a, None, Some((10_000, 100_000))),
        sample(b, None, Some((20_000, 100_000))),
    ];
    let (joules, has_power) = integrate_joules(&samples, None);
    assert!(!has_power);
    assert_eq!(joules, 0.0);
    assert_eq!(joules_per_token(&samples, None, 100), None);

    let p = profile(&samples, None, 100);
    assert_eq!(p.joules_per_token, None);
    assert_eq!(p.peak_power_w, None);
    assert_eq!(p.avg_power_w, None);
    // VRAM stats are still available without power data.
    assert_eq!(p.peak_vram_bytes, Some(20_000));
    assert_eq!(p.peak_vram_ratio, Some(0.2));
    assert_eq!(p.fragmentation_warning, None);
}

#[test]
fn fragmentation_warning_fires_at_the_threshold() {
    // 100 GB total; 90 GB used = exactly the threshold → warning.
    let gb = 1_000_000_000u64;
    assert!(fragmentation_warning(90 * gb, 100 * gb).is_some());
    assert!(fragmentation_warning(95 * gb, 100 * gb).is_some());
    assert!(fragmentation_warning(89 * gb, 100 * gb).is_none());
    // Unknown total → N/A (never a false warning).
    assert!(fragmentation_warning(95 * gb, 0).is_none());
    assert_eq!(
        VRAM_FRAGMENTATION_THRESHOLD, 0.9,
        "threshold must stay 90% (blueprint §5D)"
    );

    // The profile surfaces the warning for a hot VRAM window.
    let a = MonotonicInstant::now();
    let b = MonotonicInstant::now();
    let samples = vec![
        sample(a, Some(50_000), Some((80 * gb, 100 * gb))),
        sample(b, Some(50_000), Some((92 * gb, 100 * gb))),
    ];
    let p = profile(&samples, None, 10);
    assert!(p.fragmentation_warning.is_some());
    assert!(p.fragmentation_warning.unwrap().contains("92%"));
}

#[test]
fn poller_degrades_to_na_without_a_gpu() {
    // The Chunk 17 acceptance: on a machine without a GPU, all hardware
    // fields report N/A and the app runs normally (never panics).
    let mut poller = HwPoller::new();
    for _ in 0..3 {
        let s = poller.poll();
        if !poller.has_gpu() {
            assert!(s.vram_used_bytes.is_none(), "VRAM must be N/A");
            assert!(s.vram_total_bytes.is_none(), "VRAM must be N/A");
            assert!(s.power_mw.is_none(), "power must be N/A");
            assert!(s.gpu_clock_mhz.is_none(), "clock must be N/A");
            assert!(s.gpu_temp_c.is_none(), "temp must be N/A");
        }
        // Cross-platform CPU/RAM (sysinfo) is always available on a
        // supported host.
        assert!(s.cpu_usage_pct.is_some());
        assert!(s.ram_total_bytes.is_some());
        assert!(s.ram_used_bytes.is_some());
    }
}

#[test]
fn poller_tick_publishes_to_the_lock_free_snapshot() {
    // Blueprint §4.2: the poller merges into the `ArcSwap<MetricsSnapshot>`;
    // the reader side (`load`) is lock-free and sees a complete snapshot.
    let state = MetricsState::new();
    let mut poller = HwPoller::new();
    poller.tick(&state);

    let snap = state.load();
    if poller.has_gpu() {
        assert!(snap.vram_total_gb > 0.0);
        assert!(snap.power_w >= 0.0);
    } else {
        // N/A → the `0.0` sentinels the views render as "N/A".
        assert_eq!(snap.vram_used_gb, 0.0);
        assert_eq!(snap.vram_total_gb, 0.0);
        assert_eq!(snap.power_w, 0.0);
        assert_eq!(snap.gpu_clock_mhz, 0.0);
    }
    // Non-hardware snapshot fields pass through the merge untouched.
    let before = state.load().endpoint.clone();
    poller.tick(&state);
    assert_eq!(state.load().endpoint, before);
}

#[test]
fn merge_hw_is_a_pure_na_mapping() {
    let base = MetricsSnapshot::default();
    let na = HwSample::default(); // every field None
    let merged = merge_hw(&base, &na, None, None);
    assert_eq!(merged.vram_used_gb, 0.0);
    assert_eq!(merged.vram_total_gb, 0.0);
    assert_eq!(merged.power_w, 0.0);
    assert_eq!(merged.gpu_clock_mhz, 0.0);
    assert_eq!(merged.joules_per_token, 0.0);
    assert!(merged.gpu.is_none()); // no GPU sample → the panel is hidden

    let full = HwSample {
        vram_used_bytes: Some(21_400_000_000),
        vram_total_bytes: Some(24_000_000_000),
        power_mw: Some(285_000),
        gpu_clock_mhz: Some(1410),
        ..HwSample::default()
    };
    let gpu = crucible_llm::hw::GpuSample {
        power_watts: Some(285.0),
        ..Default::default()
    };
    let merged = merge_hw(&base, &full, Some(0.338), Some(&gpu));
    assert!((merged.vram_used_gb - 21.4).abs() < 1e-9);
    assert!((merged.vram_total_gb - 24.0).abs() < 1e-9);
    assert!((merged.power_w - 285.0).abs() < 1e-9);
    assert_eq!(merged.gpu_clock_mhz, 1410.0);
    assert_eq!(merged.joules_per_token, 0.338);
    assert!(merged.gpu.is_some()); // the full sample is carried for the panel
}

// ── NVML feature-gated path ────────────────────────────────────────────────

#[cfg(feature = "nvml")]
#[test]
fn nvml_init_degrades_gracefully_without_a_driver() {
    use crucible_llm::hw::GpuBackend;
    // Chunk 17 acceptance: no panic on a driver-less host — the `None`
    // is the graceful path (and a `Some` means a real GPU answered).
    if let Some(b) = crucible_llm::hw::nvml::NvmlBackend::try_init() {
        let s = b.poll();
        assert!(s.memory_total_mb.is_some());
        assert!(!b.name().is_empty());
    }
}
