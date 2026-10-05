//! Engine D — Hardware & Energy Efficiency Profiler (plan Chunk 17,
//! blueprint §5D).
//!
//! Correlates the GPU power trace `P(t)` with the active token output
//! windows and computes the **Silicon Efficiency Metric**:
//!
//! ```text
//! Joules/Token = ∫ P(t) dt / Total_Generated_Tokens
//! ```
//!
//! The integral is a trapezoidal rule over the [`HwSample`] power trace
//! (the 100 ms hardware poller, blueprint §4.3), optionally sliced to a
//! measured run window (`T0` → `Tn`). Every function here is **pure** —
//! no NVML, no locks, no quanta timing path (measurement-isolation
//! invariant, blueprint §4) — so the whole metric is unit-testable with
//! synthetic traces.
//!
//! **N/A rule** (blueprint §5D graceful degradation): when no power
//! telemetry is available (no GPU, no driver, feature off) the joules
//! integrals are `0.0` **and** every derived `joules_per_token` is
//! `None` — rendered `N/A`, never `0.0` (a `0.0` J/token would claim
//! "free" inference).
//!
//! **VRAM fragmentation:** the blueprint asks the profiler to warn users
//! when VRAM usage approaches the fragmentation threshold
//! ([`VRAM_FRAGMENTATION_THRESHOLD`]).

use crate::hw::{hw_sample_from_gpu, GpuBackend, HwSample, HW_POLL_INTERVAL_MS};
use crate::timing::MonotonicInstant;

/// VRAM occupancy (used / total) at which the fragmentation warning fires
/// (blueprint §5D: "warns users when approaching memory fragmentation
/// thresholds").
pub const VRAM_FRAGMENTATION_THRESHOLD: f64 = 0.90;

/// The silicon-efficiency result for one measured window.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyResult {
    /// `∫ P(t) dt` over the window, in joules (`0.0` when no power
    /// samples exist — see the module N/A rule).
    pub joules: f64,
    /// `joules / total_generated_tokens`. `None` when no power telemetry
    /// was available **or** no tokens were generated (N/A, never `0.0`).
    pub joules_per_token: Option<f64>,
    /// Mean power over the window, watts (`None` without power data).
    pub avg_power_w: Option<f64>,
    /// Peak instantaneous power, watts.
    pub peak_power_w: Option<f64>,
    /// Peak VRAM occupancy observed in the window, bytes.
    pub peak_vram_bytes: Option<u64>,
    /// Peak VRAM ratio (0.0..=1.0).
    pub peak_vram_ratio: Option<f64>,
    /// `Some` when the peak VRAM ratio reached the fragmentation
    /// threshold (the blueprint §5D warning text).
    pub fragmentation_warning: Option<String>,
    /// Mean GPU compute utilization over the window, % (0–100). `None`
    /// without utilization telemetry (e.g. the Intel sysfs path).
    pub avg_utilization_pct: Option<f64>,
    /// The GPU vendor (`"NVIDIA"` / `"AMD"` / `"Intel"`), when a backend
    /// produced the samples.
    pub vendor: Option<String>,
    /// The GPU's model / driver label, when known.
    pub model: Option<String>,
}

/// True when `t` lies within the closed window `[start, end]`.
///
/// `MonotonicInstant` is a clamped-delta clock (no `Ord` impl), so
/// ordering is derived from [`MonotonicInstant::delta_nanos`]: `x` is not
/// after `y` iff `y.delta_nanos(&x) == 0`.
fn within(t: &MonotonicInstant, start: &MonotonicInstant, end: &MonotonicInstant) -> bool {
    t.delta_nanos(start) == 0 && end.delta_nanos(t) == 0
}

/// Trapezoidal integration of the power trace: `∫ P(t) dt` in joules.
///
/// `window` (optional) slices the trace to a measured run (`T0` → `Tn`);
/// `None` integrates the whole trace. Samples without a power reading are
/// skipped — the trapezoid spans from the last known reading to the next,
/// i.e. linear interpolation across gaps (the standard treatment for
/// missing sensor samples).
///
/// Returns `(joules, has_power)`: `has_power` is `false` when no sample
/// in scope carried a reading, in which case `joules` is `0.0` and
/// callers **must** report any derived metric as N/A (module N/A rule).
#[must_use]
pub fn integrate_joules(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
) -> (f64, bool) {
    let mut joules = 0.0f64;
    let mut has_power = false;
    let mut prev: Option<(&MonotonicInstant, f64)> = None;

    for s in samples {
        let in_window = match window {
            Some((start, end)) => within(&s.t, &start, &end),
            None => true,
        };
        if !in_window {
            continue;
        }
        // Total *system* draw (GPU + CPU), watts. A sample with neither
        // contributes nothing (the N/A rule) — the trapezoid spans the gap.
        let p_w = s.power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0)
            + s.cpu_power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0);
        if p_w <= 0.0 {
            continue;
        }
        has_power = true;
        if let Some((t_prev, p_prev)) = prev {
            let dt_s = t_prev.delta_nanos(&s.t) as f64 / 1e9;
            joules += (p_prev + p_w) / 2.0 * dt_s;
        }
        prev = Some((&s.t, p_w));
    }

    (joules, has_power)
}

/// The Silicon Efficiency Metric for a window: `∫P(t)dt / tokens`.
///
/// `None` when no power telemetry is available or `tokens` is zero
/// (N/A rule — never a spurious `0.0`).
#[must_use]
pub fn joules_per_token(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
    tokens: u64,
) -> Option<f64> {
    let (joules, has_power) = integrate_joules(samples, window);
    if has_power && tokens > 0 {
        Some(joules / tokens as f64)
    } else {
        None
    }
}

/// Full Engine D profile for a window: energy, power stats, peak VRAM,
/// and the fragmentation warning.
#[must_use]
pub fn profile(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
    total_tokens: u64,
) -> EnergyResult {
    let (joules, has_power) = integrate_joules(samples, window);

    // In-scope samples (window applied once for the stats).
    let in_scope: Vec<&HwSample> = samples
        .iter()
        .filter(|s| match window {
            Some((start, end)) => within(&s.t, &start, &end),
            None => true,
        })
        .collect();

    // Peak *system* draw (GPU + CPU) over the in-scope samples.
    let peak = in_scope.iter().fold(0.0_f64, |max, s| {
        let p = s.power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0)
            + s.cpu_power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0);
        max.max(p)
    });
    let peak_power_w = (peak > 0.0).then_some(peak);

    // Mean power = energy / span of the power-bearing samples in scope.
    let avg_power_w = if has_power {
        let ts: Vec<&MonotonicInstant> = in_scope
            .iter()
            .filter(|s| {
                (s.power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0)
                    + s.cpu_power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0))
                    > 0.0
            })
            .map(|s| &s.t)
            .collect();
        match (ts.first(), ts.last()) {
            (Some(first), Some(last)) if first != last => {
                let span_s = first.delta_nanos(last) as f64 / 1e9;
                (span_s > 0.0).then(|| joules / span_s)
            }
            _ => None,
        }
    } else {
        None
    };

    let peak_vram_bytes = in_scope.iter().filter_map(|s| s.vram_used_bytes).max();
    let peak_vram_ratio = match (
        peak_vram_bytes,
        in_scope.iter().filter_map(|s| s.vram_total_bytes).max(),
    ) {
        (Some(used), Some(total)) if total > 0 => {
            Some((used as f64 / total as f64).clamp(0.0, 1.0))
        }
        _ => None,
    };
    let fragmentation_warning = peak_vram_bytes
        .zip(in_scope.iter().filter_map(|s| s.vram_total_bytes).max())
        .and_then(|(used, total)| fragmentation_warning(used, total));

    // Mean compute utilization over the in-scope samples that report it.
    let avg_utilization_pct = {
        let utils: Vec<f64> = in_scope
            .iter()
            .filter_map(|s| s.gpu_util_pct)
            .map(|u| u as f64)
            .collect();
        (!utils.is_empty()).then(|| utils.iter().sum::<f64>() / utils.len() as f64)
    };

    EnergyResult {
        joules,
        joules_per_token: (has_power && total_tokens > 0).then(|| joules / total_tokens as f64),
        avg_power_w,
        peak_power_w,
        peak_vram_bytes,
        peak_vram_ratio,
        fragmentation_warning,
        avg_utilization_pct,
        // Vendor / model are filled by the caller (the backend that
        // produced the samples) — the pure trace math never knows them.
        vendor: None,
        model: None,
    }
}

/// The Engine D entry point that works **from a backend directly**: poll
/// `gpu` at 100 ms for `duration`, collect [`HwSample`]s, and compute the
/// full energy profile (avg/peak power, total joules, J/token, avg
/// utilization) over `tokens` — with the backend's vendor + model
/// attached to the result.
///
/// `None` backend → an all-N/A [`EnergyResult`] (the N/A rule: no power
/// telemetry, no spurious `0.0` J/token). This is a blocking call (it
/// sleeps at the poll cadence) — the standalone / headless path. The TUI
/// prefers the shared 100 ms background poller, which accumulates the
/// same trace without a second poller.
#[must_use]
pub fn profile_backend(
    gpu: Option<&dyn GpuBackend>,
    duration: std::time::Duration,
    tokens: u64,
) -> EnergyResult {
    let (vendor, model) = gpu
        .map(|g| (Some(g.vendor().to_string()), g.model().map(str::to_string)))
        .unwrap_or((None, None));
    let mut samples: Vec<HwSample> = Vec::new();
    let interval = std::time::Duration::from_millis(HW_POLL_INTERVAL_MS);
    let start = std::time::Instant::now();
    while start.elapsed() < duration {
        if let Some(g) = gpu {
            samples.push(hw_sample_from_gpu(MonotonicInstant::now(), &g.poll()));
        }
        std::thread::sleep(interval);
    }
    let mut r = profile(&samples, None, tokens);
    r.vendor = vendor;
    r.model = model;
    r
}

/// The blueprint §5D VRAM fragmentation warning: `Some` when occupancy
/// (used / total) reaches [`VRAM_FRAGMENTATION_THRESHOLD`].
///
/// `None` when the total is unknown (N/A rule) or the ratio is below the
/// threshold.
#[must_use]
pub fn fragmentation_warning(used_bytes: u64, total_bytes: u64) -> Option<String> {
    if total_bytes == 0 {
        return None;
    }
    let ratio = used_bytes as f64 / total_bytes as f64;
    (ratio >= VRAM_FRAGMENTATION_THRESHOLD).then(|| {
        format!(
            "VRAM at {:.0}% ({:.1} / {:.1} GB) — approaching fragmentation threshold ({:.0}%)",
            ratio * 100.0,
            used_bytes as f64 / 1e9,
            total_bytes as f64 / 1e9,
            VRAM_FRAGMENTATION_THRESHOLD * 100.0
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::GpuSample;
    use crate::timing::MonotonicInstant;

    #[test]
    fn integrate_constant_power_matches_hand_calculation() {
        // 100 W constant for ~1 s (10 samples, 100 ms apart):
        // ∫ = 100 J (trapezoid of a constant is exact).
        let t0 = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(1000));
        let t1 = MonotonicInstant::now();
        let samples = vec![
            hw_sample(t0, Some(100_000), None, None),
            hw_sample(t1, Some(100_000), None, None),
        ];
        let (joules, has_power) = integrate_joules(&samples, None);
        assert!(has_power);
        let span = t0.delta_nanos(&t1) as f64 / 1e9;
        assert!(
            (joules - 100.0 * span).abs() < 1e-6,
            "joules={joules} span={span}"
        );
    }

    #[test]
    fn window_slicing_excludes_out_of_scope_samples() {
        let a = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let b = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let c = MonotonicInstant::now();
        // The middle sample carries no power reading (a sensor gap).
        let samples = vec![
            hw_sample(a, Some(1_000), None, None),
            hw_sample(b, None, None, None),
            hw_sample(c, Some(1_000), None, None),
        ];
        // Full trace: the trapezoid spans the gap a → c:
        // (1 W + 1 W)/2 × ~0.4 s ≈ 0.4 J.
        let (full, has_full) = integrate_joules(&samples, None);
        assert!(has_full);
        assert!((0.3..0.5).contains(&full), "full={full}");
        // Window [b, c]: b has no power, c is a single point → 0 J but
        // `has_power` stays true (a reading exists in scope).
        let (win, has) = integrate_joules(&samples, Some((b, c)));
        assert!(has);
        assert_eq!(win, 0.0);
        // A window that contains only the power-less sample → N/A.
        let (empty, has_empty) = integrate_joules(&samples, Some((b, b)));
        assert!(!has_empty);
        assert_eq!(empty, 0.0);
    }

    #[test]
    fn missing_power_samples_yield_na_not_zero_claim() {
        let a = MonotonicInstant::now();
        let samples = vec![hw_sample(a, None, Some(10), Some(20))];
        let (joules, has_power) = integrate_joules(&samples, None);
        assert!(!has_power);
        assert_eq!(joules, 0.0);
        assert_eq!(joules_per_token(&samples, None, 100), None);
        assert_eq!(profile(&samples, None, 100).joules_per_token, None);
    }

    #[test]
    fn zero_tokens_is_na() {
        let a = MonotonicInstant::now();
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(50_000), None, None),
            hw_sample(b, Some(50_000), None, None),
        ];
        assert_eq!(joules_per_token(&samples, None, 0), None);
        assert_eq!(profile(&samples, None, 0).joules_per_token, None);
    }

    #[test]
    fn profile_reports_power_and_vram_stats() {
        let a = MonotonicInstant::now();
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(200_000), Some(90 * 1000), Some(100 * 1000)),
            hw_sample(b, Some(400_000), Some(95 * 1000), Some(100 * 1000)),
        ];
        let p = profile(&samples, None, 100);
        assert_eq!(p.peak_power_w, Some(400.0));
        assert!(p.avg_power_w.is_some());
        assert_eq!(p.peak_vram_bytes, Some(95_000));
        assert_eq!(p.peak_vram_ratio, Some(0.95));
        assert!(p.fragmentation_warning.is_some());
        assert!(p.joules_per_token.is_some());
    }

    #[test]
    fn fragmentation_warning_fires_at_threshold() {
        assert!(fragmentation_warning(90, 100).is_some());
        assert!(fragmentation_warning(95, 100).is_some());
        assert!(fragmentation_warning(89, 100).is_none());
        assert!(fragmentation_warning(100, 0).is_none()); // N/A total
        let msg = fragmentation_warning(95_000, 100_000).unwrap();
        assert!(msg.contains("95%"), "{msg}");
        assert!(msg.contains("fragmentation"), "{msg}");
    }

    fn hw_sample(
        t: MonotonicInstant,
        power_mw: Option<u64>,
        vram_used: Option<u64>,
        vram_total: Option<u64>,
    ) -> HwSample {
        HwSample {
            t,
            power_mw,
            vram_used_bytes: vram_used,
            vram_total_bytes: vram_total,
            ..HwSample::default()
        }
    }

    // ── profile_backend (the backend-driven Engine D path) ──────────────

    /// A deterministic mock backend so `profile_backend` is testable
    /// without any hardware (the N/A-never-panic contract holds here).
    #[derive(Debug)]
    struct MockBackend {
        vendor: &'static str,
        model: Option<&'static str>,
        watts: f64,
        util: u8,
    }

    impl GpuBackend for MockBackend {
        fn vendor(&self) -> &str {
            self.vendor
        }
        fn model(&self) -> Option<&str> {
            self.model
        }
        fn poll(&self) -> GpuSample {
            GpuSample {
                power_watts: Some(self.watts),
                utilization_pct: Some(self.util),
                ..GpuSample::default()
            }
        }
    }

    #[test]
    fn profile_backend_none_is_all_na() {
        let r = profile_backend(None, std::time::Duration::from_millis(20), 100);
        assert_eq!(r.joules_per_token, None);
        assert_eq!(r.avg_utilization_pct, None);
        assert_eq!(r.vendor, None);
        assert_eq!(r.model, None);
        assert_eq!(r.joules, 0.0);
    }

    #[test]
    fn profile_backend_collects_and_computes() {
        let be = MockBackend {
            vendor: "NVIDIA",
            model: Some("RTX 4090"),
            watts: 285.0,
            util: 94,
        };
        let r = profile_backend(
            Some(&be as &dyn GpuBackend),
            std::time::Duration::from_millis(300),
            1000,
        );
        assert!(r.joules > 0.0, "energy integrated: {}", r.joules);
        assert!(r.joules_per_token.is_some());
        assert_eq!(r.avg_utilization_pct, Some(94.0));
        assert_eq!(r.vendor.as_deref(), Some("NVIDIA"));
        assert_eq!(r.model.as_deref(), Some("RTX 4090"));
        assert!((r.peak_power_w.unwrap() - 285.0).abs() < 1.0);
    }

    /// A CPU-only sample (no GPU power) still contributes to the energy
    /// integral — the total *system* draw (GPU + CPU) is what `∫P dt` uses.
    #[test]
    fn integrate_joules_includes_cpu_only_samples() {
        let a = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let b = MonotonicInstant::now();
        let samples = vec![
            HwSample {
                t: a,
                cpu_power_mw: Some(40_000), // 40 W CPU, no GPU
                ..HwSample::default()
            },
            HwSample {
                t: b,
                cpu_power_mw: Some(40_000),
                ..HwSample::default()
            },
        ];
        let (joules, has_power) = integrate_joules(&samples, None);
        assert!(has_power, "a CPU-only sample carries power");
        assert!(joules > 0.0, "CPU-only energy is positive: {joules}");
    }
}
