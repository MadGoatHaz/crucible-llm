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
    /// `true` when the CPU component of the system draw is the fixed
    /// estimate (no RAPL / Super-I-O sensor), so the energy total includes a
    /// constant, not a measurement (CF-53). The pure trace math never knows
    /// the CPU power method — the caller sets this from
    /// [`crate::hw::cpu_power::CpuPowerBackend::is_estimated`] so the
    /// `--json` energy object and the methodology block can flag it.
    pub cpu_power_estimated: bool,
}

impl EnergyResult {
    /// Mark the CPU component as the fixed estimate (CF-53). The pure trace
    /// math never knows the CPU power method, so the caller that holds the
    /// [`crate::hw::cpu_power::CpuPowerBackend`] sets this from
    /// [`CpuPowerBackend::is_estimated`] before the result reaches the
    /// `--json` energy object / methodology block.
    #[must_use]
    pub fn with_cpu_power_estimated(mut self, estimated: bool) -> Self {
        self.cpu_power_estimated = estimated;
        self
    }

    /// `true` when **no power telemetry was available** (the module N/A
    /// rule): no sample in scope carried a power reading, so `joules` is
    /// `0.0` and every derived figure (`joules_per_token`, `avg_power_w`,
    /// `peak_power_w`) is `None`.
    ///
    /// This is the distinct "no power telemetry" case — a driverless host,
    /// or a sensor that returned no usable reading (e.g. the RAPL quirk on a
    /// shared host). It is reported **N/A, clearly**, never a spurious
    /// `0.0` J/token ("free inference") and never a silent gap. A profile
    /// that *did* measure power (even an all-CPU-estimate one) is not N/A.
    #[must_use]
    pub fn is_na(&self) -> bool {
        self.avg_power_w.is_none() && self.peak_power_w.is_none()
    }

    /// The N/A reason for this result, when `joules_per_token` is `None`
    /// (the N/A case) — distinguishing the **root causes** so a consumer
    /// never sees a silent gap (B3 / I7):
    ///
    /// * **`"no power telemetry"`** — [`Self::is_na`]: no sample in scope
    ///   carried a reading (a driverless host, or a sensor with no usable
    ///   reading, e.g. the RAPL quirk on a shared host).
    /// * **`"no energy measured"`** — power telemetry *is* present and
    ///   tokens were generated, but the (idle-subtracted) energy is `0.0`
    ///   (the draw never rose above the idle floor) — I7: the `0.0` N/A
    ///   value, labeled, never a silent `0.0` J/token.
    /// * **`"no tokens generated"`** — power telemetry *is* present, but the
    ///   run produced zero tokens (a D-only / no-work run), so there is
    ///   nothing to divide the energy by.
    ///
    /// `None` when a `joules_per_token` value exists (not N/A). All three
    /// surfaces — the TUI summary, the headless line, and the `--json`/DB
    /// `na_reason` — render from this one method so they can never disagree
    /// on *why* a number is N/A.
    #[must_use]
    pub fn na_reason(&self) -> Option<&'static str> {
        if self.joules_per_token.is_some() {
            return None;
        }
        if self.is_na() {
            Some("no power telemetry")
        } else if self.joules <= 0.0 {
            Some("no energy measured")
        } else {
            Some("no tokens generated")
        }
    }
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
/// `idle_w` (F9 / H4) is the total system idle floor (watts) subtracted
/// from every sample before integrating (clamped non-negative), so the
/// result is **load-attributable** energy — the same subtraction the
/// poller's cumulative integral and the View 4 monitor apply. `0.0`
/// leaves the integral unchanged (the pre-F9 / no-idle-baseline case,
/// e.g. a standalone `profile_backend` run with no measured floor).
///
/// Returns `(joules, has_power)`: `has_power` is `false` when no sample
/// in scope carried a reading, in which case `joules` is `0.0` and
/// callers **must** report any derived metric as N/A (module N/A rule).
#[must_use]
pub fn integrate_joules(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
    idle_w: f64,
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
        // Total *system* draw (GPU + CPU), watts, minus the idle floor
        // (clamped non-negative). A sample with no reading at all is
        // skipped — the trapezoid spans from the last known reading to the
        // next (linear interpolation across gaps, the standard treatment
        // for missing sensor samples). A reading *at or below* the floor
        // contributes 0 W to the integral (mirroring the poller's
        // cumulative integral, which carries the post-subtraction value
        // forward and only adds an interval when the current sample is
        // above the floor).
        let raw = s.power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0)
            + s.cpu_power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0);
        if raw <= 0.0 {
            continue;
        }
        has_power = true;
        let p_w = (raw - idle_w).max(0.0);
        if p_w > 0.0 {
            if let Some((t_prev, p_prev)) = prev {
                let dt_s = t_prev.delta_nanos(&s.t) as f64 / 1e9;
                joules += (p_prev + p_w) / 2.0 * dt_s;
            }
        }
        prev = Some((&s.t, p_w));
    }

    (joules, has_power)
}

/// The Silicon Efficiency Metric for a window: `∫P(t)dt / tokens`.
///
/// `None` when no power telemetry is available or `tokens` is zero
/// (N/A rule — never a spurious `0.0`).
///
/// This is the **per-stream** (Engine A, per-iteration window) concept —
/// it integrates the raw trace with no idle floor (`idle_w = 0.0`), distinct
/// from the run-level `energy.joules_per_token` D headline (idle-subtracted
/// full load window ÷ all-engine tokens). The JSON export labels the two
/// (I7c) so a consumer never conflates them.
#[must_use]
pub fn joules_per_token(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
    tokens: u64,
) -> Option<f64> {
    let (joules, has_power) = integrate_joules(samples, window, 0.0);
    if has_power && tokens > 0 {
        Some(joules / tokens as f64)
    } else {
        None
    }
}

/// Full Engine D profile for a window: energy, power stats, peak VRAM,
/// and the fragmentation warning.
///
/// `idle_w` (F9 / H4) is the total system idle floor (watts) subtracted
/// from the energy integral **and** the mean power, so `joules` and
/// `avg_power_w` are on the *same* (load-attributable) basis —
/// `joules ≈ avg_power_w × duration` holds. `0.0` for a profile with no
/// measured idle baseline (the standalone `profile_backend` path).
#[must_use]
pub fn profile(
    samples: &[HwSample],
    window: Option<(MonotonicInstant, MonotonicInstant)>,
    total_tokens: u64,
    idle_w: f64,
) -> EnergyResult {
    let (joules, has_power) = integrate_joules(samples, window, idle_w);

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
        // I7: a `0.0` energy (the draw never rose above the idle floor, or
        // no interval resolved) is the N/A value — `joules_per_token` stays
        // `None` (labeled by `na_reason`), never a spurious `0.0` J/token
        // ("free inference").
        joules_per_token:
            (has_power && total_tokens > 0 && joules > 0.0).then(|| joules / total_tokens as f64),
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
        // The CPU "estimated" flag is set by the caller from the CPU power
        // backend (the pure trace math never knows it) — default `false`.
        cpu_power_estimated: false,
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
    // `idle_w = 0.0`: a standalone backend profile has no measured idle
    // baseline (the TUI / headless paths pass the monitor's floor).
    let mut r = profile(&samples, None, tokens, 0.0);
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
        let (joules, has_power) = integrate_joules(&samples, None, 0.0);
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
        let (full, has_full) = integrate_joules(&samples, None, 0.0);
        assert!(has_full);
        assert!((0.3..0.5).contains(&full), "full={full}");
        // Window [b, c]: b has no power, c is a single point → 0 J but
        // `has_power` stays true (a reading exists in scope).
        let (win, has) = integrate_joules(&samples, Some((b, c)), 0.0);
        assert!(has);
        assert_eq!(win, 0.0);
        // A window that contains only the power-less sample → N/A.
        let (empty, has_empty) = integrate_joules(&samples, Some((b, b)), 0.0);
        assert!(!has_empty);
        assert_eq!(empty, 0.0);
    }

    #[test]
    fn missing_power_samples_yield_na_not_zero_claim() {
        let a = MonotonicInstant::now();
        let samples = vec![hw_sample(a, None, Some(10), Some(20))];
        let (joules, has_power) = integrate_joules(&samples, None, 0.0);
        assert!(!has_power);
        assert_eq!(joules, 0.0);
        assert_eq!(joules_per_token(&samples, None, 100), None);
        assert_eq!(profile(&samples, None, 100, 0.0).joules_per_token, None);
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
        assert_eq!(profile(&samples, None, 0, 0.0).joules_per_token, None);
    }

    /// CF-53: the CPU "estimated" flag defaults to `false` on a profile (the
    /// pure trace math never knows the CPU method) and is settable via the
    /// builder — the caller copies it from
    /// [`crate::hw::cpu_power::CpuPowerBackend::is_estimated`].
    #[test]
    fn cpu_power_estimated_flag_defaults_false_and_is_settable() {
        let a = MonotonicInstant::now();
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(50_000), None, None),
            hw_sample(b, Some(50_000), None, None),
        ];
        // `profile` defaults the flag to `false`.
        let r = profile(&samples, None, 100, 0.0);
        assert!(!r.cpu_power_estimated, "profile defaults the flag to false");
        // The builder sets it (what the caller does from `is_estimated`).
        let r_est = r.with_cpu_power_estimated(true);
        assert!(r_est.cpu_power_estimated, "builder sets the flag true");
    }

    #[test]
    fn profile_reports_power_and_vram_stats() {
        let a = MonotonicInstant::now();
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(200_000), Some(90 * 1000), Some(100 * 1000)),
            hw_sample(b, Some(400_000), Some(95 * 1000), Some(100 * 1000)),
        ];
        let p = profile(&samples, None, 100, 0.0);
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
        let (joules, has_power) = integrate_joules(&samples, None, 0.0);
        assert!(has_power, "a CPU-only sample carries power");
        assert!(joules > 0.0, "CPU-only energy is positive: {joules}");
    }

    /// F9: `avg_power_w` is on the **same idle-subtracted basis** as
    /// `joules` — `joules ≈ avg_power_w × span` holds. Before F9 the mean
    /// power was computed from the raw (non-idle-subtracted) samples while
    /// `joules` (once `apply_load_window` overwrites it) was idle-subtracted,
    /// so the two disagreed.
    #[test]
    fn avg_power_w_is_on_the_idle_subtracted_basis() {
        // 100 W total system draw with a 50 W idle floor → 50 W
        // load-attributable.
        let a = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(200));
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(100_000), None, None),
            hw_sample(b, Some(100_000), None, None),
        ];
        let p = profile(&samples, None, 100, 50.0);
        let span = a.delta_nanos(&b) as f64 / 1e9;
        // The energy is the 50 W above-floor draw over the span.
        assert!(
            (p.joules - 50.0 * span).abs() < 50.0 * span * 0.01,
            "idle-subtracted joules: {}",
            p.joules
        );
        // The mean power is the same 50 W — not the raw 100 W.
        let avg = p.avg_power_w.expect("avg power present");
        assert!((avg - 50.0).abs() < 1.0, "avg is the load draw: {avg}");
        // The identity F9 guarantees: joules ≈ avg × span.
        assert!(
            (p.joules - avg * span).abs() <= p.joules.max(1.0) * 1e-6 + 1e-9,
            "joules ({} ) ≈ avg ({avg}) × span ({span})",
            p.joules
        );
    }

    /// F9 / I7: a floor at or above every sample yields zero
    /// load-attributable energy — `has_power` stays true (telemetry
    /// existed), `joules_per_token` stays `None` (never a spurious
    /// `0.0`), and `na_reason` names the cause ("no energy measured").
    #[test]
    fn floor_at_or_above_all_samples_yields_zero_energy() {
        let a = MonotonicInstant::now();
        std::thread::sleep(std::time::Duration::from_millis(100));
        let b = MonotonicInstant::now();
        let samples = vec![
            hw_sample(a, Some(80_000), None, None),
            hw_sample(b, Some(90_000), None, None),
        ];
        let (joules, has_power) = integrate_joules(&samples, None, 100.0);
        assert!(has_power, "readings exist — this is not the no-telemetry N/A");
        assert_eq!(joules, 0.0, "nothing above the floor → zero load energy");
        let p = profile(&samples, None, 100, 100.0);
        assert_eq!(p.joules, 0.0);
        assert_eq!(
            p.joules_per_token, None,
            "zero energy is the N/A value — never a spurious 0.0 J/token"
        );
        assert_eq!(
            p.na_reason(),
            Some("no energy measured"),
            "the 0.0 N/A carries its labeled cause"
        );
    }

    // ── B3: the N/A reason distinguishes its two root causes ─────────────

    /// A telemetry-bearing profile (power measured) with a `joules_per_token`
    /// of `None` only because the run produced zero tokens — the B3
    /// "no tokens generated" case, distinct from "no power telemetry".
    fn energy_with_power_no_tokens() -> EnergyResult {
        EnergyResult {
            joules: 100.0,
            joules_per_token: None, // 0 tokens → N/A, but power IS present
            avg_power_w: Some(200.0),
            peak_power_w: Some(300.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(90.0),
            vendor: Some("NVIDIA".into()),
            model: Some("Mock GPU".into()),
            cpu_power_estimated: false,
        }
    }

    /// A no-telemetry profile (driverless host): no sample carried a reading.
    fn energy_no_power() -> EnergyResult {
        EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: None,
            peak_power_w: None,
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: None,
            vendor: None,
            model: None,
            cpu_power_estimated: false,
        }
    }

    /// B3: telemetry present but zero tokens → "no tokens generated" (NOT
    /// "no power telemetry" — the old mislabel).
    #[test]
    fn na_reason_no_tokens_when_power_is_present() {
        let e = energy_with_power_no_tokens();
        assert!(!e.is_na(), "power is present → not the no-telemetry N/A");
        assert_eq!(e.joules_per_token, None, "0 tokens → no J/token");
        assert_eq!(
            e.na_reason(),
            Some("no tokens generated"),
            "the reason names the real cause"
        );
    }

    /// B3: no power telemetry at all → "no power telemetry" (the original
    /// N/A case, unchanged).
    #[test]
    fn na_reason_no_power_telemetry_when_driverless() {
        let e = energy_no_power();
        assert!(e.is_na(), "no sample carried a reading → N/A");
        assert_eq!(e.na_reason(), Some("no power telemetry"));
    }

    /// B3: a real J/token → not N/A at all (`na_reason` is `None`).
    #[test]
    fn na_reason_is_none_when_a_value_exists() {
        let mut e = energy_with_power_no_tokens();
        e.joules_per_token = Some(0.4);
        assert_eq!(e.na_reason(), None, "a value is not N/A");
    }

    /// I7: telemetry + tokens present, but the (idle-subtracted) energy is
    /// `0.0` → the third, distinct cause "no energy measured" (the draw
    /// never rose above the idle floor) — the `0.0` N/A value is labeled,
    /// never a silent `0.0` J/token.
    #[test]
    fn na_reason_no_energy_measured_when_the_floor_covers_the_draw() {
        let e = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: Some(0.0),
            peak_power_w: Some(80.0),
            avg_utilization_pct: Some(90.0),
            ..energy_no_power()
        };
        assert!(!e.is_na(), "telemetry is present → not the no-telemetry N/A");
        assert_eq!(
            e.na_reason(),
            Some("no energy measured"),
            "the distinct zero-energy cause"
        );
    }
}
