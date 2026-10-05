//! The GPU & Power monitor (View 4): a dedicated, information-dense hardware
//! panel for AI-server operators.
//!
//! This module is **pure data + math** — no NVML, no locks, no timing path —
//! so every number the panel shows is unit-testable with synthetic samples.
//! The 100 ms [`crate::hw::HwPoller`] is the *writer*: it polls the
//! [`crate::hw::GpuBackend`], folds each poll into a [`GpuPowerMonitor`]
//! (this module), and copies the monitor into the lock-free
//! [`crate::metrics::state::MetricsSnapshot`] the render loop reads.
//!
//! What the monitor tracks:
//!
//! * **Per-GPU** — one [`crate::hw::GpuSample`] per device (power, utilization,
//!   temperature, VRAM, clocks, throttle) for the multi-GPU table, plus
//!   *whole-run* statistics per device (running-average utilization,
//!   running-average temperature, peak temperature) so the table shows the
//!   full picture, not just "right now".
//! * **Aggregate power** — all GPUs summed: current, peak, and the *idle
//!   baseline* measured before load (so `compute = total − idle` is
//!   meaningful, not "the whole draw").
//! * **Energy** — `∫P(t)dt` over the 1 Hz power history (trapezoidal rule),
//!   in joules / kWh, and a `$` estimate at a user-set `$/kWh` rate.
//! * **Efficiency** — J/token, J/ktoken, tokens/W, and `$`/million tokens.
//! * **Duration** — the load-window span, *frozen* at its final value once
//!   the run ends ([`Self::end_load`]) — a completed run's duration must
//!   not keep counting on screen.
//!
//! **N/A rule** (blueprint §5D): with no power telemetry every derived
//! metric is `None` (rendered `N/A`), never a spurious `0.0`.

use crate::timing::MonotonicInstant;

use super::GpuSample;

/// The idle-baseline sampling window (seconds) taken **before** load begins.
pub const IDLE_WINDOW_SECS: u64 = 5;

/// The 1 Hz power-history cadence (one aggregate sample per second).
pub const HISTORY_PERIOD_SECS: f64 = 1.0;

/// Bounded power-history capacity: 1800 s = 30 min at 1 Hz (plenty for a
/// single benchmark run; the oldest samples drop first).
pub const HISTORY_CAPACITY: usize = 1800;

/// One 1 Hz **aggregate** power/telemetry sample (all GPUs combined): the
/// point the power-over-time graph plots and the energy integral integrates.
#[derive(Debug, Clone, Copy)]
pub struct PowerSample {
    /// Sample timestamp (monotonic clock).
    pub t: MonotonicInstant,
    /// Aggregate power draw, watts (all GPUs summed).
    pub power_w: f64,
    /// Aggregate (max) compute utilization, % (0–100).
    pub util_pct: f64,
    /// Aggregate (max) temperature, °C.
    pub temp_c: f64,
    /// Aggregate (summed) VRAM in use, GB.
    pub vram_gb: f64,
}

impl Default for PowerSample {
    fn default() -> Self {
        Self {
            t: MonotonicInstant::now(),
            power_w: 0.0,
            util_pct: 0.0,
            temp_c: 0.0,
            vram_gb: 0.0,
        }
    }
}

/// The full GPU & Power monitor state for View 4.
///
/// Plain `Clone` data (no interior mutability, no locks) so a copy travels in
/// the `ArcSwap<MetricsSnapshot>` double buffer and the render loop reads it
/// lock-free (measurement-isolation invariant, blueprint §4). The poller is
/// the only writer.
#[derive(Debug, Clone)]
pub struct GpuPowerMonitor {
    // ---- per-GPU (the multi-GPU table) ----
    /// One sample per device, in device order.
    pub gpus: Vec<GpuSample>,
    /// Per-device display names (parallel to `gpus`); empty until the first
    /// poll that reports names.
    pub gpu_names: Vec<String>,
    /// Running-average utilization per device, % (the whole-run figure the
    /// table's `AvgU` column shows — not just the current reading).
    pub avg_util_per_gpu: Vec<f64>,
    /// Running-average temperature per device, °C (the table's `AvgT`).
    pub avg_temp_per_gpu: Vec<f64>,
    /// Peak temperature seen per device, °C (the table's `MaxT`).
    pub max_temp_per_gpu: Vec<i32>,

    // ---- aggregate power (all GPUs summed) ----
    /// Current total power draw, watts.
    pub total_power_w: f64,
    /// Peak total power draw observed this run, watts.
    pub peak_power_w: f64,
    /// Idle baseline power (mean of the pre-load [`IDLE_WINDOW_SECS`]
    /// window), watts. `0.0` until measured.
    pub idle_power_w: f64,
    /// Peak temperature (max across devices) this run, °C.
    pub max_temp_c: f64,
    /// Mean utilization (over the 1 Hz history), %.
    pub avg_util_pct: f64,
    /// Number of distinct throttle events observed (a device newly reporting
    /// a non-empty throttle reason counts once per transition).
    pub throttle_events: u32,
    /// `true` once any power reading has been seen (the N/A rule's gate).
    pub has_power: bool,

    // ---- tokens (the efficiency denominator) ----
    /// Total tokens generated this run (cumulative across all engines).
    pub total_tokens: u64,

    // ---- 1 Hz power history (the power-over-time graph) ----
    pub history: Vec<PowerSample>,
    /// When the load (post-idle) window began; `None` while idle / before a
    /// run. The history and `duration` are measured from here.
    pub load_started: Option<MonotonicInstant>,
    /// When the load window **closed** (the run ended and the metrics
    /// pipeline froze); `None` while the run is live. Once set,
    /// [`Self::duration_sec`] reports the frozen final span — a completed
    /// run's duration must not keep counting on screen.
    pub load_ended: Option<MonotonicInstant>,
    /// When the (idle) run began — set by [`Self::begin_run`].
    pub run_started: Option<MonotonicInstant>,

    // ---- idle-measurement scratch (writer side) ----
    /// Aggregate power samples collected during the pre-load idle window.
    pub idle_samples: Vec<f64>,
    /// `true` once the idle baseline has been finalized.
    pub idle_finalized: bool,
    /// Per-device poll counts weighting the running averages above
    /// (writer side: `avg = (avg·n + x) / (n+1)`).
    pub util_sample_counts: Vec<u64>,
    pub temp_sample_counts: Vec<u64>,

    // ---- config ----
    /// The electricity rate for cost estimation, $/kWh.
    pub rate_per_kwh: f64,
}

impl Default for GpuPowerMonitor {
    fn default() -> Self {
        Self {
            gpus: Vec::new(),
            gpu_names: Vec::new(),
            avg_util_per_gpu: Vec::new(),
            avg_temp_per_gpu: Vec::new(),
            max_temp_per_gpu: Vec::new(),
            total_power_w: 0.0,
            peak_power_w: 0.0,
            idle_power_w: 0.0,
            max_temp_c: 0.0,
            avg_util_pct: 0.0,
            throttle_events: 0,
            has_power: false,
            total_tokens: 0,
            history: Vec::new(),
            load_started: None,
            load_ended: None,
            run_started: None,
            idle_samples: Vec::new(),
            idle_finalized: false,
            util_sample_counts: Vec::new(),
            temp_sample_counts: Vec::new(),
            rate_per_kwh: 0.15,
        }
    }
}

impl GpuPowerMonitor {
    /// Set the `$/kWh` rate (builder-style).
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.rate_per_kwh = rate.max(0.0);
        self
    }

    /// Arm a **new** run: reset every accumulator (including the per-GPU
    /// run statistics and the load-window clock) and start the idle clock.
    /// The poller calls this when a benchmark sequence begins.
    pub fn begin_run(&mut self) {
        self.total_power_w = 0.0;
        self.peak_power_w = 0.0;
        self.idle_power_w = 0.0;
        self.max_temp_c = 0.0;
        self.avg_util_pct = 0.0;
        self.throttle_events = 0;
        self.has_power = false;
        self.total_tokens = 0;
        self.history.clear();
        self.idle_samples.clear();
        self.idle_finalized = false;
        self.load_started = None;
        self.load_ended = None;
        self.run_started = Some(MonotonicInstant::now());
        self.avg_util_per_gpu.clear();
        self.avg_temp_per_gpu.clear();
        self.max_temp_per_gpu.clear();
        self.util_sample_counts.clear();
        self.temp_sample_counts.clear();
    }

    /// Record one poll **during the idle window** (before load): accumulate
    /// the aggregate power for the baseline. No-op once the idle window is
    /// finalized (the run is no longer idle).
    pub fn record_idle(&mut self, power_w: f64) {
        if self.idle_finalized {
            return;
        }
        if power_w > 0.0 {
            self.has_power = true;
            if self.idle_samples.len() < 1024 {
                self.idle_samples.push(power_w);
            }
        }
        self.total_power_w = power_w;
    }

    /// Finalize the idle baseline (mean of the pre-load samples) and open the
    /// load window. Called by the poller when the benchmark load begins.
    pub fn start_load(&mut self) {
        if !self.idle_finalized {
            self.idle_power_w = if self.idle_samples.is_empty() {
                0.0
            } else {
                self.idle_samples.iter().sum::<f64>() / self.idle_samples.len() as f64
            };
            self.idle_finalized = true;
        }
        self.load_started = Some(MonotonicInstant::now());
    }

    /// Record one poll **during the load window**: update the per-GPU table
    /// (including the whole-run per-device statistics), the aggregate power
    /// stats, and append a 1 Hz point to the history.
    ///
    /// `agg` is the backend's aggregated sample (all GPUs combined);
    /// `per_gpu` is one sample per device; `names` the per-device labels;
    /// `tokens` the cumulative token count (the efficiency denominator).
    pub fn record(
        &mut self,
        agg: &GpuSample,
        per_gpu: &[GpuSample],
        names: &[String],
        tokens: u64,
    ) {
        // Capture the *previous* throttle state before overwriting the
        // per-GPU table (a rising edge is one event, not one per poll).
        let prev_throttled = self.gpus_was_throttled();
        self.total_tokens = tokens;
        // Per-GPU table.
        self.gpus = per_gpu.to_vec();
        if !names.is_empty() {
            self.gpu_names = names.to_vec();
        }
        // Per-GPU whole-run statistics: running means (utilization,
        // temperature) + the peak temperature, one slot per device. A
        // growing device count adds fresh slots; a shrinking one trims the
        // tail (a hot-plug event — rare, but never out of bounds).
        let n = per_gpu.len();
        self.avg_util_per_gpu.resize(n, 0.0);
        self.avg_temp_per_gpu.resize(n, 0.0);
        self.max_temp_per_gpu.resize(n, 0);
        self.util_sample_counts.resize(n, 0);
        self.temp_sample_counts.resize(n, 0);
        for (i, g) in per_gpu.iter().enumerate() {
            if let Some(u) = g.utilization_pct {
                let c = self.util_sample_counts[i];
                self.avg_util_per_gpu[i] =
                    (self.avg_util_per_gpu[i] * c as f64 + u as f64) / (c + 1) as f64;
                self.util_sample_counts[i] = c + 1;
            }
            if let Some(t) = g.temperature_c {
                let c = self.temp_sample_counts[i];
                self.avg_temp_per_gpu[i] =
                    (self.avg_temp_per_gpu[i] * c as f64 + t as f64) / (c + 1) as f64;
                self.max_temp_per_gpu[i] = self.max_temp_per_gpu[i].max(t);
                self.temp_sample_counts[i] = c + 1;
            }
        }
        // Aggregate power.
        if let Some(w) = agg.power_watts {
            self.has_power = true;
            self.total_power_w = w;
            self.peak_power_w = self.peak_power_w.max(w);
        }
        if let Some(t) = agg.temperature_c {
            self.max_temp_c = self.max_temp_c.max(t as f64);
        }
        // Throttle events: a transition from "no throttle" to "throttle" on
        // any device (a rising edge) counts once.
        let now_throttled = per_gpu.iter().any(|g| {
            g.throttle_reasons
                .as_deref()
                .is_some_and(|r| !r.trim().is_empty() && r.trim() != "None")
        });
        if now_throttled && !prev_throttled {
            self.throttle_events += 1;
        }

        // 1 Hz history (only once the load window is open).
        if self.load_started.is_some() {
            let due = self
                .history
                .last()
                .map(|last| {
                    last.t.delta_nanos(&MonotonicInstant::now()) as f64 / 1e9 >= HISTORY_PERIOD_SECS
                })
                .unwrap_or(true);
            if due && self.has_power {
                self.push_history(agg);
            }
        }
    }

    /// `true` when the *previous* per-GPU set (before this poll) was
    /// throttling — used to detect a throttle rising edge.
    fn gpus_was_throttled(&self) -> bool {
        self.gpus.iter().any(|g| {
            g.throttle_reasons
                .as_deref()
                .is_some_and(|r| !r.trim().is_empty() && r.trim() != "None")
        })
    }

    fn push_history(&mut self, agg: &GpuSample) {
        let now = MonotonicInstant::now();
        let sample = PowerSample {
            t: now,
            power_w: agg.power_watts.unwrap_or(0.0),
            util_pct: agg.utilization_pct.map(|u| u as f64).unwrap_or(0.0),
            temp_c: agg.temperature_c.map(|t| t as f64).unwrap_or(0.0),
            vram_gb: agg
                .memory_used_mb
                .map(|mb| mb as f64 / 1024.0)
                .unwrap_or(0.0),
        };
        self.history.push(sample);
        if self.history.len() > HISTORY_CAPACITY {
            self.history.drain(0..self.history.len() - HISTORY_CAPACITY);
        }
        // The mean utilization is recomputed from the history (time-weighted).
        let utils: Vec<f64> = self.history.iter().map(|s| s.util_pct).collect();
        if !utils.is_empty() {
            self.avg_util_pct = utils.iter().sum::<f64>() / utils.len() as f64;
        }
    }

    // ── run-window lifecycle ──────────────────────────────────────────────

    /// Close the load window (the run is over): freeze the duration at its
    /// final value.
    ///
    /// The 100 ms poller stops ticking the moment the metrics pipeline
    /// freezes, so without this close the `load_started`-anchored clock
    /// would keep growing on a completed run's panel — the "duration keeps
    /// counting" bug. **Idempotent**: the second call (the `AllComplete`
    /// safety-net freeze) is a no-op, and it is ignored before a load
    /// window has opened.
    pub fn end_load(&mut self) {
        if self.load_ended.is_none() && self.load_started.is_some() {
            self.load_ended = Some(MonotonicInstant::now());
        }
    }

    // ── derived metrics (pure; the panel reads these) ─────────────────────

    /// The load-window duration in seconds. While the run is live this
    /// grows with the clock; once [`Self::end_load`] has closed the window
    /// it is **frozen** at the final value (`0.0` before load / no clock).
    pub fn duration_sec(&self) -> f64 {
        match (self.load_started, self.load_ended) {
            (Some(start), Some(end)) => end.delta_nanos(&start) as f64 / 1e9,
            (Some(start), None) => start.elapsed().as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Total energy `∫P(t)dt` over the load history, joules.
    pub fn energy_joules(&self) -> f64 {
        integrate_power(&self.history)
    }

    /// Total energy in kWh (joules / 3.6e6).
    pub fn energy_kwh(&self) -> f64 {
        self.energy_joules() / 3_600_000.0
    }

    /// Estimated cost at the configured `$/kWh` rate.
    pub fn cost_usd(&self) -> f64 {
        self.energy_kwh() * self.rate_per_kwh
    }

    /// Compute power = total draw − idle baseline (the power the *work*
    /// added, not the machine's floor). Clamped non-negative.
    pub fn compute_power_w(&self) -> f64 {
        (self.total_power_w - self.idle_power_w).max(0.0)
    }

    /// Mean power over the load window: `energy / duration` (time-weighted).
    /// `None` before the window has any span (the N/A rule).
    pub fn avg_power_w(&self) -> Option<f64> {
        let d = self.duration_sec();
        (d > 0.0 && self.has_power).then(|| self.energy_joules() / d)
    }

    /// Joules per token. `None` without power or without tokens (N/A rule).
    pub fn joules_per_token(&self) -> Option<f64> {
        (self.has_power && self.total_tokens > 0)
            .then(|| self.energy_joules() / self.total_tokens as f64)
    }

    /// Joules per 1000 tokens (more intuitive than J/token).
    pub fn joules_per_ktoken(&self) -> Option<f64> {
        self.joules_per_token().map(|j| j * 1000.0)
    }

    /// Throughput efficiency: tokens per watt of *compute* power.
    /// `None` without power, tokens, or a non-zero compute draw.
    pub fn tokens_per_watt(&self) -> Option<f64> {
        let compute = self.compute_power_w();
        (self.has_power && self.total_tokens > 0 && compute > 0.0)
            .then(|| self.total_tokens as f64 / compute)
    }

    /// Cost per million tokens at the configured rate.
    pub fn cost_per_million_tokens(&self) -> Option<f64> {
        (self.has_power && self.total_tokens > 0)
            .then(|| self.cost_usd() * 1_000_000.0 / self.total_tokens as f64)
    }
}

/// Trapezoidal integration of a 1 Hz power history: `∫P(t)dt` in joules.
///
/// A gap between two samples is bridged by linear interpolation (the
/// standard treatment for missing sensor samples). An empty history or a
/// single point integrates to `0.0`.
#[must_use]
pub fn integrate_power(history: &[PowerSample]) -> f64 {
    let mut joules = 0.0f64;
    for pair in history.windows(2) {
        let dt = pair[0].t.delta_nanos(&pair[1].t) as f64 / 1e9;
        joules += (pair[0].power_w + pair[1].power_w) / 2.0 * dt;
    }
    joules
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrate_power_is_zero_for_empty_or_single() {
        assert_eq!(integrate_power(&[]), 0.0);
        assert_eq!(integrate_power(&[PowerSample::default()]), 0.0);
    }

    #[test]
    fn integrate_power_is_nonnegative_and_scales_with_power() {
        // Two samples ~60 ms apart in time: positive energy, and a higher
        // power over the *same* gap yields more energy.
        let a = PowerSample::default(); // t = now()
        std::thread::sleep(std::time::Duration::from_millis(60));
        let t_b = MonotonicInstant::now();
        let b = PowerSample {
            t: t_b,
            power_w: 100.0,
            ..Default::default()
        };
        let b2 = PowerSample {
            t: t_b,
            power_w: 200.0,
            ..Default::default()
        };
        let j = integrate_power(&[a, b]);
        assert!(j > 0.0, "positive energy over a real gap: {j}");
        assert!(integrate_power(&[a, b2]) > j);
    }

    #[test]
    fn idle_baseline_is_the_mean_of_idle_samples() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.record_idle(120.0);
        m.record_idle(140.0);
        m.start_load();
        assert!((m.idle_power_w - 120.0).abs() < 1e-9, "mean of 100/120/140");
        assert!(m.idle_finalized);
        // Once finalized, further idle polls are ignored.
        m.record_idle(999.0);
        assert!((m.idle_power_w - 120.0).abs() < 1e-9);
    }

    #[test]
    fn compute_power_is_total_minus_idle() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        m.total_power_w = 300.0;
        assert!((m.compute_power_w() - 200.0).abs() < 1e-9);
        // Never negative.
        m.total_power_w = 50.0;
        assert_eq!(m.compute_power_w(), 0.0);
    }

    #[test]
    fn energy_kwh_and_cost_scale_with_rate() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        // Two samples ~50 ms apart at 100 W: a small but positive joule count.
        let a = PowerSample {
            power_w: 100.0,
            ..Default::default()
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        let b = PowerSample {
            power_w: 100.0,
            ..Default::default()
        };
        m.history = vec![a, b];
        let j = m.energy_joules();
        assert!(j > 0.0, "positive energy: {j}");
        let kwh = m.energy_kwh();
        assert!((kwh - j / 3_600_000.0).abs() < 1e-12);
        m.rate_per_kwh = 0.30;
        assert!((m.cost_usd() - kwh * 0.30).abs() < 1e-12);
    }

    #[test]
    fn efficiency_metrics_are_na_without_power_or_tokens() {
        let m = GpuPowerMonitor::default();
        assert_eq!(m.joules_per_token(), None);
        assert_eq!(m.joules_per_ktoken(), None);
        assert_eq!(m.tokens_per_watt(), None);
        assert_eq!(m.cost_per_million_tokens(), None);
    }

    #[test]
    fn efficiency_metrics_compute_with_power_and_tokens() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(10.0);
        m.start_load();
        m.has_power = true;
        m.total_power_w = 110.0; // compute = 100 W
        m.total_tokens = 1000;
        // A real (small) time gap so the trapezoid integrates to a positive
        // energy (two back-to-back reads can land on the same instant).
        let a = PowerSample {
            power_w: 100.0,
            ..Default::default()
        };
        std::thread::sleep(std::time::Duration::from_millis(1));
        let b = PowerSample {
            power_w: 100.0,
            ..Default::default()
        };
        m.history = vec![a, b];
        let jpt = m.joules_per_token().unwrap();
        assert!(jpt > 0.0);
        // J/ktoken is exactly 1000× J/token.
        assert!((m.joules_per_ktoken().unwrap() - jpt * 1000.0).abs() < 1e-9);
        // tokens/W uses the *compute* power (100 W), not the total (110 W).
        let tpw = m.tokens_per_watt().unwrap();
        assert!((tpw - 1000.0 / 100.0).abs() < 1e-9);
        // $/M tokens is positive.
        assert!(m.cost_per_million_tokens().unwrap() >= 0.0);
    }

    #[test]
    fn begin_run_resets_all_accumulators() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.total_power_w = 500.0;
        m.peak_power_w = 600.0;
        m.max_temp_c = 80.0;
        m.total_tokens = 12345;
        m.history.push(PowerSample::default());
        m.avg_util_per_gpu = vec![90.0];
        m.avg_temp_per_gpu = vec![65.0];
        m.max_temp_per_gpu = vec![72];
        m.util_sample_counts = vec![10];
        m.temp_sample_counts = vec![10];
        m.load_started = Some(MonotonicInstant::now());
        m.load_ended = Some(MonotonicInstant::now());
        m.begin_run();
        assert_eq!(m.total_power_w, 0.0);
        assert_eq!(m.peak_power_w, 0.0);
        assert_eq!(m.max_temp_c, 0.0);
        assert_eq!(m.total_tokens, 0);
        assert!(m.history.is_empty());
        assert!(m.avg_util_per_gpu.is_empty());
        assert!(m.avg_temp_per_gpu.is_empty());
        assert!(m.max_temp_per_gpu.is_empty());
        assert!(m.util_sample_counts.is_empty());
        assert!(m.temp_sample_counts.is_empty());
        assert!(m.load_started.is_none());
        assert!(m.load_ended.is_none());
        assert!(!m.idle_finalized);
        assert!(m.run_started.is_some());
    }

    #[test]
    fn per_gpu_running_averages_and_peak() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let g0a = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(65),
            ..Default::default()
        };
        let g0b = GpuSample {
            power_watts: Some(320.0),
            utilization_pct: Some(95),
            temperature_c: Some(70),
            ..Default::default()
        };
        let g1a = GpuSample {
            power_watts: Some(350.0),
            utilization_pct: Some(80),
            temperature_c: Some(55),
            ..Default::default()
        };
        let g1b = GpuSample {
            power_watts: Some(340.0),
            utilization_pct: Some(84),
            temperature_c: Some(58),
            ..Default::default()
        };
        let agg = GpuSample::aggregate(&[g0a.clone(), g1a.clone()]);
        m.record(&agg, &[g0a, g1a], &[], 0);
        let agg = GpuSample::aggregate(&[g0b.clone(), g1b.clone()]);
        m.record(&agg, &[g0b, g1b], &[], 10);
        // Whole-run means: (90+95)/2, (65+70)/2; peaks: 70, 58.
        assert!((m.avg_util_per_gpu[0] - 92.5).abs() < 1e-9);
        assert!((m.avg_temp_per_gpu[0] - 67.5).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[0], 70);
        assert!((m.avg_util_per_gpu[1] - 82.0).abs() < 1e-9);
        assert!((m.avg_temp_per_gpu[1] - 56.5).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[1], 58);
    }

    #[test]
    fn per_gpu_stats_survive_a_missing_reading() {
        // A poll with no temperature must not stall the temperature
        // average: the next real reading folds in cleanly.
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let g0 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(60),
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g0));
        m.record(&agg, std::slice::from_ref(&g0), &[], 0);
        let g1 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: None, // sensor glitch on this poll
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g1));
        m.record(&agg, std::slice::from_ref(&g1), &[], 0);
        let g2 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(80),
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g2));
        m.record(&agg, std::slice::from_ref(&g2), &[], 0);
        // Temp average over the two real readings: (60+80)/2 = 70.
        assert!((m.avg_temp_per_gpu[0] - 70.0).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[0], 80);
        // Util average over all three: 90.
        assert!((m.avg_util_per_gpu[0] - 90.0).abs() < 1e-9);
    }

    #[test]
    fn end_load_freezes_the_duration() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        let before = m.duration_sec();
        std::thread::sleep(std::time::Duration::from_millis(60));
        let live = m.duration_sec();
        assert!(live > before, "the live duration grows: {live} > {before}");
        // Close the window: the duration freezes at its final value…
        m.end_load();
        let frozen = m.duration_sec();
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert_eq!(
            m.duration_sec(),
            frozen,
            "the frozen duration does not grow"
        );
        // …and a second close (the AllComplete safety-net) is a no-op.
        m.end_load();
        assert_eq!(m.duration_sec(), frozen);
        // A fresh run re-arms: the clock grows again from zero.
        m.begin_run();
        assert!(m.duration_sec() == 0.0);
        m.start_load();
        std::thread::sleep(std::time::Duration::from_millis(60));
        assert!(m.duration_sec() > 0.0);
    }

    #[test]
    fn end_load_before_a_load_window_is_a_noop() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.end_load();
        assert!(m.load_ended.is_none());
        assert_eq!(m.duration_sec(), 0.0);
    }

    #[test]
    fn record_updates_per_gpu_and_peak() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let g0 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(65),
            memory_used_mb: Some(8000),
            memory_total_mb: Some(16000),
            ..Default::default()
        };
        let g1 = GpuSample {
            power_watts: Some(350.0),
            utilization_pct: Some(95),
            temperature_c: Some(70),
            ..Default::default()
        };
        let agg = GpuSample {
            power_watts: Some(650.0),
            utilization_pct: Some(95),
            temperature_c: Some(70),
            memory_used_mb: Some(16000),
            memory_total_mb: Some(32000),
            ..Default::default()
        };
        let per_gpu = vec![g0, g1];
        m.record(&agg, &per_gpu, &["A4000".into(), "A4000".into()], 100);
        assert_eq!(m.gpus.len(), 2);
        assert!((m.total_power_w - 650.0).abs() < 1e-9);
        assert!((m.peak_power_w - 650.0).abs() < 1e-9);
        assert!((m.max_temp_c - 70.0).abs() < 1e-9);
        assert_eq!(m.gpu_names, vec!["A4000", "A4000"]);
        assert_eq!(m.total_tokens, 100);
    }

    #[test]
    fn throttle_rising_edge_counts_one_event() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        // First poll: no throttle.
        let ok = GpuSample {
            power_watts: Some(100.0),
            throttle_reasons: None,
            ..Default::default()
        };
        m.record(&ok, std::slice::from_ref(&ok), &[], 0);
        assert_eq!(m.throttle_events, 0);
        // Second poll: throttle appears → one rising edge.
        let throttled = GpuSample {
            power_watts: Some(100.0),
            throttle_reasons: Some("thermal".into()),
            ..Default::default()
        };
        m.record(&throttled, std::slice::from_ref(&throttled), &[], 0);
        assert_eq!(m.throttle_events, 1);
        // Third poll: still throttling → no new edge.
        m.record(&throttled, std::slice::from_ref(&throttled), &[], 0);
        assert_eq!(m.throttle_events, 1);
    }
}
