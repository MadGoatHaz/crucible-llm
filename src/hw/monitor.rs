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
//! * **Phase power** — the prefill window (load start → first token) and
//!   the decode window (first token → last token) are measured
//!   *separately* from the 1 Hz power history: `avg_prefill_power` and
//!   `avg_decode_power`. This is what makes input vs output token costs
//!   meaningful (prefill and decode draw very different power).
//! * **Energy** — `∫P(t)dt` over the 1 Hz power history (trapezoidal rule),
//!   in joules / kWh, and a `$` estimate at a user-set `$/kWh` rate.
//! * **Cost per 1M tokens** — [`calculate_token_costs`] turns the phase
//!   power + phase durations + token counts into **separate** $/1M rates
//!   for *input* (prefill) and *output* (decode) tokens, plus a blended
//!   rate and the run's total cost — directly comparable to cloud API
//!   pricing (GPT-4o / Claude list prices shown beside them in View 4).
//! * **Efficiency** — J/token, J/ktoken, tokens/W.
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

/// The token / phase context for one poll, read from the current metrics
/// snapshot by the [`crate::hw::HwPoller`].
///
/// * `prompt_tokens` / `completion_tokens` are the **cumulative** run
///   totals (the [`crate::metrics::state::OverallStats`] the metrics
///   pipeline folds on every stream completion) — the cost denominators;
/// * `active` — a stream is decoding right now (the last-token clock
///   advances while this is true);
/// * `tokens_seen` — at least one token has arrived this run (latches the
///   prefill→decode boundary the moment the first token lands).
#[derive(Debug, Clone, Copy, Default)]
pub struct TokenPhase {
    /// Cumulative prompt (input) tokens this run.
    pub prompt_tokens: u64,
    /// Cumulative completion (output) tokens this run.
    pub completion_tokens: u64,
    /// A stream is actively decoding right now.
    pub active: bool,
    /// At least one token has been observed this run.
    pub tokens_seen: bool,
}

/// The $/1M-token cost breakdown for a run (the View 4 **COST ANALYSIS**
/// panel's data).
///
/// * **Input** (prefill) and **output** (decode) are *separate* rates —
///   like the cloud providers' own pricing tables (GPT-4o charges
///   differently for prompt vs completion tokens);
/// * **Blended** is the all-in rate over `input + output` tokens;
/// * **Total cost** is what this run's measured energy actually cost at
///   the user's `$/kWh` rate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TokenCosts {
    /// $ per 1M **input** (prompt / prefill) tokens.
    pub cost_per_1m_input: f64,
    /// $ per 1M **output** (completion / decode) tokens.
    pub cost_per_1m_output: f64,
    /// $ per 1M tokens blended (input + output together).
    pub blended: f64,
    /// The run's total electricity cost at the configured rate.
    pub total_cost: f64,
    /// The input tokens the rates are normalized to.
    pub prompt_tokens: u64,
    /// The output tokens the rates are normalized to.
    pub completion_tokens: u64,
}

/// The $/1M-token cost math (pure — unit-testable without any hardware).
///
/// Each phase's energy is its **measured average power** times its
/// **measured duration**:
///
/// ```text
/// prefill_kWh = avg_prefill_power_W × TTFT_s / 3.6e6
/// decode_kWh  = avg_decode_power_W × decode_s / 3.6e6
///
/// $/1M input  = prefill_kWh × rate / (prompt_tokens / 1e6)
/// $/1M output = decode_kWh  × rate / (completion_tokens / 1e6)
/// blended     = (prefill_kWh + decode_kWh) × rate / (total_tokens / 1e6)
/// ```
///
/// A phase with zero tokens yields a `0.0` rate (nothing to normalize);
/// the N/A *display* decision (show `N/A` vs `0.00`) is the UI's, made on
/// the presence of a [`GpuPowerMonitor::token_costs`] result.
#[must_use]
pub fn calculate_token_costs(
    prompt_tokens: u64,
    completion_tokens: u64,
    ttft_secs: f64,
    decode_secs: f64,
    avg_prefill_power_w: f64,
    avg_decode_power_w: f64,
    rate_per_kwh: f64,
) -> TokenCosts {
    let prefill_kwh = avg_prefill_power_w * ttft_secs / 3_600_000.0;
    let decode_kwh = avg_decode_power_w * decode_secs / 3_600_000.0;

    let cost_per_1m_input = if prompt_tokens > 0 {
        (prefill_kwh * rate_per_kwh) / (prompt_tokens as f64 / 1_000_000.0)
    } else {
        0.0
    };
    let cost_per_1m_output = if completion_tokens > 0 {
        (decode_kwh * rate_per_kwh) / (completion_tokens as f64 / 1_000_000.0)
    } else {
        0.0
    };

    let total_tokens = prompt_tokens + completion_tokens;
    let total_kwh = prefill_kwh + decode_kwh;
    let blended = if total_tokens > 0 {
        (total_kwh * rate_per_kwh) / (total_tokens as f64 / 1_000_000.0)
    } else {
        0.0
    };

    TokenCosts {
        cost_per_1m_input,
        cost_per_1m_output,
        blended,
        total_cost: total_kwh * rate_per_kwh,
        prompt_tokens,
        completion_tokens,
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

    // ---- tokens (the efficiency + cost denominators) ----
    /// Total **output** tokens generated this run (cumulative across all
    /// engines) — the J/token and $/1M-output denominators.
    pub total_tokens: u64,
    /// Total **input** (prompt) tokens processed this run (cumulative
    /// across all engines) — the $/1M-input denominator.
    pub prompt_tokens: u64,

    // ---- phase boundaries (the prefill / decode split) ----
    /// When the **first token** was observed this run (the prefill→decode
    /// boundary). Latched by [`Self::record`] the first poll that reports
    /// `TokenPhase::tokens_seen`. `None` until a token arrives.
    pub first_token_at: Option<MonotonicInstant>,
    /// When the **last token activity** was seen (the decode window's end):
    /// updated on every poll that reports `TokenPhase::active`.
    pub last_token_at: Option<MonotonicInstant>,

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
            prompt_tokens: 0,
            first_token_at: None,
            last_token_at: None,
            history: Vec::new(),
            load_started: None,
            load_ended: None,
            run_started: None,
            idle_samples: Vec::new(),
            idle_finalized: false,
            util_sample_counts: Vec::new(),
            temp_sample_counts: Vec::new(),
            rate_per_kwh: 0.16,
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
    /// run statistics, the phase clocks, and the load-window clock) and
    /// start the idle clock. The poller calls this when a benchmark
    /// sequence begins.
    pub fn begin_run(&mut self) {
        self.total_power_w = 0.0;
        self.peak_power_w = 0.0;
        self.idle_power_w = 0.0;
        self.max_temp_c = 0.0;
        self.avg_util_pct = 0.0;
        self.throttle_events = 0;
        self.has_power = false;
        self.total_tokens = 0;
        self.prompt_tokens = 0;
        self.first_token_at = None;
        self.last_token_at = None;
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
    /// stats, the token / phase state, and append a 1 Hz point to the
    /// history.
    ///
    /// `agg` is the backend's aggregated sample (all GPUs combined);
    /// `per_gpu` is one sample per device; `names` the per-device labels;
    /// `phase` the cumulative token totals + the prefill/decode boundary
    /// signals from the current snapshot.
    pub fn record(
        &mut self,
        agg: &GpuSample,
        per_gpu: &[GpuSample],
        names: &[String],
        phase: &TokenPhase,
    ) {
        // Capture the *previous* throttle state before overwriting the
        // per-GPU table (a rising edge is one event, not one per poll).
        let prev_throttled = self.gpus_was_throttled();
        self.prompt_tokens = phase.prompt_tokens;
        self.total_tokens = phase.completion_tokens;
        // The prefill→decode boundary: latch the first-token instant on
        // the first poll that reports any token, and advance the
        // last-token clock while a stream is decoding.
        if self.first_token_at.is_none() && phase.tokens_seen {
            self.first_token_at = Some(MonotonicInstant::now());
        }
        if phase.active {
            self.last_token_at = Some(MonotonicInstant::now());
        }
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

    /// The **average power draw during the prefill phase** (load start →
    /// first token), watts, from the 1 Hz history.
    ///
    /// The window is **half-open** (`[load_start, first_token)`): a sample
    /// sitting exactly on the first-token instant belongs to the decode
    /// phase. A sub-second TTFT rarely contains a whole 1 Hz sample; when
    /// the window is empty the **first** history sample (the one closest
    /// to load start) stands in — the best available estimate of the
    /// prefill draw. `None` when the history is empty (the N/A rule).
    pub fn avg_prefill_power(&self) -> Option<f64> {
        let first = self.first_token_at?;
        let start = self.load_started.unwrap_or(first);
        self.phase_avg_power(start, first, false)
            .or_else(|| self.history.first().map(|s| s.power_w))
    }

    /// The **average power draw during the decode phase** (first token →
    /// last token activity), watts, from the 1 Hz history (the window is
    /// **closed**: a sample on the first-token instant is decode's).
    ///
    /// When the window is empty (a very short decode) the **last** history
    /// sample stands in. `None` when the history is empty (the N/A rule).
    pub fn avg_decode_power(&self) -> Option<f64> {
        let first = self.first_token_at?;
        let last = self.last_token_at.unwrap_or(first);
        self.phase_avg_power(first, last, true)
            .or_else(|| self.history.last().map(|s| s.power_w))
    }

    /// Mean power of the 1 Hz history samples inside `[start, end]`
    /// (`end` included only when `end_inclusive`; monotonic ordering via
    /// the clamped `delta_nanos`, the same test `within()` in
    /// `engines/hardware.rs` uses). `None` when no sample falls in the
    /// window.
    fn phase_avg_power(
        &self,
        start: MonotonicInstant,
        end: MonotonicInstant,
        end_inclusive: bool,
    ) -> Option<f64> {
        let vals: Vec<f64> = self
            .history
            .iter()
            .filter(|s| {
                // `start <= s.t <= end` (clamped deltas are `0` exactly
                // when the receiver is not after the argument)…
                let in_window = s.t.delta_nanos(&start) == 0 && end.delta_nanos(&s.t) == 0;
                // …and a sample exactly on a half-open boundary belongs to
                // the next phase.
                in_window && (end_inclusive || s.t != end)
            })
            .map(|s| s.power_w)
            .collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// The **$/1M-token cost breakdown** for this run: separate *input*
    /// (prefill) and *output* (decode) rates, a blended rate, and the run's
    /// total cost — the data the View 4 **COST ANALYSIS** panel renders
    /// beside cloud reference prices.
    ///
    /// `None` (the N/A rule) when there is no power telemetry, no token
    /// data, no first-token latch, or no power history to measure the
    /// phase draws from.
    #[must_use]
    pub fn token_costs(&self) -> Option<TokenCosts> {
        if !self.has_power {
            return None;
        }
        let first = self.first_token_at?;
        if self.prompt_tokens + self.total_tokens == 0 {
            return None;
        }
        let last = self.last_token_at.unwrap_or(first);
        let load_start = self.load_started.unwrap_or(first);

        let ttft_secs = load_start.delta_nanos(&first) as f64 / 1e9;
        let decode_secs = first.delta_nanos(&last) as f64 / 1e9;

        let (Some(prefill_w), Some(decode_w)) = (self.avg_prefill_power(), self.avg_decode_power())
        else {
            return None;
        };

        Some(calculate_token_costs(
            self.prompt_tokens,
            self.total_tokens,
            ttft_secs,
            decode_secs,
            prefill_w,
            decode_w,
            self.rate_per_kwh,
        ))
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

    /// A [`TokenPhase`] test helper (the common "nothing yet" case).
    fn phase(prompt: u64, completion: u64, active: bool, seen: bool) -> TokenPhase {
        TokenPhase {
            prompt_tokens: prompt,
            completion_tokens: completion,
            active,
            tokens_seen: seen,
        }
    }

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
    }

    #[test]
    fn begin_run_resets_all_accumulators() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.total_power_w = 500.0;
        m.peak_power_w = 600.0;
        m.max_temp_c = 80.0;
        m.total_tokens = 12345;
        m.prompt_tokens = 6789;
        m.first_token_at = Some(MonotonicInstant::now());
        m.last_token_at = Some(MonotonicInstant::now());
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
        assert_eq!(m.prompt_tokens, 0);
        assert!(m.first_token_at.is_none());
        assert!(m.last_token_at.is_none());
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
        m.record(&agg, &[g0a, g1a], &[], &phase(0, 0, false, false));
        let agg = GpuSample::aggregate(&[g0b.clone(), g1b.clone()]);
        m.record(&agg, &[g0b, g1b], &[], &phase(100, 10, false, true));
        // Whole-run means: (90+95)/2, (65+70)/2; peaks: 70, 58.
        assert!((m.avg_util_per_gpu[0] - 92.5).abs() < 1e-9);
        assert!((m.avg_temp_per_gpu[0] - 67.5).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[0], 70);
        assert!((m.avg_util_per_gpu[1] - 82.0).abs() < 1e-9);
        assert!((m.avg_temp_per_gpu[1] - 56.5).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[1], 58);
        // The token totals and the first-token latch carried through.
        assert_eq!(m.prompt_tokens, 100);
        assert_eq!(m.total_tokens, 10);
        assert!(m.first_token_at.is_some());
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
        m.record(
            &agg,
            std::slice::from_ref(&g0),
            &[],
            &phase(0, 0, false, false),
        );
        let g1 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: None, // sensor glitch on this poll
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g1));
        m.record(
            &agg,
            std::slice::from_ref(&g1),
            &[],
            &phase(0, 0, true, true),
        );
        let g2 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(80),
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g2));
        m.record(
            &agg,
            std::slice::from_ref(&g2),
            &[],
            &phase(0, 5, true, true),
        );
        // Temp average over the two real readings: (60+80)/2 = 70.
        assert!((m.avg_temp_per_gpu[0] - 70.0).abs() < 1e-9);
        assert_eq!(m.max_temp_per_gpu[0], 80);
        // Util average over all three: 90.
        assert!((m.avg_util_per_gpu[0] - 90.0).abs() < 1e-9);
        // The last-token clock advanced while active.
        assert!(m.last_token_at.is_some());
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
        m.record(
            &agg,
            &per_gpu,
            &["A4000".into(), "A4000".into()],
            &phase(2000, 100, true, true),
        );
        assert_eq!(m.gpus.len(), 2);
        assert!((m.total_power_w - 650.0).abs() < 1e-9);
        assert!((m.peak_power_w - 650.0).abs() < 1e-9);
        assert!((m.max_temp_c - 70.0).abs() < 1e-9);
        assert_eq!(m.gpu_names, vec!["A4000", "A4000"]);
        assert_eq!(m.total_tokens, 100);
        assert_eq!(m.prompt_tokens, 2000);
        assert!(m.first_token_at.is_some());
        assert!(m.last_token_at.is_some());
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
        m.record(
            &ok,
            std::slice::from_ref(&ok),
            &[],
            &phase(0, 0, false, false),
        );
        assert_eq!(m.throttle_events, 0);
        // Second poll: throttle appears → one rising edge.
        let throttled = GpuSample {
            power_watts: Some(100.0),
            throttle_reasons: Some("thermal".into()),
            ..Default::default()
        };
        m.record(
            &throttled,
            std::slice::from_ref(&throttled),
            &[],
            &phase(0, 0, false, false),
        );
        assert_eq!(m.throttle_events, 1);
        // Third poll: still throttling → no new edge.
        m.record(
            &throttled,
            std::slice::from_ref(&throttled),
            &[],
            &phase(0, 0, false, false),
        );
        assert_eq!(m.throttle_events, 1);
    }

    // ── $/1M token cost math ───────────────────────────────────────────────

    /// The task's worked example, hand-computed:
    /// 2 000 input tokens, 8 000 output, TTFT 2 s @ 300 W prefill,
    /// decode 40 s @ 500 W, $0.16/kWh.
    ///
    /// prefill_kWh = 300 × 2 / 3.6e6 = 1.6667e-4
    /// decode_kWh  = 500 × 40 / 3.6e6 = 5.5556e-3
    /// $/1M in     = 1.6667e-4 × 0.16 / 0.002 = 0.01333
    /// $/1M out    = 5.5556e-3 × 0.16 / 0.008 = 0.11111
    /// blended     = (1.6667e-4 + 5.5556e-3) × 0.16 / 0.01 = 0.09156
    /// total       = 5.7222e-3 × 0.16 = 9.1556e-4
    #[test]
    fn token_costs_match_the_hand_calculation() {
        let c = calculate_token_costs(2_000, 8_000, 2.0, 40.0, 300.0, 500.0, 0.16);
        assert!(
            (c.cost_per_1m_input - 0.013333).abs() < 1e-5,
            "in: {}",
            c.cost_per_1m_input
        );
        assert!(
            (c.cost_per_1m_output - 0.111111).abs() < 1e-5,
            "out: {}",
            c.cost_per_1m_output
        );
        assert!(
            (c.blended - 0.091556).abs() < 1e-5,
            "blended: {}",
            c.blended
        );
        assert!(
            (c.total_cost - 0.00091556).abs() < 1e-8,
            "total: {}",
            c.total_cost
        );
        assert_eq!(c.prompt_tokens, 2_000);
        assert_eq!(c.completion_tokens, 8_000);
    }

    /// Zero tokens in a phase → that phase's rate is 0.0 (nothing to
    /// normalize), the other phase is unaffected.
    #[test]
    fn token_costs_zero_tokens_yield_zero_rate() {
        let c = calculate_token_costs(0, 1000, 1.0, 10.0, 200.0, 400.0, 0.16);
        assert_eq!(c.cost_per_1m_input, 0.0);
        assert!(c.cost_per_1m_output > 0.0);
        assert!(c.blended > 0.0);
        // All-zero run: every rate 0.0, total 0.0.
        let empty = calculate_token_costs(0, 0, 0.0, 0.0, 0.0, 0.0, 0.16);
        assert_eq!(empty.cost_per_1m_input, 0.0);
        assert_eq!(empty.cost_per_1m_output, 0.0);
        assert_eq!(empty.blended, 0.0);
        assert_eq!(empty.total_cost, 0.0);
    }

    /// Input and output are *separate* numbers (the headline requirement):
    /// decode draws more power than prefill, so the output rate exceeds
    /// the input rate.
    #[test]
    fn input_and_output_rates_are_separate() {
        let c = calculate_token_costs(1_000, 9_000, 0.5, 30.0, 150.0, 600.0, 0.16);
        assert!(
            c.cost_per_1m_output > c.cost_per_1m_input,
            "output ({}) > input ({})",
            c.cost_per_1m_output,
            c.cost_per_1m_input
        );
    }

    /// The monitor derives the phase powers from its 1 Hz history:
    /// samples inside [load_start, first_token) average the prefill draw,
    /// samples inside [first_token, last_token] the decode draw.
    #[test]
    fn monitor_phase_powers_split_prefill_and_decode() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let load_start = m.load_started.unwrap();
        // ~1.1 s later: the first token lands (the prefill window holds
        // the low-prefill samples).
        std::thread::sleep(std::time::Duration::from_millis(1100));
        let first = MonotonicInstant::now();
        m.first_token_at = Some(first);
        // History: two prefill samples (100 W) before the first token…
        for w in [100.0, 110.0] {
            m.history.push(PowerSample {
                t: load_start,
                power_w: w,
                ..Default::default()
            });
        }
        // …and three decode samples (300 W) after it.
        for w in [300.0, 310.0, 320.0] {
            m.history.push(PowerSample {
                t: first,
                power_w: w,
                ..Default::default()
            });
        }
        m.last_token_at = Some(first);
        m.has_power = true;
        assert!((m.avg_prefill_power().unwrap() - 105.0).abs() < 1e-9);
        assert!((m.avg_decode_power().unwrap() - 310.0).abs() < 1e-9);
    }

    /// A sub-second phase (no 1 Hz sample inside the window) falls back to
    /// the nearest history sample — never `None` while power exists.
    #[test]
    fn monitor_phase_power_falls_back_to_nearest_sample() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let load_start = m.load_started.unwrap();
        // One sample right at load start, the first token a few ms later
        // (no sample inside the prefill window).
        m.history.push(PowerSample {
            t: load_start,
            power_w: 80.0,
            ..Default::default()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let first = MonotonicInstant::now();
        m.first_token_at = Some(first);
        m.last_token_at = Some(first);
        m.history.push(PowerSample {
            t: first,
            power_w: 250.0,
            ..Default::default()
        });
        m.has_power = true;
        assert!((m.avg_prefill_power().unwrap() - 80.0).abs() < 1e-9);
        assert!((m.avg_decode_power().unwrap() - 250.0).abs() < 1e-9);
    }

    /// `token_costs` is `None` (the N/A rule) without power, without a
    /// first-token latch, without tokens, or without a power history.
    #[test]
    fn token_costs_is_na_without_the_prerequisites() {
        // No power.
        let m = GpuPowerMonitor::default();
        assert_eq!(m.token_costs(), None);
        // Power + history, but no first token.
        let m = GpuPowerMonitor {
            has_power: true,
            history: vec![PowerSample {
                power_w: 100.0,
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(m.token_costs(), None);
        // First token + tokens, but no history.
        let m = GpuPowerMonitor {
            has_power: true,
            first_token_at: Some(MonotonicInstant::now()),
            prompt_tokens: 100,
            ..Default::default()
        };
        assert_eq!(m.token_costs(), None);
    }

    /// The full monitor path: a run with a measured prefill + decode
    /// phase produces the $/1M breakdown the COST ANALYSIS panel shows.
    ///
    /// Physically realistic shape: prefill 0.5 s at ~300 W (2 000 prompt
    /// tokens), decode 1.5 s at ~600 W (8 000 completion tokens) → the
    /// output $/1M rate exceeds the input one.
    #[test]
    fn token_costs_end_to_end_from_the_monitor() {
        let mut m = GpuPowerMonitor::default().with_rate(0.16);
        m.begin_run();
        m.record_idle(50.0);
        m.start_load();
        let load_start = m.load_started.unwrap();

        // Prefill: 0.5 s at ~300 W (two samples at load start).
        for w in [295.0, 305.0] {
            m.history.push(PowerSample {
                t: load_start,
                power_w: w,
                ..Default::default()
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(500));
        let first = MonotonicInstant::now();
        m.first_token_at = Some(first);
        m.has_power = true;

        // Decode: 1.5 s at ~600 W (three samples at the first-token mark).
        for w in [590.0, 600.0, 610.0] {
            m.history.push(PowerSample {
                t: first,
                power_w: w,
                ..Default::default()
            });
        }
        std::thread::sleep(std::time::Duration::from_millis(1500));
        m.last_token_at = Some(MonotonicInstant::now());
        m.prompt_tokens = 2_000;
        m.total_tokens = 8_000;

        let c = m.token_costs().expect("costs derived from a full run");
        // Hand check: prefill ≈ 300 W × 0.5 s = 150 J = 4.1667e-5 kWh;
        // decode ≈ 600 W × 1.5 s = 900 J = 2.5e-4 kWh.
        // $/1M in  = 4.1667e-5 × 0.16 / 0.002 = 0.00333
        // $/1M out = 2.5e-4 × 0.16 / 0.008 = 0.005
        assert!(
            (c.cost_per_1m_input - 0.00333).abs() < 0.0005,
            "in: {}",
            c.cost_per_1m_input
        );
        assert!(
            (c.cost_per_1m_output - 0.005).abs() < 0.001,
            "out: {}",
            c.cost_per_1m_output
        );
        assert!(c.blended > 0.0, "blended rate: {}", c.blended);
        assert!(c.total_cost > 0.0, "total cost: {}", c.total_cost);
        // Decode's longer window at 2× the power → the output rate wins.
        assert!(c.cost_per_1m_output > c.cost_per_1m_input);
    }
}
