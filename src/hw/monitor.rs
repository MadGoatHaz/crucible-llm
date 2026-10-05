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
//!   rate and the run's total cost at the user's `$/kWh` rate.
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

/// The manual "Measure Idle" window (seconds) — the GPU tab's `[i]` action
/// sits the system idle for this long and records the mean as the new idle
/// baseline. Longer than the automatic [`IDLE_WINDOW_SECS`] pre-test window
/// for a cleaner no-load floor.
pub const MANUAL_IDLE_WINDOW_SECS: u64 = 10;

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
    /// CPU power draw, watts (RAPL / Super I-O / estimate). The total
    /// *system* draw is `power_w + cpu_power_w` — the energy integral and
    /// the `$/1M` cost math use the sum, not the GPU alone.
    pub cpu_power_w: f64,
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
            cpu_power_w: 0.0,
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
    /// Run-average **prefill** (input) throughput, tokens/sec (total input
    /// tokens ÷ total prefill time). `0.0` until a stream completes — the
    /// $/1M-*input* cost's phase rate.
    pub prefill_throughput: f64,
    /// Run-average **decode** (output) throughput, tokens/sec (total output
    /// tokens ÷ total decode time). `0.0` until a stream completes — the
    /// $/1M-*output* cost's phase rate.
    pub decode_throughput: f64,
}

/// The $/1M-token cost breakdown for a run (the View 4 **COST ANALYSIS**
/// panel's data).
///
/// * **Input** (prefill) and **output** (decode) are *separate* rates —
///   prefill and decode draw very different power;
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
/// Each phase's energy is its **measured average power** applied over the
/// **time that phase actually occupied** during the run:
///
/// ```text
/// primary (throughput model — the multi-request-correct one):
///   prefill_J = prompt_tokens   × avg_prefill_power_W / prefill_throughput
///   decode_J  = completion_tokens × avg_decode_power_W / decode_throughput
///
/// fallback (no throughput measured yet — a single-request run):
///   prefill_J = avg_prefill_power_W × TTFT_s
///   decode_J  = avg_decode_power_W  × decode_s
///
/// $/1M input  = prefill_kWh × rate / (prompt_tokens / 1e6)
/// $/1M output = decode_kWh  × rate / (completion_tokens / 1e6)
/// blended     = (prefill_kWh + decode_kWh) × rate / (total_tokens / 1e6)
/// ```
///
/// The throughput model is a **strict generalization** of the old
/// `power × duration` formula: for a single request, `throughput =
/// tokens / duration`, so the two are identical. For a multi-engine run
/// (the case that produced the "$0 input" bug), the cumulative input token
/// total is matched against the *total* prefill time across every request —
/// not just the first request's TTFT — so the input cost is a real,
/// non-zero number.
///
/// The **blended** rate is the token-count-weighted mean of the input and
/// output rates (`(in·N_in + out·N_out) / (N_in + N_out)`), which is always
/// *between* the two — it can never fall below the cheaper phase (the
/// "blended < output" bug is now structurally impossible).
///
/// A phase with zero tokens yields a `0.0` rate (nothing to normalize);
/// the N/A *display* decision (show `N/A` vs `0.00`) is the UI's, made on
/// the presence of a [`GpuPowerMonitor::token_costs`] result.
#[must_use]
// Nine scalar inputs (two token counts, two phase durations, two phase
// powers, two phase throughputs, one rate) — a pure, positional math
// function; grouping them into a struct would obscure the formula.
#[allow(clippy::too_many_arguments)]
pub fn calculate_token_costs(
    prompt_tokens: u64,
    completion_tokens: u64,
    ttft_secs: f64,
    decode_secs: f64,
    avg_prefill_power_w: f64,
    avg_decode_power_w: f64,
    prefill_throughput: f64,
    decode_throughput: f64,
    rate_per_kwh: f64,
) -> TokenCosts {
    // Each phase's energy in joules. Primary: the throughput model (total
    // phase tokens × measured phase power ÷ measured phase throughput =
    // the energy that phase consumed over the whole run). Fallback (no
    // throughput measured yet): phase power × the measured phase window.
    let prefill_joules = if prefill_throughput > 0.0 && prompt_tokens > 0 {
        prompt_tokens as f64 * avg_prefill_power_w / prefill_throughput
    } else {
        avg_prefill_power_w * ttft_secs
    };
    let decode_joules = if decode_throughput > 0.0 && completion_tokens > 0 {
        completion_tokens as f64 * avg_decode_power_w / decode_throughput
    } else {
        avg_decode_power_w * decode_secs
    };

    let prefill_kwh = prefill_joules / 3_600_000.0;
    let decode_kwh = decode_joules / 3_600_000.0;

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
    // The blended rate = the token-count-weighted mean of the input and
    // output rates, so it is always *between* them (never below the cheaper
    // phase) — the "blended < output" bug is structurally impossible now.
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

    // ---- CPU power (the system-draw component the GPU-only math missed) ----
    /// Current CPU power draw, watts (RAPL / Super I-O / estimate).
    pub cpu_power_w: f64,
    /// Peak CPU power draw observed this run, watts.
    pub cpu_power_peak_w: f64,

    // ---- tokens (the efficiency + cost denominators) ----
    /// Total **output** tokens generated this run (cumulative across all
    /// engines) — the J/token and $/1M-output denominators.
    pub total_tokens: u64,
    /// Total **input** (prompt) tokens processed this run (cumulative
    /// across all engines) — the $/1M-input denominator.
    pub prompt_tokens: u64,
    /// Run-average prefill (input) throughput, tokens/sec — the latest
    /// snapshot's figure; `0.0` until a stream completes. The $/1M-input
    /// cost's phase rate (converts the input token total into prefill time).
    pub prefill_throughput: f64,
    /// Run-average decode (output) throughput, tokens/sec — the $/1M-output
    /// cost's phase rate.
    pub decode_throughput: f64,

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
    /// `true` while a *manual* "Measure Idle" window is running (the GPU
    /// tab's `[i]` action): the poller accumulates a fresh no-load baseline
    /// and finalizes it when the window elapses.
    pub measure_idle: bool,
    /// When the manual idle measurement started (the window's origin).
    pub idle_measure_started: Option<MonotonicInstant>,
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
            cpu_power_w: 0.0,
            cpu_power_peak_w: 0.0,
            total_tokens: 0,
            prompt_tokens: 0,
            prefill_throughput: 0.0,
            decode_throughput: 0.0,
            first_token_at: None,
            last_token_at: None,
            history: Vec::new(),
            load_started: None,
            load_ended: None,
            run_started: None,
            idle_samples: Vec::new(),
            idle_finalized: false,
            measure_idle: false,
            idle_measure_started: None,
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

    /// Record the current CPU power draw (watts). Called by the poller on
    /// each tick, *before* [`Self::record`] / [`Self::record_idle`], so the
    /// 1 Hz history point carries the CPU component. The running peak is
    /// tracked too.
    pub fn set_cpu_power(&mut self, watts: f64) {
        self.cpu_power_w = watts.max(0.0);
        self.cpu_power_peak_w = self.cpu_power_peak_w.max(self.cpu_power_w);
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
        self.cpu_power_w = 0.0;
        self.cpu_power_peak_w = 0.0;
        self.total_tokens = 0;
        self.prompt_tokens = 0;
        self.prefill_throughput = 0.0;
        self.decode_throughput = 0.0;
        self.first_token_at = None;
        self.last_token_at = None;
        self.history.clear();
        self.idle_samples.clear();
        self.idle_finalized = false;
        self.measure_idle = false;
        self.idle_measure_started = None;
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
        // Accumulate while the pre-load window is open, *or* during a
        // manual "Measure Idle" window (which runs after a run has already
        // finalized its baseline).
        if self.idle_finalized && !self.measure_idle {
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

    /// Arm a **manual** idle-baseline measurement (the GPU tab's `[i]`
    /// "Measure Idle" action): clear the idle samples and start a fresh
    /// [`MANUAL_IDLE_WINDOW_SECS`] no-load window. Unlike [`Self::begin_run`]
    /// this does *not* reset the run stats — it only re-measures the idle
    /// floor, so the last run's figures stay on screen. The poller
    /// accumulates the window and calls [`Self::finalize_idle_measurement`]
    /// when it elapses.
    pub fn begin_idle_measurement(&mut self) {
        self.measure_idle = true;
        self.idle_samples.clear();
        self.idle_measure_started = Some(MonotonicInstant::now());
    }

    /// Finalize a manual idle measurement: set the idle baseline to the mean
    /// of the accumulated window and clear the measurement flag. The existing
    /// run stats (a completed run's power / energy / cost) are untouched.
    pub fn finalize_idle_measurement(&mut self) {
        if !self.idle_samples.is_empty() {
            self.idle_power_w =
                self.idle_samples.iter().sum::<f64>() / self.idle_samples.len() as f64;
        }
        self.measure_idle = false;
        self.idle_measure_started = None;
    }

    /// Cancel an in-progress manual idle measurement (the `[i]` key pressed
    /// again): clear the flag and the partial samples without touching the
    /// existing idle baseline.
    pub fn cancel_idle_measurement(&mut self) {
        self.measure_idle = false;
        self.idle_measure_started = None;
        self.idle_samples.clear();
    }

    /// `true` while a manual idle measurement is in progress (the GPU tab
    /// shows a "measuring idle…" status line).
    #[must_use]
    pub fn is_measuring_idle(&self) -> bool {
        self.measure_idle
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

        // The run-average phase throughputs (the $/1M cost model's inputs)
        // — carried from the current snapshot each poll.
        self.prefill_throughput = phase.prefill_throughput;
        self.decode_throughput = phase.decode_throughput;

        // The *current* per-GPU table + total power are live "right now"
        // readings — they keep updating even after the run ends.
        self.gpus = per_gpu.to_vec();
        if !names.is_empty() {
            self.gpu_names = names.to_vec();
        }
        if let Some(w) = agg.power_watts {
            self.has_power = true;
            self.total_power_w = w;
        }

        // Once the load window is closed ([`Self::end_load`]), freeze every
        // *run-accumulating* stat: the peak, the per-GPU whole-run averages,
        // the throttle counter, the prefill/decode phase clocks, and the 1 Hz
        // energy history (the source of `energy_joules` → the avg-power and
        // cost figures). This is the "avg power keeps running after the tests
        // end" fix — a completed run's numbers must not drift on screen. The
        // current readings above stay live.
        if self.load_ended.is_some() {
            return;
        }

        // Run-accumulating token / phase state.
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
        // Aggregate power peak + max temperature.
        if let Some(w) = agg.power_watts {
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
            // The CPU component of the system draw (set by `set_cpu_power`
            // before `record` on each tick) — the energy / cost math sums it
            // with the GPU power.
            cpu_power_w: self.cpu_power_w,
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
            (Some(start), Some(end)) => start.delta_nanos(&end) as f64 / 1e9,
            (Some(start), None) => start.elapsed().as_secs_f64(),
            _ => 0.0,
        }
    }

    /// Total energy `∫P(t)dt` over the **test window**, joules.
    ///
    /// Window-based (the "statistics only count what happened during the
    /// test" rule): the integral spans `[load_started, load_ended]`.
    /// While the run is live (`load_ended` is `None`) that is everything
    /// from load start to now; once the run completes, any samples the
    /// still-running 100 ms poller records **after** `load_ended` are
    /// excluded — the energy, and the avg-power / cost / J-token figures
    /// derived from it, are final the moment the window closes.
    pub fn energy_joules(&self) -> f64 {
        match (self.load_started, self.load_ended) {
            (Some(start), Some(end)) => {
                // Single pass, no allocation: trapezoids over consecutive
                // in-window samples (the same math as `integrate_power`,
                // with the window filter applied). A sample outside the
                // window breaks the chain, so no interval ever bridges
                // across an edge.
                let mut joules = 0.0f64;
                let mut prev: Option<&PowerSample> = None;
                for s in &self.history {
                    // `start <= s.t <= end` (clamped-delta ordering, the
                    // same predicate `phase_avg_power` uses).
                    let in_window = s.t.delta_nanos(&start) == 0 && end.delta_nanos(&s.t) == 0;
                    if in_window {
                        if let Some(p) = prev {
                            let dt = p.t.delta_nanos(&s.t) as f64 / 1e9;
                            // Total *system* draw (GPU + CPU) at each end
                            // of the interval.
                            let p0 = p.power_w + p.cpu_power_w;
                            let p1 = s.power_w + s.cpu_power_w;
                            joules += (p0 + p1) / 2.0 * dt;
                        }
                        prev = Some(s);
                    } else {
                        prev = None;
                    }
                }
                joules
            }
            // No closed window (live run, or no load window yet): the
            // whole history is in scope.
            _ => integrate_power(&self.history),
        }
    }

    /// Total energy in kWh (joules / 3.6e6).
    pub fn energy_kwh(&self) -> f64 {
        self.energy_joules() / 3_600_000.0
    }

    /// Estimated cost at the configured `$/kWh` rate.
    pub fn cost_usd(&self) -> f64 {
        self.energy_kwh() * self.rate_per_kwh
    }

    /// Estimated cost at an explicit `$/kWh` rate (the live-config variant).
    pub fn cost_usd_at_rate(&self, rate: f64) -> f64 {
        self.energy_kwh() * rate
    }

    /// Compute power = total draw − idle baseline (the power the *work*
    /// added, not the machine's floor). Clamped non-negative.
    pub fn compute_power_w(&self) -> f64 {
        (self.total_power_w - self.idle_power_w).max(0.0)
    }

    /// Total **system** power draw (all GPUs + the CPU), watts — the figure
    /// the "Total" line in the View 4 system-power panel and the energy /
    /// cost math are built from (GPU alone underreports the real draw).
    pub fn system_power_w(&self) -> f64 {
        self.total_power_w + self.cpu_power_w
    }

    /// Mean power over the test window: `energy / duration` (time-weighted).
    /// Both the energy and the duration are windowed to
    /// `[load_started, load_ended]`, so once the run completes this is
    /// **final** — post-completion samples (the poller keeps running for
    /// the live table) never dilute it. `None` before the window has any
    /// span (the N/A rule).
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
    /// fewer than 2 samples fall in the window the **overall average
    /// power** stands in (see [`Self::prefill_uses_fallback`] for the UI
    /// warning). `None` when the history is empty (the N/A rule).
    pub fn avg_prefill_power(&self) -> Option<f64> {
        let first = self.first_token_at?;
        let start = self.load_started.unwrap_or(first);
        if self.prefill_uses_fallback() {
            // Fewer than 2 samples in the TTFT window: use the overall
            // average power (the best available estimate).
            self.avg_power_w()
                .or_else(|| self.history.first().map(|s| s.power_w))
        } else {
            self.phase_avg_power(start, first, false)
        }
    }

    /// `true` when the prefill window contains fewer than 2 power samples
    /// (a sub-second TTFT at 1 Hz sampling). The UI shows a warning note
    /// when this is set: the prefill power is the overall average, not a
    /// phase-specific measurement.
    pub fn prefill_uses_fallback(&self) -> bool {
        let Some(first) = self.first_token_at else {
            return true; // no first token → no prefill measurement possible
        };
        let start = self.load_started.unwrap_or(first);
        let count = self
            .history
            .iter()
            .filter(|s| {
                let in_window = s.t.delta_nanos(&start) == 0 && first.delta_nanos(&s.t) == 0;
                in_window && s.t != first
            })
            .count();
        count < 2
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
            // Total *system* draw (GPU + CPU) — the phase powers that feed
            // the `$/1M` cost math include the CPU, not just the GPU.
            .map(|s| s.power_w + s.cpu_power_w)
            .collect();
        (!vals.is_empty()).then(|| vals.iter().sum::<f64>() / vals.len() as f64)
    }

    /// The **$/1M-token cost breakdown** for this run: separate *input*
    /// (prefill) and *output* (decode) rates, a blended rate, and the run's
    /// total cost — the data the View 4 **COST ANALYSIS** panel renders.
    ///
    /// `None` (the N/A rule) when there is no power telemetry, no token
    /// data, no first-token latch, or no power history to measure the
    /// phase draws from.
    #[must_use]
    pub fn token_costs(&self) -> Option<TokenCosts> {
        self.token_costs_at_rate(self.rate_per_kwh)
    }

    /// The **$/1M-token cost breakdown** at an explicit `rate` ($/kWh).
    ///
    /// This is the live-rate variant: the UI calls it with the user's
    /// current `$/kWh` from the Config form so a rate change takes effect
    /// immediately (the monitor's own `rate_per_kwh` is set at startup and
    /// may be stale).
    #[must_use]
    pub fn token_costs_at_rate(&self, rate: f64) -> Option<TokenCosts> {
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
            self.prefill_throughput,
            self.decode_throughput,
            rate,
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
        // Total *system* draw (GPU + CPU) at each end of the interval — the
        // energy the `$`/1M cost is computed from (GPU alone underreports).
        let p0 = pair[0].power_w + pair[0].cpu_power_w;
        let p1 = pair[1].power_w + pair[1].cpu_power_w;
        joules += (p0 + p1) / 2.0 * dt;
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
            ..Default::default()
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
        // Throughputs `0.0` → the duration fallback (a single-request run),
        // which is the original `power × duration` model.
        let c = calculate_token_costs(2_000, 8_000, 2.0, 40.0, 300.0, 500.0, 0.0, 0.0, 0.16);
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
        let c = calculate_token_costs(0, 1000, 1.0, 10.0, 200.0, 400.0, 0.0, 0.0, 0.16);
        assert_eq!(c.cost_per_1m_input, 0.0);
        assert!(c.cost_per_1m_output > 0.0);
        assert!(c.blended > 0.0);
        // All-zero run: every rate 0.0, total 0.0.
        let empty = calculate_token_costs(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16);
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
        let c = calculate_token_costs(1_000, 9_000, 0.5, 30.0, 150.0, 600.0, 0.0, 0.0, 0.16);
        assert!(
            c.cost_per_1m_output > c.cost_per_1m_input,
            "output ({}) > input ({})",
            c.cost_per_1m_output,
            c.cost_per_1m_input
        );
    }

    /// The throughput model is a **strict generalization** of the old
    /// `power × duration` formula: for a *single* request, `throughput =
    /// tokens / duration`, so the two give identical costs. This pins the
    /// backward-compatibility guarantee.
    #[test]
    fn token_costs_throughput_model_matches_duration_for_a_single_request() {
        // One request: 2 000 input tokens prefilling in 2 s, 8 000 output
        // tokens decoding over 40 s.
        let dur = calculate_token_costs(2_000, 8_000, 2.0, 40.0, 300.0, 500.0, 0.0, 0.0, 0.16);
        let thr = calculate_token_costs(
            2_000,
            8_000,
            2.0,
            40.0,
            300.0,
            500.0,
            2_000.0 / 2.0,  // = 1 000 tok/s (the single request's prefill rate)
            8_000.0 / 40.0, // = 200 tok/s (its decode rate)
            0.16,
        );
        assert!(
            (dur.cost_per_1m_input - thr.cost_per_1m_input).abs() < 1e-9,
            "input matches: {} vs {}",
            dur.cost_per_1m_input,
            thr.cost_per_1m_input
        );
        assert!(
            (dur.cost_per_1m_output - thr.cost_per_1m_output).abs() < 1e-9,
            "output matches: {} vs {}",
            dur.cost_per_1m_output,
            thr.cost_per_1m_output
        );
        assert!((dur.blended - thr.blended).abs() < 1e-9);
    }

    /// The headline regression (the user's "$0.000 input" bug): a
    /// **multi-request** run with 2 M cumulative input tokens and a measured
    /// prefill throughput must yield a **non-zero** input cost, and the
    /// blended rate must sit **between** input and output. The old
    /// `power × first-request-TTFT` model divided ~0.4 s of prefill energy
    /// by 2 M tokens → $0; the throughput model matches the 2 M tokens
    /// against the *total* prefill time → a real number.
    #[test]
    fn token_costs_throughput_model_fixes_multi_request_input() {
        // 2 M input tokens prefilling at 2 000 tok/s = 1 000 s of prefill;
        // 14 686 output tokens decoding at 50 tok/s = 293.7 s of decode.
        // Prefill power 200 W, decode power 300 W, $0.16/kWh.
        let c = calculate_token_costs(
            2_000_000, 14_686, 0.4,    // first-request TTFT (would give ~$0 in the old model)
            1158.0, // whole-run decode span (would inflate output in the old model)
            200.0, 300.0, 2_000.0, // prefill throughput
            50.0,    // decode throughput
            0.16,
        );
        // Input energy = 2 000 000 × 200 / 2 000 = 200 000 J = 0.0556 kWh.
        // $/1M input = 0.0556 × 0.16 / 2.0 = 0.00444 — non-zero.
        assert!(
            c.cost_per_1m_input > 0.0,
            "input cost must be non-zero: {}",
            c.cost_per_1m_input
        );
        assert!(
            (c.cost_per_1m_input - 0.004444).abs() < 1e-4,
            "input: {}",
            c.cost_per_1m_input
        );
        // Output energy = 14 686 × 300 / 50 = 88 116 J = 0.0245 kWh.
        // $/1M output = 0.0245 × 0.16 / 0.014686 = 0.266.
        assert!(
            (c.cost_per_1m_output - 0.266).abs() < 0.01,
            "output: {}",
            c.cost_per_1m_output
        );
        // Blended is the token-weighted mean → always between the two.
        assert!(
            c.blended >= c.cost_per_1m_input - 1e-9 && c.blended <= c.cost_per_1m_output + 1e-9,
            "blended ({}) between input ({}) and output ({})",
            c.blended,
            c.cost_per_1m_input,
            c.cost_per_1m_output
        );
    }

    /// Property: the blended rate is the token-count-weighted mean of the
    /// input and output rates, so it is **always** between them — never
    /// below the cheaper phase (the "blended < output" bug).
    #[test]
    fn blended_rate_is_always_between_input_and_output() {
        // A spread of (N_in, N_out, prefill_rate, decode_rate) shapes,
        // including the input-dominated and output-dominated extremes.
        for (nin, nout, tin, tout) in [
            (2_000_000u64, 14_686u64, 2_000.0, 50.0),
            (100u64, 900u64, 100.0, 90.0),
            (10_000u64, 10u64, 1_000.0, 10.0),
            (1u64, 1_000_000u64, 1.0, 100.0),
        ] {
            let c = calculate_token_costs(nin, nout, 1.0, 10.0, 200.0, 300.0, tin, tout, 0.16);
            let lo = c.cost_per_1m_input.min(c.cost_per_1m_output);
            let hi = c.cost_per_1m_input.max(c.cost_per_1m_output);
            assert!(
                (c.blended >= lo - 1e-9) && (c.blended <= hi + 1e-9),
                "blended ({}) must be between [{}, {}] for N_in={} N_out={}",
                c.blended,
                lo,
                hi,
                nin,
                nout
            );
        }
    }

    /// The "avg power keeps running after the tests end" fix: once the load
    /// window is closed (`end_load`), `record()` no longer grows the 1 Hz
    /// history, the energy integral, the peak, or the phase clocks — while
    /// the *current* per-GPU table and total power stay live.
    #[test]
    fn record_freezes_run_stats_after_end_load() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        let g0 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            temperature_c: Some(65),
            ..Default::default()
        };
        let agg = GpuSample::aggregate(std::slice::from_ref(&g0));
        m.record(
            &agg,
            std::slice::from_ref(&g0),
            &[],
            &phase(1000, 100, true, true),
        );
        let history_before = m.history.len();
        let peak_before = m.peak_power_w;
        let energy_before = m.energy_joules();
        assert!(history_before > 0, "a history point was recorded");

        // Close the load window, then a *higher* reading arrives after the
        // run ends.
        m.end_load();
        let g_hot = GpuSample {
            power_watts: Some(999.0),
            utilization_pct: Some(99),
            temperature_c: Some(99),
            ..Default::default()
        };
        let agg_hot = GpuSample::aggregate(std::slice::from_ref(&g_hot));
        m.record(
            &agg_hot,
            std::slice::from_ref(&g_hot),
            &[],
            &phase(5000, 500, true, true),
        );

        // The run aggregates are frozen…
        assert_eq!(m.history.len(), history_before, "history is frozen");
        assert_eq!(m.peak_power_w, peak_before, "peak is frozen");
        assert_eq!(m.energy_joules(), energy_before, "energy is frozen");
        assert_eq!(m.prompt_tokens, 1000, "run input-token total is frozen");
        assert_eq!(m.total_tokens, 100, "run output-token total is frozen");
        // …while the *current* readings stay live (the 999 W shows now).
        assert_eq!(m.total_power_w, 999.0, "current power stays live");
        assert_eq!(
            m.gpus[0].power_watts,
            Some(999.0),
            "current per-GPU stays live"
        );
    }

    /// Window-based statistics: once the test window closes
    /// (`end_load`), the energy / avg power / duration are **final** —
    /// samples the still-running 100 ms poller records afterwards are in
    /// the history (the live table keeps working) but contribute nothing
    /// to the run statistics. The "duration + avg power keep running
    /// after the tests end" regression: the poller does not stop, the
    /// *math* does.
    #[test]
    fn stats_are_final_once_the_test_window_closes() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        m.has_power = true;

        // Two in-window samples ~50 ms apart at 300 W (a real gap so the
        // trapezoid integrates to a positive energy).
        let a = PowerSample {
            power_w: 300.0,
            ..Default::default()
        };
        std::thread::sleep(std::time::Duration::from_millis(50));
        let b = PowerSample {
            power_w: 300.0,
            ..Default::default()
        };
        m.history = vec![a, b];

        // Close the window (the run is over)…
        m.end_load();
        let frozen_energy = m.energy_joules();
        let frozen_avg = m.avg_power_w().expect("avg power");
        let frozen_duration = m.duration_sec();
        assert!(
            frozen_energy > 0.0,
            "positive in-window energy: {frozen_energy}"
        );
        assert!(frozen_avg > 0.0, "positive avg power: {frozen_avg}");
        assert!(
            frozen_duration > 0.0,
            "positive duration: {frozen_duration}"
        );

        // …then the *still-running* poller records more samples after
        // `load_ended` (a higher draw, to make any drift obvious).
        std::thread::sleep(std::time::Duration::from_millis(50));
        let c = PowerSample {
            power_w: 999.0,
            ..Default::default()
        };
        m.history.push(c);
        std::thread::sleep(std::time::Duration::from_millis(50));
        let d = PowerSample {
            power_w: 999.0,
            ..Default::default()
        };
        m.history.push(d);

        // Every statistic is locked to the window: the post-end samples
        // are recorded in `history` but change nothing.
        assert_eq!(m.history.len(), 4, "the poller's samples are recorded");
        assert!(
            (m.energy_joules() - frozen_energy).abs() < 1e-9,
            "energy is final: {} vs {frozen_energy}",
            m.energy_joules()
        );
        assert!(
            (m.avg_power_w().unwrap() - frozen_avg).abs() < 1e-9,
            "avg power is final: {} vs {frozen_avg}",
            m.avg_power_w().unwrap()
        );
        assert_eq!(m.duration_sec(), frozen_duration, "duration is final");
    }

    /// The "Measure Idle" action: arming a manual measurement clears the
    /// samples, accumulating during the window, and finalizing sets the idle
    /// baseline to the mean — without touching the run stats.
    #[test]
    fn manual_idle_measurement_arms_accumulates_and_finalizes() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        assert!((m.idle_power_w - 100.0).abs() < 1e-9, "prior baseline");
        // Arm a manual re-measurement after the run.
        m.end_load();
        m.begin_idle_measurement();
        assert!(m.is_measuring_idle());
        m.record_idle(150.0);
        m.record_idle(170.0);
        // Finalize: the baseline is the new mean (160).
        m.finalize_idle_measurement();
        assert!(!m.is_measuring_idle());
        assert!((m.idle_power_w - 160.0).abs() < 1e-9, "mean of 150/170");
    }

    /// Cancelling a manual measurement leaves the existing baseline intact.
    #[test]
    fn manual_idle_measurement_cancel_preserves_baseline() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.record_idle(100.0);
        m.start_load();
        m.end_load();
        let before = m.idle_power_w;
        m.begin_idle_measurement();
        m.record_idle(999.0); // a partial sample
        m.cancel_idle_measurement();
        assert!(!m.is_measuring_idle());
        assert!((m.idle_power_w - before).abs() < 1e-9, "baseline unchanged");
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

    /// A sub-second prefill phase (fewer than 2 samples inside the window)
    /// falls back to the overall average power — never `None` while power
    /// exists.
    #[test]
    fn monitor_prefill_falls_back_to_overall_average() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let load_start = m.load_started.unwrap();
        // One sample right at load start, the first token a few ms later
        // (only 1 sample inside the prefill window → fallback triggered).
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
        // Freeze the duration so `avg_power_w` is stable across calls.
        m.end_load();
        // The fallback is detected…
        assert!(m.prefill_uses_fallback(), "fallback detected");
        // …and the value equals the overall average (not the single 80 W).
        let prefill = m.avg_prefill_power().unwrap();
        let overall = m.avg_power_w().unwrap();
        assert!((prefill - overall).abs() < 1e-9);
        // The prefill value is *not* just the 80 W sample — it's the
        // time-weighted average of both samples (~165 W).
        assert!(
            prefill > 100.0,
            "overall avg > single 80W sample: {prefill}"
        );
        // Decode still uses its own window (the 250 W sample).
        assert!((m.avg_decode_power().unwrap() - 250.0).abs() < 1e-9);
    }

    /// Two or more samples in the prefill window → the real phase average
    /// is used and no fallback is indicated.
    #[test]
    fn monitor_prefill_uses_phase_average_with_enough_samples() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.start_load();
        let load_start = m.load_started.unwrap();
        // Two prefill samples (100 W, 200 W) before the first token.
        m.history.push(PowerSample {
            t: load_start,
            power_w: 100.0,
            ..Default::default()
        });
        m.history.push(PowerSample {
            t: load_start,
            power_w: 200.0,
            ..Default::default()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let first = MonotonicInstant::now();
        m.first_token_at = Some(first);
        m.last_token_at = Some(first);
        m.has_power = true;
        assert!(!m.prefill_uses_fallback(), "no fallback with 2 samples");
        let prefill = m.avg_prefill_power().unwrap();
        assert!((prefill - 150.0).abs() < 1e-9, "mean of 100+200");
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

    // ── CPU power integration (the system-draw component) ────────────────

    /// `set_cpu_power` tracks the current + peak CPU draw (a lower later
    /// reading does not lower the peak).
    #[test]
    fn set_cpu_power_tracks_current_and_peak() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.set_cpu_power(50.0);
        m.set_cpu_power(73.0);
        m.set_cpu_power(20.0);
        assert!(
            (m.cpu_power_w - 20.0).abs() < 1e-9,
            "current is the last read"
        );
        assert!(
            (m.cpu_power_peak_w - 73.0).abs() < 1e-9,
            "peak is the max seen"
        );
    }

    /// `system_power_w` is the GPU + CPU sum (the "Total" the panel shows).
    #[test]
    fn system_power_is_gpu_plus_cpu() {
        let mut m = GpuPowerMonitor {
            total_power_w: 2847.0,
            ..Default::default()
        };
        m.set_cpu_power(73.0);
        assert!((m.system_power_w() - 2920.0).abs() < 1e-9);
    }

    /// The energy integral uses the *total system draw* (GPU + CPU per
    /// sample): the same GPU power with a nonzero CPU draw integrates to
    /// more energy.
    #[test]
    fn energy_joules_includes_the_cpu_draw() {
        let mut m = GpuPowerMonitor::default().with_rate(0.16);
        m.begin_run();
        m.start_load();
        let load_start = m.load_started.unwrap();
        m.set_cpu_power(40.0);
        m.history.push(PowerSample {
            t: load_start,
            power_w: 300.0,
            cpu_power_w: 40.0,
            ..Default::default()
        });
        std::thread::sleep(std::time::Duration::from_millis(50));
        let now = MonotonicInstant::now();
        m.history.push(PowerSample {
            t: now,
            power_w: 300.0,
            cpu_power_w: 40.0,
            ..Default::default()
        });
        let with_cpu = m.energy_joules();
        // The identical samples with no CPU draw.
        let gpu_only = GpuPowerMonitor {
            history: vec![
                PowerSample {
                    t: load_start,
                    power_w: 300.0,
                    ..Default::default()
                },
                PowerSample {
                    t: now,
                    power_w: 300.0,
                    ..Default::default()
                },
            ],
            ..Default::default()
        }
        .energy_joules();
        assert!(
            with_cpu > gpu_only,
            "system energy ({with_cpu}) > GPU-only ({gpu_only})"
        );
    }

    /// `begin_run` resets the CPU power fields.
    #[test]
    fn begin_run_resets_cpu_power() {
        let mut m = GpuPowerMonitor::default();
        m.begin_run();
        m.set_cpu_power(73.0);
        m.begin_run();
        assert_eq!(m.cpu_power_w, 0.0);
        assert_eq!(m.cpu_power_peak_w, 0.0);
    }
}
