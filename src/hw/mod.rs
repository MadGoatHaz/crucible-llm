//! Hardware telemetry (plan Chunk 17, blueprint §4.3 / §5D).
//!
//! [`HwPoller`] samples the host once every 100 ms:
//!
//! * **NVIDIA GPU** — VRAM (used / total), SM clock, core temperature,
//!   instantaneous power (mW), and compute/memory-bus utilization, via
//!   `nvml-wrapper`. This is feature-gated (`nvml`, default-off) and
//!   runtime-detected: when no GPU/driver is present every GPU field
//!   degrades to `None` (rendered `N/A`) — never a panic, so the app runs
//!   normally on driver-less machines (blueprint §5D / Chunk 17
//!   acceptance).
//! * **Cross-platform CPU / RAM** counters via `sysinfo` (not
//!   feature-gated).
//!
//! The poller also accumulates:
//!
//! * a bounded rolling [`HwSample`] trace — the `P(t)` power trace that
//!   the Engine D energy math ([`crate::engines::hardware`]) integrates;
//! * a cumulative `∫P(t)dt` (joules) powering the live J/token gauge.
//!
//! [`HwPoller::tick`] is the metrics-pipeline seam (blueprint §4.2): it
//! merges the latest sample into the current `ArcSwap<MetricsSnapshot>`
//! (load → clone → merge → atomic store). The render loop never writes the
//! snapshot itself and this path never touches the stream workers' quanta
//! timing path (measurement-isolation invariant, blueprint §4).
//!
//! **Weights vs KV-cache:** NVML reports *aggregate* FB memory
//! (used / reserved / total). The split between static model weights and
//! the dynamic KV-cache reservation is server-internal (vLLM /
//! llama.cpp) and not exposed by the driver, so the telemetry carries the
//! aggregate `used` (weights + KV cache) against `total`.

#[cfg(feature = "nvml")]
pub mod nvml;

use crate::metrics::state::{MetricsSnapshot, MetricsState};
use crate::timing::MonotonicInstant;

/// The hardware poll period (blueprint §4.3: "polls system sensors every
/// 100 milliseconds").
pub const HW_POLL_INTERVAL_MS: u64 = 100;

/// Bounded rolling trace capacity: 100 ms period → 3600 samples = 6 min
/// of power trace (plenty for a single benchmark run; the oldest samples
/// drop first).
pub const HW_TRACE_CAPACITY: usize = 3600;

/// One hardware telemetry sample — a `P(t)` trace point (blueprint §5D).
///
/// Every field is `Option`: the N/A rule. A `None` field is rendered
/// `N/A` / zeroed in the UI snapshot, never a panic. GPU fields are
/// `None` when the `nvml` feature is off or no NVIDIA driver is present;
/// CPU/RAM fields come from `sysinfo` on every supported host.
#[derive(Debug, Clone, Default)]
pub struct HwSample {
    /// Sample timestamp (monotonic clock).
    pub t: MonotonicInstant,
    /// VRAM in use (model weights + KV cache), bytes.
    pub vram_used_bytes: Option<u64>,
    /// Total installed VRAM, bytes.
    pub vram_total_bytes: Option<u64>,
    /// Instantaneous GPU power draw, milliwatts.
    pub power_mw: Option<u64>,
    /// GPU (SM) core clock, MHz.
    pub gpu_clock_mhz: Option<u32>,
    /// GPU core temperature, °C.
    pub gpu_temp_c: Option<u32>,
    /// GPU compute utilization, % (0–100).
    pub gpu_util_pct: Option<u32>,
    /// GPU memory-bus utilization, % (0–100).
    pub mem_util_pct: Option<u32>,
    /// Global CPU usage, % (0–100).
    pub cpu_usage_pct: Option<f32>,
    /// RAM in use, bytes.
    pub ram_used_bytes: Option<u64>,
    /// Total RAM, bytes.
    pub ram_total_bytes: Option<u64>,
}

/// The 100 ms hardware telemetry poller (blueprint §4.3 "Hardware
/// Profiler").
///
/// [`HwPoller::new`] probes the NVIDIA driver at startup (feature-gated);
/// when no GPU/driver is present the poller keeps running with CPU/RAM
/// telemetry only and every GPU field reports N/A.
#[derive(Debug)]
pub struct HwPoller {
    #[cfg(feature = "nvml")]
    gpu: Option<nvml::NvmlPoller>,
    sys: sysinfo::System,
    /// The target GPU's marketing name (for `benchmark_sessions.system_gpu`).
    gpu_name: Option<String>,
    /// Cumulative `∫P(t)dt` since poller start, joules (trapezoidal rule).
    energy_joules: f64,
    /// Bounded rolling trace of recent samples (the `P(t)` power trace).
    trace: Vec<HwSample>,
    /// Rolling trace capacity (oldest samples drop first).
    capacity: usize,
}

impl HwPoller {
    /// Probe the hardware once and prime the CPU/RAM counters.
    ///
    /// Never fails: a missing driver (or the absent `nvml` feature) simply
    /// means GPU fields stay `None` (graceful degradation, blueprint
    /// §5D).
    pub fn new() -> Self {
        #[cfg(feature = "nvml")]
        let gpu = nvml::NvmlPoller::init().ok();
        #[cfg(feature = "nvml")]
        let gpu_name = gpu.as_ref().map(|g| g.name().to_string());
        #[cfg(not(feature = "nvml"))]
        let gpu_name = None;
        let mut sys = sysinfo::System::new();
        // Prime the counters: `global_cpu_usage()` is a delta-based
        // reading (inaccurate on the very first call — sysinfo docs).
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        Self {
            #[cfg(feature = "nvml")]
            gpu,
            sys,
            gpu_name,
            energy_joules: 0.0,
            trace: Vec::new(),
            capacity: HW_TRACE_CAPACITY,
        }
    }

    /// Override the rolling trace capacity (defaults to
    /// [`HW_TRACE_CAPACITY`]).
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// `true` when an NVIDIA GPU/driver is available (feature on **and**
    /// NVML initialized).
    pub fn has_gpu(&self) -> bool {
        #[cfg(feature = "nvml")]
        {
            self.gpu.is_some()
        }
        #[cfg(not(feature = "nvml"))]
        {
            false
        }
    }

    /// The target GPU's name (`benchmark_sessions.system_gpu`).
    pub fn gpu_name(&self) -> Option<&str> {
        self.gpu_name.as_deref()
    }

    /// Cumulative `∫P(t)dt` since poller start, joules.
    pub fn energy_joules(&self) -> f64 {
        self.energy_joules
    }

    /// The rolling `P(t)` power trace (bounded, oldest first).
    pub fn trace(&self) -> &[HwSample] {
        &self.trace
    }

    /// Live silicon efficiency for the UI gauge: cumulative joules ÷
    /// total tokens generated so far. `None` when no tokens yet (N/A —
    /// never a spurious `0.0`).
    pub fn live_joules_per_token(&self, total_tokens: u64) -> Option<f64> {
        (total_tokens > 0).then(|| self.energy_joules / total_tokens as f64)
    }

    /// One 100 ms poll: read the GPU (if any) + CPU/RAM, advance the
    /// cumulative energy integral, and append to the bounded trace.
    pub fn poll(&mut self) -> HwSample {
        let mut sample = HwSample {
            t: MonotonicInstant::now(),
            ..HwSample::default()
        };

        #[cfg(feature = "nvml")]
        if let Some(gpu) = &self.gpu {
            let g = gpu.read();
            sample.vram_used_bytes = g.vram_used_bytes;
            sample.vram_total_bytes = g.vram_total_bytes;
            sample.power_mw = g.power_mw;
            sample.gpu_clock_mhz = g.gpu_clock_mhz;
            sample.gpu_temp_c = g.gpu_temp_c;
            sample.gpu_util_pct = g.gpu_util_pct;
            sample.mem_util_pct = g.mem_util_pct;
        }

        // Cross-platform CPU/RAM (sysinfo): a couple of /proc reads —
        // cheap enough for the 100 ms cadence, and it never touches the
        // stream workers' timing path.
        self.sys.refresh_memory();
        self.sys.refresh_cpu_usage();
        sample.cpu_usage_pct = Some(self.sys.global_cpu_usage());
        sample.ram_used_bytes = Some(self.sys.used_memory());
        sample.ram_total_bytes = Some(self.sys.total_memory());

        // Trapezoidal update of ∫P(t)dt (watts × seconds). A missing
        // power reading contributes nothing (N/A rule) — the integral
        // simply spans from the last known reading to the next.
        if let Some(mw) = sample.power_mw {
            let p = mw as f64 / 1000.0;
            let p_prev = self
                .trace
                .last()
                .and_then(|prev| prev.power_mw)
                .map(|m| m as f64 / 1000.0)
                .unwrap_or(p);
            let dt = self
                .trace
                .last()
                .map(|prev| prev.t.delta_nanos(&sample.t) as f64 / 1e9)
                .unwrap_or(0.0);
            self.energy_joules += (p_prev + p) / 2.0 * dt;
        }

        self.trace.push(sample.clone());
        if self.trace.len() > self.capacity {
            self.trace.drain(0..self.trace.len() - self.capacity);
        }
        sample
    }

    /// The metrics-pipeline seam (blueprint §4.2): poll once and merge the
    /// sample into the current `ArcSwap<MetricsSnapshot>` (load → clone →
    /// merge → atomic store).
    ///
    /// Each publish is a complete snapshot (no torn state), and this path
    /// never touches the stream workers' quanta timing path
    /// (measurement-isolation invariant, blueprint §4).
    pub fn tick(&mut self, state: &MetricsState) {
        let sample = self.poll();
        let snap = state.load();
        let merged = merge_hw(
            &snap,
            &sample,
            self.live_joules_per_token(snap.completion_tokens),
        );
        state.update(merged);
    }
}

impl Default for HwPoller {
    fn default() -> Self {
        Self::new()
    }
}

/// Merge a hardware sample into a metrics snapshot — the N/A → `0.0`
/// sentinel mapping the UI gauges expect (blueprint §5D: `0.0` on a
/// hardware field means "no telemetry", rendered `N/A` by the views).
///
/// Pure and side-effect free: `jpt` (the live joules/token) is passed in
/// by the caller so this function stays trivially testable.
pub fn merge_hw(snap: &MetricsSnapshot, sample: &HwSample, jpt: Option<f64>) -> MetricsSnapshot {
    let mut s = snap.clone();
    s.vram_used_gb = sample
        .vram_used_bytes
        .map(|b| b as f64 / 1e9)
        .unwrap_or(0.0);
    s.vram_total_gb = sample
        .vram_total_bytes
        .map(|b| b as f64 / 1e9)
        .unwrap_or(0.0);
    s.power_w = sample.power_mw.map(|m| m as f64 / 1000.0).unwrap_or(0.0);
    s.gpu_clock_mhz = sample.gpu_clock_mhz.map(|m| m as f64).unwrap_or(0.0);
    s.joules_per_token = jpt.unwrap_or(0.0);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_hw_maps_na_to_zero_sentinels() {
        let snap = MetricsSnapshot::default();
        let sample = HwSample::default(); // every field None
        let merged = merge_hw(&snap, &sample, None);
        assert_eq!(merged.vram_used_gb, 0.0);
        assert_eq!(merged.vram_total_gb, 0.0);
        assert_eq!(merged.power_w, 0.0);
        assert_eq!(merged.gpu_clock_mhz, 0.0);
        assert_eq!(merged.joules_per_token, 0.0);
        // Non-hardware fields pass through untouched.
        assert_eq!(merged.endpoint, snap.endpoint);
        assert_eq!(merged.model, snap.model);
    }

    #[test]
    fn merge_hw_populates_from_a_full_sample() {
        let snap = MetricsSnapshot::default();
        let sample = HwSample {
            vram_used_bytes: Some(21_400_000_000),
            vram_total_bytes: Some(24_000_000_000),
            power_mw: Some(285_000),
            gpu_clock_mhz: Some(1410),
            ..HwSample::default()
        };
        let merged = merge_hw(&snap, &sample, Some(0.338));
        assert!((merged.vram_used_gb - 21.4).abs() < 1e-9);
        assert!((merged.vram_total_gb - 24.0).abs() < 1e-9);
        assert!((merged.power_w - 285.0).abs() < 1e-9);
        assert_eq!(merged.gpu_clock_mhz, 1410.0);
        assert_eq!(merged.joules_per_token, 0.338);
    }

    #[test]
    fn live_joules_per_token_is_na_without_tokens() {
        let p = HwPoller::new();
        assert_eq!(p.live_joules_per_token(0), None);
        // With tokens the gauge is a real (possibly 0.0 on a driver-less
        // host) measurement — never the N/A `None`.
        assert!(p.live_joules_per_token(100).is_some());
    }

    #[test]
    fn poll_never_panics_and_degrades_to_na() {
        let mut p = HwPoller::new();
        for _ in 0..3 {
            let s = p.poll();
            // CPU/RAM (sysinfo) are available on every supported host.
            assert!(s.cpu_usage_pct.is_some());
            assert!(s.ram_total_bytes.is_some());
            // GPU fields are N/A without a driver — and a driver-less
            // machine must keep working (Chunk 17 acceptance).
            if !p.has_gpu() {
                assert!(s.power_mw.is_none());
                assert!(s.vram_total_bytes.is_none());
            }
        }
        assert!(!p.trace().is_empty());
    }

    #[test]
    fn trace_is_bounded() {
        let mut p = HwPoller::new().with_capacity(10);
        for _ in 0..15 {
            p.poll();
        }
        assert!(p.trace().len() <= 10);
        assert_eq!(p.trace().len(), 10);
    }
}
