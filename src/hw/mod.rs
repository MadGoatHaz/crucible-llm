//! Hardware telemetry (plan Chunk 17, blueprint §4.3 / §5D).
//!
//! [`HwPoller`] samples the host once every 100 ms:
//!
//! * **GPU (auto-detected)** — VRAM (used / total), core clock,
//!   temperature, instantaneous power, and utilization, from whichever
//!   vendor backend [`detect_gpu`] finds: NVIDIA (NVML — the default
//!   feature, tried first), AMD (sysfs/hwmon), or Intel (Level Zero →
//!   sysfs fallback). Runtime-detected: when no GPU/driver is present
//!   every GPU field degrades to `None` (rendered `N/A`) — never a panic,
//!   so the app runs normally on driver-less machines (blueprint §5D /
//!   Chunk 17 acceptance).
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

/// The GPU & Power monitor (View 4): the dedicated multi-GPU power / energy /
/// cost / efficiency panel. Pure data + math (no NVML, no locks).
pub mod monitor;

pub use monitor::{GpuPowerMonitor, PowerSample, TokenPhase, HISTORY_CAPACITY, IDLE_WINDOW_SECS};

#[cfg(target_os = "linux")]
mod amd;
#[cfg(target_os = "linux")]
mod intel;
#[cfg(target_os = "linux")]
mod intel_level_zero;

#[cfg(target_os = "linux")]
pub use amd::AmdSysfsBackend;
#[cfg(target_os = "linux")]
pub use intel::{detect_intel_backend, IntelSysfsBackend};
#[cfg(target_os = "linux")]
pub use intel_level_zero::IntelLevelZeroBackend;

use std::sync::Arc;

use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamStatus};
use crate::timing::MonotonicInstant;

/// A unified, vendor-agnostic GPU telemetry sample (blueprint §5D).
///
/// Every field is `Option` (the N/A rule): a metric the driver cannot
/// provide is `None`, never a panic. Backends normalize raw sensor units
/// (µW → W, m°C → °C, Hz → MHz, bytes → MB) before populating these, so
/// downstream consumers receive consistent values regardless of vendor.
#[derive(Debug, Clone, Default)]
pub struct GpuSample {
    /// Instantaneous power draw, watts.
    pub power_watts: Option<f64>,
    /// GPU compute engine utilization, % (0–100).
    pub utilization_pct: Option<u8>,
    /// VRAM in use (weights + KV cache), MB.
    pub memory_used_mb: Option<u64>,
    /// Total installed VRAM, MB.
    pub memory_total_mb: Option<u64>,
    /// Graphics core clock, MHz.
    pub core_clock_mhz: Option<u32>,
    /// Memory clock, MHz.
    pub memory_clock_mhz: Option<u32>,
    /// Core / hotspot temperature, °C.
    pub temperature_c: Option<i32>,
    /// Human-readable throttle reasons (e.g. "thermal", "power"), if any.
    pub throttle_reasons: Option<String>,
}

impl GpuSample {
    /// Combine a set of per-device samples into one **aggregate** sample:
    /// power and VRAM are **summed**, core clock / memory clock / temperature
    /// / utilization take the **max**, and throttle reasons are **joined**
    /// (deduped, comma-separated; `None` when no device is throttling).
    ///
    /// This is the multi-GPU roll-up the 100 ms poller feeds the energy
    /// math and the View 4 aggregate panels. An empty slice yields an all-
    /// `None` sample (the N/A rule).
    #[must_use]
    pub fn aggregate(samples: &[GpuSample]) -> GpuSample {
        if samples.is_empty() {
            return GpuSample::default();
        }
        let power_sum: f64 = samples.iter().filter_map(|s| s.power_watts).sum();
        let power_watts = (power_sum > 0.0).then_some(power_sum);
        let utilization_pct = samples
            .iter()
            .filter_map(|s| s.utilization_pct)
            .max()
            .map(|u| u.min(100));
        let memory_used_mb = {
            let sum: u64 = samples
                .iter()
                .filter_map(|s| s.memory_used_mb)
                .fold(0u64, u64::saturating_add);
            (sum > 0).then_some(sum)
        };
        let memory_total_mb = {
            let sum: u64 = samples
                .iter()
                .filter_map(|s| s.memory_total_mb)
                .fold(0u64, u64::saturating_add);
            (sum > 0).then_some(sum)
        };
        let core_clock_mhz = samples.iter().filter_map(|s| s.core_clock_mhz).max();
        let memory_clock_mhz = samples.iter().filter_map(|s| s.memory_clock_mhz).max();
        let temperature_c = samples.iter().filter_map(|s| s.temperature_c).max();
        // Join the distinct, non-trivial throttle reasons (a device reporting
        // `None`/`"None"` is not throttling).
        let mut reasons: Vec<String> = Vec::new();
        for s in samples {
            if let Some(r) = s
                .throttle_reasons
                .as_deref()
                .map(str::trim)
                .filter(|r| !r.is_empty() && *r != "None")
            {
                if !reasons.iter().any(|x| x == r) {
                    reasons.push(r.to_string());
                }
            }
        }
        let throttle_reasons = (!reasons.is_empty()).then(|| reasons.join(", "));
        GpuSample {
            power_watts,
            utilization_pct,
            memory_used_mb,
            memory_total_mb,
            core_clock_mhz,
            memory_clock_mhz,
            temperature_c,
            throttle_reasons,
        }
    }
}

/// A vendor-agnostic GPU telemetry backend.
///
/// Implementations read from the platform's native telemetry surface
/// (NVML, AMD sysfs/hwmon, Intel sysfs/hwmon) and normalize into a
/// [`GpuSample`]. [`GpuBackend::poll`] is the 100 ms hot path: it must be
/// cheap (direct file reads / a single driver call — no subprocess
/// spawning) and must never panic; a missing node degrades that field to
/// `None` (the N/A rule).
pub trait GpuBackend: Send + Sync + std::fmt::Debug {
    /// Vendor name (e.g. `"AMD"`, `"Intel"`, `"NVIDIA"`).
    fn vendor(&self) -> &str;
    /// The GPU's model / driver label, if known.
    fn model(&self) -> Option<&str>;
    /// One telemetry sample (all fields optional, N/A rule).
    fn poll(&self) -> GpuSample;

    /// One sample **per device** (multi-GPU). The default returns a single
    /// element wrapping [`GpuBackend::poll`] — single-GPU backends (AMD /
    /// Intel sysfs) need no override. Multi-GPU backends (NVML) override
    /// this to expose each device so View 4 can render the per-GPU table.
    fn poll_all(&self) -> Vec<GpuSample> {
        vec![self.poll()]
    }

    /// A display name **per device** (parallel to [`Self::poll_all`]). The
    /// default repeats the single [`GpuBackend::model`] name; multi-GPU
    /// backends override to label each card.
    fn device_names(&self) -> Vec<String> {
        vec![self
            .model()
            .map(str::to_string)
            .unwrap_or_else(|| self.vendor().to_string())]
    }
}

/// Auto-detect the best available GPU backend (NVIDIA → AMD → Intel).
///
/// NVIDIA (NVML) is tried **first** — it is the most common LLM-serving
/// accelerator and the richest telemetry surface, and it is the default
/// feature. When no NVIDIA driver is present (or the `nvml` feature is
/// off) it falls through to the AMD sysfs backend, then the Intel chain
/// (Level Zero first, sysfs fallback).
///
/// Returns `None` when no GPU telemetry is available at all (the
/// graceful-degradation path, never a panic) — the caller runs with
/// CPU/RAM telemetry only and every GPU field reports N/A.
pub fn detect_gpu() -> Option<Box<dyn GpuBackend>> {
    #[cfg(feature = "nvml")]
    if let Some(be) = nvml::NvmlBackend::try_init() {
        return Some(be);
    }
    #[cfg(target_os = "linux")]
    if let Some(be) = amd::AmdSysfsBackend::try_init() {
        return Some(be);
    }
    #[cfg(target_os = "linux")]
    if let Some(be) = intel::detect_intel_backend() {
        return Some(be);
    }
    None
}

/// The vendor + model display label, de-duplicating when the model name
/// already carries the vendor prefix (NVIDIA and Intel marketing names do —
/// e.g. `"NVIDIA GeForce RTX 3080 Ti"` stays as-is, not
/// `"NVIDIA NVIDIA …"`). `None` model → the vendor alone.
pub fn gpu_display_name(g: &dyn GpuBackend) -> String {
    match g.model() {
        Some(m)
            if m.to_ascii_lowercase()
                .starts_with(&g.vendor().to_ascii_lowercase()) =>
        {
            m.to_string()
        }
        Some(m) => format!("{} {}", g.vendor(), m),
        None => g.vendor().to_string(),
    }
}

/// Convert a vendor-agnostic [`GpuSample`] into the GPU fields of an
/// [`HwSample`] (the `P(t)` trace point the energy math integrates).
///
/// Unit conversions: MB → bytes (VRAM), watts → milliwatts (power),
/// °C clamped non-negative, % carried through. The CPU/RAM fields stay
/// `None` (the poller fills them from `sysinfo`); `t` is the sample
/// timestamp.
pub fn hw_sample_from_gpu(t: MonotonicInstant, g: &GpuSample) -> HwSample {
    HwSample {
        t,
        vram_used_bytes: g.memory_used_mb.map(|mb| mb.saturating_mul(1024 * 1024)),
        vram_total_bytes: g.memory_total_mb.map(|mb| mb.saturating_mul(1024 * 1024)),
        power_mw: g.power_watts.map(|w| (w * 1000.0).round().max(0.0) as u64),
        gpu_clock_mhz: g.core_clock_mhz,
        gpu_temp_c: g.temperature_c.map(|c| c.max(0) as u32),
        gpu_util_pct: g.utilization_pct.map(|u| u as u32),
        ..HwSample::default()
    }
}

/// Shared, infallible sysfs read helpers for the AMD and Intel backends.
///
/// Every read returns `Option` — a missing node (ENOENT) or an
/// unparseable value is `None`, never a panic (the N/A rule). Linux-only:
/// the `/sys` tree does not exist on other platforms.
#[cfg(target_os = "linux")]
pub(crate) mod sysfs {
    use std::path::{Path, PathBuf};

    /// Read a file and return its trimmed contents, or `None` on any I/O
    /// failure (missing file, permission, etc.).
    pub(crate) fn read_trimmed(path: &Path) -> Option<String> {
        std::fs::read_to_string(path)
            .ok()
            .map(|s| s.trim().to_string())
    }

    /// Read a file as a `u8` (0–255), or `None`.
    pub(crate) fn read_u8(path: &Path) -> Option<u8> {
        read_trimmed(path)?.parse().ok()
    }

    /// Read a file as a `u32`, or `None`.
    pub(crate) fn read_u32(path: &Path) -> Option<u32> {
        read_trimmed(path)?.parse().ok()
    }

    /// Read a file as a `u64`, or `None`.
    pub(crate) fn read_u64(path: &Path) -> Option<u64> {
        read_trimmed(path)?.parse().ok()
    }

    /// Read a file as an `i32` (hwmon temperatures are signed), or `None`.
    pub(crate) fn read_i32(path: &Path) -> Option<i32> {
        read_trimmed(path)?.parse().ok()
    }

    /// Locate the first `hwmon` subdirectory under a DRM `device` dir
    /// (e.g. `/sys/class/drm/card0/device/hwmon/hwmon0`). `None` when the
    /// hwmon dir is absent (integrated GPUs / older drivers).
    pub(crate) fn find_hwmon(device_path: &Path) -> Option<PathBuf> {
        let hwmon_dir = device_path.join("hwmon");
        std::fs::read_dir(&hwmon_dir).ok().and_then(|rd| {
            rd.filter_map(|e| e.ok())
                .map(|e| e.path())
                .find(|p| p.is_dir())
        })
    }
}

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
/// [`HwPoller::new`] auto-detects a GPU backend at startup (NVIDIA → AMD
/// → Intel, via [`detect_gpu`]); when no GPU/driver is present the
/// poller keeps running with CPU/RAM telemetry only and every GPU field
/// reports N/A. [`HwPoller::with_backend`] injects an already-detected
/// backend (the entry point detects once and shares the result with the
/// App and Engine D).
pub struct HwPoller {
    /// The detected GPU backend (any vendor), or `None` (no GPU).
    gpu: Option<Arc<dyn GpuBackend>>,
    sys: sysinfo::System,
    /// The target GPU's marketing name (for `benchmark_sessions.system_gpu`).
    gpu_name: Option<String>,
    /// Cumulative `∫P(t)dt` since poller start, joules (trapezoidal rule).
    energy_joules: f64,
    /// Bounded rolling trace of recent samples (the `P(t)` power trace).
    trace: Vec<HwSample>,
    /// Rolling trace capacity (oldest samples drop first).
    capacity: usize,
    /// The GPU & Power monitor (View 4): per-GPU samples (plus whole-run
    /// per-device stats), aggregate power, the idle baseline, the 1 Hz
    /// power history, and the energy/cost/efficiency math. Written by
    /// [`Self::tick`], read into the snapshot.
    monitor: GpuPowerMonitor,
    /// Per-device display names (parallel to the monitor's per-GPU table),
    /// refreshed from the backend on each poll.
    gpu_names: Vec<String>,
}

impl std::fmt::Debug for HwPoller {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HwPoller")
            .field("gpu_name", &self.gpu_name)
            .field("energy_joules", &self.energy_joules)
            .field("trace_len", &self.trace.len())
            .field("gpu_count", &self.gpu_names.len())
            .finish()
    }
}

impl HwPoller {
    /// Auto-detect a GPU backend and prime the CPU/RAM counters.
    ///
    /// Never fails: no GPU/driver simply means the poller runs with
    /// CPU/RAM telemetry only and every GPU field reports N/A (graceful
    /// degradation, blueprint §5D).
    pub fn new() -> Self {
        Self::with_backend(detect_gpu().map(Arc::from))
    }

    /// Build a poller around an already-detected backend (`None` = no
    /// GPU) and prime the CPU/RAM counters. The entry point uses this to
    /// share the single detection result with the App and Engine D.
    pub fn with_backend(gpu: Option<Arc<dyn GpuBackend>>) -> Self {
        let gpu_name = gpu.as_ref().map(|g| gpu_display_name(g.as_ref()));
        let mut sys = sysinfo::System::new();
        // Prime the counters: `global_cpu_usage()` is a delta-based
        // reading (inaccurate on the very first call — sysinfo docs).
        sys.refresh_memory();
        sys.refresh_cpu_usage();
        Self {
            gpu,
            sys,
            gpu_name,
            energy_joules: 0.0,
            trace: Vec::new(),
            capacity: HW_TRACE_CAPACITY,
            monitor: GpuPowerMonitor::default(),
            gpu_names: Vec::new(),
        }
    }

    /// Set the `$/kWh` electricity rate the View 4 cost estimate uses
    /// (builder-style; the entry point wires it from the resolved config).
    #[must_use]
    pub fn with_rate(mut self, rate: f64) -> Self {
        self.monitor = self.monitor.with_rate(rate);
        self
    }

    /// Override the rolling trace capacity (defaults to
    /// [`HW_TRACE_CAPACITY`]).
    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(1);
        self
    }

    /// Arm a **new** benchmark run on the monitor: reset every accumulator
    /// and open the idle clock. The sequence executor calls this before the
    /// pre-load idle window.
    pub fn begin_run(&mut self) {
        self.monitor.begin_run();
    }

    /// Open the **load** window on the monitor (finalize the idle baseline
    /// and start recording the 1 Hz power history). The sequence executor
    /// calls this once the idle window elapses and the benchmark load begins.
    pub fn start_load(&mut self) {
        self.monitor.start_load();
    }

    /// The GPU & Power monitor (View 4) state.
    pub fn monitor(&self) -> &GpuPowerMonitor {
        &self.monitor
    }

    /// Close the monitor's load window (the run is over): the duration
    /// freezes at its final value. Idempotent — the sequence calls it at
    /// both freeze sites (the last engine's `Complete` and the `AllComplete`
    /// safety net), and it is a no-op before a load window has opened.
    pub fn end_load(&mut self) {
        self.monitor.end_load();
    }

    /// `true` when a GPU backend is available (any vendor).
    pub fn has_gpu(&self) -> bool {
        self.gpu.is_some()
    }

    /// The detected GPU backend, if any (for Engine D / the App to read
    /// vendor + model, or to poll directly).
    pub fn backend(&self) -> Option<&dyn GpuBackend> {
        self.gpu.as_deref()
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

    /// One 100 ms poll: read the GPU backend (if any) + CPU/RAM, advance
    /// the cumulative energy integral, and append to the bounded trace.
    pub fn poll(&mut self) -> HwSample {
        let gpu = self.gpu.as_ref().map(|g| g.poll());
        self.advance(&gpu)
    }

    /// The shared sample-advance logic: fold the (optional) GPU sample
    /// into a fresh [`HwSample`], add the CPU/RAM counters, advance the
    /// trapezoidal `∫P(t)dt` integral, and append to the bounded trace.
    fn advance(&mut self, gpu: &Option<GpuSample>) -> HwSample {
        let mut sample = HwSample {
            t: MonotonicInstant::now(),
            ..HwSample::default()
        };
        if let Some(g) = gpu {
            let g_sample = hw_sample_from_gpu(sample.t, g);
            sample.vram_used_bytes = g_sample.vram_used_bytes;
            sample.vram_total_bytes = g_sample.vram_total_bytes;
            sample.power_mw = g_sample.power_mw;
            sample.gpu_clock_mhz = g_sample.gpu_clock_mhz;
            sample.gpu_temp_c = g_sample.gpu_temp_c;
            sample.gpu_util_pct = g_sample.gpu_util_pct;
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
        // One sample per device (multi-GPU), then the aggregate roll-up the
        // energy math and the View 4 aggregate panels consume.
        let per_gpu = self.gpu.as_ref().map(|g| g.poll_all()).unwrap_or_default();
        if let Some(g) = &self.gpu {
            self.gpu_names = g.device_names();
        }
        let agg = GpuSample::aggregate(&per_gpu);

        // Read the current snapshot once (lock-free) for both the token
        // count and the merge base.
        let snap = state.load();
        // Feed the GPU & Power monitor (View 4): before the load window
        // opens, accumulate the idle baseline; once open, record the
        // per-GPU table (including the whole-run per-device stats) + the
        // 1 Hz power history + efficiency + the $/1M-token phase split.
        // The phase context carries the cumulative prompt/completion
        // totals (the cost denominators), whether a stream is decoding
        // right now (advances the last-token clock), and whether any token
        // has arrived (latches the prefill→decode boundary).
        let phase = TokenPhase {
            prompt_tokens: snap.overall.total_prompt_tokens,
            completion_tokens: snap.overall.total_tokens,
            active: snap.status == StreamStatus::Streaming,
            tokens_seen: snap.completion_tokens > 0 || snap.observed_frames > 0,
        };
        if self.monitor.idle_finalized {
            self.monitor.record(&agg, &per_gpu, &self.gpu_names, &phase);
        } else {
            self.monitor.record_idle(agg.power_watts.unwrap_or(0.0));
        }

        // The existing 100 ms `P(t)` trace + cumulative energy integral
        // (Engine D's Silicon Efficiency Metric) over the aggregate.
        let sample = self.advance(&Some(agg.clone()));
        let merged = merge_hw(
            &snap,
            &sample,
            self.live_joules_per_token(snap.completion_tokens),
            Some(&agg),
            Some(&self.monitor),
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
pub fn merge_hw(
    snap: &MetricsSnapshot,
    sample: &HwSample,
    jpt: Option<f64>,
    gpu: Option<&GpuSample>,
    monitor: Option<&GpuPowerMonitor>,
) -> MetricsSnapshot {
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
    // The full vendor-agnostic GPU sample (temperature, utilization,
    // throttle, …) the Live view's GPU panel renders. `None` (no GPU) →
    // the panel is hidden entirely (never an N/A box).
    s.gpu = gpu.cloned();
    // The GPU & Power monitor (View 4): per-GPU table, aggregate power,
    // idle baseline, 1 Hz history, energy/cost/efficiency. `None` (no
    // GPU) → View 4 shows its "no telemetry" placeholder.
    s.gpu_monitor = monitor.cloned();
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_hw_maps_na_to_zero_sentinels() {
        let snap = MetricsSnapshot::default();
        let sample = HwSample::default(); // every field None
        let merged = merge_hw(&snap, &sample, None, None, None);
        assert_eq!(merged.vram_used_gb, 0.0);
        assert_eq!(merged.vram_total_gb, 0.0);
        assert_eq!(merged.power_w, 0.0);
        assert_eq!(merged.gpu_clock_mhz, 0.0);
        assert_eq!(merged.joules_per_token, 0.0);
        assert!(merged.gpu.is_none()); // no GPU → the panel is hidden
        assert!(merged.gpu_monitor.is_none()); // no monitor → View 4 placeholder
                                               // Non-hardware fields pass through untouched.
        assert_eq!(merged.endpoint, snap.endpoint);
        assert_eq!(merged.model, snap.model);
    }

    #[test]
    fn merge_hw_carries_the_gpu_monitor() {
        let snap = MetricsSnapshot::default();
        let sample = HwSample::default();
        let mon = GpuPowerMonitor::default().with_rate(0.25);
        let merged = merge_hw(&snap, &sample, None, None, Some(&mon));
        assert!(merged.gpu_monitor.is_some());
        assert!((merged.gpu_monitor.as_ref().unwrap().rate_per_kwh - 0.25).abs() < 1e-9);
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
        let gpu = GpuSample {
            power_watts: Some(285.0),
            utilization_pct: Some(94),
            temperature_c: Some(68),
            ..GpuSample::default()
        };
        let merged = merge_hw(&snap, &sample, Some(0.338), Some(&gpu), None);
        assert!((merged.vram_used_gb - 21.4).abs() < 1e-9);
        assert!((merged.vram_total_gb - 24.0).abs() < 1e-9);
        assert!((merged.power_w - 285.0).abs() < 1e-9);
        assert_eq!(merged.gpu_clock_mhz, 1410.0);
        assert_eq!(merged.joules_per_token, 0.338);
        assert!(merged.gpu.is_some()); // the full sample is carried for the panel
    }

    /// `hw_sample_from_gpu` normalizes vendor units (MB→bytes, W→mW,
    /// °C clamped, % carried) into the trace-point shape the energy math
    /// integrates.
    #[test]
    fn hw_sample_from_gpu_normalizes_units() {
        let t = MonotonicInstant::now();
        let g = GpuSample {
            power_watts: Some(285.5),
            utilization_pct: Some(94),
            memory_used_mb: Some(7700),
            memory_total_mb: Some(25000),
            core_clock_mhz: Some(2520),
            temperature_c: Some(68),
            ..GpuSample::default()
        };
        let s = hw_sample_from_gpu(t, &g);
        assert_eq!(s.power_mw, Some(285_500));
        assert_eq!(s.gpu_util_pct, Some(94));
        assert_eq!(s.vram_used_bytes, Some(7700 * 1024 * 1024));
        assert_eq!(s.vram_total_bytes, Some(25000 * 1024 * 1024));
        assert_eq!(s.gpu_clock_mhz, Some(2520));
        assert_eq!(s.gpu_temp_c, Some(68));
        // A negative temperature (a sensor glitch) clamps to 0.
        let s2 = hw_sample_from_gpu(
            t,
            &GpuSample {
                temperature_c: Some(-5),
                ..g
            },
        );
        assert_eq!(s2.gpu_temp_c, Some(0));
    }

    /// `GpuSample::aggregate` rolls up per-device samples: power + VRAM
    /// summed, clocks / temp / util maxed, throttle reasons joined.
    #[test]
    fn aggregate_sums_power_vram_and_maxes_the_rest() {
        let g0 = GpuSample {
            power_watts: Some(300.0),
            utilization_pct: Some(90),
            memory_used_mb: Some(8000),
            memory_total_mb: Some(16000),
            core_clock_mhz: Some(1500),
            memory_clock_mhz: Some(1200),
            temperature_c: Some(65),
            throttle_reasons: None,
        };
        let g1 = GpuSample {
            power_watts: Some(350.0),
            utilization_pct: Some(97),
            memory_used_mb: Some(9000),
            memory_total_mb: Some(16000),
            core_clock_mhz: Some(1450),
            memory_clock_mhz: Some(1215),
            temperature_c: Some(70),
            throttle_reasons: Some("thermal".into()),
        };
        let a = GpuSample::aggregate(&[g0, g1]);
        assert!((a.power_watts.unwrap() - 650.0).abs() < 1e-9);
        assert_eq!(a.utilization_pct, Some(97));
        assert_eq!(a.memory_used_mb, Some(17000));
        assert_eq!(a.memory_total_mb, Some(32000));
        assert_eq!(a.core_clock_mhz, Some(1500));
        assert_eq!(a.memory_clock_mhz, Some(1215));
        assert_eq!(a.temperature_c, Some(70));
        assert_eq!(a.throttle_reasons.as_deref(), Some("thermal"));
    }

    #[test]
    fn aggregate_dedupes_and_joins_throttle_reasons() {
        let a = GpuSample {
            throttle_reasons: Some("thermal".into()),
            ..Default::default()
        };
        let b = GpuSample {
            throttle_reasons: Some("power".into()),
            ..Default::default()
        };
        let c = GpuSample {
            throttle_reasons: Some("thermal".into()), // duplicate
            ..Default::default()
        };
        let d = GpuSample {
            throttle_reasons: Some("None".into()), // not a real throttle
            ..Default::default()
        };
        let agg = GpuSample::aggregate(&[a, b, c, d]);
        assert_eq!(agg.throttle_reasons.as_deref(), Some("thermal, power"));
    }

    #[test]
    fn aggregate_empty_is_all_na() {
        let a = GpuSample::aggregate(&[]);
        assert!(a.power_watts.is_none());
        assert!(a.utilization_pct.is_none());
        assert!(a.throttle_reasons.is_none());
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
