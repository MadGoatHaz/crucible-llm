//! NVML 100 ms poll worker (plan Chunk 17, blueprint §5D): NVIDIA VRAM,
//! SM clock, temperature, and instantaneous power via `nvml-wrapper`.
//!
//! Feature-gated (`nvml`, default-off). [`NvmlPoller::init`] probes the
//! NVIDIA driver (`libnvidia-ml`) at runtime; when no GPU/driver is
//! present it returns `Err` and the caller ([`crate::hw::HwPoller`])
//! degrades every GPU field to N/A — never a panic, so the app runs
//! normally on driver-less machines (Chunk 17 acceptance).
//!
//! **Weights vs KV-cache:** NVML reports *aggregate* FB memory
//! (used / reserved / total). The split between static model weights and
//! the dynamic KV-cache reservation is server-internal (vLLM /
//! llama.cpp) and not exposed by the driver, so the sample carries the
//! aggregate `used` (weights + KV cache) against `total`.
//!
//! **Multi-GPU:** every visible device is read on each pass and
//! aggregated — VRAM and power are summed, clock / temperature /
//! utilization take the max.
//!
//! **Per-field degradation:** any single reading the driver cannot
//! provide (e.g. `power_usage()` on some vGPU hosts) is `None` for that
//! field only — the rest of the sample is unaffected.

use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::error::NvmlError;
use nvml_wrapper::Nvml;

/// A single NVML read pass over all visible NVIDIA GPUs.
///
/// All fields are `Option` (per-field graceful degradation, blueprint
/// §5D): a reading the driver cannot provide is `None`, never a panic.
#[derive(Debug, Clone, Default)]
pub struct GpuSample {
    /// VRAM in use (weights + KV cache), bytes (summed across devices).
    pub vram_used_bytes: Option<u64>,
    /// Total installed VRAM, bytes.
    pub vram_total_bytes: Option<u64>,
    /// Instantaneous power draw, milliwatts.
    pub power_mw: Option<u64>,
    /// GPU (SM) core clock, MHz (the max across devices).
    pub gpu_clock_mhz: Option<u32>,
    /// GPU core temperature, °C (the max across devices).
    pub gpu_temp_c: Option<u32>,
    /// GPU compute utilization, % (0–100, max across devices).
    pub gpu_util_pct: Option<u32>,
    /// GPU memory-bus utilization, % (0–100, max across devices).
    pub mem_util_pct: Option<u32>,
}

/// The NVML-backed GPU poller.
///
/// Owns the initialized `Nvml` handle (its `Drop` impl calls
/// `nvmlShutdown`); device handles are re-derived on each read (a cheap
/// `nvmlDeviceGetHandleByIndex_v2` call — no self-referential storage).
#[derive(Debug)]
pub struct NvmlPoller {
    nvml: Nvml,
    count: u32,
    /// The target (first) GPU's marketing name.
    name: String,
}

impl NvmlPoller {
    /// Initialize NVML and enumerate the visible GPUs.
    ///
    /// `Err` when the driver is not loaded (`DriverNotLoaded` /
    /// `LibraryNotFound`), when NVML refuses to initialize, or when the
    /// driver reports zero devices (`NotFound`). The caller degrades to
    /// N/A on any error.
    pub fn init() -> Result<Self, NvmlError> {
        let nvml = Nvml::init()?;
        let count = nvml.device_count()?;
        if count == 0 {
            return Err(NvmlError::NotFound);
        }
        let name = nvml
            .device_by_index(0)?
            .name()
            .unwrap_or_else(|_| "NVIDIA GPU".to_string());
        Ok(Self { nvml, count, name })
    }

    /// The target (first) GPU's marketing name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of visible NVIDIA GPUs.
    pub fn device_count(&self) -> u32 {
        self.count
    }

    /// One read pass: aggregate every visible device into a
    /// [`GpuSample`] (per-field graceful degradation — a failed read
    /// yields `None` for that field only).
    pub fn read(&self) -> GpuSample {
        let mut vram_used = 0u64;
        let mut vram_total = 0u64;
        let mut power_mw = 0u64;
        let mut gpu_clock_mhz = None;
        let mut gpu_temp_c = None;
        let mut gpu_util_pct = None;
        let mut mem_util_pct = None;

        for i in 0..self.count {
            let Ok(device) = self.nvml.device_by_index(i) else {
                continue; // per-device degradation: skip a lost device
            };

            if let Ok(mi) = device.memory_info() {
                vram_used += mi.used;
                vram_total += mi.total;
            }
            if let Ok(mw) = device.power_usage() {
                power_mw += mw as u64;
            }
            if let Ok(mhz) = device.clock_info(Clock::Graphics) {
                gpu_clock_mhz = Some(gpu_clock_mhz.unwrap_or(0).max(mhz));
            }
            if let Ok(c) = device.temperature(TemperatureSensor::Gpu) {
                gpu_temp_c = Some(gpu_temp_c.unwrap_or(0).max(c));
            }
            if let Ok(u) = device.utilization_rates() {
                gpu_util_pct = Some(gpu_util_pct.unwrap_or(0).max(u.gpu));
                mem_util_pct = Some(mem_util_pct.unwrap_or(0).max(u.memory));
            }
        }

        GpuSample {
            vram_used_bytes: (vram_total > 0).then_some(vram_used),
            vram_total_bytes: (vram_total > 0).then_some(vram_total),
            power_mw: (power_mw > 0).then_some(power_mw),
            gpu_clock_mhz,
            gpu_temp_c,
            gpu_util_pct,
            mem_util_pct,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `init` must return a `Result` — never panic — on a driver-less
    /// host (Chunk 17 acceptance: graceful degradation).
    #[test]
    fn init_degrades_gracefully_without_a_driver() {
        let result = NvmlPoller::init();
        match result {
            Ok(p) => {
                // A GPU is present: a read pass must not panic and must
                // report at least the VRAM total.
                let s = p.read();
                assert!(s.vram_total_bytes.is_some());
                assert!(!p.name().is_empty());
            }
            Err(_) => {
                // No driver: the Err is the graceful path.
            }
        }
    }
}
