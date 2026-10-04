//! NVML GPU backend (blueprint §5D): NVIDIA VRAM, SM clock, temperature,
//! power, and utilization via `nvml-wrapper`.
//!
//! Feature-gated (`nvml`, **on by default**). [`NvmlBackend::try_init`]
//! probes the NVIDIA driver (`libnvidia-ml`) at runtime; when no
//! GPU/driver is present it returns `None` and the caller
//! ([`crate::hw::detect_gpu`]) falls through to the AMD / Intel
//! backends — never a panic, so the app runs normally on driver-less
//! machines (Chunk 17 acceptance).
//!
//! [`NvmlBackend`] implements the vendor-agnostic [`GpuBackend`] trait:
//! its `poll` is the 100 ms hot path (a single driver call per visible
//! device — no subprocess) and normalizes NVML's raw units into a
//! [`GpuSample`].
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
use nvml_wrapper::Nvml;

use super::GpuBackend;
use super::GpuSample;

/// The NVML-backed GPU backend (NVIDIA).
///
/// Owns the initialized `Nvml` handle (its `Drop` impl calls
/// `nvmlShutdown`); device handles are re-derived on each read (a cheap
/// `nvmlDeviceGetHandleByIndex_v2` call — no self-referential storage).
#[derive(Debug)]
pub struct NvmlBackend {
    nvml: Nvml,
    count: u32,
    /// The target (first) GPU's marketing name.
    name: String,
}

impl NvmlBackend {
    /// Initialize NVML and enumerate the visible GPUs.
    ///
    /// `None` (the graceful-degradation path) when the driver is not
    /// loaded, NVML refuses to initialize, or the driver reports zero
    /// devices — never a panic.
    pub fn try_init() -> Option<Box<Self>> {
        let nvml = Nvml::init().ok()?;
        let count = nvml.device_count().ok()?;
        if count == 0 {
            return None;
        }
        let name = nvml
            .device_by_index(0)
            .ok()?
            .name()
            .unwrap_or_else(|_| "NVIDIA GPU".to_string());
        Some(Box::new(Self { nvml, count, name }))
    }

    /// The target (first) GPU's marketing name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Number of visible NVIDIA GPUs.
    pub fn device_count(&self) -> u32 {
        self.count
    }
}

impl GpuBackend for NvmlBackend {
    fn vendor(&self) -> &str {
        "NVIDIA"
    }

    fn model(&self) -> Option<&str> {
        Some(&self.name)
    }

    fn poll(&self) -> GpuSample {
        let mut vram_used = 0u64;
        let mut vram_total = 0u64;
        let mut power_w = 0.0f64;
        let mut power_some = false;
        let mut core_clock_mhz = None;
        let mut memory_clock_mhz = None;
        let mut gpu_temp_c = None;
        let mut gpu_util_pct = None;

        for i in 0..self.count {
            let Ok(device) = self.nvml.device_by_index(i) else {
                continue; // per-device degradation: skip a lost device
            };
            if let Ok(mi) = device.memory_info() {
                vram_used += mi.used;
                vram_total += mi.total;
            }
            if let Ok(mw) = device.power_usage() {
                power_w += mw as f64 / 1000.0;
                power_some = true;
            }
            if let Ok(mhz) = device.clock_info(Clock::Graphics) {
                core_clock_mhz = Some(core_clock_mhz.unwrap_or(0).max(mhz));
            }
            if let Ok(mhz) = device.clock_info(Clock::Memory) {
                memory_clock_mhz = Some(memory_clock_mhz.unwrap_or(0).max(mhz));
            }
            if let Ok(c) = device.temperature(TemperatureSensor::Gpu) {
                gpu_temp_c = Some(gpu_temp_c.unwrap_or(0).max(c));
            }
            if let Ok(u) = device.utilization_rates() {
                gpu_util_pct = Some(gpu_util_pct.unwrap_or(0).max(u.gpu));
            }
        }

        GpuSample {
            power_watts: power_some.then_some(power_w),
            utilization_pct: gpu_util_pct.map(|u| u.min(100) as u8),
            memory_used_mb: (vram_total > 0).then_some(vram_used / (1024 * 1024)),
            memory_total_mb: (vram_total > 0).then_some(vram_total / (1024 * 1024)),
            core_clock_mhz,
            memory_clock_mhz,
            temperature_c: gpu_temp_c.map(|c| c as i32),
            throttle_reasons: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_init` must return `None` — never panic — on a driver-less
    /// host (Chunk 17 acceptance: graceful degradation).
    #[test]
    fn try_init_degrades_gracefully_without_a_driver() {
        let result = NvmlBackend::try_init();
        match result {
            Some(b) => {
                // A GPU is present: a poll must not panic and must report
                // at least the VRAM total.
                let s = b.poll();
                assert!(s.memory_total_mb.is_some());
                assert!(!b.name().is_empty());
                assert_eq!(b.vendor(), "NVIDIA");
            }
            None => {
                // No driver: the `None` is the graceful path.
            }
        }
    }
}
