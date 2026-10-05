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

use std::sync::Mutex;

use nvml_wrapper::enum_wrappers::device::{Clock, TemperatureSensor};
use nvml_wrapper::Nvml;

use super::GpuBackend;
use super::GpuSample;
use super::MAX_GPU_POWER_W;

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
    /// Per-device "power anomaly already warned" latches (the `GpuBackend`
    /// poll is `&self`; a stuck anomalous reading must not spam the console
    /// every 100 ms). Parallel to the device indices.
    warned: Mutex<Vec<bool>>,
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
        Some(Box::new(Self {
            nvml,
            count,
            name,
            warned: Mutex::new(vec![false; count as usize]),
        }))
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

    /// One [`GpuSample`] **per visible device** (the multi-GPU source for
    /// View 4's per-GPU table). Each device is read independently with
    /// per-field degradation (a reading the driver cannot provide is `None`
    /// for that field only); a lost device is skipped entirely.
    fn poll_all(&self) -> Vec<GpuSample> {
        let mut out = Vec::with_capacity(self.count as usize);
        let mut guard = self.warned.lock().unwrap_or_else(|e| e.into_inner());
        guard.resize(self.count as usize, false);
        for i in 0..self.count {
            let Ok(device) = self.nvml.device_by_index(i) else {
                continue; // per-device degradation: skip a lost device
            };
            let (vram_used, vram_total) = device
                .memory_info()
                .map(|mi| (mi.used, mi.total))
                .unwrap_or((0, 0));
            // Power: NVML reports *milliwatts* (mW / 1000 = W) — the units
            // are correct. The TDP drives the plausibility clamp: a reading
            // > 2× the TDP (e.g. the 943 W on a single 200 W A4000) is
            // flagged and capped at 1.5× the TDP.
            //
            // The TDP reference is the **default** power-management limit
            // (the card's factory TDP — not user-settable). We fall back to
            // the *current* limit only if the default query is unsupported.
            // Relying on the current limit alone is the bug this fixes:
            // `nvidia-smi -pl` can raise it (defeating the clamp), and on
            // some hosts the current-limit query fails outright (returns
            // `None`), which disabled the clamp and let a phantom reading
            // (the 974 W peak on this A4000) flow into the `$/1M` cost math
            // ~10× too high. The default limit is a static property of the
            // card, so it is available even when the current one is not.
            let tdp_w = device
                .power_management_limit_default()
                .or_else(|_| device.power_management_limit())
                .ok()
                .map(|mw| mw as f64 / 1000.0);
            let power_watts = device.power_usage().ok().and_then(|mw| {
                clamp_nvml_power(mw as f64 / 1000.0, tdp_w, i, &mut guard[i as usize])
            });
            out.push(GpuSample {
                power_watts,
                utilization_pct: device
                    .utilization_rates()
                    .ok()
                    .map(|u| u.gpu.min(100) as u8),
                memory_used_mb: (vram_total > 0).then_some(vram_used / (1024 * 1024)),
                memory_total_mb: (vram_total > 0).then_some(vram_total / (1024 * 1024)),
                core_clock_mhz: device.clock_info(Clock::Graphics).ok(),
                memory_clock_mhz: device.clock_info(Clock::Memory).ok(),
                temperature_c: device
                    .temperature(TemperatureSensor::Gpu)
                    .ok()
                    .map(|c| c as i32),
                // NVML throttle reasons are not read on this path (kept `None`
                // — the N/A rule); the Intel Level Zero backend does surface
                // them. The aggregate joins whatever a backend reports.
                throttle_reasons: None,
            });
        }
        out
    }

    /// A display name **per device** (parallel to [`Self::poll_all`]).
    fn device_names(&self) -> Vec<String> {
        (0..self.count)
            .map(|i| {
                self.nvml
                    .device_by_index(i)
                    .ok()
                    .and_then(|d| d.name().ok())
                    .unwrap_or_else(|| self.name.clone())
            })
            .collect()
    }

    /// The single **aggregate** sample: every visible device rolled up
    /// (power + VRAM summed, clocks / temp / utilization maxed) — the value
    /// the 100 ms energy math and the Live panel consume.
    fn poll(&self) -> GpuSample {
        GpuSample::aggregate(&self.poll_all())
    }
}

/// Plausibility-clamp an NVML power reading (watts) against the card's
/// TDP (watts) and the global ceiling.
///
/// NVML's `power_usage()`, `power_management_limit_default()` **and**
/// `power_management_limit()` all return *milliwatts*; the caller converts
/// to watts before calling this. The TDP reference is the **default**
/// power-management limit (the factory TDP, not user-settable), falling back
/// to the current limit. A reading **> 2× the TDP** is a unit/scale bug or a
/// phantom spike — the canonical case is the **943 W reported on a single
/// 200 W A4000** (4.7× its TDP, physically impossible). It is flagged with a
/// `[WARN]` (once per anomaly episode, via the `warned` latch) and **capped
/// at 1.5× the TDP** (transient spikes are possible, 3× is not). When no TDP
/// is available (both limit queries unsupported on some hosts) the global
/// [`MAX_GPU_POWER_W`] ceiling stands in.
#[must_use]
fn clamp_nvml_power(
    power_w: f64,
    tdp_w: Option<f64>,
    index: u32,
    warned: &mut bool,
) -> Option<f64> {
    if !power_w.is_finite() {
        return None;
    }
    // TDP-based bound (the 943 W-on-a-300 W A4000 guard).
    if let Some(tdp) = tdp_w {
        if tdp > 0.0 && power_w > 2.0 * tdp {
            if !*warned {
                eprintln!(
                    "[WARN] [HW] GPU {index} reported {power_w:.0} W but TDP is {tdp:.0} W — possible unit error"
                );
                *warned = true;
            }
            return Some(power_w.min(1.5 * tdp));
        }
    }
    // Global ceiling (no consumer / datacenter GPU draws 5 kW).
    if power_w > MAX_GPU_POWER_W {
        if !*warned {
            eprintln!(
                "[WARN] [HW] GPU {index} reported {power_w:.0} W — implausibly high (>{MAX_GPU_POWER_W:.0} W), capping"
            );
            *warned = true;
        }
        return Some(MAX_GPU_POWER_W);
    }
    *warned = false;
    Some(power_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// NVML reports power in *milliwatts*: the `/1000.0` in `poll_all` is
    /// the mW → W conversion.
    #[test]
    fn nvml_power_unit_is_milliwatts() {
        let mw = 300_000u32; // a 300 W A4000
        assert!((mw as f64 / 1000.0 - 300.0).abs() < 1e-9);
        assert!((19_500_f64 / 1000.0 - 19.5).abs() < 1e-9);
    }

    /// The 943 W-on-a-300 W A4000 bug: 943 > 2×300 = 600 → flagged and
    /// capped at 1.5×300 = 450 W. A stuck reading does not re-warn (the
    /// latch holds until the reading normalizes).
    #[test]
    fn nvml_clamp_flags_and_caps_a_3x_tdp_reading() {
        let mut warned = false;
        let out = clamp_nvml_power(943.0, Some(300.0), 0, &mut warned);
        assert_eq!(out, Some(450.0), "capped at 1.5× the TDP");
        assert!(warned, "the anomaly was flagged");
        // A second identical reading: the latch is already set (no new warn
        // would fire), the value is still capped.
        let out2 = clamp_nvml_power(943.0, Some(300.0), 0, &mut warned);
        assert_eq!(out2, Some(450.0));
    }

    /// The A4000 phantom from `work/testing/latest.log` (Engine D "peak 974
    /// W" on a single 200 W-TDP card): 974 > 2×200 = 400 → flagged and
    /// capped at 1.5×200 = 300 W. This is the reading that, when the TDP
    /// reference was the *current* limit (which fails on some hosts →
    /// `None`), slipped past the clamp through the 5 000 W global cap and
    /// inflated the `$/1M` output cost ~10×. With the TDP now sourced from
    /// the *default* limit, the clamp engages and caps it.
    #[test]
    fn nvml_clamp_caps_the_a4000_phantom_reading() {
        let mut warned = false;
        let out = clamp_nvml_power(974.0, Some(200.0), 0, &mut warned);
        assert_eq!(
            out,
            Some(300.0),
            "974 W on a 200 W TDP is capped at 1.5× = 300 W"
        );
        assert!(warned, "the anomaly was flagged");
    }

    /// A normal (sub-2×TDP) reading passes through untouched and resets the
    /// latch.
    #[test]
    fn nvml_clamp_passes_a_normal_reading() {
        let mut warned = true; // start "latched"
        let out = clamp_nvml_power(285.0, Some(300.0), 0, &mut warned);
        assert_eq!(out, Some(285.0), "a sub-2×TDP reading is untouched");
        assert!(!warned, "a normal reading resets the latch");
    }

    /// Without a TDP (the limit query is unsupported), a 5 kW+ reading hits
    /// the global ceiling.
    #[test]
    fn nvml_clamp_falls_back_to_the_global_cap_without_a_tdp() {
        let mut warned = false;
        let out = clamp_nvml_power(355_000.0, None, 1, &mut warned);
        assert_eq!(out, Some(MAX_GPU_POWER_W));
        assert!(warned);
    }

    /// A non-finite reading degrades to `None` (the N/A rule).
    #[test]
    fn nvml_clamp_non_finite_is_na() {
        let mut warned = false;
        assert_eq!(
            clamp_nvml_power(f64::NAN, Some(300.0), 0, &mut warned),
            None
        );
        assert_eq!(clamp_nvml_power(f64::INFINITY, None, 0, &mut warned), None);
    }

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
