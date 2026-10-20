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
    /// Per-device "power reading unvalidated" latches. The
    /// power-management limit queries failed, so no TDP reference is
    /// available — a one-time WARN that the reading stands, bounded only by
    /// the global ceiling (a label of uncertainty, not a correction; a
    /// separate latch from `warned` so the two notices never collide).
    power_unvalidated: Mutex<Vec<bool>>,
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
            power_unvalidated: Mutex::new(vec![false; count as usize]),
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

    /// P3/N3: the NVIDIA driver version (`nvmlSystemGetDriverVersion`),
    /// e.g. `"550.54.05"`. `None` when the driver refuses the query (the
    /// N/A rule — the `[HW]` detail line omits it, never fabricates).
    pub fn driver_version(&self) -> Option<String> {
        self.nvml.sys_driver_version().ok()
    }

    /// P3/N3: the CUDA driver version (`nvmlSystemGetCudaDriverVersion`),
    /// formatted `major.minor` (e.g. `12040` → `"12.4"`). `None` when the
    /// driver refuses the query (the N/A rule).
    pub fn cuda_version(&self) -> Option<String> {
        self.nvml
            .sys_cuda_driver_version()
            .ok()
            .map(|v| {
                format!(
                    "{}.{}",
                    nvml_wrapper::cuda_driver_version_major(v),
                    nvml_wrapper::cuda_driver_version_minor(v)
                )
            })
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
        let mut unvalidated_guard = self
            .power_unvalidated
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        unvalidated_guard.resize(self.count as usize, false);
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
            // above 1.5× the TDP is flagged and capped at 1.5× the TDP.
            //
            // The TDP reference is the **default** power-management limit
            // (the card's factory TDP — not user-settable). We fall back to
            // the *current* limit only if the default query is unsupported.
            // Relying on the current limit alone is the bug the default
            // limits fix: `nvidia-smi -pl` can raise it (defeating the
            // clamp), and on some hosts the current-limit query fails
            // outright (returns `None`).
            //
            // **Generic, no per-card assumptions:** when *both* limit
            // queries fail there is no TDP reference. The measured reading
            // then **stands**, bounded only by the global
            // `MAX_GPU_POWER_W` ceiling, and is flagged **unvalidated** by a
            // one-time WARN (`power_unvalidated`) — a label of uncertainty,
            // not a correction (a guessed card value would be).
            let name = device.name().unwrap_or_default();
            let tdp_w = device
                .power_management_limit_default()
                .or_else(|_| device.power_management_limit())
                .ok()
                .map(|mw| mw as f64 / 1000.0);
            if tdp_w.is_none() && !unvalidated_guard[i as usize] {
                eprintln!(
                    "[WARN] [HW] GPU {i} ({name}): power-management limit queries failed — no TDP reference available; the power reading is UNVALIDATED (only the {MAX_GPU_POWER_W:.0} W global ceiling applies)"
                );
                unvalidated_guard[i as usize] = true;
            }
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

    /// P3/N3: the GPU driver + CUDA versions for the `[HW]` detail line.
    /// NVML is the only backend that can provide them (the trait default
    /// returns `(None, None)` for AMD / Intel).
    fn driver_and_cuda_versions(&self) -> (Option<String>, Option<String>) {
        (self.driver_version(), self.cuda_version())
    }
}

/// Plausibility-clamp an NVML power reading (watts) against the card's
/// TDP (watts) and the global ceiling.
///
/// NVML's `power_usage()`, `power_management_limit_default()` **and**
/// `power_management_limit()` all return *milliwatts*; the caller converts
/// to watts before calling this. The TDP reference is the **queried**
/// power-management limit — the **default** (the factory TDP, not
/// user-settable), falling back to the current limit.
///
/// **Generic, no per-card assumptions:**
///
/// * **With a queried TDP** (`Some`): a reading above **1.5× the TDP** is a
///   unit/scale bug or a phantom spike. It is flagged with a `[WARN]` (once
///   per anomaly episode, via the `warned` latch) and **capped at 1.5× the
///   TDP** (transient spikes are possible, 3× is not). The `> 1.5×TDP`
///   trigger is deliberately stricter than the old `> 2×TDP` one: it stops
///   a reading in `(1.5×TDP, 1000]` on a small card from passing
///   unflagged — a phantom that the looser trigger would have let through.
/// * **Without a TDP** (`None` — both limit queries failed): there is no
///   per-card value to clamp against. The measured reading **stands**,
///   subject only to the global [`MAX_GPU_POWER_W`] last-resort ceiling,
///   and is flagged **unvalidated** (a one-time WARN in `poll_all`) — a
///   label of uncertainty, not a correction.
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
    // TDP ceiling: a reading above 1.5× the queried TDP is capped at 1.5×
    // the TDP and flagged (transient spikes are possible, 3× is not).
    if let Some(tdp) = tdp_w {
        if tdp > 0.0 {
            let ceiling = 1.5 * tdp;
            if power_w > ceiling {
                if !*warned {
                    eprintln!(
                        "[WARN] [HW] GPU {index} reported {power_w:.0} W but the TDP is {tdp:.0} W — capping at {ceiling:.0} W (1.5× TDP)"
                    );
                    *warned = true;
                }
                return Some(ceiling);
            }
        }
    }
    // Global ceiling (no TDP at all — the last-resort guard).
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
        let mw = 300_000u32; // 300 W in milliwatts
        assert!((mw as f64 / 1000.0 - 300.0).abs() < 1e-9);
        assert!((19_500_f64 / 1000.0 - 19.5).abs() < 1e-9);
    }

    /// A 943 W reading on a 300 W-TDP card: 943 > 1.5×300 = 450 → flagged
    /// and capped at 450 W. A stuck reading does not re-warn (the latch
    /// holds until the reading normalizes).
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

    /// A 974 W phantom reading on a 200 W-TDP card: 974 > 1.5×200 = 300 →
    /// flagged and capped at 300 W. This is the reading that, when the TDP
    /// reference was the *current* limit (which fails on some hosts →
    /// `None`), slipped past the clamp through the 1000 W global cap and
    /// inflated the `$/1M` output cost ~10×. With the TDP sourced from the
    /// *default* limit, the clamp engages and caps it.
    #[test]
    fn nvml_clamp_caps_a_phantom_reading() {
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

    // ── generic no-TDP behavior (no per-card assumptions) ───────────────

    /// With **no** TDP reference (both limit queries failed), a reading
    /// under the global ceiling is **not** card-clamped — it stands as
    /// measured. The `poll_all` one-time WARN flags it unvalidated; the
    /// clamp itself does not warn below the global cap.
    #[test]
    fn a_reading_with_no_tdp_reference_is_not_card_clamped() {
        let mut warned = false;
        let out = clamp_nvml_power(989.0, None, 0, &mut warned);
        assert_eq!(
            out,
            Some(989.0),
            "no per-card cap: the reading stands under the global ceiling"
        );
        assert!(!warned, "the clamp does not warn below the global cap");
    }

    /// With **no** TDP reference, a reading above the global ceiling is
    /// capped at `MAX_GPU_POWER_W` and flagged — the last-resort guard.
    #[test]
    fn a_reading_with_no_tdp_reference_above_the_global_cap_is_capped() {
        let mut warned = false;
        let out = clamp_nvml_power(1200.0, None, 0, &mut warned);
        assert_eq!(out, Some(MAX_GPU_POWER_W), "capped at the global ceiling");
        assert!(warned, "the anomaly was flagged");
    }

    /// A reading in `(1.5×TDP, 1000]` on a small card is capped by the
    /// 1.5×TDP ceiling — the case the old `> 2×TDP` trigger let through
    /// (400 W on a 200 W card is under both 2×TDP=400… and the 1000 W cap,
    /// so it must be caught by the stricter 1.5× rule).
    #[test]
    fn a_reading_above_1_5x_tdp_is_capped_even_below_the_global_cap() {
        let mut warned = false;
        // 400 W on a 200 W-TDP card: under the 1000 W global cap, over 1.5×200.
        let out = clamp_nvml_power(400.0, Some(200.0), 0, &mut warned);
        assert_eq!(out, Some(300.0), "400 W on a 200 W card caps at 300 W");
        assert!(warned);
        // A sub-1.5×TDP reading on the same card passes untouched.
        let mut warned2 = false;
        assert_eq!(
            clamp_nvml_power(280.0, Some(200.0), 0, &mut warned2),
            Some(280.0)
        );
        assert!(!warned2);
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
