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
    /// H1: per-device "TDP lookup degraded to the table" latches. The
    /// power-management limit queries failed, so the TDP came from the
    /// per-model table — a one-time WARN (a separate latch from `warned` so
    /// the TDP-degradation notice never collides with the power-anomaly WARN).
    tdp_degraded: Mutex<Vec<bool>>,
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
            tdp_degraded: Mutex::new(vec![false; count as usize]),
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
        let mut tdp_guard = self.tdp_degraded.lock().unwrap_or_else(|e| e.into_inner());
        tdp_guard.resize(self.count as usize, false);
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
            // above 1.5× the TDP (e.g. the 943 W on a single 200 W A4000) is
            // flagged and capped at 1.5× the TDP.
            //
            // The TDP reference is the **default** power-management limit
            // (the card's factory TDP — not user-settable). We fall back to
            // the *current* limit only if the default query is unsupported.
            // Relying on the current limit alone is the bug the default
            // limits fix: `nvidia-smi -pl` can raise it (defeating the
            // clamp), and on some hosts the current-limit query fails
            // outright (returns `None`).
            //
            // **H1:** on some hosts *both* limit queries fail (the A4000 in
            // `work/scratch/latest.log`), leaving `tdp_w = None` and
            // disabling the TDP-based clamp — so a phantom ~989 W reading
            // passed the 1000 W global cap silently. When the queries fail
            // we fall back to the **per-model TDP table** keyed on the
            // marketing name, so the clamp still has a TDP to bound against.
            // The degradation is WARNed once per device (`tdp_degraded`).
            let name = device.name().unwrap_or_default();
            let queried_tdp = device
                .power_management_limit_default()
                .or_else(|_| device.power_management_limit())
                .ok()
                .map(|mw| mw as f64 / 1000.0);
            let tdp_w = tdp_from_queries_or_model(queried_tdp, &name);
            if queried_tdp.is_none() && !tdp_guard[i as usize] {
                match tdp_w {
                    Some(tdp) => eprintln!(
                        "[WARN] [HW] GPU {i} ({name}): power-management limit queries failed — using the per-model TDP table ({tdp:.0} W) for the plausibility clamp"
                    ),
                    None => eprintln!(
                        "[WARN] [HW] GPU {i} ({name}): power-management limit queries failed and no per-model TDP is known — only the {MAX_GPU_POWER_W:.0} W global ceiling applies"
                    ),
                }
                tdp_guard[i as usize] = true;
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
}

/// The per-model TDP table (watts) for common accelerator cards, keyed on a
/// **case-insensitive substring** of the marketing name.
///
/// **H1:** NVML's `power_management_limit_default()` / `power_management_limit()`
/// both fail on some hosts (the A4000 in `work/scratch/latest.log`), leaving
/// `tdp_w = None` and disabling the TDP-based clamp — so a phantom ~989 W
/// reading passed the 1000 W global cap silently. This table is the fallback:
/// when the limit queries fail, the TDP is looked up from the card's name.
/// The values are the cards' *factory* TDPs — conservative by design: a
/// wrong-**high** TDP would let a phantom reading through, a wrong-**low**
/// one just caps a real reading a little early.
///
/// Ordered so a more specific name matches before a substring of it (e.g.
/// `l40s` before `l40` before `l4`; `rtx 4090` before `4090`), because
/// [`tdp_for_model`] takes the *first* match.
pub const TDP_TABLE: &[(&str, f64)] = &[
    ("a4000", 200.0),
    ("a5000", 230.0),
    ("a6000", 300.0),
    ("l40s", 350.0),
    ("l40", 350.0),
    ("l4", 72.0),
    ("a100", 400.0),
    ("h100", 700.0),
    ("h200", 700.0),
    ("rtx 4090", 450.0),
    ("4090", 450.0),
    ("rtx 4080", 250.0),
    ("4080", 250.0),
    ("3090", 350.0),
    ("3080", 320.0),
    ("t4", 70.0),
    ("v100", 250.0),
    ("p40", 250.0),
];

/// Resolve a per-model TDP (watts) from a marketing name (case-insensitive
/// substring match against [`TDP_TABLE`]). `None` when no entry matches — the
/// caller then has only the global ceiling (and a stronger WARN).
#[must_use]
pub fn tdp_for_model(name: &str) -> Option<f64> {
    let n = name.to_ascii_lowercase();
    TDP_TABLE
        .iter()
        .find(|(key, _)| n.contains(&key.to_ascii_lowercase()))
        .map(|(_, tdp)| *tdp)
}

/// The TDP reference for the plausibility clamp: prefer the queried
/// power-management limit, fall back to the per-model table (H1) when both
/// limit queries failed (`queried` is `None`).
#[must_use]
pub fn tdp_from_queries_or_model(queried: Option<f64>, name: &str) -> Option<f64> {
    queried.or_else(|| tdp_for_model(name))
}

/// Plausibility-clamp an NVML power reading (watts) against the card's
/// TDP (watts) and the global ceiling.
///
/// NVML's `power_usage()`, `power_management_limit_default()` **and**
/// `power_management_limit()` all return *milliwatts*; the caller converts
/// to watts before calling this. The TDP reference is the **default**
/// power-management limit (the factory TDP, not user-settable), falling back
/// to the current limit, and — **H1** — to the per-model [`TDP_TABLE`] when
/// both limit queries failed.
///
/// **Per-model ceiling:** a reading above **1.5× the TDP** is a unit/scale
/// bug or a phantom spike — the canonical case is the **989 W reported on a
/// single 200 W A4000** (4.9× its TDP, physically impossible). It is flagged
/// with a `[WARN]` (once per anomaly episode, via the `warned` latch) and
/// **capped at 1.5× the TDP** (transient spikes are possible, 3× is not).
/// The `> 1.5×TDP` trigger is deliberately stricter than the old `> 2×TDP`
/// one: it stops a reading in `(1.5×TDP, 1000]` on a small card (e.g. 400 W
/// on a 200 W A4000) from passing unflagged — a phantom that the looser
/// trigger would have let through.
///
/// When no TDP is available at all (the limit queries failed *and* the model
/// is not in the table) the global [`MAX_GPU_POWER_W`] ceiling stands in.
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
    // Per-model ceiling (the 989 W-on-a-200 W A4000 guard). A reading above
    // 1.5× the TDP is capped at 1.5× the TDP and flagged.
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

    // ── H1: the per-model TDP table fallback ────────────────────────────

    /// H1: the per-model TDP table resolves known cards by case-insensitive
    /// name substring (the more-specific name matches before its substrings),
    /// and returns `None` for an unknown name.
    #[test]
    fn tdp_for_model_resolves_known_names() {
        assert_eq!(tdp_for_model("NVIDIA RTX A4000"), Some(200.0));
        assert_eq!(tdp_for_model("nvidia a5000"), Some(230.0));
        assert_eq!(tdp_for_model("NVIDIA A6000"), Some(300.0));
        assert_eq!(tdp_for_model("NVIDIA L40S"), Some(350.0));
        assert_eq!(tdp_for_model("NVIDIA L40"), Some(350.0));
        // `l40` must not shadow `l4`: a plain L4 resolves to 72 W.
        assert_eq!(tdp_for_model("NVIDIA L4"), Some(72.0));
        assert_eq!(tdp_for_model("NVIDIA H100"), Some(700.0));
        assert_eq!(tdp_for_model("NVIDIA A100"), Some(400.0));
        assert_eq!(tdp_for_model("NVIDIA GeForce RTX 4090"), Some(450.0));
        assert_eq!(tdp_for_model("Some Unknown GPU"), None);
    }

    /// H1: the fallback prefers the queried limit and falls back to the
    /// table only when the query is `None`.
    #[test]
    fn tdp_from_queries_or_model_prefers_the_query() {
        assert_eq!(
            tdp_from_queries_or_model(Some(180.0), "NVIDIA RTX A4000"),
            Some(180.0)
        );
        assert_eq!(
            tdp_from_queries_or_model(None, "NVIDIA RTX A4000"),
            Some(200.0)
        );
        assert_eq!(tdp_from_queries_or_model(None, "Unknown"), None);
    }

    /// H1 (the headline regression): a 989 W reading with **no** queried TDP
    /// on an A4000 resolves its TDP from the table (200 W) and is capped at
    /// 1.5×200 = 300 W — it can never pass unflagged. Before H1, `tdp_w` was
    /// `None` and the reading passed the 1000 W global cap silently.
    #[test]
    fn a4000_phantom_with_no_queried_tdp_is_capped_by_the_table() {
        let mut warned = false;
        // Both limit queries failed (`None`) → the table provides 200 W.
        let tdp = tdp_from_queries_or_model(None, "NVIDIA RTX A4000");
        assert_eq!(tdp, Some(200.0), "the table resolves the A4000 TDP");
        let out = clamp_nvml_power(989.0, tdp, 0, &mut warned);
        assert_eq!(
            out,
            Some(300.0),
            "989 W on a 200 W card is capped at 1.5× = 300 W"
        );
        assert!(warned, "the anomaly was flagged");
    }

    /// H1: a reading in `(1.5×TDP, 1000]` on a known small card is capped by
    /// the per-model ceiling — the case the old `> 2×TDP` trigger let through
    /// (400 W on a 200 W card is under both 2×TDP=400… and the 1000 W cap, so
    /// it must be caught by the stricter 1.5× rule).
    #[test]
    fn a_reading_above_1_5x_tdp_is_capped_even_below_the_global_cap() {
        let mut warned = false;
        // 400 W on a 200 W A4000: under the 1000 W global cap, over 1.5×200.
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
