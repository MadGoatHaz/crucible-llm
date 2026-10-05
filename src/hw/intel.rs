//! Intel GPU telemetry via the in-tree `i915` / `xe` driver's sysfs +
//! hwmon nodes.
//!
//! Zero external dependencies, no root: pure `std::fs` reads (research
//! "Linux GPU Monitoring APIs" §Intel; blueprint §5D).
//!
//! **Multi-GPU:** [`IntelSysfsBackend::try_init`] collects **every** DRM
//! card whose `device/vendor` is `0x8086` (Intel) — dual-Arc rigs and
//! discrete + integrated combos all report, not just the first card.
//! [`GpuBackend::poll_all`] yields one [`GpuSample`] per card and
//! [`GpuBackend::poll`] is their aggregate roll-up (the same shape NVML
//! produces).
//!
//! **Known sysfs limitations (documented, per-field N/A):**
//! * **Utilization is always `None`.** Neither `i915` nor `xe` exposes a
//!   scalar `gpu_busy_percent` in sysfs; engine load requires the perf PMU
//!   or Level Zero Sysman (out of scope for the zero-dep sysfs path).
//! * **VRAM is `None` on most systems.** There is no universal
//!   `mem_info_vram_used` node for Intel; only some discrete Arc cards
//!   expose `lmem_used_bytes` / `lmem_total_bytes`.
//! * **Integrated iGPU power is `None`.** iGPU power lives in the CPU
//!   package RAPL powercap node, not the DRM device; discrete Arc cards
//!   expose it via hwmon `power1_average` / `power1_input`.
//! * **Multi-tile `xe` reads `tile0` only** (the primary tile).
//!
//! **Driver divergence:** `i915` reports the active GT frequency at
//! `card{N}/gt_act_freq_mhz`; `xe` reports it under the tile hierarchy at
//! `card{N}/device/tile0/gt0/freq0/act_freq`.
//! [`IntelSysfsBackend::try_init`] detects which driver binds **each**
//! card (via the `device/driver` symlink) and selects the correct path.
//!
//! **Detection:** `None` when no Intel GPU is present (or
//! `/sys/class/drm` is unreadable) — the caller degrades to N/A, never a
//! panic.

use std::path::{Path, PathBuf};

use super::intel_level_zero::IntelLevelZeroBackend;
use super::sysfs::{find_hwmon, read_i32, read_trimmed, read_u32, read_u64};
use super::{GpuBackend, GpuSample};

/// One Intel DRM card and the telemetry nodes resolved for it at init.
#[derive(Debug)]
struct IntelCard {
    /// The DRM card root (e.g. `/sys/class/drm/card0`).
    card_path: PathBuf,
    /// The card's `device/hwmon/hwmon{M}` dir, if registered.
    hwmon_path: Option<PathBuf>,
    /// `true` when the `xe` driver (not legacy `i915`) binds the card.
    is_xe: bool,
    /// The DRM driver label ("i915" / "xe" model string), read once at
    /// init so per-device names can be returned without re-reading sysfs.
    model: Option<String>,
}

/// The Intel (`i915` / `xe`) sysfs + hwmon GPU backend — **all** Intel
/// cards on the host, one [`GpuSample`] each.
#[derive(Debug)]
pub struct IntelSysfsBackend {
    cards: Vec<IntelCard>,
}

impl IntelSysfsBackend {
    /// Scan `/sys/class/drm` for **all** Intel GPUs (`device/vendor ==
    /// 0x8086`) and initialize a backend holding every one of them.
    ///
    /// Returns `None` when no Intel GPU is present (or `/sys/class/drm`
    /// is unreadable) — the graceful-degradation path, never a panic.
    pub fn try_init() -> Option<Box<Self>> {
        let drm = PathBuf::from("/sys/class/drm");
        let cards = std::fs::read_dir(&drm).ok()?;
        let mut found = Vec::new();
        for entry in cards.flatten() {
            let card_path = entry.path();
            if !card_path.is_dir() {
                continue;
            }
            let vendor = card_path.join("device/vendor");
            if read_trimmed(&vendor).as_deref() != Some("0x8086") {
                continue;
            }
            let device = card_path.join("device");
            // The `device/driver` symlink resolves to `.../i915` or `.../xe`.
            let is_xe = std::fs::read_link(device.join("driver"))
                .map(|p| p.to_string_lossy().contains("xe"))
                .unwrap_or(false);
            found.push(IntelCard {
                card_path,
                hwmon_path: find_hwmon(&device),
                is_xe,
                model: read_trimmed(&device.join("label")),
            });
        }
        (!found.is_empty()).then(|| Box::new(Self { cards: found }))
    }
}

impl GpuBackend for IntelSysfsBackend {
    fn vendor(&self) -> &str {
        "Intel"
    }

    /// The first card's label: `gpu_display_name` composes the panel
    /// title from it, and the per-device table shows every card
    /// individually via [`Self::device_names`].
    fn model(&self) -> Option<&str> {
        self.cards.first().and_then(|c| c.model.as_deref())
    }

    fn poll(&self) -> GpuSample {
        GpuSample::aggregate(&self.poll_all())
    }

    /// One sample per Intel card (the multi-GPU table's source).
    fn poll_all(&self) -> Vec<GpuSample> {
        self.cards.iter().map(sample_card).collect()
    }

    /// The display name of each card (parallel to [`Self::poll_all`]).
    fn device_names(&self) -> Vec<String> {
        self.cards
            .iter()
            .enumerate()
            .map(|(i, c)| c.model.clone().unwrap_or_else(|| format!("Intel GPU {i}")))
            .collect()
    }
}

/// Read one card's full telemetry sample (all fields optional, N/A rule).
fn sample_card(card: &IntelCard) -> GpuSample {
    let device = card.card_path.join("device");

    // Temperature + power from the hwmon child (millidegrees C,
    // microwatts); `None` for both when no hwmon node is registered
    // (e.g. integrated iGPUs, whose power lives in the CPU RAPL node).
    let (temperature_c, power_watts) = card.hwmon_path.as_ref().map_or((None, None), |hwmon| {
        (
            read_i32(&hwmon.join("temp1_input")).map(|milli| milli / 1000),
            read_hwmon_power(hwmon),
        )
    });

    GpuSample {
        // Utilization: NOT in Intel sysfs (documented limitation) —
        // engine load requires the perf PMU / Level Zero Sysman.
        utilization_pct: None,
        // VRAM: only some discrete Arc cards expose `lmem_*` nodes.
        memory_used_mb: read_u64(&device.join("lmem_used_bytes")).map(|b| b / (1024 * 1024)),
        memory_total_mb: read_u64(&device.join("lmem_total_bytes")).map(|b| b / (1024 * 1024)),
        temperature_c,
        power_watts,
        // Core clock: driver-dependent path (i915 vs xe).
        core_clock_mhz: core_clock(card),
        // Memory clock: not in Intel sysfs.
        memory_clock_mhz: None,
        // Throttle reasons: not in Intel sysfs.
        throttle_reasons: None,
    }
}

/// Active GT core clock in MHz, selecting the node by driver:
/// `xe` reads the tile hierarchy, `i915` reads the top-level card node
/// (with the per-GT `rps_act_freq_mhz` fallback). `None` when neither
/// is present.
fn core_clock(card: &IntelCard) -> Option<u32> {
    if card.is_xe {
        let path = card.card_path.join("device/tile0/gt0/freq0/act_freq");
        read_u32(&path)
    } else {
        let primary = card.card_path.join("gt_act_freq_mhz");
        if let Some(mhz) = read_u32(&primary) {
            return Some(mhz);
        }
        read_u32(&card.card_path.join("gt/gt0/rps_act_freq_mhz"))
    }
}

/// The Intel GPU detection chain: **Level Zero Sysman first**, then the
/// sysfs/hwmon fallback.
///
/// 1. [`IntelLevelZeroBackend`] — preferred: full telemetry for discrete
///    Arc GPUs (power, utilization, VRAM, core + memory clocks,
///    temperature, throttle reasons) via `libze_loader.so.1`, loaded
///    dynamically. `None` when the Level Zero loader / Intel Compute
///    Runtime is not installed — never a panic.
/// 2. [`IntelSysfsBackend`] — fallback: works on every Intel GPU
///    (including integrated) with the documented per-field limitations.
///
/// Returns `None` only when *neither* surface is available (no Intel
/// GPU at all) — the caller degrades to N/A.
pub fn detect_intel_backend() -> Option<Box<dyn GpuBackend>> {
    if let Some(level_zero) = IntelLevelZeroBackend::try_init() {
        return Some(level_zero);
    }
    IntelSysfsBackend::try_init().map(|sysfs| sysfs as Box<dyn GpuBackend>)
}

/// The marketing labels of **all** Intel DRM cards, in
/// `/sys/class/drm` order (e.g. `["Intel Arc A770", "Intel Arc A770"]`).
///
/// The Level Zero backend pairs these with its enumerated devices (by
/// index) for per-device names. Empty when no readable Intel card
/// exists (the N/A rule).
pub fn intel_card_labels() -> Vec<String> {
    let drm = PathBuf::from("/sys/class/drm");
    let Ok(cards) = std::fs::read_dir(&drm) else {
        return Vec::new();
    };
    let mut labels = Vec::new();
    for entry in cards.flatten() {
        let card = entry.path();
        if !card.is_dir() {
            continue;
        }
        if read_trimmed(&card.join("device/vendor")).as_deref() != Some("0x8086") {
            continue;
        }
        if let Some(label) = read_trimmed(&card.join("device/label")) {
            labels.push(label);
        }
    }
    labels
}

/// hwmon power draw: `power1_average` on discrete Arc, `power1_input` on
/// some designs. Microwatts → watts. `None` for integrated iGPUs (power
/// lives in the CPU RAPL powercap node, not the DRM device).
fn read_hwmon_power(hwmon: &Path) -> Option<f64> {
    let path = hwmon.join("power1_average");
    let path = if path.exists() {
        path
    } else {
        hwmon.join("power1_input")
    };
    read_u64(&path).map(|uw| uw as f64 / 1_000_000.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::sysfs::{read_i32, read_trimmed, read_u32, read_u64};
    use std::path::Path;

    /// `try_init` must never panic: `None` on a machine without an Intel
    /// GPU, or a working backend (whose `poll` and `poll_all` also never
    /// panic) when one is present.
    #[test]
    fn try_init_is_graceful() {
        let result = IntelSysfsBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "Intel");
            assert!(!backend.cards.is_empty(), "a backend holds ≥1 card");
            let _sample = backend.poll(); // must not panic
            let all = backend.poll_all();
            assert_eq!(all.len(), backend.cards.len(), "one sample per card");
            assert_eq!(
                all.len(),
                backend.device_names().len(),
                "names parallel to samples"
            );
        }
    }

    /// `poll` is the aggregate roll-up of `poll_all` (power summed).
    #[test]
    fn poll_is_the_aggregate_of_poll_all() {
        let Some(backend) = IntelSysfsBackend::try_init() else {
            return; // no Intel GPU on this machine — nothing to verify
        };
        let all = backend.poll_all();
        let agg = GpuSample::aggregate(&all);
        let single = backend.poll();
        assert_eq!(single.power_watts, agg.power_watts);
        assert_eq!(single.memory_used_mb, agg.memory_used_mb);
        assert_eq!(single.temperature_c, agg.temperature_c);
    }

    /// The detection chain (Level Zero → sysfs) must never panic: `None`
    /// on a machine with no Intel GPU, or a working backend (Level Zero
    /// when the loader is installed, sysfs otherwise) whose `poll` also
    /// never panics.
    #[test]
    fn detect_intel_backend_is_graceful() {
        let result = detect_intel_backend();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "Intel");
            let _sample = backend.poll(); // must not panic
        }
    }

    /// The sysfs label scan is infallible (empty on a non-Intel host).
    #[test]
    fn intel_card_labels_is_infallible() {
        let _labels = intel_card_labels();
    }

    /// The shared read helpers are infallible: a missing path is `None`.
    #[test]
    fn read_helpers_are_infallible() {
        let missing = Path::new("/nonexistent/crucible-test-node");
        assert_eq!(read_u64(missing), None);
        assert_eq!(read_u32(missing), None);
        assert_eq!(read_i32(missing), None);
        assert_eq!(read_trimmed(missing), None);
    }
}
