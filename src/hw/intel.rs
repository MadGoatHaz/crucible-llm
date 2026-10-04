//! Intel GPU telemetry via the in-tree `i915` / `xe` driver's sysfs +
//! hwmon nodes.
//!
//! Zero external dependencies, no root: pure `std::fs` reads (research
//! "Linux GPU Monitoring APIs" §Intel; blueprint §5D).
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
//! [`IntelSysfsBackend::try_init`] detects which driver binds the card
//! (via the `device/driver` symlink) and selects the correct path.
//!
//! **Detection:** [`IntelSysfsBackend::try_init`] scans `/sys/class/drm`
//! for a card whose `device/vendor` is `0x8086` (Intel). `None` when no
//! Intel GPU is present — the caller degrades to N/A, never a panic.

use std::path::{Path, PathBuf};

use super::sysfs::{find_hwmon, read_i32, read_trimmed, read_u32, read_u64};
use super::{GpuBackend, GpuSample};

/// The Intel (`i915` / `xe`) sysfs + hwmon GPU backend.
#[derive(Debug)]
pub struct IntelSysfsBackend {
    /// The DRM card root (e.g. `/sys/class/drm/card0`).
    card_path: PathBuf,
    /// The card's `device/hwmon/hwmon{M}` dir, if registered.
    hwmon_path: Option<PathBuf>,
    /// `true` when the `xe` driver (not legacy `i915`) binds the card.
    is_xe: bool,
    /// The DRM driver label ("i915" / "xe"), read once at init so
    /// [`GpuBackend::model`] can return a `&str` borrowed from `self`.
    model: Option<String>,
}

impl IntelSysfsBackend {
    /// Scan `/sys/class/drm` for an Intel GPU (`device/vendor == 0x8086`)
    /// and initialize a backend for the first one found.
    ///
    /// Returns `None` when no Intel GPU is present (or `/sys/class/drm`
    /// is unreadable) — the graceful-degradation path, never a panic.
    pub fn try_init() -> Option<Box<Self>> {
        let drm = PathBuf::from("/sys/class/drm");
        let cards = std::fs::read_dir(&drm).ok()?;
        for entry in cards.flatten() {
            let card_path = entry.path();
            let vendor = card_path.join("device/vendor");
            if read_trimmed(&vendor).as_deref() != Some("0x8086") {
                continue;
            }
            let device = card_path.join("device");
            // The `device/driver` symlink resolves to `.../i915` or `.../xe`.
            let is_xe = std::fs::read_link(device.join("driver"))
                .map(|p| p.to_string_lossy().contains("xe"))
                .unwrap_or(false);
            let hwmon_path = find_hwmon(&device);
            let model = read_trimmed(&device.join("label"));
            return Some(Box::new(Self {
                card_path,
                hwmon_path,
                is_xe,
                model,
            }));
        }
        None
    }

    /// Active GT core clock in MHz, selecting the node by driver:
    /// `xe` reads the tile hierarchy, `i915` reads the top-level card node
    /// (with the per-GT `rps_act_freq_mhz` fallback). `None` when neither
    /// is present.
    fn core_clock(&self) -> Option<u32> {
        if self.is_xe {
            let path = self.card_path.join("device/tile0/gt0/freq0/act_freq");
            read_u32(&path)
        } else {
            let primary = self.card_path.join("gt_act_freq_mhz");
            if let Some(mhz) = read_u32(&primary) {
                return Some(mhz);
            }
            read_u32(&self.card_path.join("gt/gt0/rps_act_freq_mhz"))
        }
    }
}

impl GpuBackend for IntelSysfsBackend {
    fn vendor(&self) -> &str {
        "Intel"
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn poll(&self) -> GpuSample {
        let device = self.card_path.join("device");

        // Temperature + power from the hwmon child (millidegrees C,
        // microwatts); `None` for both when no hwmon node is registered
        // (e.g. integrated iGPUs, whose power lives in the CPU RAPL node).
        let (temperature_c, power_watts) = self.hwmon_path.as_ref().map_or((None, None), |hwmon| {
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
            core_clock_mhz: self.core_clock(),
            // Memory clock: not in Intel sysfs.
            memory_clock_mhz: None,
            // Throttle reasons: not in Intel sysfs.
            throttle_reasons: None,
        }
    }
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
    /// GPU, or a working backend (whose `poll` also never panics) when one
    /// is present.
    #[test]
    fn try_init_is_graceful() {
        let result = IntelSysfsBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "Intel");
            let _sample = backend.poll(); // must not panic
        }
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
