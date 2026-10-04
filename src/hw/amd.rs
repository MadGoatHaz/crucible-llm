//! AMD GPU telemetry via the in-tree `amdgpu` driver's sysfs + hwmon nodes.
//!
//! Zero external dependencies, no root, no ROCm stack: pure `std::fs` reads
//! of `/sys/class/drm/card{N}/device/` and its `hwmon` child (research
//! "Linux GPU Monitoring APIs" §AMD; blueprint §5D).
//!
//! **Per-field N/A degradation:** not every node exists on every AMD GPU —
//! older Polaris/Vega cards lack `freq1_input` and `mem_info_vram_*`, and
//! some APU designs expose `power1_input` instead of `power1_average`.
//! Every read is infallible: a missing node is `None` for that field only,
//! never a panic, so a partially-exposed GPU still reports what it can.
//!
//! **Unit normalization** (raw sensor units → [`GpuSample`] units):
//! * power `power1_average` / `power1_input` — microwatts → watts (÷1e6)
//! * temperature `temp1_input` — millidegrees C → degrees C (÷1000)
//! * clock `freq1_input` — hertz → megahertz (÷1e6)
//! * VRAM `mem_info_vram_{used,total}` — bytes → MB (÷1 MiB)
//!
//! **Detection:** [`AmdSysfsBackend::try_init`] scans `/sys/class/drm` for
//! a card whose `device/vendor` is `0x1002` (AMD). `None` when no AMD GPU
//! is present — the caller degrades to N/A, never a panic.

use std::path::{Path, PathBuf};

use super::sysfs::{find_hwmon, read_i32, read_trimmed, read_u64, read_u8};
use super::{GpuBackend, GpuSample};

/// The AMD (`amdgpu`) sysfs + hwmon GPU backend.
#[derive(Debug)]
pub struct AmdSysfsBackend {
    /// The DRM card root (e.g. `/sys/class/drm/card0`).
    card_path: PathBuf,
    /// The card's `device/hwmon/hwmon{M}` dir, if the driver registered one.
    hwmon_path: Option<PathBuf>,
    /// The DRM driver label (e.g. "radeonsi", "amdgpu"), read once at init
    /// so [`GpuBackend::model`] can return a `&str` borrowed from `self`.
    model: Option<String>,
}

impl AmdSysfsBackend {
    /// Scan `/sys/class/drm` for an AMD GPU (`device/vendor == 0x1002`)
    /// and initialize a backend for the first one found.
    ///
    /// Returns `None` when no AMD GPU is present (or `/sys/class/drm` is
    /// unreadable) — the graceful-degradation path, never a panic.
    pub fn try_init() -> Option<Box<Self>> {
        let drm = PathBuf::from("/sys/class/drm");
        let cards = std::fs::read_dir(&drm).ok()?;
        for entry in cards.flatten() {
            let card_path = entry.path();
            let vendor = card_path.join("device/vendor");
            if read_trimmed(&vendor).as_deref() != Some("0x1002") {
                continue;
            }
            let device = card_path.join("device");
            let hwmon_path = find_hwmon(&device);
            let model = read_trimmed(&device.join("label"));
            return Some(Box::new(Self {
                card_path,
                hwmon_path,
                model,
            }));
        }
        None
    }

    /// Active graphics core clock in MHz.
    ///
    /// Primary: the hwmon frequency sensor (`freq1_input`, hertz). Fallback:
    /// the DPM state table (`pp_dpm_sclk`), whose active row is
    /// asterisk-marked. `None` when neither node is present.
    fn core_clock(&self) -> Option<u32> {
        if let Some(hwmon) = &self.hwmon_path {
            if let Some(hz) = read_u64(&hwmon.join("freq1_input")) {
                return Some((hz / 1_000_000) as u32);
            }
        }
        let sclk = self.card_path.join("device/pp_dpm_sclk");
        read_trimmed(&sclk).and_then(|content| parse_pp_dpm_mhz(&content))
    }

    /// Memory clock in MHz from the `pp_dpm_mclk` DPM table, `None` when
    /// the node is absent (not exposed on all cards).
    fn memory_clock(&self) -> Option<u32> {
        let mclk = self.card_path.join("device/pp_dpm_mclk");
        read_trimmed(&mclk).and_then(|content| parse_pp_dpm_mhz(&content))
    }
}

impl GpuBackend for AmdSysfsBackend {
    fn vendor(&self) -> &str {
        "AMD"
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn poll(&self) -> GpuSample {
        let device = self.card_path.join("device");

        // Temperature + power from the hwmon child (millidegrees C,
        // microwatts); `None` for both when no hwmon node is registered.
        let (temperature_c, power_watts) = self.hwmon_path.as_ref().map_or((None, None), |hwmon| {
            (
                read_i32(&hwmon.join("temp1_input")).map(|milli| milli / 1000),
                read_hwmon_power(hwmon),
            )
        });

        GpuSample {
            // Utilization: 0–100 integer (time-averaged SMU compute activity).
            utilization_pct: read_u8(&device.join("gpu_busy_percent")),
            // VRAM: raw bytes → MB.
            memory_used_mb: read_u64(&device.join("mem_info_vram_used")).map(|b| b / (1024 * 1024)),
            memory_total_mb: read_u64(&device.join("mem_info_vram_total"))
                .map(|b| b / (1024 * 1024)),
            temperature_c,
            power_watts,
            // Clocks: hwmon freq sensor (primary) / DPM table (fallback).
            core_clock_mhz: self.core_clock(),
            memory_clock_mhz: self.memory_clock(),
            // Throttle reasons are not exposed via amdgpu sysfs.
            throttle_reasons: None,
        }
    }
}

/// hwmon power draw: `power1_average` on most cards, `power1_input` on some
/// APU designs. Microwatts → watts.
fn read_hwmon_power(hwmon: &Path) -> Option<f64> {
    let path = hwmon.join("power1_average");
    let path = if path.exists() {
        path
    } else {
        hwmon.join("power1_input")
    };
    read_u64(&path).map(|uw| uw as f64 / 1_000_000.0)
}

/// Parse the active (asterisk-marked) row of an `amdgpu` DPM state table
/// (e.g. `pp_dpm_sclk` / `pp_dpm_mclk`) into MHz. The format is one
/// frequency per line, the currently active state suffixed with `*`:
///
/// ```text
///     0Mhz *
///    571Mhz
///   1600Mhz
/// ```
///
/// Robust to driver variation (asterisk attached or space-separated).
/// Returns `None` when no row is marked active.
fn parse_pp_dpm_mhz(content: &str) -> Option<u32> {
    for line in content.lines() {
        let line = line.trim();
        if !line.contains('*') {
            continue;
        }
        // Drop the asterisk, then find the "<N>Mhz" token on that row.
        let cleaned = line.replace('*', " ");
        let token = cleaned.split_whitespace().find(|t| t.ends_with("Mhz"))?;
        return token.trim_end_matches("Mhz").parse::<u32>().ok();
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::sysfs::{read_i32, read_trimmed, read_u32, read_u64, read_u8};
    use std::path::Path;

    /// `try_init` must never panic: `None` on a machine without an AMD GPU,
    /// or a working backend (whose `poll` also never panics) when one is
    /// present.
    #[test]
    fn try_init_is_graceful() {
        let result = AmdSysfsBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "AMD");
            let _sample = backend.poll(); // must not panic
        }
    }

    /// The DPM table parser finds the asterisk-marked active row.
    #[test]
    fn parse_pp_dpm_mhz_finds_active_row() {
        assert_eq!(
            parse_pp_dpm_mhz("    0Mhz *\n   571Mhz\n  1600Mhz"),
            Some(0)
        );
        assert_eq!(parse_pp_dpm_mhz("  1412Mhz *\n  1600Mhz"), Some(1412));
        // Asterisk directly attached (no separating space) is handled too.
        assert_eq!(parse_pp_dpm_mhz("  800Mhz*\n  900Mhz"), Some(800));
    }

    /// A DPM table with no active marker (or empty) yields `None` — never
    /// a panic.
    #[test]
    fn parse_pp_dpm_mhz_none_without_active_row() {
        assert_eq!(parse_pp_dpm_mhz("  571Mhz\n  1600Mhz"), None);
        assert_eq!(parse_pp_dpm_mhz(""), None);
    }

    /// The shared read helpers are infallible: a missing path is `None`.
    #[test]
    fn read_helpers_are_infallible() {
        let missing = Path::new("/nonexistent/crucible-test-node");
        assert_eq!(read_u64(missing), None);
        assert_eq!(read_u32(missing), None);
        assert_eq!(read_u8(missing), None);
        assert_eq!(read_i32(missing), None);
        assert_eq!(read_trimmed(missing), None);
    }
}
