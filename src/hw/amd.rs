//! AMD GPU telemetry via the in-tree `amdgpu` driver's sysfs + hwmon nodes.
//!
//! Zero external dependencies, no root, no ROCm stack: pure `std::fs` reads
//! of `/sys/class/drm/card{N}/device/` and its `hwmon` child (research
//! "Linux GPU Monitoring APIs" §AMD; blueprint §5D).
//!
//! **Multi-GPU:** [`AmdSysfsBackend::try_init`] collects **every** DRM card
//! whose `device/vendor` is `0x1002` (AMD) — a quad-9700 rig reports all
//! four cards, not just the first. [`GpuBackend::poll_all`] yields one
//! [`GpuSample`] per card (the per-GPU table), and
//! [`GpuBackend::poll`] is their aggregate roll-up (summed power / VRAM,
//! maxed clocks / temp / util — the same shape NVML produces).
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
//! **Detection:** `None` when no AMD GPU is present (or
//! `/sys/class/drm` is unreadable) — the caller degrades to N/A, never a
//! panic.

use std::path::{Path, PathBuf};

use super::sysfs::{find_hwmon, read_i32, read_trimmed, read_u64, read_u8};
use super::{GpuBackend, GpuSample};

/// One AMD DRM card and the telemetry nodes resolved for it at init.
#[derive(Debug)]
struct AmdCard {
    /// The DRM card root (e.g. `/sys/class/drm/card0`).
    card_path: PathBuf,
    /// The card's `device/hwmon/hwmon{M}` dir, if the driver registered one.
    hwmon_path: Option<PathBuf>,
    /// The DRM driver label (e.g. "Radeon RX 9700"), read once at init so
    /// per-device names can be returned without re-reading sysfs.
    model: Option<String>,
}

/// The AMD (`amdgpu`) sysfs + hwmon GPU backend — **all** AMD cards on the
/// host, one [`GpuSample`] each.
#[derive(Debug)]
pub struct AmdSysfsBackend {
    cards: Vec<AmdCard>,
}

impl AmdSysfsBackend {
    /// Scan `/sys/class/drm` for **all** AMD GPUs (`device/vendor ==
    /// 0x1002`) and initialize a backend holding every one of them.
    ///
    /// Returns `None` when no AMD GPU is present (or `/sys/class/drm` is
    /// unreadable) — the graceful-degradation path, never a panic.
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
            if read_trimmed(&vendor).as_deref() != Some("0x1002") {
                continue;
            }
            let device = card_path.join("device");
            found.push(AmdCard {
                card_path,
                hwmon_path: find_hwmon(&device),
                model: read_trimmed(&device.join("label")),
            });
        }
        (!found.is_empty()).then(|| Box::new(Self { cards: found }))
    }
}

impl GpuBackend for AmdSysfsBackend {
    fn vendor(&self) -> &str {
        "AMD"
    }

    /// The first card's label: `gpu_display_name` composes the panel
    /// title from it (e.g. "AMD Radeon RX 9700"), and the per-device
    /// table shows every card individually via [`Self::device_names`].
    fn model(&self) -> Option<&str> {
        self.cards.first().and_then(|c| c.model.as_deref())
    }

    fn poll(&self) -> GpuSample {
        GpuSample::aggregate(&self.poll_all())
    }

    /// One sample per AMD card (the multi-GPU table's source).
    fn poll_all(&self) -> Vec<GpuSample> {
        self.cards.iter().map(sample_card).collect()
    }

    /// The display name of each card (parallel to [`Self::poll_all`]).
    fn device_names(&self) -> Vec<String> {
        self.cards
            .iter()
            .enumerate()
            .map(|(i, c)| c.model.clone().unwrap_or_else(|| format!("AMD GPU {i}")))
            .collect()
    }
}

/// Read one card's full telemetry sample (all fields optional, N/A rule).
fn sample_card(card: &AmdCard) -> GpuSample {
    let device = card.card_path.join("device");

    // Temperature + power from the hwmon child (millidegrees C,
    // microwatts); `None` for both when no hwmon node is registered.
    let (temperature_c, power_watts) = card.hwmon_path.as_ref().map_or((None, None), |hwmon| {
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
        memory_total_mb: read_u64(&device.join("mem_info_vram_total")).map(|b| b / (1024 * 1024)),
        temperature_c,
        power_watts,
        // Clocks: hwmon freq sensor (primary) / DPM table (fallback).
        core_clock_mhz: core_clock(card),
        memory_clock_mhz: memory_clock(card),
        // Throttle reasons are not exposed via amdgpu sysfs.
        throttle_reasons: None,
    }
}

/// Active graphics core clock in MHz.
///
/// Primary: the hwmon frequency sensor (`freq1_input`, hertz). Fallback:
/// the DPM state table (`pp_dpm_sclk`), whose active row is
/// asterisk-marked. `None` when neither node is present.
fn core_clock(card: &AmdCard) -> Option<u32> {
    if let Some(hwmon) = &card.hwmon_path {
        if let Some(hz) = read_u64(&hwmon.join("freq1_input")) {
            return Some((hz / 1_000_000) as u32);
        }
    }
    let sclk = card.card_path.join("device/pp_dpm_sclk");
    read_trimmed(&sclk).and_then(|content| parse_pp_dpm_mhz(&content))
}

/// Memory clock in MHz from the `pp_dpm_mclk` DPM table, `None` when
/// the node is absent (not exposed on all cards).
fn memory_clock(card: &AmdCard) -> Option<u32> {
    let mclk = card.card_path.join("device/pp_dpm_mclk");
    read_trimmed(&mclk).and_then(|content| parse_pp_dpm_mhz(&content))
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
    /// or a working backend (whose `poll` and `poll_all` also never panic)
    /// when one is present.
    #[test]
    fn try_init_is_graceful() {
        let result = AmdSysfsBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "AMD");
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
        let Some(backend) = AmdSysfsBackend::try_init() else {
            return; // no AMD GPU on this machine — nothing to verify
        };
        let all = backend.poll_all();
        let agg = GpuSample::aggregate(&all);
        let single = backend.poll();
        assert_eq!(single.power_watts, agg.power_watts);
        assert_eq!(single.memory_used_mb, agg.memory_used_mb);
        assert_eq!(single.temperature_c, agg.temperature_c);
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
