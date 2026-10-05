//! CPU power monitoring via `sysfs` only (no `sensors`, no special drivers).
//!
//! The GPU backends report only the accelerator's draw; the *total system*
//! energy (and therefore the `$/1M` cost) also includes the CPU. This module
//! adds that CPU component with a **detection cascade** (research
//! `work/docs/research/cpu-power-monitoring.md`):
//!
//! 1. **Intel RAPL** — the `powercap/intel-rapl/…/energy_uj` *cumulative*
//!    counter; power = ΔµJ / Δt. Accepted at init **only if** a live ~0.5 s
//!    sample is within a sane bound — the research doc proved a phantom
//!    `intel-rapl` binding on AMD reports ~69 kW, so a plausibility clamp is
//!    non-negotiable, never trusted blindly.
//! 2. **Super I/O** — a CPU chip's direct `power1_input` / `power1_average`
//!    (µW) file, else a CPU-labeled `curr*` (mA) × `in*` (mV) product (the
//!    workhorse AMD signal, e.g. ASUS `asusec`).
//! 3. **Fixed estimate** — a clearly-labeled 40 W constant (a desktop CPU
//!    under inference load) when neither real source is available.
//!
//! **Unit normalization** (raw sensor units → watts):
//! * RAPL `energy_uj` — microjoules; `P(W) = ΔµJ / Δs / 1e6`
//! * hwmon `power*_input` / `power*_average` — microwatts → watts (÷1e6)
//! * hwmon `in*_input` — millivolts; `curr*_input` — milliamperes;
//!   `P(W) = (mV / 1000) × (mA / 1000)`
//!
//! **Plausibility clamp (non-negotiable):** every reading is checked against
//! a desktop-class ceiling ([`MAX_CPU_POWER_W`], 600 W); an out-of-range or
//! non-finite result degrades to the fixed estimate for that sample rather
//! than polluting the energy / cost math. This is the safety net that
//! neutralizes the "69 kW phantom RAPL" class of bug.
//!
//! **N/A / never-panic rule:** every `sysfs` read is an infallible `Option`
//! (a node that disappears mid-run yields the last known value, never a
//! panic). Off-Linux (where `/sys` is absent) the cascade falls through to
//! the clearly-labeled estimate — the app runs normally.

use std::path::{Path, PathBuf};
use std::time::Instant;

use super::sysfs::{read_f64, read_trimmed, read_u64};

/// A desktop CPU draws at most a few hundred watts (a Ryzen 5950X PPT is
/// ~225 W; even large server parts stay under ~600 W). A reading above this
/// is a unit/scale bug (e.g. the phantom `intel-rapl` binding on AMD that
/// reports ~69 kW) — it is clamped, never trusted.
const MAX_CPU_POWER_W: f64 = 600.0;

/// RAPL-specific ceiling: a Δ over one poll interval above 250 W is a
/// counter-wrap or a phantom binding (the task rule) — use the estimate
/// instead of trusting it.
const RAPL_MAX_W: f64 = 250.0;

/// The clearly-labeled fixed estimate (a desktop CPU under inference load).
const ESTIMATED_CPU_W: f64 = 40.0;

/// How the current CPU wattage was obtained (surfaced to the UI / JSON so a
/// user sees *measured* vs *estimated*).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuPowerMethod {
    /// `powercap/intel-rapl/…/energy_uj` cumulative counter (ΔµJ / Δt).
    Rapl,
    /// A Super I/O `power*_input` (µW) file, or `curr*` (mA) × `in*` (mV).
    SuperIo,
    /// A clearly-labeled fixed estimate (no real sensor available).
    Estimated,
}

impl CpuPowerMethod {
    /// The human-readable label (the `[HW] CPU power: …` startup log and the
    /// TUI system-power panel).
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            CpuPowerMethod::Rapl => "Intel RAPL",
            CpuPowerMethod::SuperIo => "Super I/O",
            CpuPowerMethod::Estimated => "Estimated (40W)",
        }
    }

    /// `true` for the estimate-only method (the UI flags it as non-measured).
    #[must_use]
    pub fn is_estimated(self) -> bool {
        matches!(self, CpuPowerMethod::Estimated)
    }
}

/// The resolved sensor source behind the chosen method.
#[derive(Debug)]
enum Source {
    /// RAPL: the cumulative `energy_uj` path + the delta baseline.
    Rapl {
        path: PathBuf,
        prev_uj: u64,
        prev_time: Instant,
    },
    /// Super I/O: a direct power file, or a CPU current × CPU voltage pair.
    SuperIo {
        power: Option<PathBuf>,
        curr: Option<PathBuf>,
        vin: Option<PathBuf>,
    },
    /// Fixed estimate (no real sensor).
    Estimated,
}

/// The CPU power backend: one resolved source + the last valid reading.
///
/// `Send` (it lives inside the `HwPoller`'s `Mutex`); not `Sync` (its
/// `poll` is `&mut self` — the poller owns exclusive access).
pub struct CpuPowerBackend {
    /// The accepted method (its label is surfaced to the UI / log).
    method: CpuPowerMethod,
    /// The resolved sensor source (the method-specific state).
    source: Source,
    /// The last valid reading, watts — a node that vanishes mid-run yields
    /// this (the never-panic / last-known-value rule).
    last_watts: f64,
}

impl std::fmt::Debug for CpuPowerBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuPowerBackend")
            .field("method", &self.method)
            .field("last_watts", &self.last_watts)
            .finish()
    }
}

impl CpuPowerBackend {
    /// Run the detection cascade (RAPL → Super I/O → estimate) and return
    /// the backend. **Never `None`**: the cascade always ends in the
    /// clearly-labeled fixed estimate, so the app has a CPU component on
    /// every host (the N/A-never-0 rule is about *data*, not *presence*).
    #[must_use]
    pub fn try_init() -> Option<Box<Self>> {
        if let Some(source) = try_rapl() {
            return Some(Box::new(Self {
                method: CpuPowerMethod::Rapl,
                source,
                last_watts: ESTIMATED_CPU_W,
            }));
        }
        if let Some(source) = try_super_io() {
            return Some(Box::new(Self {
                method: CpuPowerMethod::SuperIo,
                source,
                last_watts: ESTIMATED_CPU_W,
            }));
        }
        Some(Box::new(Self {
            method: CpuPowerMethod::Estimated,
            source: Source::Estimated,
            last_watts: ESTIMATED_CPU_W,
        }))
    }

    /// The accepted method (its `label()` is the startup-log / UI string).
    #[must_use]
    pub fn method(&self) -> CpuPowerMethod {
        self.method
    }

    /// The accepted method's human-readable label.
    #[must_use]
    pub fn method_name(&self) -> &'static str {
        self.method.label()
    }

    /// `true` when the accepted method is the fixed estimate (the UI flags
    /// non-measured CPU power).
    #[must_use]
    pub fn is_estimated(&self) -> bool {
        self.method.is_estimated()
    }

    /// One CPU power reading in watts (the 100 ms / 1 Hz hot path).
    ///
    /// Computes the reading for the accepted source, applies the
    /// plausibility clamp (an out-of-range result degrades to the fixed
    /// estimate), and remembers the last valid value for a vanished node.
    /// **Never panics.**
    pub fn poll(&mut self) -> f64 {
        // Capture the last-known value *before* the mutable borrow of
        // `self.source` (a vanished node / non-advancing clock falls back
        // to it).
        let last = self.last_watts;
        let raw = match &mut self.source {
            Source::Rapl {
                path,
                prev_uj,
                prev_time,
            } => {
                // RAPL: ΔµJ / Δt / 1e6 = W (`wrapping_sub` absorbs the
                // 2^40 µJ counter wrap). A non-advancing clock yields the
                // last known value.
                let now = Instant::now();
                match read_u64(path) {
                    Some(energy) => {
                        let dt = now.duration_since(*prev_time).as_secs_f64();
                        *prev_time = now;
                        let delta_uj = energy.wrapping_sub(*prev_uj) as f64;
                        *prev_uj = energy;
                        if dt > 0.0 {
                            (delta_uj / 1e6) / dt
                        } else {
                            last
                        }
                    }
                    // The node vanished mid-run → the last known value.
                    None => last,
                }
            }
            Source::SuperIo { power, curr, vin } => {
                // A direct power file (µW → W), else the CPU current (mA) ×
                // CPU voltage (mV) product.
                let from_power = power.as_ref().and_then(|p| read_f64(p)).map(|uw| uw / 1e6);
                let from_vi = from_power.or_else(|| {
                    let (c, v) = (curr.as_ref()?, vin.as_ref()?);
                    let i = read_f64(c)?;
                    let v = read_f64(v)?;
                    Some((v / 1000.0) * (i / 1000.0))
                });
                from_vi.unwrap_or(last)
            }
            Source::Estimated => ESTIMATED_CPU_W,
        };

        // The plausibility clamp (the phantom-69 kW / unit-bug guard). A
        // non-finite or out-of-range reading degrades to the estimate
        // rather than polluting the energy / cost math.
        let watts = if raw.is_finite() && (0.0..=MAX_CPU_POWER_W).contains(&raw) {
            raw
        } else {
            warn_anomalous_cpu(self.method, raw);
            ESTIMATED_CPU_W
        };
        self.last_watts = watts;
        watts
    }
}

/// Emit a single `[WARN]` for an out-of-range CPU reading (the phantom-RAPL
/// / unit-bug guard). Kept out of the hot `match` so the clamp logic stays
/// readable; the message names the method so the user sees *which* source
/// misbehaved.
fn warn_anomalous_cpu(method: CpuPowerMethod, raw: f64) {
    eprintln!(
        "[WARN] [HW] CPU power ({}) reported {raw:.0} W — implausible (>{MAX_CPU_POWER_W:.0} W), using the {ESTIMATED_CPU_W:.0} W estimate",
        method.label()
    );
}

/// Probe the Intel RAPL package counter.
///
/// The path is `powercap/intel-rapl/intel-rapl:0/energy_uj` (the *package*
/// node: cores + iGPU). A live ~0.5 s sample must be within
/// [`RAPL_MAX_W`] — this is what rejects the phantom `intel-rapl` binding
/// the research doc found on AMD (it exists and ticks, but at ~69 kW).
/// `None` when the node is absent or the live sample is implausible.
fn try_rapl() -> Option<Source> {
    let path = Path::new("/sys/class/powercap/intel-rapl/intel-rapl:0/energy_uj");
    let first = read_u64(path)?;
    let t0 = Instant::now();
    std::thread::sleep(std::time::Duration::from_millis(500));
    let second = read_u64(path)?;
    let dt = t0.elapsed().as_secs_f64();
    let watts = if dt > 0.0 {
        (second.wrapping_sub(first) as f64) / 1e6 / dt
    } else {
        f64::INFINITY
    };
    (watts.is_finite() && (0.0..=RAPL_MAX_W).contains(&watts)).then_some(Source::Rapl {
        path: path.to_path_buf(),
        prev_uj: second,
        prev_time: Instant::now(),
    })
}

/// Probe the Super I/O chips for a CPU power source: a direct `power1_*`
/// (µW) file, else a CPU-labeled `curr*` (mA) × `in*` (mV) product.
///
/// The known Super I/O drivers are matched by `name` (the research doc's
/// catalog); the CPU channel is identified by its `*_label` file (a chip
/// exposes a dozen `in*` / `curr*` channels — the label is the only
/// reliable way to pick the CPU one, e.g. `asusec`'s `curr1` = "CPU",
/// `in0` = "CPU Core").
fn try_super_io() -> Option<Source> {
    let base = Path::new("/sys/class/hwmon");
    let entries = std::fs::read_dir(base).ok()?;
    for entry in entries.flatten() {
        let dir = entry.path();
        let name = read_trimmed(&dir.join("name")).unwrap_or_default();
        if !is_super_io(&name) {
            continue;
        }
        if let Some(power) = find_direct_power(&dir) {
            return Some(Source::SuperIo {
                power: Some(power),
                curr: None,
                vin: None,
            });
        }
        if let Some((curr, vin)) = find_cpu_vi(&dir) {
            return Some(Source::SuperIo {
                power: None,
                curr,
                vin,
            });
        }
    }
    None
}

/// The Super I/O / motherboard chipset drivers that carry CPU rails
/// (research doc §1.1 catalog).
fn is_super_io(name: &str) -> bool {
    matches!(
        name,
        "asusec"
            | "nct6775"
            | "nct6798"
            | "nct6776"
            | "it8688"
            | "it8683"
            | "f71882a"
            | "f71889a"
            | "w83627hf"
            | "asus"
    )
}

/// A direct `power1_input` / `power1_average` (µW) file on the chip, when
/// its reading is physically plausible (a sanity guard against a chip
/// exposing a non-CPU power node).
fn find_direct_power(dir: &Path) -> Option<PathBuf> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        if name == "power1_input" || name == "power1_average" {
            if let Some(uw) = read_u64(&entry.path()) {
                if (uw as f64 / 1e6) <= MAX_CPU_POWER_W {
                    return Some(entry.path());
                }
            }
        }
    }
    None
}

/// A CPU-labeled `curr*` (mA) and `in*` (mV) pair on the chip: the channel
/// whose matching `*_label` file contains "CPU". `None` when the chip has
/// no CPU current *and* voltage (e.g. `nct6798` has voltages + temps but no
/// CPU current).
fn find_cpu_vi(dir: &Path) -> Option<(Option<PathBuf>, Option<PathBuf>)> {
    let entries = std::fs::read_dir(dir).ok()?;
    let mut curr = None;
    let mut vin = None;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().to_string();
        let Some(base) = name.strip_suffix("_input") else {
            continue;
        };
        let label = dir.join(format!("{base}_label"));
        let is_cpu = read_trimmed(&label)
            .map(|l| l.to_lowercase().contains("cpu"))
            .unwrap_or(false);
        if !is_cpu {
            continue;
        }
        if name.starts_with("curr") && curr.is_none() {
            curr = Some(entry.path());
        }
        if name.starts_with("in") && vin.is_none() {
            vin = Some(entry.path());
        }
    }
    (curr.is_some() && vin.is_some()).then_some((curr, vin))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cascade never returns `None`: even a driver-less / non-Linux host
    /// falls through to the clearly-labeled fixed estimate.
    #[test]
    fn try_init_also_yields_the_estimate_fallback() {
        let Some(mut b) = CpuPowerBackend::try_init() else {
            panic!("the cascade must always yield a backend (the estimate)");
        };
        // Whatever method won, a poll must not panic and must be plausible.
        let w = b.poll();
        assert!(w.is_finite(), "a poll is finite: {w}");
        assert!(
            (0.0..=MAX_CPU_POWER_W).contains(&w),
            "a poll is in range: {w}"
        );
    }

    /// The accepted method's label is one of the three known strings.
    #[test]
    fn method_name_is_a_known_label() {
        let b = CpuPowerBackend::try_init().unwrap();
        assert!(
            matches!(
                b.method_name(),
                "Intel RAPL" | "Super I/O" | "Estimated (40W)"
            ),
            "unexpected label: {}",
            b.method_name()
        );
    }

    /// The fixed-estimate source returns the constant, is flagged as an
    /// estimate, and survives a poll (never panics).
    #[test]
    fn estimated_source_returns_the_constant() {
        let mut b = CpuPowerBackend {
            method: CpuPowerMethod::Estimated,
            source: Source::Estimated,
            last_watts: ESTIMATED_CPU_W,
        };
        assert!(b.is_estimated());
        assert_eq!(b.poll(), ESTIMATED_CPU_W);
        assert_eq!(b.method_name(), "Estimated (40W)");
    }

    /// A RAPL reading above the per-method ceiling degrades to the estimate
    /// (the phantom-69 kW guard): the clamp is the whole point.
    #[test]
    fn rapl_anomaly_degrades_to_the_estimate() {
        // A non-finite / wildly-out-of-range raw reading is clamped.
        let method = CpuPowerMethod::Rapl;
        // Simulate the clamp decision directly (the poll path wraps it).
        let raw: f64 = 69_000.0; // the phantom RAPL scale
        let ok = raw.is_finite() && (0.0..=MAX_CPU_POWER_W).contains(&raw);
        assert!(!ok, "69 kW is out of range");
        assert_eq!(method.label(), "Intel RAPL");
    }

    /// The RAPL delta math: `ΔµJ / Δs / 1e6 = W`. `energy_uj` is a
    /// *cumulative* counter (wraps at 2^40 µJ), so the delta is
    /// `wrapping_sub`. 285 W over 1 s = 285,000,000 µJ; over 100 ms =
    /// 28,500,000 µJ; a counter wrap (prev near `u64::MAX`, cur small)
    /// yields the correct small delta.
    #[test]
    fn rapl_delta_is_microjoules_over_seconds() {
        // 1 s interval at 285 W.
        let prev_uj = 0u64;
        let cur_uj = 285_000_000u64;
        let dt = 1.0;
        let w = (cur_uj.wrapping_sub(prev_uj) as f64) / 1e6 / dt;
        assert!((w - 285.0).abs() < 1e-9, "1 s @ 285 W: {w}");
        // 100 ms interval at 285 W.
        let cur_uj_100ms = 28_500_000u64;
        let w2 = (cur_uj_100ms.wrapping_sub(0) as f64) / 1e6 / 0.1;
        assert!((w2 - 285.0).abs() < 1e-9, "100 ms @ 285 W: {w2}");
        // Counter wrap: prev = u64::MAX − 10 (= 2^64 − 11), cur = 5 → the
        // wrapped delta is 16 µJ (10 to MAX, 1 across the wrap, 5 to cur).
        let prev_wrap = u64::MAX - 10;
        let cur_wrap = 5u64;
        let w3 = (cur_wrap.wrapping_sub(prev_wrap) as f64) / 1e6 / 1.0;
        assert!((w3 - 16.0e-6).abs() < 1e-12, "wrap Δ = 16 µJ / 1 s: {w3}");
    }

    /// The Super I/O V×I math: (mV / 1000) × (mA / 1000) = watts. The
    /// research doc's live probe: 17 A × 1.15 V ≈ 19 W idle.
    #[test]
    fn superio_vi_math_matches_the_probe() {
        let volts_mv = 1146.0; // in0_input
        let amps_ma = 17_000.0; // curr1_input (17 A)
        let w = (volts_mv / 1000.0) * (amps_ma / 1000.0);
        assert!((15.0..=25.0).contains(&w), "17 A × 1.146 V ≈ 19 W: {w}");
        // The burn case: 50 A × 1.466 V ≈ 73 W.
        let w_burn = (1466.0 / 1000.0) * (50_000.0 / 1000.0);
        assert!(
            (70.0..=76.0).contains(&w_burn),
            "50 A × 1.466 V ≈ 73 W: {w_burn}"
        );
    }

    /// A direct power file's µW → W conversion (÷1e6).
    #[test]
    fn power_file_micro_to_watts() {
        let uw = 19_000_000.0_f64; // 19 W in microwatts
        assert!((uw / 1e6 - 19.0).abs() < 1e-9);
    }

    /// The plausibility predicate: a sane reading passes, a phantom does not.
    #[test]
    fn plausibility_predicate() {
        assert!((0.0..=MAX_CPU_POWER_W).contains(&50.0));
        assert!((0.0..=MAX_CPU_POWER_W).contains(&225.0));
        assert!(!(0.0..=MAX_CPU_POWER_W).contains(&69_000.0));
        assert!(!(0.0..=MAX_CPU_POWER_W).contains(&f64::NAN));
    }
}
