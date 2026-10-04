//! Intel GPU telemetry via the Level Zero Sysman (`zes*`) API, dynamically
//! loaded from `libze_loader.so.1` (the oneAPI Level Zero ICD loader).
//!
//! This is the **preferred** Intel backend: it delivers the full metric set
//! for discrete Arc GPUs — power (W), engine utilization (%), VRAM
//! (used / total), core + memory clocks (MHz), temperature (°C), and
//! throttle reasons — through in-process FFI calls with no subprocess
//! overhead (research "Intel Level Zero Sysman Guide" §7: ~0.2 % of one
//! core at 10 Hz, NVML-class).
//!
//! **Dynamic loading, no link dependency:** the loader library and every
//! `zes*` symbol are resolved at runtime via `libloading` (research §2:
//! target the SONAME `libze_loader.so.1`). On a machine without the
//! Intel Compute Runtime (`intel-level-zero-gpu` / `intel-compute-runtime`)
//! or the loader, [`IntelLevelZeroBackend::try_init`] returns `None` —
//! never a panic — and the caller falls back to
//! [`super::intel::IntelSysfsBackend`] (see
//! [`super::intel::detect_intel_backend`]).
//!
//! **Metric derivation (research §1):**
//! * **Power** — `zesPowerGetEnergyCounter` returns a *cumulative*
//!   energy counter (µJ) plus a monotonic timestamp (µs); watts =
//!   ΔµJ / Δµs across consecutive samples (µJ/µs ≡ W;
//!   `wrapping_sub` absorbs 64-bit counter wrap).
//! * **Utilization** — `zesEngineGetActivity` on an engine group whose
//!   `type` is `ZES_ENGINE_GROUP_ALL` / `_COMPUTE_ALL`; % =
//!   ΔactiveTime / Δtimestamp × 100 (fraction of wall-clock time the
//!   engines were active).
//! * **VRAM** — `zesMemoryGetState` on the first memory module:
//!   used = `size` − `free` (bytes → MB).
//! * **Clocks** — `zesFrequencyGetState` on the `ZES_FREQ_DOMAIN_GPU`
//!   domain (core clock) and `ZES_FREQ_DOMAIN_MEMORY` domain (memory
//!   clock); `actual` is in MHz.
//! * **Temperature** — `zesTemperatureGetState` on the
//!   `ZES_TEMP_SENSORS_GPU` sensor (falling back to `_GLOBAL`), °C.
//! * **Throttle** — the `throttleReasons` bitmask of
//!   `zes_freq_state_t` (power / thermal / sw-range / hw-range flags).
//!
//! Every failure path degrades per-field to `None` (the N/A rule);
//! [`GpuBackend::poll`] never panics.

use std::ffi::c_void;
use std::fmt;
use std::ops::DerefMut;
use std::path::PathBuf;
use std::sync::Mutex;

use libloading::Library;

use super::sysfs::read_trimmed;
use super::{GpuBackend, GpuSample};

// ── Level Zero result codes (research §3) ─────────────────────────────────
const ZE_RESULT_SUCCESS: i32 = 0;
const ZE_RESULT_NOT_SUPPORTED: i32 = 0x102;
const ZE_RESULT_NOT_FOUND: i32 = 0x101;

// ── Structure-type tags for the extensible preamble (research §3) ─────────
const STYPE_ENGINE_PROPERTIES: u32 = 0x5;
const STYPE_FREQ_PROPERTIES: u32 = 0x9;
const STYPE_TEMP_PROPERTIES: u32 = 0x13;
const STYPE_FREQ_STATE: u32 = 0x1a;
const STYPE_MEM_STATE: u32 = 0x1d;

// ── Engine groups / frequency domains / temperature sensors (research §3) ─
const ENGINE_GROUP_ALL: u32 = 0;
const ENGINE_GROUP_COMPUTE_ALL: u32 = 1;
const FREQ_DOMAIN_GPU: u32 = 0;
const FREQ_DOMAIN_MEMORY: u32 = 1;
const TEMP_SENSOR_GLOBAL: u32 = 0;
const TEMP_SENSOR_GPU: u32 = 1;

// ── Throttle-reason bitmask flags (research §3) ───────────────────────────
const THROTTLE_AVE_PWR_CAP: u32 = 1 << 0;
const THROTTLE_BURST_PWR_CAP: u32 = 1 << 1;
const THROTTLE_CURRENT_LIMIT: u32 = 1 << 2;
const THROTTLE_THERMAL_LIMIT: u32 = 1 << 3;
const THROTTLE_PSU_ALERT: u32 = 1 << 4;
const THROTTLE_SW_RANGE: u32 = 1 << 5;
const THROTTLE_HW_RANGE: u32 = 1 << 6;

/// Loader shared objects to try in order (research §2: target the SONAME
/// `libze_loader.so.1`; the plain names cover distros that ship a
/// versionless symlink).
const LIB_CANDIDATES: &[&str] = &["libze_loader.so.1", "libze_loader.so", "ze_loader"];

// ── Opaque FFI types ──────────────────────────────────────────────────────
/// All Sysman handles are opaque pointers (research §3). A raw
/// `*mut c_void` is `!Send + !Sync`, but a handle is a pure address owned
/// by the loaded driver: safe to share across threads for the lifetime of
/// the backend, which keeps the `Library` mapped.
/// `#[repr(transparent)]`: the newtype has exactly the pointer's layout,
/// so it is FFI-safe in the `extern "C"` fn-item types below.
#[repr(transparent)]
#[derive(Copy, Clone)]
struct ZeHandle(*mut c_void);

unsafe impl Send for ZeHandle {}
unsafe impl Sync for ZeHandle {}

impl fmt::Debug for ZeHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("ZeHandle").field(&self.0).finish()
    }
}

type ZePtr = ZeHandle;

type FnZesInit = unsafe extern "C" fn(flags: u32) -> i32;
type FnZesDriverGet = unsafe extern "C" fn(p_count: *mut u32, p_drivers: *mut ZePtr) -> i32;
type FnZesDeviceGet =
    unsafe extern "C" fn(h_driver: ZePtr, p_count: *mut u32, p_devices: *mut ZePtr) -> i32;
/// The five `zesDeviceEnum*` component queries share one signature.
type FnZesEnum =
    unsafe extern "C" fn(h_device: ZePtr, p_count: *mut u32, p_handles: *mut ZePtr) -> i32;
/// The `zes*GetProperties` / `zes*GetState` / `zes*GetActivity` /
/// `zesPowerGetEnergyCounter` queries share one signature (the output
/// struct is caller-owned).
type FnZesGet = unsafe extern "C" fn(h: ZePtr, p_out: *mut c_void) -> i32;

// ── Telemetry structures, field-for-field per research §3 ─────────────────

/// `zes_power_energy_counter_t` — cumulative energy in microjoules plus a
/// monotonic hardware timestamp in microseconds.
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct EnergyCounter {
    energy: u64,
    timestamp: u64,
}

/// `zes_engine_properties_t` — the preamble must carry the structure type.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct EngineProperties {
    stype: u32,
    p_next: *mut c_void,
    engine_type: u32,
    on_subdevice: u32,
    subdevice_id: u32,
}

impl Default for EngineProperties {
    fn default() -> Self {
        Self {
            stype: 0,
            p_next: std::ptr::null_mut(),
            engine_type: 0,
            on_subdevice: 0,
            subdevice_id: 0,
        }
    }
}

/// `zes_engine_stats_t` — cumulative active time + timestamp (µs).
#[repr(C)]
#[derive(Debug, Default, Copy, Clone)]
struct EngineStats {
    active_time: u64,
    timestamp: u64,
}

/// `zes_mem_state_t` — health + free/total bytes.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct MemState {
    stype: u32,
    p_next: *const c_void,
    health: u32,
    free: u64,
    size: u64,
}

impl Default for MemState {
    fn default() -> Self {
        Self {
            stype: 0,
            p_next: std::ptr::null(),
            health: 0,
            free: 0,
            size: 0,
        }
    }
}

/// `zes_freq_properties_t` — identifies GPU vs MEMORY clock domains.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct FreqProperties {
    stype: u32,
    p_next: *mut c_void,
    domain_type: u32,
    on_subdevice: u32,
    subdevice_id: u32,
    min: f64,
    max: f64,
    can_control: u32,
}

impl Default for FreqProperties {
    fn default() -> Self {
        Self {
            stype: 0,
            p_next: std::ptr::null_mut(),
            domain_type: 0,
            on_subdevice: 0,
            subdevice_id: 0,
            min: 0.0,
            max: 0.0,
            can_control: 0,
        }
    }
}

/// `zes_freq_state_t` — `actual` is the operating clock in MHz;
/// `throttle_reasons` is the hardware throttle bitmask.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct FreqState {
    stype: u32,
    p_next: *const c_void,
    current_voltage: f64,
    request: f64,
    tdp: f64,
    efficient: f64,
    actual: f64,
    throttle_reasons: u32,
}

impl Default for FreqState {
    fn default() -> Self {
        Self {
            stype: 0,
            p_next: std::ptr::null(),
            current_voltage: 0.0,
            request: 0.0,
            tdp: 0.0,
            efficient: 0.0,
            actual: 0.0,
            throttle_reasons: 0,
        }
    }
}

/// `zes_temp_properties_t` — GLOBAL / GPU / MEMORY sensor types.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct TempProperties {
    stype: u32,
    p_next: *mut c_void,
    sensor_type: u32,
    on_subdevice: u32,
    subdevice_id: u32,
}

impl Default for TempProperties {
    fn default() -> Self {
        Self {
            stype: 0,
            p_next: std::ptr::null_mut(),
            sensor_type: 0,
            on_subdevice: 0,
            subdevice_id: 0,
        }
    }
}

// ── Pure metric math (unit-testable without any hardware) ─────────────────

/// Instantaneous power in watts from two cumulative energy-counter
/// samples: Δmicrojoules / Δmicroseconds ≡ watts (research §1). `None`
/// when the timestamp did not advance (no valid interval).
fn power_watts(prev: &EnergyCounter, cur: &EnergyCounter) -> Option<f64> {
    let d_energy = cur.energy.wrapping_sub(prev.energy) as f64;
    let d_time = cur.timestamp.wrapping_sub(prev.timestamp) as f64;
    (d_time > 0.0).then(|| d_energy / d_time)
}

/// Engine utilization, % (0–100) from two activity samples: the fraction
/// of the wall-clock window the engines spent active (research §1).
/// `None` when the timestamp did not advance.
fn utilization_pct(prev: &EngineStats, cur: &EngineStats) -> Option<u8> {
    let d_active = cur.active_time.wrapping_sub(prev.active_time) as f64;
    let d_time = cur.timestamp.wrapping_sub(prev.timestamp) as f64;
    (d_time > 0.0).then(|| (d_active / d_time * 100.0).clamp(0.0, 100.0) as u8)
}

/// Human-readable throttle labels from the `throttleReasons` bitmask
/// (research §3): the power-cap bits collapse to one `"power"` label,
/// then thermal / software-range / hardware-range. `None` when no flag
/// is set.
fn format_throttle(flags: u32) -> Option<String> {
    let mut names: Vec<&str> = Vec::new();
    if flags
        & (THROTTLE_AVE_PWR_CAP
            | THROTTLE_BURST_PWR_CAP
            | THROTTLE_CURRENT_LIMIT
            | THROTTLE_PSU_ALERT)
        != 0
    {
        names.push("power");
    }
    if flags & THROTTLE_THERMAL_LIMIT != 0 {
        names.push("thermal");
    }
    if flags & THROTTLE_SW_RANGE != 0 {
        names.push("sw_range");
    }
    if flags & THROTTLE_HW_RANGE != 0 {
        names.push("hw_range");
    }
    (!names.is_empty()).then(|| names.join(","))
}

// ── Loader / symbol plumbing ──────────────────────────────────────────────

/// `dlopen` the first candidate that loads (research §2). `None` when the
/// Level Zero loader is not installed — the graceful-degradation entry.
fn load_library() -> Option<Library> {
    LIB_CANDIDATES
        .iter()
        .find_map(|name| unsafe { Library::new(name) }.ok())
}

/// Resolve one `Copy` symbol (a function pointer) from the loaded library.
fn get_sym<T: Copy>(lib: &Library, name: &[u8]) -> Option<T> {
    let sym = unsafe { lib.get::<T>(name) }.ok()?;
    Some(*sym)
}

/// Every `zes*` entry point we use (research §3 signatures).
#[derive(Copy, Clone)]
struct SysmanFns {
    init: FnZesInit,
    driver_get: FnZesDriverGet,
    device_get: FnZesDeviceGet,
    enum_power: FnZesEnum,
    power_energy: FnZesGet,
    enum_engine: FnZesEnum,
    engine_props: FnZesGet,
    engine_activity: FnZesGet,
    enum_mem: FnZesEnum,
    mem_state: FnZesGet,
    enum_freq: FnZesEnum,
    freq_props: FnZesGet,
    freq_state: FnZesGet,
    enum_temp: FnZesEnum,
    temp_props: FnZesGet,
    temp_state: FnZesGet,
}

/// Resolve all 16 symbols; `None` if the library is missing any of them
/// (e.g. a loader built without Sysman) — graceful, never a panic.
fn load_symbols(lib: &Library) -> Option<SysmanFns> {
    Some(SysmanFns {
        init: get_sym(lib, b"zesInit")?,
        driver_get: get_sym(lib, b"zesDriverGet")?,
        device_get: get_sym(lib, b"zesDeviceGet")?,
        enum_power: get_sym(lib, b"zesDeviceEnumPowerDomains")?,
        power_energy: get_sym(lib, b"zesPowerGetEnergyCounter")?,
        enum_engine: get_sym(lib, b"zesDeviceEnumEngineGroups")?,
        engine_props: get_sym(lib, b"zesEngineGetProperties")?,
        engine_activity: get_sym(lib, b"zesEngineGetActivity")?,
        enum_mem: get_sym(lib, b"zesDeviceEnumMemoryModules")?,
        mem_state: get_sym(lib, b"zesMemoryGetState")?,
        enum_freq: get_sym(lib, b"zesDeviceEnumFrequencyDomains")?,
        freq_props: get_sym(lib, b"zesFrequencyGetProperties")?,
        freq_state: get_sym(lib, b"zesFrequencyGetState")?,
        enum_temp: get_sym(lib, b"zesDeviceEnumTemperatureSensors")?,
        temp_props: get_sym(lib, b"zesTemperatureGetProperties")?,
        temp_state: get_sym(lib, b"zesTemperatureGetState")?,
    })
}

/// Two-pass component enumeration (research §3): pass `NULL` to get the
/// count, then allocate and fill. `NOT_SUPPORTED` / other errors / zero
/// count → empty vector (the N/A rule).
unsafe fn enumerate(f: FnZesEnum, device: ZePtr) -> Vec<ZePtr> {
    let mut count: u32 = 0;
    match f(device, &mut count, std::ptr::null_mut()) {
        ZE_RESULT_SUCCESS => {}
        // Some drivers report the count via NOT_FOUND for a NULL buffer.
        ZE_RESULT_NOT_FOUND if count > 0 => {}
        // NOT_SUPPORTED → the device exposes no such components (N/A rule).
        ZE_RESULT_NOT_SUPPORTED => return Vec::new(),
        _ => return Vec::new(),
    }
    if count == 0 {
        return Vec::new();
    }
    let mut handles = vec![ZeHandle(std::ptr::null_mut()); count as usize];
    match f(device, &mut count, handles.as_mut_ptr()) {
        ZE_RESULT_SUCCESS => {}
        _ => return Vec::new(),
    }
    handles.truncate((count as usize).min(handles.len()));
    handles
}

/// Two-pass `zesDriverGet` (no device argument).
unsafe fn enumerate_drivers(f: FnZesDriverGet) -> Vec<ZePtr> {
    let mut count: u32 = 0;
    if f(&mut count, std::ptr::null_mut()) != ZE_RESULT_SUCCESS || count == 0 {
        return Vec::new();
    }
    let mut handles = vec![ZeHandle(std::ptr::null_mut()); count as usize];
    if f(&mut count, handles.as_mut_ptr()) != ZE_RESULT_SUCCESS {
        return Vec::new();
    }
    handles.truncate((count as usize).min(handles.len()));
    handles
}

/// Two-pass `zesDeviceGet` for one driver.
unsafe fn enumerate_devices(f: FnZesDeviceGet, driver: ZePtr) -> Vec<ZePtr> {
    let mut count: u32 = 0;
    if f(driver, &mut count, std::ptr::null_mut()) != ZE_RESULT_SUCCESS || count == 0 {
        return Vec::new();
    }
    let mut handles = vec![ZeHandle(std::ptr::null_mut()); count as usize];
    if f(driver, &mut count, handles.as_mut_ptr()) != ZE_RESULT_SUCCESS {
        return Vec::new();
    }
    handles.truncate((count as usize).min(handles.len()));
    handles
}

/// The Intel card's marketing label from sysfs (e.g. "Intel Arc A770"),
/// so [`GpuBackend::model`] matches the sysfs backend's naming. `None`
/// when no Intel DRM card is present (the N/A rule).
fn intel_card_label() -> Option<String> {
    let drm = PathBuf::from("/sys/class/drm");
    let cards = std::fs::read_dir(&drm).ok()?;
    for entry in cards.flatten() {
        let card = entry.path();
        if read_trimmed(&card.join("device/vendor")).as_deref() != Some("0x8086") {
            continue;
        }
        if let Some(label) = read_trimmed(&card.join("device/label")) {
            return Some(label);
        }
    }
    None
}

// ── The backend ───────────────────────────────────────────────────────────

/// The Intel Level Zero Sysman GPU backend (discrete Arc first-class).
///
/// `poll` is `&self` (the [`GpuBackend`] contract) with interior
/// mutability for the cumulative-counter baselines; the `Mutex`es are
/// uncontended in practice (single poller) and a poisoned lock degrades
/// the field to `None` instead of panicking.
pub struct IntelLevelZeroBackend {
    /// Keeps `libze_loader.so` mapped for the backend's lifetime.
    _lib: Library,
    fns: SysmanFns,
    /// Marketing name from sysfs (e.g. "Intel Arc A770"), if readable.
    model: Option<String>,
    // Resolved component handles (each `None` degrades its metric).
    pwr: Option<ZePtr>,
    engine: Option<ZePtr>,
    mem: Option<ZePtr>,
    gpu_freq: Option<ZePtr>,
    mem_freq: Option<ZePtr>,
    temp: Option<ZePtr>,
    // Cumulative-counter baselines for delta math.
    prev_energy: Mutex<EnergyCounter>,
    prev_engine: Mutex<EngineStats>,
}

impl fmt::Debug for IntelLevelZeroBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntelLevelZeroBackend")
            .field("model", &self.model)
            .field("power", &self.pwr.is_some())
            .field("engine", &self.engine.is_some())
            .field("memory", &self.mem.is_some())
            .field("gpu_freq", &self.gpu_freq.is_some())
            .field("mem_freq", &self.mem_freq.is_some())
            .field("temp", &self.temp.is_some())
            .finish()
    }
}

impl IntelLevelZeroBackend {
    /// Load the Level Zero loader, initialize Sysman (`zesInit(0)` —
    /// research §3: the modern explicit init, no `ZES_ENABLE_SYSMAN` env),
    /// discover the first driver + device, and resolve the telemetry
    /// component handles.
    ///
    /// Returns `None` — never panics — when any of: the loader library
    /// is absent, a required symbol is missing, `zesInit` fails, or no
    /// Sysman driver / device is discovered. A backend with no usable
    /// telemetry components is also rejected so the sysfs fallback is
    /// used instead.
    pub fn try_init() -> Option<Box<Self>> {
        let lib = load_library()?;
        let fns = load_symbols(&lib)?;

        let device = unsafe {
            // Explicit Sysman initialization (research §3).
            if (fns.init)(0) != ZE_RESULT_SUCCESS {
                return None;
            }
            let driver = enumerate_drivers(fns.driver_get).into_iter().next()?;
            enumerate_devices(fns.device_get, driver)
                .into_iter()
                .next()?
        };

        // Power domain: the first one (single board rail on consumer Arc).
        let pwr = unsafe { enumerate(fns.enum_power, device) }
            .into_iter()
            .next();

        // Engine group: prefer `ALL`, then `COMPUTE_ALL` (research §1).
        let engine = unsafe {
            let mut chosen = None;
            for h in enumerate(fns.enum_engine, device) {
                let mut props = EngineProperties {
                    stype: STYPE_ENGINE_PROPERTIES,
                    ..EngineProperties::default()
                };
                if (fns.engine_props)(h, &mut props as *mut EngineProperties as *mut c_void)
                    == ZE_RESULT_SUCCESS
                    && (props.engine_type == ENGINE_GROUP_ALL
                        || props.engine_type == ENGINE_GROUP_COMPUTE_ALL)
                {
                    chosen = Some(h);
                    break;
                }
            }
            chosen
        };

        // Memory module: the first one (single GDDR6 domain on consumer Arc).
        let mem = unsafe { enumerate(fns.enum_mem, device) }
            .into_iter()
            .next();

        // Frequency domains: GPU (core clock) + MEMORY (memory clock).
        let (gpu_freq, mem_freq) = unsafe {
            let mut gpu = None;
            let mut memd = None;
            for h in enumerate(fns.enum_freq, device) {
                let mut props = FreqProperties {
                    stype: STYPE_FREQ_PROPERTIES,
                    ..FreqProperties::default()
                };
                if (fns.freq_props)(h, &mut props as *mut FreqProperties as *mut c_void)
                    == ZE_RESULT_SUCCESS
                {
                    match props.domain_type {
                        FREQ_DOMAIN_GPU => gpu = Some(h),
                        FREQ_DOMAIN_MEMORY => memd = Some(h),
                        _ => {}
                    }
                    if gpu.is_some() && memd.is_some() {
                        break;
                    }
                }
            }
            (gpu, memd)
        };

        // Temperature sensor: prefer the GPU sensor, fall back to GLOBAL.
        let temp = unsafe {
            let mut gpu_sensor = None;
            let mut global_sensor = None;
            for h in enumerate(fns.enum_temp, device) {
                let mut props = TempProperties {
                    stype: STYPE_TEMP_PROPERTIES,
                    ..TempProperties::default()
                };
                if (fns.temp_props)(h, &mut props as *mut TempProperties as *mut c_void)
                    == ZE_RESULT_SUCCESS
                {
                    match props.sensor_type {
                        TEMP_SENSOR_GPU => gpu_sensor = Some(h),
                        TEMP_SENSOR_GLOBAL if global_sensor.is_none() => global_sensor = Some(h),
                        _ => {}
                    }
                    if gpu_sensor.is_some() {
                        break;
                    }
                }
            }
            gpu_sensor.or(global_sensor)
        };

        // A backend with no usable telemetry is useless — reject it so
        // the sysfs fallback (detect_intel_backend) is used.
        if pwr.is_none()
            && engine.is_none()
            && mem.is_none()
            && gpu_freq.is_none()
            && temp.is_none()
        {
            return None;
        }

        // Seed the cumulative-counter baselines (research §4: delta math
        // starts from the first successful read).
        let mut prev_energy = EnergyCounter::default();
        if let Some(p) = pwr {
            unsafe {
                (fns.power_energy)(p, &mut prev_energy as *mut EnergyCounter as *mut c_void);
            }
        }
        let mut prev_engine = EngineStats::default();
        if let Some(e) = engine {
            unsafe {
                (fns.engine_activity)(e, &mut prev_engine as *mut EngineStats as *mut c_void);
            }
        }

        Some(Box::new(Self {
            _lib: lib,
            fns,
            model: intel_card_label(),
            pwr,
            engine,
            mem,
            gpu_freq,
            mem_freq,
            temp,
            prev_energy: Mutex::new(prev_energy),
            prev_engine: Mutex::new(prev_engine),
        }))
    }
}

impl GpuBackend for IntelLevelZeroBackend {
    fn vendor(&self) -> &str {
        "Intel"
    }

    fn model(&self) -> Option<&str> {
        self.model.as_deref()
    }

    fn poll(&self) -> GpuSample {
        let mut sample = GpuSample::default();

        // Power: ΔµJ / Δµs ≡ W (research §1).
        if let Some(pwr) = self.pwr {
            let mut cur = EnergyCounter::default();
            if unsafe {
                (self.fns.power_energy)(pwr, &mut cur as *mut EnergyCounter as *mut c_void)
            } == ZE_RESULT_SUCCESS
            {
                if let Ok(mut guard) = self.prev_energy.lock() {
                    // Explicit `deref_mut()` (not `&*`): a write through the
                    // guard needs `&mut`, and the method call avoids
                    // clippy::explicit_auto_deref on the read.
                    let prev = guard.deref_mut();
                    sample.power_watts = power_watts(prev, &cur);
                    *prev = cur;
                }
            }
        }

        // Utilization: ΔactiveTime / Δtimestamp × 100 (research §1).
        if let Some(eng) = self.engine {
            let mut cur = EngineStats::default();
            if unsafe {
                (self.fns.engine_activity)(eng, &mut cur as *mut EngineStats as *mut c_void)
            } == ZE_RESULT_SUCCESS
            {
                if let Ok(mut guard) = self.prev_engine.lock() {
                    let prev = guard.deref_mut();
                    sample.utilization_pct = utilization_pct(prev, &cur);
                    *prev = cur;
                }
            }
        }

        // VRAM: used = size − free, bytes → MB (research §1).
        if let Some(mem) = self.mem {
            let mut state = MemState {
                stype: STYPE_MEM_STATE,
                ..MemState::default()
            };
            if unsafe { (self.fns.mem_state)(mem, &mut state as *mut MemState as *mut c_void) }
                == ZE_RESULT_SUCCESS
            {
                sample.memory_total_mb = Some(state.size / (1024 * 1024));
                sample.memory_used_mb = Some(state.size.saturating_sub(state.free) / (1024 * 1024));
            }
        }

        // Core clock + throttle reasons: GPU frequency domain (research §1).
        if let Some(freq) = self.gpu_freq {
            let mut state = FreqState {
                stype: STYPE_FREQ_STATE,
                ..FreqState::default()
            };
            if unsafe { (self.fns.freq_state)(freq, &mut state as *mut FreqState as *mut c_void) }
                == ZE_RESULT_SUCCESS
            {
                sample.core_clock_mhz = Some(state.actual.round() as u32);
                sample.throttle_reasons = format_throttle(state.throttle_reasons);
            }
        }

        // Memory clock: MEMORY frequency domain (research §3 domain enum).
        if let Some(freq) = self.mem_freq {
            let mut state = FreqState {
                stype: STYPE_FREQ_STATE,
                ..FreqState::default()
            };
            if unsafe { (self.fns.freq_state)(freq, &mut state as *mut FreqState as *mut c_void) }
                == ZE_RESULT_SUCCESS
            {
                sample.memory_clock_mhz = Some(state.actual.round() as u32);
            }
        }

        // Temperature: °C (research §1).
        if let Some(temp) = self.temp {
            let mut val: f64 = 0.0;
            if unsafe { (self.fns.temp_state)(temp, &mut val as *mut f64 as *mut c_void) }
                == ZE_RESULT_SUCCESS
            {
                sample.temperature_c = Some(val.round() as i32);
            }
        }

        sample
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_init` must never panic: `None` on a machine without the Level
    /// Zero loader / Intel Compute Runtime (the sysfs fallback path), or a
    /// working backend (whose `poll` also never panics) when one is present.
    #[test]
    fn try_init_is_graceful() {
        let result = IntelLevelZeroBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "Intel");
            let _sample = backend.poll(); // must not panic
        }
    }

    /// A missing library must yield `None`, never a panic or a link
    /// error (the same code path `load_library` takes for real names).
    #[test]
    fn missing_loader_is_none() {
        // A bogus name can never resolve.
        let result = unsafe { Library::new("libze_loader_definitely_not_present.so.1") }.ok();
        assert!(result.is_none());
    }

    /// 285 W over 1 s = 285 000 000 µJ / 1 000 000 µs (research §1: µJ/µs ≡ W).
    #[test]
    fn power_is_microjoules_over_microseconds() {
        let prev = EnergyCounter {
            energy: 0,
            timestamp: 0,
        };
        let cur = EnergyCounter {
            energy: 285_000_000,
            timestamp: 1_000_000,
        };
        assert!((power_watts(&prev, &cur).unwrap() - 285.0).abs() < 1e-9);
    }

    /// A non-advancing timestamp is N/A, never a divide-by-zero.
    #[test]
    fn power_is_na_on_zero_interval() {
        let prev = EnergyCounter {
            energy: 100,
            timestamp: 50,
        };
        let cur = EnergyCounter {
            energy: 200,
            timestamp: 50,
        };
        assert_eq!(power_watts(&prev, &cur), None);
    }

    /// Half the window active = 50 % (research §1).
    #[test]
    fn utilization_is_the_active_fraction() {
        let prev = EngineStats {
            active_time: 0,
            timestamp: 0,
        };
        let cur = EngineStats {
            active_time: 500_000,
            timestamp: 1_000_000,
        };
        assert_eq!(utilization_pct(&prev, &cur), Some(50));
    }

    /// A degenerate sample (more active time than elapsed) clamps to 100.
    #[test]
    fn utilization_clamps_to_hundred() {
        let prev = EngineStats {
            active_time: 0,
            timestamp: 0,
        };
        let cur = EngineStats {
            active_time: 2_000_000,
            timestamp: 1_000_000,
        };
        assert_eq!(utilization_pct(&prev, &cur), Some(100));
    }

    /// The 64-bit counters wrap: `wrapping_sub` keeps the delta correct
    /// (research §1: implementations must account for counter wrapping).
    #[test]
    fn counter_wrap_is_handled() {
        let prev = EngineStats {
            active_time: u64::MAX - 4, // 2^64 − 5
            timestamp: 100,
        };
        let cur = EngineStats {
            active_time: 5,
            timestamp: 200,
        };
        // Δactive = 5 − (2^64 − 5) mod 2^64 = 10; Δtime = 100 → 10 %.
        assert_eq!(utilization_pct(&prev, &cur), Some(10));
    }

    /// Throttle bitmask → readable labels (research §3 flags): power bits
    /// dedupe to one label, zero flags → `None`.
    #[test]
    fn throttle_flags_map_to_readable_names() {
        assert_eq!(format_throttle(0), None);
        assert_eq!(
            format_throttle(THROTTLE_THERMAL_LIMIT),
            Some("thermal".to_string())
        );
        assert_eq!(
            format_throttle(THROTTLE_AVE_PWR_CAP | THROTTLE_BURST_PWR_CAP | THROTTLE_PSU_ALERT),
            Some("power".to_string())
        );
        assert_eq!(
            format_throttle(THROTTLE_THERMAL_LIMIT | THROTTLE_HW_RANGE),
            Some("thermal,hw_range".to_string())
        );
    }
}
