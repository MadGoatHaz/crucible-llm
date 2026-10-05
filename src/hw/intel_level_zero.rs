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
//! **Multi-GPU:** `try_init` enumerates **every** driver's **every**
//! device (`zesDriverGet` → `zesDeviceGet`, both stable core API —
//! research "Level-Zero-Multi-GPU-Device-Names": a driver reports
//! `pCount=2` for a dual-GPU box) and resolves the telemetry component
//! handles **per device**, each with its own cumulative-counter baselines.
//! [`GpuBackend::poll_all`] yields one [`GpuSample`] per GPU;
//! [`GpuBackend::poll`] is their aggregate roll-up. Per-device names come
//! from the sysfs `device/label` of each Intel card (paired by index —
//! the same source the sysfs fallback backend uses); the version-sensitive
//! `zesDeviceGetProperties` struct is deliberately **not** bound.
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
use std::sync::Mutex;

use libloading::Library;

use super::intel::intel_card_labels;
use super::{clamp_gpu_power, GpuBackend, GpuSample};

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

/// Two-pass `zesDriverGet` (no device argument) — **all** drivers.
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

/// Two-pass `zesDeviceGet` for one driver — **all** of its devices
/// (multi-GPU: a dual-Arc box reports two).
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

// ── The backend ───────────────────────────────────────────────────────────

/// One Level Zero device and its resolved telemetry component handles.
///
/// Each device carries its own cumulative-counter baselines
/// (`prev_energy` / `prev_engine`) so the per-GPU delta math
/// (µJ/µs → W, active-fraction → %) is independent per card.
struct ZeDevice {
    /// Display name (sysfs `device/label` paired by index, else
    /// `"Intel GPU {i}"`).
    name: String,
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
    /// "Power anomaly already warned" latch (a stuck unit-bug reading must
    /// not spam the console every 100 ms poll).
    power_warned: Mutex<bool>,
}

impl fmt::Debug for ZeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ZeDevice")
            .field("name", &self.name)
            .field("power", &self.pwr.is_some())
            .field("engine", &self.engine.is_some())
            .field("memory", &self.mem.is_some())
            .field("gpu_freq", &self.gpu_freq.is_some())
            .field("mem_freq", &self.mem_freq.is_some())
            .field("temp", &self.temp.is_some())
            .finish()
    }
}

impl ZeDevice {
    /// `true` when at least one telemetry component resolved (a device
    /// with none is dropped — the N/A rule).
    fn has_telemetry(&self) -> bool {
        self.pwr.is_some()
            || self.engine.is_some()
            || self.mem.is_some()
            || self.gpu_freq.is_some()
            || self.mem_freq.is_some()
            || self.temp.is_some()
    }
}

/// The Intel Level Zero Sysman GPU backend — **all** Intel GPUs the
/// loader reports, one [`GpuSample`] each.
///
/// `poll` is `&self` (the [`GpuBackend`] contract) with interior
/// mutability for the cumulative-counter baselines; the `Mutex`es are
/// uncontended in practice (single poller) and a poisoned lock degrades
/// the field to `None` instead of panicking.
pub struct IntelLevelZeroBackend {
    /// Keeps `libze_loader.so` mapped for the backend's lifetime.
    _lib: Library,
    fns: SysmanFns,
    /// Every discovered device with telemetry (≥1 component each).
    devices: Vec<ZeDevice>,
}

impl fmt::Debug for IntelLevelZeroBackend {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("IntelLevelZeroBackend")
            .field("devices", &self.devices)
            .finish()
    }
}

/// Resolve one device's telemetry component handles (power, engine,
/// memory, GPU + memory frequency domains, temperature sensor) — the
/// per-device half of the old single-device init.
fn resolve_device(fns: &SysmanFns, device: ZePtr, name: String) -> ZeDevice {
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

    ZeDevice {
        name,
        pwr,
        engine,
        mem,
        gpu_freq,
        mem_freq,
        temp,
        prev_energy: Mutex::new(prev_energy),
        prev_engine: Mutex::new(prev_engine),
        power_warned: Mutex::new(false),
    }
}

impl IntelLevelZeroBackend {
    /// Load the Level Zero loader, initialize Sysman (`zesInit(0)` —
    /// research §3: the modern explicit init, no `ZES_ENABLE_SYSMAN` env),
    /// discover **every** driver's **every** device, and resolve the
    /// telemetry component handles per device.
    ///
    /// Returns `None` — never panics — when any of: the loader library
    /// is absent, a required symbol is missing, `zesInit` fails, or no
    /// Sysman driver / device with usable telemetry is discovered.
    pub fn try_init() -> Option<Box<Self>> {
        let lib = load_library()?;
        let fns = load_symbols(&lib)?;

        unsafe {
            // Explicit Sysman initialization (research §3).
            if (fns.init)(0) != ZE_RESULT_SUCCESS {
                return None;
            }
        }

        // All drivers → all devices (multi-GPU: `zesDeviceGet` returns
        // every GPU the driver exposes, e.g. pCount=2 on a dual-Arc box).
        let drivers = unsafe { enumerate_drivers(fns.driver_get) };
        let labels = intel_card_labels(); // sysfs names, paired by index
        let mut index = 0usize;
        let mut devices = Vec::new();
        for driver in drivers {
            for device in unsafe { enumerate_devices(fns.device_get, driver) } {
                let name = labels
                    .get(index)
                    .cloned()
                    .unwrap_or_else(|| format!("Intel GPU {index}"));
                index += 1;
                let dev = resolve_device(&fns, device, name);
                if dev.has_telemetry() {
                    devices.push(dev);
                }
            }
        }

        // No device with usable telemetry → the sysfs fallback
        // (detect_intel_backend) is used instead.
        if devices.is_empty() {
            return None;
        }

        Some(Box::new(Self {
            _lib: lib,
            fns,
            devices,
        }))
    }
}

/// Read one device's full telemetry sample (all fields optional, N/A
/// rule) — the per-GPU half of the old single-device `poll`.
fn poll_device(fns: &SysmanFns, dev: &ZeDevice) -> GpuSample {
    let mut sample = GpuSample::default();

    // Power: ΔµJ / Δµs ≡ W (research §1).
    if let Some(pwr) = dev.pwr {
        let mut cur = EnergyCounter::default();
        if unsafe { (fns.power_energy)(pwr, &mut cur as *mut EnergyCounter as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            if let Ok(mut guard) = dev.prev_energy.lock() {
                // Explicit `deref_mut()` (not `&*`): a write through the
                // guard needs `&mut`, and the method call avoids
                // clippy::explicit_auto_deref on the read.
                let prev = guard.deref_mut();
                sample.power_watts = power_watts(prev, &cur);
                *prev = cur;
            }
            // Plausibility clamp (the Sysman energy counter is µJ/µs ≡ W; a
            // unit/scale bug would show as a 5 kW+ reading).
            let mut warned = dev.power_warned.lock().unwrap_or_else(|e| e.into_inner());
            sample.power_watts = clamp_gpu_power(
                sample.power_watts,
                &format!("Intel GPU ({})", dev.name),
                &mut warned,
            );
        }
    }

    // Utilization: ΔactiveTime / Δtimestamp × 100 (research §1).
    if let Some(eng) = dev.engine {
        let mut cur = EngineStats::default();
        if unsafe { (fns.engine_activity)(eng, &mut cur as *mut EngineStats as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            if let Ok(mut guard) = dev.prev_engine.lock() {
                let prev = guard.deref_mut();
                sample.utilization_pct = utilization_pct(prev, &cur);
                *prev = cur;
            }
        }
    }

    // VRAM: used = size − free, bytes → MB (research §1).
    if let Some(mem) = dev.mem {
        let mut state = MemState {
            stype: STYPE_MEM_STATE,
            ..MemState::default()
        };
        if unsafe { (fns.mem_state)(mem, &mut state as *mut MemState as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            sample.memory_total_mb = Some(state.size / (1024 * 1024));
            sample.memory_used_mb = Some(state.size.saturating_sub(state.free) / (1024 * 1024));
        }
    }

    // Core clock + throttle reasons: GPU frequency domain (research §1).
    if let Some(freq) = dev.gpu_freq {
        let mut state = FreqState {
            stype: STYPE_FREQ_STATE,
            ..FreqState::default()
        };
        if unsafe { (fns.freq_state)(freq, &mut state as *mut FreqState as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            sample.core_clock_mhz = Some(state.actual.round() as u32);
            sample.throttle_reasons = format_throttle(state.throttle_reasons);
        }
    }

    // Memory clock: MEMORY frequency domain (research §3 domain enum).
    if let Some(freq) = dev.mem_freq {
        let mut state = FreqState {
            stype: STYPE_FREQ_STATE,
            ..FreqState::default()
        };
        if unsafe { (fns.freq_state)(freq, &mut state as *mut FreqState as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            sample.memory_clock_mhz = Some(state.actual.round() as u32);
        }
    }

    // Temperature: °C (research §1).
    if let Some(temp) = dev.temp {
        let mut val: f64 = 0.0;
        if unsafe { (fns.temp_state)(temp, &mut val as *mut f64 as *mut c_void) }
            == ZE_RESULT_SUCCESS
        {
            sample.temperature_c = Some(val.round() as i32);
        }
    }

    sample
}

impl GpuBackend for IntelLevelZeroBackend {
    fn vendor(&self) -> &str {
        "Intel"
    }

    /// The first device's name: `gpu_display_name` composes the panel
    /// title from it; the per-device table shows every GPU individually
    /// via [`Self::device_names`].
    fn model(&self) -> Option<&str> {
        self.devices.first().map(|d| d.name.as_str())
    }

    fn poll(&self) -> GpuSample {
        GpuSample::aggregate(&self.poll_all())
    }

    /// One sample per Intel GPU (the multi-GPU table's source).
    fn poll_all(&self) -> Vec<GpuSample> {
        self.devices
            .iter()
            .map(|d| poll_device(&self.fns, d))
            .collect()
    }

    /// The display name of each GPU (parallel to [`Self::poll_all`]).
    fn device_names(&self) -> Vec<String> {
        self.devices.iter().map(|d| d.name.clone()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `try_init` must never panic: `None` on a machine without the Level
    /// Zero loader / Intel Compute Runtime (the sysfs fallback path), or a
    /// working backend (whose `poll` and `poll_all` also never panic) when
    /// one is present.
    #[test]
    fn try_init_is_graceful() {
        let result = IntelLevelZeroBackend::try_init();
        if let Some(backend) = &result {
            assert_eq!(backend.vendor(), "Intel");
            assert!(!backend.devices.is_empty(), "a backend holds ≥1 device");
            let _sample = backend.poll(); // must not panic
            let all = backend.poll_all();
            assert_eq!(all.len(), backend.devices.len(), "one sample per GPU");
            assert_eq!(
                all.len(),
                backend.device_names().len(),
                "names parallel to samples"
            );
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

    /// A device with no resolved components is dropped (the N/A rule) —
    /// the backend keeps only devices that can report something.
    #[test]
    fn device_without_telemetry_is_dropped() {
        let bare = ZeDevice {
            name: "none".into(),
            pwr: None,
            engine: None,
            mem: None,
            gpu_freq: None,
            mem_freq: None,
            temp: None,
            prev_energy: Mutex::new(EnergyCounter::default()),
            prev_engine: Mutex::new(EngineStats::default()),
            power_warned: Mutex::new(false),
        };
        assert!(!bare.has_telemetry());
    }
}
