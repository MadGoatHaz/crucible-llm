# **Architecture and Implementation Blueprint for Linux Non-NVIDIA GPU Telemetry**

Hardware monitoring on Linux has historically been centered on NVIDIA’s proprietary ecosystem through the NVIDIA Management Library (NVML) and its command-line interface, nvidia-smi1. When engineering vendor-agnostic infrastructure, telemetry pollers face a fragmented landscape across AMD and Intel accelerators. NVIDIA unifies power, thermals, clocks, memory, and engine utilization behind a single user-space dynamic library (libnvidia-ml.so), whereas AMD and Intel split their monitoring surfaces across user-space runtimes, direct Linux kernel file systems (sysfs, hwmon), and kernel profiling interfaces (DRM fdinfo, perf PMU counters)2. Designing a native, cross-vendor Rust hardware poller requires analyzing the operational mechanics of the AMD amdgpu driver, the Intel i915 and xe drivers, and cross-vendor Linux kernel standards.

## **AMD GPU Telemetry Ecosystem**

### **User-Space Telemetry: ROCm SMI and AMDSMI Evolution**

AMD’s official telemetry offerings have undergone a multi-year architectural transition5. The legacy tool, rocm-smi (historically packaged under rocm-smi-lib), is a Python CLI wrapper built over an underlying C shared library (librocm\_smi64.so)6. It exposes socket power consumption in watts, engine and memory clock frequencies in megahertz, memory allocation (VRAM and GTT) in megabytes, temperatures across multiple physical diodes (edge, junction, and memory), and engine utilization percentages1. The tool exposes these metrics programmatically through standard flags such as rocm-smi \--showuse \--showmemuse \--showpower \--showclocks \--showtemp \--json7.  
However, invoking rocm-smi as an external subprocess incurs substantial overhead. The Python interpreter startup, dynamic linking of librocm\_smi64.so, and repetitive parsing of standard input introduce latencies between 80 ms and 250 ms per invocation. Furthermore, between ROCm minor versions, the JSON schema of rocm-smi has exhibited schema instability, altering key identifiers and data types across releases9.  
AMD has deprecated rocm-smi-lib in favor of AMD SMI (AMDSMI), backed by libamd\_smi.so6. AMD SMI standardizes system management across compute accelerators such as the Instinct MI-series and client Radeon GPUs6. Its modern CLI binary, amd-smi, provides structured JSON outputs via amd-smi metric \--json5.  
The underlying C API (amdsmi/amdsmi.h) provides fine-grained, DCGM-like programmatic querying through functions such as amdsmi\_get\_power\_info(), amdsmi\_get\_gpu\_activity(), amdsmi\_get\_clock\_info(), and amdsmi\_get\_vram\_usage(). Official Python bindings and community wrappers like pyrsmi wrap this native C shared object6.  
In the Rust ecosystem, AMD does not publish an official, vendor-maintained equivalent to nvml-wrapper. While third-party crates such as amdsmi-sys exist, they are frequently unmaintained or tied to specific ROCm distributions. Generating low-level FFI bindings directly via bindgen against /opt/rocm/include/amd\_smi/amdsmi.h requires the full ROCm development stack to be present on the host compilation and deployment environments. This requirement limits deployment portability across standard consumer distributions, bare-metal container images, or lightweight edge nodes lacking ROCm runtime packages6.

### **Direct Kernel Subsystems: amdgpu Sysfs and hwmon Architecture**

The most resilient and performant mechanism for querying AMD GPU telemetry on Linux bypasses user-space ROCm shared libraries entirely, reading directly from the Linux kernel’s native sysfs and hwmon driver nodes3. The in-tree amdgpu kernel module exposes hardware state registers via standard virtual file paths rooted at /sys/class/drm/card{N}/device/4.  
GPU compute utilization is accessible through /sys/class/drm/card{N}/device/gpu\_busy\_percent, which returns an ASCII-encoded integer from 0 to 1007. The System Management Unit (SMU) coprocessor periodically samples whether internal GPU compute blocks—including the Release-from-Compute, MicroEngine Compute, Pipe Prefetch, and Command Processor blocks—are processing command buffers, calculating a time-averaged utilization ratio4. The memory controller activity is similarly exposed via /sys/class/drm/card{N}/device/mem\_busy\_percent4.  
Regarding power consumption, AMD GPUs do not populate a top-level power\_consumption file directly in the device directory. Instead, power metrics are routed through the standard Linux hardware monitoring (hwmon) subsystem14. The path is resolved via the symbolic link /sys/class/drm/card{N}/device/hwmon/hwmon{M}/power1\_average, or power1\_input on select APU designs14. This node reports the real-time graphics package or socket power draw in microwatts (![][image1])8. Converting this integer value to watts requires dividing by ![][image2].  
Video memory footprint data is exposed via dedicated driver nodes14. The /sys/class/drm/card{N}/device/mem\_info\_vram\_used and /sys/class/drm/card{N}/device/mem\_info\_vram\_total nodes report allocated and total addressable video memory in raw bytes14. System memory allocated to the GPU through the Graphics Translation Table is available at /sys/class/drm/card{N}/device/mem\_info\_gtt\_used7. Converting these values to megabytes requires dividing by ![][image3].  
Core clock frequencies can be parsed via /sys/class/drm/card{N}/device/pp\_dpm\_sclk4. This virtual file outputs available Dynamic Power Management (DPM) states line-by-line, suffixing the currently active state with an asterisk7. Alternatively, modern kernels populate direct hwmon frequency sensors at /sys/class/drm/card{N}/device/hwmon/hwmon{M}/freq1\_input, which provides the instantaneous core clock in hertz (![][image4]), requiring division by ![][image5] to obtain megahertz (![][image6]).  
Thermals are read from /sys/class/drm/card{N}/device/hwmon/hwmon{M}/temp1\_input in millidegrees Celsius (![][image7]), where dividing by ![][image8] yields an integer Celsius reading14. Accompanying files such as temp1\_label denote whether the channel measures the edge diode, the silicon junction, or the memory stack10. Accessing these sysfs nodes requires zero dynamic library dependencies, operates with standard non-root read permissions, and incurs an overhead of less than ![][image9] per read when reusing open file descriptors via standard Linux file I/O.

### **Profiling Tools and High-Level Runtimes: amdgpu\_top and ravn**

The open-source AMD monitoring ecosystem features two notable projects: amdgpu\_top and ravn (in connection with LibreHardwareMonitor)17.  
The amdgpu\_top utility is a monitoring tool written natively in Rust17. It displays engine utilization, sensor telemetry, and process allocation by interfacing directly with low-level kernel APIs: the Graphics Register Bus Manager (GRBM/GRBM2) performance counters, DRM client fdinfo, sysfs, and direct AMDGPU DRM ioctls via libdrm\_amdgpu17. For programmatic consumption, the binary provides continuous JSON streaming via amdgpu\_top \--json \-s 1000 \-n 1 or output to a FIFO pipe via \--json-fifo17.  
The maintainer exposes the core functionality as a library crate published on crates.io under the name libamdgpu\_top20. By importing libamdgpu\_top directly, an application can construct an AMD poller using native Rust structures without invoking CLI subprocesses20. However, because it binds directly to low-level DRM character devices (/dev/dri/card\*) and queries performance counter registers, reading GRBM metrics requires elevated capabilities (CAP\_SYS\_ADMIN or membership in the render or video groups)19. Furthermore, continuous counter reads can inhibit the GPU's dynamic power-saving states19.  
In contrast, ravn (often referenced alongside LibreHardwareMonitor) belongs to the Windows hardware monitoring ecosystem18. LibreHardwareMonitor is an open-source C\#/.NET library and GUI utility designed as the successor to OpenHardwareMonitor18. On Windows, it reads hardware states through a bundled kernel driver (WinRing0) and vendor APIs such as NVAPI, AMD ADL, and Intel IGCL21.  
While LibreHardwareMonitor can run on cross-platform .NET runtimes under Linux, its Linux hardware backend relies on lm-sensors via a managed libsensors C-wrapper18. It lacks support for discrete GPU VRAM allocation tables, engine scheduling rings, or modern DRM compute interfaces on Linux21. Furthermore, LibreHardwareMonitor exposes its programmatic interface solely via a built-in local HTTP web server emitting JSON at /data.json or a /metrics Prometheus endpoint24. Relying on a .NET-based HTTP daemon to monitor AMD hardware from a high-performance Rust poller introduces prohibitive memory and runtime overhead.

| Monitoring Path | Metrics Available | Programmatic Access Method | Rust Crate Availability | Distro & Driver Requirements | Latency & Reliability |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Direct sysfs / hwmon** | Watts, Busy %, MHz, Total/Used MB, °C4 | Direct file reads: /sys/class/drm/card\*/device/ \[cite: 4, 14\] | Pure Rust file I/O (std::fs) | Linux kernel ![][image10]; in-tree amdgpu driver15 | ![][image11]; exceptional stability; zero external C dependencies |
| **libamdgpu\_top** | Watts, Engine % (GFX, Media, Compute), MHz, MB, °C19 | In-process native Rust library API20 | libamdgpu\_top (crates.io)20 | Linux kernel ![][image12]; requires libdrm development headers | ![][image13]; high fidelity; can disable GPU power gating if GRBM counters are polled continuously19 |
| **amd-smi (AMDSMI)** | Watts, % (SMU load), MHz, MB, °C, PCIe BW, RAS errors5 | Dynamic library linking (libamd\_smi.so) or CLI JSON5 | Custom bindgen FFI wrapper required | ROCm ![][image14] stack installed; RDNA/CDNA enterprise hardware6 | ![][image15] via C API; CLI subprocess adds ![][image16] overhead |
| **rocm-smi (Legacy)** | Watts, % (Busy), MHz, MB, °C, Fan %7 | Subprocess execution: rocm-smi \--json \[cite: 8\] | rocm-smi-sys (community, unmaintained) | ROCm stack installed; deprecated by AMD6 | ![][image16] per sample; fragile JSON schema across ROCm releases9 |
| **LibreHardwareMonitor** | Temperatures, Core Watts, Clocks (minimal VRAM/Engine metrics)24 | HTTP GET request to embedded web server (/data.json)25 | None; requires HTTP client (reqwest) | .NET runtime installed; Linux lm-sensors installed18 | ![][image17]; poor metric completeness on Linux21 |

## **Intel GPU Telemetry Stack (Discrete Arc and Integrated)**

### **Architectural Divergence: i915 versus Xe Kernel Drivers**

Evaluating Intel GPU telemetry on modern Linux requires accounting for the kernel architectural transition from the legacy i915 driver to the modern xe driver27. Historically, all Intel graphics—spanning early integrated chips through Gen12 (Iris Xe) and initial discrete Arc Alchemist (DG2) cards—were managed by the monolithic i915.ko driver3. Recognizing that legacy register infrastructure impeded scalability, Intel developed the ground-up xe.ko driver27.  
The xe driver was stabilized in the Linux 6.8 kernel series and became the default driver for newer architectures such as Lunar Lake, Arrow Lake, and Arc Battlemage (BMG/Xe2)3. It is also selectively supported for discrete Arc Alchemist hardware27. Because i915 and xe organize virtual file systems and performance monitoring units differently, a robust telemetry collector must detect which driver binds the target PCI device node3.

### **Programmatic Telemetry via Intel Level Zero Sysman API**

The architectural equivalent of NVIDIA’s NVML for Intel silicon is the Level Zero Systems Management (Sysman) API29. Sysman is a sub-specification of the Intel oneAPI Level Zero runtime designed specifically to expose low-level node management and hardware metrics across integrated (iGPU) and discrete (Arc, Flex, Max) architectures29.  
The Sysman API provides access to the necessary telemetry domains through a structured C interface29:

* zesDeviceGetPowerProperties() and zesDeviceGetPowerEnergyCounter() return instantaneous power consumption in milliwatts (![][image18]) or continuous energy counters in microjoules (![][image19]) across the GPU package and individual memory subsystems.  
* zesDeviceGetEngineActivity() returns per-engine physical utilization records (activeTime versus timestamp), permitting the exact calculation of compute, copy, and video engine utilization percentages over any chosen polling delta.  
* zesDeviceGetMemoryState() returns local memory (VRAM/LMEM) allocation profiles, providing total and free memory in bytes.  
* zesDeviceGetFrequencyState() yields instantaneous actual and requested clock frequencies in megahertz (![][image6]).  
* zesDeviceGetTemperatureState() returns thermal readings across global GPU sensors, compute tiles, and memory arrays in degrees Celsius (°C).

Level Zero is implemented in user space by libze\_loader.so (the Level Zero Loader), which dynamically dispatches calls to the underlying hardware driver backend (libze\_intel\_gpu.so)29. In the Rust ecosystem, crates such as all-smi leverage Level Zero Sysman directly by loading libze\_loader.so dynamically at runtime via libloading29. By utilizing dynamic loading, the binary avoids hard compilation dependencies on Intel oneAPI packages, initializing the Sysman backend only when the driver library is discovered on the target system29.

### **Profiling and Counter Scraping: intel\_gpu\_top and PMU Events**

The standard command-line utility for monitoring Intel graphics is intel\_gpu\_top, part of the intel-gpu-tools (IGT) suite31. Rather than reading purely from sysfs, intel\_gpu\_top measures GPU load via Linux Performance Monitoring Unit (PMU) perf events exposed by the kernel driver31.  
When executed with the \-J flag, intel\_gpu\_top formats its output as structured JSON via intel\_gpu\_top \-J \-s 100031. This yields real-time streaming telemetry covering engine clocks (actual and requested) in megahertz, percentage busy time for individual engine classes (Render/3D, Blitter, Video, VideoEnhance), real-time wattage across the graphics package and socket via Running Average Power Limit (RAPL) MSR interfaces, and read/write bus memory bandwidth in megabytes per second3.  
However, incorporating intel\_gpu\_top \-J into an automated Rust polling pipeline presents operational hurdles:

* **JSON Stream Formatting**: intel\_gpu\_top continuously streams comma-delimited JSON objects designed to represent an infinite array, but it fails to emit the closing square bracket until terminated with SIGINT33. If the process is halted prematurely, standard parsers fail with syntax errors unless the calling application actively buffers stdout and wraps the payload in brackets \[...\]32.  
* **Kernel Privilege Constraints**: Because it opens kernel perf event file descriptors via perf\_event\_open(), unprivileged execution fails unless the administrator lowers the kernel profiling barrier (sysctl \-w kernel.perf\_event\_paranoid=1 or lower) or grants the binary explicit capabilities: setcap cap\_perfmon,cap\_sys\_ptrace=ep \$(which intel\_gpu\_top)31.  
* **Xe Driver Compatibility**: Historically, intel\_gpu\_top supported only the i915 kernel driver PMU27. Intel's newer xe driver did not expose equivalent PMU performance counter hooks until Linux 6.163. On kernels operating discrete Arc cards under xe prior to 6.16, running intel\_gpu\_top frequently results in blank engine outputs or process initialization failures27.

### **Intel Sysfs Hierarchies and Tooling Clarifications**

For environments where running external binaries or linking Level Zero is impractical, direct file reads from sysfs provide lightweight access to clocks, power, and temperatures, though utilization metrics require alternative handling3.  
Under the **i915** driver architecture, instantaneous core frequency is read from /sys/class/drm/card{N}/gt\_act\_freq\_mhz (or /sys/class/drm/card{N}/gt/gt0/rps\_act\_freq\_mhz) in megahertz36. Discrete Arc GPUs expose standard hwmon nodes under their PCI device tree at /sys/class/drm/card{N}/device/hwmon/hwmon{M}/energy1\_input or power1\_average3. The energy1\_input node provides a cumulative microjoule counter; calculating watts requires computing the delta between two reads divided by elapsed time (![][image20])38. For integrated Intel GPUs, power is not exposed under the DRM device node; instead, it is found in the CPU package RAPL powercap interface at /sys/class/powercap/intel-rapl/intel-rapl:0/intel-rapl:0:1/energy\_uj. Discrete Arc cards expose local memory boundaries through driver-specific interfaces, but there is no universal mem\_info\_vram\_used sysfs file matching AMD’s implementation3. Crucially, the i915 kernel driver omits a scalar gpu\_busy\_percent file in sysfs3. Determining GPU load without Level Zero requires sampling DRM client engine busyness via /proc/\[pid\]/fdinfo/ or attaching to the PMU3.  
Under the **xe** driver architecture, engine frequency resides under the tile hierarchy at /sys/class/drm/card{N}/device/tile0/gt0/freq0/act\_freq in megahertz39. Power is mapped through standard hwmon at /sys/class/drm/card{N}/device/hwmon/hwmon{M}/power1\_input (or power1\_average) in microwatts3. GPU utilization is similarly absent from sysfs under xe and must be derived via DRM fdinfo or Level Zero Sysman3.  
Additional diagnostic tools often cited in this domain serve distinct operational functions:

* **xperf** is a Windows-specific CLI component of the Microsoft Windows Performance Toolkit (part of the Windows SDK)40. It controls Event Tracing for Windows (ETW) sessions and operates strictly on Windows kernel traces (.etl files)40. It has no operational standing on Linux and cannot monitor Linux GPU performance41.  
* **xpu-smi (Intel XPU Manager / XPUM)** is Intel's enterprise command-line utility designed for server environments (Intel Data Center GPU Flex and Max series)43. It interfaces with Level Zero Sysman to export hardware state, thermal metrics, and power in human-readable and JSON formats43. On consumer Intel Arc cards, however, xpu-smi frequently outputs N/A for engine utilization percentages, as its internal monitoring modules assume multi-tile data-center architectures44.  
* **gpu\_monitor** generally refers to ad-hoc Python integration scripts (such as those maintained in containerized media-server monitoring utilities) that execute intel\_gpu\_top \-J as a background subprocess, parse the output line-by-line, and re-export the data45.

| Monitoring Path | Metrics Available | Programmatic Access Method | Rust Crate Availability | Distro & Driver Requirements | Latency & Reliability |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Level Zero Sysman** | Watts, Engine % (Render, Video), MHz, Total/Used MB, °C29 | C Dynamic Library (libze\_loader.so) via Sysman API29 | all-smi (dynamic loading pattern)29 | Linux kernel ![][image12]; intel-level-zero-gpu package | ![][image21]; exceptional precision across integrated and Arc dGPUs |
| **intel\_gpu\_top** | Watts (RAPL), Engine %, MHz, Memory BW31 | CLI execution with JSON output (-J)31 | None (handled via std::process::Command) | intel-gpu-tools package; i915 driver (limited on xe)27 | High polling overhead (![][image22] sample window); requires elevated perf permissions31 |
| **sysfs \+ hwmon / RAPL** | MHz (clocks), Watts (hwmon/RAPL), °C3 | Direct file reads (gt\_act\_freq\_mhz, power1\_average)3 | Pure Rust file I/O (std::fs) | Linux kernel ![][image12]; in-tree i915 or xe drivers3 | ![][image11]; completely stable, but **lacks** GPU utilization and VRAM MB nodes3 |
| **DRM fdinfo** | Engine Time (ns), Total/Resident Memory (KiB)46 | Direct file reads: /proc/\[pid\]/fdinfo/\* \[cite: 46\] | Custom file parsing in Rust | Linux kernel ![][image12] (standardized DRM fdinfo format) | ![][image23]; requires aggregating all process file descriptors to find total system load |
| **xpu-smi** | Watts, MHz, MB, °C, Tile metrics43 | CLI JSON output or C library linking (libxpum.so)43 | None | Intel Data Center GPUs (Flex/Max); Arc consumer support is partial43 | High CLI latency; frequently reports N/A for utilization on Arc desktop cards44 |

## **Cross-Vendor Telemetry Abstractions**

### **Architectural Bounds of gpustat and Unified hwmon Subsystem**

The Python monitoring utility gpustat does not support AMD or Intel GPUs47. Internally, gpustat is a thin wrapper around nvidia-ml-py, the official Python bindings for NVIDIA's NVML library (libnvidia-ml.so)48. It relies exclusively on NVIDIA-specific C symbols, such as nvmlDeviceGetUtilizationRates, nvmlDeviceGetMemoryInfo, and nvmlDeviceGetTemperature48. It contains no abstraction layer, fallback parser, or plugin mechanism for AMD sysfs or Intel Level Zero47. Invoking gpustat on non-NVIDIA systems results in immediate driver initialization exceptions48. Third-party projects with similar names, such as Unraid's gpustat plugin, are distinct shell script collections that invoke vendor-specific CLI tools conditionally based on hardware detection52.  
Linux provides a vendor-neutral hardware monitoring subsystem in /sys/class/hwmon/3. The kernel drivers for AMD (amdgpu), Intel discrete graphics (i915, xe), and the open-source NVIDIA driver (nouveau) all register standard hwmon driver classes3. The hwmon subsystem standardizes power consumption through power{N}\_average or power{N}\_input in microwatts (![][image1]), or cumulative energy counters via energy{N}\_input in microjoules (![][image19])14. It also standardizes temperatures via temp{N}\_input in millidegrees Celsius (![][image7])14.  
While hwmon provides a clean, cross-vendor interface for power and thermal management, its scope is strictly confined to hardware physical sensor lines2. It does not expose compute engine utilization percentages, memory pool usage (VRAM), or execution pipeline states3. Consequently, hwmon can serve as a unified backend for power and thermals, but it must be supplemented with driver-specific mechanisms for compute load and memory telemetry3.

### **Standardized Engine Telemetry via DRM Client fdinfo**

To resolve the historical absence of a unified GPU utilization interface, the upstream Linux kernel introduced the DRM client fdinfo protocol, standardized in Linux kernel 5.1917. Whenever an active user-space process opens a Direct Rendering Manager character device node (such as /dev/dri/card0 or /dev/dri/renderD128), the kernel exposes driver scheduling and allocation statistics inside that process’s virtual file descriptor table at /proc/\[pid\]/fdinfo/\[fd\]17.  
When read, a DRM file descriptor emits structured text metadata including drm-driver, drm-pdev, drm-client-id, engine execution times in nanoseconds (such as drm-engine-gfx or drm-engine-compute), and memory usage entries like drm-memory-vram and drm-memory-gtt in kibibytes46. The exact engine labels vary slightly by vendor—such as drm-engine-render on Intel i915/xe versus drm-engine-gfx on AMD amdgpu—but the syntax and semantics are consistent across modern Linux drivers46.  
To calculate system-wide GPU utilization via fdinfo, the monitoring agent iterates through all process directories in /proc/, inspects the descriptors in /proc/\[pid\]/fdinfo/, and identifies files declaring a drm-driver entry. It records the accumulated engine run times for each unique drm-client-id and, at the subsequent sampling tick (![][image24]), recalculates the elapsed runtime. System-wide utilization is the sum of all active client engine time deltas divided by total wall-clock time multiplied by available engine hardware instances:  
![][image25]  
This mechanism provides a unified interface across AMD, Intel, and modern open-source NVIDIA drivers, tracking per-process VRAM allocation without vendor SDK dependencies3. However, calculating utilization via fdinfo incurs higher overhead (![][image23]) because it must scan numerous process file descriptors on every collection cycle.

### **Multi-Vendor Tooling Landscape: nvtop, all-smi, and silicon-monitor**

Several production monitoring systems provide cross-vendor coverage across NVIDIA, AMD, and Intel hardware:

* **nvtop** is an interactive terminal monitor written in C1. It provides comprehensive cross-vendor GPU support across NVIDIA (NVML), AMD (amdgpu sysfs and DRM fdinfo), and Intel (i915/xe perf PMU and DRM fdinfo)27. nvtop isolates vendor quirks into discrete C extraction units, making its codebase an effective reference implementation for Linux DRM scraping27.  
* **all-smi** is an open-source unified CLI and daemon written natively in Rust29. It monitors heterogeneous compute hardware including NVIDIA, AMD, Intel Arc/Iris, Apple Silicon, Google TPU, and Tenstorrent NPUs30. Its internal architecture employs a modular trait-based poller design30: it utilizes NVML for NVIDIA, sysfs/AMDSMI for AMD, and dynamically binds libze\_loader.so (Level Zero Sysman) for Intel29.  
* **silicon-monitor (simon)** is a Rust-based system monitoring framework featuring a structured hardware ontology58. Its GPU engine exposes a single interface across vendors, utilizing nvml-wrapper for NVIDIA, direct sysfs/DRM for AMD, and i915/xe sysfs/PMU parsers for Intel58.  
* **btop** is a C++ terminal resource monitor that includes multi-vendor GPU telemetry60. It uses NVML for NVIDIA, reads /sys/class/drm/card\*/device/ directly for AMD, and parses Intel GPU frequencies and RAPL powercap interfaces for Intel60.

| Approach / Architecture | Vendors Supported | Metrics Extracted | Performance Overhead | Primary Advantages | Critical Limitations |
| :---- | :---- | :---- | :---- | :---- | :---- |
| **Direct sysfs \+ hwmon** | AMD, Intel (Arc & iGPU), NVIDIA (Nouveau)3 | Watts, MHz, °C (plus MB and % on AMD)4 | Minimal (![][image11]) | Zero external library dependencies; zero elevated capabilities3 | Intel lacks utilization and memory nodes in sysfs; path divergence across drivers3 |
| **Level Zero Sysman** | Intel (Integrated, Arc Discrete, Data Center)29 | Watts, Utilization %, MHz, VRAM MB, °C29 | Low (![][image26]) | Complete programmatic NVML counterpart for Intel; highly stable C ABI29 | Intel-only; requires Intel compute runtime packages on the host29 |
| **DRM fdinfo Scanning** | AMD (amdgpu), Intel (i915, xe), NVIDIA (nouveau)17 | Engine Utilization %, VRAM/GTT Allocation MB17 | Moderate (![][image23]) | Standardized across all modern Linux DRM drivers; accurate per-process attribution46 | Requires traversing /proc/\*/fdinfo/; does not provide power, clocks, or thermals46 |
| **Perf PMU Counters** | Intel (i915, xe ![][image12]), AMD3 | Engine Utilization %, Bus Bandwidth, Package Power31 | Low (![][image27]) | Direct hardware counter sampling without runtime overhead | Requires CAP\_PERFMON or relaxed perf\_event\_paranoid; driver-specific PMU formats31 |
| **all-smi / simon Trait Architecture** | NVIDIA, AMD, Intel, Apple Silicon30 | Comprehensive: Watts, %, MHz, MB, °C30 | Optimized per vendor backend (![][image28]) | Native Rust ecosystem; encapsulates vendor divergence behind a unified API58 | Requires maintaining multiple driver-specific query paths58 |

## **Implementation Strategy for the Rust HwPoller Trait**

### **Core Trait Architecture**

To build a high-performance, non-blocking telemetry engine in Rust, hardware polling should be structured around an abstraction trait whose interface is uniform regardless of whether the underlying target is an NVIDIA H100, an AMD Radeon RX 7900 / Instinct MI300, or an Intel Arc A770.

Rust  
/// Unified hardware polling interface for graphics and compute accelerators.  
pub trait HwPoller: Send \+ Sync {  
    /// Instantaneous power consumption in Watts.  
    fn power\_watts(&self) \-\> Option\<f64\>;

    /// Overall GPU compute engine utilization percentage (0 \- 100).  
    fn utilization\_pct(&self) \-\> Option\<u8\>;

    /// Currently allocated video memory (VRAM / LMEM) in Megabytes.  
    fn memory\_used\_mb(&self) \-\> Option\<u64\>;

    /// Total available video memory (VRAM / LMEM) in Megabytes.  
    fn memory\_total\_mb(&self) \-\> Option\<u64\>;

    /// Instantaneous graphics core clock frequency in Megahertz.  
    fn clock\_mhz(&self) \-\> Option\<u32\>;

    /// Core or hotspot temperature in degrees Celsius.  
    fn temperature\_c(&self) \-\> Option\<i32\>;  
}

### **Backend Strategy and Selection Logic**

A multi-backend hardware monitor must select the most reliable telemetry source per vendor while preserving runtime portability:  
For NVIDIA devices, the backend binds to nvml-wrapper, calling Device::power\_usage(), Device::utilization\_rates(), Device::memory\_info(), Device::clock\_info(), and Device::temperature().  
For AMD devices, the optimal backend is a pure AmdSysfsPoller. Querying /sys/class/drm/card{N}/device/ and its child hwmon directory satisfies all six trait methods natively without requiring the ROCm user-space stack, external C compilers, or elevated root permissions4.  
For Intel devices, the backend uses a dual-path strategy. In environments where the Intel oneAPI compute runtime is installed, an IntelSysmanPoller dynamically binds libze\_loader.so at runtime using libloading, extracting all six metrics via Level Zero Sysman29. In minimal or containerized environments lacking Level Zero, an IntelSysfsPoller fallback reads clocks from gt\_act\_freq\_mhz or act\_freq, power and thermals from the associated hwmon or RAPL nodes, and estimates utilization and memory via DRM fdinfo sweeps3.

### **Reference Implementation: AMD Sysfs and hwmon Backend**

The following standalone Rust module implements the HwPoller trait for AMD GPUs by resolving the sysfs tree and converting raw sensor units:

Rust  
use std::fs::{self, File};  
use std::io::Read;  
use std::path::{Path, PathBuf};

pub struct AmdSysfsPoller {  
    device\_path: PathBuf,  
    hwmon\_path: Option\<PathBuf\>,  
}

impl AmdSysfsPoller {  
    pub fn new(card\_index: usize) \-\> Result\<Self, std::io::Error\> {  
        let device\_path \= PathBuf::from(format\!("/sys/class/drm/card{}/device", card\_index));  
        if \!device\_path.exists() {  
            return Err(std::io::Error::new(  
                std::io::ErrorKind::NotFound,  
                format\!("AMD GPU at {} does not exist", device\_path.display()),  
            ));  
        }

        let hwmon\_dir \= device\_path.join("hwmon");  
        let hwmon\_path \= if hwmon\_dir.exists() {  
            fs::read\_dir(hwmon\_dir)?  
                .filter\_map(|entry| entry.ok())  
                .map(|entry| entry.path())  
                .find(|path| path.is\_dir())  
        } else {  
            None  
        };

        Ok(Self {  
            device\_path,  
            hwmon\_path,  
        })  
    }

    fn read\_trimmed(&self, file\_path: \&Path) \-\> Option\<String\> {  
        let mut file \= File::open(file\_path).ok()?;  
        let mut buffer \= String::with\_capacity(32);  
        file.read\_to\_string(&mut buffer).ok()?;  
        Some(buffer.trim().to\_string())  
    }

    fn read\_u64(&self, file\_path: \&Path) \-\> Option\<u64\> {  
        self.read\_trimmed(file\_path)?.parse::\<u64\>().ok()  
    }  
}

impl HwPoller for AmdSysfsPoller {  
    fn power\_watts(&self) \-\> Option\<f64\> {  
        let hwmon \= self.hwmon\_path.as\_ref()?;  
        let p\_path \= hwmon.join("power1\_average");  
        let p\_path \= if p\_path.exists() { p\_path } else { hwmon.join("power1\_input") };  
          
        let microwatts \= self.read\_u64(\&p\_path)?;  
        Some(microwatts as f64 / 1\_000\_000.0)  
    }

    fn utilization\_pct(&self) \-\> Option\<u8\> {  
        let path \= self.device\_path.join("gpu\_busy\_percent");  
        let val \= self.read\_u64(\&path)?;  
        Some(val.min(100) as u8)  
    }

    fn memory\_used\_mb(&self) \-\> Option\<u64\> {  
        let path \= self.device\_path.join("mem\_info\_vram\_used");  
        let bytes \= self.read\_u64(\&path)?;  
        Some(bytes / (1024 \* 1024))  
    }

    fn memory\_total\_mb(&self) \-\> Option\<u64\> {  
        let path \= self.device\_path.join("mem\_info\_vram\_total");  
        let bytes \= self.read\_u64(\&path)?;  
        Some(bytes / (1024 \* 1024))  
    }

    fn clock\_mhz(&self) \-\> Option\<u32\> {  
        if let Some(hwmon) \= &self.hwmon\_path {  
            let freq\_path \= hwmon.join("freq1\_input");  
            if freq\_path.exists() {  
                if let Some(hz) \= self.read\_u64(\&freq\_path) {  
                    return Some((hz / 1\_000\_000) as u32);  
                }  
            }  
        }

        let sclk\_path \= self.device\_path.join("pp\_dpm\_sclk");  
        let content \= self.read\_trimmed(\&sclk\_path)?;  
        for line in content.lines() {  
            if line.ends\_with('\*') {  
                let parts: Vec\<&str\> \= line.split\_whitespace().collect();  
                if parts.len() \>= 2 {  
                    let freq\_str \= parts\[1\].trim\_end\_matches('\*').trim\_end\_matches("Mhz");  
                    if let Ok(freq) \= freq\_str.parse::\<u32\>() {  
                        return Some(freq);  
                    }  
                }  
            }  
        }  
        None  
    }

    fn temperature\_c(&self) \-\> Option\<i32\> {  
        let hwmon \= self.hwmon\_path.as\_ref()?;  
        let temp\_path \= hwmon.join("temp1\_input");  
        let millidegrees \= self.read\_trimmed(\&temp\_path)?.parse::\<i64\>().ok()?;  
        Some((millidegrees / 1000) as i32)  
    }  
}

### **Reference Implementation: Intel Level Zero / Sysfs Backend**

For Intel platforms, the poller detects driver state (i915 versus xe) and falls back to sysfs and hwmon when Level Zero Sysman is unavailable:

Rust  
use std::fs::{self, File};  
use std::io::Read;  
use std::path::{Path, PathBuf};

pub struct IntelSysfsPoller {  
    card\_index: usize,  
    device\_path: PathBuf,  
    hwmon\_path: Option\<PathBuf\>,  
    is\_xe\_driver: bool,  
}

impl IntelSysfsPoller {  
    pub fn new(card\_index: usize) \-\> Result\<Self, std::io::Error\> {  
        let device\_path \= PathBuf::from(format\!("/sys/class/drm/card{}/device", card\_index));  
        if \!device\_path.exists() {  
            return Err(std::io::Error::new(  
                std::io::ErrorKind::NotFound,  
                format\!("Intel GPU at {} does not exist", device\_path.display()),  
            ));  
        }

        let driver\_link \= device\_path.join("driver");  
        let is\_xe\_driver \= fs::read\_link(\&driver\_link)  
            .map(|p| p.to\_string\_lossy().contains("xe"))  
            .unwrap\_or(false);

        let hwmon\_dir \= device\_path.join("hwmon");  
        let hwmon\_path \= if hwmon\_dir.exists() {  
            fs::read\_dir(hwmon\_dir)?  
                .filter\_map(|e| e.ok())  
                .map(|e| e.path())  
                .find(|p| p.is\_dir())  
        } else {  
            None  
        };

        Ok(Self {  
            card\_index,  
            device\_path,  
            hwmon\_path,  
            is\_xe\_driver,  
        })  
    }

    fn read\_u64(&self, file\_path: \&Path) \-\> Option\<u64\> {  
        let mut file \= File::open(file\_path).ok()?;  
        let mut buffer \= String::with\_capacity(32);  
        file.read\_to\_string(&mut buffer).ok()?;  
        buffer.trim().parse::\<u64\>().ok()  
    }  
}

impl HwPoller for IntelSysfsPoller {  
    fn power\_watts(&self) \-\> Option\<f64\> {  
        if let Some(hwmon) \= &self.hwmon\_path {  
            let p\_avg \= hwmon.join("power1\_average");  
            let p\_path \= if p\_avg.exists() { p\_avg } else { hwmon.join("power1\_input") };  
            if let Some(uw) \= self.read\_u64(\&p\_path) {  
                return Some(uw as f64 / 1\_000\_000.0);  
            }  
        }  
        None  
    }

    fn utilization\_pct(&self) \-\> Option\<u8\> {  
        // Intel sysfs does not expose a static gpu\_busy\_percent node.  
        // Production implementations derive load by sampling aggregated DRM  
        // client execution time from /proc/\[pid\]/fdinfo/\* across sampling intervals.  
        None  
    }

    fn memory\_used\_mb(&self) \-\> Option\<u64\> {  
        let lmem\_used \= self.device\_path.join("lmem\_used\_bytes");  
        if lmem\_used.exists() {  
            return self.read\_u64(\&lmem\_used).map(|b| b / (1024 \* 1024));  
        }  
        None  
    }

    fn memory\_total\_mb(&self) \-\> Option\<u64\> {  
        let lmem\_total \= self.device\_path.join("lmem\_total\_bytes");  
        if lmem\_total.exists() {  
            return self.read\_u64(\&lmem\_total).map(|b| b / (1024 \* 1024));  
        }  
        None  
    }

    fn clock\_mhz(&self) \-\> Option\<u32\> {  
        if self.is\_xe\_driver {  
            let freq\_path \= self.device\_path.join("tile0/gt0/freq0/act\_freq");  
            self.read\_u64(\&freq\_path).map(|f| f as u32)  
        } else {  
            let freq\_path \= PathBuf::from(format\!("/sys/class/drm/card{}/gt\_act\_freq\_mhz", self.card\_index));  
            self.read\_u64(\&freq\_path).map(|f| f as u32)  
        }  
    }

    fn temperature\_c(&self) \-\> Option\<i32\> {  
        let hwmon \= self.hwmon\_path.as\_ref()?;  
        let temp\_path \= hwmon.join("temp1\_input");  
        let millidegrees \= self.read\_u64(\&temp\_path)?;  
        Some((millidegrees / 1000) as i32)  
    }  
}

### **Operational Considerations: Sleep States, Resets, and Normalization**

Deploying these telemetry extractors in production environments requires handling several operational constraints:  
Reading dynamic performance management registers (such as pp\_dpm\_sclk on AMD or hardware performance counters on Intel) forces the driver to wake low-power execution blocks if the GPU is completely idle14. Continuously polling performance counters every ![][image29] can increase baseline idle power consumption by ![][image30] on high-end discrete GPUs. Setting the collection loop interval between ![][image31] and ![][image32] balances telemetry freshness with low-power sleep retention1.  
When a GPU undergoes an internal dynamic reset following a compute ring timeout or firmware crash, sysfs and hwmon file handles are invalidated by the kernel driver10. The HwPoller implementation must avoid panicking on failed reads, treating ENOENT and ENODEV as transient None returns, and re-resolving the base paths when errors persist across multiple cycles16.  
Furthermore, kernel subsystems maintain different conventions for raw sensor units. Power on hwmon is reported in microwatts (![][image1]), whereas Level Zero Sysman reports in milliwatts (![][image18])14. Temperatures are reported in millidegrees Celsius (![][image7]) in hwmon, but as pure integer Celsius (°C) in Sysman14. Frequencies are reported in megahertz (![][image6]) in i915, in hertz (![][image4]) in AMD freq1\_input, and as textual string tables in AMD pp\_dpm\_sclk7. Normalizing these values in the poller backend ensures that downstream consumers receive consistent metrics regardless of hardware platform.

## **Conclusions and Architectural Synthesis**

Implementing a vendor-agnostic HwPoller in Rust requires matching each hardware vendor's telemetry architecture to the most stable, performant Linux kernel and user-space interfaces.  
For AMD hardware, avoiding the overhead and instability of external ROCm CLI tools (rocm-smi, amd-smi) yields significant reliability benefits5. The in-tree Linux amdgpu driver exposes all necessary telemetry directly via standard sysfs and hwmon virtual files4. This approach provides sub-millisecond query performance, requires zero external C libraries, operates without elevated user privileges, and works reliably across consumer Radeon and enterprise Instinct accelerators on kernels ![][image10]6.  
For Intel hardware, a bifurcated backend provides the best balance between metric completeness and system portability. In environments with Intel compute runtimes installed, interfacing dynamically with Level Zero Sysman (libze\_loader.so) provides full parity with NVML, exposing fine-grained energy, utilization, clock, and memory metrics across both integrated and discrete Arc platforms29. In minimal or containerized environments lacking the Level Zero loader, falling back to direct sysfs file parsing (gt\_act\_freq\_mhz for i915, act\_freq for xe) paired with hwmon power nodes ensures basic observability3. Where compute utilization is required without Level Zero, aggregating DRM fdinfo run times across active clients or reading the driver's perf PMU bridges the gap3.  
By encapsulating these vendor-specific access patterns behind a unified HwPoller trait, a Rust application achieves low-overhead, reliable hardware telemetry across all major GPU platforms on Linux.

[image1]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACIAAAAbCAYAAAAZMl2nAAACrUlEQVR4Xs1VTaiOQRQ+I0SUnxvilqxsrFjIxoZCInVLUtfuUhZWlq6U/60bCynlrxTZYUFioSykFDtlc8tCoZQF6XrOnPPOnPl7v+9u8NQz75nnnDPnvDPzvR9RDS551ASDqpjKNr0aXhWpmpE21UrsQ19O4nOxROVthkXaZiW5IkWx6qw0RXQWfKecAX+DzzXkJPhZdeYv8JW4PF573XnffWZZtlQ8/n4js8NKJS/4oDtSHVaTNMd80iXoy/B4AxxPZbVm1VWo6Ivz2/4AF9udw2OaCfMnuDwpRvQwRjZR+kslAr7jJLuyz8RtAO8q2TdufKPgZTHlZRpwKzC893S0yjgm4NMFGGGRdSTFpoJMdAJcq/wK3jHlDsPeFadt7Ae/K+eajl+CZ7pJAkdvMH4wyrVgObpH0sw8FW5imB/8rNiJAb/ZM2WHBSS3f6+flZnnwBnIm/Bcg4DzxncEc96xHeAikotartHNo+7eYrjg6UU/bCPZ/pEQFo+GH1vYj9kpksJbTdioNnIJHCPZ8YFYRlJwj7LDJBbkeyMIDYb22fhE8o24bR0KPrqP4FVwYZSTNRL8N43sJGmEP0jMDo/B6+BSJ8cWEHqSu8W5XCzHRRLfrdxRQPuaJEnYrGSMOdH40z0Be3vxEtJNd48OpE6G83cIPNTagRyPwBfgUyX/9I6Bu5HP/yt8GVvAT52+IG5J7tDifDQjbMdW6kczB/wGx8G626DhtAXSNWxCbuvcyBtJtm99lBiNqi348DzH/iEGqYmjcE43I2zj/SHBzuelLVH5mqfBK3Hag2oj9uxNI3mVBH3t6rTIV7vQqzABrYSKVIj/vpGqsyIXgqAqV8QosZUeadSHQSOuIRfIG4loreD1llMwMKTiizlmR7o422W1z4EVh0CRnlci+gND+3gaWUm2LQAAAABJRU5ErkJggg==>

[image2]: <data:image/png;base64,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>

[image3]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAZCAYAAABNcRIKAAAFeUlEQVR4Xu1YV8heRRCdxZiIxoImdkQhFlQsKGKUGGyg2LCjIopIsFfEAkaQBPFBfbBhgz+KDTsasJIoYonmKUbBqIjY8EVUFHyQ33N2Zu+Wu/d+l7+ID/+B8927M7O7M7P1fiIlnLEQtSUuk2c2vtC26UJnOylMEfWJZWelEmVwZcWyHJDE0WXSi7JSk6ASNdkAjKjW2V0VpWFZHoAhVYbYVK2GRFNTu7q4RN2G0mGzmUjsdgD38DThqDZUz75GWSq6rOpy58WHgmtJlOblBkS9KnA9+Bn4IXg/uGmubuFw8MtQKFq9FHwFfAm8D9zaKC3LiCvBceOP4Fvge+DSpM62iU2NY8HQwIpXge+Itndzru7DTDLHgqFhwsl8AFyN+islNj4v970zkDPAT8EtrXwLbF8XreAr6U9Tfza4HtyQmARcBK4AZ3lzJ8eK98sztlUAsqvx+Mv4O/gaZGcHpWExCl8ofXJAZ0/5GzywsVQ8Ct4tvgV3GJ5/gLvnJv1YIjGZ872k5n0OzshrkvIW4DiqHUE20pi3ZeBHkszMBKvBQwrZB55ODtBicCjb7ZjMM41dYGzHGCOcXI7f5Ulj55AYzJ/xnGNG54kOFPflAjGwMldLMIPGSeeTmRgmb4l0H9HE6ywIcPIbfm9VZj3sLTrid4mfmRFmtQrkrN7KxLPAr43bmKxB4g+WuTtLWWqb4tGiA00G7C+6jGcl5muMY41kAJLemtf2zPSomXocJ97WnRQEpv4JfNgYhBuB3EaYFCTTbUgGJYCD8g/4HcjgnwIvNvbhMvBB46uiq4VJ2ik1quBdcF999Z7sKDH+e0W3HK4ingUL1W4EkoDSZLYPoCJy0f2SticW8h/A540B3Aq4hAifzESXgokLPjAp9CP6EvfeFJeALxhpQaOHRA+hLhwJvl0ExYSFvr+V2C8Gy/Fgs4MwouqNYQl2onES7/PVsM9cThZ27FrJ/F5icMTOkie2WOahD8dTfg2Kp+H5iWhQnD0kl3wXePiVS5irhfUXRxH7aeKhP+UJfZTEZD6SyHc1GW8tUrTTiXRm8ioxCotEbU8p5PkyF3lObMmZC609U/Q8+UZ0HyM2B5+BkPs3+6BvnaiEx4OM9W7ypVy5meh2cnomFdlPYvy3JXLOSMrGEtlIzCRzcsnMeoHDXOJ+mW+XKhq73CnuKQz23FTo9K53nXGOtVcwk10hGjz3SEXs57GEObyNY6AcmOXGAG4Z9O2GIsnE8cJ+XXK3VKNNwD+NSxtdTObjqWOtT9Gs6LKZmSSzseJs3TjKPXiNuLYxcU3HBxuDPMXTUs5MkYNEr0AGF+pwnyJvjLrMDz+g4D3GAF6TKF9YxgzcITqgC0rHRG8DJD9kAnYTTX736mg1ky/z7Qsdr0q/gE8U8vPB90WXDsFPMF4leBUia/08C2GRTMdTmKcvA51tQg7MKmO4e9b84N11T9L64mH1huSHSIoVojHuwkLhHz+pSfoXYloGm/VOZ65ojda8NDi5UPSz6iuJyeQnImULzIqzgV8Fv/oaiQ94uUDUQVw15E3J7qgZ+FnGNpkM9vG58VTTM2EvguvAl0X32r2MHq7tBx9z8bKSFG1/LXindA/m7aJfYJacKk5ARX5ArEP9jyXmYUrxZHwNsUTJf4jox3T1P13tWsNzpXKilRN+tA81i5qsgh4/Jo5K364mbUsytNWUdCaH/6Isqtl4VETdGGHcrzY/EiQu6SP9nTqk7dU/xvqQV/D/AnWkciCG1BxpE/+NkgHWfWiCGdVKXV+XdmEmmYa6vi4lujX9uv8rEp/9xatEKQqJzerF9xayQXDt9iaOqWuJyPycJKammb5W+nQBQ2wmgWISxPdp7nc6wBn8LzvaMIQd8FvCAAAAAElFTkSuQmCC>

[image4]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAZCAYAAADE6YVjAAACHElEQVR4Xr1Uu0pkQRCtQtA1MDMwMnYR/ILN/YT5gxFkNfQv/AYzM8FY2cDA2EQUDUXGbBEGBAUZT1VXdVf3bR+JHuZMdz26Tlffe5voM7D+7K8GVwFuUnyhzd/DB6EGKTPKdGZRtUUn0LhSRzqL7qr8DoYJdjFDspHv4N9GcAWxc/AV9gzjcyKfYZwvxT/Bj4gU8C14k2gezqc8JhEnHgk9rjlKrs+mPc/ggwhdG9s0iPAMzpHSkWvHwj2RAhG5EnaOYQzKUY6EHxUZSlSWHFcSKVvMCUmEIEKlE8wPMExxXKcYT+F4AJ+UREuel2txEpGHW8jGYtfHRXSEpZs23wCn4J6xbUkdoZPo10zthFIX0o3j2HvG/wn+z8A5od8GHkxgdMIQEKqzlOLucWn8t6XssrzmTOtmhw9Urez9RpGCLFJLyAKGCOfjikFM1/D/hHHfXFvG1ZLlSM8kfyfBLxjLTUDhmZiODP9S9/RLd890qCRaTmbdUvniedBu9eCD/y+lN/CP2QtYdi9EBXkBMuTuwnvOrxhflKz2NuZyd8k3IFcNRPgyUX2L4H/Yj+TfCdEF1k6UFbQn33Y4iIgcHrbowfiythkGXxxT3knNCPGBsPhaRw+i2zj0Vq66Hk6/jNSPddUr0O68Mm1dZXfwDSLuyGVzqFerbKAD25OvHWY1Ce6srjhPKOHsGBY0lHD9Avq8Xd7VyKNZIcnjbxR5dwq8XVS5AAAAAElFTkSuQmCC>

[image5]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFMAAAAZCAYAAABNcRIKAAAE30lEQVR4Xu1YO6udVRCdQUXQEMEYEwhY2aVSLGwiaGWRR2FACFwUC5UUYkwn+Be00kKbSNBClFSCL4iQSEy4QQgKFmKlXBFRYmN5ndkzs2f283w5N2JzF6xzzrdnZu21Z3+vewEIyB8N+qPLwfVBYyY3i/Vg+YO6ZrgZaLEgZQ2EHlTtWA87FhghdnTVJG28HbmN6IqHpnaxKl6gl7WkEY46E5sRB45D0CrtFLWeHD9O39cTAR6oYjOcJX5PvEJ8m3iPBcKiDhEvEq8RvyHNp3Kk1M9a6FpZbwKuG/pQsIfSBwQfJc6Sr1Zr0ItyeLeZNdZu5jvEr4mfEreVuZme2Os/nCRuEu/TgdeJn2nYCu4k8ga9oMcHKPIzfT+pNLiWwLRMb4STFN1MtFosfDDIA14XBh+A6iPLs4eRj7GHTuRF8Gbuj4HYmQq8e6+G2F6Q+ieUXPcsff0hPzPeJH6uNCSt9EsmNC3TE6hKELM6qRXspQSvw+wh+9B682FgLfchiGuqEFxUDaJm4rawbOYAh0EmYaMRN4lvKBkfEm94OIEvyX+EeDdxpiV62WyxgFiXazXDfDDMww3bKUXyQUfkAQ7T9zYTkhb98vtU1FqE4Zk5wNMgkx+rxreI7yoZl8jXtzkq/k5DngsfAtWCvtYWulYNrcNjQoXMYT4Yl4jsIfvQNp0GOYOjh74P7HsoTkg7wLKZ/gCKKCrTvYVzj3osffxK/FjJ+A7kRi4QjZfB53oEohbHfR7WMj1B1wMeFRaIdeYh+QgS5iN68DU5Sg8LsODMLFZyHLoT4y/08Ykw5V/F2EzOAHgJfK5HIWqlZuZ5WMv0FJUHObPYQ92AWHeV8q4klkg+kD1g9pA2R2YpfOh6hOWet4jNfLAMdcuOgOSe8KGUly5Lps5IT0K8VinI5eWXWEcLuKlbic1lntWO0LK4GSeYZai4zPlpzK9DzAi73UQPrY+g5e2cY7eZt6OZGojNPBDCjlKB76vckFOV8N/E14Qp8hbxB/4RTNBTFP8Uwh3AWtLYUxLOYK2kNzAvHqTOayXZfDDMQ/IRwE9z9+Dr7/kwrYzWk494MzG90OYA/0Y5W++qtoV3+kw+ArgfROMxpqbxLvOrRQS/311QGlotP3tZz6YVHw6rOxMcRx8M8oA3hQXMh8HO3u6a2uaNEc/MgzKUy/mB9DvxvA1oZIN+Xabve3X4FeCHDdJOMyVvD8hfQM9ozj7ij1C/znS0UP6s5IdG0gLzgXA+LGyDnleXmVD74DrZfPMQfOA+GicfxWvQhrLxAe6hDzGEz9PHl8SfwJu5qWMPayqfDb8R/+ICpQDhOfp8n/gV8QvgBZcZDN6c92jsI5DXFPuTLiAVVFq4X5hR+zBwHRFLHyXYg/iA2kdpFldrDVDqjIYMH9QDPRS9nIglrIr3schH1I6e5LtpYOdgPXMKmbIvgXSp4Ll6dIbS/K1hUsOX7LlZQkbs4H+FpB8mWTgf36z59WECVZouYhhYigU+FgCXOpllBZHROWiQaM7p/NdkCeZzGGZeqsiaPnaGsTvFbAGM3WY6xu4UMSH9OZxZtXEJZsmmWwz0sMa8MbsurI/XgnqqtOzQh291shX5K8I7Ru/sbfeol9VHmcdHxgUIactnnMG2q92mIXS9LXpj/xOGy6kG4uG/+uVOCGBkCUwAAAAASUVORK5CYII=>

[image6]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACsAAAAZCAYAAACo79dmAAADNklEQVR4XtVWO2gVQRSdW6iFFkEblWCen1ItjP0r7QTFLl0UrAQRFcTGQBB7CyWIsdBC0CpqCH6wEBUERQg2WtgpCGoslBRBz517Z+bOzO57KzbmwNnZ+5s9c3d23nOuCRQvGMjf8VU9CSHFRCSvyhS0uJO/qdJ46uBgiPBkURT8rwgz5E8Inm7wmaG7pX+go0DRIZs+rNRgEskfMP4mJe4PtyhbD35Rct53RGYw9sHX6gtcQuw6F2GpN3D9YWJvlFzXAUnMahIbb+bB90K6HZwFJsDnSn7oZnaGGVS4CCK3U90BY8jg2Gzhb9kSfJN1LMMcOC2kXxhHrArFFdxOC/2DtxQJSaxz24uHtYhtF1QjrQRiabfQb4dJkwU3jeB6EflTnl4QqdiIKBb1vaJlYxqbtU3L+1E4K2Rio4v330I4ltTL4veBLFTFSmdN2gwM3gIc6wWnxhvFOmkK+5+CD8FFtQ8qDVLRnJnkNEnnRk3CZR1FrAhq7SyTJEfpt4AXm2t1J2Fd8nfk1uG6iNiCptS/EmrfMzZEuhWMp9Q1ivtzGp0S+le9VeMBVmwvfzOmszkuIG1cUukMLl+d5ApKseq5X3geYZ++VPOskwVwIO5ZFusr04SZ2OgVVGK1bAfIHR0Hf4InQrwFq0qsx4PCPuZk8l3gVeNnoV6sC2IT0mlAKjYlbNOY/6EQxOBa8C34Cg1ag3E/eEjpoU2JBUkseT+fs8uwboLHzcRWbOsHRnzOmpVAfNXZCHlTfLbvVc9RJz9AzAobwHfgJmXAHXAZk21kQ58tYuVLrz4wEqE4vsj8gvlKL5ZqsXwcrjj9gLWDt3A9oMzA59w3J6teYqLgvAo74kQwt2aPk3Pwo5If/AwBbBHqk8Q++0XIQl6A13SF/OrZZv8nnYfZB++q/4n46LHavAimM2/Vv6LMLjH0P6u2o4wXR1eNEB+Wl9CcOfRBjJjTJVlRTNxm1f3LOhbC+XLr8noaixTlTrfnVaBBsw5Ac1GxpOakBB/XpCy3rbAptwOa8/9TsQGhrmt9sZzMy4P1dJ2zE5omy31lhulMFkpLsGNtVObfI01QTpXElZGB8MnVijrhD11TxrjzF/CSAAAAAElFTkSuQmCC>

[image7]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACgAAAAZCAYAAABD2GxlAAAC3UlEQVR4Xs1WO2tWQRCd9YGipgsigtqIglgKPlAUAyJaiCg2WvhqUsR3oUgQwU78AzZ2CloqCgYURBSRFBqwshNBbHx1WsSzO7N7Z3dn75dgYU5yvr135uzs2bk3+4XIgBPq+xp51Nb0gWfUVZyRcVZ5K2YjKcudVTFL0Fi+Q8zaqiyq67vu1p6pIIKWtivJP3WmD43ttRaLKHPh3rndgUSPwc/gS/AuuBrcCl4TMsoiJua8wQjHzTeMLQEvglfAc8LFmYJxCfwlPKKqrAIfoPwHjNeF/w5l8jw4JNFhTyw2moscDNE0eCJQ7VLeuWXgJ9IGg2ZGLaxFReRyGSA2HTGE/FdwkuIDyPTJ7Tg1OngB/AHRb4zbhBOY8g7ja3AleIP4vXkjPN7ZdCPY6AG5PsqkjZ0Jd8hx926lSPrI4N/Bk8IM88m7d6HI7UBHizDOA6fAj+CIL4jfPZ64/0bxsfJCG8Az4BqhStFNYoOjXbMKWLGALrGfQhG3iZnwBHyhdOuEfsEtKWoiTYoGT2mDmScVa3kVg7RcGJUPwXtRBKwVTiPvH0lAXTRGwngQo699tc8AEn7dXcJKN5cNhsu9xAaHhRx1yiDP6AwSbQ8RlbSAzFJ8+nf2qQ4aMw6Dx5ic1ZrQQQSGPbuke+RaHSSXGTQWFITMWQx/MG4WFmm3AB/PwBWe1vHHj9i3ObQ6Lej/SLTB9ULfwR0+YNTKwYKFGO9j/CLcqeb5E+MO8XFnYgxi/zXjF43n3D6c4hMYf2I334nfRbxL9DaQj6QpjKdjkV6wG39s+a9D0E0i9ArXz1H/Pe7HtLxG2o6/kIcbQ7pHXboKp4uipfmtIehDJi0ffP9tYbDKBtTmLLTiClEycH9RUImqQI4B6Rmg1YMSprsGButaHU4PU4VmUM7jPxgMlH9GZ4Okb0xshNtQE6o2zLpYY0ZXeMCGW0ntLIwtoQVLq4rZh4A1SSAGehQBfwElNmtm0kk/wgAAAABJRU5ErkJggg==>

[image8]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAC8AAAAZCAYAAAChBHccAAADfklEQVR4XtVWS4iPURQ/F4Uo5TGM3cxGYWNBo5jF2FgxxEaUGCwsxNSUlWzlUVKSkvEsSTSREOWZt0IWiPJIUl4rNuN37jnffX8zf1bmV7/7OOd37z33fue730c0AExcRPbQ4tuxrh41uoI5X7mIWodHupG0XyE1pf0KbrwK4sojkLh+iniuwkylQf+Iv5oqDd4ZgxpVG6qHQjNJvR7ZBA4d8D1BfRuSC2i35iuaZhTXwPvgTXCB2pUVzBYUz0h0B8CxgbMQwlANXjv7wOto96HuV+bBl9ECfgRnaH8e+B7kYJm8yHCU98A1sqBpQvGK3AYcloIPwHHa3wpeAocJ60+P0UU++MnePOCgveDZxHYH3KZkLAe/Ek/kp9qB9pXk5J+C3U4hp86xdAjD4fkzqAleUd7DB5j3Rz5Dp1FeZcpB03GUzwMFYxP4C85RQppOsu4KL7Gjv4DblYHZZPHUBB9HFmA0+qzdnfiOkqQFk3GDJIcdoFxPsk6rcqH2OxPdO1SHlDF8KLbVharfkqgp8eUw9r3oh3dX4jkCflcyHpPkfAg9KDNbSMt03cXWW50u0VuStExTM0MYfJ42OcbpybvgdZu94E8l4y54LzmMtSTBzxHSIulr8B5vID9nSWHOV/CWMG2mDHry1mF+k0sbZz2G4hVTD/AiyS0SYgPFaTNf+0uSdfnmKqRNpRr6wdsiDL7Zm8Pgs43wR40/JhT4zqDdJ7TmPShfVE7FZvAbOEI5kWTdlaGI5IrtUQqimHw8Qc7bL2IaK78HIyOL5Lt7mVSOHDc9QgvO4x+hANhJ8lEMgZfadAeHxh8r3hB/+ZmCKKYweH/yU51VBHyzfAZPeLvFTPA1yaNnzAI/YUwLU6ceQ5I2fKMwxhu59ztlbhfAKvCW6hkbSW6p6ukkZylYDV4GX5IPntOBbdNUM4FsUPZRRysCc8HDMJ0nCbItl9indhCmU6gfgetCp4dZZfi2Mry24fX5V6IAU95JjuiKOmkN6dDGJorRwJhUkvYdUofvuxb/a/Q6cwFWGUwUzul96UolSxmN6gSq1jX5SmzndmFjCWJVEekmM3lmKGBQjRO0h9YSilOFxvAwApQPIxCnAxxqHRX+5+AHhb6w6QRpP0LNipkp12USh+TfJhPmc1kUTAU0pop1wYJx5VETU9FWQs1yrt/IRA1IEqR/kXHvD+fZr3VnxvH1AAAAAElFTkSuQmCC>

[image9]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACwAAAAbCAYAAAAH+20UAAADLUlEQVR4XtWXz4tOURjHn5OUEcaUoiilFDVZKIVkochfYGMnJTWzURLyY4WFkSgihSi/jVmMURQbJT8KKUlKyo+VBdnYjO9znufe85xzz73vfWdm41Pf+97znO95znPOPfd9Z4gMTtWMuLyvs7klXSRqslb6KoFumOQiG8dlOjOhLmnKoIspW50IjvWqN9CsMkqlZRP0moX2UwTv4X6hsSXoqM4VCK3t/13BRGdhe4LP+6pxSgoGS5DpGz6XqZiN0EeoT5pmxi4mL2ljTjyDKi54to8EwynoVtkK8NPYlQYj7CRtikqoGeIP/iCLbMGBr9DJaPs8bgyXYROoWpqo89XFE6o77HE9GjsaYhz212F8vIziBp13NzQUanA9uH/GgmG5btZMdAxAe6HL0Ai0lVWtPUR4wADaKM7ZMzyPpOAjJlZwB/pcNHyqcofLxK+gg0UDbCDJNw5Ln/p2kBRbsJ+0YBOLcdGRcPYMz5FYtuDb0Jc0aOglybfZxA5Ab1UFF0ie1FptL4VWsMLand1cfydHwu+wL9J2/sX1mIkV8JHgHaxjC0nBNt8odF5VsBr6Q7IxH6DT0HRV8sRC1eYMO96ZElhekE/C5jAAPICu2kDA+4ZIxhahabj+grarLP0w8O6PyKa5PaKU/67ggP2W6I3KIjoB3YxDjHvn5IWp4zF0zrRXkeTvVx2G5kM3oLnBRvtIjg6rFnOGnQwOVXNy/lVbrGLWQD+gBYUpWSSfv98IjmoPf3U9JCl4huo6dzj+LnfyLeGdjg6RLIaV4GgbSaJPKi74ucYWGec66KKKf8L5rV7pe5JKFV4QF3cFhkckLxm//ddgH2ORvGzMJeTgx3+G5Jgdp7CoDujktgZddZYQjv9MJPnBeC+3NqlxVXKmvoohQ8Zji8ospRqS9l3yu1slzpKZsHviQsoFZ6jvc99x2Vm2TE9MfU9JoyXTmQmVZPr4R4L1k+TP0slT5JiKXB3JTMKhEI4NcTwZnMk19WQmmXDBWdp4mDpfdp7qP5oViycfnRhtc9VXo0jx6QKKO3tf9jTmU/JJmqn31fdMnrrcdXHG9zUZYuIdbj/uH2gih94PhruYAAAAAElFTkSuQmCC>

[image10]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADoAAAAZCAYAAABggz2wAAADfklEQVR4Xt1WO2gWQRDeVasYQfAtomIjBEQQwScKvmqxsRErH4WFIAhJkVgJEUXEwkcUVEwKu6BoE0EhnaWNigZt1UrESiR+szN3uzu7e3eJKdSPfP/dzHw7u7OvnDEFWO0weV8JOW2bL4pbrSeLPd6v7SZ0U3m06F24RZPCF9EGPRlFc3YDEeTGI7Z2px5b/XVApcqr897ZYqbZZAYbhngWHK3nqpogKT6kxBaBg+BNViYgVT/4DpyE8QzPtWGwG7orGf9IoafBE2Av91ySzRrrkPE7nmM6oGGtXYXHZ/Ae+BW8FStqDCDnB7RYJmUNgS9k6lQBsXkIHAevCpezOxDVr/Xcp4g0NR6Db8HRJKLTxPaUoUKr7nxsJfgLvFh7jOlDfBrPbUKGzs9w3q3C++BlcEMd0cg6K7jgSeEpmJPgWNIkcUTgQlPsB6fRtp/rd0mWOh9v90HniScnD4lvBEcsr/QRPOeHmhasMNxuXLJNmmTrVqMojqhU6B7jirIDgY9WmYq/4UjIpkwQqKxZjd8r4Gtwuw80grbpeiEhU6iGv8wEOIPZQhdD+APPS4Fvl+EVfSQM0K1iAs0WnYcHaLTTr4AksEmqw+D52FUV6pWl7p2fg9kVlXZDeKEbd4m4qWgq9K6wOyxf13Qx0WquUeEA4ZBtL9aFLh+9zZNCGbyK2itAofa2dgqoybDhi47ukqOGt/MQUxRd8L8X2ie8Bl6AutoaNaqBJSeKcdDwNZ8nx3YE+hhBSstbd8R7YqjeNxnuY5+wBLsFP9fBM8Ie8XtJYuuYh5+MGh9NehnNM/wFxIjSOSMoNJlaOodPvWnOGf5KUt3GYzwOHgMXRF4lKpdFaI4Cn0xa6BPwDbhQqNNM4RvnTuTxeAk+lPfNaPcFjQ8EcYbLlxubmo+sJhJlBRXos5I4AdVPw1vrleGjQZhAc1pp2j09QaYJ8Lnhrf4NfVD7YaKLsnC34X93pH1v6MgoNI4sRnepVuqvzRmhqa0Nw0qYaccuH8D/PUufTe7ziRi+K+6tWwky+ecEnfIGIjrBZHZqVyOnFl+ULFk6bQdoCKXoLm5XVoqggLlAkkf10wQt0XaNYiADvyrK/kMkef6aQgseZ9FPYSIiO9B5R9n0KAbSU5FDg8aHCgXMBZpy8gQ2KVLE8xi0bauhDhQVM0WYqHvSorIYaEY8Id75G4dVhWOI+mXvAAAAAElFTkSuQmCC>

[image11]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEUAAAAVCAYAAAAQAyPeAAACMUlEQVR4XuWYPU7DQBCFbSRQKm4B1NDDASiRqFAaJGg5BBKiQEgIcQYq7sUxmF3v2vPzxjtOQBT5YHB23pvZnyQm0HWZPn/nKLCHbRaZE0OBLsvjmtQiJGQSjHN4pb0nBJjrG2euQz047tlkVt+Puo3jOj1nGumyCm/peRqUMrS4GMuq6jzTqoevWbLMK0f/8ECWg630JjPhCgnQiwMltKYGi8xL2OY9J2j00bIei0R5NrRHj/+M3TkU1HEzml2EAbjVXlvwu4Kxl7MKETZ2wntBo2e63kwp3GtYJlI4nu7lFUFbjKFZtGXa3zVdvyjuKA6ULsld3ecNZAbkESpXbzIZ4Xc8v0B9GfarIbpbig+KNQn73CkQBwGQO3YImQyoQuRmDN6vYp09oniieExB4om1/A/TKpzD62HWSXpYc7pPfFOcasEsQ9XaVoy5wibcb1ahkLodORTj5LfeY4oXivcUJJ9xz+wkIJUwab5aI3Y4B/Fe/BU+kaV+qmi4xCjdT1aUW9P1leKKYm9U/S6AiNnxOOkM0+ZskFCBMZnzu6T4pMwDxWF1xIg7RzbdcMNsZJOolP1n3TEx7ZzijeKeyRanT2ZOA0z2stAoo13XqE9QbCBy6QesLzBthw5lMUuqkLd1k5QgL8oJ0BnqsQO2odPbArS+TE7WifoSWl9KtEj96ypahulLg9hfN4moDzHWiQdTt21650rWwOkFUgQ/RezwWerfhjyXN2HNK70ehC77ASKbD6gaxsyFAAAAAElFTkSuQmCC>

[image12]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAWCAYAAABtwKSvAAAB40lEQVR4XtWXvUoFMRCFc1FE8AcbazvFSt/AB7Cw18ZOOztBQQTxAcTOVmxtfAyx8Y08yWaTyUxmkru38X5ybnbnnGQzuaugc13M8BMGXqZDSaVYKSUsLyACvkCLLMDtVBSFsigiVWQqVGI5u8PBybTBGJ5rEoFupElHriOSmStcwZqfG5MpeuJNqpFq0Q11zWtgTftvzZxB+7wYqOeVvfP3XQQU7Bx1zWTc1C4+n6DPqHMU16yJdMN6Tnc8tjsBdsJbQTN3h/EXeoR2kkvo3Uhfzk6xPSoUAZFeh66hH+gW2i7tRfDPKp9H77irXZcwhy8SWYUuoW+YLxg3vSo5V52diJ4VCSgBpZxpBhL+W7qA3qADrzA1fvQtI3P8XiD6t2ZYXslyNENtf80m7mF4xngP+b94ImU/oOKKAsHyIh2RAbLNI+gD+oIOi0AD3modO5VcPZLIuRxegU5w++6F61c3NkGDHYsnaFa57t1whr6q/orua3Rm7gafV9BG0PxP6YjL35k+eAPjQJoZobcVu4KV0uoDU86oRE4Wjdahp0BL05vJ8DWmfmvdLG8zp1jwASMRv086juoi9Z+Hfuhki3ZOM/mpKoiIKAyIjfB/I7SZPMVItniAJLsLviJxcnONZsC5P9QqF2sGr2hEAAAAAElFTkSuQmCC>

[image13]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAVCAYAAAANfR1FAAAC6UlEQVR4Xt2WvY8OURTGz0VBIlgUIhu7FEJlE6WPUK1C4atUaWkQnwWNj6gWhSgQCiFRbKKg2LARJEKNSiIRkYjwByjWc+45c+fcuffOzCubbLJP8puZe+45Z865M3Pflygn17y2BpE1J+6FkTXxKTM7gJrRzTGrULxXp7ncVrCk5lqm0UghsOQwS4obsTcb8I4DurepNVV5sjxTlkuimuM+i9FcxHxIYshYvLx5h3IPLI2mi0rSHQIHlLVq2w7OB49ESY5UA7lEzvOxMUd7cXwAniszYEXlJi69NQnnGY/kYT6BzbFbrLSoWZNPeEzhYoZqc+FmefMkuCq4kzgfBMtjF/lu8uF5q0jnbE0mUVtk2piViWxJ8gSsVv5LLbkjtfo1Jk1jbqg1MCjeCZ00dko5A56BW2BJNp8YlxG/rvL68tM+Dc6CF8plsBHObLsEHoGxRr6d4Ca4Ar+H8RTRUSX5xrzCK5Bu65Vgv4jTLoW1ELwDj3Wclab+iOM3sEbNWxgn9dwOflL8dFyD4wXYVirMvoorjX0P/CfI45RqTBN+ztGFcFujuhCfcySypnpN8QKMKhzLu2wlfhN+NLJw7GeAOmi8eYtSY4nqOJvBbXLymmxVKp0jycnbvhdHZdqbBnfMeFjh2HFj51f1pxlDbh0Ob0k+I/ZnW5i139iqYO2QD5fDYcexjvZ7KjPRdfCXchtSLG7srlm0ESHXmPtlxpC7pheLwD47w7KNRbtaZnUzcsPwe4WLxQqLV/I7yUffJV5x/nNQ3XBUCY1pHdhE3G9/VT/6P2CDTNc6AqbAF4UTfVDbeuMXK9/tbpLmmJckeU6ABdY91CNG/p2bwjU/VS74KcATd+8FXw//yB8n2fm+qo2f8JjyhmRHvgHuk2r+NZavzajToa/sv/S2pG1zVupnViiKLKcxgZFCph6yNyv/9gV1Oqj6+vVx1EXRQxxQDs/MmPXKzDbUz0tkEs+lQsmlQswitvoV9A/+oo2ui0ZJ6QAAAABJRU5ErkJggg==>

[image14]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACoAAAAWCAYAAAC2ew6NAAAB2klEQVR4XtVUu0pFMRBMUMEHigj+gILaWPgF/oCFtYWNCLY2wgVFsLEVS3uxsfI3RPwlJ8nmsZtNTtBC7sCcZHdmN5tzjxprAtwaKaHlAmwSpSfFvqlUOYJaeLrDaEkPatPUe7D9M6uN7mtlTS1QLNN1ogBdUMQMyqSKy/CsDaFuDBh5q1M9pg0Z3lZ552TQU3Cv3SPcQtdGUVe7i8usjD2K5DaiB6zvxDNwqb6Svm82V4UpKEW+FxvGrgeaGfgN3kPZVEpVsGvEvpRszq0mM1Jdy4eDlrFcYfeJ9QbcEJb6m00ds9AckMCczYm0nAPPL4IX4BfST9BWPRnC/8s0VTxzEE2vFIr+lUZYAc+hvkDfdZQGDTS+CqZ0DvaIP93cDNqENTt4PoK3sG6FVO6Zy6cadRyloA5cJRiOwDfwAzwQWoGiSTm9ONy/2ZSThmH4ogXwGA1fHbF/RnrfK/nxB5SD/gK51l7jcQmuEVlvuapIJlc4+NboJQy6Te1K8VgL/kczVuOrFBvN7il0xa1CVsk6itMbHYHzys4CWezaCvz7oB4n4J0lun2Hh45p0HiQe3RPdOgbKjUfkQPFlsEKesgO7m1Xyu9bOmVcIw4XnUqFklIx+SW09CL/A91gF5XWd0aIAAAAAElFTkSuQmCC>

[image15]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAVCAYAAADxaDaPAAADo0lEQVR4XuVWXYhOQRieoeTniiglQilFWRvyc+lmpUhSRFxwLSlJJO36WYV14a9VNkT7lbghF8hPLsTSuiHLhTuUyCU363nmnZkzZ86c852P9mqfer6Zed533nnfOXPmfEqrOqjnlcb/zG0Bbpmq5apsxhg7xONK6KK7jkMkfAgvSodehXRSEzUXcHNyhtxohBAtEg5tUrH876gZr8pYZasGd1maUCoEjLXYXoLCmUg+VR3uqRknF0hIRH6uYCnkXWgXCM2qs9C/KObEjIQUYKzlVvBUZCOmgXssV4DjwAmIeQbtDONRiB8L8SZYLZZqYFQXvxcc9tSm/QmuM1ZGTEVN6yzomuVb8GXebLBEhetl7Gw1/1QK8bgZWPwN8ACJydvQznbG4g4LCrIXfEoN8FXCE8Xr55aHMN4p2sgjy0T7wnaDG71ubZUw9tS+52CLV5m/uLfhAZ8lvWc+rTwKgkOZoUwvwDiieM33e79lL3gfnJO51EHkqIPivWTQBt615HrHwBewrc4ck4v2I8BvtM/AzSBOjW4I1T1wKrgPPgfR3gQ3mVkMJeF4n1wAu8CrRgE2KHn6FubFO43OZ8WLKImSJ5XPuV+l3/m5Sja4F+5jrLYek/9gfnvgF8AHPo/uL7RrrOz+D30DB9CbaP06lLm39BQ7Jo4oebWbgpN5CW1HwA6wB/0e07p+jvowJ0XPqwFhwA2Sz5IQw3gl610WqdT7OPiVnchjSEsuDvyKIJ5eGGic+wPs1qxLyZO9Am4JnIhVSpLxOyWL8bc0MQvvw+Poi89s5qifsDSSjcgjzdctsYR36gSH8jaDd0peH4flSvJfFGjc4D4lXzNSzdTSCXbNrMILkPraMA9XVj43Oyq+pyz+dSwC75XooY3v6zBCnAy0COY8sPiPsQX4oNLFLw40ns5Jtr+MP0z5Kdp5QVn80/EIfOIEh7Bw0xbqzYHFv5Gu2yDzyz8zvLBIpx1F8x3tdHEsRTf8P8WiktMQFr9SSfHtQc53lD3uGbT5N8fb8oElEz6HCZONOXCthDjy8+Xi8I8SE+BG3vJ+cvwuWT5WcineBucHPinwPviiJOZDJfeSW4sabX1Ig/9WB5Uc70FwhyVt15V8CexJH9XF10JweNgz3bxWD5G/j1NE9hULtTTK9HJDBVqaY51bmtMKagSOXeJNE63ol0Y9r9poGi5xAlLJxuMMoSXt5dXYXPxKxWjq0Bypaizckyox51DHx6OisL9VFK3eEVKgsAAAAABJRU5ErkJggg==>

[image16]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEkAAAAVCAYAAAAKP8NQAAACQklEQVR4Xu2YvVFsMQyFfYEKGBIgoAFmiKgAEkgISUhogYCEjIghIyKihlfAi6mCZpDXf/KxJNvLLhHfjME+OpJt7bJ3B+c0FhQ8IIqeMZbJZNm9ufOI9Ovxa/hZWC0sMc+tYlZsmngGUBv4Gct0y/CdltA8efOoysGtgFvheghMWq1R5EAHZKtgykv4Q2oKxHijRzTd1dsYNqDnxKq4Qy8/wu7/S8gbyaoFy+D3Z9LG+GvSltEOoumzzNbBT9QwS+sSSx8ag/VHbMuAre8JT8uOqULx+kL83kAjNYLLGj/SHY0PGle03vUjyDZS7S69pF7cpE1ulXFYbpjS22uPZvc0/R/HE6mHxafg08WT1IHKIvpHmE1cx2/ksAf3Thy3JH2S+k7z4+Rq0Oo2eufRrxILxe9n1bfZGLWw4iVmuRwL675LGv/I4Jt1hkGhG7XahtZmqhSY5VMawIuRsxu95pTGF3kfk6C+EnYdk9HU2me1YNFDFOjcuUfO3qfxTKs3ks55YNOEwlr5GT02TQpJCL4sCTHOETle6Pcr/T7BoEfN5wE2H924EC/r7M+z5NLBaFqjbhG8/rF/4VZPtdV4oHFQX2ymqACmlx6kHy1SB3AtwQ8u+GsprqS9MiHw1yRpL8a18585zt248hVgTYxdMrIHeyW7RoBMdYkB31D134FJTq3snBQsCUGaZKZCuYyf1VfT6+iRUYQejQNJuU65yhzRD+eZrfJjcENcI903osTAFxHJwVsLfapXUnImBNVzD/IN1loRTjKmjy4AAAAASUVORK5CYII=>

[image17]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAVCAYAAAD2KuiaAAACFElEQVR4Xu2XsVEEMQxFdzkqYEiAgAaYIaICSCAhJCGhBQISMiKGjIiIGiiAmCpoBnnttaUvyfbeXsib0a0tff2VzRwMw8AY+WbGSIaUkR687Jx2qoBU+e/ScN20Lh9r6R3Kzq4B37X1G9BoK5KB9NKX41GrcTzvXbHMVV3/WLnMWOsBdcl6S+JEO784dXa29l7hH0Q6mBJOU5BAXdqXOVgCD9HC06MXh79TIh1MCacpSKCOnVcMg0Nj32qa5vVDexeX51YHSxg9CEpw72C8pbMzkkeXGCknuZiqCxun6MrqnuKT4ppymynqdoJJ2ZQ3BYleXT9tx3iCffp4oPhO8UxxpE/H1m1nLVE/BaQpyKAE923sjr0Y4x09fyg+KE6mirqMiM7sjuBd9cci7u1USZaiKQtcUXyN8SLOQ8IcKiVUnsFrvo4b+ap+HI/Z3ykXyiBnFL8UT7LGPWpu/TVfyStaZd8ZJnDf5oDihdreqfeCDGwHOysux5Eo7IMgC/3MPVYkxxSvpHmj5ykvhDa7lWeZCtOQ6kce2veQlaqO3caG1pdD/O0f4pHiMJcniditJ9+RP3CW5F0Phk4aleT/BcTVzRC+48NwO+Q/g3w21SlQ1dyTKvyhxDXwP8/+771Cj4NgGvd2KsAHRFRNS2yYvN0SVW0dAh3LDZA0xHojhx0cUjDXapqF+BdgJiVKohLLSO01lz+7dBEvXFgJOQAAAABJRU5ErkJggg==>

[image18]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAACYAAAAZCAYAAABdEVzWAAAC+UlEQVR4Xs1Wy4uPURh+DyKTZMFCTUo2LoVYWCjk9gcoGwsbrEZsFRKKMiUNNpOahYWUQpJC2UhNUQwbheRWbFhRpPG8533P+d5z+82w0Dz1fOec5718z3e+y+9HxHD+2ETPsA2GeV7gSkmEQrWy7WYSXa0bw+WZTdg8X6WLem1d9YhuKzkS607TTqrMLeJJJoVWYl0PtjR6GrPvGMeVj8GZCB6A/lEYY0/AfqlzI0ZnvkJNnw91qBto6f/R2MQIFutWaR74G8H7soxZs5VfwTfpo+AB83RG6KalIYtMrOY0VOAB+AucH5VuWx+S7MqaoHvZ0TCO/coOif/q+YxYXGyBAZKT7zYam2Re5hjezCMh4OTlvh3WEXFXiQ6C30iabkDknpBGwafgKrTYi/EmSl5gPGu3wvhcSNKDTQTsUa4Hn4GjJn8d5tFoAQT53h4iburoKtZ9TA3fAl+TPAvck5+lH+BmZY5H4BdwBi9QMMLEdDo4SGJ8keaeAFfqPIGa98MWkqJtXZhD7iLJbnLjUPABPOrpkp1nHCbpw/3mgPwMDWvlVowc26/r66GoAIe1bzC2zIQZQ+CYyWO8JWMshVtB0gd1bgfGnUrGLOKLdHQH4xLwVNI1x5Q1xkB4E0nDpUZjnMNszArE3yOiY0wvBb2LvyQxfwmcqxQ4uobjT5Lna3XVWPZs8IPsjWW7c550xwzeYX2cmcoKRydJet3IQ8A+ktjzqOgJKxY9wq1cHhXJvIBRdqzDexJT3ljcsO6K1pLv5fi7lmMByY4NcnJuRtadOoA5f5/YGP+W7VJeAT+rzt+17SD/5IzDxCcm8e2KznQU8O1enCgM2Zm7mG30y3AxucPJIKmRxiUKMRX8qshhaMN/N9hV/H1tifImdigihdBA+tKERVbdbNYMpLD/QvISv45iZoAHrsle7Qlhmjb/uZPVK0lTw1i56gGT2KzxrpOlhVrrFrU+hdZKjAgJ9evqWephDPVGnmFOmocCEl/1pEStpzR0Ef8Af0eJcbpRaykAAAAASUVORK5CYII=>

[image19]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABcAAAAcCAYAAACK7SRjAAAB90lEQVR4XrVUsUpdQRCdVRsRP0BIqnSCSeUP2BoipDL6ChOJ2FhqIQQhEksrsdRSrGzEQvQXBMEmhViJTTCQHzBnd+buzs7uPrV4B87u3XPOnZ27+3jkCEgDQ6+VnJCL1UiHZPonl4WtV8A2JiIrVbOCRu0Aq8eaAy2eK33WGsFbwHSD+SnQ0WMguY0yatfF7ty6VYBZ4g16QgtbXuNZ7yOF4q7nmdLmWLg3q/cB52Zdu/NY6S2eb5g01smYfmBa70IFHB8Lcr7rXrUxaF8xPASqyhjuMX0zWT3ImfOxqFjCQIqnDB1gOvFU3jsM/jynOi0Dv8cX6sJPc8GL0lqGO3BTKHCLCP7FwxBXyr6oW30ivtAvQgUOThAHZoTdi/vgGT9GDIPvKUbQefg6N8/skPqfI77xcU9lXoNb4Afwu3DbZ8EVyX2WtT8eT5HT/FMCk0KPZTheWwJ3yJ8/6IJOf4g39NgDf4OjQgU+lnMMF+Al0x2RL+pC4StwNeZhOv6aU/AYPETujfIVBlqcaAT8B875fdI1CHjzmthA7k07Pu+4e/GqFYoN1f9SBkdrGG6trGFfqtRu4hfM3b6Jl6DVukXqrPQCnvWqC4OqV/nvKIQOTYMa3muKR9gblyIyRK8oVAha6WeWXoDaUAltxK6KnPqCimcRs8B/uI9PpkVKauEAAAAASUVORK5CYII=>

[image20]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAbCAYAAADFyymQAAAECklEQVR4Xu2Xu2tVQRCHZ32AWqiQQsGg4LOKBB+oiJXiA2xEsFELwUKC0cZGRETjoxAj/gWKCgGFCD5iYVSwUoyNCFZC0EYtDCSIhYjO7M45Z/Yxm703YJUPfvfunZmdnZ0999xzAdrAhAYFitNiNbuPKYxrkZKkJTE1pcFKRxKmFCdQnaHRUZgBWolU8Pcg0+VT63P4lNncjJV8jXkgDvGrK6IgXA3xHWpYjn2o96i/qJ+oZ6znqA+oMUxLPtJqNyVcx35ejm8XAkdDrqF51qA2h0aV6YYIlPRPUW9Do4NmyFn1V2MBvv5G3RJOyVbUH9Sc2hKmAjiN6vIsaZr6lA1Y2Idvr/CtPx+sswEnjhs6TQO7yVCYZhe4K+BQ6BDQVeMQScVwqBlKvM5hfTAObi1bX5ImKR0UxR6oLRHpHVarvgS36C/UR9RsJT7kIriFl3jlA5xrhnBXjJttuuC1qDPCnSJdX+3yC+VPO4Hrkm6xrspBVrWBy+ASHa8j8gyDu4/I2rpRD6uAaH1/D3TvWFl/SuPqc3Mmq287uCvyM+oHj697ERlmoN6wFrJtPuorahQ1l20adEp0YvQ1wyINyxYsrxBL1BjHvdAjPlF9M4yoz9T1mVHw6ouyD6PpZtKToboywuLpIYk2dZI+ZBJuAdeMHvpQH7yBO/i6Q8QJ3M2Yc+KVZOiGKrwe7uo1+foSUCPpfnMsdOSg0x0B13GSrIZ8n8BdKXRzYmf0aH0KXGHd3Igqx23UPI45gtrjhlFrLwH95Kahe9iIVVUfY+L6uMaadcB1BXaG6wjKoe72iNMKOQwu6XnPKoMNDOLLGLgTEWZvTD+Vs4RJMhgaRNNtfaQgXzWS9fk1AvRi3AQEdVUo+51uiLTQJUgPLTOlMYCS0c/bBHoWh05wE76hnoQOkWov6krjqNnEoq9cClmfqNGjro8laxxAPeLxCtRV4UtsFUwf2h7j4GxeZsi4U7hRT23YCM7n3RQFXbjwF8yxKnQg11jav9s+SNVnwvrsAx3V4Ndo7KNAv3H3sQeoTupB3AeL6TDuf0eVyBM3AFX/fJLo0Xwpaxm433b+D2PolKr/MJVe87wXEEN13Wd5Rh51QGF9zdirkdiP+o5J32EcXYmToLTKkvMlaCncBpttxn3He0P3/2WSyidxt4dM2ozpqXERS8ABpYWUxiWZ0uQALZdmJ/zG0A0vQ5wo3VfRwHhKBjU4dlhLiwdlSQYnjetRR0OjRr1Xfkk2owQRF+UopsU1p070FJynILggpAWmG5Ijl1r6aOxvzI5z0wtoe3rxRHn6bg9pFLtiLqR8dq60kCaudIZKblnNHlMeWYaeT/FIsx2zQQlvg2iFaOiZEnYd5RASppDStQpCLP8AET7GbAOXQoMAAAAASUVORK5CYII=>

[image21]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAVCAYAAADVcblPAAAEf0lEQVR4Xu1XXehnQxh+h8UKoXBBRHJDSvlolZKPkM36CLHZtJKb3TZqN5RlV+S7FRGh3aXlQuzVKrTCjSupFXJDWe3iYiW5oMTzzLxzZuY9c86Zs/+93Kf/c+a8H/Oe933nnPnNX2QEzvmrVTeBs2bNtM5WNpgwe0znMG61qHtHbd3ax4BfkewCGt+KOdHnljiEhc7PYEO1rXXPo6pIyp55Ak3+FaeKyqOas+xvhiZauJ0KMGY3tqFsG5BPmxdinvcwBuIY9aPgNvBpErbXMJ5ZulRxIXg3eI6EXfUU8JXUL/twKxc4DHwM3AF+D26F9xkkjWHPzmDlsgaff98lw6hx/3GwkQcATOD9IrKTm3D9GjwkKb3ee2We94H/eTo//g7ekMwB1ZxNIOB58BK9Pwn8EvyOhNtRpXMxkflrDYqh/FugNc7FEeCf4D1Gv1hCg66YCMpG8i14ELwDPL2wZujFKRt5HPgXuKrTiFsucZGYnwmg02P+tobFurBT+VcwMoOfhWeh9Fd+mnwgV9AU5/bh8kSUVFeKImvAm6PQszZA5xwP/gphc2ZaKl0j3SOZPkfMP9WQsA/xTP4WtYyLJjRjma7cddYA7EW4rVEYCL0G+pdhvB/3r4IfS29vHZg5hjDlOUlNuio3ZPfLJPnYGvZKln8F74B/g5+Dt4DrlW+DH4IngGvBhyR8dfzicpwsrF3kcaTiuIpMgqvfQdPdjZvtub6CG8F7w63jvGdw8xNujvYcwcS6nyrhU39P6V17/mEvjI1cahx2g6P5w/0FDH+A1xrDHly/AmMNl0vYPk5UEg9L2NKEEy6TkARX1uIXGV/RPpxcKSHeXcprwE3QbyrGkht1dgT3vU/BLeAipSJvvx9j/r6GwtKWP08Jv1kl8C34YiZfIOEZHEmCc7n9PUnhXHW4VY0Jjr/A7tm+uuORGN4Ab8/MSyTsaeuVZdkRhdDDFgm/4AnWP8kx/1oNPEH08jexNoA/dPpIkV0Yn4pOwHkSnnGRkjhcQv3Uu0WY8DNuVqsx4hh1qL2pEfz8eOzhWxXBow/nXa9sQlYb96OXYv9xWULi9oHkUoBvK/MvanAt+YeD6QYXG5kMvHwD5o08X0K8i5XERvHHsoQVmPxJrgDuBL8AD1WZifGgvK7zCE/kRn2WylyUjzB+prbYjAyl5DXJ6zYJn1TY9J0f31VyL6Yv3wzGv5qyYoUyr8HmX4fDD4XIj1YtzMM5baTjX/y080Zuh35lcOnK9Q/m5/S6kk0Lm2qwHys8noi85aVkOA38wIVfax6g+TbxKNN5pNaZJhZGxzj/iH6mLhv1/mx1vBT8V/rnXiKvYQeKiz8KdTjH/372aPydEvZzkrXwGdxj35Sw1/OHh367lPhvzm2W0I9tBxt5oBpZwhRqZWv2qCoNXIPbpMMIyuXqo2JTVWmZilPDXP+GCdMeLZgfJf7XZrRWkaFv62sC0pnDC3oZ8p6D1hgTz2vJKZgmnDJYLyvPw8JmdxgOYy1WbkVtXtTZMcLKdYx7Jeu4nzQ4JPwPLcvtuzQuADoAAAAASUVORK5CYII=>

[image22]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAVCAYAAADVcblPAAACXUlEQVR4Xu2XMXLbMBBFSScnyKRxUvgCnnHlE8SN3bhMkyZXcJFGnStPOleqcoYcILVP4ct4QQDELnY/sOBQVfxmVhL//7sASUmWp2kX5looMKuRcjI7Z1Qp1aQEDY8srx09ktDgb5rnWKcj7sdcorNV0LUPzdHQhAYDZ4STD8INqL1d8E7UOb0fz61gCT3AoBVoeQJ3UBE7+eM2fL055Uu78I7COe7IlL6BsxDwzFHSXDAQyC6WXs8ATyaAc//5hTQ9QzQkhScT8OZ6bJvT6rK81vcndgBVQzocHhPY1NShtR8oRmO1Zx61Xkl+kPOHnm+pPqQqlPkDsCbVq4TJ1kZBM5DOaERa771CueIfqX5S/Ut1oDpfcw56y/V8k5GbWM6lCBZARuT4YNt0luo71QvVkepr+4x8S61uHiVGqqsgSZ6MNBoqqyylFq6jEPfagBv6z+UvPR+p9ao2FePzV3IrGoH09r9XSN8IuwfyIx/2UF62uKR6pfrFZnVZcusDWqmncyyNYdqmOE4+BXvcjIzMJ6pHqmeq64ml1+tjIlcUMdRTvavqWH3sxeqLWnb4s5UGCTsaYcEvVE9Uv6kuSiKAF+zR6to+9TT09oL88JPnG9mHWNMD1ecqswvq7qIdAcNWsS7xpWxQr9TfL2QX1Fv0uyl+B95P5efPAGwBtJagCqVDu3W93CbrjcARP+Cvv1YS0vD9bs/gLHAs2dIMUSuJbITzFoYDR4MjEhFB1RUELqrAaQHLmbIpFha7PGyg32dcS7vJVgEpjHq658Q/Vo1cw5KmM2jvKwi5+rwBbQURSQ6qPrwAAAAASUVORK5CYII=>

[image23]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD8AAAAVCAYAAADxaDaPAAADkklEQVR4Xu1XTYhPURS/10eJIlIWI2Y1KxJJjBUbLGSlSFGysbNgIUyiYUa+UmzEDGoYlCilUGhsfFuQz5XFGERjoWQxfud+nnvefe//hmbFr35z3/ndc84759777swoZaDtYEb/7J+iXY46PnmESF7CnyAXJ7TU1FKIkLq0c4pBiVyFeiGZ1akM5M3RQ865TB8xDONlsrYktG6eun4BNQJkYYUqZQ5p51Duk2RnRvmbpColZ2AYi6EdvO74CjwNNjvP4UGr0YZKrQMPykmH1Y7XwP3Qz2Fc4iezxWcR268bYfG/eYMjMBazmWngU/AlOJ7pdbAFPOv4HHyQThssAz84+vzTUcMgapoZ3UYeE8FBcLPQ14ND4EahByQrzY24Gb3gw2gG3FBxgfhkH9hW8Pa2lDMwLswvDSkmmAwOgKeEvgq+1PyOVM60UszpYZrnAi6rUXD/jphDhl63wxXwrtckGjVCyKt14Ppyq4dPwTS/NPGpAbY8onmjTgGH8NRJjHMGl8D3QmPQF/DjJ3gPsWsw7sTYS1T2rpoKblN2w3qUvVM4muB7AuNebMIZMRfQDP5QNmkFXIvly03Fym+eCqRFpcZD8y7FefBLENgqMhxX5jPVK9jbiQP4+UjFO2Q5+E3ZxfaOuzFs97srQbc+8Q7YjekxTqdEdBKq2OZ8OWjxqCAOyknNH3BUVIgr5SL41j6WYh/4UYrI8UbZOry9SNn3zIqaif0Kdih7pyXodjxsrOLipAgrWOpIR9I2n7p8gnDMMsFV8L7QLGL8HpAajbqdo99M7cxxobLNz/ECMA7s0lYnBtDO0ZEierSCW5ldQGnbFrTzj6WXtt82fRJEDrofOsijEOEVbZqPpyNOvVb2bxUPNK/pbpnLoqnHCc5Y4B3Xgi+g7SKqyMvgSu9kUrBMEhmZdv5JsKLDbGUbIE5yWgv4GWwiw79G5tT2yL7zBvOi0xCb12bjqPl5QcNvEy2O+wzwlzLHQDuGY0HBLcUSKnEUvOlIFw7lua3sQnJQca2abl9tPrM+PM8XPhInwX5lc95CXXQX+XeR1o8zQ8d6E56fOY3GDYZadWHEX5K6R7v74Z9uXrGjk0GZXheZ+IJUEApo7GHRyM/cJomTN8IaWKE6UfVsI1Qtdw7Bt0YQayG+SMTxf4uzaDRPSHzqBNRCJlFGagzeOU+gi/mki4cI+7uGs/42ayF3FsKjKijjyvEbdoC5C3aw28cAAAAASUVORK5CYII=>

[image24]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABkAAAAbCAYAAACJISRoAAAB6UlEQVR4XrVUzyuGQRDe8TO54aiUi5NyduCkuIu4SkocXJXclJKbg4scOCknDuIkpaRc+P4MB/XlIDG7O+/uzOzui/I99ey388yzOzOv92UMAxB/DWcunBBy1shj2muLA89B3uKhLyxZUz0qdbmyEvHjpAHtyAUp8emyZwj/VsSFtES1DdlA4SRbh8Mb5nBp4r7TUeQEWBUws7i+IT+Rw9HjcjkcoH4jlFx3rEQ38h45gfxC9TK6inhE306I8o1wwAYuSxScgStkJpmB4xh5h2esxzZ2TRwSg8QAeizBdWU6SBxBfiCvyJ7DDDXSXwlxkHSkdeKqTMIhXTLumXx428gXKVkIlwt6jZ/A0r8d0TOIfDfxUegGL1A4Ck8kbT5gE7lomXi8sGenoYmmgwz2VYdX3K0Ev4C8rXVFyNiH661OqmIDxn83lg9MH6PCozbA/amlqR55BOxi4hw3W4LgfyFqT0R76Ty1OEVxF0bL+LtG5ADbYdN4o6PrDNg+S2jQBfbDtZM9I/d94dz/LLUTSlyq839AsVgd6qpoXccB6ajJNFkkL4lH7lzoM3vCFAwFc+KrEC5hsZICsiJHYbqWFtGT6bgSZA+8A21WcfSIuipOc8k5bXEQYupISqSWCG2u86Yo/fF+CeruG6yxTWcBndlIAAAAAElFTkSuQmCC>

[image25]: <data:image/png;base64,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>

[image26]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADYAAAAVCAYAAAANfR1FAAAC6UlEQVR4Xt1WPYxOQRSdG6Ih1l9QbFgiWioKsQkNCQW6TSREyTYaNFR+ohChEn+hEBKJQohCUKxtbEQUqCSbKCQiFEoRzp2538yd3+99X6LZk5xv3pw59757Z957u8Z4ULisKgxWZcUOVPExaisVvSJn0D5Vjp9XE9X0FM0kHVELr+kp0sZ6SJsteRg1PUVfX7oZfQMEBV8sJYaCX1AIq5u9oRBVjGOpIHeDTltLFDR7tV14G1zklyLUMnmsAu8KX4LvwXOIWRgszfgMA7l75rne2F5ccSHPhH/BJWIJriKitQWYPcK4UsjYCv4yrtEKWvm7o5VlUsiNLQ1yKyR6r/aAnyHstLSrFjeMy7nBO4dEq5II1hjceWNiSBOmc8EO6H8wHhI6kLlkXM7xWqCFrKWWdO6gzGVDANYnmaZwYiFWZcoSZoJTyLzDDz+OI5V6FoMfjbvvBfAEeAp8IcQ7ajYigrWzCLyPcbONDInGwavgefCeVRSOCfu+Y5mSCR4TWON8R+PNKYDMB/x+AVeLsMnRxl8LRlv8KzVn8AZsSzQP/Sguk91m7AYvN0nmjHXGGDWu0MOJnkFanQIfqPuOMcnVc8BZ7NpJ/H71c4cp+D5h5Dp2JWuFxobDiCO9JV0QI00az/kUboYpjTraenSx/Kh+U3OA1iDVtHs6iP0RVGO0PF3siHngU+FBVTe/H1vCtACyjd1SwlrHXmM+Gxqj78keXZRxPrhPbZi9co25d2KFX4rAvvwrqYTruH5iacxpS7LjDPEf/SwwwjQS8T8HAhpzTE6MeJPoh587/ATXp/mPgM8N/w1y5EQzViOzThvzxqJME4ZjeWPc5mjOWmexMeJHl+//G+SCH8O3H+MbIcfzV/O44S8fmVnR+IT56wjSa4wPwSvgHSOYs41ZuJLjeQ2tNQ29AfXG/ivUHaUCXUixnoJYkCqioLXWB0OHZqc3dKbBoE954Jv2Ch0wrIDWkRY2J0VR7IJ24D9maZVX9CJGtgAAAABJRU5ErkJggg==>

[image27]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAFIAAAAVCAYAAADVcblPAAAEaElEQVR4Xu1YSageRRCuVnFBRYPLQVEE9SRCQA3JSVzA4BIXVBAUUcRDkEcEQwgYScQQN1AURXEJUZ7e9KQhioJevCiCoidRUEnUQ4J48iDxq67qmerqZeYtx3y878101dfV39TfM/O/R7QkhPjjkYfSiI8V8bJhasVTN+4t1csleI2OfbgPY6zraSKR0k1ZRD9bIBlSj/lKHj7uxwadlCAXeHnzWouAR0sQWglBP9tGc17zCibA+rD0aW2YSlo7D64EM8rMkEzAXcBqQBvMe6K/L1xybGCGJ8FFxJ9l4vwN8OIhW5mgsavAh8DLNHI++NooUqhZHyuDEeuReBvH032imDB6mOkjPR4cqsGZcHOPNXK5MHO5iR+Mw4g7wO/B4yZWeRQ8CslRPkJ5BPrbxnRtbjN2M7gX3B+4JtGZLt8CexAfI49givHhUCnX/lzr8NqTwH/Ah+NorHYyiaHrVGegItHxBSyC28F7wYsGWQ/eRT7eTPLhnDVEvD5HauQidNtB4yNNbLWpFqtjSsm3BDeMd6BbLhzGrz22QMobzQJ+3TkMY6h6WsDWcTppJJlG9sEeFlAk9zGg4qLV1yZKcR4JtInE9C1ZXHAI3Ne4WEGIF/EquA18HfyUzLO1Ooc43spElDsyoj4H0QUmdXw08D74L/gleBe4Q/keeAA8G3wMfJziXRd4p1ucR3HNsJtd8E7kRt6UawL//IaTD/O4Rbyw28EtZvwc+CvOTmMmZYZ6P0YEaSRka0ysB/YABvUREX0gNuXhJZz+jeONYzLiIDTf0ngN1wZ5BJ6jZDxB8kiLuIakkZsqZv8A91XiJVSDw/Uku+nBSKKNiL1AkUGPjoF2yWxBGG9t3hEOYaqpCdEHpOyhh6fAv7KIfJn8EXzZRK8k8cRHJoPnHob6aR5cTvGNG+7WpAXewPS8DxqcAr4F3mNi+OoSF0y3SYEZfSgb6SaZYfIw+NAv1dZHDzvBn30Q+A58Jg1Qci1JvXVKxokk63KcToDsd/ARTSbwdzjZqWq7aECgC0g0vLMS+CsHx25VLgHDCqmR6RZqIOqTB94MLR897KTYyOLqfiDTSOAKknoblJgRduFwqtHQfSjzuQ0A94NfQX68jrmxH4FbR0lcnR/Ul6oRfCj0CfiF5mKwsDhgkHikRp7rE8DawPUD3cCDMHpQHxHeRw94UdAvPkhya9tGplvbNDK+Px4YFLoSN+5F8E0lN83uiDPAP8F3TYxxIfgxyVvyG/AVcE2rRbWYAT/PuM5PJKa/1jHzEtVcjRr/UfreywjwIDQ+QvTRWzDIX28HSdb6LMjzfCNJDV6D3xHvkPjiFw/r+JZn8l9Re0n6wd+jjzWSVquRq4lQ+/eJD/kxoxbL4ASTekH8j04eGcYzSwhq19UHTxgnFdOLwMrRLNlMKHKrS4fOjYeV1Ckxp5rRWHlxXvlAXPlaH4ZxGPNeszysoFJvWi+XMEeTRNKoWTMUE11N5z7ew9z1a7JazCLVbvj5H0W91IamyVvgAAAAAElFTkSuQmCC>

[image28]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEQAAAAVCAYAAAD/wUjgAAADx0lEQVR4Xu2YS4jPURTHz9WwEpKNvDZekYWyYDfyWijvkPHYSnluUIYhI4qwkUeTosjGWChKysZjlJUSpow8NzYoE9L4nvv63Xvuvf//f5DVfOo7v/8959xz7z33/u7/3xD1G2WVI2/X1rxLoKKwuFWNHMeEmB65oXI2STXPYnTRkaE/sf0nyZ4YDGFBqhBZNvvHmpWSpf+PqHS2AuPwk66FigtQj0bjavAPUljSTP9wV2xhyhUSRteM4pJO1AZdhY6z4O6ApsYhabegPRY6As2vTDlkBoGiNfh7BXoJ3YLmBb5i94K5aA9xMTK2jQYK4jkI3QzazCIE9EBDhF0yDToDXYT6oJb8JPJWJljnMnzaZ81NsF3A8wdss62tn+THjKyZEF7wV2iLsA8iPRlake3lqFYziUxB1ofuRggKch+6DA22rhFkcnbWnIOmnt9SDFPeN4vMoKtDt+U1md0vEuT3BSmOSS6+eKHehl7BbwqidGF68XwcRXkKWRzeHY4X98llWEpmIculQ5n3+IY0JpjiNnhCqp1ogAVkcp6OrMFCSd97qhdPLhpOMx2w4pN2FxoF7YT2QpegzbprzAnoKHSKG3hv9aBcGIF6hjE5aWzNrAiWiWTvEOnLke5TmhM8gN5A47hRjDIF+0L+9fZR3fj0gkxRmOnQL2hyENYM3Qk2Ss0lsxCurqSb5Akp407IBm9RKBKL6GRRSj8nWIXsh54giS4G49cgqoLmITw+k7n3Qg+fGj4pDs7Fc1wc2Nwp7IC2s2GGNayTA5HZHb7pG8EWRFUFcbi8aX6/MXZznJW/qR7iOdwZ6tAKvZVGZU7YWd8yPw14rUv4lAcnfSv0zfqoiUyyHc5rYft3qnsneNwJ2SgdGvebPd1dSTP0CBrpigWuZyM9qhXed9IKukgXxPd1BeFrwsF35xz+gKgx1qbW4s+9eEi1CsKRtb9DFA0l81tlTxgV4AqyqTJxxloLIe0OdmsK8TebonZldp3VDp0Pu0SY9Ieh97FDw6+MPiF2FuMpLQi/JnzZJrSgF9+y56z4K3B0sJxh0EekvlaZNDNJX0p68D7k6LFt8eqYTKLoUQs8JTNhqV3GnS0wv9IfyMTxF8BCK57DT+gT+nSi20p87lIm7jm022obmYLwZh+jgIGCiIKk2LGTKQRonw+oFVmigT6lkMK9VBFvQjzXwFY0+R6ZsAD3xsuoyFZ/4JrUja8bIFHZPv7/MzlKztIi/4RcIXPoGBkoOpuPIigZQCb5W2y+NG0ycoY4pl60pxTYyJAJjY/P/t/107ExrJTnxwAAAABJRU5ErkJggg==>

[image29]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAADMAAAAZCAYAAACclhZ6AAADN0lEQVR4Xt1WzYuOURQ/NxEzUsT42FiRULZSZEfYzR8gNhY+RkmMkigpCzUUCllYGBvfpHxkoYSFJGNINhSWFhaUxu/ce97nnvv5vtOMmuZXv+ee+zvnnnvPfe5z35eIYezTQ/q28Y80ziIUuZcNy6DTOIsk2M2kl+d1aVKnEhK9glJsSVcob4hWy1GUOuJ+Xiqi0/qTALfbVs74QjsKiAatF74FpzeqD9gAvhE+g3wT7aIwJERBzqMWrBcax7X6k68YovPgU9gPmBBHKCjGYgn4FVwq5MGb8HgPa1YriPPFc7Xg9c6ixoo+oS0mSssFD2YW8gHcqQWLZE2JEMB6JSSZIYv2UUExVvHxP2CfCpK45glxkfW8EVwOXUAjjy5RBn68K0YfM+ebaVyBx5vIBuYuHs9La4B8Fc1vckVvIz7GZD470nWwB/Y5tI8khr9LjS7k2IW2H7wC+zba3nRT0wXsEUbHzMwXLVOMTc5HzUJN0UI3BF70T3Cfc5vZjvQHfEnue2TsJnf56DQ7wINiM47C06v6Efzs6TGzMHNMuZg74LAz491p+jyOi5kShXwHz7Ih8hZYPM9cIeMS+ApcI3HL8FguPpUunjssZobSOfIveEILgnvE13QdXMxwPB3639AcUtJmcAT6PKZoq2H/Yh3km/M0Bk5tRlQwaYvpinyvkWQg0hj80V5sdXjB8aIhHMNzKJaBL/D1qxH88fPcC4QW8K7E4zDMW8b597d8KfzsqhijirEBA2gGvdbgE7xbkwp039g3MxSHkHszuhj7Zigs5hqFP8pHyN6C2W3zgKuPSe5q7g48RKvAj+BipqRZR7y7zcdaTG6PWSxS4ZiBC4WAuYHHAR+CXPZtlrEdfAjy3c/khC9E61Fxa8HLwvtIyrfMCuXP4QK5RXNOzrcRfMyUI4O/SOYM2r3gO4nj65rJxfFcvHi+9fAbRSfBaaRR3MMG8gstttazo0VqvJmQzqAGSqJavppvnFGaxZVc94b9MUJtdSZdqjDiXZ1AaLueer2jQvwmfD/2lFDzUVv3BC2mFvO/UdsJvVmRNQaMR5J6jpq35iui/Ho7R2c5VFQppAY1xprS/wd9V5LWYbNkOwAAAABJRU5ErkJggg==>

[image30]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAE4AAAAYCAYAAABUfcv3AAAEzUlEQVR4Xu1XR8hdRRQ+YyQaYsGIPbEsVRQsCxeixEKsaCyIWH+QWBEEsxBFEOJGRURBI2JEY8eFiiIaYwWxJlixgSCoCGJdiIrE8805986ZmTPv3fe/P7jQD753Z06dOXfuzDyiuUAoGn3f7eaISvyMc6oECj/n5kGZpOy7GDJAX+lLh2Ea3wrNYA2FK3ZesGtXIbfqer40R2jIZ4OpIw2fcI7kl3kfxnxHudgqFLsyXybRv6FcNqsBEB3FfIv5OvNJ5WmdMoUMV/HPRyS2a5Tb9uo6+YvMTR1Z+zs/71Wze5i/qO4P5QviFvGh6sBnRWQrXOWiO5ivMZ+m5Li4MNuCZPAXa38n5efMEzqjBDN1I1XswfyepHjAsWCQvOeLKHot5+f7/Fykdlcr8fK2rCKn7onCgHg3FLpDSfI8pLQIbHY/P88t5FTl6iHyGUqFW2LVjNOZP5MU0GIV+2LlJbRyJGBiyLFaTMMiIWRhvbHbyPqVpr9ACV/EoEaybZRYUe9FSTLbitt/8vMn5fwiwvPMeV5YR0T/Fy5hwsIJRhXuQeYnsZVHuJTEfgdH18JCtruGn/tp/wAl4twmorCv9s9RG4tvmbeUQgcYM2LsY2TLmI/wN7kJ5PbxRrcnc7Xp+yjnGEYXDpv4u10Hvup/IcE+0IGdbkJsz3w8MtDH/NxR5ceRjGN5Z2jwBfOxXBTqCclXghiXGNntzN1VDt5pdFgExX7dzbQObmELl05V8dlIcpr2Ig2FzRz2SztdhWbOAN+3SbYA8BBjegZJ3FOTqMdn7LuuFDp5cPric31KVfMIJ6zgVeVXlDwfZc7XthGPRbCFW1L4vUlauPw+Fs4LYn+kkaVmBVd3pvJX5jEqO1njeoX7lPJrREIdHteKv5jbMY9grohnJ9G1SuTANQwr/wH1SYBpFrNOQFI4HOHxGMf3bvEcc4Pjpz60P3Mtt9fGZ0DbUuUUToJTHSbiO+aPJBv74RQnFfC5lfiS+XBsOYGKua4gKc7ZzJspXoWixcHCeNe7kWQv9XINwgzesr7pvQrdrRQ/kQqXMXFKbZ1GW72mErgnYePWl9Pbv0JSrINI7m4Yh97rDEIs8PXSzBUOdmP+TXJne6LQAd+QzAsvdoEbQZFNr8AMpU+1LNwpzN/6XqrNKma930Q0Cwh75IirzwCfICa5c+yFuP+tNDG66whslkJuo7uZBOuD/IO4Dp3C7i6SsawZ4a/wLERmC7e30QILSU7Vs7SPPQHEzb6TKbwEGXA3+4AkJnC0Eqv97t5KVibuiPh0gYuUOKji/WtkpqS8kiQ2VnKGIP9aoMMeOzEu4Ajrgvx96gqHwWFl4H7VYReS/3u4CmxQXhE1I2dgEQ1xul3O7WdITjYcCuBNhAtqDhTvPpKxvKTEdUIwLC+uVl+XQgX/daMfKPv/+29Bl0JaEfnshs11M2LkANKom0jqMYYVRtgb1YAhNDG5X8ujMQojdrQGDX9BU9HACPv/VuF8jDKvwvWdUV7DMTdREsbH8yxKWdn3RYJO4VfJ6dUYp7eYxLaD/BHoOhKjFzkBHdH0cJPGp4xodkmN16wCOE7Thmxg6lh1gFoyCbIXMgC+bRfF17bEk2NooKZdUzHH8IuRSWq1oqnoMd5icvwDFaLowp7WiKkAAAAASUVORK5CYII=>

[image31]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAD0AAAAZCAYAAACCXybJAAAEQ0lEQVR4XuVWSeiXRRh+p6w0LaRFo4MJLXSIImjxoLgcLAQzIzpIUXbIFtovRoKo4dYCkkiHFFEvRoK2HRJM2ohWLyYu6cGNCDx0KAjCnmfe95vt237/n57ygef3m3nmmW/ed2a+mU/kXOBKwUDd2tosATVDTWhFMkwuDoRBjYP6Atoy13C9XGsbHrVHlUJRvx7KLaFWmnsQgy87lvUGqaz3oMve1VaHk5fxexY8ZdwNfoWHvJL5RGaD+4zfgDvFT5ikI44Ctxjp+x5ti0KrB8220oOiwaxPSWsRDXZD1XJBJi3yKvgX+KfxI3BeaBVvvRF/J8BbjcR94EFwfOUD3gU3GC8Gx4Cfg8/nATaE5SXTS0sil6UMLbJH0cak55fjeESBSWyvy7IffNHKk8B/wNuNCicL8PMHShcFbQSoxWToibcXL4EPhYc0dzwNfV2mqI+vwoemPCv6mlxrrDDD9DsT7Tyj7XVJpqYy2P8L4AaUd5IoHwA/Qfkas10uGvRqqycDuI/x84MJq4Q+J2M9I6bAx/4PJlqK6eBJ8WM47ApMopO9niKHwQdQnoN/jvUzypvguzR295iBEBbjfy24zak3IIk3lJ4Dt6PqSGvZCn5m7deJJr3S6il2gb9Zeb1TH9/jMckA91jST1ZCDu+bJdr3F9HxKnBMTsiboofkJeBR8LHgcP7s+BW8zJQrwa9De4aYNA+iKwr9EYlb8ipfdo1Jc2ccsT7vSJZ0GOJu0xdqtRGjYaZncQzL42nRvpOSgL8E3wsOkRtEPa9JzGN+kl87qiW22ixbNb6nlP8F14TmCG6j77ToXrc+44wVsL29PjfRSowW9TxR6E+ZnoQmeyRPmuAhS9/f4Bfg1LTRd26ahAspaW7dI+CyKBGOHyLc0txeBN813sEleHpvtvLDogNPNCqczDT9pqAVcPq+0vN4JViQmnS+Skz6/VDTNr7XjPUt8Dj4O/R04g1qniA6GE+9FAtNv82MvK4+yBwKnvTVAcXTngHe4RnxKHgsXaps2rXMQygmHVGtdAquZExaz51ws4gu5Gnw3kSrAVeA3BxqzgewF3w7aOKTOAROJn2Yzm8hnqycOA+nz+IErrVkuIL8wltSeZrg4vbmZKd4xuv+Gg4TxYNsY7TIXeAx8Gqz8GBm0jzFW8Ft8Cn8u0mUf8IAyyVfDmKa6Fbe7PQ6+xGW+OWlYJI85cltwm9vvW7KZ6WYDn4rmjTv5aXgG0a+ekyaV9D9ovHxq+8MuMPIpHnFrhB/hfpvh6m1Eat6qrPs66W5QpvegSG69GPYh2b9LNM86bbPuhLltKUo6w3o6j4idHeOrZ2+POm8k9XSv9Jce3ZNyNAVVO35XRjUNzR6B8gNXfbexLraBkDXpI4Mvf3/j0kPg5CdDd4QQ2kZGsP0H2b2G6Tzj8ZBGsQGqSZ2TW6pl/VmmGswc7evdwUMLUOm9f8AlZvFree/OVUAAAAASUVORK5CYII=>

[image32]: <data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEcAAAAZCAYAAABjNDOYAAAEqElEQVR4Xu1ZW8hmUxh+V8ghjJkJEUqUcWiGuZAiLpRciRwKZaQ0NQwupsYkRUioIeNUJiIljKnJaTQuxOQQmZErjMOFUEoiF7gYz7Pfd+39rrXXWt/+/4ar/6nn22u/h7Xe/ay1117/jEgVITcYaPe+Whxhvu7SjpvZY9EoSTkzhzHMHGsK5pPYymn5RpgZ7BSpqVIwzRtuEhrIo9oZZa+ulLk/U91TRDG8aJTxc41xfscgn+N6WO6E/Tj8vmf8GPxANMf3uz/4vHE3+Al4UxrS4SIjY3bC9zqux6chilHJIwNRNO5TLIgzQpAn8fsu+GbHIHthVHHSh/4MvJ7szEGOxe/34Ll9lMgm8CnjfuDBov2uc+WfhPaPJNrLzHYh+vsG16VDWBvT5JgWlaCUAtsaUsWRRYlH5GrwF7vx6RRjm1k483+DZxojrkLKr6ICA+EJ/Lxi9OCKXTfcuqHSy9DOqzEUTAVMi4pYY8zE6fCiaPEZwgb8/C66SmLu0cY4PFcW7eforfwkKio5IHQr7LXENgu5OLV2BRNCerTE2YmePsxsxK2i8UeB91ubr6Tfs84y+5XgIdZ+0OixRXQPqoET9Bfq4J53Hbhd+Crq6/gGeCT4KLgDfB+8rM9UcOybQU7oc+A2iHPtJIVCfK20+MMz924EcAPOsTZo/KmihbF9qDFihdlXow+KuBc5D5AuhngZVfyQ2VztgQ+3FQau1NvNFcf6I+jKPt0yVonuhwdobofV4AanxR1CcRwSnTLR3MoJunKGAH51SuLcIoM4G7t2QLGk9Cdf7j8qjshitnNxLI570EgcD8Tdh8ufEh+aiV1y+E70CxnN/OpyzJP7EJHNaH2K63kWRt9ya8+EiqOdHuEd6ORtXD7yNsNtog/OJc2ZiKuuW3lW1EqzX2Gmf8CHjB6vwr0rm7AcFGfPaFol7DHhIigAxzzF2c5GHIXlBH6J62Pggc7fxII4DfgNeXE2/CPo8AstKvFwc+N5hfbLg+YeY4yIhZ5h9zwvPW4U9mc9viW66bZwN/gVG1l9tBXECfEcFRNYw53C44fWxAltYBglEae3Ki4Ff8tsxKYwPNAS0Vx+ncgInHO6Q1/ERrS3kINJWAc3VI4/xpB8r5g4GfjFKogjy9x0vgQucnVwYnl8SBEDMvW9OHxQD26ynPFLjARfJRbKPwMinhbdmEmC55+t4D19hM7e1x2DnGg2noF+RkU8dbdgr9UI30oqzgWiz3Gas/G15VcugivoLndfxA1I4tmA6pNcjvw67UDxJ7g4viqbjZx1iBVWOT/BUzCLJF8Q3aceltE8dDNLPis6e/yKrByHJaDwfIVZ3zu4XgzySlII+p4BbxT9rNO2C7zGyLrXgzyhbxedwINkNppFlTGnlAnBE0I6TIhrhcz0tQLazv8Go6IqNQzmSoDB98dLLa/dS49KWMU8YGbAJEzQRaZG5UjFmYqY4WXOUTF7TAipY2ryKG5kUFTMHVo+jyRuQZwUMc7Hs53fl9r9fW6MaHVUwYSQMvKxPCr2ijlDFjWPceaOlqr7QtBRH+4f80coW/8npP8fZab4U76fA+pZdU+xJgfv+xeNSOwcLrPzVAAAAABJRU5ErkJggg==>