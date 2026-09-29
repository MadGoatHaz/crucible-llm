//! Hardware telemetry. The NVML poller is feature-gated (`nvml`, default-off)
//! and degrades to `"N/A"` when no GPU/driver is present.

#[cfg(feature = "nvml")]
pub mod nvml;
