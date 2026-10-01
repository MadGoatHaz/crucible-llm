//! High-resolution timing core: a `MonotonicInstant` wrapper over the `quanta`
//! cycle clock and the `T0..Tn` per-stream timestamp record type.
//!
//! `MonotonicInstant` exposes nanosecond deltas computed from the `quanta`
//! source clock (TSC / System Counter, falling back to the OS reference clock)
//! with no syscalls on the fast path. `StreamTimestamps` records the lifecycle
//! milestones `T0..Tn` (blueprint §4.1, §7) for a single stream.

use std::time::Duration;

/// A monotonic, high-resolution clock instant backed by the `quanta` source
/// clock (RDTSC / System Counter), falling back to the OS `clock_gettime`
/// reference clock where the CPU lacks invariant counters.
///
/// Deltas between two instants are always reported as non-negative `u64`
/// nanosecond counts: because the source clock is monotonic, a "negative" gap
/// can only come from a clock-source hand-off anomaly and is clamped to `0`
/// rather than surfacing as a signed value, keeping all downstream latency
/// math in the non-negative integer domain.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct MonotonicInstant {
    inner: quanta::Instant,
}

impl std::fmt::Debug for MonotonicInstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MonotonicInstant").finish_non_exhaustive()
    }
}

impl MonotonicInstant {
    /// Capture the current monotonic instant.
    #[inline]
    pub fn now() -> Self {
        Self {
            inner: quanta::Instant::now(),
        }
    }

    /// Elapsed nanoseconds since `self`.
    #[inline]
    #[must_use]
    pub fn elapsed_nanos(&self) -> u64 {
        u64::try_from(self.elapsed().as_nanos()).unwrap_or(u64::MAX)
    }

    /// Elapsed [`Duration`] since `self`.
    #[inline]
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.inner.elapsed()
    }

    /// Non-negative nanosecond delta from `self` to `other`.
    ///
    /// Returns `0` when `other` is not strictly after `self` (the source clock
    /// is monotonic, so this only reflects an ordering anomaly, never real
    /// backward time).
    #[inline]
    #[must_use]
    pub fn delta_nanos(&self, other: &Self) -> u64 {
        match other.inner.checked_duration_since(self.inner) {
            Some(d) => u64::try_from(d.as_nanos()).unwrap_or(u64::MAX),
            None => 0,
        }
    }
}

impl Default for MonotonicInstant {
    fn default() -> Self {
        Self::now()
    }
}

/// Per-stream lifecycle timestamps (blueprint §4.1 / §7).
///
/// `T0` socket start · `T1` request write complete · `T2` first byte ·
/// `T3` first token frame · `Tn` stream close.
///
/// Every milestone is optional because not every request records every point:
/// a non-SSE plain-JSON fallback has no `T3`, and a connection refused never
/// reaches `T1`. `None` means "not recorded".
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StreamTimestamps {
    /// T0 — socket connection start.
    pub t0: Option<MonotonicInstant>,
    /// T1 — request write complete.
    pub t1: Option<MonotonicInstant>,
    /// T2 — first byte received.
    pub t2: Option<MonotonicInstant>,
    /// T3 — first token frame decoded.
    pub t3: Option<MonotonicInstant>,
    /// Tn — stream close.
    pub t_end: Option<MonotonicInstant>,
}

impl StreamTimestamps {
    /// Create an empty record (no milestones recorded yet).
    pub fn new() -> Self {
        Self::default()
    }

    /// `TTFT` in nanoseconds = `T3 - T1` (first token arrival − request
    /// dispatched, blueprint §7.1). `None` if either milestone is missing.
    pub fn ttft_nanos(&self) -> Option<u64> {
        match (self.t1, self.t3) {
            (Some(t1), Some(t3)) => Some(t1.delta_nanos(&t3)),
            _ => None,
        }
    }

    /// First-byte arrival in nanoseconds = `T2 - T1` (write complete → first
    /// byte), the network / prefill hand-off window.
    pub fn first_byte_nanos(&self) -> Option<u64> {
        match (self.t1, self.t2) {
            (Some(t1), Some(t2)) => Some(t1.delta_nanos(&t2)),
            _ => None,
        }
    }

    /// Total request duration in nanoseconds = `Tn - T0`.
    pub fn total_nanos(&self) -> Option<u64> {
        match (self.t0, self.t_end) {
            (Some(t0), Some(t_end)) => Some(t0.delta_nanos(&t_end)),
            _ => None,
        }
    }

    /// True once the stream has been fully closed (`Tn` recorded).
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.t_end.is_some()
    }
}
