//! `ArcSwap<MetricsSnapshot>` double buffer: the stream worker / engine
//! pushes a unified snapshot per batch and the TUI reads it lock-free.
//!
//! This is the measurement-isolation seam (blueprint §4): the render loop
//! only ever calls [`MetricsState::load`] — an atomic, lock-free read of the
//! current pointee — and never blocks, never takes a mutex, and never touches
//! the quanta timing path. The worker writes via [`MetricsState::update`],
//! which atomically swaps in a fresh `Arc<MetricsSnapshot>`.
//!
//! A snapshot is plain `Clone` data (no interior mutability, no locks), so a
//! single `MetricsState` is shared between writer and readers by wrapping it
//! in an `Arc`; both `update` and `load` take `&self`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;

use super::histogram::LatencyHistogram;

/// Per-stream / overall run status (blueprint §6 stream-matrix `STATE` column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StreamStatus {
    /// Awaiting dispatch / first byte.
    #[default]
    Waiting,
    /// Actively receiving tokens.
    Streaming,
    /// Stream closed (`[DONE]` / `Tn` recorded).
    Done,
    /// The stream failed (connection refused, timeout, premature close).
    Error,
}

impl StreamStatus {
    /// `STATE` column label for the stream matrix.
    pub fn label(self) -> &'static str {
        match self {
            StreamStatus::Waiting => "Waiting",
            StreamStatus::Streaming => "Streaming",
            StreamStatus::Done => "Done",
            StreamStatus::Error => "Error",
        }
    }
}

/// One row of the Active Streams Monitor (blueprint §6, View 1).
///
/// Travels inside [`MetricsSnapshot`] so per-stream detail is part of the same
/// double-buffered swap — the UI reads it in the same lock-free read as the
/// aggregate metrics.
#[derive(Debug, Clone, Default)]
pub struct StreamMetric {
    pub id: u32,
    /// `Reasoning` | `Content` | `Tool-Call`.
    pub kind: String,
    pub state: StreamStatus,
    /// Prompt-processing (prefill) tokens.
    pub pp_tokens: Option<u64>,
    /// Token-generation (decode) tokens.
    pub tg_tokens: Option<u64>,
    /// Time-to-first-token, in seconds.
    pub ttft_s: Option<f64>,
    /// Generation speed, tokens/sec.
    pub gen_tps: Option<f64>,
    /// MTP multiplier (`tokens / packets`).
    pub mtp: Option<f64>,
    /// 0.0..=1.0.
    pub progress: f64,
}

/// The unified metrics snapshot (blueprint §4.2: the Engine Core "pushes
/// unified snapshots to an atomic, double-buffered state cache read by the
/// UI").
///
/// This is the *only* data the TUI render loop reads. Every field is plain
/// `Clone` data — no interior mutability, no locks — so an
/// `Arc<MetricsSnapshot>` can be swapped atomically and read from any number
/// of threads simultaneously.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    // ---- identity / context (status bar) ----
    pub endpoint: String,
    pub backend: String,
    pub model: String,
    pub mode: String,

    // ---- throughput ----
    /// Aggregate tokens/sec across all active streams.
    pub aggregate_tps: f64,
    /// Prompt-processing (prefill) throughput, tokens/sec:
    /// `prompt_tokens / TTFT`. A key metric for AI rig builders — it is
    /// distinct from generation speed. `0.0` is the N/A sentinel (no prompt
    /// data / no TTFT yet), rendered `--` by the views.
    pub prompt_throughput: f64,
    pub active_streams: usize,
    pub total_streams: usize,

    // ---- hardware telemetry (0.0 == N/A when no GPU / feature off) ----
    pub vram_used_gb: f64,
    pub vram_total_gb: f64,
    pub power_w: f64,
    pub joules_per_token: f64,
    /// GPU core clock in MHz (blueprint §6 View 1 top-left key metric).
    pub gpu_clock_mhz: f64,

    // ---- inter-token latency percentiles (nanoseconds) ----
    pub itl_p50_ns: u64,
    pub itl_p90_ns: u64,
    pub itl_p99_ns: u64,
    pub itl_p999_ns: u64,
    /// Normalized ITL histogram bins (0.0..=1.0), low → high latency.
    pub itl_bins: Vec<f64>,

    // ---- token counts ----
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,

    // ---- overall run status ----
    pub status: StreamStatus,

    // ---- per-stream detail (stream matrix) ----
    pub streams: Vec<StreamMetric>,

    // ---- rolling aggregate-throughput window (one sample/sec, last 60) ----
    pub throughput_series: Vec<f64>,
}

impl Default for MetricsSnapshot {
    fn default() -> Self {
        Self {
            endpoint: String::new(),
            backend: String::new(),
            model: String::new(),
            mode: String::new(),
            aggregate_tps: 0.0,
            prompt_throughput: 0.0,
            active_streams: 0,
            total_streams: 0,
            vram_used_gb: 0.0,
            vram_total_gb: 0.0,
            power_w: 0.0,
            joules_per_token: 0.0,
            gpu_clock_mhz: 0.0,
            itl_p50_ns: 0,
            itl_p90_ns: 0,
            itl_p99_ns: 0,
            itl_p999_ns: 0,
            itl_bins: Vec::new(),
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            status: StreamStatus::default(),
            streams: Vec::new(),
            throughput_series: Vec::new(),
        }
    }
}

impl MetricsSnapshot {
    /// p50 inter-token latency in milliseconds (for display).
    pub fn itl_p50_ms(&self) -> f64 {
        self.itl_p50_ns as f64 / 1_000_000.0
    }

    /// p90 inter-token latency in milliseconds.
    pub fn itl_p90_ms(&self) -> f64 {
        self.itl_p90_ns as f64 / 1_000_000.0
    }

    /// p99 inter-token latency in milliseconds.
    pub fn itl_p99_ms(&self) -> f64 {
        self.itl_p99_ns as f64 / 1_000_000.0
    }

    /// p99.9 inter-token latency in milliseconds.
    pub fn itl_p999_ms(&self) -> f64 {
        self.itl_p999_ns as f64 / 1_000_000.0
    }

    /// Display label for the GPU core clock: `1410 MHz`, or `N/A` when no
    /// GPU/driver telemetry is available (the `0.0` sentinel, blueprint
    /// §5D graceful-degradation rule).
    pub fn gpu_clock_label(&self) -> String {
        if self.gpu_clock_mhz > 0.0 {
            format!("{:.0} MHz", self.gpu_clock_mhz)
        } else {
            "N/A".to_string()
        }
    }

    /// Fill the ITL percentile fields from a [`LatencyHistogram`] (Chunk 2).
    ///
    /// This is the bridge between the worker-side latency histogram and the
    /// UI-facing snapshot: the engine records inter-token deltas into the
    /// histogram and copies the percentiles into each published snapshot.
    pub fn with_itl_percentiles(mut self, h: &LatencyHistogram) -> Self {
        self.itl_p50_ns = h.p50() as u64;
        self.itl_p90_ns = h.p90() as u64;
        self.itl_p99_ns = h.p99() as u64;
        self.itl_p999_ns = h.p999() as u64;
        self
    }

    /// Fill [`prompt_throughput`](Self::prompt_throughput) from the
    /// snapshot's own data when an engine did not set it:
    /// `prompt_tokens / mean(TTFT)` — how fast the server processes the
    /// input (prefill). The prompt total is the top-level
    /// `prompt_tokens` when present, else the sum of the per-stream
    /// `pp_tokens` (Engine B reports prompt counts only per stream).
    /// `0.0` stays the N/A sentinel when neither a prompt count nor a
    /// TTFT is available yet.
    pub fn with_derived_prompt_throughput(mut self) -> Self {
        if self.prompt_throughput > 0.0 {
            return self;
        }
        let prompt = if self.prompt_tokens > 0 {
            self.prompt_tokens
        } else {
            self.streams.iter().filter_map(|s| s.pp_tokens).sum::<u64>()
        };
        let ttfts: Vec<f64> = self
            .streams
            .iter()
            .filter_map(|s| s.ttft_s)
            .filter(|t| *t > 0.0)
            .collect();
        if prompt > 0 && !ttfts.is_empty() {
            let mean_ttft = ttfts.iter().sum::<f64>() / ttfts.len() as f64;
            if mean_ttft > 0.0 {
                self.prompt_throughput = prompt as f64 / mean_ttft;
            }
        }
        self
    }
}

/// The rolling aggregate-throughput window kept by [`MetricsState`]
/// (writer side): one sample per second, at most [`ROLLING_WINDOW`]
/// (60 = the last 60 seconds), copied into every published snapshot's
/// [`MetricsSnapshot::throughput_series`] for the render loop.
///
/// The engines publish *fresh* snapshots (`..Default::default()`) on every
/// batch, which would wipe a series carried inside the snapshot — so the
/// series lives here, outside the double buffer, and `update()` re-stamps
/// it. Sampling is time-gated (one per second) and gap-aware: updates
/// arrive in batches (every 8/16 events) and not at a steady cadence, so
/// `update()` backfills at most the elapsed whole seconds (capped) with
/// the current value and never backfills across a long silence (> 2 s),
/// which would fake a plateau that never happened.
#[derive(Debug, Default)]
struct RollingSeries {
    samples: VecDeque<f64>,
    last: Option<Instant>,
}

/// The rolling window length: one sample per second, last 60 seconds.
const ROLLING_WINDOW: usize = 60;

/// The minimum gap between two rolling samples.
const SAMPLE_PERIOD: Duration = Duration::from_secs(1);

/// Gaps longer than this are not backfilled (a silent engine is a flat
/// line at its last value, not a wall of faked samples).
const MAX_BACKFILL: Duration = Duration::from_secs(2);

impl RollingSeries {
    /// Fold one `update()` into the window. Returns `true` when a new
    /// sample was taken (at most one per second, with bounded backfill).
    fn sample(&mut self, value: f64, now: Instant) -> bool {
        match self.last {
            None => {
                self.last = Some(now);
                self.samples.push_back(value);
                true
            }
            Some(t) => {
                let gap = now.duration_since(t);
                if gap < SAMPLE_PERIOD {
                    false
                } else {
                    if gap <= MAX_BACKFILL {
                        let count =
                            (gap.as_secs_f64() / SAMPLE_PERIOD.as_secs_f64()).floor() as usize;
                        for _ in 0..count {
                            self.samples.push_back(value);
                        }
                    } else {
                        self.samples.push_back(value);
                    }
                    self.last = Some(now);
                    while self.samples.len() > ROLLING_WINDOW {
                        self.samples.pop_front();
                    }
                    true
                }
            }
        }
    }

    /// The window as a plain `Vec` (oldest → newest) for the snapshot.
    fn as_vec(&self) -> Vec<f64> {
        self.samples.iter().copied().collect()
    }
}

/// Lock-free double-buffered holder for the current [`MetricsSnapshot`].
///
/// Shared between the stream worker (writer) and the TUI (reader) by wrapping
/// in an `Arc`; both [`MetricsState::update`] and [`MetricsState::load`] take
/// `&self`, so a single instance serves all parties.
#[derive(Debug)]
pub struct MetricsState {
    inner: ArcSwap<MetricsSnapshot>,
    /// Writer-side rolling throughput window (see [`RollingSeries`]).
    /// Only `update()` (the engine / hw-poller side) touches it; the render
    /// loop reads the copied series from the published snapshot, so the
    /// measurement-isolation invariant (blueprint §4) is intact.
    rolling: Mutex<RollingSeries>,
}

impl MetricsState {
    /// Create an empty state holding a default (zeroed) snapshot.
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(MetricsSnapshot::default()),
            rolling: Mutex::new(RollingSeries::default()),
        }
    }

    /// Atomically publish a new snapshot.
    ///
    /// Writer side — called by the stream worker / engine per batch. The
    /// previous pointee stays alive for any reader that already holds its
    /// `Arc`, so an in-flight render never observes a torn state.
    ///
    /// Before storing, two derived fields are completed when the engine
    /// left them empty (never overridden when the engine set them):
    ///
    /// * [`MetricsSnapshot::prompt_throughput`] — `prompt_tokens / mean
    ///   TTFT` (the prefill speed, Issue: prompt throughput);
    /// * [`MetricsSnapshot::throughput_series`] — the rolling 60 s window
    ///   maintained by this state (the Live view's hero chart reads it).
    pub fn update(&self, mut snapshot: MetricsSnapshot) {
        snapshot = snapshot.with_derived_prompt_throughput();
        {
            let mut rs = self
                .rolling
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if rs.samples.is_empty() && !snapshot.throughput_series.is_empty() {
                // A writer seeding the window (tests / a resumed session)
                // is adopted wholesale; production engines publish empty
                // series, so the window fills one sample per second.
                for s in snapshot.throughput_series.drain(..) {
                    rs.samples.push_back(s);
                }
                while rs.samples.len() > ROLLING_WINDOW {
                    rs.samples.pop_front();
                }
                rs.last = Some(Instant::now());
            } else {
                rs.sample(snapshot.aggregate_tps, Instant::now());
            }
            snapshot.throughput_series = rs.as_vec();
        }
        self.inner.store(Arc::new(snapshot));
    }

    /// Lock-free read of the current snapshot.
    ///
    /// Reader side — called by the TUI render loop. `load_full()` is a
    /// lock-free, wait-free atomic load that returns the pointee as an owned
    /// `Arc`; reading any field on it is a plain immutable read that never
    /// blocks.
    pub fn load(&self) -> Arc<MetricsSnapshot> {
        self.inner.load_full()
    }
}

impl Default for MetricsState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snap(agg: f64) -> MetricsSnapshot {
        MetricsSnapshot {
            aggregate_tps: agg,
            ..Default::default()
        }
    }

    #[test]
    fn rolling_series_samples_once_per_second() {
        let state = MetricsState::new();
        state.update(snap(100.0));
        assert_eq!(state.load().throughput_series, vec![100.0]);

        // A second update within the 1 s period adds no sample.
        state.update(snap(110.0));
        assert_eq!(state.load().throughput_series, vec![100.0]);

        // After the period elapses, the latest value is sampled.
        std::thread::sleep(Duration::from_millis(1100));
        state.update(snap(120.0));
        assert_eq!(state.load().throughput_series, vec![100.0, 120.0]);
    }

    #[test]
    fn rolling_series_caps_at_60_samples() {
        let rs = &mut RollingSeries::default();
        let mut t = Instant::now();
        // 70 "updates" spaced just over the sample period (simulated by
        // advancing `t` directly — no real sleeping).
        for i in 0..70 {
            t += SAMPLE_PERIOD + Duration::from_millis(1);
            rs.sample(i as f64, t);
        }
        assert_eq!(rs.samples.len(), ROLLING_WINDOW);
        // The window keeps the *newest* 60 (10..=70 → 11 kept from 0).
        assert_eq!(rs.samples.front().copied(), Some(10.0));
        assert_eq!(rs.samples.back().copied(), Some(69.0));
    }

    #[test]
    fn rolling_series_does_not_backfill_long_silence() {
        let rs = &mut RollingSeries::default();
        let t0 = Instant::now();
        rs.sample(50.0, t0);
        // A 10 s silence: one flat sample at the last value, not ten.
        rs.sample(50.0, t0 + Duration::from_secs(10));
        assert_eq!(rs.samples.len(), 2);
        // A short (1.5 s) gap after that backfills exactly one extra second.
        rs.sample(50.0, t0 + Duration::from_millis(11_500));
        assert_eq!(rs.samples.len(), 3);
        // A sub-second gap adds nothing.
        rs.sample(50.0, t0 + Duration::from_millis(11_800));
        assert_eq!(rs.samples.len(), 3);
    }

    #[test]
    fn prompt_throughput_derives_from_prompt_tokens_and_mean_ttft() {
        let s = MetricsSnapshot {
            prompt_tokens: 4096,
            streams: vec![
                StreamMetric {
                    id: 1,
                    ttft_s: Some(0.2),
                    ..Default::default()
                },
                StreamMetric {
                    id: 2,
                    ttft_s: Some(0.4),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // Mean TTFT = 0.3 s → 4096 / 0.3.
        let s = s.with_derived_prompt_throughput();
        assert!((s.prompt_throughput - 4096.0 / 0.3).abs() < 1e-9);
    }

    #[test]
    fn prompt_throughput_falls_back_to_stream_pp_tokens() {
        // Engine B shape: no top-level prompt total, per-stream counts.
        let s = MetricsSnapshot {
            streams: vec![
                StreamMetric {
                    id: 1,
                    pp_tokens: Some(64),
                    ttft_s: Some(0.1),
                    ..Default::default()
                },
                StreamMetric {
                    id: 2,
                    pp_tokens: Some(64),
                    ttft_s: Some(0.3),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        // (64 + 64) / mean(0.2) = 640 t/s.
        let s = s.with_derived_prompt_throughput();
        assert!((s.prompt_throughput - 640.0).abs() < 1e-9);
    }

    #[test]
    fn prompt_throughput_stays_na_without_prompt_data() {
        let s = MetricsSnapshot::default().with_derived_prompt_throughput();
        assert_eq!(s.prompt_throughput, 0.0);
    }

    #[test]
    fn update_stamps_the_rolling_series_and_derives_prompt_throughput() {
        let state = MetricsState::new();
        state.update(MetricsSnapshot {
            prompt_tokens: 100,
            streams: vec![StreamMetric {
                id: 1,
                ttft_s: Some(0.5),
                ..Default::default()
            }],
            ..Default::default()
        });
        let loaded = state.load();
        assert!((loaded.prompt_throughput - 200.0).abs() < 1e-9);
        assert_eq!(loaded.throughput_series, vec![0.0]);
    }
}
