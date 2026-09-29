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

use std::sync::Arc;

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
#[derive(Debug, Clone)]
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

    /// Sample values mirroring the blueprint §6 mockup, used to seed the
    /// initial snapshot so the dashboard is verifiable before a real stream
    /// worker publishes data.
    pub fn sample() -> Self {
        let itl_bins = [
            0.92, 0.86, 0.79, 0.71, 0.62, 0.53, 0.44, 0.36, 0.29, 0.23, 0.18, 0.14, 0.11, 0.08,
            0.06, 0.045, 0.033, 0.024, 0.017, 0.012, 0.008, 0.005, 0.003, 0.002,
        ];
        Self {
            endpoint: "http://127.0.0.1:8000/v1".into(),
            backend: "vLLM".into(),
            model: "Qwen3.6-35B-A3B-UD-Q4_K_XL".into(),
            mode: "Concurrency".into(),
            aggregate_tps: 842.3,
            active_streams: 16,
            total_streams: 16,
            vram_used_gb: 21.4,
            vram_total_gb: 24.0,
            power_w: 285.0,
            joules_per_token: 0.338,
            gpu_clock_mhz: 1410.0,
            // 12.1 ms / 16.4 ms / 41.2 ms / 55.0 ms.
            itl_p50_ns: 12_100_000,
            itl_p90_ns: 16_400_000,
            itl_p99_ns: 41_200_000,
            itl_p999_ns: 55_000_000,
            itl_bins: itl_bins.to_vec(),
            prompt_tokens: 4096,
            completion_tokens: 1332,
            reasoning_tokens: 1152,
            status: StreamStatus::Streaming,
            streams: vec![
                StreamMetric {
                    id: 1,
                    kind: "Reasoning".into(),
                    state: StreamStatus::Streaming,
                    pp_tokens: Some(2048),
                    tg_tokens: Some(312),
                    ttft_s: Some(0.182),
                    gen_tps: Some(72.4),
                    mtp: Some(1.84),
                    progress: 0.55,
                },
                StreamMetric {
                    id: 2,
                    kind: "Content".into(),
                    state: StreamStatus::Streaming,
                    pp_tokens: Some(512),
                    tg_tokens: Some(180),
                    ttft_s: Some(0.045),
                    gen_tps: Some(88.1),
                    mtp: Some(1.02),
                    progress: 0.70,
                },
                StreamMetric {
                    id: 3,
                    kind: "Tool-Call".into(),
                    state: StreamStatus::Waiting,
                    pp_tokens: Some(4096),
                    tg_tokens: None,
                    ttft_s: None,
                    gen_tps: None,
                    mtp: None,
                    progress: 0.0,
                },
                StreamMetric {
                    id: 4,
                    kind: "Reasoning".into(),
                    state: StreamStatus::Done,
                    pp_tokens: Some(2048),
                    tg_tokens: Some(840),
                    ttft_s: Some(0.191),
                    gen_tps: Some(68.9),
                    mtp: Some(1.79),
                    progress: 1.0,
                },
            ],
            throughput_series: (0..60)
                .map(|i| 842.3 + 18.0 * ((i as f64) * 0.31).sin())
                .collect(),
        }
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
}

impl MetricsState {
    /// Create an empty state holding a default (zeroed) snapshot.
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(MetricsSnapshot::default()),
        }
    }

    /// Atomically publish a new snapshot.
    ///
    /// Writer side — called by the stream worker / engine per batch. The
    /// previous pointee stays alive for any reader that already holds its
    /// `Arc`, so an in-flight render never observes a torn state.
    pub fn update(&self, snapshot: MetricsSnapshot) {
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
