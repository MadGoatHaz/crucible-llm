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

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use arc_swap::ArcSwap;
use serde::{Deserialize, Serialize};

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
    /// The decode-loop guard (v0.1.1) flagged this stream: its repeating
    /// output is excluded from throughput tallies.
    pub looping: bool,
}

/// The JSON export's `loop_guard` block (v0.1.1): how many streams the
/// decode-loop guard excluded, and how many of their tokens were kept out
/// of the throughput numbers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LoopGuardSummary {
    /// Streams flagged `looping` (excluded from throughput).
    pub detected_streams: usize,
    /// Tokens excluded with them.
    pub excluded_tokens: u64,
}

/// A `(max, avg, p5)` summary of a metric's observed distribution.
///
/// `p5` is the 5th percentile — the *worst* 5% of samples (for a
/// throughput, the slowest; for a latency, the slowest). All values are
/// `0.0` when no sample has been recorded yet (rendered `--` by the views).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct StatTriple {
    pub max: f64,
    pub avg: f64,
    pub p5: f64,
}

/// Cumulative "overall" statistics across **every** benchmark that has run
/// (FIX 1).
///
/// The Live view's **KEY METRICS** panel shows *these* (in contrast to the
/// hero chart, which shows the *live* rolling window): `max` / `avg` / `p5`
/// per metric, the total tokens generated, the active-stream average/peak,
/// and the total elapsed time. It is built by the writer-side
/// [`OverallAccumulator`] in [`MetricsState::update`] and copied into every
/// published [`MetricsSnapshot`] so the render loop reads it lock-free.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct OverallStats {
    /// Generation throughput (per-stream decode t/s).
    pub gen: StatTriple,
    /// Prompt throughput (prefill t/s).
    pub prompt: StatTriple,
    /// Time-to-first-token (seconds).
    pub ttft: StatTriple,
    /// Inter-token latency p50 (ms).
    pub itl_p50: StatTriple,
    /// Inter-token latency p99 (ms).
    pub itl_p99: StatTriple,
    /// Total tokens generated — the sum of the server-reported
    /// `usage.completion_tokens` across every completed stream (NOT the SSE
    /// frame count).
    pub total_tokens: u64,
    /// Number of streams that reached a terminal state.
    pub completed_streams: u64,
    /// Mean active streams (time-weighted).
    pub active_avg: f64,
    /// Peak active streams.
    pub active_max: usize,
    /// Total elapsed benchmark time (seconds).
    pub duration_sec: f64,
}

/// A marker at an engine transition, for the hero chart's vertical lines
/// (FIX 2): `at_sec` is the elapsed time since the first update when the
/// running engine changed, `label` is the engine (the snapshot's `mode`).
#[derive(Debug, Clone)]
pub struct EngineMarker {
    pub at_sec: f64,
    pub label: String,
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
    /// The inference backend name — **only when known** (we never assume:
    /// the server may be vLLM, llama.cpp, LM Studio, Unsloth, SGLang, …).
    /// Empty by default; the TUI status bar shows the endpoint host
    /// (`host:port`) instead of a backend label.
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
    /// Labeled metric layers (v0.1.1, blueprint §v0.1.1-B) — the three
    /// throughput numbers a benchmark must keep *separate* (never blended):
    ///
    /// * **prefill** — `prompt_tokens / TTFT`: how fast the server ingests
    ///   the input;
    /// * **decode** — `completion_tokens / (T_last − T_first)`: sustained
    ///   generation speed;
    /// * **e2e** — `total_tokens / total wall time`: everything included
    ///   (connection, TTFT, generation).
    ///
    /// `0.0` is the N/A sentinel for each.
    pub prefill_throughput: f64,
    pub decode_throughput: f64,
    pub e2e_throughput: f64,
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

    // ---- nanosecond timing publication (v0.1.1) ----
    /// The measured overhead of a single [`crate::timing::MonotonicInstant::now`]
    /// call, in nanoseconds (10 000 calls averaged at state construction).
    /// This is the precision the timestamps are claimed at; the JSON export
    /// publishes it as `timing.overhead_ns`. `0` until
    /// [`MetricsState::update`] stamps it.
    pub timing_resolution_ns: u64,

    // ---- token counts ----
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub reasoning_tokens: u64,

    // ---- decode-loop guard (v0.1.1) ----
    /// Streams excluded from throughput because the loop guard flagged a
    /// repeating 32-token pattern × 3.
    pub loop_excluded_streams: usize,
    /// Tokens excluded with those streams.
    pub loop_excluded_tokens: u64,

    // ---- overall run status ----
    pub status: StreamStatus,

    // ---- per-stream detail (stream matrix) ----
    pub streams: Vec<StreamMetric>,

    // ---- rolling aggregate-throughput window (one sample/sec, last 60) ----
    pub throughput_series: Vec<f64>,

    // ---- cumulative overall stats (FIX 1: the KEY METRICS panel) ----
    pub overall: OverallStats,
    /// Engine-transition markers (FIX 2: the hero chart's vertical lines).
    pub engine_markers: Vec<EngineMarker>,
    /// Total elapsed benchmark time (seconds) since the first update.
    pub elapsed_sec: f64,
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
            prefill_throughput: 0.0,
            decode_throughput: 0.0,
            e2e_throughput: 0.0,
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
            timing_resolution_ns: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            reasoning_tokens: 0,
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            status: StreamStatus::default(),
            streams: Vec::new(),
            throughput_series: Vec::new(),
            overall: OverallStats::default(),
            engine_markers: Vec::new(),
            elapsed_sec: 0.0,
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

    /// Fill the v0.1.1 **labeled metric layers**
    /// ([`prefill_throughput`](Self::prefill_throughput),
    /// [`decode_throughput`](Self::decode_throughput),
    /// [`e2e_throughput`](Self::e2e_throughput)) from the snapshot's own
    /// data when an engine did not set them. The three numbers are kept
    /// separate — never blended into one:
    ///
    /// * **prefill** — the derived prompt throughput (input tokens / mean
    ///   TTFT);
    /// * **decode** — the token-weighted mean of the non-looping
    ///   streams' `gen_tps` (each is `tokens / (T_last − T_first)`);
    /// * **e2e** — the live aggregate (total tokens / total wall time).
    ///
    /// Looping streams (the v0.1.1 decode-loop guard) are excluded from
    /// the decode layer. `0.0` stays the N/A sentinel when no data exists.
    pub fn with_derived_labeled_throughputs(mut self) -> Self {
        if self.prefill_throughput <= 0.0 {
            self.prefill_throughput = self.prompt_throughput;
        }
        if self.decode_throughput <= 0.0 {
            let mut tokens = 0u64;
            let mut weighted = 0.0;
            for s in &self.streams {
                if s.looping {
                    continue;
                }
                let t = s.tg_tokens.unwrap_or(0);
                tokens += t;
                if let Some(g) = s.gen_tps.filter(|g| *g > 0.0) {
                    weighted += g * t as f64;
                }
            }
            if tokens > 0 && weighted > 0.0 {
                self.decode_throughput = weighted / tokens as f64;
            }
        }
        if self.e2e_throughput <= 0.0 {
            self.e2e_throughput = self.aggregate_tps;
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

/// The `p`-th percentile (0–100) of `samples` via linear interpolation.
///
/// `p = 5` yields the 5th percentile — the value at the bottom of the
/// distribution (the *worst* 5% for a throughput / the slowest 5% for a
/// latency). `0.0` when `samples` is empty.
fn percentile(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = (p.clamp(0.0, 100.0) / 100.0) * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil().min((sorted.len() - 1) as f64) as usize;
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// A [`StatTriple`] `(max, avg, p5)` over a sample distribution
/// (`0.0`/default when empty).
fn triple(samples: &[f64]) -> StatTriple {
    if samples.is_empty() {
        return StatTriple::default();
    }
    StatTriple {
        max: samples.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        avg: samples.iter().sum::<f64>() / samples.len() as f64,
        p5: percentile(samples, 5.0),
    }
}

/// Writer-side accumulator for the cumulative [`OverallStats`] (FIX 1).
///
/// Lives in [`MetricsState`] *outside* the double buffer (like the
/// [`RollingSeries`]) so it survives the engines' fresh
/// `..Default::default()` snapshots. `update()` folds each published
/// snapshot in and re-stamps the computed [`OverallStats`] into the
/// snapshot the render loop reads lock-free (measurement-isolation
/// invariant, blueprint §4).
///
/// Two sampling regimes:
/// * **per-completed-stream** (event-based): generation throughput, TTFT,
///   and the total-token count are recorded once per stream, on its
///   non-terminal → terminal transition (a stream is counted exactly once,
///   and a stream id re-used by a later level/iteration re-counts cleanly);
/// * **time-weighted** (~1 sample/sec): prompt throughput, the ITL
///   percentiles, and the active-stream count (mean + peak).
#[derive(Debug, Default)]
struct OverallAccumulator {
    gen: Vec<f64>,
    prompt: Vec<f64>,
    ttft: Vec<f64>,
    itl_p50: Vec<f64>,
    itl_p99: Vec<f64>,
    total_tokens: u64,
    completed_streams: u64,
    active_max: usize,
    active_sum: f64,
    active_count: u64,
    first: Option<Instant>,
    last: Option<Instant>,
    last_agg: Option<Instant>,
    prev_mode: String,
    markers: Vec<EngineMarker>,
    /// Previous status per stream id (completion-transition detection).
    prev_status: HashMap<u32, StreamStatus>,
}

impl OverallAccumulator {
    /// Defensive cap on any per-sample vector (a benchmark produces at most
    /// a few hundred completed streams / a few thousand time samples).
    const CAP: usize = 100_000;

    fn elapsed(&self) -> f64 {
        match (self.first, self.last) {
            (Some(f), Some(l)) => l.duration_since(f).as_secs_f64(),
            _ => 0.0,
        }
    }

    fn push(v: &mut Vec<f64>, x: f64) {
        if v.len() < Self::CAP {
            v.push(x);
        }
    }

    /// Fold one published snapshot into the running totals.
    fn fold(&mut self, s: &MetricsSnapshot, now: Instant) {
        if self.first.is_none() {
            self.first = Some(now);
        }
        self.last = Some(now);

        // Engine-transition marker: the snapshot's `mode` identifies the
        // engine ("short"/"long" = A, "Concurrency" = B, "NIAH" = C1,
        // "Reasoning" = C2, "Structured" = C3). Record where each new
        // engine begins so the hero chart can draw a vertical line.
        if s.mode != self.prev_mode {
            self.markers.push(EngineMarker {
                at_sec: self.elapsed(),
                label: s.mode.clone(),
            });
            self.prev_mode = s.mode.clone();
        }

        // Per-completed-stream metrics (event-based): count a stream once,
        // on its non-terminal → terminal transition.
        for st in &s.streams {
            let terminal = matches!(st.state, StreamStatus::Done | StreamStatus::Error);
            let was_terminal = self
                .prev_status
                .get(&st.id)
                .is_some_and(|p| matches!(p, StreamStatus::Done | StreamStatus::Error));
            if terminal && !was_terminal {
                self.completed_streams += 1;
                // Looping streams (v0.1.1 decode-loop guard) are excluded
                // from the throughput tallies — their repeating output is
                // not a measurement of the server's real generation speed.
                // The TTFT sample is kept (a latency, not a throughput).
                if !st.looping {
                    if let Some(g) = st.gen_tps.filter(|g| *g > 0.0) {
                        Self::push(&mut self.gen, g);
                    }
                    if let Some(tok) = st.tg_tokens {
                        self.total_tokens += tok;
                    }
                }
                if let Some(t) = st.ttft_s.filter(|t| *t > 0.0) {
                    Self::push(&mut self.ttft, t);
                }
            }
            self.prev_status.insert(st.id, st.state);
        }

        // Time-weighted aggregate metrics (~1 sample/sec, gap-aware): prompt
        // throughput, the ITL percentiles, and the active-stream count.
        let due = self
            .last_agg
            .is_none_or(|t| now.duration_since(t) >= SAMPLE_PERIOD);
        if due {
            self.last_agg = Some(now);
            if s.prompt_throughput > 0.0 {
                Self::push(&mut self.prompt, s.prompt_throughput);
            }
            if s.itl_p50_ns > 0 {
                Self::push(&mut self.itl_p50, s.itl_p50_ns as f64 / 1e6);
            }
            if s.itl_p99_ns > 0 {
                Self::push(&mut self.itl_p99, s.itl_p99_ns as f64 / 1e6);
            }
            self.active_sum += s.active_streams as f64;
            self.active_count += 1;
            self.active_max = self.active_max.max(s.active_streams);
        }
    }

    /// The computed [`OverallStats`] for the current snapshot.
    fn stats(&self) -> OverallStats {
        OverallStats {
            gen: triple(&self.gen),
            prompt: triple(&self.prompt),
            ttft: triple(&self.ttft),
            itl_p50: triple(&self.itl_p50),
            itl_p99: triple(&self.itl_p99),
            total_tokens: self.total_tokens,
            completed_streams: self.completed_streams,
            active_avg: if self.active_count > 0 {
                self.active_sum / self.active_count as f64
            } else {
                0.0
            },
            active_max: self.active_max,
            duration_sec: self.elapsed(),
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
    /// Writer-side rolling throughput window (see [`RollingSeries`]).
    /// Only `update()` (the engine / hw-poller side) touches it; the render
    /// loop reads the copied series from the published snapshot, so the
    /// measurement-isolation invariant (blueprint §4) is intact.
    rolling: Mutex<RollingSeries>,
    /// Writer-side cumulative overall-stats accumulator (FIX 1, see
    /// [`OverallAccumulator`]): folded on each `update()` and re-stamped
    /// into the published snapshot for the lock-free render.
    overall_acc: Mutex<OverallAccumulator>,
    /// Latch set when the benchmark sequence completes. While set,
    /// [`update`](Self::update) is a no-op: the rolling window, the
    /// overall accumulator, and the published snapshot all freeze at
    /// their final values, so the UI never shows numbers shifting after
    /// the run is done. This is what stops the forever-running 100 ms
    /// hardware poller (and any other writer) from re-stamping `0.0`
    /// throughput samples / growing `elapsed_sec` / dragging `active_avg`
    /// toward zero after the last engine finishes. One-way *within a run*:
    /// [`unfreeze`](Self::unfreeze) re-arms it when a new run starts.
    frozen: AtomicBool,
    /// The measured overhead of a single
    /// [`crate::timing::MonotonicInstant::now`] call (ns) — timed once at
    /// construction (10 000 calls averaged) and stamped into every
    /// published snapshot as `timing_resolution_ns` (v0.1.1 nanosecond
    /// timing publication).
    timing_overhead_ns: u64,
}

impl MetricsState {
    /// Create an empty state holding a default (zeroed) snapshot.
    pub fn new() -> Self {
        Self {
            inner: ArcSwap::from_pointee(MetricsSnapshot::default()),
            rolling: Mutex::new(RollingSeries::default()),
            overall_acc: Mutex::new(OverallAccumulator::default()),
            frozen: AtomicBool::new(false),
            timing_overhead_ns: crate::timing::measure_timestamp_overhead(),
        }
    }

    /// Freeze all metric updates. Called by the sequence executor when
    /// the last engine completes (and again at `AllComplete`, idempotently).
    /// After this, [`update`](Self::update) is a no-op — the rolling
    /// window, the overall-stats accumulator, and the published snapshot
    /// all hold their final values, so the UI shows static numbers, the
    /// throughput graph stops animating, and the 100 ms hardware poller
    /// (which checks [`is_frozen`](Self::is_frozen)) goes idle.
    pub fn freeze(&self) {
        self.frozen.store(true, Ordering::Relaxed);
    }

    /// Re-arm the pipeline for a **new** run. The freeze is one-way
    /// within a run (so the final numbers can never drift), but a fresh
    /// benchmark sequence must be able to update metrics again — the App
    /// calls this when the user starts the next run (`r` / `F5` /
    /// launch). Also wakes the idle hardware poller.
    pub fn unfreeze(&self) {
        self.frozen.store(false, Ordering::Relaxed);
    }

    /// `true` while the pipeline is frozen (a run completed and no new
    /// run has started).
    pub fn is_frozen(&self) -> bool {
        self.frozen.load(Ordering::Relaxed)
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
        // One-way freeze (sequence `AllComplete`): no further updates. The
        // rolling window, the overall accumulator, and the published
        // snapshot all stay at their final values — the UI shows static
        // numbers and the graph stops animating. This short-circuits the
        // forever-running hardware poller, which would otherwise keep
        // re-stamping `0.0` samples and growing `elapsed_sec`.
        if self.frozen.load(Ordering::Relaxed) {
            return;
        }
        // v0.1.1: stamp the measured timestamp overhead when the engine
        // left the field empty, then complete the labeled metric layers.
        if snapshot.timing_resolution_ns == 0 {
            snapshot.timing_resolution_ns = self.timing_overhead_ns;
        }
        snapshot = snapshot
            .with_derived_prompt_throughput()
            .with_derived_labeled_throughputs();
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
        {
            let mut acc = self
                .overall_acc
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            acc.fold(&snapshot, Instant::now());
            snapshot.overall = acc.stats();
            snapshot.engine_markers = acc.markers.clone();
            snapshot.elapsed_sec = acc.elapsed();
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

    // ── overall stats (FIX 1) ────────────────────────────────────────────

    fn done_stream(
        id: u32,
        gen: Option<f64>,
        ttft: Option<f64>,
        tokens: Option<u64>,
    ) -> StreamMetric {
        StreamMetric {
            id,
            state: StreamStatus::Done,
            gen_tps: gen,
            ttft_s: ttft,
            tg_tokens: tokens,
            ..Default::default()
        }
    }

    fn streaming(id: u32) -> StreamMetric {
        StreamMetric {
            id,
            state: StreamStatus::Streaming,
            ..Default::default()
        }
    }

    #[test]
    fn overall_counts_each_completed_stream_once() {
        let state = MetricsState::new();
        // Streaming (not counted) → Done (counted) → Done again (not re-counted).
        state.update(MetricsSnapshot {
            streams: vec![streaming(0)],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            streams: vec![done_stream(0, Some(100.0), Some(0.5), Some(256))],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            streams: vec![done_stream(0, Some(100.0), Some(0.5), Some(256))],
            ..Default::default()
        });
        let o = state.load().overall.clone();
        assert_eq!(o.completed_streams, 1, "exactly one completion counted");
        assert_eq!(o.total_tokens, 256);
        assert!((o.gen.avg - 100.0).abs() < 1e-9);
        assert!((o.ttft.avg - 0.5).abs() < 1e-9);
    }

    #[test]
    fn overall_recounts_when_a_stream_id_is_reused() {
        // A new iteration reuses stream id 0: Streaming → Done counts again.
        let state = MetricsState::new();
        state.update(MetricsSnapshot {
            streams: vec![done_stream(0, Some(100.0), None, Some(100))],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            streams: vec![streaming(0)],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            streams: vec![done_stream(0, Some(120.0), None, Some(200))],
            ..Default::default()
        });
        let o = state.load().overall.clone();
        assert_eq!(o.completed_streams, 2, "each iteration's completion counts");
        assert_eq!(o.total_tokens, 300);
    }

    #[test]
    fn overall_computes_max_avg_p5_triples() {
        let state = MetricsState::new();
        // Three completed streams with distinct gen rates: 100, 200, 300.
        for g in [100.0, 200.0, 300.0] {
            state.update(MetricsSnapshot {
                streams: vec![done_stream(0, Some(g), Some(0.1), Some(50))],
                ..Default::default()
            });
            // A fresh iteration re-arms the (re-used) stream id.
            state.update(MetricsSnapshot {
                streams: vec![streaming(0)],
                ..Default::default()
            });
        }
        let o = state.load().overall.clone();
        assert_eq!(o.completed_streams, 3);
        assert!((o.gen.max - 300.0).abs() < 1e-9, "max");
        assert!((o.gen.avg - 200.0).abs() < 1e-9, "avg");
        // p5 of [100, 200, 300]: idx = 0.05 × 2 = 0.1 → 100·0.9 + 200·0.1 = 110.
        assert!((o.gen.p5 - 110.0).abs() < 1e-9, "p5: {}", o.gen.p5);
    }

    #[test]
    fn overall_tracks_duration_and_engine_markers() {
        let state = MetricsState::new();
        state.update(MetricsSnapshot {
            mode: "short".into(),
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            mode: "Concurrency".into(),
            ..Default::default()
        });
        let loaded = state.load();
        // Two distinct engines → two transition markers, in order.
        assert_eq!(loaded.engine_markers.len(), 2, "a marker per engine");
        assert_eq!(loaded.engine_markers[0].label, "short");
        assert_eq!(loaded.engine_markers[1].label, "Concurrency");
        assert!(loaded.elapsed_sec >= 0.0);
    }

    // ── one-way freeze (AllComplete) ────────────────────────────────────

    #[test]
    fn frozen_state_ignores_all_further_updates() {
        let state = MetricsState::new();
        state.update(snap(100.0));
        assert!(!state.is_frozen(), "fresh state is not frozen");

        // The sequence completes: everything freezes in place.
        state.freeze();
        assert!(state.is_frozen());
        let before = state.load();
        let series_before = before.throughput_series.clone();
        let elapsed_before = before.elapsed_sec;
        let overall_before = before.overall.clone();
        let agg_before = before.aggregate_tps;

        // A later update (e.g. from the forever-running 100 ms hardware
        // poller, now with `0.0` throughput and no active streams) must be
        // a no-op: no new rolling sample, no elapsed growth, no overall
        // drift, and the published snapshot is untouched.
        std::thread::sleep(Duration::from_millis(1100));
        state.update(snap(0.0));
        let after = state.load();
        assert_eq!(after.throughput_series, series_before, "series frozen");
        assert_eq!(after.elapsed_sec, elapsed_before, "elapsed frozen");
        assert_eq!(after.overall, overall_before, "overall frozen");
        assert_eq!(after.aggregate_tps, agg_before, "snapshot frozen");

        // The freeze is idempotent (setting it twice is a no-op).
        state.freeze();
        assert!(state.is_frozen());
    }

    // ── v0.1.1: labeled layers, timing publication, loop guard ─────────

    #[test]
    fn update_stamps_the_measured_timing_resolution() {
        let state = MetricsState::new();
        state.update(snap(100.0));
        // The measured overhead is stamped onto every published snapshot
        // (a real clock measures a small positive number of ns).
        assert!(
            state.load().timing_resolution_ns > 0,
            "timing_resolution_ns must be the measured overhead"
        );
    }

    #[test]
    fn labeled_layers_derive_from_stream_data() {
        let s = MetricsSnapshot {
            prompt_tokens: 4096,
            streams: vec![
                StreamMetric {
                    id: 1,
                    ttft_s: Some(0.2),
                    tg_tokens: Some(100),
                    gen_tps: Some(50.0),
                    ..Default::default()
                },
                StreamMetric {
                    id: 2,
                    ttft_s: Some(0.4),
                    tg_tokens: Some(300),
                    gen_tps: Some(100.0),
                    ..Default::default()
                },
            ],
            aggregate_tps: 42.0,
            ..Default::default()
        };
        let s = s
            .with_derived_prompt_throughput()
            .with_derived_labeled_throughputs();
        // prefill = the derived prompt throughput (4096 / mean TTFT 0.3).
        assert!((s.prefill_throughput - 4096.0 / 0.3).abs() < 1e-9);
        // decode = token-weighted mean gen: (50·100 + 100·300) / 400.
        assert!((s.decode_throughput - 87.5).abs() < 1e-9);
        // e2e = the live aggregate.
        assert!((s.e2e_throughput - 42.0).abs() < 1e-9);
    }

    #[test]
    fn labeled_layers_exclude_looping_streams_from_decode() {
        let s = MetricsSnapshot {
            streams: vec![
                StreamMetric {
                    id: 1,
                    tg_tokens: Some(100),
                    gen_tps: Some(50.0),
                    looping: true,
                    ..Default::default()
                },
                StreamMetric {
                    id: 2,
                    tg_tokens: Some(300),
                    gen_tps: Some(100.0),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let s = s.with_derived_labeled_throughputs();
        // The looping stream's 100 tokens at 50 t/s are out: decode is
        // exactly stream 2's rate.
        assert!((s.decode_throughput - 100.0).abs() < 1e-9);
    }

    #[test]
    fn overall_excludes_looping_streams_from_tokens_and_gen() {
        let state = MetricsState::new();
        // Stream 0 loops (excluded), stream 1 is clean (counted).
        state.update(MetricsSnapshot {
            streams: vec![
                StreamMetric {
                    id: 0,
                    state: StreamStatus::Done,
                    gen_tps: Some(999.0),
                    tg_tokens: Some(1000),
                    looping: true,
                    ..Default::default()
                },
                StreamMetric {
                    id: 1,
                    state: StreamStatus::Done,
                    gen_tps: Some(50.0),
                    tg_tokens: Some(200),
                    ..Default::default()
                },
            ],
            ..Default::default()
        });
        let o = state.load().overall.clone();
        assert_eq!(o.completed_streams, 2, "both streams completed");
        assert_eq!(o.total_tokens, 200, "looping stream's tokens excluded");
        assert!((o.gen.avg - 50.0).abs() < 1e-9, "looping rate excluded");
    }

    #[test]
    fn unfreeze_re_arms_the_pipeline_for_a_new_run() {
        let state = MetricsState::new();
        state.update(snap(100.0));
        state.freeze();
        assert!(state.is_frozen());

        // A new run starts: the latch clears and updates flow again.
        state.unfreeze();
        assert!(!state.is_frozen());
        std::thread::sleep(Duration::from_millis(1100));
        state.update(snap(120.0));
        let loaded = state.load();
        assert!(
            loaded.throughput_series.len() >= 2,
            "series fills again after unfreeze: {:?}",
            loaded.throughput_series
        );
        // … and a later completion freezes it once more (one-way within
        // the new run).
        state.freeze();
        assert!(state.is_frozen());
    }
}
