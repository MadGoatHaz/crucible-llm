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
use crate::hw::GpuSample;

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
    /// The latest vendor-agnostic GPU sample — the full telemetry set the
    /// Live view's GPU panel renders (power, utilization, temperature,
    /// clocks, VRAM, throttle reasons). `None` when no GPU is present:
    /// the panel is hidden entirely, never shown as an N/A box.
    pub gpu: Option<GpuSample>,

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
    /// The token frames **observed on the wire** (each `Reasoning` /
    /// `Content` delta), cumulative across the stream. This is the
    /// ground-truth numerator for the live decode rate: unlike
    /// [`completion_tokens`](Self::completion_tokens) it never jumps to the
    /// server's self-reported `usage` (which some backends — Unsloth /
    /// llama.cpp — inflate to the requested `max_tokens` target). `0` when
    /// the publishing engine does not report it (the rate tracker then falls
    /// back to `completion_tokens`).
    pub observed_frames: u64,
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
            gpu: None,
            itl_p50_ns: 0,
            itl_p90_ns: 0,
            itl_p99_ns: 0,
            itl_p999_ns: 0,
            itl_bins: Vec::new(),
            timing_resolution_ns: 0,
            prompt_tokens: 0,
            completion_tokens: 0,
            observed_frames: 0,
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

/// The rolling decode-rate window kept by [`MetricsState`] (writer side):
/// one sample per second, at most [`ROLLING_WINDOW`] (60 = the last 60
/// seconds), copied into every published snapshot's
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

/// The minimum time (seconds) that must elapse after the first token before
/// the tracker emits a rate sample.
///
/// Without this gate the very first sample divides the token count by a
/// near-zero elapsed time (`first_token_instant` is latched to the same
/// instant it is first read), producing a spike of `tokens / ~0` (e.g.
/// 8 000 t/s for 8 tokens). That single spike becomes the first rolling
/// sample, the hero chart's y-axis auto-scales to it, and every real
/// (~60 t/s) bar collapses to a sub-pixel sliver — the "graph shows the
/// wrong number" symptom. Requiring a real elapsed time keeps the sample
/// bounded (and the rate still converges to the true number as more tokens
/// arrive).
const MIN_ELAPSED_SECS: f64 = 0.05;

/// The minimum time (seconds) of *real decoding* before the **first**
/// cumulative sample is emitted.
///
/// The 50 ms floor alone is not enough: the first publish carrying tokens
/// arrives a batch (8/16 frames) after the true first token, so a sample
/// taken 50 ms after that latch divides a *small* token count by a *short*
/// duration and spikes (10 tokens / 0.05 s = 200 t/s for a ~55 t/s stream).
/// That spike auto-scales the hero chart's y-axis to ~220 and collapses
/// every real (~55 t/s) bar to a sliver — the graph then reads *lower*
/// than the OVERALL METRICS panel next to it. Requiring half a second of
/// decoding keeps the first sample close to the true rate; the cumulative
/// curve then converges, and the stream's *exact* final rate
/// (`gen_tps` = `total_tokens / generation_duration`, the same value the
/// Overall Metrics panel records) lands as the last sample.
const FIRST_SAMPLE_ELAPSED: f64 = 0.5;

/// One decoded-rate sample for the rolling series.
#[derive(Debug, Clone, Copy, PartialEq)]
enum RateSample {
    /// A live cumulative sample (`tokens_received / elapsed`) — appended
    /// through the window's normal 1 s cadence.
    Live(f64),
    /// The stream's **exact** final rate (its `gen_tps`): `total_tokens /
    /// generation_duration` — the same formula and value the OVERALL
    /// METRICS panel records for the completed stream. Forced into the
    /// series even when it lands inside the 1 s cadence of the last live
    /// sample (a short stream), so the graph's last bar is the true
    /// number, never a decaying estimate.
    Final(f64),
}

/// Writer-side tracker for the **cumulative decode rate** that the rolling
/// series samples.
///
/// * While a stream is open the tracker emits the cumulative rate
///   `tokens_received / (now − first_token_instant)` (gated by
///   [`FIRST_SAMPLE_ELAPSED`] for the first sample so it cannot spike).
///   `tokens_received` is the snapshot's **observed token frames** (the
///   wire truth) — it never jumps to the server's self-reported `usage`
///   (which some backends inflate to the requested `max_tokens` target).
///   Writers that don't report `observed_frames` fall back to
///   `completion_tokens`. `first_token_instant` is latched the first time
///   the token count is non-zero.
/// * When a stream row completes (non-terminal → terminal — the *same*
///   transition the [`OverallAccumulator`] folds on), the tracker emits
///   [`RateSample::Final`] with that stream's **exact** `gen_tps`
///   (`total_tokens / generation_duration`) — the same value the OVERALL
///   METRICS panel shows. The graph's last bar is therefore the same
///   number as the panel's.
/// * After a `Final`, the tracker is *finished*: it emits no further
///   samples. Post-completion updates (the engine's duplicate final
///   publish, the 100 ms hardware poller re-publishing the merged
///   snapshot, the summary hold) would otherwise divide the fixed token
///   count by a *growing* elapsed and drag the series below the true rate
///   — the "graph shows a lower number than Overall Metrics" symptom.
///   A new stream in the same mode (a re-used stream id, the next
///   iteration, the next Engine B level) re-arms the tracker.
///
/// Reset on engine transition (the snapshot's `mode` changes) so each
/// engine gets a fresh convergence curve.
#[derive(Debug, Default)]
struct DecodeRateTracker {
    /// Total tokens received so far (latest snapshot's `observed_frames`,
    /// else `completion_tokens`).
    tokens_received: u64,
    /// When the first token was observed (the rate denominator's origin).
    first_token_instant: Option<Instant>,
    /// The engine mode the tracker is currently bound to.
    current_mode: String,
    /// The previous per-stream state (completion-transition detection —
    /// the same signal the [`OverallAccumulator`] folds on).
    prev_state: HashMap<u32, StreamStatus>,
    /// No cumulative sample has been emitted yet for the current stream
    /// (the first one needs [`FIRST_SAMPLE_ELAPSED`], not just
    /// [`MIN_ELAPSED_SECS`]).
    first_sample: bool,
    /// The current stream completed and its exact `gen_tps` was pushed as
    /// the final sample: no further samples until a new stream starts.
    finished: bool,
}

impl DecodeRateTracker {
    /// Fold one published snapshot. Returns the rate sample to record —
    /// [`RateSample::Final`] on a stream completion (the exact `gen_tps`),
    /// [`RateSample::Live`] once a meaningful elapsed time has accumulated,
    /// or `None` when no sample should be recorded yet (no tokens, below
    /// the first-sample gate, or the stream is already finished).
    fn update(&mut self, snapshot: &MetricsSnapshot, now: Instant) -> Option<RateSample> {
        // Engine transition: reset for the new engine's convergence curve.
        if snapshot.mode != self.current_mode {
            self.current_mode = snapshot.mode.clone();
            self.tokens_received = 0;
            self.first_token_instant = None;
            self.first_sample = true;
            self.finished = false;
            self.prev_state.clear();
        }
        // The numerator: the observed token frames on the wire (the ground
        // truth — it stays monotonic and never jumps to an over-reported
        // `usage`), falling back to `completion_tokens` for writers that do
        // not report `observed_frames`.
        let observed = snapshot.observed_frames;
        self.tokens_received = if observed > 0 {
            observed
        } else {
            snapshot.completion_tokens
        };

        // Completion: any stream row transitions non-terminal → terminal
        // (the same detection the `OverallAccumulator` uses to fold the
        // stream's `gen_tps` into the overall stats). The final rolling
        // sample is that exact `gen_tps` — `total_tokens /
        // generation_duration` — so the graph's last bar carries the same
        // number the OVERALL METRICS panel reports.
        let mut any_transition = false;
        let mut final_rate: Option<f64> = None;
        for st in &snapshot.streams {
            let terminal = matches!(st.state, StreamStatus::Done | StreamStatus::Error);
            let was_terminal = self
                .prev_state
                .get(&st.id)
                .is_some_and(|p| matches!(p, StreamStatus::Done | StreamStatus::Error));
            if terminal && !was_terminal {
                any_transition = true;
                // Looping streams are excluded — exactly as the overall
                // accumulator excludes them from its `gen` samples.
                if !st.looping {
                    if let Some(g) = st.gen_tps.filter(|g| *g > 0.0) {
                        final_rate = Some(final_rate.map_or(g, |c| c.max(g)));
                    }
                }
            }
            self.prev_state.insert(st.id, st.state);
        }
        if any_transition {
            self.finished = true;
            return final_rate.map(RateSample::Final);
        }

        // A finished stream emits no further samples (no post-completion
        // decay). A *new* stream in the same mode — a re-used stream id
        // (next iteration / next Engine B level) reporting a non-terminal
        // state with tokens — re-arms the tracker. Snapshots with no stream
        // rows (the hardware poller's merge, the app seed) do not.
        if self.finished {
            let new_stream = snapshot
                .streams
                .iter()
                .any(|st| matches!(st.state, StreamStatus::Waiting | StreamStatus::Streaming))
                && self.tokens_received > 0;
            if !new_stream {
                return None;
            }
            self.finished = false;
            self.first_sample = true;
            self.first_token_instant = None;
        }

        // Latch the first-token instant.
        if self.tokens_received > 0 && self.first_token_instant.is_none() {
            self.first_token_instant = Some(now);
        }
        match self.first_token_instant {
            Some(first) => {
                let elapsed_s = now.duration_since(first).as_secs_f64();
                // The first sample needs real decoding time
                // ([`FIRST_SAMPLE_ELAPSED`]): an earlier one divides a
                // small token count by a short duration and spikes the
                // chart's y-axis. After the first sample, the window's 1 s
                // cadence gates the rest (and [`MIN_ELAPSED_SECS`] keeps a
                // same-instant re-publish from re-sampling).
                let gate = if self.first_sample {
                    FIRST_SAMPLE_ELAPSED
                } else {
                    MIN_ELAPSED_SECS
                };
                if elapsed_s < gate {
                    return None;
                }
                self.first_sample = false;
                Some(RateSample::Live(self.tokens_received as f64 / elapsed_s))
            }
            None => None,
        }
    }

    /// Reset for a new run (called from [`MetricsState::unfreeze`]).
    fn reset(&mut self) {
        self.tokens_received = 0;
        self.first_token_instant = None;
        self.current_mode.clear();
        self.prev_state.clear();
        self.first_sample = true;
        self.finished = false;
    }
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

    /// Append `value` **unconditionally** (bypassing the 1 s cadence) and
    /// advance the cadence anchor.
    ///
    /// Used for a stream's exact final rate ([`RateSample::Final`]): the
    /// last bar of the graph must be the true number even when the
    /// stream completes less than a second after the previous live sample
    /// (a short stream), where the normal cadence would drop it.
    fn force_sample(&mut self, value: f64, now: Instant) {
        self.samples.push_back(value);
        self.last = Some(now);
        while self.samples.len() > ROLLING_WINDOW {
            self.samples.pop_front();
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
    /// Writer-side rolling decode-rate window (see [`RollingSeries`]).
    /// Only `update()` (the engine / hw-poller side) touches it; the render
    /// loop reads the copied series from the published snapshot, so the
    /// measurement-isolation invariant (blueprint §4) is intact.
    rolling: Mutex<RollingSeries>,
    /// Writer-side cumulative decode-rate tracker (see [`DecodeRateTracker`]):
    /// computes `tokens_received / (now − first_token)` — the same formula
    /// the Overall Metrics panel uses — and feeds it to the rolling series.
    rate_tracker: Mutex<DecodeRateTracker>,
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
            rate_tracker: Mutex::new(DecodeRateTracker::default()),
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
    /// launch). Also resets the rate tracker and clears the rolling
    /// series so the new run starts with a clean convergence curve.
    pub fn unfreeze(&self) {
        self.frozen.store(false, Ordering::Relaxed);
        self.rate_tracker
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .reset();
        let mut rs = self.rolling.lock().unwrap_or_else(|p| p.into_inner());
        rs.samples.clear();
        rs.last = None;
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
            // The cumulative decode rate: `tokens_received / (now −
            // first_token)` — computed in real time while the stream is
            // open. On stream completion the tracker instead emits the
            // stream's *exact* `gen_tps` (`total_tokens /
            // generation_duration`) — the same value the Overall Metrics
            // panel records — as the series' final sample, so the graph
            // ends on the true number. The tracker returns `None` until a
            // meaningful elapsed time has accumulated (no
            // divide-by-near-zero spike), and emits nothing after a
            // completed stream (no post-completion decay).
            let now = Instant::now();
            let rate_sample = {
                let mut tracker = self
                    .rate_tracker
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                tracker.update(&snapshot, now)
            };
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
                rs.last = Some(now);
            } else if let Some(sample) = rate_sample {
                match sample {
                    RateSample::Live(rate) => {
                        rs.sample(rate, now);
                    }
                    // The exact final rate bypasses the 1 s cadence: the
                    // graph's last bar is the stream's true number even
                    // for a short stream.
                    RateSample::Final(rate) => rs.force_sample(rate, now),
                }
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

    /// A snapshot with both `aggregate_tps` and a token count set —
    /// the rolling series samples the cumulative decode rate, so tests that
    /// exercise the series need a non-zero token count (`observed_frames`,
    /// the wire-truth numerator the tracker reads).
    fn snap_tokens(agg: f64, tokens: u64) -> MetricsSnapshot {
        MetricsSnapshot {
            aggregate_tps: agg,
            completion_tokens: tokens,
            observed_frames: tokens,
            ..Default::default()
        }
    }

    #[test]
    fn rolling_series_samples_once_per_second() {
        let state = MetricsState::new();
        // The first update latches the first-token origin. With ~0 elapsed
        // time the tracker emits no sample (a divide-by-near-zero would
        // spike to `tokens / 0.001` and dominate the chart's y-axis).
        state.update(snap_tokens(100.0, 50));
        assert_eq!(
            state.load().throughput_series.len(),
            0,
            "no sample on the zero-elapsed first update"
        );

        // A second update in the same instant still has no real elapsed.
        state.update(snap_tokens(110.0, 60));
        assert_eq!(state.load().throughput_series.len(), 0);

        // After a real elapsed (1.1 s) the first sample lands — a bounded
        // cumulative rate (120 tokens / ~1.1 s), never a spike.
        std::thread::sleep(Duration::from_millis(1100));
        state.update(snap_tokens(120.0, 120));
        let series = state.load().throughput_series.clone();
        assert_eq!(
            series.len(),
            1,
            "one sample after a real elapsed: {series:?}"
        );
        assert!(series[0] > 0.0, "sample is positive: {}", series[0]);
        assert!(
            series[0] < 1000.0,
            "first sample is bounded (no divide-by-floor spike): {}",
            series[0]
        );
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
            completion_tokens: 0, // no tokens yet → cumulative rate is 0.0
            streams: vec![StreamMetric {
                id: 1,
                ttft_s: Some(0.5),
                ..Default::default()
            }],
            ..Default::default()
        });
        let loaded = state.load();
        assert!((loaded.prompt_throughput - 200.0).abs() < 1e-9);
        // No tokens → no decode-rate sample yet (the series stays empty;
        // the hero renders "Awaiting first tokens…").
        assert!(loaded.throughput_series.is_empty());
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
    fn overall_counts_every_engine_when_stream_ids_are_reused() {
        // The user-reported Live-vs-Overall discrepancy, reproduced:
        // Engine A (mode "short") completes *naturally* at 64.1 t/s;
        // Engine F (mode "FlatOut") re-uses stream id 0, streams live, and
        // its *final* publish (the 60 s window aborted the worker — no
        // terminal event on the wire) reports Done at 27.9 t/s. Both
        // streams must be counted, so the OVERALL METRICS panel covers
        // *all* engines of the run.
        let state = MetricsState::new();
        state.update(MetricsSnapshot {
            mode: "short".into(),
            streams: vec![done_stream(0, Some(64.1), Some(0.393), Some(256))],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            mode: "FlatOut".into(),
            streams: vec![streaming(0)],
            ..Default::default()
        });
        state.update(MetricsSnapshot {
            mode: "FlatOut".into(),
            streams: vec![done_stream(0, Some(27.9), Some(0.198), Some(1667))],
            ..Default::default()
        });
        let o = state.load().overall.clone();
        assert_eq!(o.completed_streams, 2, "both engines' streams counted");
        assert_eq!(
            o.total_tokens,
            256 + 1667,
            "total tokens across all engines"
        );
        assert!(
            (o.gen.avg - (64.1 + 27.9) / 2.0).abs() < 1e-9,
            "avg across all engines: {}",
            o.gen.avg
        );
        assert!((o.gen.max - 64.1).abs() < 1e-9, "max: {}", o.gen.max);
        assert!(
            (o.gen.p5 - (27.9 * 0.95 + 64.1 * 0.05)).abs() < 1e-9,
            "p5: {}",
            o.gen.p5
        );
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
        state.update(snap_tokens(100.0, 50));
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
        state.update(snap_tokens(0.0, 0));
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
        state.update(snap_tokens(100.0, 10));
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
        state.update(snap_tokens(100.0, 50));
        state.freeze();
        assert!(state.is_frozen());

        // A new run starts: the latch clears, the tracker resets, and the
        // rolling series is cleared.
        state.unfreeze();
        assert!(!state.is_frozen());
        // The first update after unfreeze re-latches the origin (no sample
        // yet — the elapsed since the latch is ~0).
        state.update(snap_tokens(120.0, 100));
        assert!(state.load().throughput_series.is_empty());
        // After a real elapsed the series fills again.
        std::thread::sleep(Duration::from_millis(1100));
        state.update(snap_tokens(130.0, 200));
        let loaded = state.load();
        assert!(
            !loaded.throughput_series.is_empty(),
            "series fills again after unfreeze: {:?}",
            loaded.throughput_series
        );
        // … and a later completion freezes it once more (one-way within
        // the new run).
        state.freeze();
        assert!(state.is_frozen());
    }

    // ── DecodeRateTracker convergence ──────────────────────────────────

    #[test]
    fn rate_tracker_converges_as_tokens_arrive() {
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();

        // No tokens yet: no sample.
        let s0 = MetricsSnapshot {
            completion_tokens: 0,
            observed_frames: 0,
            mode: "short".into(),
            ..Default::default()
        };
        assert!(tracker.update(&s0, t0).is_none());

        // First token: the origin latches, but with ~0 elapsed no sample is
        // emitted — this is exactly what prevents the divide-by-floor spike
        // (1 token / 0.001 s = 1000 t/s) from reaching the rolling series.
        let s1 = MetricsSnapshot {
            completion_tokens: 1,
            observed_frames: 1,
            mode: "short".into(),
            ..Default::default()
        };
        assert!(
            tracker
                .update(&s1, t0 + Duration::from_millis(10))
                .is_none(),
            "zero-elapsed first token emits no sample"
        );

        // 10 tokens, 90 ms since the latch: the first-sample gate (0.5 s of
        // real decoding) still holds. Sampling here would be 10 / 0.09 and
        // earlier still 10 / 0.05 = 200 t/s for a ~100 t/s stream — a spike
        // that would auto-scale the chart's y-axis and collapse every real
        // bar to a sliver.
        let s2 = MetricsSnapshot {
            completion_tokens: 10,
            observed_frames: 10,
            mode: "short".into(),
            ..Default::default()
        };
        assert!(
            tracker
                .update(&s2, t0 + Duration::from_millis(100))
                .is_none(),
            "no first sample before 0.5 s of real decoding"
        );

        // Half a second of real decoding: the first sample is bounded and
        // close to the true rate.
        let s3 = MetricsSnapshot {
            completion_tokens: 60,
            observed_frames: 60,
            mode: "short".into(),
            ..Default::default()
        };
        let r3 = tracker
            .update(&s3, t0 + Duration::from_millis(560))
            .expect("real elapsed yields a rate");
        match r3 {
            RateSample::Live(r) => {
                assert!(
                    (r - 100.0).abs() < 25.0,
                    "60 tok / ~0.5 s ≈ 100 t/s: got {r}"
                );
            }
            RateSample::Final(_) => panic!("no stream completed in this test"),
        }

        // Even more tokens: the rate converges to the true 100 t/s.
        let s4 = MetricsSnapshot {
            completion_tokens: 100,
            observed_frames: 100,
            mode: "short".into(),
            ..Default::default()
        };
        let r4 = tracker
            .update(&s4, t0 + Duration::from_millis(1000))
            .expect("real elapsed yields a rate");
        match r4 {
            RateSample::Live(r) => {
                assert!(
                    (r - 100.0).abs() < 15.0,
                    "100 tok / ~0.99 s ≈ 101 t/s: got {r}"
                );
            }
            RateSample::Final(_) => panic!("no stream completed in this test"),
        }
    }

    #[test]
    fn rate_tracker_resets_on_engine_transition() {
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();

        // Engine A: 50 tokens.
        let sa = MetricsSnapshot {
            completion_tokens: 50,
            observed_frames: 50,
            mode: "short".into(),
            ..Default::default()
        };
        let _ = tracker.update(&sa, t0);
        assert_eq!(tracker.tokens_received, 50);

        // Engine B: resets the tracker.
        let sb = MetricsSnapshot {
            completion_tokens: 10,
            observed_frames: 10,
            mode: "Concurrency".into(),
            ..Default::default()
        };
        let _ = tracker.update(&sb, t0 + Duration::from_secs(5));
        assert_eq!(tracker.tokens_received, 10);
        assert_eq!(tracker.current_mode, "Concurrency");
    }

    #[test]
    fn rate_tracker_reset_clears_state() {
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();
        let s = MetricsSnapshot {
            completion_tokens: 100,
            observed_frames: 100,
            mode: "short".into(),
            ..Default::default()
        };
        let _ = tracker.update(&s, t0);
        assert!(tracker.first_token_instant.is_some());

        tracker.reset();
        assert_eq!(tracker.tokens_received, 0);
        assert!(tracker.first_token_instant.is_none());
        assert!(tracker.current_mode.is_empty());
        assert!(tracker.prev_state.is_empty());
        assert!(tracker.first_sample);
        assert!(!tracker.finished);
    }

    // ── completion: the graph ends on the Overall Metrics' number ─────

    fn done_stream_snap(id: u32, gen: Option<f64>, tokens: u64, looping: bool) -> MetricsSnapshot {
        MetricsSnapshot {
            completion_tokens: tokens,
            observed_frames: tokens,
            mode: "short".into(),
            status: StreamStatus::Done,
            streams: vec![StreamMetric {
                id,
                state: StreamStatus::Done,
                gen_tps: gen,
                tg_tokens: (tokens > 0).then_some(tokens),
                looping,
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    fn live_stream_snap(id: u32, tokens: u64) -> MetricsSnapshot {
        MetricsSnapshot {
            completion_tokens: tokens,
            observed_frames: tokens,
            mode: "short".into(),
            status: StreamStatus::Streaming,
            streams: vec![StreamMetric {
                id,
                state: StreamStatus::Streaming,
                tg_tokens: (tokens > 0).then_some(tokens),
                ..Default::default()
            }],
            ..Default::default()
        }
    }

    #[test]
    fn rate_tracker_final_sample_is_the_exact_gen_tps() {
        // The user-reported Live-vs-Overall discrepancy: the rolling
        // series' last sample must be the stream's exact `gen_tps`
        // (`total_tokens / generation_duration`) — the same value the
        // OVERALL METRICS panel records — not the tracker's decaying
        // cumulative estimate.
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();

        // In flight at ~100 t/s.
        let _ = tracker.update(&live_stream_snap(0, 100), t0 + Duration::from_millis(600));

        // The stream completes at exactly 55.16 t/s (the overall number).
        let final_rate = tracker
            .update(
                &done_stream_snap(0, Some(55.16), 256, false),
                t0 + Duration::from_millis(2000),
            )
            .expect("completion yields the final sample");
        assert_eq!(final_rate, RateSample::Final(55.16));

        // Post-completion updates (the engine's duplicate final publish,
        // the 100 ms hardware poller re-publishing the merged snapshot)
        // must NOT sample: the fixed token count over a growing elapsed
        // would decay the series below the true rate.
        assert!(
            tracker
                .update(
                    &done_stream_snap(0, Some(55.16), 256, false),
                    t0 + Duration::from_millis(3000)
                )
                .is_none(),
            "no sample after completion (duplicate publish)"
        );
        assert!(
            tracker
                .update(
                    &done_stream_snap(0, Some(55.16), 256, false),
                    t0 + Duration::from_millis(4500)
                )
                .is_none(),
            "no sample after completion (hardware poller, 1.5 s later)"
        );
    }

    #[test]
    fn rate_tracker_final_sample_excludes_looping_streams() {
        // A looping stream's repeating output is not a measurement: the
        // overall panel excludes it from its `gen` samples, and so must
        // the final rolling sample.
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();
        let _ = tracker.update(&live_stream_snap(0, 100), t0 + Duration::from_millis(600));
        // A snapshot with a looping Done row and a clean Done row: the
        // final sample is the clean row's rate, matching the overall
        // accumulator's exclusion.
        let snap = MetricsSnapshot {
            completion_tokens: 150,
            observed_frames: 150,
            mode: "short".into(),
            status: StreamStatus::Done,
            streams: vec![
                StreamMetric {
                    id: 0,
                    state: StreamStatus::Done,
                    gen_tps: Some(999.0),
                    tg_tokens: Some(100),
                    looping: true,
                    ..Default::default()
                },
                StreamMetric {
                    id: 1,
                    state: StreamStatus::Done,
                    gen_tps: Some(42.0),
                    tg_tokens: Some(50),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        assert_eq!(
            tracker.update(&snap, t0 + Duration::from_millis(2000)),
            Some(RateSample::Final(42.0)),
            "looping stream's rate is excluded from the final sample"
        );
    }

    #[test]
    fn rate_tracker_re_arms_when_a_new_stream_starts() {
        let mut tracker = DecodeRateTracker::default();
        let t0 = Instant::now();

        // Complete a stream.
        let _ = tracker.update(&live_stream_snap(0, 100), t0 + Duration::from_millis(600));
        let _ = tracker.update(
            &done_stream_snap(0, Some(55.0), 256, false),
            t0 + Duration::from_millis(2000),
        );
        assert!(tracker.finished);

        // A new iteration re-uses stream id 0: the tracker re-arms and
        // latches a fresh convergence curve.
        let next = tracker.update(&live_stream_snap(0, 12), t0 + Duration::from_millis(2100));
        assert!(
            next.is_none(),
            "re-armed stream has no sample yet (0 elapsed)"
        );
        assert!(!tracker.finished, "re-armed for the new stream");
        assert!(tracker.first_token_instant.is_some(), "fresh latch");

        // A snapshot with no stream rows (the hardware poller's merge —
        // same mode, same tokens, no new stream state) does NOT re-arm a
        // finished tracker.
        let mut tracker2 = DecodeRateTracker::default();
        let _ = tracker2.update(
            &done_stream_snap(0, Some(55.0), 256, false),
            t0 + Duration::from_millis(600),
        );
        assert!(tracker2.finished);
        let poller_like = MetricsSnapshot {
            mode: "short".into(),
            completion_tokens: 256,
            observed_frames: 256,
            ..Default::default()
        };
        assert!(tracker2
            .update(&poller_like, t0 + Duration::from_millis(700))
            .is_none());
        assert!(
            tracker2.finished,
            "stream-less snapshot keeps the tracker finished"
        );
    }

    #[test]
    fn final_sample_bypasses_the_cadence_for_short_streams() {
        // A short stream can complete less than a second after its last
        // live sample; the normal 1 s cadence would drop the exact final
        // rate. `force_sample` guarantees the series ends on it.
        let state = MetricsState::new();
        state.update(live_stream_snap(0, 100));
        std::thread::sleep(Duration::from_millis(600));
        state.update(live_stream_snap(0, 200));
        // Complete 300 ms later (inside the 1 s cadence of the last sample).
        std::thread::sleep(Duration::from_millis(300));
        state.update(done_stream_snap(0, Some(55.16), 256, false));

        let series = state.load().throughput_series.clone();
        assert!(series.len() >= 2, "live sample + final sample: {series:?}");
        assert_eq!(
            series.last().copied(),
            Some(55.16),
            "the series ends on the stream's exact gen_tps: {series:?}"
        );
    }

    #[test]
    fn live_series_ends_where_the_overall_metrics_begin() {
        // The end-to-end regression for the user-reported discrepancy:
        // the LIVE THROUGHPUT graph (the rolling series) must end on the
        // *same number* the OVERALL METRICS panel reports — one value,
        // one source. The stream runs at ~55 t/s and completes at
        // exactly 55.16 (256 tokens / its decode window).
        let state = MetricsState::new();
        state.update(live_stream_snap(0, 100));
        std::thread::sleep(Duration::from_millis(600));
        state.update(live_stream_snap(0, 200));
        state.update(done_stream_snap(0, Some(55.16), 256, false));

        // A post-completion update (the hardware poller re-publishing
        // the merged snapshot a second later) must not move the series.
        std::thread::sleep(Duration::from_millis(1100));
        state.update(done_stream_snap(0, Some(55.16), 256, false));

        let loaded = state.load();
        let series = &loaded.throughput_series;
        assert!(!series.is_empty(), "the graph has samples: {series:?}");
        assert_eq!(
            series.last().copied(),
            Some(55.16),
            "the graph's final bar is the stream's exact gen_tps: {series:?}"
        );
        // The OVERALL METRICS panel reads the same value from the same
        // completion transition.
        assert!(
            (loaded.overall.gen.avg - 55.16).abs() < 1e-9,
            "overall avg: {}",
            loaded.overall.gen.avg
        );
        assert!(
            (loaded.overall.gen.max - 55.16).abs() < 1e-9,
            "overall max: {}",
            loaded.overall.gen.max
        );
        assert_eq!(
            series.last().copied(),
            Some(loaded.overall.gen.avg),
            "graph final == overall avg: the two panels read the same number"
        );
    }
}
