//! Engine F — Flat Out: real-world maximum throughput at the optimal
//! user count.
//!
//! The goal is to measure what a production deployment actually serves:
//! the server's **total** tokens/sec when it is loaded at its
//! **sweet-spot concurrency** — the number of users Engine B's sweep
//! recommends (the highest level where every user still gets ≥ 29.4
//! t/s: 30 t/s ideal with a 2% margin). That aggregate number, with the
//! per-stream number confirming each user stays comfortable, is the
//! figure to quote when comparing servers or configurations.
//!
//! Method:
//!
//! * **`n` concurrent streams** (the sweet spot, or
//!   [`DEFAULT_STREAM_COUNT`] when Engine B did not run) — all spawned
//!   together via the shared [`WorkerPool`];
//! * **one minimal (~15-token) open-ended prompt per stream** — prefill
//!   takes <100 ms, negligible against the 60-second decode window;
//! * **`max_tokens` = [`MAX_TOKENS`] (100,000) clamped to the model's
//!   max context + `ignore_eos`** — effectively unlimited, so the
//!   *only* stop condition is the 60-second window
//!   ([`WINDOW_SECS`]); `ignore_eos` keeps llama.cpp servers from
//!   letting the model end a stream on its own end-token (other
//!   backends ignore the field). Before the run, the engine probes the
//!   model's max context (`GET /v1/models`, CF-29) and clamps the
//!   requested `max_tokens` to it — servers that cap generation
//!   reject a 100,000 request with a uniform HTTP 400, which is then
//!   surfaced as a clear **configuration error** ("requested tokens
//!   exceed model limit"), never a silent 0 t/s;
//! * **all streams start together and are all aborted at the 60 s mark**
//!   (aborting the supervisor drops every worker task, closing its
//!   socket so the server stops generating);
//! * **authoritative token counting** — each stream's count is the
//!   server's `usage.completion_tokens` when it arrives, else the
//!   re-tokenized text ([`authoritative_tokens`]) — never the raw frame
//!   tally, which undercounts batched servers (vLLM MTP) by 30–40 %;
//! * **an honest headline** — the 60 s window aborts each stream before
//!   its terminal `usage` frame arrives, so by default the count falls to
//!   the `chars/4` text estimate. [`FlatOutResult`] flags that:
//!   `tokens_estimated` + `counting_method`, and the terminal summary
//!   prefixes the headline `≈ … (est., …)` — an estimate is never
//!   presented as a precise measurement.
//!
//! The result records the **aggregate t/s** (the headline), the
//! **per-stream t/s in two labeled views** (CF-04: `÷ completed
//! streams` — the capacity view — and `÷ requested streams` — the
//! user-experience view; the legacy `per_stream_tps` remains the
//! requested view for back-compat), the total tokens, and the pooled
//! TTFT / ITL evidence.
//!
//! Measurement-isolation note: all timing is captured in the workers
//! (quanta, `T0..Tn`); this module only takes deltas of those records and
//! counts the tokens that arrive. Nothing here touches the TUI render
//! path.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::client::pool::{PoolEvent, WorkerPool};
use crate::client::{authoritative_tokens, StreamError, StreamEvent};
use crate::config::Config;
use crate::engines::concurrency::SweepResult;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::log::{Context, RunLogger};
use crate::metrics::histogram::LatencyHistogram;
use crate::metrics::state::{MetricsSnapshot, MetricsState, StreamMetric, StreamStatus};
use crate::prompt::tokenizer::count_tokens;
use crate::prompt::{GeneratedPrompt, PromptGenerator, Tokenizer, TokenizerError};
use crate::sse::{Chunk, Usage};
use crate::timing::MonotonicInstant;

/// The continuous run window (seconds). All streams start together and
/// are all aborted when it elapses — the only real stop condition of the
/// test.
pub const WINDOW_SECS: f64 = 60.0;

/// The run window in nanoseconds (the same domain as
/// [`MonotonicInstant::delta_nanos`]) for the exact window comparison.
/// Kept in sync with [`WINDOW_SECS`].
const WINDOW_NS: u64 = 60 * 1_000_000_000;

/// The `max_tokens` requested per stream: 100,000 — effectively
/// unlimited. The 60-second window, not this cap, ends the test.
/// (CF-29: the cap actually sent is this value clamped to the
/// probed model max context — see [`clamp_max_tokens`].)
pub const MAX_TOKENS: u32 = 100_000;

/// (CF-04) The `failed_stream_ratio` above which the summary line /
/// run log flags the run: more than a fifth of the requested streams
/// produced no tokens, so the capacity and user-experience views
/// differ materially.
pub const FAILED_STREAM_RATIO_FLAG: f64 = 0.20;

/// (CF-29) The `/v1/models` fields that carry the model's max context,
/// in priority order — the first present, parseable value wins. vLLM
/// exposes `max_model_len`; other OpenAI-compatible backends may use
/// the OpenAI-style names. A server that exposes none of them degrades
/// to the unclamped [`MAX_TOKENS`] (N/A-never-fail).
const MODEL_CONTEXT_FIELDS: [&str; 5] = [
    "max_model_len",
    "max_context",
    "context_length",
    "max_context_length",
    "max_output_tokens",
];

/// (CF-29) Overall budget for the max-context probe: model lists are
/// small, so a server that cannot answer within 10 s is stuck — the
/// run proceeds unclamped rather than hanging.
const PROBE_TIMEOUT: Duration = Duration::from_secs(10);

/// The stream count when Engine B did not run (there is no sweet spot to
/// read): a small, always-valid load.
pub const DEFAULT_STREAM_COUNT: usize = 3;

/// The minimal open-ended prompt (~15 tokens) every stream runs.
///
/// Short enough that prefill takes <100 ms (negligible against the 60 s
/// window), open-ended enough that the model keeps generating without
/// stopping, and not a question (questions get short answers).
pub const MINIMAL_PROMPT: &str =
    "Continue this story: The old lighthouse keeper walked down the spiral stairs and";

/// Where the run's stream count came from (the JSON export's
/// `stream_count_source` field).
///
/// The distinct sources are all real, and the label names which one drove
/// the run:
///
/// * [`Self::ConcurrencyMaxThroughput`] — the **max-throughput** level
///   (the sweep *envelope*: the last level before the saturation knee,
///   else the peak-aggregate level). This is the count Flat Out runs at.
/// * [`Self::ConcurrencySweetSpot`] — the **practical** sweet spot (the
///   highest level where every stream still gets ≥ 29.4 t/s — the
///   hosting recommendation). Retained for back-compat: it is what
///   pre-envelope runs (and the JSON/DB records they produced) carry.
/// * [`Self::SweepFallbackSingleStream`] — the sweep **ran** but produced
///   no usable curve (the envelope is `None`), so Flat Out falls back to a
///   single stream. B4: a 1-stream floor is *not* a measured max-throughput
///   level, so it gets its own tag — the log + JSON name the actual source.
/// * [`Self::Default`] — Engine B did not run, so there was no
///   concurrency data to read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StreamCountSource {
    /// The **practical** sweet spot from Engine B's sweep — the highest
    /// level where every stream still gets ≥ 29.4 t/s (the recommended
    /// production hosting concurrency). Retained for back-compat: new runs
    /// tag the max-throughput level via [`Self::ConcurrencyMaxThroughput`].
    ConcurrencySweetSpot,
    /// Engine B's **max-throughput** level (the sweep envelope: the last
    /// level before the saturation knee, else the peak-aggregate level) —
    /// the real-world full-load count Flat Out runs at.
    ConcurrencyMaxThroughput,
    /// The sweep ran but produced no usable curve (the envelope is `None`),
    /// so Flat Out falls back to a single stream (B4). Distinct from
    /// [`Self::ConcurrencyMaxThroughput`] — a 1-stream floor is not a
    /// measured max-throughput level.
    SweepFallbackSingleStream,
    /// The built-in default ([`DEFAULT_STREAM_COUNT`]) — Engine B did not
    /// run, so there was no concurrency data to read.
    Default,
}

impl StreamCountSource {
    /// The JSON label (`"concurrency_sweet_spot"` /
    /// `"concurrency_max_throughput"` / `"sweep_fallback_single_stream"` /
    /// `"default"`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConcurrencySweetSpot => "concurrency_sweet_spot",
            Self::ConcurrencyMaxThroughput => "concurrency_max_throughput",
            Self::SweepFallbackSingleStream => "sweep_fallback_single_stream",
            Self::Default => "default",
        }
    }

    /// The human-readable descriptor for the startup log line — names what
    /// actually drove the count so the log matches the JSON label.
    pub fn log_descriptor(self) -> &'static str {
        match self {
            Self::ConcurrencyMaxThroughput => "concurrency max-throughput level",
            Self::ConcurrencySweetSpot => "practical sweet spot",
            Self::SweepFallbackSingleStream => "empty sweep — single-stream fallback",
            Self::Default => "default",
        }
    }
}

/// The `(stream count, source)` Flat Out should use given a completed
/// Engine B sweep: the **max-throughput level** (the sweep *envelope* —
/// the last level before the saturation knee, else the peak-aggregate
/// level), falling back to the practical sweet spot only when no curve
/// exists ([`SweepResult::envelope`] is `None` for a sweep that ran but
/// produced no usable levels).
///
/// A single shared pure helper so the TUI (`sequence.rs`) and headless
/// (`headless.rs`) paths can never diverge on which level — or which
/// source label — Flat Out consumes. **B4:** the no-curve fallback is
/// tagged [`StreamCountSource::SweepFallbackSingleStream`] (a 1-stream
/// floor is *not* a measured max-throughput level), so the log + JSON name
/// the actual source.
#[must_use]
pub fn flatout_streams_for(r: &SweepResult) -> (usize, StreamCountSource) {
    match r.envelope() {
        Some(e) => (e.sweet_spot, StreamCountSource::ConcurrencyMaxThroughput),
        None => (r.sweet_spot, StreamCountSource::SweepFallbackSingleStream),
    }
}

/// The complete Flat Out result: `n` concurrent streams (the max-
/// throughput level, or the default) running for one [`WINDOW_SECS`]-
/// second window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlatOutResult {
    /// The number of concurrent streams (Engine B's max-throughput level,
    /// the practical sweet spot, or the default — see
    /// [`stream_count_source`](Self::stream_count_source)).
    pub stream_count: usize,
    /// Where the stream count came from.
    pub stream_count_source: StreamCountSource,
    /// Total tokens generated across **all** streams in the window.
    pub total_tokens: u64,
    /// How long the run took (seconds; ~[`WINDOW_SECS`], or less when
    /// every stream ended on its own).
    pub duration_secs: f64,
    /// The headline number: `total_tokens / duration_secs` — the
    /// server's real-world maximum throughput at full load.
    pub aggregate_tps: f64,
    /// **Back-compat alias of [`tps_per_requested_stream`](Self::tps_per_requested_stream)**
    /// — `aggregate_tps / stream_count`: what each *requested* user
    /// gets at the recommended load (≈ 30 t/s confirms the sweet
    /// spot). Kept so older JSON/DB consumers keep reading the same
    /// number; new consumers should use the two labeled CF-04 views.
    pub per_stream_tps: f64,
    /// **Capacity view** (CF-04): `aggregate_tps / completed_streams`
    /// — what each *healthy* stream sustains. Failed / zero-token
    /// streams are excluded from the denominator, so they no longer
    /// dilute the number. `0.0` when no stream completed.
    #[serde(default)]
    pub tps_per_completed_stream: f64,
    /// **User-experience view** (CF-04): `aggregate_tps /
    /// requested_streams` — the average per *requested* stream,
    /// failures included (the honest "per user" number at the
    /// recommended load; identical to [`per_stream_tps`](Self::per_stream_tps)).
    #[serde(default)]
    pub tps_per_requested_stream: f64,
    /// (CF-04): `failed_streams / requested_streams`. A stream that
    /// never produced a token (a terminal failure, or a stall before
    /// any frame) counts as failed. Flagged in the summary line / run
    /// log when above [`FAILED_STREAM_RATIO_FLAG`] (0.20) — more than a
    /// fifth of the requested load is not producing tokens.
    #[serde(default)]
    pub failed_stream_ratio: f64,
    /// (CF-29) `Some` when the run is a **configuration error, not a
    /// measurement**: every requested stream was rejected by the server
    /// with a uniform HTTP 400 (request too large) and no stream
    /// produced a single token. The message names the condition
    /// ("requested tokens exceed model limit") so a 0 t/s headline is
    /// never read as "the server is slow". `None` for normal runs and
    /// for partial failures (those are dilution, surfaced via
    /// [`failed_stream_ratio`](Self::failed_stream_ratio)).
    #[serde(default)]
    pub config_error: Option<String>,
    /// (CF-29) The `max_tokens` actually sent per stream:
    /// [`MAX_TOKENS`] when the model's max context could not be probed,
    /// else the clamped value (`model max context − prompt tokens`,
    /// floored at 1 — see [`clamp_max_tokens`]). `0` in pre-CF-29
    /// documents (the field did not exist yet).
    #[serde(default)]
    pub max_tokens_used: u32,
    /// Mean time to first token across the streams (milliseconds).
    pub ttft_avg_ms: f64,
    /// Inter-token latency p50, pooled across all streams (milliseconds).
    pub itl_p50_ms: f64,
    /// Inter-token latency p99, pooled across all streams (milliseconds).
    pub itl_p99_ms: f64,
    /// `true` when the headline token total (and therefore the t/s) relies
    /// on at least one *estimated* per-stream count — a stream had no
    /// `usage` frame and no exact tokenizer, so it fell back to the
    /// `chars/4` text estimate. This is the **default** configuration: the
    /// 60 s window aborts each stream before its terminal `usage` frame
    /// arrives. `false` when every stream's count is authoritative (the
    /// server's `usage` or an exact re-tokenization).
    #[serde(default)]
    pub tokens_estimated: bool,
    /// How the headline tokens were counted: `"usage"` (the server's
    /// `usage.completion_tokens` — authoritative), `"tokenizer"` (exact
    /// re-tokenization when `--tokenizer` is supplied), `"chars/4"` (the
    /// flagged text estimate), `"mixed"` (a blend across streams), or
    /// `"none"` (no stream produced a count). The provenance marker that
    /// makes `tokens_estimated` actionable.
    #[serde(default)]
    pub counting_method: String,
}

impl FlatOutResult {
    /// Build the result from the events observed per stream.
    ///
    /// Each stream's token count follows the **Authoritative Counting
    /// Rule** ([`authoritative_tokens`]): the server's
    /// `usage.completion_tokens` when it arrived (the rare path — the
    /// 60 s window normally aborts the streams first), else the
    /// re-tokenized reasoning + content text (the exact tokenizer when
    /// one is supplied, else `chars/4`). Counting SSE frames is never
    /// used: batched servers (vLLM MTP) pack multiple tokens into one
    /// frame, which is the 30–40 % undercount this engine used to show.
    ///
    /// The result records **how** the total was counted
    /// ([`FlatOutResult::counting_method`]) and flags it
    /// ([`FlatOutResult::tokens_estimated`]) whenever any stream fell to
    /// the `chars/4` estimate — the default case, since the 60 s window
    /// aborts each stream before its terminal `usage` frame arrives. An
    /// estimate is never presented as a precise measurement.
    ///
    /// The **aggregate t/s** is the total over the *wall* window
    /// (`total_tokens / duration_secs`) — prefill and any trailing gap
    /// included, exactly what a full-load deployment sustains. The ITL
    /// percentiles are the gaps between consecutive token frames, pooled
    /// across all streams.
    ///
    /// The **per-stream t/s** is reported in two labeled views
    /// (CF-04): `÷ completed streams` (capacity — failed / zero-token
    /// streams don't dilute it) and `÷ requested streams`
    /// (user-experience — the honest per-user average; the legacy
    /// `per_stream_tps` equals this one). A stream *completed* when it
    /// produced at least one token frame (the 60 s abort still counts —
    /// it ran to the window); it *failed* when it has a terminal
    /// `Failed` event or produced no tokens at all.
    #[must_use]
    pub fn from_events(
        per_stream: &[Vec<StreamEvent>],
        duration_secs: f64,
        tokenizer: Option<&Tokenizer>,
        source: StreamCountSource,
    ) -> Self {
        let stream_count = per_stream.len().max(1);
        let mut total_tokens = 0u64;
        let mut tokens_estimated = false;
        let mut methods: Vec<&'static str> = Vec::new();
        let mut ttfts_ms: Vec<f64> = Vec::new();
        let mut itl = LatencyHistogram::default();
        // CF-04: the dual-denominator stream classification.
        let mut completed_streams = 0usize;
        let mut failed_streams = 0usize;
        for events in per_stream {
            // The stream's authoritative count (usage → text estimate),
            // with the `estimated` flag that separates the server's
            // `usage` / an exact tokenizer from the `chars/4` fallback.
            let usage = stream_usage(events);
            let (tokens, estimated) = authoritative_tokens(events, usage, tokenizer);
            total_tokens += tokens;
            if estimated {
                tokens_estimated = true;
            }
            methods.push(counting_method(usage, estimated));
            // CF-04: classify the stream for the dual denominators —
            // *completed* when it produced at least one token frame
            // (the 60 s abort still counts: it ran to the window), else
            // *failed* (a terminal error, or a stall before any frame).
            if events
                .iter()
                .any(|e| matches!(e, StreamEvent::Frame { frame, .. } if frame.chunk.is_token()))
            {
                completed_streams += 1;
            } else {
                failed_streams += 1;
            }
            // Pooled ITL: the gaps between this stream's consecutive
            // token frames.
            let mut prev: Option<MonotonicInstant> = None;
            for e in events {
                if let StreamEvent::Frame { frame, at, .. } = e {
                    if frame.chunk.is_token() {
                        if let Some(p) = prev {
                            itl.record(p.delta_nanos(at));
                        }
                        prev = Some(*at);
                    }
                }
            }
            // TTFT of the stream's first token frame.
            if let Some(ts) = events.iter().find_map(|e| match e {
                StreamEvent::Frame {
                    frame, timestamps, ..
                } if frame.chunk.is_token() => Some(*timestamps),
                _ => None,
            }) {
                if let Some(ns) = ts.ttft_nanos() {
                    ttfts_ms.push(ns as f64 / 1e6);
                }
            }
        }
        let aggregate_tps = if duration_secs > 0.0 {
            total_tokens as f64 / duration_secs
        } else {
            0.0
        };
        // CF-04: the dual per-stream denominators. `requested` is the
        // streams the run asked for (the legacy single view);
        // `completed` excludes the failed / zero-token streams so they
        // stop diluting the capacity number (no divide-by-zero: an all-
        // failed run reports 0.0, not ∞).
        let tps_per_requested_stream = aggregate_tps / stream_count as f64;
        let tps_per_completed_stream = if completed_streams > 0 {
            aggregate_tps / completed_streams as f64
        } else {
            0.0
        };
        let failed_stream_ratio = failed_streams as f64 / stream_count as f64;
        let (itl_p50_ms, itl_p99_ms) = itl_percentiles_ms(&itl);
        let ttft_avg_ms = if ttfts_ms.is_empty() {
            0.0
        } else {
            ttfts_ms.iter().sum::<f64>() / ttfts_ms.len() as f64
        };
        Self {
            stream_count,
            stream_count_source: source,
            total_tokens,
            duration_secs,
            aggregate_tps,
            // Back-compat alias: the legacy number is the requested view.
            per_stream_tps: tps_per_requested_stream,
            tps_per_completed_stream,
            tps_per_requested_stream,
            failed_stream_ratio,
            // CF-29: `from_events` does not know the probe outcome —
            // `run()` fills the real cap (and a config error, when the
            // server uniformly rejects every stream) after building the
            // result. `0` marks a result built without a run (the
            // pre-CF-29 shape).
            config_error: None,
            max_tokens_used: 0,
            ttft_avg_ms,
            itl_p50_ms,
            itl_p99_ms,
            tokens_estimated,
            counting_method: summarize_methods(&methods),
        }
    }

    /// The one-line summary for the sequence header / headless report.
    ///
    /// When the token total is an estimate (no `usage` frame and no exact
    /// tokenizer — the default configuration), the headline t/s is
    /// prefixed with `≈` and annotated `(est., <method>)`, so the number
    /// is never read as a precise measurement.
    ///
    /// The per-stream figure is shown in both labeled views (CF-04):
    /// `t/s per stream` (÷ requested — the user-experience view) and
    /// `t/s per completed stream` (÷ completed — the capacity view).
    /// A run where more than a fifth of the requested streams produced
    /// no tokens (above [`FAILED_STREAM_RATIO_FLAG`]) carries a `⚠`
    /// failure-ratio flag.
    ///
    /// (CF-29) When [`config_error`](Self::config_error) is set — every
    /// stream was rejected with a uniform HTTP 400 — the line leads
    /// with the configuration error so it is never read as "0 t/s,
    /// slow server".
    #[must_use]
    pub fn summary_line(&self) -> String {
        if let Some(e) = &self.config_error {
            return format!("⚠ CONFIG: {e}");
        }
        let (agg_prefix, est_note) = if self.tokens_estimated {
            ("≈", format!(" (est., {})", self.counting_method))
        } else {
            ("", String::new())
        };
        // CF-04: flag a run where more than a fifth of the requested
        // streams produced no tokens.
        let fail_note = if self.failed_stream_ratio > FAILED_STREAM_RATIO_FLAG {
            format!(
                " · ⚠ {:.0}% of streams failed",
                self.failed_stream_ratio * 100.0
            )
        } else {
            String::new()
        };
        format!(
            "{agg_prefix}{agg:.1} t/s aggregate ({n} streams){est_note} · {req:.1} t/s per stream · {comp:.1} t/s per completed stream{fail_note} · {tok} tok in {d:.0}s",
            agg = self.aggregate_tps,
            n = self.stream_count,
            req = self.tps_per_requested_stream,
            comp = self.tps_per_completed_stream,
            tok = self.total_tokens,
            d = self.duration_secs
        )
    }

    /// The `--json` object for this result.
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        let mut obj = serde_json::json!({
            "stream_count": self.stream_count,
            "stream_count_source": self.stream_count_source,
            "total_tokens": self.total_tokens,
            "duration_secs": self.duration_secs,
            "aggregate_tps": self.aggregate_tps,
            "per_stream_tps": self.per_stream_tps,
            // CF-04: the dual per-stream denominators (exact contract
            // names, shared with Engine B): capacity (÷ completed) and
            // user-experience (÷ requested), plus the failed ratio.
            "tps_per_completed_stream": self.tps_per_completed_stream,
            "tps_per_requested_stream": self.tps_per_requested_stream,
            "failed_stream_ratio": self.failed_stream_ratio,
            "ttft_avg_ms": self.ttft_avg_ms,
            "itl_p50_ms": self.itl_p50_ms,
            "itl_p99_ms": self.itl_p99_ms,
            // CF-07: provenance of the headline token count. `true` when
            // the total relies on a `chars/4` text estimate (the default —
            // the 60 s window aborts each stream before its terminal
            // `usage` frame arrives); `counting_method` names the source.
            "tokens_estimated": self.tokens_estimated,
            "counting_method": self.counting_method,
            // CF-29: the `max_tokens` actually sent per stream
            // ([`MAX_TOKENS`] when the model's max context could not be
            // probed, else the clamped value).
            "max_tokens_used": self.max_tokens_used,
        });
        // CF-29: the configuration-error diagnostic (uniform HTTP 400
        // rejection of every stream) — present only when set.
        if let Some(e) = &self.config_error {
            obj.as_object_mut()
                .unwrap()
                .insert("config_error".into(), serde_json::json!(e));
        }
        obj
    }
}

/// `(p50 ms, p99 ms)` of a pooled inter-token-latency histogram (the
/// same histogram Engine A uses for its ITL distribution).
fn itl_percentiles_ms(h: &LatencyHistogram) -> (f64, f64) {
    (h.p50() / 1e6, h.p99() / 1e6)
}

/// The per-stream counting-method label: the server's `usage` when it
/// carries a positive completion count, an exact re-tokenization when a
/// tokenizer was supplied (no usage), else the `chars/4` text estimate
/// (the only case [`authoritative_tokens`] flags `estimated`).
fn counting_method(usage: Option<Usage>, estimated: bool) -> &'static str {
    if estimated {
        "chars/4"
    } else if usage.is_some_and(|u| u.completion_tokens > 0) {
        "usage"
    } else {
        "tokenizer"
    }
}

/// Collapse the per-stream method labels into one aggregate label: a
/// single label when every stream agrees, `"mixed"` when they differ,
/// `"none"` when no stream produced a count.
fn summarize_methods(methods: &[&'static str]) -> String {
    if methods.is_empty() {
        return "none".to_string();
    }
    let distinct: Vec<&'static str> = methods
        .iter()
        .copied()
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    match distinct.as_slice() {
        [only] => only.to_string(),
        _ => "mixed".to_string(),
    }
}

// ── CF-29: respect the model's max context ─────────────────────────────

/// Extract the model's max context from a `/v1/models` response body
/// (CF-29): find the entry whose `id` matches `model` and read the
/// first present max-context field ([`MODEL_CONTEXT_FIELDS`]). Pure (no
/// network) — unit-testable with mock bodies. `None` when the body is
/// not JSON, the model is absent, or no field is present (graceful:
/// the run proceeds unclamped).
#[must_use]
pub fn parse_model_max_context(body: &str, model: &str) -> Option<u64> {
    let v: serde_json::Value = serde_json::from_str(body).ok()?;
    let data = v.get("data")?.as_array()?;
    let entry = data
        .iter()
        .find(|m| m.get("id").and_then(|id| id.as_str()) == Some(model))?;
    MODEL_CONTEXT_FIELDS
        .iter()
        .find_map(|f| entry.get(*f))
        .and_then(|v| {
            v.as_u64()
                .or_else(|| v.as_str().and_then(|s| s.parse().ok()))
        })
}

/// The `max_tokens` to request per stream (CF-29): [`MAX_TOKENS`] when
/// the model's max context could not be probed, else clamped to
/// `max context − prompt tokens` (servers enforce
/// `prompt + max_tokens ≤ max context`) — floored at 1 so a tiny
/// context still yields a legal request.
#[must_use]
pub fn clamp_max_tokens(probed_max_context: Option<u64>, prompt_tokens: u64) -> u32 {
    match probed_max_context {
        None => MAX_TOKENS,
        Some(max_ctx) => {
            let headroom = max_ctx.saturating_sub(prompt_tokens);
            (MAX_TOKENS as u64).min(headroom.max(1)) as u32
        }
    }
}

/// Detect the CF-29 condition: **every** requested stream was rejected
/// by the server with a uniform HTTP 400 (request too large) and no
/// stream produced a single token. That is a *configuration* error
/// ("requested tokens exceed model limit"), not a measurement — the run
/// must not be reported as a silent 0 t/s. `None` for any partial
/// failure (those are dilution, flagged via `failed_stream_ratio`), a
/// non-400 failure, or a stream that produced tokens before failing.
#[must_use]
pub fn uniform_400_config_error(
    per_stream: &[Vec<StreamEvent>],
    max_tokens_used: u32,
    probed_max_context: Option<u64>,
) -> Option<String> {
    if per_stream.is_empty() {
        return None;
    }
    let mut tokens_seen = false;
    let mut all_400 = true;
    for events in per_stream {
        // A stream with no terminal `Failed` event at all (stalled,
        // aborted in flight) is not a 400 rejection.
        if !events
            .iter()
            .any(|e| matches!(e, StreamEvent::Failed { .. }))
        {
            all_400 = false;
        }
        for e in events {
            match e {
                StreamEvent::Frame { frame, .. } if frame.chunk.is_token() => {
                    tokens_seen = true;
                }
                StreamEvent::Failed { error, .. }
                    if !matches!(error, StreamError::Http { status: 400, .. }) =>
                {
                    all_400 = false;
                }
                _ => {}
            }
        }
    }
    if all_400 && !tokens_seen {
        let probed_note = match probed_max_context {
            Some(max) => format!(" (probed model max context: {max})"),
            None => " (the model's max context could not be probed)".to_string(),
        };
        Some(format!(
            "requested tokens exceed model limit: the server rejected every stream with a uniform HTTP 400 (request too large){probed_note}; the per-stream max_tokens was {max_tokens_used}. The server's effective generation cap is lower — reduce the request size or raise the server's max-model-len. This is a configuration error, not a 0 t/s measurement."
        ))
    } else {
        None
    }
}

/// A stream's authoritative `usage` (the last usage frame, or the
/// terminal `Complete` event's usage — they mirror each other).
fn stream_usage(events: &[StreamEvent]) -> Option<Usage> {
    let mut usage = None;
    for e in events {
        match e {
            StreamEvent::Frame { frame, .. } => {
                if let Chunk::Usage(u) = &frame.chunk {
                    usage = Some(*u);
                }
            }
            StreamEvent::Complete { usage: u, .. } => usage = *u,
            StreamEvent::Failed { .. } => {}
        }
    }
    usage
}

/// Convert one pooled (stream-tagged) event back to the untagged
/// [`StreamEvent`] of the stream it came from (the per-stream event
/// lists the result / snapshot builders consume).
fn to_stream_event(e: &PoolEvent) -> StreamEvent {
    match e {
        PoolEvent::Frame {
            frame,
            at,
            timestamps,
            ..
        } => StreamEvent::Frame {
            frame: frame.clone(),
            at: *at,
            timestamps: *timestamps,
        },
        PoolEvent::Complete {
            timestamps,
            usage,
            premature,
            malformed_frames,
            looping,
            loop_excluded_tokens,
            ..
        } => StreamEvent::Complete {
            timestamps: *timestamps,
            usage: *usage,
            premature: *premature,
            malformed_frames: *malformed_frames,
            looping: *looping,
            loop_excluded_tokens: *loop_excluded_tokens,
        },
        PoolEvent::Failed {
            timestamps,
            error,
            looping,
            loop_excluded_tokens,
            ..
        } => StreamEvent::Failed {
            timestamps: *timestamps,
            error: error.clone(),
            looping: *looping,
            loop_excluded_tokens: *loop_excluded_tokens,
        },
    }
}

/// Engine construction errors.
#[derive(Debug, Error)]
pub enum FlatOutError {
    /// The shared `reqwest` client could not be built.
    #[error("failed to build HTTP client: {0}")]
    Client(#[from] reqwest::Error),
    /// An explicit `--tokenizer` file failed to load/parse.
    #[error("transparent")]
    Tokenizer(#[from] TokenizerError),
}

/// The Flat Out engine: `n` concurrent streams (the sweet spot) with the
/// minimal prompt, an effectively-unlimited `max_tokens` cap, and one
/// [`WINDOW_SECS`]-second window that ends the test.
#[derive(Debug)]
pub struct FlatOutEngine {
    cfg: Config,
    client: reqwest::Client,
    generator: PromptGenerator,
    /// The number of concurrent streams (the sweet spot, or the default).
    stream_count: usize,
    /// Where the stream count came from (the result's
    /// `stream_count_source`).
    stream_count_source: StreamCountSource,
    /// Optional TUI seam: publish live [`MetricsSnapshot`]s.
    metrics: Option<Arc<MetricsState>>,
    /// Optional sequence seam: publish [`EngineProgress`].
    progress: Option<Arc<ProgressBus>>,
    /// Optional `Space`-key pause gate.
    pause: Option<Arc<RunPause>>,
    /// Optional run logger.
    logger: Option<Arc<RunLogger>>,
}

impl FlatOutEngine {
    /// Build the engine from the resolved config (the stream count
    /// defaults to [`DEFAULT_STREAM_COUNT`] — use
    /// [`stream_count`](Self::stream_count) to drive it from Engine B's
    /// max-throughput level).
    pub fn new(cfg: &Config) -> Result<Self, FlatOutError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        let generator = match &cfg.tokenizer {
            Some(path) => PromptGenerator::new(Some(Tokenizer::from_file(path)?)),
            None => PromptGenerator::new(None),
        };
        Ok(Self {
            cfg: cfg.clone(),
            client,
            generator,
            stream_count: DEFAULT_STREAM_COUNT,
            stream_count_source: StreamCountSource::Default,
            metrics: None,
            progress: None,
            pause: None,
            logger: None,
        })
    }

    /// Set the stream count (e.g. Engine B's sweet spot) and its source.
    pub fn stream_count(mut self, n: usize, source: StreamCountSource) -> Self {
        self.stream_count = n.max(1);
        self.stream_count_source = source;
        self
    }

    /// Attach a [`MetricsState`] publisher.
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach a [`ProgressBus`].
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach the `Space`-key pause gate.
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger.
    pub fn logger(mut self, logger: Arc<RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// Probe the model's max context via `GET {base}/v1/models` (CF-29).
    ///
    /// Best-effort by design (N/A-never-fail): any failure — no
    /// `/models` route (404), a stalled server (the [`PROBE_TIMEOUT`]
    /// budget), a body with no recognizable max-context field
    /// ([`MODEL_CONTEXT_FIELDS`]), or the configured model not being
    /// listed — yields `None`, and the run proceeds with the unclamped
    /// [`MAX_TOKENS`] (today's behavior). The probe rides the engine's
    /// shared `reqwest` client and the configured API key.
    async fn probe_model_max_context(&self) -> Option<u64> {
        let url = format!(
            "{}/models",
            crate::client::models::normalize_base_url(&self.cfg.url)
        );
        let fut = async {
            let mut req = self.client.get(&url);
            if let Some(key) = &self.cfg.api_key {
                req = req.bearer_auth(key);
            }
            let resp = req.send().await.ok()?;
            if !resp.status().is_success() {
                return None;
            }
            resp.text().await.ok()
        };
        match tokio::time::timeout(PROBE_TIMEOUT, fut).await {
            Ok(Some(body)) => parse_model_max_context(&body, &self.cfg.model),
            Ok(None) => None,
            Err(_) => None, // stalled: degrade to unclamped
        }
    }

    /// Run the `n`-stream window and return the result.
    ///
    /// All streams are spawned together (one [`WorkerPool`] level),
    /// drained until the [`WINDOW_SECS`] window elapses (or every stream
    /// ends on its own, whichever comes first), then aborted: the
    /// supervisor's task tree is dropped, every socket closes, and the
    /// server stops generating.
    pub async fn run(&self) -> FlatOutResult {
        let prompt = self.generate_prompt();
        let n = self.stream_count;

        // The `Space`-key pause: hold before the streams go out.
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }

        // CF-29: probe the model's max context and clamp the requested
        // `max_tokens` to it — servers that cap generation reject a
        // 100,000 request with a uniform HTTP 400 (previously: all
        // streams failed → a silent 0 t/s with no diagnostic). A
        // failed probe degrades to the unclamped cap.
        let probed_max_context = self.probe_model_max_context().await;
        let max_tokens_used = clamp_max_tokens(probed_max_context, prompt.token_count as u64);

        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: 0.0,
                streams: n,
                tokens: 0,
                tps: 0.0,
            });
        }

        if self.stream_count_source == StreamCountSource::Default {
            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!("No concurrency data — using default {n} streams"),
                );
            }
        } else {
            // The non-default sources name what actually drove the count so
            // the run log matches the JSON `stream_count_source` label.
            let source_desc = self.stream_count_source.log_descriptor();
            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!(
                        "Real-world max throughput: {n} streams ({source_desc}) for {WINDOW_SECS:.0}s, max_tokens {max_tokens_used}"
                    ),
                );
            }
        }
        // CF-29: make the clamp visible in the run log (the probe
        // succeeded and reduced the cap).
        if max_tokens_used != MAX_TOKENS {
            if let Some(l) = &self.logger {
                l.info(
                    Context::EngineF,
                    format!(
                        "CF-29: max_tokens clamped {MAX_TOKENS} → {max_tokens_used} to fit the model's max context ({} prompt tokens reserved)",
                        prompt.token_count
                    ),
                );
            }
        }

        let pool = WorkerPool::new(
            self.client.clone(),
            &self.cfg.url,
            &self.cfg.model,
            &prompt.text,
            max_tokens_used,
        )
        .read_timeout(Duration::from_secs(self.cfg.timeout.max(1)))
        // The 60 s window — not the model's own end-token — is the only
        // stop: llama.cpp servers honor `ignore_eos`, other backends
        // ignore the field (graceful degradation).
        .ignore_eos(true)
        // H2: the pool's tokenizer reaches every worker, so the
        // per-stream `stream complete` lines count with the SAME
        // authoritative method as this summary (usage first, else
        // re-tokenized text) — the two reconcile, no 3.5× gap.
        .with_tokenizer(self.generator.tokenizer());
        let pool = match &self.cfg.api_key {
            Some(key) => pool.api_key(key.clone()),
            None => pool,
        };
        let pool = match &self.logger {
            Some(l) => pool.logger(l.clone()),
            None => pool,
        };

        // Spawn all `n` streams together; drain the aggregate channel
        // until the window elapses (then abort) or every stream ends on
        // its own. `tokio::time::timeout` bounds each `recv` to the time
        // remaining in the window, so a slow server is cut at 60 s.
        let (mut rx, supervisor) = pool.spawn_tagged(n, "F");
        let start = MonotonicInstant::now();
        let mut per_stream: Vec<Vec<StreamEvent>> = vec![Vec::new(); n];
        let mut terminals = 0usize;
        let mut batch = 0u32;
        loop {
            let elapsed_ns = start.delta_nanos(&MonotonicInstant::now());
            if elapsed_ns >= WINDOW_NS {
                break;
            }
            let remaining = Duration::from_nanos(WINDOW_NS - elapsed_ns);
            match tokio::time::timeout(remaining, rx.recv()).await {
                Ok(Some(event)) => {
                    if let Some(bucket) = per_stream.get_mut(event.stream() as usize) {
                        bucket.push(to_stream_event(&event));
                    }
                    if event.is_terminal() {
                        terminals += 1;
                    }
                    batch += 1;
                    if batch >= 16 || terminals == n {
                        // Live TUI snapshot + sequence progress bus (the
                        // 10 Hz ticker mirrors the bus into the state
                        // slot). While the window is open the run is live
                        // (`finished = false`).
                        if let Some(state) = &self.metrics {
                            state.update(self.live_snapshot(&per_stream, &start, false));
                        }
                        if let Some(bus) = &self.progress {
                            let (tokens, tps) =
                                Self::observed(&per_stream, self.generator.tokenizer().as_deref());
                            bus.publish(EngineProgress::FlatOut {
                                elapsed_secs: start.elapsed().as_secs_f64(),
                                streams: n,
                                tokens,
                                tps,
                            });
                        }
                        batch = 0;
                    }
                    if terminals == n {
                        break; // every stream ended before the window
                    }
                }
                Ok(None) => break, // channel closed (all workers done)
                Err(_) => break,   // 60 s window elapsed
            }
        }

        // Stop the run: abort the supervisor task — it owns every worker
        // task (and its forwarder), so they are aborted with it and their
        // sockets close (the server stops generating). A no-op when all
        // streams already ended on their own.
        supervisor.abort();
        drop(rx);

        let elapsed = start.elapsed().as_secs_f64();
        let mut result = FlatOutResult::from_events(
            &per_stream,
            elapsed,
            self.generator.tokenizer().as_deref(),
            self.stream_count_source,
        );
        // CF-29: record the cap actually sent. A uniform server
        // rejection of every stream (HTTP 400, no tokens produced) is a
        // *configuration* error — "requested tokens exceed model
        // limit" — surfaced in the result, the summary line, and the
        // run log. It is never a silent 0 t/s measurement.
        result.max_tokens_used = max_tokens_used;
        if let Some(err) =
            uniform_400_config_error(&per_stream, max_tokens_used, probed_max_context)
        {
            result.config_error = Some(err.clone());
            if let Some(l) = &self.logger {
                l.error(Context::EngineF, format!("CONFIG: {err}"));
            }
        }

        // Final publish (the last batch may not have flushed). The 60 s
        // window is over and the workers are aborted: every stream is a
        // *completed* benchmark stream, reported `Done` (the
        // accumulator counts each one's decode rate — the last engine
        // must appear in OVERALL METRICS).
        if let Some(state) = &self.metrics {
            state.update(self.live_snapshot(&per_stream, &start, true));
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::FlatOut {
                elapsed_secs: elapsed,
                streams: n,
                tokens: result.total_tokens,
                tps: result.aggregate_tps,
            });
        }
        if let Some(l) = &self.logger {
            // CF-07: never log the headline as precise when it is an
            // estimate (the default — the window aborts each stream before
            // its terminal `usage` frame arrives).
            let est_note = if result.tokens_estimated {
                format!(" [token count estimated: {}]", result.counting_method)
            } else {
                String::new()
            };
            l.info(
                Context::EngineF,
                format!(
                    "Flat Out: {n} streams, {} tok in {:.1}s — {:.1} t/s aggregate, {:.1} t/s per stream (requested), {:.1} t/s per completed stream (TTFT avg {:.0} ms, ITL p50 {:.1} / p99 {:.1} ms){est_note}",
                    result.total_tokens,
                    result.duration_secs,
                    result.aggregate_tps,
                    result.tps_per_requested_stream,
                    result.tps_per_completed_stream,
                    result.ttft_avg_ms,
                    result.itl_p50_ms,
                    result.itl_p99_ms
                ),
            );
            // CF-04: a level where more than a fifth of the requested
            // streams produced no tokens is flagged, not silently
            // averaged.
            if result.failed_stream_ratio > FAILED_STREAM_RATIO_FLAG {
                l.warn(
                    Context::EngineF,
                    format!(
                        "CF-04: {:.0}% of the requested streams produced no tokens (failed-stream ratio {:.2} > {} threshold) — the capacity and user-experience per-stream views differ materially",
                        result.failed_stream_ratio * 100.0,
                        result.failed_stream_ratio,
                        FAILED_STREAM_RATIO_FLAG
                    ),
                );
            }
        }

        result
    }

    /// The prompt for this run: the minimal (~15-token) open-ended
    /// [`MINIMAL_PROMPT`], so prefill is <100 ms — negligible against the
    /// 60-second decode window.
    pub fn generate_prompt(&self) -> GeneratedPrompt {
        let c = count_tokens(MINIMAL_PROMPT, self.generator.tokenizer().as_deref());
        GeneratedPrompt {
            text: MINIMAL_PROMPT.to_string(),
            token_count: c.tokens,
            estimated: c.estimated,
            nocache: false,
        }
    }

    /// The live progress figures: the authoritative token count so far
    /// (summed across all streams) + its decode rate over the global
    /// first→last token window (prefill/TTFT excluded — the research's
    /// Timing Boundary Rule).
    fn observed(per_stream: &[Vec<StreamEvent>], tokenizer: Option<&Tokenizer>) -> (u64, f64) {
        let mut total = 0u64;
        let mut first: Option<MonotonicInstant> = None;
        let mut last: Option<MonotonicInstant> = None;
        for events in per_stream {
            total += authoritative_tokens(events, stream_usage(events), tokenizer).0;
            for e in events {
                if let StreamEvent::Frame { frame, at, .. } = e {
                    if frame.chunk.is_token() {
                        if first.is_none() {
                            first = Some(*at);
                        }
                        last = Some(*at);
                    }
                }
            }
        }
        let tps = match (first, last) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 {
                    total as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        (total, tps)
    }

    /// Build a live [`MetricsSnapshot`] from the per-stream events
    /// collected so far in the run.
    ///
    /// The decode rate uses the same formula as Engine A and the Overall
    /// Metrics panel: `tokens_received / (T_last − T_first)` — over all
    /// streams combined (the live aggregate).
    ///
    /// `finished` marks the **final** publish (the 60 s window elapsed
    /// and the workers were aborted): every stream is a *completed*
    /// benchmark stream and is reported `Done`. The abort means no
    /// terminal `Complete` event ever arrives for the in-flight streams,
    /// so without this flag they would stay `Streaming` forever and the
    /// `OverallAccumulator`'s non-terminal → terminal detector would
    /// never count this engine's decode rate — the last engine would
    /// silently vanish from the OVERALL METRICS panel.
    fn live_snapshot(
        &self,
        per_stream: &[Vec<StreamEvent>],
        start: &MonotonicInstant,
        finished: bool,
    ) -> MetricsSnapshot {
        let n = per_stream.len().max(1);
        let elapsed_ns = start.delta_nanos(&MonotonicInstant::now()).max(1);
        let (tokens_received, decode_rate) =
            Self::observed(per_stream, self.generator.tokenizer().as_deref());

        // Pooled ITL + one stream row per stream.
        let mut itl = LatencyHistogram::default();
        let mut active = 0usize;
        // CF-30: the estimated-token seam — tainted when any stream's
        // count is the `chars/4` fallback (the 60 s window aborts before
        // the terminal `usage` frame arrives, so without a tokenizer this
        // is the norm — the same rule `from_events` applies to
        // `FlatOutResult.tokens_estimated`).
        let mut any_estimated = false;
        let mut streams: Vec<StreamMetric> = Vec::with_capacity(n);
        for (id, events) in per_stream.iter().enumerate() {
            let mut prev: Option<MonotonicInstant> = None;
            let mut first_token: Option<MonotonicInstant> = None;
            let mut last_token: Option<MonotonicInstant> = None;
            let mut ttft_s: Option<f64> = None;
            let mut complete = false;
            let mut failed = false;
            for e in events {
                match e {
                    StreamEvent::Frame {
                        frame,
                        at,
                        timestamps,
                    } => {
                        if frame.chunk.is_token() {
                            if let Some(p) = prev {
                                itl.record(p.delta_nanos(at));
                            }
                            prev = Some(*at);
                            if first_token.is_none() {
                                first_token = Some(*at);
                                if let Some(ns) = timestamps.ttft_nanos() {
                                    ttft_s = Some(ns as f64 / 1e9);
                                }
                            }
                            last_token = Some(*at);
                        }
                    }
                    StreamEvent::Complete { .. } => complete = true,
                    StreamEvent::Failed { .. } => failed = true,
                }
            }
            let (tokens, gen_tps, estimated) = Self::stream_rate(
                events,
                first_token,
                last_token,
                self.generator.tokenizer().as_deref(),
            );
            if estimated {
                any_estimated = true;
            }
            let state = if finished {
                StreamStatus::Done
            } else if failed {
                StreamStatus::Error
            } else if complete {
                StreamStatus::Done
            } else if tokens > 0 {
                StreamStatus::Streaming
            } else {
                StreamStatus::Waiting
            };
            if state == StreamStatus::Streaming {
                active += 1;
            }
            streams.push(StreamMetric {
                id: id as u32,
                kind: "Content".to_string(),
                state,
                tg_tokens: (tokens > 0).then_some(tokens),
                gen_tps: (gen_tps > 0.0).then_some(gen_tps),
                ttft_s,
                progress: (elapsed_ns as f64 / 1e9 / WINDOW_SECS).min(1.0),
                ..Default::default()
            });
        }

        MetricsSnapshot {
            endpoint: self.cfg.url.clone(),
            backend: String::new(),
            model: self.cfg.model.clone(),
            mode: "FlatOut".to_string(),
            aggregate_tps: decode_rate,
            active_streams: active,
            total_streams: n,
            itl_p50_ns: itl.p50() as u64,
            itl_p90_ns: itl.p90() as u64,
            itl_p99_ns: itl.p99() as u64,
            itl_p999_ns: itl.p999() as u64,
            completion_tokens: tokens_received,
            // The decode-rate numerator: the authoritative token count
            // (usage → text estimate), summed across all streams.
            observed_frames: tokens_received,
            prompt_tokens: 0,
            status: if finished {
                StreamStatus::Done
            } else {
                StreamStatus::Streaming
            },
            estimated: any_estimated,
            streams,
            ..Default::default()
        }
    }

    /// One stream's authoritative token count + its decode rate over its
    /// own first→last token window (`tokens / span` — the same formula
    /// the Overall Metrics panel records per stream) + the CF-30
    /// estimated flag (`true` when the count is the `chars/4` fallback —
    /// no server `usage` and no exact tokenizer).
    fn stream_rate(
        events: &[StreamEvent],
        first_token: Option<MonotonicInstant>,
        last_token: Option<MonotonicInstant>,
        tokenizer: Option<&Tokenizer>,
    ) -> (u64, f64, bool) {
        let (tokens, estimated) = authoritative_tokens(events, stream_usage(events), tokenizer);
        let gen_tps = match (first_token, last_token) {
            (Some(f), Some(l)) => {
                let span_s = f.delta_nanos(&l) as f64 / 1e9;
                if span_s > 0.0 && tokens > 0 {
                    tokens as f64 / span_s
                } else {
                    0.0
                }
            }
            _ => 0.0,
        };
        (tokens, gen_tps, estimated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::concurrency::SweepLevel;
    use crate::sse::{Chunk, ParsedFrame, Usage};
    use crate::timing::StreamTimestamps;

    /// A single content frame at `at` with the given timestamps.
    fn content_frame(at: MonotonicInstant, ts: StreamTimestamps) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content("a".into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: ts,
        }
    }

    /// A content frame carrying `text` (for the text-estimate paths).
    fn content_frame_text(text: &str, at: MonotonicInstant, ts: StreamTimestamps) -> StreamEvent {
        StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Content(text.into()),
                t_nanos: 0,
                done: false,
            },
            at,
            timestamps: ts,
        }
    }

    fn ts_at(t: MonotonicInstant) -> StreamTimestamps {
        StreamTimestamps {
            t0: Some(t),
            t1: Some(t),
            t2: Some(t),
            t3: Some(t),
            t_end: Some(t),
        }
    }

    /// A timestamp record with `t_end` unset — the shape of every frame
    /// captured from an *aborted* stream (the worker is dropped before it
    /// can send the terminal event, so `Tn` is never recorded).
    fn ts_open(t: MonotonicInstant) -> StreamTimestamps {
        StreamTimestamps {
            t0: Some(t),
            t1: Some(t),
            t2: Some(t),
            t3: Some(t),
            t_end: None,
        }
    }

    #[test]
    fn window_is_60_seconds() {
        assert!((WINDOW_SECS - 60.0).abs() < 1e-9);
    }

    #[test]
    fn max_tokens_is_effectively_unlimited() {
        assert_eq!(MAX_TOKENS, 100_000);
    }

    #[test]
    fn minimal_prompt_is_short() {
        // ~15 tokens; the chars/4 estimate keeps this a soft bound.
        let c = count_tokens(MINIMAL_PROMPT, None);
        assert!(c.tokens <= 30, "prompt should be ~15 tokens: {}", c.tokens);
    }

    #[test]
    fn default_stream_count_is_three() {
        assert_eq!(DEFAULT_STREAM_COUNT, 3);
    }

    #[test]
    fn stream_count_source_labels() {
        assert_eq!(
            StreamCountSource::ConcurrencySweetSpot.as_str(),
            "concurrency_sweet_spot"
        );
        assert_eq!(
            StreamCountSource::ConcurrencyMaxThroughput.as_str(),
            "concurrency_max_throughput"
        );
        assert_eq!(
            StreamCountSource::SweepFallbackSingleStream.as_str(),
            "sweep_fallback_single_stream"
        );
        assert_eq!(StreamCountSource::Default.as_str(), "default");
    }

    #[test]
    fn stream_count_source_log_descriptors_name_the_actual_source() {
        // The startup log line must reflect the real source: max-throughput
        // (the envelope) vs practical (the hosting recommendation) vs the
        // empty-sweep fallback vs default.
        assert_eq!(
            StreamCountSource::ConcurrencyMaxThroughput.log_descriptor(),
            "concurrency max-throughput level"
        );
        assert_eq!(
            StreamCountSource::ConcurrencySweetSpot.log_descriptor(),
            "practical sweet spot"
        );
        assert_eq!(
            StreamCountSource::SweepFallbackSingleStream.log_descriptor(),
            "empty sweep — single-stream fallback"
        );
        assert_eq!(StreamCountSource::Default.log_descriptor(), "default");
    }

    /// A minimal sweep level (the Flat Out tests only read `concurrency`,
    /// `aggregate_tps`, and `p90_tpot_ns`).
    fn level(concurrency: usize, tps: f64, p90_ms: f64) -> SweepLevel {
        SweepLevel {
            concurrency,
            context: 0,
            aggregate_tps: tps,
            per_stream_tps: tps / concurrency.max(1) as f64,
            p50_tpot_ns: 0,
            p90_tpot_ns: (p90_ms * 1e6) as u64,
            p99_tpot_ns: 0,
            ttft_p50_ns: 0,
            ttft_p90_ns: 0,
            total_tokens: 0,
            completed_streams: concurrency,
            failed_streams: 0,
            timed_out_streams: 0,
            timed_out_tokens: 0,
            aborted: false,
            wall_ns: 0,
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            estimated: false,
            streams: Vec::new(),
        }
    }

    // ── flatout_streams_for: the shared TUI/headless seam ───────────────

    #[test]
    fn flatout_streams_for_uses_the_envelope_not_the_practical() {
        // The incident shape: practical sweet spot = 1 (level 3 measures
        // 87.3 / 3 = 29.1 t/s, missing the 29.4 bar by 0.3), envelope
        // (max-throughput) = 4 (monotone, no knee). The helper must return
        // the envelope (4), NOT the practical (1) — and tag it as
        // max-throughput.
        let r = SweepResult::from_levels(
            vec![
                level(1, 31.6, 5.0),
                level(2, 56.5, 6.0),
                level(3, 87.3, 7.0),
                level(4, 106.0, 8.0),
            ],
            None,
        );
        assert_eq!(r.sweet_spot, 1, "the practical sweet spot is 1");
        let (n, source) = flatout_streams_for(&r);
        assert_eq!(n, 4, "the envelope (max-throughput), not the practical");
        assert_ne!(n, r.sweet_spot, "consumed value is distinct from the practical");
        assert_eq!(source, StreamCountSource::ConcurrencyMaxThroughput);
    }

    #[test]
    fn flatout_streams_for_is_the_last_level_before_the_knee() {
        // A knee-present curve (throughput plateaus 340→345, +1.5% ≤ 5 %,
        // while p90 spikes 10→30 ms, 3× ≥ 2×) → the envelope is the last
        // level before the knee (8), not the peak.
        let r = SweepResult::from_levels(
            vec![
                level(1, 100.0, 5.0),
                level(2, 190.0, 6.0),
                level(4, 280.0, 7.0),
                level(8, 340.0, 10.0),
                level(16, 345.0, 30.0),
            ],
            None,
        );
        assert_eq!(r.envelope().expect("envelope").knee.map(|k| k.concurrency), Some(16));
        let (n, source) = flatout_streams_for(&r);
        assert_eq!(n, 8, "the last level before the knee");
        assert_eq!(source, StreamCountSource::ConcurrencyMaxThroughput);
    }

    #[test]
    fn flatout_streams_for_falls_back_to_the_practical_when_no_curve() {
        // A sweep that ran but produced no usable levels: envelope() is
        // None, so the helper falls back to the practical sweet spot
        // (from_levels computes it as 1) — B4: tagged with the distinct
        // `SweepFallbackSingleStream` source (a 1-stream floor is *not* a
        // measured max-throughput level).
        let r = SweepResult::from_levels(vec![], None);
        assert!(r.envelope().is_none(), "no curve → no envelope");
        let (n, source) = flatout_streams_for(&r);
        assert_eq!(n, r.sweet_spot, "falls back to the practical sweet spot");
        assert_eq!(n, 1);
        assert_eq!(
            source,
            StreamCountSource::SweepFallbackSingleStream,
            "the empty-sweep fallback is named, not mislabeled max-throughput"
        );
    }

    #[test]
    fn to_dict_carries_the_max_throughput_source_label() {
        let r = FlatOutResult {
            stream_count: 4,
            stream_count_source: StreamCountSource::ConcurrencyMaxThroughput,
            total_tokens: 17_124,
            duration_secs: 60.0,
            aggregate_tps: 285.4,
            per_stream_tps: 71.35,
            tps_per_completed_stream: 71.35,
            tps_per_requested_stream: 71.35,
            failed_stream_ratio: 0.0,
            config_error: None,
            max_tokens_used: MAX_TOKENS,
            ttft_avg_ms: 312.5,
            itl_p50_ms: 28.3,
            itl_p99_ms: 65.1,
            tokens_estimated: false,
            counting_method: "usage".into(),
        };
        let v = r.to_dict();
        assert_eq!(v["stream_count"], 4);
        assert_eq!(v["stream_count_source"], "concurrency_max_throughput");
    }

    #[test]
    fn engine_defaults_to_three_streams() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        assert_eq!(engine.stream_count, DEFAULT_STREAM_COUNT);
        assert_eq!(engine.stream_count_source, StreamCountSource::Default);
    }

    #[test]
    fn engine_stream_count_builder_sets_count_and_source() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        let engine = engine.stream_count(8, StreamCountSource::ConcurrencySweetSpot);
        assert_eq!(engine.stream_count, 8);
        assert_eq!(
            engine.stream_count_source,
            StreamCountSource::ConcurrencySweetSpot
        );
        // The builder floors at one stream (never zero).
        let engine = FlatOutEngine::new(&Config::default())
            .unwrap()
            .stream_count(0, StreamCountSource::Default);
        assert_eq!(engine.stream_count, 1);
    }

    #[test]
    fn from_events_prefers_server_usage_over_the_text_estimate() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now(); // 0 ms
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // 100 ms (first token)
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // 200 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 300 ms (last token)
            let ts = ts_at(t0);
            // Three content frames over 100→300 ms, plus a usage frame
            // claiming 100,000 tokens.
            let events = vec![
                content_frame(t1, ts),
                content_frame(t2, ts),
                content_frame(t3, ts),
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(Usage {
                            prompt_tokens: 15,
                            completion_tokens: 100_000,
                        }),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t3,
                    timestamps: ts,
                },
            ];
            // The server's usage is the authoritative count (primary
            // method); the aggregate is total / wall window (10 s).
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.stream_count, 1);
            assert_eq!(r.total_tokens, 100_000, "server usage is authoritative");
            assert!(!r.tokens_estimated, "usage present → not an estimate");
            assert_eq!(r.counting_method, "usage");
            assert!(
                (r.aggregate_tps - 10_000.0).abs() < 1.0,
                "aggregate: {}",
                r.aggregate_tps
            );
            assert!(
                (r.per_stream_tps - 10_000.0).abs() < 1.0,
                "per-stream: {}",
                r.per_stream_tps
            );
            assert!((r.duration_secs - 10.0).abs() < 1e-9);
        });
    }

    #[test]
    fn from_events_sums_streams_and_pools_itl() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // 100 ms
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // 200 ms
            mock.increment(100_000_000);
            let t3 = MonotonicInstant::now(); // 300 ms
            let ts = ts_at(t0);
            // Two streams: 4-char content frames (chars/4 → 1 token each).
            // Stream 0: two tokens (100→200 ms, a 100 ms gap); stream 1:
            // one token (300 ms). Total 3 tokens over the 10 s window.
            let s0 = vec![
                content_frame_text("abcd", t1, ts),
                content_frame_text("efgh", t2, ts),
            ];
            let s1 = vec![content_frame_text("ijkl", t3, ts)];
            let r = FlatOutResult::from_events(&[s0, s1], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.stream_count, 2);
            assert_eq!(r.total_tokens, 3, "1 + 1 + 1 token per 4-char frame");
            assert!(
                r.tokens_estimated,
                "no usage + no tokenizer → chars/4 estimate"
            );
            assert_eq!(r.counting_method, "chars/4");
            assert!(
                (r.aggregate_tps - 0.3).abs() < 1e-9,
                "aggregate: {}",
                r.aggregate_tps
            );
            assert!(
                (r.per_stream_tps - 0.15).abs() < 1e-9,
                "per-stream: {}",
                r.per_stream_tps
            );
            // The pooled ITL has one 100 ms sample.
            assert!((r.itl_p50_ms - 100.0).abs() < 0.5, "p50: {}", r.itl_p50_ms);
            assert!((r.itl_p99_ms - 100.0).abs() < 0.5, "p99: {}", r.itl_p99_ms);
        });
    }

    #[test]
    fn from_events_no_usage_estimates_from_the_text() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now(); // first token
            mock.increment(100_000_000);
            let t2 = MonotonicInstant::now(); // last token
            let ts = ts_at(t0);
            // No usage frame (the aborted-stream case), no tokenizer: the
            // count is estimated from the full text (`chars/4`), NOT the
            // raw frame tally — counting frames undercounts batched
            // servers. Two frames of 4 chars each → 8 chars → 2 tokens.
            let events = vec![
                content_frame_text("abcd", t1, ts),
                content_frame_text("efgh", t2, ts),
            ];
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(
                r.total_tokens, 2,
                "no usage + no tokenizer → chars/4 estimate"
            );
            assert!(r.tokens_estimated, "the chars/4 fallback is flagged");
            assert_eq!(r.counting_method, "chars/4");
            assert!(
                (r.aggregate_tps - 0.2).abs() < 1e-9,
                "aggregate: {}",
                r.aggregate_tps
            );
        });
    }

    #[test]
    fn from_events_no_usage_retokenizes_with_the_tokenizer() {
        let tok =
            Tokenizer::from_json(include_str!("../../tests/fixtures/mini_tokenizer.json")).unwrap();
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(50_000_000);
            let t1 = MonotonicInstant::now(); // first token
            mock.increment(50_000_000);
            let t2 = MonotonicInstant::now(); // last token
            let ts = ts_at(t0);
            // No usage frame; two content frames "apple " + "apple" → the
            // word-level tokenizer counts "apple apple" as exactly 2
            // tokens.
            let events = vec![
                content_frame_text("apple ", t1, ts),
                content_frame_text("apple", t2, ts),
            ];
            let r =
                FlatOutResult::from_events(&[events], 10.0, Some(&tok), StreamCountSource::Default);
            assert_eq!(r.total_tokens, 2, "re-tokenized count (apple apple)");
            assert!(!r.tokens_estimated, "exact tokenizer → not an estimate");
            assert_eq!(r.counting_method, "tokenizer");
        });
    }

    #[test]
    fn from_events_zero_duration_gives_zero_tps() {
        let t0 = MonotonicInstant::now();
        let events = vec![content_frame_text("abcd", t0, ts_at(t0))];
        let r = FlatOutResult::from_events(&[events], 0.0, None, StreamCountSource::Default);
        assert_eq!(r.total_tokens, 1); // 4 chars → chars/4 = 1 token
        assert_eq!(r.aggregate_tps, 0.0);
        assert_eq!(r.per_stream_tps, 0.0);
    }

    #[test]
    fn from_events_ignores_control_and_counts_reasoning() {
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let mut reasoning = content_frame(t0, ts);
        if let StreamEvent::Frame { frame, .. } = &mut reasoning {
            frame.chunk = Chunk::Reasoning("think".into());
        }
        let control = StreamEvent::Frame {
            frame: ParsedFrame {
                chunk: Chunk::Control,
                t_nanos: 0,
                done: true,
            },
            at: t0,
            timestamps: ts,
        };
        let events = vec![reasoning, control];
        let r = FlatOutResult::from_events(&[events], 5.0, None, StreamCountSource::Default);
        // One reasoning token frame; the `[DONE]` control frame is not a
        // token.
        assert_eq!(r.total_tokens, 1);
    }

    #[test]
    fn from_events_averages_ttft_across_streams() {
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            // Stream 0: request at t0, first token at 100 ms (TTFT 100 ms);
            // stream 1: first token at 300 ms (TTFT 300 ms) → mean 200 ms.
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            mock.increment(200_000_000);
            let t2 = MonotonicInstant::now();
            // TTFT = T3 − T1: the request went out at t0, the first token
            // frame decoded at its arrival instant.
            let ts1 = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t0),
                t2: Some(t1),
                t3: Some(t1),
                t_end: None,
            };
            let ts2 = StreamTimestamps {
                t0: Some(t0),
                t1: Some(t0),
                t2: Some(t2),
                t3: Some(t2),
                t_end: None,
            };
            let s0 = content_frame_text("abcd", t1, ts1);
            let s1 = content_frame_text("efgh", t2, ts2);
            let r = FlatOutResult::from_events(
                &[vec![s0], vec![s1]],
                5.0,
                None,
                StreamCountSource::Default,
            );
            assert!(
                (r.ttft_avg_ms - 200.0).abs() < 1.0,
                "ttft avg: {}",
                r.ttft_avg_ms
            );
        });
    }

    #[test]
    fn to_stream_event_round_trips_all_variants() {
        let (clock, _mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            let ts = ts_at(t0);
            let frame = PoolEvent::from_event(
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Content("tok".into()),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t0,
                    timestamps: ts,
                },
                3,
            );
            assert!(matches!(to_stream_event(&frame), StreamEvent::Frame { .. }));
            let complete = PoolEvent::from_event(
                StreamEvent::Complete {
                    timestamps: ts,
                    usage: Some(Usage {
                        prompt_tokens: 10,
                        completion_tokens: 20,
                    }),
                    premature: false,
                    malformed_frames: 0,
                    looping: false,
                    loop_excluded_tokens: 0,
                },
                1,
            );
            assert!(matches!(
                to_stream_event(&complete),
                StreamEvent::Complete {
                    usage: Some(u),
                    ..
                } if u.completion_tokens == 20
            ));
            let failed = PoolEvent::from_event(
                StreamEvent::Failed {
                    timestamps: ts,
                    error: crate::client::StreamError::Connection("refused".into()),
                    looping: false,
                    loop_excluded_tokens: 0,
                },
                2,
            );
            assert!(matches!(
                to_stream_event(&failed),
                StreamEvent::Failed { .. }
            ));
        });
    }

    #[test]
    fn live_snapshot_reports_per_stream_states() {
        let engine = FlatOutEngine::new(&Config {
            url: "http://127.0.0.1:9".to_string(),
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap();
        let t0 = MonotonicInstant::now();
        // Two streams: one streaming (tokens, no terminal), one failed.
        let s0 = vec![content_frame_text("abcd", t0, ts_open(t0))];
        let s1 = vec![StreamEvent::Failed {
            timestamps: ts_at(t0),
            error: crate::client::StreamError::Connection("refused".into()),
            looping: false,
            loop_excluded_tokens: 0,
        }];
        // Live publish: stream 0 Streaming, stream 1 Error.
        let live = engine.live_snapshot(&[s0.clone(), s1.clone()], &t0, false);
        assert_eq!(live.total_streams, 2);
        assert_eq!(live.active_streams, 1);
        assert_eq!(live.streams[0].state, StreamStatus::Streaming);
        assert_eq!(live.streams[1].state, StreamStatus::Error);
        // Final publish (window elapsed, workers aborted): every stream
        // is a completed benchmark stream, reported Done (the accumulator
        // counts each one's decode rate).
        let final_snap = engine.live_snapshot(&[s0, s1], &t0, true);
        assert_eq!(final_snap.status, StreamStatus::Done);
        assert_eq!(final_snap.active_streams, 0);
        assert_eq!(final_snap.streams[0].state, StreamStatus::Done);
        assert_eq!(final_snap.streams[1].state, StreamStatus::Done);
    }

    #[test]
    fn summary_line_carries_the_headline_numbers() {
        let r = FlatOutResult {
            stream_count: 8,
            stream_count_source: StreamCountSource::ConcurrencySweetSpot,
            total_tokens: 17_124,
            duration_secs: 60.0,
            aggregate_tps: 285.4,
            per_stream_tps: 35.7,
            // CF-04: no failures in this fixture — both views agree.
            tps_per_completed_stream: 35.7,
            tps_per_requested_stream: 35.7,
            failed_stream_ratio: 0.0,
            // CF-29: a normal run — no config error, the unclamped cap.
            config_error: None,
            max_tokens_used: MAX_TOKENS,
            ttft_avg_ms: 312.5,
            itl_p50_ms: 28.3,
            itl_p99_ms: 65.1,
            tokens_estimated: false,
            counting_method: "usage".into(),
        };
        let line = r.summary_line();
        assert!(line.contains("285.4 t/s aggregate (8 streams)"), "{line}");
        assert!(line.contains("35.7 t/s per stream"), "{line}");
        // CF-04: the capacity view is labeled separately.
        assert!(line.contains("35.7 t/s per completed stream"), "{line}");
        assert!(line.contains("17124 tok in 60s"), "{line}");
        // No failure flag when nothing failed.
        assert!(!line.contains("of streams failed"), "{line}");
    }

    #[test]
    fn to_dict_is_the_full_export_object() {
        let r = FlatOutResult {
            stream_count: 8,
            stream_count_source: StreamCountSource::ConcurrencySweetSpot,
            total_tokens: 17_124,
            duration_secs: 60.0,
            aggregate_tps: 285.4,
            per_stream_tps: 35.7,
            // CF-04: no failures in this fixture — both views agree.
            tps_per_completed_stream: 35.7,
            tps_per_requested_stream: 35.7,
            failed_stream_ratio: 0.0,
            // CF-29: a normal run — no config error, the unclamped cap.
            config_error: None,
            max_tokens_used: MAX_TOKENS,
            ttft_avg_ms: 312.5,
            itl_p50_ms: 28.3,
            itl_p99_ms: 65.1,
            tokens_estimated: false,
            counting_method: "usage".into(),
        };
        let v = r.to_dict();
        let keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec![
                "stream_count",
                "stream_count_source",
                "total_tokens",
                "duration_secs",
                "aggregate_tps",
                "per_stream_tps",
                // CF-04: the dual per-stream denominators (exact
                // contract names, shared with Engine B).
                "tps_per_completed_stream",
                "tps_per_requested_stream",
                "failed_stream_ratio",
                "ttft_avg_ms",
                "itl_p50_ms",
                "itl_p99_ms",
                "tokens_estimated",
                "counting_method",
                // CF-29: the cap actually sent (a normal run: unclamped).
                "max_tokens_used"
            ]
        );
        // CF-29: a normal run carries no `config_error` key at all.
        assert!(v.get("config_error").is_none(), "no config_error when None");
        assert_eq!(v["max_tokens_used"], MAX_TOKENS);
        assert_eq!(v["stream_count"], 8);
        assert_eq!(v["stream_count_source"], "concurrency_sweet_spot");
        assert_eq!(v["total_tokens"], 17_124);
        assert!((v["aggregate_tps"].as_f64().unwrap() - 285.4).abs() < 1e-9);
        assert!((v["per_stream_tps"].as_f64().unwrap() - 35.7).abs() < 1e-9);
        assert!((v["tps_per_completed_stream"].as_f64().unwrap() - 35.7).abs() < 1e-9);
        assert!((v["tps_per_requested_stream"].as_f64().unwrap() - 35.7).abs() < 1e-9);
        assert_eq!(v["failed_stream_ratio"].as_f64().unwrap(), 0.0);
        // CF-07: the JSON carries the token-count provenance.
        assert_eq!(v["tokens_estimated"], false);
        assert_eq!(v["counting_method"], "usage");
    }

    #[test]
    fn cf07_usage_present_uses_real_tokens_unflagged() {
        // CF-07 (a): when the server's `usage` frame is present, the
        // headline uses the real `completion_tokens` and is NOT flagged as
        // an estimate — in the terminal summary or the JSON.
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            let ts = ts_at(t0);
            let events = vec![
                content_frame_text("abcd", t1, ts),
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(Usage {
                            prompt_tokens: 15,
                            completion_tokens: 42,
                        }),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t1,
                    timestamps: ts,
                },
            ];
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(
                r.total_tokens, 42,
                "usage.completion_tokens is the real count"
            );
            assert!(!r.tokens_estimated, "usage present → not an estimate");
            assert_eq!(r.counting_method, "usage");
            // Terminal: no estimate marker.
            let line = r.summary_line();
            assert!(!line.starts_with("≈"), "no ≈ when authoritative: {line}");
            assert!(
                !line.contains("(est.,"),
                "no (est.) when authoritative: {line}"
            );
            // JSON: explicit that the count is real.
            let v = r.to_dict();
            assert_eq!(v["tokens_estimated"], false);
            assert_eq!(v["counting_method"], "usage");
        });
    }

    #[test]
    fn cf07_no_usage_flags_the_estimate_terminal_and_json() {
        // CF-07 (b): when no `usage` frame arrives (the default — the 60 s
        // window aborts each stream first) and no tokenizer is supplied,
        // the count is a `chars/4` estimate and MUST be flagged in both the
        // terminal summary and the JSON — never presented as a precise
        // measurement.
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            let ts = ts_at(t0);
            // No usage frame; 8 chars of content → chars/4 = 2 tokens.
            let events = vec![content_frame_text("abcdefgh", t1, ts)];
            let r = FlatOutResult::from_events(&[events], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.total_tokens, 2, "8 chars → chars/4 = 2 tokens");
            assert!(r.tokens_estimated, "no usage + no tokenizer → estimate");
            assert_eq!(r.counting_method, "chars/4");
            // Terminal: the headline is prefixed ≈ and names the estimate.
            let line = r.summary_line();
            assert!(
                line.starts_with("≈"),
                "terminal headline must carry ≈: {line}"
            );
            assert!(
                line.contains("(est., chars/4)"),
                "terminal must name the estimate: {line}"
            );
            // JSON: explicit provenance marker.
            let v = r.to_dict();
            assert_eq!(v["tokens_estimated"], true);
            assert_eq!(v["counting_method"], "chars/4");
        });
    }

    #[test]
    fn cf07_mixed_streams_report_mixed_and_flagged() {
        // One stream with a real `usage` count, one without (chars/4): the
        // aggregate is flagged and the method reads "mixed".
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            let ts = ts_at(t0);
            // Stream 0: usage present (real count 42).
            let s0 = vec![
                content_frame_text("abcd", t1, ts),
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(Usage {
                            prompt_tokens: 15,
                            completion_tokens: 42,
                        }),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t1,
                    timestamps: ts,
                },
            ];
            // Stream 1: no usage (chars/4 → 2 tokens from 8 chars).
            let s1 = vec![content_frame_text("abcdefgh", t1, ts)];
            let r = FlatOutResult::from_events(&[s0, s1], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.total_tokens, 44, "42 (usage) + 2 (chars/4)");
            assert!(r.tokens_estimated, "one estimated stream taints the total");
            assert_eq!(r.counting_method, "mixed");
            assert!(
                r.summary_line().starts_with("≈"),
                "mixed is flagged in the terminal: {}",
                r.summary_line()
            );
            let v = r.to_dict();
            assert_eq!(v["tokens_estimated"], true);
            assert_eq!(v["counting_method"], "mixed");
        });
    }

    // ── H2: one counting method — the summary and the per-stream lines reconcile ──

    #[test]
    fn h2_summary_total_equals_sum_of_per_stream_counts() {
        // H2: the Flat Out summary and the per-stream `stream complete`
        // lines must use ONE counting method — the summary total equals
        // the sum of the per-stream authoritative counts (the worker's
        // line drives from the same `authoritative_count` core over the
        // same text), so the two reconcile exactly (no 3.5× gap).
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            let ts = ts_at(t0);
            // Stream 0: no usage → the text (8 chars → chars/4 = 2).
            let s0 = vec![content_frame_text("abcdefgh", t1, ts)];
            // Stream 1: no usage → the text (4 chars → chars/4 = 1).
            let s1 = vec![content_frame_text("abcd", t1, ts)];
            // Stream 2: usage present → the server's figure (42).
            let s2 = vec![
                content_frame_text("abcd", t1, ts),
                StreamEvent::Frame {
                    frame: ParsedFrame {
                        chunk: Chunk::Usage(Usage {
                            prompt_tokens: 15,
                            completion_tokens: 42,
                        }),
                        t_nanos: 0,
                        done: false,
                    },
                    at: t1,
                    timestamps: ts,
                },
            ];
            let per_stream = vec![s0, s1, s2];
            let r =
                FlatOutResult::from_events(&per_stream, 10.0, None, StreamCountSource::Default);
            // The per-stream counts the worker's completion line computes
            // (the same core, the same text source as `stream_text`).
            let mut sum = 0u64;
            for events in &per_stream {
                let (n, _) = crate::client::stream::authoritative_count(
                    stream_usage(events),
                    &crate::client::stream::stream_text(events),
                    None,
                );
                sum += n;
            }
            assert_eq!(
                r.total_tokens, sum,
                "Σ(per-stream counts) must equal the summary total"
            );
            assert_eq!(r.total_tokens, 45, "2 (chars/4) + 1 (chars/4) + 42 (usage)");
        });
    }

    // ── CF-04: the dual per-stream denominators ─────────────────────────

    /// A stream that failed before producing a single token frame.
    fn failed_stream(t0: MonotonicInstant, status: u16) -> StreamEvent {
        StreamEvent::Failed {
            timestamps: ts_at(t0),
            error: crate::client::StreamError::Http {
                status,
                body: "rejected".into(),
            },
            looping: false,
            loop_excluded_tokens: 0,
        }
    }

    #[test]
    fn cf04_dual_denominators_completed_vs_requested() {
        // 3 requested: two produce tokens (completed — the 60 s abort
        // still counts), one fails before any frame. The capacity view
        // divides by 2, the user-experience view by 3 — and the
        // back-compat `per_stream_tps` equals the requested view.
        let (clock, mock) = quanta::Clock::mock();
        quanta::with_clock(&clock, || {
            let t0 = MonotonicInstant::now();
            mock.increment(100_000_000);
            let t1 = MonotonicInstant::now();
            let ts = ts_at(t0);
            let s0 = vec![content_frame_text("abcd", t1, ts)]; // 1 token
            let s1 = vec![content_frame_text("efgh", t1, ts)]; // 1 token
            let s2 = vec![failed_stream(t0, 400)]; // failed, no tokens
            let r =
                FlatOutResult::from_events(&[s0, s1, s2], 10.0, None, StreamCountSource::Default);
            assert_eq!(r.stream_count, 3, "requested streams");
            assert_eq!(r.total_tokens, 2);
            assert!(
                (r.aggregate_tps - 0.2).abs() < 1e-9,
                "aggregate: {}",
                r.aggregate_tps
            );
            // Capacity view: ÷ 2 completed streams.
            assert!(
                (r.tps_per_completed_stream - 0.1).abs() < 1e-9,
                "capacity view: {}",
                r.tps_per_completed_stream
            );
            // User-experience view: ÷ 3 requested streams.
            assert!(
                (r.tps_per_requested_stream - 0.2 / 3.0).abs() < 1e-9,
                "requested view: {}",
                r.tps_per_requested_stream
            );
            // Back-compat alias: the legacy number is the requested view.
            assert!(
                (r.per_stream_tps - r.tps_per_requested_stream).abs() < 1e-12,
                "per_stream_tps must equal the requested view"
            );
            // The failed ratio: 1 of 3.
            assert!(
                (r.failed_stream_ratio - 1.0 / 3.0).abs() < 1e-9,
                "ratio: {}",
                r.failed_stream_ratio
            );
            // 33% > 20% → the summary line carries the failure flag.
            assert!(
                r.summary_line().contains("⚠ 33% of streams failed"),
                "{}",
                r.summary_line()
            );
        });
    }

    #[test]
    fn cf04_all_failed_no_divide_by_zero_and_full_ratio() {
        // Every stream failed (a uniform rejection): the capacity view
        // is 0.0 (not ∞ / NaN), the ratio is 1.0, and the summary flags
        // it.
        let t0 = MonotonicInstant::now();
        let r = FlatOutResult::from_events(
            &[
                vec![failed_stream(t0, 400)],
                vec![failed_stream(t0, 400)],
                vec![failed_stream(t0, 400)],
            ],
            10.0,
            None,
            StreamCountSource::Default,
        );
        assert_eq!(r.total_tokens, 0);
        assert_eq!(r.aggregate_tps, 0.0);
        assert_eq!(
            r.tps_per_completed_stream, 0.0,
            "no completed streams → 0.0, never a divide-by-zero"
        );
        assert_eq!(r.tps_per_requested_stream, 0.0);
        assert_eq!(r.failed_stream_ratio, 1.0, "all requested streams failed");
        assert!(
            r.summary_line().contains("⚠ 100% of streams failed"),
            "{}",
            r.summary_line()
        );
    }

    #[test]
    fn cf04_failure_flag_only_above_20_percent() {
        // 5 requested, 1 failed → exactly 0.20 → NOT flagged (the flag
        // is strictly greater than the threshold).
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let ok = || vec![content_frame_text("abcd", t0, ts)];
        let r5 = FlatOutResult::from_events(
            &[ok(), ok(), ok(), ok(), vec![failed_stream(t0, 400)]],
            10.0,
            None,
            StreamCountSource::Default,
        );
        assert!((r5.failed_stream_ratio - 0.20).abs() < 1e-9);
        assert!(
            !r5.summary_line().contains("of streams failed"),
            "exactly 20% is not flagged: {}",
            r5.summary_line()
        );
        // 4 requested, 1 failed → 0.25 → flagged.
        let r4 = FlatOutResult::from_events(
            &[ok(), ok(), ok(), vec![failed_stream(t0, 400)]],
            10.0,
            None,
            StreamCountSource::Default,
        );
        assert!((r4.failed_stream_ratio - 0.25).abs() < 1e-9);
        assert!(
            r4.summary_line().contains("⚠ 25% of streams failed"),
            "{}",
            r4.summary_line()
        );
    }

    #[test]
    fn cf04_to_dict_carries_the_exact_contract_names() {
        // The JSON export carries the dual denominators under the exact
        // contract field names (shared with Engine B), and the
        // back-compat `per_stream_tps` is the requested view.
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let s0 = vec![content_frame_text("abcd", t0, ts)];
        let s1 = vec![failed_stream(t0, 400)];
        let r = FlatOutResult::from_events(&[s0, s1], 10.0, None, StreamCountSource::Default);
        let v = r.to_dict();
        assert!(
            (v["tps_per_completed_stream"].as_f64().unwrap() - 0.1).abs() < 1e-9,
            "÷ 1 completed stream: {}",
            v["tps_per_completed_stream"]
        );
        assert!(
            (v["tps_per_requested_stream"].as_f64().unwrap() - 0.05).abs() < 1e-9,
            "÷ 2 requested streams: {}",
            v["tps_per_requested_stream"]
        );
        assert!(
            (v["failed_stream_ratio"].as_f64().unwrap() - 0.5).abs() < 1e-9,
            "1 of 2 failed: {}",
            v["failed_stream_ratio"]
        );
        assert_eq!(
            v["per_stream_tps"].as_f64().unwrap(),
            v["tps_per_requested_stream"].as_f64().unwrap(),
            "the back-compat field is the requested view"
        );
    }

    // ── CF-29: respect the model's max context ──────────────────────────

    #[test]
    fn cf29_parse_vllm_max_model_len() {
        // vLLM exposes `max_model_len` on its `/v1/models` entries.
        let body = r#"{
            "object": "list",
            "data": [
                {"id": "qwen3.8-27b", "object": "model", "owned_by": "vllm", "max_model_len": 851968},
                {"id": "other-model", "object": "model"}
            ]
        }"#;
        assert_eq!(parse_model_max_context(body, "qwen3.8-27b"), Some(851968));
        // A model with no max-context field → None (no clamp).
        assert_eq!(parse_model_max_context(body, "other-model"), None);
        // A model not in the list → None.
        assert_eq!(parse_model_max_context(body, "missing"), None);
    }

    #[test]
    fn cf29_parse_tolerates_missing_and_string_fields() {
        // String-valued limits (some backends quote the number).
        let body = r#"{"data": [{"id": "m", "max_context": "4096"}]}"#;
        assert_eq!(parse_model_max_context(body, "m"), Some(4096));
        // OpenAI-style `context_length`.
        let body = r#"{"data": [{"id": "m", "context_length": 32768}]}"#;
        assert_eq!(parse_model_max_context(body, "m"), Some(32768));
        // No recognizable field → None.
        let body = r#"{"data": [{"id": "m", "object": "model"}]}"#;
        assert_eq!(parse_model_max_context(body, "m"), None);
        // Not JSON at all → None (graceful, never a panic).
        assert_eq!(parse_model_max_context("this is not json", "m"), None);
        // Empty `data` → None.
        assert_eq!(
            parse_model_max_context(r#"{"object":"list","data":[]}"#, "m"),
            None
        );
    }

    #[test]
    fn cf29_clamp_subtracts_prompt_headroom() {
        // 1,000-token context, 20-token prompt → 980 requested.
        assert_eq!(clamp_max_tokens(Some(1_000), 20), 980);
        // A context exactly at MAX_TOKENS still reserves the prompt.
        assert_eq!(clamp_max_tokens(Some(100_000), 20), 99_980);
        // No probe (unknown backend) → the unclamped cap.
        assert_eq!(clamp_max_tokens(None, 20), MAX_TOKENS);
        // A huge context → the cap is unchanged.
        assert_eq!(clamp_max_tokens(Some(851_968), 20), MAX_TOKENS);
        // A tiny context → floored at 1 (always a legal request).
        assert_eq!(clamp_max_tokens(Some(10), 20), 1);
    }

    #[test]
    fn cf29_uniform_400_is_a_config_error() {
        // Every stream rejected with HTTP 400, no tokens: the
        // "requested tokens exceed model limit" configuration error.
        let t0 = MonotonicInstant::now();
        let per_stream = vec![
            vec![failed_stream(t0, 400)],
            vec![failed_stream(t0, 400)],
            vec![failed_stream(t0, 400)],
        ];
        let err = uniform_400_config_error(&per_stream, 980, Some(1_000));
        assert!(err.is_some(), "uniform 400 must be detected");
        let msg = err.unwrap();
        assert!(msg.contains("requested tokens exceed model limit"), "{msg}");
        assert!(msg.contains("980"), "names the cap that was sent: {msg}");
        assert!(msg.contains("1000"), "names the probed context: {msg}");
    }

    #[test]
    fn cf29_partial_failure_is_not_a_config_error() {
        // Two streams 400, one produced tokens: that is dilution (the
        // CF-04 failed_stream_ratio flags it), not a uniform config
        // rejection.
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let per_stream = vec![
            vec![failed_stream(t0, 400)],
            vec![failed_stream(t0, 400)],
            vec![content_frame_text("abcd", t0, ts)],
        ];
        assert!(uniform_400_config_error(&per_stream, 980, Some(1_000)).is_none());
    }

    #[test]
    fn cf29_non400_failure_is_not_a_config_error() {
        // All streams failed, but with connection errors (not 400): a
        // server problem, not a request-size configuration error.
        let t0 = MonotonicInstant::now();
        let per_stream = vec![
            vec![StreamEvent::Failed {
                timestamps: ts_at(t0),
                error: crate::client::StreamError::Connection("refused".into()),
                looping: false,
                loop_excluded_tokens: 0,
            }],
            vec![StreamEvent::Failed {
                timestamps: ts_at(t0),
                error: crate::client::StreamError::Connection("refused".into()),
                looping: false,
                loop_excluded_tokens: 0,
            }],
        ];
        assert!(uniform_400_config_error(&per_stream, MAX_TOKENS, None).is_none());
    }

    #[test]
    fn cf29_400_after_tokens_is_not_a_config_error() {
        // A stream that produced tokens before its 400 terminal has
        // evidence the request size was accepted — not uniform.
        let t0 = MonotonicInstant::now();
        let ts = ts_at(t0);
        let per_stream = vec![
            vec![content_frame_text("abcd", t0, ts), failed_stream(t0, 400)],
            vec![failed_stream(t0, 400)],
        ];
        assert!(uniform_400_config_error(&per_stream, 980, Some(1_000)).is_none());
    }

    #[test]
    fn cf29_config_error_surfaces_in_summary_and_json() {
        // A result carrying a config error leads with it in the
        // one-line summary (never a bare "0.0 t/s") and in the JSON.
        let r = FlatOutResult {
            stream_count: 3,
            stream_count_source: StreamCountSource::Default,
            total_tokens: 0,
            duration_secs: 1.2,
            aggregate_tps: 0.0,
            per_stream_tps: 0.0,
            tps_per_completed_stream: 0.0,
            tps_per_requested_stream: 0.0,
            failed_stream_ratio: 1.0,
            config_error: Some(
                "requested tokens exceed model limit: the server rejected every stream with a uniform HTTP 400 (request too large)"
                    .into(),
            ),
            max_tokens_used: 980,
            ttft_avg_ms: 0.0,
            itl_p50_ms: 0.0,
            itl_p99_ms: 0.0,
            tokens_estimated: true,
            counting_method: "none".into(),
        };
        assert!(
            r.summary_line().starts_with("⚠ CONFIG:"),
            "{}",
            r.summary_line()
        );
        assert!(
            r.summary_line()
                .contains("requested tokens exceed model limit"),
            "{}",
            r.summary_line()
        );
        let v = r.to_dict();
        assert!(v["config_error"].is_string());
        assert_eq!(v["max_tokens_used"], 980);
    }

    // ── CF-29: mock model server (loopback only — no live server) ───────

    /// A minimal one-shot HTTP server on 127.0.0.1 (ephemeral port):
    /// serves `GET …/v1/models` with `models_status` + `models_body` and
    /// answers every `POST …/chat/completions` with `chat_status` +
    /// `chat_body`. Records the `max_tokens` it saw in each chat body
    /// so a test can assert the clamp reached the wire.
    ///
    /// It accepts at most `max_connections` connections and then closes
    /// (the listener drops with the task at runtime teardown). No
    /// external network is touched.
    async fn spawn_mock_model_server(
        models_status: u16,
        models_body: &str,
        chat_status: u16,
        chat_body: &str,
        max_connections: usize,
    ) -> (String, std::sync::Arc<tokio::sync::Mutex<Vec<u32>>>) {
        use tokio::io::{AsyncReadExt, AsyncWriteExt};

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let seen: std::sync::Arc<tokio::sync::Mutex<Vec<u32>>> =
            std::sync::Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let models_body = models_body.to_string();
        let chat_body = chat_body.to_string();
        let seen_cl = seen.clone();
        tokio::spawn(async move {
            for _ in 0..max_connections {
                let (mut socket, _) = match listener.accept().await {
                    Ok(c) => c,
                    Err(_) => break,
                };
                // Read the full request (headers + Content-Length body).
                let mut buf: Vec<u8> = Vec::new();
                let mut tmp = [0u8; 8192];
                while !buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    match socket.read(&mut tmp).await {
                        Ok(0) => break,
                        Ok(k) => buf.extend_from_slice(&tmp[..k]),
                        Err(_) => break,
                    }
                }
                let header_end = buf
                    .windows(4)
                    .position(|w| w == b"\r\n\r\n")
                    .unwrap_or(buf.len());
                let head = String::from_utf8_lossy(&buf[..header_end]).to_string();
                let content_length = head
                    .lines()
                    .filter_map(|l| {
                        l.to_lowercase()
                            .strip_prefix("content-length:")
                            .and_then(|v| v.trim().parse::<usize>().ok())
                    })
                    .next()
                    .unwrap_or(0);
                // The body starts *after* the blank line (header_end + 4).
                let body_start = header_end + 4;
                while buf.len() < body_start + content_length {
                    match socket.read(&mut tmp).await {
                        Ok(0) => break,
                        Ok(k) => buf.extend_from_slice(&tmp[..k]),
                        Err(_) => break,
                    }
                }
                let body_len = (buf.len() - body_start).min(content_length);
                let body =
                    String::from_utf8_lossy(&buf[body_start..body_start + body_len]).to_string();
                let first_line = head.lines().next().unwrap_or_default().to_string();
                let (status, resp_body) = if first_line.contains("/models") {
                    (models_status, models_body.clone())
                } else {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&body) {
                        if let Some(mt) = v.get("max_tokens").and_then(|m| m.as_u64()) {
                            seen_cl.lock().await.push(mt as u32);
                        }
                    }
                    (chat_status, chat_body.clone())
                };
                let reason = if status == 200 { "OK" } else { "Bad Request" };
                let resp = format!(
                    "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                    resp_body.len()
                );
                let _ = socket.write_all(resp.as_bytes()).await;
                let _ = socket.shutdown().await;
            }
        });
        (format!("http://{addr}"), seen)
    }

    #[tokio::test]
    async fn cf29_mock_small_context_clamps_and_uniform_400_is_a_config_error() {
        // A mock model with a 1,000-token max context (vLLM-style
        // `max_model_len`) whose true generation cap is lower: it
        // rejects every chat request with HTTP 400. The run must
        // (1) clamp `max_tokens` to the probed context (minus the
        // prompt) before sending, and (2) surface the uniform 400 as a
        // configuration error — not a silent 0 t/s.
        let (url, seen) = spawn_mock_model_server(
            200,
            r#"{"object":"list","data":[{"id":"m","object":"model","max_model_len":1000}]}"#,
            400,
            r#"{"error":"max_tokens exceeds the model limit"}"#,
            3, // 1 probe + 2 chat streams
        )
        .await;
        let engine = FlatOutEngine::new(&Config {
            url,
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap()
        .stream_count(2, StreamCountSource::Default);
        let r = engine.run().await;
        // The clamp: 1,000 context − the ~20-token prompt (chars/4).
        let prompt_tokens = count_tokens(MINIMAL_PROMPT, None).tokens;
        let expected = clamp_max_tokens(Some(1_000), prompt_tokens as u64);
        assert!(expected < MAX_TOKENS, "the probe must reduce the cap");
        assert_eq!(
            r.max_tokens_used, expected,
            "max_tokens clamped to the probed model max context"
        );
        // The clamp reached the wire on every request.
        let seen = seen.lock().await;
        assert!(
            !seen.is_empty() && seen.iter().all(|&mt| mt == expected),
            "every chat request carried the clamped cap: {seen:?}"
        );
        // The uniform 400 is a configuration error, not 0 t/s.
        assert!(r.config_error.is_some(), "uniform 400 → config error");
        assert!(r
            .config_error
            .as_ref()
            .unwrap()
            .contains("requested tokens exceed model limit"));
        assert!(
            r.summary_line().starts_with("⚠ CONFIG"),
            "the summary leads with the config error, not a bare 0 t/s: {}",
            r.summary_line()
        );
        assert!(r.to_dict()["config_error"].is_string());
    }

    #[tokio::test]
    async fn cf29_mock_without_context_field_runs_unclamped() {
        // A backend whose `/v1/models` entries carry no max-context field
        // (llama.cpp-style): the probe finds nothing, the run proceeds
        // with the unclamped cap — and a uniform 400 is *still* surfaced
        // as a configuration error (the server's cap is simply
        // unprobed).
        let (url, seen) = spawn_mock_model_server(
            200,
            r#"{"object":"list","data":[{"id":"m","object":"model","owned_by":"llama.cpp"}]}"#,
            400,
            r#"{"error":"too large"}"#,
            3,
        )
        .await;
        let engine = FlatOutEngine::new(&Config {
            url,
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap()
        .stream_count(2, StreamCountSource::Default);
        let r = engine.run().await;
        assert_eq!(
            r.max_tokens_used, MAX_TOKENS,
            "no context field → no clamp (graceful degradation)"
        );
        let seen = seen.lock().await;
        assert!(seen.iter().all(|&mt| mt == MAX_TOKENS), "{seen:?}");
        assert!(
            r.config_error.is_some(),
            "the uniform 400 is still a config error"
        );
        assert!(r
            .config_error
            .as_ref()
            .unwrap()
            .contains("could not be probed"));
    }

    #[tokio::test]
    async fn cf29_models_endpoint_404_degrades_gracefully() {
        // No `/v1/models` route at all (404): the probe degrades to
        // `None`, the run proceeds unclamped, and a uniform 400 on the
        // chat requests is still surfaced as a configuration error.
        let (url, _seen) = spawn_mock_model_server(
            404,
            r#"{"error":"no route"}"#,
            400,
            r#"{"error":"too large"}"#,
            3,
        )
        .await;
        let engine = FlatOutEngine::new(&Config {
            url,
            model: "m".to_string(),
            ..Config::default()
        })
        .unwrap()
        .stream_count(2, StreamCountSource::Default);
        let r = engine.run().await;
        assert_eq!(r.max_tokens_used, MAX_TOKENS, "404 probe → unclamped");
        assert!(r.config_error.is_some(), "uniform 400 still surfaced");
    }
}
