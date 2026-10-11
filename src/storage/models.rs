//! `serde` row structs matching `schema.sql`
//! (`benchmark_sessions`, `stream_metrics`, `needle_evaluations`).
//!
//! Field nullability mirrors the DDL exactly: `NOT NULL` columns are plain
//! fields, nullable columns are `Option<T>` (stored as / read from SQL
//! `NULL`). Every struct round-trips through `serde_json` for the export
//! path (Chunk 13) and the history diff (Chunk 14).

use serde::{Deserialize, Serialize};

use crate::engines::capability::{ReasoningResult, StructuredResult};
use crate::engines::concurrency::SweepResult;
use crate::engines::flatout::FlatOutResult;
use crate::engines::hardware::EnergyResult;
use crate::engines::speed::SpeedResult;

/// One `benchmark_sessions` row: a completed benchmark run.
///
/// `timestamp` is `None` on insert → SQLite's `DEFAULT CURRENT_TIMESTAMP`
/// fills it (UTC `YYYY-MM-DD HH:MM:SS`).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkSession {
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timestamp: Option<String>,
    pub target_url: String,
    pub model_name: String,
    /// `'vllm'` | `'llamacpp'` | `'sglang'` | `'ollama'` (when known).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub backend_type: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quantization: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub system_gpu: Option<String>,
    /// Wall-clock duration of the whole run (all iterations).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub total_duration_sec: Option<f64>,
    /// R2 (traceability): the engines this run selected, as a JSON name
    /// list (e.g. `["speed","niah"]`). `None` for pre-migration rows.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engines: Option<String>,
    /// R2: the headline results of the non-stream engines (B/C2/C3/D/F),
    /// serialized as a JSON [`EngineResults`] object. `None` when no such
    /// engine ran (or for pre-migration rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_results: Option<String>,
}

/// One `stream_metrics` row: the per-stream §7 metrics for a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamMetricRow {
    /// `AUTOINCREMENT`; `None` on insert, filled by SQLite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metric_id: Option<i64>,
    pub session_id: String,
    /// Concurrency level of the sweep step this stream belongs to
    /// (the single-stream headless engine records `1`).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concurrency_level: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub completion_tokens: Option<i64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reasoning_tokens: Option<i64>,
    /// Time-to-first-token, milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<f64>,
    /// Mean time-per-output-token, milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tpot_ms: Option<f64>,
    /// MTP η (tokens per content chunk).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mtp_efficiency: Option<f64>,
    /// Silicon efficiency (Chunk 17); `NULL` without GPU telemetry.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub joules_per_token: Option<f64>,
    /// Prefix-cache heuristic result (blueprint §7.5); `NULL` when no
    /// cold baseline is available to compare against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_hit: Option<bool>,
}

impl StreamMetricRow {
    /// One `stream_metrics` row from a headless speed-test result
    /// (Chunk 7).
    ///
    /// `tpot_ms` is derived from the decode speed (TG): the mean
    /// inter-token time, `1000 / tg_speed`. `reasoning_tokens` and
    /// `joules_per_token` stay `NULL` until the capability/hardware
    /// engines (Chunks 15/17) produce them. `cache_hit` (CF-26) also
    /// stays `NULL` here: the Engine A `SpeedResult` does not yet carry
    /// the server's `cached_tokens` detail (the CF-02 parser captures it,
    /// but `client/stream.rs` — a P2-A seam — does not propagate it into
    /// the outcome). Use [`with_cache_hit`](Self::with_cache_hit) to set it
    /// once the value is available.
    #[must_use]
    pub fn from_speed_result(
        result: &SpeedResult,
        session_id: &str,
        concurrency_level: i64,
    ) -> Self {
        let tpot_ms = if result.tg_speed > 0.0 {
            Some(1000.0 / result.tg_speed)
        } else {
            None
        };
        Self {
            metric_id: None,
            session_id: session_id.to_string(),
            concurrency_level: Some(concurrency_level),
            prompt_tokens: Some(result.prompt_tokens as i64),
            completion_tokens: Some(result.completion_tokens as i64),
            reasoning_tokens: None,
            ttft_ms: Some(result.ttft * 1000.0),
            tpot_ms,
            mtp_efficiency: Some(result.mtp_efficiency),
            joules_per_token: None,
            cache_hit: None,
        }
    }

    /// Set the `reasoning_tokens` column (CF-02).
    ///
    /// [`from_speed_result`](Self::from_speed_result) leaves this `NULL`
    /// because the Engine A `SpeedResult` does not yet carry the server's
    /// `reasoning_tokens` detail. When a caller *does* have the value (the
    /// `UsageDetails` the SSE parser now reads, or a future `SpeedResult`
    /// field), this builder populates the column so a run can be flagged
    /// reasoning-dominated and thinking-suppression can be verified
    /// (`reasoning_tokens ≈ 0` when disabled).
    #[must_use]
    pub fn with_reasoning_tokens(mut self, tokens: u64) -> Self {
        self.reasoning_tokens = Some(tokens as i64);
        self
    }

    /// Set the `cache_hit` column (CF-26): `true` when the server reports
    /// `UsageDetails.cached_tokens > 0` (a warm KV / prefix-cache hit).
    ///
    /// [`from_speed_result`](Self::from_speed_result) leaves this `NULL`
    /// because the Engine A `SpeedResult` does not yet carry the server's
    /// `cached_tokens` detail (the CF-02 parser captures it, but
    /// `client/stream.rs` — a P2-A seam — does not propagate it into the
    /// outcome). When a caller *does* have the value, this builder
    /// populates the column.
    #[must_use]
    pub fn with_cache_hit(mut self, hit: bool) -> Self {
        self.cache_hit = Some(hit);
        self
    }
}

/// One `needle_evaluations` row: a single NIAH size×depth cell (Chunk 15).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeedleEvaluation {
    /// `AUTOINCREMENT`; `None` on insert, filled by SQLite.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub eval_id: Option<i64>,
    pub session_id: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context_length: Option<i64>,
    /// 0.0..=100.0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub depth_percent: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieved_successfully: Option<bool>,
    /// **Option C (hybrid reasoning):** `true` when this cell's stream
    /// completed OK but the model spent its whole generation budget on
    /// chain-of-thought (all-CoT / zero-content) — an *inconclusive (budget)*
    /// outcome, distinct from a retrieval failure. `None` for rows written
    /// before the column existed (pre-Option-C databases / exports), so old
    /// data reads back cleanly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub truncated_by_reasoning: Option<bool>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}

// ── R2: the `engine_results` column (headline numbers for B/C2/C3/D/F) ──
//
// A/C1 already persist to their own tables (`stream_metrics` /
// `needle_evaluations`). The remaining engines have no dedicated table, so
// their *headline* numbers are captured here as a single JSON object on the
// session row. These are deliberately lightweight summaries (not the full
// engine result types, which carry non-`Serialize` nested fields) — just
// enough to make a stored run traceable: which engines ran, and their key
// result.

/// The `engine_results` JSON object for one session: the headline summary
/// of each non-stream engine that ran (`None` for the ones that didn't).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EngineResults {
    /// Engine B — the concurrency sweep.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub b_concurrency: Option<ConcurrencySummary>,
    /// Engine C2 — deterministic reasoning.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub c2_reasoning: Option<ReasoningSummary>,
    /// Engine C3 — structured output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub c3_structured: Option<StructuredSummary>,
    /// Engine D — hardware / energy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub d_energy: Option<EnergySummary>,
    /// Engine F — Flat Out.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub f_flatout: Option<FlatOutSummary>,
}

/// The `engine_results` column as a JSON string (`None` when no
/// non-stream engine ran).
impl EngineResults {
    /// Serialize to the JSON stored in `benchmark_sessions.engine_results`.
    /// `None` when every field is empty (nothing to store).
    pub fn to_json(&self) -> Option<String> {
        if self.b_concurrency.is_none()
            && self.c2_reasoning.is_none()
            && self.c3_structured.is_none()
            && self.d_energy.is_none()
            && self.f_flatout.is_none()
        {
            return None;
        }
        serde_json::to_string(self).ok()
    }
}

/// Engine B headline: the recommended sweet spot + how many levels ran.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConcurrencySummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sweet_spot: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub levels: Option<usize>,
}

impl ConcurrencySummary {
    #[must_use]
    pub fn from_sweep(r: &SweepResult) -> Self {
        Self {
            sweet_spot: Some(r.sweet_spot),
            levels: Some(r.levels.len()),
        }
    }
}

/// Engine C2 headline: the pass/fail score + mean decode speed.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReasoningSummary {
    pub total: usize,
    pub solved: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_tg_speed: Option<f64>,
}

impl ReasoningSummary {
    #[must_use]
    pub fn from_reasoning(r: &ReasoningResult) -> Self {
        Self {
            total: r.score.total,
            solved: r.score.solved,
            avg_tg_speed: Some(r.avg_tg_speed()),
        }
    }
}

/// Engine C3 headline: the free-vs-constrained penalty + compliance.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructuredSummary {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub constrained_tps: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub penalty_pct: Option<f64>,
    pub compliant: usize,
    pub partial: usize,
    pub failed: usize,
    /// CF-36: `true` only when the constrained case-0 run produced valid
    /// JSON — the `penalty_pct` above is meaningful (and rendered) only
    /// in that case; otherwise the constraint was not effective.
    #[serde(default)]
    pub constraint_effective: bool,
    /// CF-51: the mean per-case stability rate ([0.0, 1.0]) — the fraction
    /// of repeated runs that were fully compliant (the McNemar/Wilson
    /// A/B-comparator data prerequisite).
    #[serde(default)]
    pub stability_rate: f64,
}

impl StructuredSummary {
    #[must_use]
    pub fn from_structured(r: &StructuredResult) -> Self {
        let (compliant, partial, failed) = r.score();
        Self {
            free_tps: Some(r.free_tps),
            constrained_tps: Some(r.constrained_tps),
            penalty_pct: Some(r.penalty_pct),
            compliant,
            partial,
            failed,
            constraint_effective: r.constraint_effective(),
            stability_rate: r.stability_rate(),
        }
    }
}

/// Engine D headline: the silicon-efficiency energy profile.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergySummary {
    pub joules: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub joules_per_token: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_power_w: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_power_w: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub avg_utilization_pct: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vendor: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fragmentation_warning: Option<String>,
    /// The clear N/A reason when no power telemetry was available (the
    /// module N/A rule — a driverless host or a sensor with no usable
    /// reading, e.g. the RAPL quirk on a shared host). `Some("N/A — no
    /// power telemetry")` so the persisted/exported energy object is an
    /// *honest, labeled* N/A, never a silent gap and never a spurious
    /// `0.0` J/token. `None` when the profile measured power.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub na_reason: Option<String>,
}

impl EnergySummary {
    #[must_use]
    pub fn from_energy(e: &EnergyResult) -> Self {
        Self {
            joules: e.joules,
            joules_per_token: e.joules_per_token,
            avg_power_w: e.avg_power_w,
            peak_power_w: e.peak_power_w,
            avg_utilization_pct: e.avg_utilization_pct,
            vendor: e.vendor.clone(),
            model: e.model.clone(),
            fragmentation_warning: e.fragmentation_warning.clone(),
            na_reason: e.is_na().then(|| "N/A — no power telemetry".to_string()),
        }
    }
}

/// Engine F headline: the real-world max-throughput window.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlatOutSummary {
    pub stream_count: usize,
    pub total_tokens: u64,
    pub duration_secs: f64,
    pub aggregate_tps: f64,
    /// Back-compat alias of [`tps_per_requested_stream`](Self::tps_per_requested_stream).
    pub per_stream_tps: f64,
    /// CF-04 dual per-stream denominators, now carried in the persisted /
    /// exported summary (Wave 6a gap: they were in the `--json` stdout
    /// `flat_out` but omitted here, so the DB-built export lost them).
    /// **Capacity view** — `aggregate_tps ÷ completed streams` (failed /
    /// zero-token streams excluded). `#[serde(default)]` keeps pre-existing
    /// stored `engine_results` (without the field) deserializing.
    #[serde(default)]
    pub tps_per_completed_stream: f64,
    /// **User-experience view** — `aggregate_tps ÷ requested streams`
    /// (failures included; equals [`per_stream_tps`](Self::per_stream_tps)).
    #[serde(default)]
    pub tps_per_requested_stream: f64,
    /// (CF-04) `failed_streams ÷ requested_streams` — flagged in the run
    /// log when above the 0.20 threshold.
    #[serde(default)]
    pub failed_stream_ratio: f64,
}

impl FlatOutSummary {
    #[must_use]
    pub fn from_flatout(r: &FlatOutResult) -> Self {
        Self {
            stream_count: r.stream_count,
            total_tokens: r.total_tokens,
            duration_secs: r.duration_secs,
            aggregate_tps: r.aggregate_tps,
            per_stream_tps: r.per_stream_tps,
            tps_per_completed_stream: r.tps_per_completed_stream,
            tps_per_requested_stream: r.tps_per_requested_stream,
            failed_stream_ratio: r.failed_stream_ratio,
        }
    }
}
