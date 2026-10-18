//! `serde` row structs matching `schema.sql`
//! (`benchmark_sessions`, `stream_metrics`, `needle_evaluations`).
//!
//! Field nullability mirrors the DDL exactly: `NOT NULL` columns are plain
//! fields, nullable columns are `Option<T>` (stored as / read from SQL
//! `NULL`). Every struct round-trips through `serde_json` for the export
//! path (Chunk 13) and the history diff (Chunk 14).

use serde::{Deserialize, Serialize};

use crate::engines::capability::{CaseVerdict, ReasoningResult, StructuredResult};
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
    /// **R2 hybrid reasoning:** the number of cases
    /// `truncated_by_reasoning` (all-CoT / zero-content completions) — the
    /// distinct "inconclusive (budget)" bucket, reported *separately* from
    /// `failed` (a budget exhaustion is not a capability failure).
    /// `#[serde(default)]` keeps pre-existing stored `engine_results`
    /// documents deserializing.
    #[serde(default)]
    pub truncated: usize,
    /// **R2:** the per-case detail (verdict + the `truncated_by_reasoning`
    /// flag + the valid-JSON signal) so the DB headline is self-contained.
    /// `#[serde(default)]` keeps pre-existing documents deserializing.
    #[serde(default)]
    pub cases: Vec<StructuredCaseSummary>,
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

/// One C3 case in the persisted `engine_results` JSON column (R2): the
/// folded verdict + the distinct `truncated_by_reasoning` flag, so a
/// truncated case is reported separately from a compliance failure.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StructuredCaseSummary {
    /// The case name (`Simple` / `Medium` / `Complex`).
    pub name: String,
    /// The folded per-case verdict (`compliant` / `partial` / `failed` /
    /// `truncated`).
    #[serde(default)]
    pub verdict: CaseVerdict,
    /// `true` when a repeat's stream completed OK but committed no content
    /// (budget exhausted in chain-of-thought) — the distinct
    /// `truncated_by_reasoning` outcome.
    #[serde(default)]
    pub truncated_by_reasoning: bool,
    /// The CF-17 valid-JSON signal (a truncated case is `false` — there is
    /// no content to parse).
    #[serde(default)]
    pub valid_json: bool,
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
            truncated: r.truncated_count(),
            cases: r
                .cases
                .iter()
                .map(|c| StructuredCaseSummary {
                    name: c.name.clone(),
                    verdict: c.verdict,
                    truncated_by_reasoning: c.truncated_by_reasoning,
                    valid_json: c.is_valid_json(),
                })
                .collect(),
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
    /// The clear N/A reason when `joules_per_token` is `None` (the module
    /// N/A rule) — **labeled**, never a silent gap and never a spurious
    /// `0.0` J/token. B3 distinguishes the two root causes:
    /// `Some("N/A — no power telemetry")` (a driverless host, or a sensor
    /// with no usable reading, e.g. the RAPL quirk on a shared host) vs
    /// `Some("N/A — no tokens generated")` (power telemetry present, but the
    /// run produced zero tokens to divide by). `None` when a J/token value
    /// was measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub na_reason: Option<String>,
    /// I4 / CF-53: `true` when the CPU component of the system draw is (or
    /// included) the labeled fixed-estimate fallback — including a
    /// measured backend that fell back mid-run (the sample-level latch in
    /// `CpuPowerBackend::is_estimated`). The DB / export can now tell
    /// measured from estimated (previously the flag existed only in the
    /// `--json` stdout object, never persisted).
    #[serde(default, skip_serializing_if = "is_false")]
    pub cpu_power_estimated: bool,
}

/// `skip_serializing_if` predicate for a `false`-is-absent bool field.
fn is_false(b: &bool) -> bool {
    !*b
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
            // B3: the labeled N/A reason (distinguishing "no power
            // telemetry" from "no tokens generated"), or `None` when a
            // J/token value was measured.
            na_reason: e.na_reason().map(|r| format!("N/A — {r}")),
            // I4: the estimate flag is copied from the result (the
            // production callers set it from `CpuPowerBackend::is_estimated`
            // — including the mid-run sample-level fallback).
            cpu_power_estimated: e.cpu_power_estimated,
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::capability::{
        evaluate_case, CaseCheck, CountingMethod, StructuredCaseResult,
    };

    /// A C3 result with one compliant case, one partial case, and one
    /// **truncated** case (the R2 `truncated_by_reasoning` bucket).
    fn result_with_truncated() -> StructuredResult {
        StructuredResult {
            free_tps: 65.2,
            constrained_tps: 61.8,
            penalty_pct: 5.2,
            free_ttft: 0.2,
            constrained_ttft: 0.21,
            cases: vec![
                evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
                evaluate_case("Medium", r#"[{"id":1,"label":"a","active":true}]"#),
                StructuredCaseResult {
                    name: "Complex".to_string(),
                    checks: vec![CaseCheck {
                        label: "Valid JSON".to_string(),
                        passed: false,
                    }],
                    verdict: CaseVerdict::Truncated,
                    output: String::new(),
                    repeats: 1,
                    compliant_repeats: 0,
                    truncated_by_reasoning: true,
                    reasoning_tokens: None,
                },
            ],
            constrained_body: String::new(),
            free_body: "prose".to_string(),
            counting_method: CountingMethod::ServerUsage,
        }
    }

    #[test]
    fn structured_summary_carries_the_truncated_bucket_and_per_case_flags() {
        // R2: the persisted C3 headline reports `truncated` separately from
        // `failed`, and the per-case detail carries the flag.
        let s = StructuredSummary::from_structured(&result_with_truncated());
        assert_eq!(s.compliant, 1);
        assert_eq!(s.partial, 1);
        assert_eq!(s.failed, 0, "a truncated case is not a compliance failure");
        assert_eq!(s.truncated, 1);
        assert_eq!(s.cases.len(), 3);
        assert_eq!(s.cases[2].name, "Complex");
        assert_eq!(s.cases[2].verdict, CaseVerdict::Truncated);
        assert!(s.cases[2].truncated_by_reasoning);
        assert!(!s.cases[2].valid_json, "no content ⇒ not valid JSON");
        assert!(!s.cases[0].truncated_by_reasoning);
        assert!(s.cases[0].valid_json);
    }

    #[test]
    fn legacy_structured_summary_document_still_deserializes() {
        // `#[serde(default)]` on the R2 fields: a pre-R2 `engine_results`
        // document (no `truncated`, no `cases`) parses with honest zeros.
        let legacy = r#"{
            "free_tps": 65.2,
            "constrained_tps": 61.8,
            "penalty_pct": 5.2,
            "compliant": 3,
            "partial": 0,
            "failed": 0,
            "constraint_effective": true,
            "stability_rate": 1.0
        }"#;
        let s: StructuredSummary = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.compliant, 3);
        assert_eq!(s.truncated, 0);
        assert!(s.cases.is_empty());
    }

    #[test]
    fn structured_summary_round_trips_through_json() {
        let s = StructuredSummary::from_structured(&result_with_truncated());
        let json = serde_json::to_string(&s).unwrap();
        let back: StructuredSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back.truncated, 1);
        assert_eq!(back.cases[2].verdict, CaseVerdict::Truncated);
        assert!(back.cases[2].truncated_by_reasoning);
    }

    /// B3: `EnergySummary::from_energy` sets the labeled `na_reason` for
    /// **both** N/A root causes (and none when a value is measured) — so the
    /// persisted / `--json` energy object is never a silent gap.
    #[test]
    fn energy_summary_na_reason_labels_both_causes() {
        // No power telemetry (driverless host).
        let no_power = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: None,
            peak_power_w: None,
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: None,
            vendor: None,
            model: None,
            cpu_power_estimated: false,
        };
        let s = EnergySummary::from_energy(&no_power);
        assert_eq!(
            s.na_reason.as_deref(),
            Some("N/A — no power telemetry"),
            "the original N/A cause"
        );

        // Power telemetry present, but zero tokens generated.
        let no_tokens = EnergyResult {
            joules: 100.0,
            joules_per_token: None,
            avg_power_w: Some(200.0),
            peak_power_w: Some(300.0),
            avg_utilization_pct: Some(90.0),
            ..no_power
        };
        let s2 = EnergySummary::from_energy(&no_tokens);
        assert_eq!(
            s2.na_reason.as_deref(),
            Some("N/A — no tokens generated"),
            "the distinct zero-tokens cause, not a mislabel"
        );

        // A measured value → no reason at all.
        let with_value = EnergyResult {
            joules_per_token: Some(0.5),
            ..no_tokens
        };
        let s3 = EnergySummary::from_energy(&with_value);
        assert_eq!(s3.na_reason, None, "a value is not N/A");
        assert_eq!(s3.joules_per_token, Some(0.5));
    }

    /// I7a: `EnergySummary.joules` is a non-`Option` `f64` — the `0.0` N/A
    /// value must **always** carry `na_reason` (never a silent `0.0` in the
    /// DB / export).
    #[test]
    fn energy_summary_zero_joules_always_carries_na_reason() {
        // No telemetry: joules 0.0 + the no-telemetry reason.
        let no_power = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: None,
            peak_power_w: None,
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: None,
            vendor: None,
            model: None,
            cpu_power_estimated: false,
        };
        let s = EnergySummary::from_energy(&no_power);
        assert_eq!(s.joules, 0.0);
        assert!(
            s.na_reason.is_some(),
            "the 0.0 N/A value carries a reason: {:?}",
            s.na_reason
        );

        // Zero energy above the floor (telemetry + tokens present): the
        // distinct "no energy measured" reason.
        let no_energy = EnergyResult {
            joules: 0.0,
            joules_per_token: None,
            avg_power_w: Some(0.0),
            peak_power_w: Some(80.0),
            avg_utilization_pct: Some(90.0),
            ..no_power
        };
        let s2 = EnergySummary::from_energy(&no_energy);
        assert_eq!(s2.joules, 0.0);
        assert_eq!(
            s2.na_reason.as_deref(),
            Some("N/A — no energy measured"),
            "the 0.0 N/A value is labeled: {:?}",
            s2.na_reason
        );
    }

    /// I4: `from_energy` copies the `cpu_power_estimated` flag into the
    /// persisted summary (previously the flag existed only in the `--json`
    /// stdout object — the DB / export could not tell measured from
    /// estimated).
    #[test]
    fn energy_summary_carries_the_cpu_power_estimated_flag() {
        let base = EnergyResult {
            joules: 100.0,
            joules_per_token: Some(0.5),
            avg_power_w: Some(200.0),
            peak_power_w: Some(300.0),
            peak_vram_bytes: None,
            peak_vram_ratio: None,
            fragmentation_warning: None,
            avg_utilization_pct: Some(90.0),
            vendor: Some("NVIDIA".into()),
            model: Some("Mock".into()),
            cpu_power_estimated: false,
        };
        // Estimated (a mid-run RAPL fallback latched the estimate).
        let est = EnergyResult {
            cpu_power_estimated: true,
            ..base.clone()
        };
        let s = EnergySummary::from_energy(&est);
        assert!(s.cpu_power_estimated, "the flag is copied");
        // Measured → the flag is false and omitted from the JSON.
        let m = EnergySummary::from_energy(&base);
        assert!(!m.cpu_power_estimated);
        let json = serde_json::to_string(&m).unwrap();
        assert!(
            !json.contains("cpu_power_estimated"),
            "false is omitted (not a noisy zero): {json}"
        );
        let json_est = serde_json::to_string(&s).unwrap();
        assert!(json_est.contains("\"cpu_power_estimated\":true"), "{json_est}");
        // A legacy stored document (no field) deserializes as `false`.
        let legacy = r#"{"joules":100.0,"joules_per_token":0.5}"#;
        let back: EnergySummary = serde_json::from_str(legacy).unwrap();
        assert!(!back.cpu_power_estimated, "legacy rows are not estimated");
    }
}
