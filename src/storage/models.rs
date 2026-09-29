//! `serde` row structs matching `schema.sql`
//! (`benchmark_sessions`, `stream_metrics`, `needle_evaluations`).
//!
//! Field nullability mirrors the DDL exactly: `NOT NULL` columns are plain
//! fields, nullable columns are `Option<T>` (stored as / read from SQL
//! `NULL`). Every struct round-trips through `serde_json` for the export
//! path (Chunk 13) and the history diff (Chunk 14).

use serde::{Deserialize, Serialize};

use crate::engines::speed::SpeedResult;

/// One `benchmark_sessions` row: a completed benchmark run.
///
/// `timestamp` is `None` on insert → SQLite's `DEFAULT CURRENT_TIMESTAMP`
/// fills it (UTC `YYYY-MM-DD HH:MM:SS`).
#[derive(Debug, Clone, Serialize, Deserialize)]
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
    /// inter-token time, `1000 / tg_speed`. `reasoning_tokens`,
    /// `joules_per_token`, and `cache_hit` stay `NULL` until the
    /// capability/hardware engines (Chunks 15/17) and a cold TTFT
    /// baseline produce them.
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<f64>,
}
