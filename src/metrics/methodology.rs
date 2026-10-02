//! The v0.1.1 "Methodology Transparency" JSON blocks (blueprint
//! §v0.1.1-A/E).
//!
//! Every JSON export carries two related blocks:
//!
//! * `timing` — *how* the measurements were taken: nanosecond resolution,
//!   the quanta TSC-based monotonic clock, the measured per-timestamp
//!   overhead, and the measurement-isolation guarantee;
//! * `methodology` — *what each number means*: the exact formula behind
//!   every published metric (the three labeled throughput layers, ITL,
//!   aggregate / per-stream, token counting, the loop guard, warmup).
//!
//! These blocks are the trust signal: a reader of the JSON can verify
//! exactly how every number was produced without reading the source.
//!
//! Both are pure `serde_json::Value` builders (no timing path, no locks —
//! the one `measure_timestamp_overhead()` call a consumer makes is a
//! one-shot calibration, not the measurement path).

use serde_json::{json, Value};

/// The `timing` block of a JSON export.
///
/// `overhead_ns` is the measured cost of a single
/// [`crate::timing::MonotonicInstant::now`] call (the caller measures it —
/// 10 000 calls averaged — so the published number is the one from *this*
/// machine).
#[must_use]
pub fn timing_block(overhead_ns: u64) -> Value {
    json!({
        "resolution": "nanosecond",
        "method": "quanta TSC-based monotonic clock, single Instant::now() per token",
        "overhead_ns": overhead_ns,
        "isolation": "measurement thread separate from UI/storage, ArcSwap publish is wait-free"
    })
}

/// The `methodology` block of a JSON export: the package version plus the
/// exact formula behind every published metric.
#[must_use]
pub fn methodology_block(overhead_ns: u64) -> Value {
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "timing": {
            "resolution": "nanosecond",
            "method": "quanta TSC",
            "overhead_ns": overhead_ns
        },
        "prefill_throughput": "prompt_tokens / (T_first_content - T_request_sent)",
        "decode_throughput": "completion_tokens / (T_last_content - T_first_content)",
        "e2e_throughput": "total_tokens / total_wall_time",
        "itl": "(T_last_content - T_first_content) / (output_tokens - 1)",
        "aggregate_throughput": "sum(all_completion_tokens) / wall_time",
        "per_stream_throughput": "aggregate_throughput / active_streams",
        "token_counting": "server usage.completion_tokens (primary); exact-tokenizer re-encoding of the full reasoning+content text (fallback when a proxy omits usage); chars/4 estimate of that text (last resort, flagged estimated) — never the raw SSE frame count, which undercounts servers that batch tokens per frame",
        "loop_guard": "32-token sequence × 3 consecutive repetitions",
        "warmup": "1 warmup request discarded before measurement"
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timing_block_carries_the_published_fields() {
        let v = timing_block(42);
        assert_eq!(v["resolution"], "nanosecond");
        assert_eq!(v["overhead_ns"], 42);
        assert!(v["method"].as_str().unwrap().contains("quanta"));
        assert!(v["isolation"].as_str().unwrap().contains("wait-free"));
    }

    #[test]
    fn methodology_block_documents_every_layer() {
        let v = methodology_block(42);
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["timing"]["overhead_ns"], 42);
        for key in [
            "prefill_throughput",
            "decode_throughput",
            "e2e_throughput",
            "itl",
            "aggregate_throughput",
            "per_stream_throughput",
            "token_counting",
            "loop_guard",
            "warmup",
        ] {
            assert!(
                v.get(key).is_some_and(|x| !x.as_str().unwrap().is_empty()),
                "{key}"
            );
        }
        // The three labeled layers are distinct formulas (never blended).
        assert_ne!(v["prefill_throughput"], v["decode_throughput"]);
        assert_ne!(v["decode_throughput"], v["e2e_throughput"]);
    }
}
