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
//!   aggregate / per-stream, MTP η, token counting, the loop guard, the
//!   discarded warmup, and the energy $/kWh rate source).
//!
//! These blocks are the trust signal: a reader of the JSON can verify
//! exactly how every number was produced without reading the source.
//! **CF-11: every string below states what the code actually does** — the
//! dual-TTFT definition, the CF-05 latency/bandwidth split, the CF-26
//! decode-window split, the CF-12 canonical MTP η (mirrored verbatim from
//! `engines/speed.rs::mtp_eta`), the CF-04 dual per-stream denominators
//! (mirrored verbatim from `engines/concurrency.rs`), and the real
//! discarded warmup that `SpeedEngine::run` / `Sweep::run` perform.
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
        "method": "quanta TSC-based monotonic clock, one Instant::now() per parsed token frame (CF-09: a batched MTP frame stamps once, so its tokens keep a real inter-frame gap)",
        "overhead_ns": overhead_ns,
        "isolation": "measurement thread separate from UI/storage, ArcSwap publish is wait-free"
    })
}

/// The `methodology` block of a JSON export: the package version plus the
/// exact formula behind every published metric.
///
/// CF-11: every string states what the code does. The MTP η line is
/// verbatim `engines/speed.rs::mtp_eta` (CF-12 canonical:
/// `completion_tokens / total_token_frames`, all Reasoning + Content
/// frames); the per-stream line is verbatim `engines/concurrency.rs`'s
/// CF-04 dual denominators (`tps_per_completed_stream` /
/// `tps_per_requested_stream`); the warmup line describes the real
/// discarded warmup run both engines perform before measuring.
#[must_use]
pub fn methodology_block(overhead_ns: u64) -> Value {
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "timing": {
            "resolution": "nanosecond",
            "method": "quanta TSC, one Instant::now() per parsed token frame",
            "overhead_ns": overhead_ns
        },
        "ttft": "dual: perceived = T3 - T0 (first token − request start; the full client round-trip, incl. network + server queue/prefill/first-decode); server-side = T3 - T1 (first token − response-headers; excludes the request-send network leg, isolates server prefill). T1 is stamped when the server's response headers arrive, not when the last request byte is written.",
        "first_token_latency": "CF-05: a LATENCY metric (reported in ms, never divided into a token count), measured on the SHORT prompt. perceived = T3 - T0 (client round-trip, network-included); server-side = T3 - T1 (server queue + prefill + first-decode). This is 'how responsive is this server'.",
        "prefill_throughput": "CF-05: a BANDWIDTH metric (tok/s) that scales with context, measured on a LONG prompt (a real context size, --tokens): prompt_tokens / (T3 - T1). The numerator is large enough that prefill compute dominates the TTFT denominator — unlike the old short-prompt 'PP speed', which was a latency floor wearing a bandwidth label.",
        "prefill_scaling": "Engine A's prefill_throughput is one point on the prefill-degradation curve (the compute→memory-bandwidth crossover as context grows). The full 2k→128k curve is measured by Engine C1's NIAH matrix (per-cell prefill_tps = prompt_tokens / TTFT); see that curve for how prefill bandwidth changes with context length.",
        "decode_throughput": "completion_tokens / (T_last_token - T_first_token) — the decode window runs from the first token frame to the last (prefill/TTFT and the trailing gap to [DONE] excluded). CF-26: when the server reports reasoning-token detail, the window splits at the reasoning→content transition and the rate is content_tokens / content window (the answer phase); without that detail it is the full-window blended rate.",
        "e2e_throughput": "total_tokens / total_wall_time (T0 → Tn)",
        "itl": "mean = (T_last_token - T_first_token) / (output_tokens - 1); the published percentiles (p50/p90/p99/p99.9) are the distribution of per-frame arrival deltas — one timestamp per parsed token frame (CF-09), so a batched MTP frame yields its real inter-frame gap, not a 1 ns artifact",
        "aggregate_throughput": "sum(all_completion_tokens) / wall_time",
        "per_stream_throughput": "CF-04 dual denominators: tps_per_completed_stream = aggregate_throughput / completed_streams (capacity view — what the healthy streams got) and tps_per_requested_stream = aggregate_throughput / requested_streams (user-experience view — a failed stream dilutes it, it gets 0). failed_stream_ratio = failed / requested; a level is flagged high_failure when that ratio exceeds 20%.",
        "mtp_eta": "CF-12 canonical MTP acceptance ratio: completion_tokens / total_token_frames, where total_token_frames is every token frame observed in the stream (Reasoning + Content combined). η = 1.0 means one token per frame (standard auto-regressive); η > 1 is legitimate and quantifies multi-token / speculative batching (vLLM MTP, EAGLE) — one SSE frame carries several accepted tokens. 0.0 = N/A when no token frame arrived (never a division by zero).",
        "token_counting": "server usage.completion_tokens (primary); exact-tokenizer re-encoding of the full reasoning+content text (fallback when a proxy omits usage); chars/4 estimate of that text (last resort, flagged estimated) — never the raw SSE frame count, which undercounts servers that batch tokens per frame",
        "loop_guard": "32-token sequence × 3 consecutive repetitions (the run's throughput layers are zeroed, it is excluded from aggregates, and it is flagged)",
        "warmup": "one discarded warmup run before Engine A's measured iterations and before Engine B's first ladder level: it absorbs cold-start / KV-cache effects so they don't pollute the first measured sample. It is logged, appears in no result, and is never counted.",
        "energy_rate": "$/kWh for the $/1M cost math: default 0.16 (a representative US residential/commercial blended rate — config.rs DEFAULT_RATE_KWH, GpuPowerMonitor's default), overridable via the config file, the CRUCIBLE_RATE_KWH environment variable, or the TUI Config view, and clamped to ≥ 0. The energy itself is measured (the GPU + CPU power integral), not fabricated; only the $ conversion uses this rate."
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
        // CF-09: the published method says per-frame stamping (the code's
        // reality), not the old "per token" claim.
        assert!(v["method"]
            .as_str()
            .unwrap()
            .contains("per parsed token frame"));
        assert!(v["isolation"].as_str().unwrap().contains("wait-free"));
    }

    #[test]
    fn methodology_block_documents_every_layer() {
        let v = methodology_block(42);
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["timing"]["overhead_ns"], 42);
        for key in [
            "ttft",
            "first_token_latency",
            "prefill_throughput",
            "prefill_scaling",
            "decode_throughput",
            "e2e_throughput",
            "itl",
            "aggregate_throughput",
            "per_stream_throughput",
            "mtp_eta",
            "token_counting",
            "loop_guard",
            "warmup",
            "energy_rate",
        ] {
            assert!(
                v.get(key).is_some_and(|x| !x.as_str().unwrap().is_empty()),
                "{key}"
            );
        }
        // CF-05: the latency and bandwidth metrics are distinct, and the
        // prefill metric is documented as a long-prompt (scaling) number.
        assert_ne!(v["first_token_latency"], v["prefill_throughput"]);
        assert!(v["prefill_throughput"]
            .as_str()
            .unwrap()
            .contains("LONG prompt"));
        assert!(v["prefill_scaling"].as_str().unwrap().contains("C1"));
        // The three labeled layers are distinct formulas (never blended).
        assert_ne!(v["prefill_throughput"], v["decode_throughput"]);
        assert_ne!(v["decode_throughput"], v["e2e_throughput"]);
        // CF-06: the TTFT line names the honest T1 (headers received).
        assert!(v["ttft"]
            .as_str()
            .unwrap()
            .contains("response headers arrive"));
    }

    /// CF-11: the **per-stream** line documents the CF-04 dual
    /// denominators with the exact names the code publishes
    /// (`engines/concurrency.rs` / `engines/flatout.rs`).
    #[test]
    fn methodology_per_stream_matches_the_dual_denominators() {
        let v = methodology_block(42);
        let s = v["per_stream_throughput"].as_str().unwrap();
        assert!(s.contains("tps_per_completed_stream"), "{s}");
        assert!(s.contains("tps_per_requested_stream"), "{s}");
        assert!(s.contains("failed_stream_ratio"), "{s}");
        assert!(s.contains("high_failure"), "{s}");
        // The old "/ active_streams" claim is gone.
        assert!(!s.contains("active_streams"), "{s}");
    }

    /// CF-11: the **MTP η** line states the CF-12 canonical formula
    /// verbatim (`engines/speed.rs::mtp_eta`) — and that function's
    /// behavior matches the documented arithmetic.
    #[test]
    fn methodology_mtp_eta_matches_the_canonical_formula() {
        use crate::engines::speed::mtp_eta;
        let v = methodology_block(42);
        let s = v["mtp_eta"].as_str().unwrap();
        assert!(s.contains("completion_tokens / total_token_frames"), "{s}");
        assert!(s.contains("Reasoning + Content"), "{s}");
        assert!(s.contains("η > 1 is legitimate"), "{s}");
        // The documented arithmetic, checked against the function:
        // 34 completion tokens over 4 token frames → 8.5 (the
        // headless-test mock stream's canonical example).
        assert!((mtp_eta(34, 4) - 8.5).abs() < 1e-9);
        // No token frame → 0.0 (N/A), never a division by zero.
        assert_eq!(mtp_eta(34, 0), 0.0);
    }

    /// CF-11: the **warmup** line describes a real, discarded warmup —
    /// the string must not claim more than the code does.
    #[test]
    fn methodology_warmup_is_documented_as_discarded() {
        let v = methodology_block(42);
        let s = v["warmup"].as_str().unwrap();
        assert!(s.contains("discarded"), "{s}");
        assert!(s.contains("Engine A"), "{s}");
        assert!(s.contains("Engine B"), "{s}");
        assert!(s.contains("never counted"), "{s}");
    }

    /// CF-55: the **energy rate** line documents the $0.16/kWh source
    /// (`config.rs::DEFAULT_RATE_KWH` / `GpuPowerMonitor`'s default), its
    /// override paths, and that the energy itself is measured.
    #[test]
    fn methodology_energy_rate_documents_the_default_source() {
        use crate::config::DEFAULT_RATE_KWH;
        let v = methodology_block(42);
        let s = v["energy_rate"].as_str().unwrap();
        assert!(
            s.contains(&DEFAULT_RATE_KWH.to_string()),
            "the documented default must equal config.rs: {s}"
        );
        assert!(s.contains("CRUCIBLE_RATE_KWH"), "{s}");
        assert!(s.contains("measured"), "{s}");
        // The published default and the code default are the same number.
        assert_eq!(DEFAULT_RATE_KWH, 0.16);
    }
}
