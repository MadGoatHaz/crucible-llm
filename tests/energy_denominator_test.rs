//! CF-41(b) / P3 CF-03 — the Joules/token denominator spans **all** engines
//! that produced tokens (A+B+C1+C2+C3+F), not just A+B.
//!
//! `BenchmarkSequence::all_engines_tokens` (sequence.rs:1213) is a *private*
//! lib method, so this integration test **mirrors** its summing rule over the
//! public result types and asserts the property the CF-03 fix establishes:
//! the all-engine denominator is strictly larger than the old A+B-only one,
//! and feeding it to the energy `profile` yields a *lower* (honest) J/token —
//! the numerator (the power-trace integral) always includes the most
//! energy-hungry capability cells, so the denominator must span the same
//! engine set.
//!
//! No live server, no network I/O: the result types are constructed with
//! known token contributions and the pure `profile` math is exercised
//! against a synthetic power trace.

use std::time::Duration;

use crucible_llm::engines::hardware::profile;
use crucible_llm::engines::{
    FlatOutResult, NiahResult, ReasoningResult, ReasoningScore, SpeedResult, StreamCountSource,
    StructuredResult, SweepLevel, SweepResult, NIAH_MAX_GEN_TOKENS,
};
use crucible_llm::hw::HwSample;
use crucible_llm::timing::MonotonicInstant;

// ── result-type constructors (known token contributions) ──────────────────

/// A minimal Engine A (speed) result with a known completion-token count.
fn speed_with_tokens(completion_tokens: u64) -> SpeedResult {
    SpeedResult {
        ttft: 0.2,
        prompt_tokens: 100,
        completion_tokens,
        reasoning_tokens: None,
        pp_speed: 500.0,
        tg_speed: 50.0,
        mtp_efficiency: 1.0,
        stream_time: 6.0,
        total_chunks: completion_tokens,
        content_chunks: completion_tokens,
        reasoning_chunks: 0,
        other_chunks: 0,
        estimated: false,
        model: "test-model".into(),
        mode: "short".into(),
        error: None,
        looping: false,
        usage_discrepancy: false,
        e2e_only: false,
        itl_clamped: 0,
    }
}

/// A one-level Engine B (concurrency) sweep with a known total-token count.
fn sweep_with_total_tokens(total_tokens: u64) -> SweepResult {
    SweepResult {
        levels: vec![SweepLevel {
            concurrency: 1,
            context: 0,
            aggregate_tps: 100.0,
            per_stream_tps: 100.0,
            p50_tpot_ns: 1_000_000,
            p90_tpot_ns: 2_000_000,
            p99_tpot_ns: 3_000_000,
            ttft_p50_ns: 10_000_000,
            ttft_p90_ns: 20_000_000,
            total_tokens,
            completed_streams: 1,
            failed_streams: 0,
            timed_out_streams: 0,
            timed_out_tokens: 0,
            aborted: false,
            wall_ns: 1_000_000_000,
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            estimated: false,
            streams: Vec::new(),
        }],
        ..SweepResult::default()
    }
}

/// An Engine C1 (NIAH) result with `run` cells marked as run (the rest
/// unrun). Each run cell contributes the 64-token gen cap to the estimate.
fn niah_with_run_cells(run: usize) -> NiahResult {
    let mut r = NiahResult::new(vec![2000], vec![50, 60]);
    for c in r.cells.iter_mut().take(run) {
        c.target_tokens = 2000; // mark the cell as run (an unrun cell is 0)
    }
    r
}

/// An Engine C2 (reasoning) result whose committed responses total `chars`
/// characters (the `chars/4` estimate source).
fn reasoning_with_response_chars(chars: usize) -> ReasoningResult {
    ReasoningResult {
        responses: vec!["x".repeat(chars)],
        ttfts: vec![0.2],
        tg_speeds: vec![50.0],
        score: ReasoningScore {
            total: 1,
            solved: 1,
            by_category: [(1, 1), (0, 0), (0, 0)],
        },
        estimated: false,
    }
}

/// An Engine C3 (structured) result whose free-form body is `chars`
/// characters (the `chars/4` estimate source); no scored cases.
fn structured_with_body_chars(chars: usize) -> StructuredResult {
    StructuredResult {
        free_tps: 65.2,
        constrained_tps: 61.8,
        penalty_pct: 5.2,
        free_ttft: 0.2,
        constrained_ttft: 0.21,
        cases: Vec::new(),
        constrained_body: String::new(),
        free_body: "x".repeat(chars),
    }
}

/// An Engine F (flat out) result with a known total-token count.
fn flatout_with_total_tokens(total_tokens: u64) -> FlatOutResult {
    FlatOutResult {
        stream_count: 3,
        stream_count_source: StreamCountSource::Default,
        total_tokens,
        duration_secs: 60.0,
        aggregate_tps: total_tokens as f64 / 60.0,
        per_stream_tps: total_tokens as f64 / 180.0,
        tps_per_completed_stream: total_tokens as f64 / 180.0,
        tps_per_requested_stream: total_tokens as f64 / 180.0,
        failed_stream_ratio: 0.0,
        config_error: None,
        max_tokens_used: 100_000,
        ttft_avg_ms: 100.0,
        itl_p50_ms: 20.0,
        itl_p99_ms: 40.0,
        tokens_estimated: false,
        counting_method: "usage".into(),
    }
}

// ── the mirrored all-engine denominator ───────────────────────────────────

/// Mirror of `BenchmarkSequence::all_engines_tokens` (sequence.rs:1213):
/// the J/token denominator spans every engine that produced tokens — real
/// counts for A/B/F, the labeled `chars/4` / NIAH-cap estimates for C1/C2/C3
/// (whose result types carry no output count). The production method is
/// private, so this re-states the same summing rule over the public result
/// types.
fn all_engines_tokens(
    speed: &[SpeedResult],
    sweep: &SweepResult,
    niah: &NiahResult,
    reasoning: &ReasoningResult,
    structured: &StructuredResult,
    flatout: &FlatOutResult,
) -> u64 {
    let mut total = 0u64;
    // Engine A — real completion tokens.
    total += speed.iter().map(|r| r.completion_tokens).sum::<u64>();
    // Engine B — real sweep total tokens.
    total += sweep.levels.iter().map(|l| l.total_tokens).sum::<u64>();
    // Engine C1 (NIAH) — estimate: run cells × the NIAH gen cap
    // (`NIAH_MAX_GEN_TOKENS` — 1024 after the Option C budget raise).
    let run_cells = niah.cells.iter().filter(|c| c.target_tokens > 0).count();
    total += (run_cells as u64) * NIAH_MAX_GEN_TOKENS as u64;
    // Engine C2 (Reasoning) — estimate: chars/4 over the response text.
    let chars: usize = reasoning.responses.iter().map(|s| s.chars().count()).sum();
    total += (chars / 4) as u64;
    // Engine C3 (Structured) — estimate: chars/4 over free + constrained.
    let chars: usize = structured.free_body.chars().count()
        + structured
            .cases
            .iter()
            .map(|c| c.output.chars().count())
            .sum::<usize>();
    total += (chars / 4) as u64;
    // Engine F (Flat Out) — real total tokens.
    total += flatout.total_tokens;
    total
}

/// A two-point 300 W power trace (a real gap so the integral is positive).
fn power_trace() -> Vec<HwSample> {
    let a = MonotonicInstant::now();
    std::thread::sleep(Duration::from_millis(50));
    let b = MonotonicInstant::now();
    vec![
        HwSample {
            t: a,
            power_mw: Some(300_000),
            ..HwSample::default()
        },
        HwSample {
            t: b,
            power_mw: Some(300_000),
            ..HwSample::default()
        },
    ]
}

// ── the test ───────────────────────────────────────────────────────────────

#[test]
fn all_engine_joules_per_token_denominator_spans_every_engine() {
    // Populate every engine with a known token contribution:
    //   A: 256 (real)  B: 1000 (real)  C1: 1 run cell × NIAH_MAX_GEN_TOKENS
    //   (est.)  C2: 400 chars / 4 = 100 (est.)  C3: 800 chars / 4 = 200
    //   (est.)  F: 5000 (real)
    let speed = vec![speed_with_tokens(256)];
    let sweep = sweep_with_total_tokens(1000);
    let niah = niah_with_run_cells(1);
    let reasoning = reasoning_with_response_chars(400);
    let structured = structured_with_body_chars(800);
    let flatout = flatout_with_total_tokens(5000);

    // The old A+B-only denominator.
    let a_b_only = speed.iter().map(|r| r.completion_tokens).sum::<u64>()
        + sweep.levels.iter().map(|l| l.total_tokens).sum::<u64>();
    assert_eq!(
        a_b_only,
        256 + 1000,
        "A+B-only is the old (inflating) denominator"
    );

    // The honest all-engine denominator (mirrors sequence.rs::all_engines_tokens).
    // The C1 term references `NIAH_MAX_GEN_TOKENS` so it tracks the budget
    // (1024 after the Option C raise) instead of hard-coding a stale value.
    let all = all_engines_tokens(&speed, &sweep, &niah, &reasoning, &structured, &flatout);
    assert_eq!(
        all,
        256 + 1000 + NIAH_MAX_GEN_TOKENS as u64 + 100 + 200 + 5000,
        "the denominator spans A+B+C1+C2+C3+F, not just A+B"
    );
    assert!(
        all > a_b_only,
        "the all-engine denominator ({all}) must exceed the old A+B-only ({a_b_only})"
    );

    // The observable effect: feeding the *same* power trace to `profile`
    // with the all-engine denominator yields a *lower* (honest) J/token than
    // the A+B-only one — the C1/C2/C3/F tokens the numerator already paid
    // for are now in the denominator too.
    let trace = power_trace();
    let jpt_all = profile(&trace, None, all, 0.0)
        .joules_per_token
        .expect("power + tokens");
    let jpt_ab = profile(&trace, None, a_b_only, 0.0)
        .joules_per_token
        .expect("power + tokens");
    assert!(
        jpt_all < jpt_ab,
        "all-engine J/token ({jpt_all}) < A+B-only J/token ({jpt_ab}) — the inflation is removed"
    );

    // The N/A rule: zero tokens (no engine ran) → no spurious 0.0.
    assert_eq!(profile(&trace, None, 0, 0.0).joules_per_token, None);
}
