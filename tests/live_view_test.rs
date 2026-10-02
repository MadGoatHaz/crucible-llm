//! View 1 (Live Monitor & Telemetry) acceptance tests — the **remote-user**
//! redesign.
//!
//! The remote operator does not sit on the GPU machine, so the panels that
//! read `N/A` off-box (VRAM gauge, GPU clock, power, the per-stream monitor
//! matrix, the ITL braille histogram, the benchmark queue) are gone. The
//! view now shows, top to bottom:
//!
//! * the **BENCHMARK SEQUENCE** header (current engine + progress bar);
//! * the **throughput hero** (a large real-time tokens/sec block chart)
//!   beside the **KEY METRICS** readout;
//! * the **CONCURRENCY CURVE** (Engine B) — shown while B runs / after the
//!   run / when a sweep is on screen;
//! * the **CAPABILITY SCORES** (C1/C2/C3/D horizontal bars) — shown while a
//!   C/D engine runs / after the run / when a result is on screen;
//! * a compact **EVENT LOG**.
//!
//! These tests verify the layout against *injected/synthetic* metrics and
//! result slots (the same lock-free writer seams the engines use) and confirm
//! the engine-adaptive visibility: panels that don't apply are hidden, never
//! rendered empty. Rendering runs against `ratatui::backend::TestBackend`, so
//! the suite is fully offline and deterministic — no terminal attached.

use ratatui::buffer::Buffer;

use crucible_llm::engines::{
    evaluate_case, Engine, EngineProgress, ReasoningResult, ReasoningScore, SeqPhase, SeqState,
    StructuredResult, SweepLevel, SweepResult,
};
use crucible_llm::metrics::MetricsSnapshot;
use crucible_llm::ui::app::App;
use crucible_llm::ui::theme::Theme;
use crucible_llm::ui::views::live;

mod common;
use common::render::render_buffer;

const W: u16 = 120;
const H: u16 = 40;

/// Render the Live view at `w`x`h` and return the resulting buffer.
fn render_live(app: &App, w: u16, h: u16) -> Buffer {
    render_buffer(live::render, app, w, h)
}

/// Join every cell's symbol into one flat string for substring asserts.
fn buf_text(buf: &Buffer) -> String {
    buf.content().iter().map(|c| c.symbol()).collect()
}

/// A fresh `App` whose snapshot has been replaced by `snap` via the
/// lock-free writer path (identical to what the stream worker does).
fn app_with(snap: MetricsSnapshot) -> App {
    let app = App::new();
    app.metrics.update(snap);
    app
}

/// A synthetic snapshot with known values for the Live view tests.
fn test_snapshot() -> MetricsSnapshot {
    MetricsSnapshot {
        endpoint: "http://127.0.0.1:8000/v1".into(),
        backend: "vLLM".into(),
        model: "test-model".into(),
        mode: "Concurrency".into(),
        aggregate_tps: 842.3,
        active_streams: 16,
        total_streams: 16,
        itl_p50_ns: 12_100_000,
        itl_p90_ns: 16_400_000,
        itl_p99_ns: 41_200_000,
        itl_p999_ns: 55_000_000,
        prompt_tokens: 4096,
        completion_tokens: 1332,
        reasoning_tokens: 1152,
        streams: vec![
            stream(1, Some(0.2)),
            stream(2, Some(0.2)),
            stream(3, Some(0.2)),
        ],
        throughput_series: (0..60)
            .map(|i| 842.3 + 18.0 * ((i as f64) * 0.31).sin())
            .collect(),
        ..MetricsSnapshot::default()
    }
}

/// A minimal stream row (only the TTFT is exercised by the key metrics).
fn stream(id: u32, ttft: Option<f64>) -> crucible_llm::metrics::StreamMetric {
    crucible_llm::metrics::StreamMetric {
        id,
        kind: "Content".into(),
        state: crucible_llm::metrics::StreamStatus::Streaming,
        pp_tokens: Some(1024),
        tg_tokens: Some(128),
        ttft_s: ttft,
        gen_tps: Some(72.4),
        mtp: Some(1.0),
        progress: 0.5,
        looping: false,
    }
}

/// One sweep level (concurrency, aggregate t/s, p90 TPOT in ms).
fn lvl(concurrency: usize, tps: f64, p90_ms: f64) -> SweepLevel {
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
        completed_streams: 0,
        failed_streams: 0,
        timed_out_streams: 0,
        aborted: false,
        wall_ns: 0,
        loop_excluded_streams: 0,
        loop_excluded_tokens: 0,
        streams: Vec::new(),
    }
}

/// A full A→D queue in the given phase (the shape the executor publishes).
fn seq_state(
    phase: SeqPhase,
    engine: Engine,
    progress: Option<EngineProgress>,
    completed: Vec<(Engine, String)>,
) -> SeqState {
    SeqState {
        phase,
        queue: Engine::ALL.to_vec(),
        engine,
        progress,
        summary: String::new(),
        completed,
        engine_started_ms: 0,
    }
}

// ---- acceptance: the default (idle) view shows only the core panels ----

#[test]
fn default_view_shows_hero_key_metrics_and_log() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    for title in [
        "BENCHMARK SEQUENCE",
        "THROUGHPUT",
        "OVERALL METRICS",
        "EVENT LOG",
    ] {
        assert!(text.contains(title), "missing panel: {title}");
    }
    // The removed hardware / on-local panels are gone.
    assert!(!text.contains("Target GPU VRAM"), "VRAM panel removed");
    assert!(
        !text.contains("ACTIVE STREAMS MONITOR"),
        "stream matrix removed"
    );
    assert!(!text.contains("ITL) DISTRIBUTION"), "ITL histogram removed");
    assert!(!text.contains("BENCHMARK QUEUE"), "queue panel removed");
    // No engine running → the adaptive panels stay hidden (not empty).
    assert!(
        !text.contains("CONCURRENCY CURVE"),
        "concurrency hidden when idle"
    );
    assert!(
        !text.contains("CAPABILITY SCORES"),
        "capabilities hidden when idle"
    );
}

// ---- acceptance: key metrics show real data (no N/A hardware) ----

#[test]
fn overall_metrics_panel_shows_cumulative_stats() {
    // FIX 1: the panel shows *cumulative* stats across all engines. Give it
    // three **completed** streams (state = Done) so the overall accumulator
    // records per-stream gen / TTFT / tokens, plus the aggregate prompt /
    // ITL / active samples.
    let mut s = test_snapshot();
    for st in &mut s.streams {
        st.state = crucible_llm::metrics::StreamStatus::Done;
    }
    let app = app_with(s);
    let text = buf_text(&render_live(&app, W, H));

    // The panel is clearly cumulative (max / avg / p5, not the live value).
    assert!(text.contains("OVERALL METRICS"), "{text}");
    // Per-stream gen throughput (72.4 t/s from each stream row).
    assert!(text.contains("72.4"), "gen throughput: {text}");
    // TTFT: 0.2 s → 200 ms.
    assert!(text.contains("200"), "TTFT ms: {text}");
    // ITL p50 / p99 (12.1 / 41.2 ms).
    assert!(text.contains("12.1"), "ITL p50: {text}");
    assert!(text.contains("41.2"), "ITL p99: {text}");
    // Total tokens = the sum of the server-reported stream tokens (3 × 128).
    assert!(text.contains("384"), "total tokens: {text}");
    // Prompt throughput: 4096 prompt tokens / 0.2 s mean TTFT.
    assert!(text.contains("20480"), "prompt throughput: {text}");
    // Active streams (16).
    assert!(text.contains("16"), "active streams: {text}");
    // The cumulative legend explains avg / max / p5.
    assert!(text.contains("5th percentile"), "p5 legend: {text}");
    assert!(text.contains('ℹ'), "overall info note");
}

#[test]
fn overall_metrics_show_placeholders_without_telemetry() {
    // A zeroed snapshot (no streams, no ITL) must degrade to `--`, never
    // panic or show a fake value.
    let app = app_with(MetricsSnapshot::default());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("OVERALL METRICS"));
    // The hero header still shows the live `now` (0.0 t/s).
    assert!(text.contains("0.0 t/s"));
    assert!(text.contains("--"), "missing metrics show `--`");
}

// ---- acceptance: throughput hero (block chart + axes + gradient) ----

#[test]
fn throughput_hero_renders_block_chart_with_axes() {
    let mut s = test_snapshot();
    // A high/medium/low mix across the window so all three gradient colors
    // appear and the y-axis shows the max. `now` is the latest published
    // aggregate (420.0 here); `PEAK` is the window max (950.0).
    s.throughput_series = vec![900.0, 100.0, 480.0, 950.0, 60.0, 420.0];
    s.aggregate_tps = 420.0;
    let app = app_with(s);
    let buf = render_live(&app, W, H);
    let text = buf_text(&buf);

    assert!(text.contains("THROUGHPUT"));
    // FIX 2: the header is now / peak / avg (lowercase, not `PEAK:`).
    assert!(text.contains("now 420.0 t/s"), "{text}");
    assert!(text.contains("peak 950.0 t/s"), "{text}");
    // The block chart body + the x-axis time labels (spanning the *actual*
    // 6-sample → 5 s window).
    assert!(text.contains('█'), "bars rendered");
    assert!(text.contains("0s"), "x-axis start");
    assert!(text.contains("5s"), "x-axis end");
    // The layered gradient: a hot white/cyan cap on the tall bars, a dim
    // blue floor at the base of each bar (glowing from within).
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "█" && c.fg == Theme::Cyberpunk.bright()),
        "the hot cap is bright white/cyan"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.fg == Theme::Cyberpunk.floor()),
        "the dim blue floor is present"
    );
}

#[test]
fn throughput_hero_degrades_gracefully_when_empty() {
    // Zeroed snapshot: no rolling samples, no throughput.
    let app = app_with(MetricsSnapshot::default());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("THROUGHPUT"));
    assert!(text.contains("now 0.0 t/s"));
    // Tiny terminal: the guarded render path never panics.
    let _ = render_live(&app, 12, 8);
}

// ---- acceptance: concurrency curve is engine-adaptive ----

#[test]
fn concurrency_curve_shows_while_engine_b_runs() {
    let app = app_with(test_snapshot());
    // A completed sweep on screen + Engine B running.
    app.sweep.store(SweepResult {
        levels: vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0), lvl(4, 340.0, 20.0)],
        matrix: None,
    });
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Concurrency,
        Some(EngineProgress::Concurrency {
            level: 4,
            step: 3,
            total_steps: 7,
            active: 4,
        }),
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("CONCURRENCY CURVE"), "curve panel visible");
    // The curve plots the sweep (markers for sweet spot / knee).
    assert!(
        text.contains('◆') || text.contains('▲'),
        "curve markers rendered"
    );
}

#[test]
fn concurrency_curve_hidden_during_engine_a() {
    let app = app_with(test_snapshot());
    // Sweep data exists, but Engine A is running → the curve is hidden
    // (never shown empty for the wrong engine).
    app.sweep.store(SweepResult {
        levels: vec![lvl(1, 100.0, 5.0)],
        matrix: None,
    });
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Speed,
        None,
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(
        !text.contains("CONCURRENCY CURVE"),
        "hidden during Engine A: {text}"
    );
}

#[test]
fn concurrency_curve_shows_in_progress_note_mid_sweep() {
    let app = app_with(test_snapshot());
    // Engine B running but no sweep result published yet.
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Concurrency,
        Some(EngineProgress::Concurrency {
            level: 8,
            step: 2,
            total_steps: 7,
            active: 8,
        }),
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("CONCURRENCY CURVE"), "panel visible");
    assert!(
        text.contains("in progress"),
        "mid-sweep shows an in-progress note: {text}"
    );
}

// ---- acceptance: capability scores are engine-adaptive ----

#[test]
fn capability_scores_show_for_c_engines() {
    let app = app_with(test_snapshot());
    app.reasoning_slot.store(ReasoningResult {
        responses: vec![],
        ttfts: vec![],
        tg_speeds: vec![85.0],
        score: ReasoningScore {
            total: 13,
            solved: 12,
            by_category: [(5, 5), (4, 4), (3, 4)],
        },
    });
    app.structured_slot.store(StructuredResult {
        free_tps: 100.0,
        constrained_tps: 98.0,
        penalty_pct: -0.3,
        free_ttft: 0.1,
        constrained_ttft: 0.101,
        cases: vec![
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
            evaluate_case("Medium", "not json"),
            evaluate_case("Complex", "also not json"),
        ],
        constrained_body: "{}".into(),
        free_body: "hi".into(),
    });
    // Engine C2 (Reasoning) running → the capability panel is visible.
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Reasoning,
        Some(EngineProgress::Reasoning {
            challenge: 5,
            total: 13,
        }),
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("CAPABILITY ASSESSMENT"), "panel visible");
    // Only the engines that *ran* show a bar.
    assert!(text.contains("Reasoning"), "run: reasoning bar");
    assert!(text.contains("Structured"), "run: structured bar");
    assert!(
        !text.contains("Long Context"),
        "unrun: no long-context bar: {text}"
    );
    // The stored results render their values.
    assert!(text.contains("12/13"), "reasoning score");
    assert!(text.contains("compliant"), "structured score");
    // The ⚡ warning marks the poor scores.
    assert!(text.contains('⚡'), "warning on poor scores");
    // The practical OVERALL summary.
    assert!(text.contains("OVERALL"), "overall summary");
}

#[test]
fn capability_scores_hidden_during_engine_a() {
    let app = app_with(test_snapshot());
    app.reasoning_slot.store(ReasoningResult {
        responses: vec![],
        ttfts: vec![],
        tg_speeds: vec![85.0],
        score: ReasoningScore {
            total: 13,
            solved: 12,
            by_category: [(5, 5), (4, 4), (3, 4)],
        },
    });
    // Engine A running → capabilities hidden even though data exists.
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Speed,
        None,
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(
        !text.contains("CAPABILITY SCORES"),
        "hidden during Engine A: {text}"
    );
}

// ---- acceptance: after all complete, the full summary shows every panel ----

#[test]
fn all_complete_shows_the_full_summary() {
    let app = app_with(test_snapshot());
    app.sweep.store(SweepResult {
        levels: vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0)],
        matrix: None,
    });
    app.reasoning_slot.store(ReasoningResult {
        responses: vec![],
        ttfts: vec![],
        tg_speeds: vec![85.0],
        score: ReasoningScore {
            total: 13,
            solved: 12,
            by_category: [(5, 5), (4, 4), (3, 4)],
        },
    });
    let mut state = seq_state(SeqPhase::AllComplete, Engine::Hardware, None, Vec::new());
    state.summary = "6 of 6 engines complete".to_string();
    app.seq.store(state);

    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("ALL BENCHMARKS COMPLETE"), "header");
    assert!(text.contains("THROUGHPUT"), "hero");
    assert!(text.contains("OVERALL METRICS"), "overall metrics");
    assert!(
        text.contains("CONCURRENCY CURVE"),
        "concurrency in the summary"
    );
    assert!(
        text.contains("CAPABILITY ASSESSMENT"),
        "capabilities in the summary"
    );
    assert!(text.contains("EVENT LOG"), "log");
}

// ---- benchmark sequence: header, progress bar ----

#[test]
fn sequence_header_shows_running_engine_progress_and_bar() {
    let app = app_with(test_snapshot());
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Speed,
        Some(EngineProgress::Speed {
            iteration: 2,
            total: 5,
            tokens: 248,
        }),
        Vec::new(),
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("BENCHMARK SEQUENCE"), "header panel");
    assert!(text.contains("ENGINE A: SPEED"), "current engine name");
    assert!(text.contains("Running"), "phase");
    assert!(text.contains("Iteration 2/5"), "progress line");
    assert!(text.contains("248 tok"), "tokens so far");
    assert!(text.contains('█'), "progress bar has filled cells");
    assert!(text.contains('░'), "progress bar has empty cells");
}

#[test]
fn sequence_header_shows_each_engine_progress_shape() {
    let cases = [
        (
            Engine::Concurrency,
            EngineProgress::Concurrency {
                level: 8,
                step: 3,
                total_steps: 7,
                active: 8,
            },
            "Concurrency level 8 (step 3/7)",
        ),
        (
            Engine::Niah,
            EngineProgress::Niah {
                size: 4000,
                depth: 3,
                total_depths: 11,
                cell: 13,
                total_cells: 77,
            },
            "Size 4k, Depth 3/11",
        ),
        (
            Engine::Reasoning,
            EngineProgress::Reasoning {
                challenge: 5,
                total: 13,
            },
            "Challenge 5/13",
        ),
        (
            Engine::Structured,
            EngineProgress::Structured { run: 2, total: 2 },
            "Run 2/2",
        ),
        (
            Engine::Hardware,
            EngineProgress::Sampling { elapsed: 2.5 },
            "Sampling… 2.5s",
        ),
    ];
    for (engine, progress, expect) in cases {
        let app = app_with(test_snapshot());
        app.seq.store(seq_state(
            SeqPhase::Running,
            engine,
            Some(progress),
            Vec::new(),
        ));
        let text = buf_text(&render_live(&app, W, H));
        assert!(
            text.contains(engine.title()),
            "header names {engine:?}: {text}"
        );
        assert!(text.contains(expect), "missing {expect:?}: {text}");
    }
}

#[test]
fn sequence_all_complete_header_wins() {
    let app = app_with(test_snapshot());
    let mut state = seq_state(SeqPhase::AllComplete, Engine::Hardware, None, Vec::new());
    state.summary = "6 of 6 engines complete".to_string();
    app.seq.store(state);
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("ALL BENCHMARKS COMPLETE"));
    assert!(text.contains("6 of 6 engines complete"));
}

#[test]
fn sequence_idle_before_any_run_shows_a_hint() {
    // A fresh App has never started a sequence: the header offers the
    // launch hint.
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("No benchmark running"));
}

// ---- robustness: degenerate terminal sizes never panic ----

#[test]
fn live_view_survives_60hz_frame_sequence() {
    let app = app_with(test_snapshot());
    for frame in 0..60u64 {
        // Simulate the engine publishing a fresh snapshot each tick, then
        // the 60 Hz render loop painting one frame from it.
        let tps = 842.3 + frame as f64;
        let mut s = test_snapshot();
        s.aggregate_tps = tps;
        s.completion_tokens = 1332 + frame;
        app.metrics.update(s);

        let buf = render_live(&app, W, H);
        let text = buf_text(&buf);
        assert!(text.contains("THROUGHPUT"));
        // Every frame must render the rate unit (the cumulative decode
        // rate is shown, not the raw aggregate).
        assert!(text.contains("t/s"));
    }
}

#[test]
fn renders_at_small_terminals_without_panic() {
    let app = app_with(test_snapshot());
    // Also exercise the all-complete (every panel) layout at small sizes.
    let mut state = seq_state(SeqPhase::AllComplete, Engine::Hardware, None, Vec::new());
    state.summary = "6 of 6".to_string();
    app.seq.store(state);
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let _ = render_live(&app, w, h);
    }
}
