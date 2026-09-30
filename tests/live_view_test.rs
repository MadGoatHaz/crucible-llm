//! Chunk 9 — View 1 (Live Monitor & Telemetry) acceptance tests.
//!
//! Verifies the blueprint §6 View 1 layout against *injected/synthetic*
//! metrics (the same `MetricsState::update` writer seam the stream worker
//! uses):
//!
//! - all five panels (telemetry gauges, ITL distribution, active-streams
//!   matrix, rolling throughput chart, log/event stream) render;
//! - the ITL histogram and the rolling throughput chart reflect changing
//!   snapshot data across repaints;
//! - the stream matrix shows the per-stream PP/TG split and MTP rate;
//! - the view survives a 60 Hz frame sequence (the Chunk 8 event-loop
//!   cadence) and degenerate terminal sizes without panicking.
//!
//! Rendering runs against `ratatui::backend::TestBackend`, so the suite is
//! fully offline and deterministic — no terminal attached.

use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::style::Color;
use ratatui::Terminal;

use crucible_llm::engines::{Engine, EngineProgress, SeqPhase, SeqState};
use crucible_llm::metrics::{MetricsSnapshot, StreamMetric, StreamStatus};
use crucible_llm::ui::app::App;
use crucible_llm::ui::views::live;

const W: u16 = 120;
const H: u16 = 40;

/// Render the Live view at `w`x`h` and return the resulting buffer.
fn render_live(app: &App, w: u16, h: u16) -> Buffer {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("TestBackend terminal");
    terminal
        .draw(|f| live::render(f.area(), app, f))
        .expect("render frame");
    terminal.backend().buffer().clone()
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

/// A synthetic snapshot with known values for the Live view tests
/// (replaces the removed `test_snapshot()` blueprint mock).
fn test_snapshot() -> MetricsSnapshot {
    let itl_bins = [
        0.92, 0.86, 0.79, 0.71, 0.62, 0.53, 0.44, 0.36, 0.29, 0.23, 0.18, 0.14, 0.11, 0.08, 0.06,
        0.045, 0.033, 0.024, 0.017, 0.012, 0.008, 0.005, 0.003, 0.002,
    ];
    MetricsSnapshot {
        endpoint: "http://127.0.0.1:8000/v1".into(),
        backend: "vLLM".into(),
        model: "test-model".into(),
        mode: "Concurrency".into(),
        aggregate_tps: 842.3,
        active_streams: 16,
        total_streams: 16,
        vram_used_gb: 21.4,
        vram_total_gb: 24.0,
        power_w: 285.0,
        joules_per_token: 0.338,
        gpu_clock_mhz: 1410.0,
        itl_p50_ns: 12_100_000,
        itl_p90_ns: 16_400_000,
        itl_p99_ns: 41_200_000,
        itl_p999_ns: 55_000_000,
        itl_bins: itl_bins.to_vec(),
        prompt_tokens: 4096,
        completion_tokens: 1332,
        reasoning_tokens: 1152,
        status: StreamStatus::Streaming,
        streams: vec![
            StreamMetric {
                id: 1,
                kind: "Reasoning".into(),
                state: StreamStatus::Streaming,
                pp_tokens: Some(2048),
                tg_tokens: Some(312),
                ttft_s: Some(0.182),
                gen_tps: Some(72.4),
                mtp: Some(1.84),
                progress: 0.55,
            },
            StreamMetric {
                id: 2,
                kind: "Content".into(),
                state: StreamStatus::Streaming,
                pp_tokens: Some(512),
                tg_tokens: Some(180),
                ttft_s: Some(0.045),
                gen_tps: Some(88.1),
                mtp: Some(1.02),
                progress: 0.70,
            },
            StreamMetric {
                id: 3,
                kind: "Tool-Call".into(),
                state: StreamStatus::Waiting,
                pp_tokens: Some(4096),
                tg_tokens: None,
                ttft_s: None,
                gen_tps: None,
                mtp: None,
                progress: 0.0,
            },
            StreamMetric {
                id: 4,
                kind: "Reasoning".into(),
                state: StreamStatus::Done,
                pp_tokens: Some(2048),
                tg_tokens: Some(840),
                ttft_s: Some(0.191),
                gen_tps: Some(68.9),
                mtp: Some(1.79),
                progress: 1.0,
            },
        ],
        throughput_series: (0..60)
            .map(|i| 842.3 + 18.0 * ((i as f64) * 0.31).sin())
            .collect(),
    }
}

// ---- acceptance: all five panels render with injected metrics ----

#[test]
fn all_five_panels_render_with_synthetic_metrics() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    for title in [
        "TELEMETRY GAUGES",
        "INTER-TOKEN LATENCY (ITL) DISTRIBUTION",
        "ACTIVE STREAMS MONITOR",
        "REAL-TIME SYSTEM PERFORMANCE",
        "LOG & EVENT STREAM",
    ] {
        assert!(text.contains(title), "missing panel: {title}");
    }
}

// ---- acceptance: top-left key metrics (blueprint §6) ----

#[test]
fn gauge_panel_shows_key_metrics() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("Total Aggregate"));
    assert!(text.contains("842.3 t/s"));
    assert!(text.contains("Active Streams"));
    assert!(text.contains("16 / 16"));
    assert!(text.contains("GPU Clock"));
    assert!(text.contains("1410 MHz"));
    assert!(text.contains("Current Power"));
    assert!(text.contains("285 W (0.338 J/token)"));
    // VRAM capacity bar label.
    assert!(text.contains("21.4 / 24.0 GB (89%)"));
}

#[test]
fn gpu_clock_shows_na_without_telemetry() {
    // Zeroed snapshot = no GPU/driver telemetry (the 0.0 sentinel,
    // blueprint §5D graceful degradation).
    let app = app_with(MetricsSnapshot::default());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("GPU Clock"));
    assert!(text.contains("N/A"));
}

// ---- acceptance: ITL percentiles + histogram reflect changing data ----

#[test]
fn itl_percentiles_render_from_snapshot() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("12.1 ms"));
    assert!(text.contains("16.4 ms"));
    assert!(text.contains("41.2 ms"));
}

#[test]
fn itl_histogram_and_rolling_chart_reflect_changing_data() {
    let app = app_with(test_snapshot());
    let before = render_live(&app, W, H);

    // Publish a second snapshot: mirrored ITL distribution, shifted
    // throughput series, new percentile values.
    let mut s = test_snapshot();
    s.itl_bins.reverse();
    s.throughput_series = s.throughput_series.iter().map(|v| v + 400.0).collect();
    s.itl_p50_ns = 99_000_000;
    s.itl_p90_ns = 120_000_000;
    s.itl_p99_ns = 180_000_000;
    app.metrics.update(s);

    let after = render_live(&app, W, H);
    assert_ne!(
        before, after,
        "repaint must reflect the newly published snapshot"
    );
    let text = buf_text(&after);
    assert!(text.contains("99.0 ms"));
    assert!(text.contains("120.0 ms"));
    assert!(text.contains("180.0 ms"));
}

// ---- acceptance: stream matrix shows PP/TG split and MTP rate ----

#[test]
fn stream_matrix_shows_pp_tg_split_and_mtp() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    // Row #01: Reasoning, streaming, 2048 PP / 312 TG.
    assert!(text.contains("#01"));
    assert!(text.contains("Reasoning"));
    assert!(text.contains("Streaming"));
    assert!(text.contains("2048/312"));
    assert!(text.contains("0.182 s"));
    assert!(text.contains("72.4 t/s"));
    assert!(text.contains("1.84 x"));
    // Row #02: Content split.
    assert!(text.contains("512/180"));
    assert!(text.contains("1.02 x"));
    // Row #03: waiting — TG not started, so `--` placeholder cells.
    assert!(text.contains("Waiting"));
    assert!(text.contains("4096/--"));
    // Row #04: done, 2048 PP / 840 TG.
    assert!(text.contains("Done"));
    assert!(text.contains("2048/840"));
    assert!(text.contains("1.79 x"));
}

// ---- acceptance: updates at the 60 Hz render cadence ----

#[test]
fn live_view_survives_60hz_frame_sequence() {
    let app = app_with(test_snapshot());
    for frame in 0..60u64 {
        // Simulate the engine publishing a fresh snapshot each tick, then
        // the 60 Hz render loop painting one frame from it.
        let tps = 842.3 + frame as f64;
        let mut s = test_snapshot();
        s.aggregate_tps = tps;
        s.throughput_series.push(tps.round());
        s.throughput_series.remove(0);
        app.metrics.update(s);

        let buf = render_live(&app, W, H);
        let text = buf_text(&buf);
        assert!(text.contains("TELEMETRY GAUGES"));
        // Every frame must show the *latest* published aggregate.
        assert!(text.contains(&format!("{tps:.1} t/s")));
    }
}

// ---- robustness: degenerate terminal sizes never panic ----

#[test]
fn renders_at_small_terminals_without_panic() {
    let app = app_with(test_snapshot());
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let _ = render_live(&app, w, h);
    }
}

// ---- VRAM gauge color tracks the usage thresholds ----

#[test]
fn vram_gauge_color_tracks_usage_thresholds() {
    let check = |ratio: f64, want: Color, not: Color| {
        let mut s = test_snapshot();
        s.vram_total_gb = 24.0;
        s.vram_used_gb = 24.0 * ratio;
        let app = app_with(s);
        let buf = render_live(&app, W, H);

        // Locate the "Target GPU VRAM" title, then inspect the gauge bar
        // rows directly below it. `find` yields a *byte* offset; the buffer
        // text holds multi-byte box-drawing/braille/█ glyphs, so convert to
        // a char offset before mapping onto the W x H cell grid.
        let text = buf_text(&buf);
        let byte_idx = text
            .find("Target GPU VRAM")
            .expect("VRAM gauge title in buffer");
        let char_idx = text[..byte_idx].chars().count();
        let title_row = char_idx / (W as usize);
        let mut saw_want = false;
        let mut saw_other = false;
        for (i, cell) in buf.content().iter().enumerate() {
            // `content()` is row-major over a W x H buffer.
            let row = i / (W as usize);
            if !(title_row..title_row + 3).contains(&row) {
                continue;
            }
            if cell.fg == want {
                saw_want = true;
            }
            if cell.fg == not {
                saw_other = true;
            }
        }
        assert!(saw_want, "expected {want:?} in VRAM gauge at ratio {ratio}");
        assert!(
            !saw_other,
            "unexpected {not:?} in VRAM gauge at ratio {ratio}"
        );
    };
    check(0.5, Color::Green, Color::Yellow);
    check(0.8, Color::Yellow, Color::Green);
    check(0.95, Color::Red, Color::Yellow);
}

// ---- graph rendering: throughput sparkline (block ramp + color gradient) ----

#[test]
fn throughput_sparkline_renders_block_ramp_with_color_gradient() {
    let mut s = test_snapshot();
    // A high/medium/low mix across the rolling window so all three
    // gradient colors appear.
    s.throughput_series = vec![900.0, 100.0, 480.0, 950.0, 60.0, 420.0];
    let app = app_with(s);
    let buf = render_live(&app, W, H);
    let text = buf_text(&buf);

    assert!(text.contains("REAL-TIME SYSTEM PERFORMANCE"));
    // Block ramp: the max sample is a full block, the min a sliver.
    assert!(text.contains('█'), "max sample renders a full block");
    assert!(text.contains('▁'), "min sample renders the smallest block");
    // Current value label (last sample of the window).
    assert!(text.contains("now 420.0 t/s"));
    // Color gradient: green cells at the top of the ramp, red at the
    // bottom, yellow in between.
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "█" && c.fg == Color::Green),
        "high samples are green"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "▁" && c.fg == Color::Red),
        "low samples are red"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.fg == Color::Yellow && c.symbol() != " "),
        "medium samples are yellow"
    );
}

#[test]
fn throughput_sparkline_degrades_gracefully_when_empty() {
    // Zeroed snapshot: no rolling samples, no throughput.
    let app = app_with(MetricsSnapshot::default());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("REAL-TIME SYSTEM PERFORMANCE"));
    assert!(text.contains("now 0.0 t/s"));
    // Tiny terminal: the guarded render path never panics.
    let _ = render_live(&app, 12, 8);
}

// ---- graph rendering: ITL percentile gauge bars ----

#[test]
fn itl_gauge_bars_render_percentile_values() {
    let app = app_with(test_snapshot());
    let buf = render_live(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("p50"));
    assert!(text.contains("p90"));
    assert!(text.contains("p99"));
    assert!(text.contains("12.1 ms"));
    assert!(text.contains("16.4 ms"));
    assert!(text.contains("41.2 ms"));
    // The three gauge bars carry the green/yellow/red gradient.
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "█" && c.fg == Color::Green),
        "p50 gauge bar is green"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "█" && c.fg == Color::Yellow),
        "p90 gauge bar is yellow"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "█" && c.fg == Color::Red),
        "p99 gauge bar is red"
    );
}

// ---- graph rendering: token counter ----

#[test]
fn token_counter_shows_total_generated() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("TOKENS GENERATED"));
    assert!(text.contains("1,332"));
    assert!(text.contains("1,152"));
    assert!(text.contains("4,096"));
}

// ---- benchmark sequence: header, progress bar, queue panel ----

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
    }
}

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
fn sequence_queue_marks_completed_running_and_queued() {
    let app = app_with(test_snapshot());
    app.seq.store(seq_state(
        SeqPhase::Running,
        Engine::Niah,
        Some(EngineProgress::Niah {
            size: 4000,
            depth: 3,
            total_depths: 11,
            cell: 13,
            total_cells: 77,
        }),
        vec![
            (Engine::Speed, "847 t/s decode".to_string()),
            (Engine::Concurrency, "sweet spot 16 streams".to_string()),
        ],
    ));
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("BENCHMARK QUEUE"), "queue panel");
    // Completed entries carry a check + summary.
    assert!(text.contains("A: Speed"), "completed A in queue");
    assert!(text.contains("847 t/s decode"), "A's summary");
    assert!(text.contains("B: Concurrency"), "completed B in queue");
    assert!(text.contains("sweet spot 16 streams"), "B's summary");
    // The running entry is highlighted; the rest are queued.
    assert!(text.contains("▶"), "running marker");
    assert!(text.contains("C1: NIAH"), "running C1 in queue");
    assert!(text.contains("○"), "queued marker");
    assert!(text.contains("C2: Reasoning"), "queued C2 in queue");
    assert!(text.contains("D: Energy"), "queued D in queue");
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
    // launch hint and the queue shows the full default list, dimmed.
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("No benchmark running"));
    assert!(text.contains("A: Speed"));
    assert!(text.contains("D: Energy"));
}

#[test]
fn sequence_ui_survives_small_terminals() {
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
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let _ = render_live(&app, w, h);
    }
}

// ---- engine results panel: data lines + dimmed ℹ info notes ----

#[test]
fn engine_results_panel_fills_in_as_engines_complete() {
    let app = app_with(test_snapshot());
    app.speed_slot
        .store(vec![crucible_llm::engines::SpeedResult {
            ttft: 0.25,
            prompt_tokens: 128,
            completion_tokens: 256,
            pp_speed: 1000.0,
            tg_speed: 60.6,
            mtp_efficiency: 1.0,
            stream_time: 4.2,
            total_chunks: 256,
            content_chunks: 256,
            reasoning_chunks: 0,
            other_chunks: 0,
            estimated: false,
            model: "test-model".into(),
            mode: "short".into(),
            error: None,
        }]);
    app.reasoning_slot
        .store(crucible_llm::engines::ReasoningResult {
            responses: vec![],
            ttfts: vec![],
            tg_speeds: vec![85.0],
            score: crucible_llm::engines::ReasoningScore {
                total: 13,
                solved: 12,
                by_category: [(5, 5), (4, 4), (3, 4)],
            },
        });
    app.structured_slot
        .store(crucible_llm::engines::StructuredResult {
            free_tps: 100.0,
            constrained_tps: 98.0,
            penalty_pct: -0.3,
            free_ttft: 0.1,
            constrained_ttft: 0.101,
            compliant: false,
            constrained_body: "{}".into(),
            free_body: "hi".into(),
        });

    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("ENGINE RESULTS"), "results panel title");
    // Each completed engine: its headline number …
    assert!(text.contains("A: Speed"), "{text}");
    assert!(text.contains("60.6 t/s decode"), "{text}");
    assert!(text.contains("C2: Reasoning"), "{text}");
    assert!(text.contains("12/13 solved"), "{text}");
    assert!(text.contains("C3: Structured"), "{text}");
    assert!(text.contains("compliant: NO"), "{text}");
    // … and the dimmed ℹ note explaining what it means.
    assert!(text.contains('ℹ'), "{text}");
    assert!(text.contains("Single-stream throughput"), "{text}");
    assert!(text.contains("Logical reasoning accuracy"), "{text}");
    assert!(text.contains("JSON compliance"), "{text}");
    // Engine D was never requested (no poller, no summary) → the results
    // panel carries no energy value ("D: Energy" also appears in the
    // queue panel's default list, so assert on the data value instead).
    assert!(!text.contains("N/A (no GPU telemetry)"), "{text}");
}

#[test]
fn engine_results_panel_placeholder_before_any_engine() {
    let app = app_with(test_snapshot());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("ENGINE RESULTS"), "{text}");
    assert!(text.contains("No engines completed yet"), "{text}");
}
