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

use crucible_llm::metrics::MetricsSnapshot;
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

// ---- acceptance: all five panels render with injected metrics ----

#[test]
fn all_five_panels_render_with_synthetic_metrics() {
    let app = app_with(MetricsSnapshot::sample());
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
    let app = app_with(MetricsSnapshot::sample());
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
    let app = app_with(MetricsSnapshot::sample());
    let text = buf_text(&render_live(&app, W, H));
    assert!(text.contains("12.1 ms"));
    assert!(text.contains("16.4 ms"));
    assert!(text.contains("41.2 ms"));
}

#[test]
fn itl_histogram_and_rolling_chart_reflect_changing_data() {
    let app = app_with(MetricsSnapshot::sample());
    let before = render_live(&app, W, H);

    // Publish a second snapshot: mirrored ITL distribution, shifted
    // throughput series, new percentile values.
    let mut s = MetricsSnapshot::sample();
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
    let app = app_with(MetricsSnapshot::sample());
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
    let app = app_with(MetricsSnapshot::sample());
    for frame in 0..60u64 {
        // Simulate the engine publishing a fresh snapshot each tick, then
        // the 60 Hz render loop painting one frame from it.
        let tps = 842.3 + frame as f64;
        let mut s = MetricsSnapshot::sample();
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
    let app = app_with(MetricsSnapshot::sample());
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let _ = render_live(&app, w, h);
    }
}

// ---- VRAM gauge color tracks the usage thresholds ----

#[test]
fn vram_gauge_color_tracks_usage_thresholds() {
    let check = |ratio: f64, want: Color, not: Color| {
        let mut s = MetricsSnapshot::sample();
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
