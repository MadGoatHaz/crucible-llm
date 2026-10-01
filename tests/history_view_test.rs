//! Chunk 14 — View 4 (Historical Comparison & Regression Diffing)
//! acceptance tests (blueprint §6 View 4).
//!
//! Verifies:
//! - stored sessions are loaded from the SQLite layer (newest first);
//! - selecting two stored runs (A/B) computes the signed delta metrics
//!   (% change in TTFT, tokens/sec, MTP rate, J/token) from the stored
//!   `stream_metrics` rows;
//! - the side-by-side view renders the two runs, the session list with
//!   A/B markers, and the delta table;
//! - gains are color-coded green and regressions red;
//! - the view degrades to a placeholder (no panic) with no history state,
//!   an empty session list, or a small terminal.
//!
//! Rendering runs against `ratatui::backend::TestBackend` — fully offline
//! and deterministic, no terminal attached.

use std::path::PathBuf;

use ratatui::buffer::Buffer;
use ratatui::style::Color;

use crucible_llm::storage::db::Database;
use crucible_llm::storage::models::{BenchmarkSession, StreamMetricRow};
use crucible_llm::ui::app::App;
use crucible_llm::ui::views::history::{self, session_summary, DiffReport, HistoryState};

mod common;
use common::render::render_buffer;

const W: u16 = 120;
const H: u16 = 40;

/// Render the History view at `w`x`h` and return the resulting buffer.
fn render_history(app: &App, w: u16, h: u16) -> Buffer {
    render_buffer(history::render, app, w, h)
}

/// Join every cell's symbol into one flat string for substring asserts.
fn buf_text(buf: &Buffer) -> String {
    buf.content().iter().map(|c| c.symbol()).collect()
}

/// The first buffer cell rendering `needle` in the flat text.
///
/// `str::find` returns a *byte* offset, but the buffer is indexed per
/// grapheme cell — and the buffer holds multi-byte box-drawing glyphs —
/// so map the byte offset back through the per-cell start offsets.
fn cell_at(buf: &Buffer, flat: &str, needle: &str) -> Option<ratatui::buffer::Cell> {
    let byte_idx = flat.find(needle)?;
    let mut starts = Vec::with_capacity(buf.content().len());
    let mut pos = 0usize;
    for c in buf.content() {
        starts.push(pos);
        pos += c.symbol().len();
    }
    let i = starts.binary_search(&byte_idx).ok()?;
    Some(buf.content()[i].clone())
}

fn session(id: &str, model: &str, ts: &str) -> BenchmarkSession {
    BenchmarkSession {
        session_id: id.to_string(),
        timestamp: Some(ts.to_string()),
        target_url: "http://127.0.0.1:8000/v1/chat/completions".to_string(),
        model_name: model.to_string(),
        backend_type: Some("vllm".to_string()),
        quantization: None,
        system_gpu: None,
        total_duration_sec: Some(10.0),
    }
}

fn metric(
    session_id: &str,
    ttft_ms: Option<f64>,
    tpot_ms: Option<f64>,
    mtp: Option<f64>,
) -> StreamMetricRow {
    StreamMetricRow {
        metric_id: None,
        session_id: session_id.to_string(),
        concurrency_level: Some(1),
        prompt_tokens: None,
        completion_tokens: None,
        reasoning_tokens: None,
        ttft_ms,
        tpot_ms,
        mtp_efficiency: mtp,
        joules_per_token: None,
        cache_hit: None,
    }
}

/// v0.6: TTFT 200 ms, 50 t/s (tpot 20 ms), MTP 1.0.
fn run_a() -> (BenchmarkSession, Vec<StreamMetricRow>) {
    let s = session(
        "aaaa1111-2222-3333-4444-555566667777",
        "vllm-0.6",
        "2026-09-29 20:00:00",
    );
    let m = vec![metric(&s.session_id, Some(200.0), Some(20.0), Some(1.0))];
    (s, m)
}

/// v0.7: TTFT 160 ms (−20% → gain), 62.5 t/s (+25% → gain), MTP 1.25 (+25% → gain).
fn run_b() -> (BenchmarkSession, Vec<StreamMetricRow>) {
    let s = session(
        "bbbb1111-2222-3333-4444-555566667777",
        "vllm-0.7",
        "2026-09-29 21:00:00",
    );
    let m = vec![metric(&s.session_id, Some(160.0), Some(16.0), Some(1.25))];
    (s, m)
}

/// b3300 vs b3200 regression: TTFT 240 ms (+20% → regression),
/// 40 t/s (−20% → regression), MTP 0.9 (−10% → regression).
fn run_c() -> (BenchmarkSession, Vec<StreamMetricRow>) {
    let s = session(
        "cccc1111-2222-3333-4444-555566667777",
        "llama.cpp-b3300",
        "2026-09-29 22:00:00",
    );
    let m = vec![metric(&s.session_id, Some(240.0), Some(25.0), Some(0.9))];
    (s, m)
}

fn history_with(
    a: &BenchmarkSession,
    a_rows: &[StreamMetricRow],
    b: &BenchmarkSession,
    b_rows: &[StreamMetricRow],
) -> HistoryState {
    let report = DiffReport::from_sessions(a, a_rows, b, b_rows);
    HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a.clone(), b.clone()],
        cursor: 0,
        run_a: Some(0),
        run_b: Some(1),
        summary_a: Some(report.summary_a.clone()),
        summary_b: Some(report.summary_b.clone()),
        diff: Some(report),
    }
}

// ---- storage flow: load + A/B selection computes the diff from SQLite ----

#[test]
fn load_lists_stored_sessions_newest_first() {
    let dir = std::env::temp_dir().join(format!("crucible-history-a-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db_path = dir.join("benchmarks.db");
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut db = Database::open(&db_path).unwrap();
    db.persist_run(&a, &ma).unwrap();
    db.persist_run(&b, &mb).unwrap();

    let h = HistoryState::load(&db_path).unwrap();
    assert_eq!(h.sessions.len(), 2);
    assert_eq!(h.sessions[0].session_id, b.session_id); // newest first
    assert_eq!(h.sessions[1].session_id, a.session_id);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn selecting_two_stored_runs_computes_signed_deltas() {
    let dir = std::env::temp_dir().join(format!("crucible-history-b-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db_path = dir.join("benchmarks.db");
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut db = Database::open(&db_path).unwrap();
    db.persist_run(&a, &ma).unwrap();
    db.persist_run(&b, &mb).unwrap();

    let mut h = HistoryState::load(&db_path).unwrap();
    assert_eq!(h.sessions[0].session_id, b.session_id);
    assert_eq!(h.sessions[1].session_id, a.session_id);

    // A = the older run (index 1), B = the newer run (index 0).
    h.move_cursor(1);
    h.select_a();
    h.move_cursor(-1);
    h.select_b();

    let d = h.diff.as_ref().expect("diff must be computed");
    assert_eq!(d.a.session_id, a.session_id);
    assert_eq!(d.b.session_id, b.session_id);
    // TTFT 200 → 160 ms: −20%, a gain.
    assert_eq!(d.rows[0].delta_pct, Some(-20.0));
    assert_eq!(d.rows[0].improved, Some(true));
    // 50 → 62.5 t/s: +25%, a gain.
    assert_eq!(d.rows[1].delta_pct, Some(25.0));
    assert_eq!(d.rows[1].improved, Some(true));
    // MTP 1.0 → 1.25: +25%, a gain.
    assert_eq!(d.rows[2].delta_pct, Some(25.0));
    assert_eq!(d.rows[2].improved, Some(true));
    // No J/token telemetry on either side → not comparable.
    assert_eq!(d.rows[3].delta_pct, None);
    assert_eq!(d.rows[3].improved, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn cursor_clamps_at_the_list_bounds() {
    let (a, _) = run_a();
    let (b, _) = run_b();
    let mut h = HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a, b],
        cursor: 0,
        run_a: None,
        run_b: None,
        summary_a: None,
        summary_b: None,
        diff: None,
    };
    h.move_cursor(-5);
    assert_eq!(h.cursor, 0);
    h.move_cursor(5);
    assert_eq!(h.cursor, 1);
    h.move_cursor(5);
    assert_eq!(h.cursor, 1); // clamped
    h.move_cursor(-5);
    assert_eq!(h.cursor, 0);
}

#[test]
fn same_session_on_both_sides_is_neutral() {
    let (s, rows) = run_a();
    let report = DiffReport::from_sessions(&s, &rows, &s, &rows);
    // TTFT / tokens/s / MTP are present on both sides → 0.0%, neutral.
    for r in &report.rows[..3] {
        assert_eq!(r.delta_pct, Some(0.0));
        assert_eq!(r.improved, None);
    }
    // J/token has no data on either side → not comparable.
    assert_eq!(report.rows[3].delta_pct, None);
    assert_eq!(report.rows[3].improved, None);
}

// ---- rendering: placeholder paths (no panic, no DB) ----

#[test]
fn render_without_history_shows_placeholder() {
    let app = App::new(); // history: None
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("No previous runs found"));
    assert!(text.contains("first benchmark"));
    assert!(text.contains("RUN A"));
    assert!(text.contains("RUN B"));
    assert!(text.contains("DELTA"));
}

#[test]
fn render_with_empty_session_list_shows_placeholder() {
    let mut app = App::new();
    app.history = Some(HistoryState::default());
    let text = buf_text(&render_history(&app, W, H));
    assert!(text.contains("No previous runs found"));
    assert!(text.contains("first benchmark"));
}

// ---- rendering: the side-by-side table with signed, color-coded deltas ----

#[test]
fn render_shows_side_by_side_runs_and_signed_deltas() {
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_with(&a, &ma, &b, &mb));
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);

    // Run panels carry the session identity + per-metric values.
    assert!(text.contains("RUN A"));
    assert!(text.contains("RUN B"));
    assert!(text.contains("aaaa1111"));
    assert!(text.contains("vllm-0.6"));
    assert!(text.contains("vllm-0.7"));
    assert!(text.contains("200.0 ms"));
    assert!(text.contains("160.0 ms"));
    assert!(text.contains("50.0 t/s"));
    assert!(text.contains("62.5 t/s"));

    // Session list with the A/B markers and key hints.
    assert!(text.contains("STORED SESSIONS"));
    assert!(text.contains("[j/k] move"));

    // Delta panel: signed % changes.
    assert!(text.contains("DELTA"));
    assert!(text.contains("-20.0%"));
    assert!(text.contains("+25.0%"));
}

#[test]
fn gains_render_green_and_regressions_red() {
    // Gains: v0.7 vs v0.6.
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_with(&a, &ma, &b, &mb));
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    let gain = cell_at(&buf, &text, "-20.0%").expect("TTFT gain delta rendered");
    assert_eq!(gain.fg, Color::Green, "TTFT improvement must be green");
    let gain2 = cell_at(&buf, &text, "+25.0%").expect("throughput gain delta rendered");
    assert_eq!(gain2.fg, Color::Green, "t/s improvement must be green");

    // Regressions: b3300 vs b3200.
    let (c, mc) = run_c();
    let mut app2 = App::new();
    app2.history = Some(history_with(&a, &ma, &c, &mc));
    let buf2 = render_history(&app2, W, H);
    let text2 = buf_text(&buf2);
    // b3300: TTFT 240 ms (+20%), 40 t/s (−20%), MTP 0.9 (−10%).
    let reg = cell_at(&buf2, &text2, "+20.0%").expect("TTFT regression delta rendered");
    assert_eq!(reg.fg, Color::Red, "TTFT regression must be red");
    let reg2 = cell_at(&buf2, &text2, "-20.0%").expect("t/s regression delta rendered");
    assert_eq!(reg2.fg, Color::Red, "t/s regression must be red");
}

#[test]
fn small_terminal_renders_without_panic() {
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_with(&a, &ma, &b, &mb));
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let buf = render_history(&app, w, h);
        assert!(!buf.content().is_empty());
    }
}

// ---- summary aggregation over multiple stored rows ----

#[test]
fn multi_iteration_sessions_average_their_rows() {
    let s = session(
        "dddd1111-2222-3333-4444-555566667777",
        "m",
        "2026-09-29 22:00:00",
    );
    let rows = vec![
        metric(&s.session_id, Some(100.0), Some(10.0), Some(1.0)),
        metric(&s.session_id, Some(300.0), Some(30.0), Some(2.0)),
    ];
    let sum = session_summary(&rows);
    assert_eq!(sum.ttft_ms, Some(200.0));
    // tps = mean(1000/10, 1000/30) = mean(100, 33.33…) = 66.67.
    assert!((sum.tokens_per_sec.unwrap() - 66.6667).abs() < 1e-3);
    assert_eq!(sum.mtp, Some(1.5));
}
