//! View 4 (History) acceptance tests: list, detail, compare, delete,
//! and empty-state rendering.
//!
//! Verifies:
//! - stored sessions are loaded from the SQLite layer (newest first);
//! - the List mode renders the session table with cursor highlighting;
//! - the Detail mode shows per-engine metrics for one run;
//! - the Compare mode shows signed, color-coded deltas between two runs;
//! - the DeleteConfirm mode shows the `[y/N]` prompt;
//! - the empty state shows the placeholder;
//! - the view degrades without a panic on small terminals.
//!
//! Rendering runs against `ratatui::backend::TestBackend` — fully offline
//! and deterministic, no terminal attached.

use std::path::PathBuf;

use ratatui::buffer::Buffer;

use crucible_llm::storage::db::Database;
use crucible_llm::storage::models::{BenchmarkSession, StreamMetricRow};
use crucible_llm::ui::app::App;
use crucible_llm::ui::theme::Theme;
use crucible_llm::ui::views::history::{
    self, session_summary, DiffReport, HistoryMode, HistoryState,
};

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

/// Build a `HistoryState` in Compare mode with both runs selected.
fn history_compare(
    a: &BenchmarkSession,
    a_rows: &[StreamMetricRow],
    b: &BenchmarkSession,
    b_rows: &[StreamMetricRow],
) -> HistoryState {
    let report = DiffReport::from_sessions(a, a_rows, b, b_rows);
    HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a.clone(), b.clone()],
        cursor: 1,
        mode: HistoryMode::Compare {
            first: 0,
            second: Some(1),
        },
        detail_summary: None,
        detail_needle: None,
        compare_diff: Some(report),
    }
}

// ---- storage flow: load + compare from SQLite ----

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
    assert_eq!(h.mode, HistoryMode::List);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compare_mode_computes_signed_deltas_from_sqlite() {
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

    // Select the older run (index 1) as first, then the newer (index 0) as second.
    h.cursor = 1;
    h.enter_compare();
    h.cursor = 0;
    h.select_compare_second();

    let d = h.compare_diff.as_ref().expect("diff must be computed");
    assert_eq!(d.a.session_id, a.session_id);
    assert_eq!(d.b.session_id, b.session_id);
    // TTFT 200 → 160 ms: −20%, a gain.
    assert_eq!(d.rows[0].delta_pct, Some(-20.0));
    assert_eq!(d.rows[0].improved, Some(true));
    // 50 → 62.5 t/s: +25%, a gain.
    assert_eq!(d.rows[1].delta_pct, Some(25.0));
    assert_eq!(d.rows[1].improved, Some(true));
    // MTP 1.0 → 1.25: +25%, a gain.
    assert_eq!(d.rows[3].delta_pct, Some(25.0));
    assert_eq!(d.rows[3].improved, Some(true));
    // No J/token telemetry on either side → not comparable.
    assert_eq!(d.rows[4].delta_pct, None);
    assert_eq!(d.rows[4].improved, None);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn delete_session_removes_from_db() {
    let dir = std::env::temp_dir().join(format!("crucible-history-del-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let db_path = dir.join("benchmarks.db");
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut db = Database::open(&db_path).unwrap();
    db.persist_run(&a, &ma).unwrap();
    db.persist_run(&b, &mb).unwrap();
    assert_eq!(db.list_sessions().unwrap().len(), 2);

    // Delete session A.
    db.delete_session(&a.session_id).unwrap();
    let remaining = db.list_sessions().unwrap();
    assert_eq!(remaining.len(), 1);
    assert_eq!(remaining[0].session_id, b.session_id);
    // The stream metrics for A are also gone.
    assert!(db.get_stream_metrics(&a.session_id).unwrap().is_empty());
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
        mode: HistoryMode::List,
        ..Default::default()
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
    // TTFT / Gen t/s / MTP are present on both sides → 0.0%, neutral.
    // Prompt t/s (row 2) has no data → not comparable.
    assert_eq!(report.rows[0].delta_pct, Some(0.0));
    assert_eq!(report.rows[0].improved, None);
    assert_eq!(report.rows[1].delta_pct, Some(0.0));
    assert_eq!(report.rows[1].improved, None);
    assert_eq!(report.rows[2].delta_pct, None);
    assert_eq!(report.rows[2].improved, None);
    assert_eq!(report.rows[3].delta_pct, Some(0.0));
    assert_eq!(report.rows[3].improved, None);
    // J/token has no data on either side → not comparable.
    assert_eq!(report.rows[4].delta_pct, None);
    assert_eq!(report.rows[4].improved, None);
}

// ---- rendering: empty state ----

#[test]
fn render_without_history_shows_empty_state() {
    let app = App::new(); // history: None
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("No benchmark runs saved yet"));
}

#[test]
fn render_with_empty_session_list_shows_empty_state() {
    let mut app = App::new();
    app.history = Some(HistoryState::default());
    let text = buf_text(&render_history(&app, W, H));
    assert!(text.contains("No benchmark runs saved yet"));
}

// ---- rendering: list mode ----

#[test]
fn render_list_shows_sessions_with_cursor() {
    let (a, _) = run_a();
    let (b, _) = run_b();
    let mut app = App::new();
    app.history = Some(HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a, b],
        cursor: 0,
        mode: HistoryMode::List,
        ..Default::default()
    });
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("HISTORY"));
    assert!(text.contains("2 total"));
    assert!(text.contains("vllm-0.7")); // newest first
    assert!(text.contains("vllm-0.6"));
    assert!(text.contains("> [1]"), "cursor on row 1: {text}");
}

#[test]
fn render_list_cursor_on_second_row() {
    let (a, _) = run_a();
    let (b, _) = run_b();
    let mut app = App::new();
    app.history = Some(HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a, b],
        cursor: 1,
        mode: HistoryMode::List,
        ..Default::default()
    });
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("> [2]"), "cursor on row 2: {text}");
}

// ---- rendering: detail mode ----

#[test]
fn render_detail_shows_metrics() {
    let (a, ma) = run_a();
    let sum = session_summary(&ma);
    let mut app = App::new();
    app.history = Some(HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a.clone()],
        cursor: 0,
        mode: HistoryMode::Detail(0),
        detail_summary: Some(sum),
        detail_needle: Some((36, 77)),
        ..Default::default()
    });
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("RUN DETAILS"));
    assert!(text.contains("vllm-0.6"));
    assert!(text.contains("200.0 ms"), "TTFT: {text}");
    assert!(text.contains("50.0 t/s"), "gen speed: {text}");
    assert!(text.contains("46.8%"), "NIAH 36/77: {text}");
    assert!(text.contains("[Esc] Back to list"));
}

// ---- rendering: compare mode ----

#[test]
fn render_compare_shows_signed_deltas() {
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_compare(&a, &ma, &b, &mb));
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("COMPARING RUN"));
    assert!(text.contains("-20.0%"), "TTFT delta: {text}");
    assert!(text.contains("+25.0%"), "t/s delta: {text}");
    assert!(text.contains("[Esc] Back to list"));
}

#[test]
fn render_compare_select_shows_prompt() {
    let (a, _) = run_a();
    let (b, _) = run_b();
    let mut app = App::new();
    app.history = Some(HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a, b],
        cursor: 1,
        mode: HistoryMode::Compare {
            first: 0,
            second: None,
        },
        ..Default::default()
    });
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("COMPARE"));
    assert!(text.contains("pick a second run"));
}

// ---- rendering: delete confirm ----

#[test]
fn render_delete_confirm_shows_prompt() {
    let (a, _) = run_a();
    let mut app = App::new();
    app.history = Some(HistoryState {
        db_path: PathBuf::new(),
        sessions: vec![a],
        cursor: 0,
        mode: HistoryMode::DeleteConfirm(0),
        ..Default::default()
    });
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    assert!(text.contains("DELETE RUN?"));
    assert!(text.contains("[y] Delete"));
    assert!(text.contains("[n/Esc] Cancel"));
}

// ---- rendering: color coding ----

#[test]
fn gains_render_green_and_regressions_red() {
    // Gains: v0.7 vs v0.6.
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_compare(&a, &ma, &b, &mb));
    let buf = render_history(&app, W, H);
    let text = buf_text(&buf);
    let gain = cell_at(&buf, &text, "-20.0%").expect("TTFT gain delta rendered");
    assert_eq!(gain.fg, Theme::Cyberpunk.success(), "TTFT improvement must be mint");

    // Regressions: b3300 vs v0.6.
    let (c, mc) = run_c();
    let mut app2 = App::new();
    app2.history = Some(history_compare(&a, &ma, &c, &mc));
    let buf2 = render_history(&app2, W, H);
    let text2 = buf_text(&buf2);
    let reg = cell_at(&buf2, &text2, "+20.0%").expect("TTFT regression delta rendered");
    assert_eq!(reg.fg, Theme::Cyberpunk.danger(), "TTFT regression must be hot-pink");
}

// ---- small terminal ----

#[test]
fn small_terminal_renders_without_panic() {
    let (a, ma) = run_a();
    let (b, mb) = run_b();
    let mut app = App::new();
    app.history = Some(history_compare(&a, &ma, &b, &mb));
    for (w, h) in [(80, 24), (40, 10)] {
        let buf = render_history(&app, w, h);
        assert!(!buf.content().is_empty());
    }
}

// ---- summary aggregation ----

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
