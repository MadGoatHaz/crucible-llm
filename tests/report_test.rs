//! Matrix Report (M3) — the TUI `Phase::Report` takeover.
//!
//! The `m` key opens the `[COMMUNITY-SUBMISSION]` report **only after a
//! completed run** (the gate: `SeqPhase::AllComplete`); the takeover
//! shows the auto fields (hardware / VRAM / model / speed) plus the
//! editable user fields (backend / quant / use case / notes / save
//! path); `c` copies via OSC 52 (the N/A-never-fail handler), `s` saves
//! to the (editable) path, and `Esc` returns to the dashboard.
//!
//! Render-level (ratatui::TestBackend via `common::render`), offline —
//! no server.

mod common;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

use common::render::render_text;
use crucible_llm::engines::{Engine, SeqPhase, SeqState};
use crucible_llm::ui::app::{App, Phase};
use crucible_llm::ui::views::report;

fn key(code: KeyCode) -> KeyEvent {
    KeyEvent::new(code, KeyModifiers::NONE)
}

/// An app with a **completed** run in the sequence slot (the `m` gate
/// reads `SeqPhase::AllComplete` lock-free).
fn app_with_completed_run() -> App {
    let mut app = App::new();
    app.config.model = "test-model".to_string();
    app.seq.store(SeqState {
        phase: SeqPhase::AllComplete,
        queue: vec![Engine::Speed],
        engine: Engine::Speed,
        progress: None,
        summary: String::new(),
        completed: vec![(Engine::Speed, "done".to_string())],
        engine_started_ms: 0,
    });
    app
}

// ── the `m` gate ─────────────────────────────────────────────────────────

#[test]
fn m_before_completion_stays_on_dashboard_and_logs_the_hint() {
    // A fresh app: no run has started (`seq.load()` → `None`).
    let mut app = App::new();
    app.handle_key(&key(KeyCode::Char('m')));
    assert_eq!(
        app.phase,
        Phase::Dashboard,
        "m must not open the report before a completed run"
    );
    let log: Vec<String> = app.log.iter().map(|l| l.to_string()).collect();
    assert!(
        log.iter().any(|l| l.contains("complete a run first")),
        "the hint is logged: {log:?}"
    );
}

#[test]
fn m_after_allcomplete_opens_the_report_and_shows_the_block() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    assert_eq!(
        app.phase,
        Phase::Report,
        "m opens the report after a completed run"
    );
    let text = render_text(report::render, &app, 120, 40);
    assert!(text.contains("[COMMUNITY-SUBMISSION]"), "{text}");
    assert!(text.contains("hardware:"), "{text}");
    assert!(text.contains("model:"), "{text}");
    // The auto model field comes from the Configuration form.
    assert!(text.contains("test-model"), "{text}");
}

// ── the takeover render ──────────────────────────────────────────────────

#[test]
fn report_render_shows_the_editable_fields_and_footer() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    let text = render_text(report::render, &app, 120, 40);
    // The four text user fields are in the block (the save path is in
    // the footer).
    for field in ["backend:", "quant:", "use_case:", "notes:"] {
        assert!(text.contains(field), "missing {field} in: {text}");
    }
    // The footer: the three actions + the current save path.
    assert!(text.contains("[C]opy"), "{text}");
    assert!(text.contains("[S]ave"), "{text}");
    assert!(text.contains("[Esc]lose"), "{text}");
    assert!(text.contains("/exports/matrix-"), "{text}");
}

#[test]
fn editing_a_user_field_updates_the_report_and_render() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    // Focus 0 is `backend`: type `vLLM`.
    for c in "vLLM".chars() {
        app.handle_key(&key(KeyCode::Char(c)));
    }
    // The buffer and the write-back report both carry the value.
    assert_eq!(app.report.backend, "vLLM");
    assert_eq!(app.report.report.backend, "vLLM");
    // The render reflects the edit.
    let text = render_text(report::render, &app, 120, 40);
    assert!(text.contains("vLLM"), "the typed backend renders: {text}");
}

#[test]
fn the_save_path_default_is_prefilled() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    let path = &app.report.save_path;
    // `data_dir()/exports/matrix-<unix-ts>.md`.
    let data_path = crucible_llm::config::data_dir();
    let data = data_path.to_string_lossy();
    assert!(
        path.starts_with(data.as_ref()),
        "save path under data_dir: {path}"
    );
    assert!(path.contains("/exports/matrix-"), "{path}");
    assert!(path.ends_with(".md"), "{path}");
}

#[test]
fn esc_closes_the_report_and_m_reopens_it() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    assert_eq!(app.phase, Phase::Report);
    app.handle_key(&key(KeyCode::Esc));
    assert_eq!(app.phase, Phase::Dashboard, "Esc closes the report");
    assert!(app.running, "Esc never quits");
    // The run state is untouched (`seq` still reads `AllComplete`), so
    // `m` re-opens the report.
    app.handle_key(&key(KeyCode::Char('m')));
    assert_eq!(app.phase, Phase::Report, "m re-opens after Esc");
}

// ── the `c` (OSC 52 copy) handler ────────────────────────────────────────

#[test]
fn copy_handler_logs_success_and_handles_err_gracefully() {
    // The handler decision, pure over the outcome (no real terminal
    // output asserted): Ok → the success log; Err → the "use [S]ave"
    // warning (the N/A-never-fail rule — never a crash).
    assert_eq!(
        App::report_copy_message(&Ok(())),
        "[report] copied to clipboard (OSC 52)"
    );
    assert_eq!(
        App::report_copy_message(&Err("unsupported".to_string())),
        "[report] clipboard unsupported by terminal — use [S]ave"
    );

    // The `c` key in the open report: in a piped `cargo test` the OSC 52
    // write to stdout succeeds (inert bytes on the pipe — the
    // clipboard.rs smoke test establishes the convention) → the
    // success log, and the app keeps running in the report.
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    app.handle_key(&key(KeyCode::Char('c')));
    let log: Vec<String> = app.log.iter().map(|l| l.to_string()).collect();
    assert!(
        log.iter()
            .any(|l| l == "[report] copied to clipboard (OSC 52)"),
        "the c key logs the copy: {log:?}"
    );
    assert!(app.running, "copy never quits the app");
    assert_eq!(app.phase, Phase::Report, "stays in the report after copy");
}

// ── focus cycling ────────────────────────────────────────────────────────

#[test]
fn tab_cycles_the_fields_and_shift_tab_wraps_back() {
    let mut app = app_with_completed_run();
    app.handle_key(&key(KeyCode::Char('m')));
    let start = app.report.focus;
    // Five Tabs walk the five fields and wrap to the start.
    for _ in 0..5 {
        app.handle_key(&key(KeyCode::Tab));
    }
    assert_eq!(app.report.focus, start, "five Tabs wrap to the start");
    // Shift+Tab wraps backward to the last field (the save path).
    app.handle_key(&KeyEvent::new(KeyCode::Tab, KeyModifiers::SHIFT));
    assert_eq!(app.report.focus, 4, "Shift+Tab wraps to the save path");
}
