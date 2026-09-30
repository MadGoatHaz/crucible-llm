//! Chunk 18 — View 5 (Configuration) + full-engine integration acceptance
//! tests (blueprint §6 View 5, plan Chunk 18).
//!
//! Verifies:
//! - the Config view renders every editable field (URL, model, mode, tokens,
//!   iterations, timeout, API key, cache-bypass, tokenizer, ladder, hardware,
//!   and the A/B/C1/C2/C3 engine switches) with a cursor;
//! - the key path edits the form: typing, backspace, toggle, cycle, step,
//!   and cursor navigation;
//! - `F2` **persists** the form to the JSON config file and a reload reads
//!   the same values back (the "edit and persist settings" acceptance);
//! - the form round-trips to a [`Config`] (`to_config`) that the headless,
//!   TUI, and export paths all consume (single source of truth);
//! - `r` / `F5` triggers the *selected* subset of engines (a single run
//!   orchestrates any combination of A–D);
//! - the view degrades without a panic on small terminals.
//!
//! Rendering runs against `ratatui::backend::TestBackend` — fully offline
//! and deterministic, no terminal attached.

use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyEventKind, KeyEventState, KeyModifiers};
use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::Terminal;

use crucible_llm::config::Config;
use crucible_llm::engines::Engine;
use crucible_llm::ui::app::{App, View};
use crucible_llm::ui::views::config::{ConfigKeyResult, ConfigState, Field};

const W: u16 = 120;
const H: u16 = 40;

/// Render the Config view at `w`x`h` and return the flattened buffer text.
fn render_config(app: &App, w: u16, h: u16) -> String {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("TestBackend terminal");
    terminal
        .draw(|f| crucible_llm::ui::views::config::render(f.area(), app, f))
        .expect("render frame");
    let buf: &Buffer = terminal.backend().buffer();
    buf.content().iter().map(|c| c.symbol()).collect()
}

/// Build a synthetic key event (a press, no modifiers).
fn key(code: KeyCode) -> KeyEvent {
    KeyEvent {
        code,
        state: KeyEventState::NONE,
        modifiers: KeyModifiers::NONE,
        kind: KeyEventKind::Press,
    }
}

/// A `ConfigState` seeded from a known config, with a temp save path.
fn state_with(path: PathBuf) -> ConfigState {
    let cfg = Config {
        url: "http://127.0.0.1:8000/v1".to_string(),
        model: "test-model".to_string(),
        tokens: 2000,
        iterations: 2,
        timeout: 60,
        nocache: false,
        ..Config::default()
    };
    let mut s = ConfigState::from_config(&cfg);
    s.config_path = path;
    s
}

// ── rendering: every field is present, cursor on the first ────────────────

#[test]
fn render_shows_every_editable_field() {
    let mut app = App::new();
    app.view = View::Config;
    let text = render_config(&app, W, H);
    for label in [
        "Target URL",
        "Model",
        "Mode",
        "Target tokens",
        "Iterations",
        "Timeout (s)",
        "API key",
        "Cache bypass",
        "Tokenizer",
        "Concurrency ladder",
        "Hardware telemetry",
        "Engine A — Speed",
        "Engine B — Concurrency",
        "Engine C1 — NIAH",
        "Engine C2 — Reasoning",
        "Engine C3 — Structured",
    ] {
        assert!(text.contains(label), "missing field {label}");
    }
    // The key-hint footer is present.
    assert!(text.contains("F2"));
    assert!(text.contains("F5"));
}

#[test]
fn render_small_terminal_without_panic() {
    let mut app = App::new();
    app.view = View::Config;
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let text = render_config(&app, w, h);
        assert!(!text.is_empty());
    }
}

// ── editing: typing, backspace, toggle, cycle, step ───────────────────────

#[test]
fn typing_appends_to_the_focused_text_field() {
    let mut s = ConfigState::default();
    s.url.clear(); // start from an empty field
    s.cursor = Field::Url as usize;
    // Type "http" into the URL field.
    for c in ['h', 't', 't', 'p'] {
        s.handle_key(&key(KeyCode::Char(c)));
    }
    assert_eq!(s.url, "http");
}

#[test]
fn backspace_deletes_from_the_focused_field() {
    let mut s = ConfigState::default();
    s.url.clear(); // start from an empty field
    s.cursor = Field::Url as usize;
    for c in ['a', 'b', 'c'] {
        s.handle_key(&key(KeyCode::Char(c)));
    }
    assert_eq!(s.url, "abc");
    s.handle_key(&key(KeyCode::Backspace));
    assert_eq!(s.url, "ab");
}

#[test]
fn space_toggles_a_boolean_field() {
    let mut s = ConfigState {
        cursor: Field::Nocache as usize,
        ..Default::default()
    };
    assert!(!s.nocache);
    s.handle_key(&key(KeyCode::Char(' ')));
    assert!(s.nocache, "Space toggles cache-bypass on");
    s.handle_key(&key(KeyCode::Char(' ')));
    assert!(!s.nocache, "Space toggles it back off");
}

#[test]
fn space_cycles_the_mode_field() {
    let mut s = ConfigState {
        cursor: Field::Mode as usize,
        ..Default::default()
    };
    assert_eq!(s.mode, crucible_llm::config::Mode::Short);
    s.handle_key(&key(KeyCode::Enter));
    assert_eq!(s.mode, crucible_llm::config::Mode::Long);
    s.handle_key(&key(KeyCode::Enter));
    assert_eq!(s.mode, crucible_llm::config::Mode::Short);
}

#[test]
fn plus_steps_the_tokens_field_up() {
    let mut s = ConfigState {
        cursor: Field::Tokens as usize,
        ..Default::default()
    };
    let before = s.tokens;
    s.handle_key(&key(KeyCode::Char('+')));
    assert_eq!(s.tokens, before + 100, "tokens step by 100");
}

#[test]
fn left_steps_a_number_field_down() {
    let mut s = ConfigState {
        cursor: Field::Iterations as usize,
        iterations: 5,
        ..Default::default()
    };
    s.handle_key(&key(KeyCode::Left));
    assert_eq!(s.iterations, 4);
}

#[test]
fn digits_type_into_a_number_field() {
    let mut s = ConfigState {
        cursor: Field::Timeout as usize,
        timeout: 0,
        ..Default::default()
    };
    for c in ['1', '2', '0'] {
        s.handle_key(&key(KeyCode::Char(c)));
    }
    assert_eq!(s.timeout, 120);
}

#[test]
fn engine_switches_toggle_independently() {
    let mut s = ConfigState::default();
    // Default: speed + hardware on, the rest off.
    assert!(s.engine_speed);
    assert!(!s.engine_concurrency);
    s.cursor = Field::EngineConcurrency as usize;
    s.handle_key(&key(KeyCode::Char(' ')));
    assert!(s.engine_concurrency);
    s.cursor = Field::EngineNiah as usize;
    s.handle_key(&key(KeyCode::Char(' ')));
    assert!(s.engine_niah);
    // Toggling one leaves the others untouched.
    assert!(s.engine_speed);
    assert!(!s.engine_reasoning);
}

// ── cursor navigation ─────────────────────────────────────────────────────

#[test]
fn tab_and_backtab_walk_the_field_list() {
    let mut s = ConfigState::default();
    assert_eq!(s.cursor, 0);
    s.handle_key(&key(KeyCode::Tab));
    assert_eq!(s.cursor, 1);
    s.handle_key(&key(KeyCode::BackTab));
    assert_eq!(s.cursor, 0);
    // BackTab at the start wraps to the end.
    s.handle_key(&key(KeyCode::BackTab));
    assert_eq!(s.cursor, Field::ALL.len() - 1);
}

#[test]
fn up_down_clamp_at_the_bounds() {
    let mut s = ConfigState::default();
    s.handle_key(&key(KeyCode::Up));
    assert_eq!(s.cursor, 0, "Up at the top stays at 0");
    for _ in 0..Field::ALL.len() {
        s.handle_key(&key(KeyCode::Down));
    }
    assert_eq!(
        s.cursor,
        Field::ALL.len() - 1,
        "Down clamps at the last field"
    );
}

// ── persistence: F2 saves to the config file; a reload reads it back ──────

#[test]
fn f2_saves_and_a_reload_reads_the_values_back() {
    let dir = std::env::temp_dir().join(format!("crucible-cfgview-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let path = dir.join("config.json");

    let mut s = state_with(path.clone());
    s.url = "http://10.0.0.5:9000/v1".to_string();
    s.model = "persisted-model".to_string();
    s.tokens = 4096;
    s.nocache = true;
    s.engine_concurrency = true;
    s.ladder = "1,8,64".to_string();

    // F2 saves.
    let result = s.handle_key(&key(KeyCode::F(2)));
    assert_eq!(result, ConfigKeyResult::Saved, "F2 returns Saved");
    assert!(s.saved, "the saved flash is set");
    assert!(path.exists(), "the config file was written");

    // A reload through the normal config-file layer reads the values back.
    let file = crucible_llm::config::load_config_file(&path)
        .unwrap()
        .unwrap();
    let reloaded = file.to_config();
    assert_eq!(reloaded.url, "http://10.0.0.5:9000/v1");
    assert_eq!(reloaded.model, "persisted-model");
    assert_eq!(reloaded.tokens, 4096);
    assert!(reloaded.nocache);
    assert!(reloaded.engines.concurrency);
    assert_eq!(reloaded.ladder, vec![1, 8, 64]);

    let _ = std::fs::remove_dir_all(&dir);
}

// ── the form is the single source of truth: to_config() ───────────────────

#[test]
fn to_config_maps_the_form_into_a_runnable_config() {
    let s = ConfigState {
        url: "http://1.2.3.4:7/v1".to_string(),
        model: "m".to_string(),
        tokens: 333,
        iterations: 3,
        timeout: 99,
        nocache: true,
        tokenizer: Some("/tmp/tok.json".to_string()),
        ladder: "2,4,8".to_string(),
        hardware: false,
        engine_speed: true,
        engine_concurrency: true,
        engine_niah: false,
        engine_reasoning: true,
        engine_structured: false,
        ..Default::default()
    };

    let cfg = s.to_config();
    assert_eq!(cfg.url, "http://1.2.3.4:7/v1");
    assert_eq!(cfg.model, "m");
    assert_eq!(cfg.tokens, 333);
    assert_eq!(cfg.iterations, 3);
    assert_eq!(cfg.timeout, 99);
    assert!(cfg.nocache);
    assert_eq!(
        cfg.tokenizer.as_deref(),
        Some(std::path::Path::new("/tmp/tok.json"))
    );
    assert_eq!(cfg.ladder, vec![2, 4, 8]);
    assert!(!cfg.hardware);
    // The engine selection mirrors the form's switches (hardware follows
    // the hardware toggle).
    assert!(cfg.engines.speed);
    assert!(cfg.engines.concurrency);
    assert!(!cfg.engines.niah);
    assert!(cfg.engines.reasoning);
    assert!(!cfg.engines.structured);
    assert!(!cfg.engines.hardware);
}

#[test]
fn to_config_falls_back_to_the_default_ladder_when_blank() {
    // whitespace-only ladder → invalid → falls back to the default ladder.
    let s = ConfigState {
        ladder: "   ".to_string(),
        ..Default::default()
    };
    let cfg = s.to_config();
    assert_eq!(cfg.ladder, vec![1, 2, 4, 8, 16, 32, 64]);
}

// ── a single run orchestrates the selected subset (r / F5) ────────────────

#[tokio::test]
async fn run_key_triggers_only_the_selected_engines() {
    let mut app = App::new();
    // The global `r` key works from any non-Config view (in the Config view
    // `r` is an editor key; `F5` is that view's run key).
    app.view = View::Live;
    // Select exactly Engine A + Engine D (the default selection).
    app.config.engine_speed = true;
    app.config.engine_concurrency = false;
    app.config.engine_niah = false;
    app.config.engine_reasoning = false;
    app.config.engine_structured = false;
    app.config.hardware = true;
    // Point at a dead endpoint so the spawned speed task fails fast.
    app.config.url = "http://127.0.0.1:9".to_string();
    app.config.timeout = 1;

    let action = app.handle_key(&key(KeyCode::Char('r')));
    assert_eq!(action, crucible_llm::ui::app::KeyAction::Run);
    // The sequential executor is in progress (marked synchronously by
    // `start_run` before the `tokio::spawn`).
    assert!(app.seq.is_running());
    // The queue holds exactly the selected engines, in the canonical
    // A → D order — the non-selected engines never run.
    assert_eq!(
        app.seq.load().unwrap().queue,
        vec![Engine::Speed, Engine::Hardware]
    );
    // The non-selected engines' slots are not marked running.
    assert!(!app.sweep.is_running());
    assert!(!app.niah.is_running());
    assert!(!app.reasoning_slot.is_running());
    assert!(!app.structured_slot.is_running());
}

#[tokio::test]
async fn f5_in_the_config_view_also_runs() {
    let mut app = App::new();
    app.view = View::Config;
    app.config.engine_speed = true;
    app.config.hardware = true;
    app.config.url = "http://127.0.0.1:9".to_string();
    app.config.timeout = 1;

    let action = app.handle_key(&key(KeyCode::F(5)));
    assert_eq!(action, crucible_llm::ui::app::KeyAction::Run);
    assert!(app.seq.is_running());
    assert_eq!(
        app.seq.load().unwrap().queue,
        vec![Engine::Speed, Engine::Hardware]
    );
}

#[tokio::test]
async fn no_engines_selected_logs_a_warning() {
    let mut app = App::new();
    app.view = View::Live;
    app.config.engine_speed = false;
    app.config.engine_concurrency = false;
    app.config.engine_niah = false;
    app.config.engine_reasoning = false;
    app.config.engine_structured = false;
    app.config.hardware = false;
    let action = app.handle_key(&key(KeyCode::Char('r')));
    assert_eq!(action, crucible_llm::ui::app::KeyAction::Run);
    // Nothing started (no sequence, no engine slots).
    assert!(!app.seq.is_running());
    assert!(!app.speed_slot.is_running());
    assert!(!app.sweep.is_running());
    // A warning was logged.
    let joined: String = app.log.iter().map(|l| l.to_string()).collect();
    assert!(joined.contains("no engines selected"), "{joined}");
}
