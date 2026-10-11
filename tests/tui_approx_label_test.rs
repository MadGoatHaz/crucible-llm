//! Issue #4 — TUI approximate-size labeling.
//!
//! The NIAH grid (View 3) and the Live view's Long Context capability bar
//! must carry the `(approx.)` qualifier when no `--tokenizer` was supplied
//! (the sizes are the `chars/4` estimate) and show exact labels when one is
//! — consistent with the headless run log and the `--json` `niah` section,
//! which render through the same shared `context_size_label`.
//!
//! Render-level (ratatui::TestBackend via `common::render`), offline —
//! no server.

mod common;

use std::path::PathBuf;

use crucible_llm::engines::capability::{
    NiahCell, NiahCellState, NiahEngineConfig, NiahResult, NIAH_DEPTHS, NIAH_SIZES,
};
use crucible_llm::ui::app::App;
use crucible_llm::ui::views::{live, needle};
use common::render::render_text;

/// One NIAH cell: `target_tokens` = the context size (a run cell), a
/// nominal TTFT when retrieved (built from `unrun()` so added fields keep
/// it compiling).
fn niah_cell(size: u32, retrieved: bool) -> NiahCell {
    let mut c = NiahCell::unrun();
    c.target_tokens = size;
    c.depth_percent = 50;
    c.retrieved = retrieved;
    c.ttft_s = if retrieved { 0.5 } else { 0.0 };
    c.state = if retrieved {
        NiahCellState::Nominal
    } else {
        NiahCellState::Failed
    };
    c
}

/// The test matrix: 2k + 8k retrieve (100% each), 32k fails (0%) — so the
/// reliable-up-to (≥80%) threshold is 8k.
fn test_result() -> NiahResult {
    NiahResult {
        sizes: vec![2_000, 8_000, 32_000],
        depths: vec![50],
        cells: vec![
            niah_cell(2_000, true),
            niah_cell(8_000, true),
            niah_cell(32_000, false),
        ],
    }
}

/// An app whose NIAH config snapshot carries a tokenizer path (exact
/// counts) — the shape `App::with_config` produces at the entry point.
fn app_with_tokenizer() -> App {
    let mut app = App::new();
    app.niah_config = Some(NiahEngineConfig {
        url: "http://localhost:8000/v1".to_string(),
        model: "test-model".to_string(),
        api_key: None,
        timeout: 30,
        tokenizer: Some(PathBuf::from("tests/fixtures/mini_tokenizer.json")),
        niah_contexts: NIAH_SIZES.to_vec(),
        niah_depths: NIAH_DEPTHS.to_vec(),
        niah_custom: None,
    });
    app
}

#[test]
fn without_tokenizer_the_niah_grid_and_long_context_bar_show_approx() {
    // `App::new()` supplies no tokenizer (no niah_config snapshot, empty
    // form field) → every size is the chars/4 estimate.
    let app = App::new();
    app.niah.store(test_result());

    // View 3 (the NIAH grid): every row label carries the qualifier.
    let grid = render_text(|area, app, f| needle::render(area, app, f), &app, 120, 40);
    assert!(grid.contains("2k (approx.)"), "2k row qualified: {grid}");
    assert!(grid.contains("8k (approx.)"), "8k row qualified: {grid}");
    assert!(grid.contains("32k (approx.)"), "32k row qualified: {grid}");
    assert!(
        grid.contains("128k (approx.)"),
        "largest row qualified: {grid}"
    );

    // View 1 (the Live capability bar): the reliable-up-to threshold is
    // qualified the same way.
    let live_text = render_text(|area, app, f| live::render(area, app, f), &app, 120, 40);
    assert!(
        live_text.contains("≤ 8k (approx.) reliable"),
        "long-context bar qualified: {live_text}"
    );
}

#[test]
fn with_tokenizer_the_niah_grid_and_long_context_bar_show_exact_labels() {
    let app = app_with_tokenizer();
    app.niah.store(test_result());

    // View 3: the row labels are plain sizes — no qualifier.
    let grid = render_text(|area, app, f| needle::render(area, app, f), &app, 120, 40);
    assert!(grid.contains("2k"), "2k row present: {grid}");
    assert!(grid.contains("32k"), "32k row present: {grid}");
    assert!(
        !grid.contains("(approx.)"),
        "no qualifier with a tokenizer: {grid}"
    );

    // View 1: the reliable-up-to threshold is exact.
    let live_text = render_text(|area, app, f| live::render(area, app, f), &app, 120, 40);
    assert!(
        live_text.contains("≤ 8k reliable"),
        "long-context bar exact: {live_text}"
    );
    assert!(
        !live_text.contains("(approx.)"),
        "no qualifier with a tokenizer: {live_text}"
    );
}
