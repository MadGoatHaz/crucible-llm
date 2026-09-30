//! The five dashboard views (blueprint §6) plus the pre-dashboard
//! **Setup** phase (a full-screen takeover, not a tab).
//!
//! Each view is a pure function of `&App` state: it renders into `area`
//! and never mutates anything, so a dropped frame can never perturb the
//! measurement path.

use ratatui::layout::Rect;
use ratatui::text::{Line, Span};
use ratatui::Frame;

use crate::engines::sequence::Engine;
use crate::ui::app::{App, View};
use crate::ui::theme::style;

pub mod concurrency;
pub mod config;
pub mod history;
pub mod live;
pub mod needle;
pub mod setup;

/// Dispatch the current view into `area`.
pub fn render_current(area: Rect, app: &App, f: &mut Frame) {
    match app.view {
        View::Live => live::render(area, app, f),
        View::Concurrency => concurrency::render(area, app, f),
        View::Needle => needle::render(area, app, f),
        View::History => history::render(area, app, f),
        View::Config => config::render(area, app, f),
    }
}

/// The dimmed `ℹ` info lines for a multi-line help / description text
/// (the first line carries the `ℹ` marker, continuation lines are
/// indented two spaces).
pub fn info_lines(text: &str) -> Vec<Line<'static>> {
    text.lines()
        .enumerate()
        .map(|(i, l)| {
            Line::from(Span::styled(
                if i == 0 {
                    format!("ℹ {l}")
                } else {
                    format!("  {l}")
                },
                style::info(),
            ))
        })
        .collect()
}

/// The dimmed `ℹ` info lines for one engine's
/// [`description`](Engine::description) (the setup / config engine
/// selection shows the focused engine's note; the first line carries the
/// `ℹ` marker, continuation lines are indented).
pub fn engine_info_lines(engine: Engine) -> Vec<Line<'static>> {
    info_lines(engine.description())
}
