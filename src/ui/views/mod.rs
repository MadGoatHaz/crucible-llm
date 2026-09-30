//! The five dashboard views (blueprint §6) plus the pre-dashboard
//! **Setup** phase (a full-screen takeover, not a tab).
//!
//! Each view is a pure function of `&App` state: it renders into `area`
//! and never mutates anything, so a dropped frame can never perturb the
//! measurement path.

use ratatui::layout::Rect;
use ratatui::Frame;

use crate::ui::app::{App, View};

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
