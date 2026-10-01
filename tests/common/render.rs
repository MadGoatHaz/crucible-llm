//! `ratatui::TestBackend` render helpers: draw one view into an
//! off-screen terminal and read it back as flat text or a [`Buffer`] —
//! the harness every view test used to re-implement locally.

use ratatui::backend::TestBackend;
use ratatui::buffer::Buffer;
use ratatui::layout::Rect;
use ratatui::Terminal;

use crucible_llm::ui::app::App;

/// Render `app`'s current view at `w`x`h` and return the flat buffer text
/// (all cells concatenated, row by row).
pub fn render_text(
    render: impl FnOnce(Rect, &App, &mut ratatui::Frame),
    app: &App,
    w: u16,
    h: u16,
) -> String {
    render_buffer(render, app, w, h)
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

/// Render `app`'s current view at `w`x`h` and return the resulting
/// [`Buffer`] (for color / symbol assertions).
pub fn render_buffer(
    render: impl FnOnce(Rect, &App, &mut ratatui::Frame),
    app: &App,
    w: u16,
    h: u16,
) -> Buffer {
    let backend = TestBackend::new(w, h);
    let mut terminal = Terminal::new(backend).expect("TestBackend terminal");
    terminal
        .draw(|f| render(f.area(), app, f))
        .expect("render frame");
    terminal.backend().buffer().clone()
}
