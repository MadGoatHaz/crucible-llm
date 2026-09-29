//! View 4 — Historical Comparison & Regression Diffing (blueprint §6):
//! side-by-side comparison of two stored runs with signed, color-coded
//! delta metrics (gains green, regressions red).
//!
//! Placeholder until storage (Chunk 12) provides stored sessions.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::style;

/// Render the History Diff view into `area`.
pub fn render(area: Rect, _app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    render_runs(chunks[0], f);
    render_delta(chunks[1], f);
}

/// Side-by-side run panels.
fn render_runs(area: Rect, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    render_run_panel(cols[0], "RUN A — no stored run selected", f);
    render_run_panel(cols[1], "RUN B — no stored run selected", f);
}

fn render_run_panel(area: Rect, title: &str, f: &mut Frame) {
    let metrics = ["TTFT", "Tokens/s", "MTP rate", "J/token"];
    let mut rows: Vec<Row> =
        vec![Row::new(vec![Cell::from("METRIC"), Cell::from("VALUE")]).style(style::muted_title())];
    for &metric in &metrics {
        rows.push(Row::new(vec![
            Cell::from(metric).style(style::label()),
            Cell::from("N/A").style(style::footer()),
        ]));
    }
    f.render_widget(
        Table::new(
            rows,
            [Constraint::Percentage(50), Constraint::Percentage(50)],
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title(title),
        ),
        area,
    );
}

/// Delta panel: % change in TTFT, tokens/s, MTP rate.
fn render_delta(area: Rect, f: &mut Frame) {
    let lines = Text::from(vec![
        Line::from(vec![
            Span::styled("TTFT   ", style::label()),
            Span::styled("Δ --", style::footer()),
            Span::styled("    Tokens/s   ", style::label()),
            Span::styled("Δ --", style::footer()),
            Span::styled("    MTP rate   ", style::label()),
            Span::styled("Δ --", style::footer()),
        ]),
        Line::from(Span::styled(
            "Select two stored sessions to diff (e.g. vLLM v0.6.x vs v0.7.x). Gains render green, regressions red. Storage lands in Chunk 12.",
            style::footer(),
        )),
    ]);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("DELTA (signed % change)"),
        ),
        area,
    );
}
