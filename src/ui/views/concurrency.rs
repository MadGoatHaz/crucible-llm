//! View 2 — Concurrency & Saturation matrix (blueprint §6): X = concurrency
//! (1…128), Y = latency vs throughput, with the knee / optimal operational
//! envelope highlighted.
//!
//! Placeholder until the sweep engine + knee detection land (Chunk 10/11).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::style;

/// Default concurrency ladder (blueprint §5, Engine B).
const LADDER: [usize; 7] = [1, 2, 4, 8, 16, 32, 64];

/// Render the Concurrency view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    render_matrix(chunks[0], app, f);
    render_envelope(chunks[1], app, f);
}

/// Sweep matrix: one row per ladder step; `--` until a sweep has run.
fn render_matrix(area: Rect, app: &App, f: &mut Frame) {
    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("CONC"),
        Cell::from("AGG T/S"),
        Cell::from("P90 TPOT"),
        Cell::from("STATE"),
    ])
    .style(style::muted_title())];
    for &level in &LADDER {
        let is_target = level == app.concurrency_target;
        rows.push(Row::new(vec![
            Cell::from(level.to_string()).style(
                if is_target {
                    style::value_ok()
                } else {
                    style::label()
                },
            ),
            Cell::from("--"),
            Cell::from("--"),
            Cell::from(if is_target {
                "current target"
            } else {
                "not run"
            })
            .style(if is_target {
                style::value_ok()
            } else {
                style::footer()
            }),
        ]));
    }
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ],
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("CONCURRENCY SWEEP (Aggregate t/s vs p90 TPOT)"),
        ),
        area,
    );
}

/// Optimal operational envelope: highlighted once knee detection runs.
fn render_envelope(area: Rect, app: &App, f: &mut Frame) {
    let lines = Text::from(vec![
        Line::from(vec![
            Span::styled("Recommended sweet spot: ", style::label()),
            Span::styled(
                format!("{} streams", app.concurrency_target),
                style::highlight(),
            ),
            Span::styled("  (step with [+])", style::footer()),
        ]),
        Line::from(Span::styled(
            "Run a sweep to detect the saturation knee — the transition from memory-bandwidth-bound to compute-bound (planned: Chunk 11).",
            style::footer(),
        )),
    ]);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::active_border())
                .title("OPTIMAL OPERATIONAL ENVELOPE"),
        ),
        area,
    );
}
