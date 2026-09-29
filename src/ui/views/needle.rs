//! View 3 — Context Needle Matrix (NIAH, blueprint §6): an N×M grid of
//! context token sizes (2k…128k) × needle depths (0%…100% in 10% steps),
//! color-coded: green = accurate + nominal prefill, yellow = accurate +
//! throttled prefill, red = retrieval failed / hallucinated.
//!
//! Placeholder grid (all cells unrouted) until the NIAH runner lands
//! (Chunk 15).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::{palette, style};

/// Context token sizes, 2k…128k (blueprint §5, Engine C1).
const SIZES: [&str; 7] = ["2k", "4k", "8k", "16k", "32k", "64k", "128k"];
/// Needle depths, 0%…100% in 10% steps.
const DEPTHS: [u8; 11] = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

/// Render the NIAH view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(area);
    render_grid(chunks[0], f);
    render_legend(chunks[1], app, f);
}

/// N×M grid; every cell shows `···` (dimmed) until evaluated.
fn render_grid(area: Rect, f: &mut Frame) {
    let mut header: Vec<Cell> = vec![Cell::from("CTX")];
    for &d in &DEPTHS {
        header.push(Cell::from(format!("{d:>3}%")));
    }
    let mut rows: Vec<Row> = vec![Row::new(header).style(style::muted_title())];

    let mut widths = vec![Constraint::Percentage(12)];
    widths.extend(std::iter::repeat_n(Constraint::Percentage(8), DEPTHS.len()));

    let not_run = Style::default().fg(palette::MUTED);
    for &size in &SIZES {
        let mut cells = vec![Cell::from(size).style(style::value())];
        for _ in &DEPTHS {
            cells.push(Cell::from("···").style(not_run));
        }
        rows.push(Row::new(cells));
    }

    f.render_widget(
        Table::new(rows, widths).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("NEEDLE-IN-A-HAYSTACK MATRIX (context size × depth)"),
        ),
        area,
    );
}

/// Legend + status note.
fn render_legend(area: Rect, app: &App, f: &mut Frame) {
    let lines = Text::from(vec![
        Line::from(vec![
            Span::styled("▪ ", Style::default().fg(palette::OK)),
            Span::styled("accurate + nominal prefill    ", style::label()),
            Span::styled("▪ ", Style::default().fg(palette::WARN)),
            Span::styled("accurate + throttled prefill    ", style::label()),
            Span::styled("▪ ", Style::default().fg(palette::ERR)),
            Span::styled("retrieval failed / hallucinated", style::label()),
        ]),
        Line::from(Span::styled(
            format!(
                "No NIAH runs recorded — [N] queues a new test (runner lands in Chunk 15). Concurrency target: {}",
                app.concurrency_target
            ),
            style::footer(),
        )),
    ]);
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("LEGEND"),
        ),
        area,
    );
}
