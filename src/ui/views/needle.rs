//! View 3 — Context Needle Matrix (NIAH, blueprint §6): an N×M grid of
//! context token sizes (2k…128k) × needle depths (0%…100% in 10% steps),
//! color-coded: green = accurate + nominal prefill, yellow = accurate +
//! throttled prefill, red = retrieval failed / hallucinated.
//!
//! The grid renders from the lock-free [`NiahSlot`] (Chunk 15): the `n`
//! key spawns a background matrix run that *publishes* the result, and
//! this view only ever *reads* — the render path never blocks and never
//! touches the timing path (measurement-isolation invariant,
//! blueprint §4).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::engines::capability::{NiahCellState, NIAH_DEPTHS, NIAH_SIZES};
use crate::ui::app::App;
use crate::ui::theme::{palette, style};

/// Context token sizes, 2k…128k (blueprint §5, Engine C1) — labels for the
/// grid rows (the engine's `NIAH_SIZES` are the values).
const SIZE_LABELS: [&str; 7] = ["2k", "4k", "8k", "16k", "32k", "64k", "128k"];
/// Needle depths, 0%…100% in 10% steps (the engine's `NIAH_DEPTHS`).
const DEPTHS: [u8; 11] = NIAH_DEPTHS;

/// Render the NIAH view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(area);
    render_grid(chunks[0], app, f);
    render_legend(chunks[1], app, f);
}

/// N×M grid; each cell shows its state glyph (green `●` = nominal,
/// yellow `●` = throttled, red `✗` = failed) or `···` (dimmed) until the
/// matrix has been run.
fn render_grid(area: Rect, app: &App, f: &mut Frame) {
    // Lock-free read of the published result (the runner's write seam is
    // the `n` key, never this path).
    let result = &*app.niah.load();

    let mut header: Vec<Cell> = vec![Cell::from("CTX")];
    for &d in &DEPTHS {
        header.push(Cell::from(format!("{d:>3}%")));
    }
    let mut rows: Vec<Row> = vec![Row::new(header).style(style::muted_title())];

    let mut widths = vec![Constraint::Percentage(12)];
    widths.extend(std::iter::repeat_n(Constraint::Percentage(8), DEPTHS.len()));

    for (i, label) in SIZE_LABELS.iter().enumerate() {
        let mut cells = vec![Cell::from(*label).style(style::value())];
        for &depth in &DEPTHS {
            let (glyph, st) = match result.as_ref().and_then(|r| r.cell(NIAH_SIZES[i], depth)) {
                Some(cell) => match cell.state {
                    NiahCellState::Nominal => ("●", Style::default().fg(palette::OK)),
                    NiahCellState::Throttled => ("●", Style::default().fg(palette::WARN)),
                    NiahCellState::Failed => ("✗", Style::default().fg(palette::ERR)),
                },
                None => ("···", Style::default().fg(palette::MUTED)),
            };
            cells.push(Cell::from(glyph).style(st));
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

/// Legend + run status / accuracy note.
fn render_legend(area: Rect, app: &App, f: &mut Frame) {
    let slot = &app.niah;
    let status = if slot.is_running() {
        "Running NIAH matrix… (one stream per size × depth cell)".to_string()
    } else if let Some(r) = &*slot.load() {
        format!(
            "Accuracy: {} · {} sizes × {} depths · [N] re-runs the matrix",
            r.accuracy_label(),
            r.sizes.len(),
            r.depths.len()
        )
    } else {
        "[N] queues a new NIAH test (7 sizes × 11 depths)".to_string()
    };
    let lines = Text::from(vec![
        Line::from(vec![
            Span::styled("● ", Style::default().fg(palette::OK)),
            Span::styled("accurate + nominal prefill    ", style::label()),
            Span::styled("● ", Style::default().fg(palette::WARN)),
            Span::styled("accurate + throttled prefill    ", style::label()),
            Span::styled("✗ ", Style::default().fg(palette::ERR)),
            Span::styled("retrieval failed / hallucinated", style::label()),
        ]),
        Line::from(Span::styled(status, style::footer())),
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
