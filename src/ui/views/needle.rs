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
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap};
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

/// Legend + run status / accuracy note (state-aware): locked while a
/// benchmark sequence runs (NIAH is Engine C1 in that queue), the live
/// request count while a standalone run is in flight, and the `[N]`
/// confirmation hint when idle.
fn render_legend(area: Rect, app: &App, f: &mut Frame) {
    let slot = &app.niah;
    let requests = NIAH_SIZES.len() * NIAH_DEPTHS.len();
    let status = if app.seq.is_running() {
        "Locked — NIAH runs as Engine C1 in the active benchmark sequence ([N] disabled)"
            .to_string()
    } else if slot.is_running() {
        format!(
            "Running NIAH matrix… {requests} requests ({} sizes × {} depths), one stream per cell · [Space] pauses",
            NIAH_SIZES.len(),
            NIAH_DEPTHS.len()
        )
    } else if let Some(r) = &*slot.load() {
        format!(
            "Accuracy: {} · {} sizes × {} depths · [N] re-runs the matrix (confirms)",
            r.accuracy_label(),
            r.sizes.len(),
            r.depths.len()
        )
    } else {
        format!(
            "[N] runs a new NIAH test — {requests} requests ({} sizes × {} depths); [Y] confirms",
            NIAH_SIZES.len(),
            NIAH_DEPTHS.len()
        )
    };
    // The dimmed `ℹ` note explaining what the matrix measures.
    const NIAH_INFO: &str = "Tests long-context memory: one hidden fact in a 2k–128k document. \
         Low % = the model loses track in long documents — critical for \
         RAG and document QA.";
    let mut lines: Vec<Line> = vec![
        Line::from(vec![
            Span::styled("● ", Style::default().fg(palette::OK)),
            Span::styled("accurate + nominal prefill    ", style::label()),
            Span::styled("● ", Style::default().fg(palette::WARN)),
            Span::styled("accurate + throttled prefill    ", style::label()),
            Span::styled("✗ ", Style::default().fg(palette::ERR)),
            Span::styled("retrieval failed / hallucinated", style::label()),
        ]),
        Line::from(Span::styled(status, style::footer())),
    ];
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("ℹ {NIAH_INFO}"),
        style::info(),
    )));
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title("LEGEND"),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legend_shows_the_long_context_info_note() {
        let app = crate::ui::app::App::new();
        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).expect("TestBackend terminal");
        terminal
            .draw(|f| render(f.area(), &app, f))
            .expect("render frame");
        let text: String = terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect();
        assert!(text.contains("LEGEND"), "{text}");
        assert!(text.contains('ℹ'), "long-context info note: {text}");
        assert!(text.contains("long-context memory"), "{text}");
    }
}
