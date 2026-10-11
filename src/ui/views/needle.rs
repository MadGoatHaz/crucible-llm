//! View 3 — Context Needle Matrix (NIAH, blueprint §6): an N×M grid of
//! context token sizes (2k…128k) × needle depths (0%…100% in 10% steps),
//! color-coded: green = accurate + nominal prefill, yellow = accurate +
//! throttled prefill, red = retrieval failed / hallucinated, and (Option C
//! hybrid reasoning) blue `◌` = truncated by reasoning — the stream completed
//! but the model spent its whole budget on chain-of-thought (an inconclusive
//! (budget) outcome, distinct from a retrieval failure).
//!
//! The grid renders from the lock-free [`NiahSlot`] (Chunk 15): the `n`
//! key spawns a background matrix run that *publishes* the result, and
//! this view only ever *reads* — the render path never blocks and never
//! touches the timing path (measurement-isolation invariant,
//! blueprint §4).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Cell, Paragraph, Row, Table, Wrap};
use ratatui::Frame;

use crate::engines::capability::{context_size_label, NiahCellState, NIAH_DEPTHS, NIAH_SIZES};
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, style, Theme};

/// Needle depths, 0%…100% in 10% steps (the engine's `NIAH_DEPTHS`).
const DEPTHS: [u8; 11] = NIAH_DEPTHS;

/// Render the NIAH view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(75), Constraint::Percentage(25)])
        .split(area);
    render_grid(chunks[0], app, th, f);
    render_legend(chunks[1], app, th, f);
}

/// N×M grid; each cell shows its state glyph (green `●` = nominal,
/// yellow `●` = throttled, red `✗` = failed, blue `◌` = truncated by
/// reasoning) or `···` (dimmed) until the matrix has been run.
fn render_grid(area: Rect, app: &App, th: Theme, f: &mut Frame) {
    // Lock-free read of the published result (the runner's write seam is
    // the `n` key, never this path).
    let result = &*app.niah.load();

    let mut header: Vec<Cell> = vec![Cell::from("SIZE")];
    for &d in &DEPTHS {
        header.push(Cell::from(format!("{d:>3}%")));
    }
    header.push(Cell::from("PASS %"));
    let mut rows: Vec<Row> = vec![Row::new(header).style(style::muted_title(th))];

    // The SIZE column is a fixed length wide enough for the longest
    // qualified label (`128k (approx.)` = 14 chars + margin); the depth
    // columns and the pass-rate column keep their percentages.
    let mut widths = vec![Constraint::Length(16)];
    widths.extend(std::iter::repeat_n(Constraint::Percentage(6), DEPTHS.len()));
    widths.push(Constraint::Percentage(11));

    // Issue #4: the row labels come from the shared `context_size_label`
    // (the same fn the headless run log and the `--json` `niah` section
    // use) — with no `--tokenizer` the `chars/4` sizes read
    // `128k (approx.)`, with one they read plain `128k`.
    let approx = app.niah_approximate();
    for (i, _) in NIAH_SIZES.iter().enumerate() {
        let label = context_size_label(NIAH_SIZES[i], approx);
        let mut cells = vec![Cell::from(label).style(style::value(th))];
        for &depth in &DEPTHS {
            let (glyph, st) = match result.as_ref().and_then(|r| r.cell(NIAH_SIZES[i], depth)) {
                // Option C: a truncated-by-reasoning cell (all-CoT /
                // zero-content) renders as a distinct 4th marker — it is NOT a
                // retrieval failure (red `✗`), it's an inconclusive (budget)
                // over-thinker. Checked *before* the prefill state, whose
                // `Failed` arm (a truncated cell's `retrieved == false`) would
                // otherwise paint it red.
                Some(cell) if cell.truncated_by_reasoning => {
                    ("◌", Style::default().fg(th.info()))
                }
                Some(cell) => match cell.state {
                    NiahCellState::Nominal => ("●", Style::default().fg(th.success())),
                    NiahCellState::Throttled => ("●", Style::default().fg(th.warn())),
                    NiahCellState::Failed => ("✗", Style::default().fg(th.danger())),
                },
                None => ("···", Style::default().fg(th.dim())),
            };
            cells.push(Cell::from(glyph).style(st));
        }
        // The per-size pass rate, color-coded by the ≥80 / 50–79 / <50 bar
        // (`--` until the row has been run).
        let (pr_text, pr_style) = match result
            .as_ref()
            .and_then(|r| r.size_pass_rate(NIAH_SIZES[i]))
        {
            Some(p) => (
                fmt::format_pct(p),
                Style::default().fg(pass_rate_color(th, p)),
            ),
            None => ("--".to_string(), Style::default().fg(th.dim())),
        };
        cells.push(Cell::from(pr_text).style(pr_style));
        rows.push(Row::new(cells));
    }

    f.render_widget(
        Table::new(rows, widths).block(theme::block(
            theme::panel_title(th, "NEEDLE-IN-A-HAYSTACK MATRIX (context size × depth)"),
            style::border(th),
        )),
        area,
    );
}

/// Pass-rate color: green (≥80%), yellow (50–79%), red (<50%).
fn pass_rate_color(th: Theme, p: f64) -> Color {
    if p >= 80.0 {
        th.success()
    } else if p >= 50.0 {
        th.warn()
    } else {
        th.danger()
    }
}

/// Legend + run status / accuracy note (state-aware): locked while a
/// benchmark sequence runs (NIAH is Engine C1 in that queue), the live
/// request count while a standalone run is in flight, and the `[N]`
/// confirmation hint when idle.
fn render_legend(area: Rect, app: &App, th: Theme, f: &mut Frame) {
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
            Span::styled("● ", Style::default().fg(th.success())),
            Span::styled("accurate + nominal prefill    ", style::label(th)),
            Span::styled("● ", Style::default().fg(th.warn())),
            Span::styled("accurate + throttled prefill    ", style::label(th)),
            Span::styled("✗ ", Style::default().fg(th.danger())),
            Span::styled("retrieval failed / hallucinated", style::label(th)),
        ]),
        // Option C: the 4th marker — an over-thinker that spent its whole
        // budget on chain-of-thought (all-CoT / zero-content). It is an
        // inconclusive (budget) outcome, NOT a retrieval failure; the
        // truncated count is also broken out in the `Accuracy:` status line.
        Line::from(vec![
            Span::styled("◌ ", Style::default().fg(th.info())),
            Span::styled("truncated by reasoning (over-thought, no answer)", style::label(th)),
        ]),
        Line::from(Span::styled(status, style::footer(th))),
    ];
    // Empty state: no result yet and nothing in flight → say what will
    // appear and when.
    if slot.load().is_none() && !slot.is_running() && !app.seq.is_running() {
        lines.push(Line::from(Span::styled(
            "NIAH results will appear here after Engine C1 completes.",
            style::info(th),
        )));
    }
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("ℹ {NIAH_INFO}"),
        style::info(th),
    )));
    // The practical interpretation: the "reliable up to ~Xk" threshold
    // computed from the actual grid (shown once a result exists).
    if let Some(r) = &*slot.load() {
        lines.push(Line::raw(""));
        lines.push(Line::from(Span::styled(
            format!("ℹ {}", r.interpretation()),
            style::info(th),
        )));
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(theme::block(
                theme::panel_title(th, "LEGEND"),
                style::border(th),
            ))
            .wrap(Wrap { trim: true }),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The (symbol, foreground-color) pairs present in a rendered buffer
    /// (skips blank cells) — for asserting a specific glyph renders in a
    /// specific color.
    fn buf_colors(buf: &ratatui::buffer::Buffer) -> Vec<(String, Color)> {
        buf.content()
            .iter()
            .filter(|c| c.symbol() != " ")
            .map(|c| (c.symbol().to_string(), c.fg))
            .collect()
    }

    fn buf_text(buf: &ratatui::buffer::Buffer) -> String {
        buf.content().iter().map(|c| c.symbol()).collect()
    }

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

    #[test]
    fn grid_shows_pass_rate_column_and_interpretation() {
        use crate::engines::capability::{NiahCellState, NiahResult};
        let mut app = crate::ui::app::App::new();
        app.view = crate::ui::app::View::Needle;

        // 2k: both depths retrieved (100%); 4k: one of two (50%).
        let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
        for c in &mut r.cells[0..2] {
            c.target_tokens = 2000;
            c.retrieved = true;
            c.ttft_s = 0.1;
            c.state = NiahCellState::Nominal;
        }
        r.cells[2].target_tokens = 4000;
        r.cells[2].retrieved = true;
        r.cells[2].ttft_s = 0.2;
        r.cells[2].state = NiahCellState::Nominal;
        r.cells[3].target_tokens = 4000;
        r.cells[3].retrieved = false;
        r.cells[3].state = NiahCellState::Failed;
        app.niah.store(r);

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
        // The PASS % header and the per-row percentages.
        assert!(text.contains("PASS %"), "{text}");
        assert!(text.contains("100.0%"), "2k row is 100%: {text}");
        assert!(text.contains("50.0%"), "4k row is 50%: {text}");
        // The practical interpretation line names the reliable threshold.
        assert!(text.contains("Reliable retrieval up to"), "{text}");
    }

    #[test]
    fn grid_renders_truncated_by_reasoning_as_a_distinct_marker() {
        use crate::engines::capability::NiahResult;
        let mut app = crate::ui::app::App::new();
        app.view = crate::ui::app::View::Needle;

        // 2k row: one clean pass (nominal) and one truncated-by-reasoning cell
        // (all-CoT / zero-content — NOT a retrieval failure).
        let mut r = NiahResult::new(vec![2000], vec![0, 100]);
        r.cells[0].target_tokens = 2000;
        r.cells[0].retrieved = true;
        r.cells[0].ttft_s = 0.1;
        r.cells[0].state = NiahCellState::Nominal;
        r.cells[1].target_tokens = 2000;
        r.cells[1].retrieved = false;
        r.cells[1].truncated_by_reasoning = true;
        r.cells[1].ttft_s = 0.1;
        // Its prefill state is `Failed` (retrieved == false) — but the grid
        // must render it as the distinct truncated marker, not red `✗`.
        r.cells[1].state = NiahCellState::Failed;
        app.niah.store(r);

        let backend = ratatui::backend::TestBackend::new(120, 40);
        let mut terminal = ratatui::Terminal::new(backend).expect("TestBackend terminal");
        terminal
            .draw(|f| render(f.area(), &app, f))
            .expect("render frame");
        let buf = terminal.backend().buffer().clone();

        // The truncated cell renders the blue `◌` marker (th.info()), NOT the
        // red `✗` failure glyph.
        assert!(
            buf_colors(&buf)
                .iter()
                .any(|(s, c)| s == "◌" && *c == Theme::Cyberpunk.info()),
            "a truncated-by-reasoning cell must render the ◌ marker"
        );
        // The clean pass still renders the green `●`.
        assert!(
            buf_colors(&buf)
                .iter()
                .any(|(s, c)| s == "●" && *c == Theme::Cyberpunk.success()),
            "a nominal pass cell must render"
        );
        // The legend explains the 4th marker.
        assert!(buf_text(&buf).contains("truncated by reasoning"));
        // The panel's Accuracy line breaks out the truncated count (Option C).
        assert!(buf_text(&buf).contains("1 truncated-by-reasoning"));
    }
}
