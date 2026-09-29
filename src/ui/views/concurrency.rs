//! View 2 — Concurrency & Saturation matrix (blueprint §6): X = concurrency
//! (1…128), Y = latency vs throughput, with the knee / optimal operational
//! envelope highlighted.
//!
//! Once a [`Sweep`] (plan Chunk 10) has run, the matrix renders the real
//! per-level curve — aggregate tokens/sec and client-perceived p90 TPOT
//! pooled across every concurrent stream of the level — and the envelope
//! panel reports the peak-throughput point. Knee-point detection that
//! narrows that to the recommended sweet spot lands in Chunk 11.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::engines::concurrency::{SweepLevel, DEFAULT_LADDER};
use crate::ui::app::App;
use crate::ui::theme::style;

/// The placeholder ladder shown before a sweep has run (blueprint §5,
/// Engine B default).
const LADDER: [usize; 7] = DEFAULT_LADDER;

/// Render the Concurrency view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    render_matrix(chunks[0], app, f);
    render_envelope(chunks[1], app, f);
}

/// Sweep matrix: one row per ladder step — real aggregate t/s + p90 TPOT
/// once a sweep has run, `--` until then.
fn render_matrix(area: Rect, app: &App, f: &mut Frame) {
    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("CONC"),
        Cell::from("AGG T/S"),
        Cell::from("P90 TPOT"),
        Cell::from("STATE"),
    ])
    .style(style::muted_title())];

    match app.sweep.as_deref() {
        Some(result) if !result.levels.is_empty() => {
            let peak = result.peak_throughput().map(|l| l.concurrency).unwrap_or(0);
            for level in &result.levels {
                rows.push(sweep_row(level, peak, app.concurrency_target));
            }
        }
        _ => {
            for &level in &LADDER {
                let is_target = level == app.concurrency_target;
                rows.push(Row::new(vec![
                    Cell::from(level.to_string()).style(if is_target {
                        style::value_ok()
                    } else {
                        style::label()
                    }),
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
        }
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

/// One matrix row for a completed sweep level.
fn sweep_row(level: &SweepLevel, peak: usize, target: usize) -> Row<'_> {
    let state = if level.failed_streams == level.concurrency {
        "failed".to_string()
    } else if level.failed_streams > 0 {
        format!("{}/{} ok", level.completed_streams, level.concurrency)
    } else {
        "done".to_string()
    };
    let is_peak = level.concurrency == peak;
    let is_target = level.concurrency == target;
    let value_style = if is_peak {
        style::value_ok()
    } else if is_target {
        style::highlight()
    } else {
        style::label()
    };
    Row::new(vec![
        Cell::from(level.concurrency.to_string()).style(value_style),
        Cell::from(format!("{:.1} t/s", level.aggregate_tps)).style(value_style),
        Cell::from(format!("{:.1} ms", level.p90_tpot_ms())).style(value_style),
        Cell::from(state).style(if is_peak {
            style::value_ok()
        } else {
            style::footer()
        }),
    ])
}

/// Optimal operational envelope: the peak-throughput point once a sweep has
/// run; knee detection (Chunk 11) refines it into the recommended sweet spot.
fn render_envelope(area: Rect, app: &App, f: &mut Frame) {
    let lines = match app.sweep.as_deref().and_then(|r| r.peak_throughput()) {
        Some(peak) => Text::from(vec![
            Line::from(vec![
                Span::styled("Peak aggregate: ", style::label()),
                Span::styled(
                    format!(
                        "{:.1} t/s at {} streams (p90 TPOT {:.1} ms)",
                        peak.aggregate_tps,
                        peak.concurrency,
                        peak.p90_tpot_ms()
                    ),
                    style::highlight(),
                ),
            ]),
            Line::from(Span::styled(
                "Saturation knee detection (the recommended sweet spot) lands in Chunk 11 — step the target with [+] while it refines the envelope.",
                style::footer(),
            )),
        ]),
        None => Text::from(vec![
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
        ]),
    };
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
