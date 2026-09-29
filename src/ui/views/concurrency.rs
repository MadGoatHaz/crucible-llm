//! View 2 — Concurrency & Saturation matrix (blueprint §6): X = concurrency
//! (1…128), Y = latency vs throughput, with the knee / optimal operational
//! envelope highlighted.
//!
//! Once a [`Sweep`] (plan Chunk 10) has run, the matrix renders the real
//! per-level curve — aggregate tokens/sec and client-perceived p90 TPOT
//! pooled across every concurrent stream of the level — and highlights the
//! **saturation knee** (plan Chunk 11, [`SweepResult::detect_knee`]): the
//! level where aggregate throughput plateaus while p90 TPOT spikes (the
//! memory-bandwidth-bound → compute-bound transition), plus the
//! **Optimal Operational Envelope** ([`SweepResult::envelope`]) — the
//! recommended sweet spot just before the knee.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::engines::concurrency::{Envelope, SweepLevel, DEFAULT_LADDER};
use crate::ui::app::App;
use crate::ui::theme::style;

/// The placeholder ladder shown before a sweep has run (blueprint §5,
/// Engine B default).
const LADDER: [usize; 7] = DEFAULT_LADDER;

/// Render the Concurrency view into `area`.
///
/// The envelope (knee + sweet spot) is computed once per frame from the
/// last [`SweepResult`] and shared by the matrix and the envelope panel.
/// Reading `app.sweep` is a pure `&` read — no locks, no timing path
/// (measurement isolation, blueprint §4).
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    let envelope = app.sweep.as_deref().and_then(|r| r.envelope());
    render_matrix(chunks[0], app, f, &envelope);
    render_envelope(chunks[1], app, f, &envelope);
}

/// Sweep matrix: one row per ladder step — real aggregate t/s + p90 TPOT
/// once a sweep has run, `--` until then. The knee row is flagged red
/// (`KNEE`) and the sweet-spot row green (`SWEET`).
fn render_matrix(area: Rect, app: &App, f: &mut Frame, envelope: &Option<Envelope>) {
    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("CONC"),
        Cell::from("AGG T/S"),
        Cell::from("P90 TPOT"),
        Cell::from("STATE"),
    ])
    .style(style::muted_title())];

    match app.sweep.as_deref() {
        Some(result) if !result.levels.is_empty() => {
            for level in &result.levels {
                rows.push(sweep_row(level, envelope, app.concurrency_target));
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
                Constraint::Percentage(20),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(30),
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

/// One matrix row for a completed sweep level, flagged when it is the
/// detected knee (red) or the recommended sweet spot (green).
fn sweep_row<'a>(level: &'a SweepLevel, envelope: &'a Option<Envelope>, target: usize) -> Row<'a> {
    let is_knee = envelope
        .as_ref()
        .and_then(|e| e.knee)
        .is_some_and(|k| k.concurrency == level.concurrency);
    let is_sweet = envelope.is_some_and(|e| e.sweet_spot == level.concurrency);

    let mut state = if level.failed_streams == level.concurrency {
        "failed".to_string()
    } else if level.failed_streams > 0 {
        format!("{}/{} ok", level.completed_streams, level.concurrency)
    } else {
        "done".to_string()
    };
    if is_knee {
        state.push_str(" · KNEE");
    }
    if is_sweet {
        state.push_str(" · SWEET");
    }

    let value_style = if is_knee {
        style::value_err()
    } else if is_sweet {
        style::value_ok()
    } else if level.concurrency == target {
        style::highlight()
    } else {
        style::label()
    };
    Row::new(vec![
        Cell::from(level.concurrency.to_string()).style(value_style),
        Cell::from(format!("{:.1} t/s", level.aggregate_tps)).style(value_style),
        Cell::from(format!("{:.1} ms", level.p90_tpot_ms())).style(value_style),
        Cell::from(state).style(if is_knee {
            style::value_err()
        } else if is_sweet {
            style::value_ok()
        } else {
            style::footer()
        }),
    ])
}

/// Optimal Operational Envelope (blueprint §6 View 2): the recommended
/// sweet spot, with the detected saturation knee (and the rationale —
/// throughput plateau + p90 spike) when the curve showed a transition.
fn render_envelope(area: Rect, app: &App, f: &mut Frame, envelope: &Option<Envelope>) {
    let lines = match envelope {
        Some(env) => {
            let mut ls = vec![Line::from(vec![
                Span::styled("Recommended sweet spot: ", style::label()),
                Span::styled(
                    format!(
                        "{} streams ({} t/s @ p90 {} ms)",
                        env.sweet_spot,
                        env.aggregate_tps,
                        env.p90_tpot_ns as f64 / 1_000_000.0
                    ),
                    style::value_ok(),
                ),
            ])];
            match env.knee {
                Some(k) => ls.push(Line::from(vec![
                    Span::styled("Saturation knee: ", style::label()),
                    Span::styled(
                        format!(
                            "{} streams — throughput plateaus ({} t/s) while p90 TPOT spikes to {} ms",
                            k.concurrency,
                            k.aggregate_tps,
                            k.p90_tpot_ns as f64 / 1_000_000.0
                        ),
                        style::value_err(),
                    ),
                ])),
                None => ls.push(Line::from(Span::styled(
                    "No saturation knee detected — running at the peak-throughput point.",
                    style::footer(),
                ))),
            }
            Text::from(ls)
        }
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
                "Run a sweep to detect the saturation knee — the transition from memory-bandwidth-bound to compute-bound.",
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
