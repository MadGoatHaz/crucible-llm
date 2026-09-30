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
//!
//! The top panel renders the sweep as a block-based throughput-vs-
//! concurrency curve: one vertical bar per ladder level (x on a
//! `log2(concurrency)` scale, y ∝ aggregate t/s), with the sweet spot
//! capped by a green `●` and the saturation knee by a red `▲`.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::engines::concurrency::{Envelope, SweepLevel, DEFAULT_LADDER};
use crate::ui::app::App;
use crate::ui::theme::{palette, style};

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
        .constraints([
            Constraint::Percentage(40), // throughput vs concurrency curve
            Constraint::Percentage(40), // sweep matrix
            Constraint::Percentage(20), // optimal operational envelope
        ])
        .split(area);
    // Chunk 18: the sweep result is a lock-free `ResultSlot` (a background
    // `r`-key sweep publishes it); reading is a pure lock-free load.
    let envelope = app
        .sweep
        .load()
        .as_ref()
        .as_ref()
        .and_then(|r| r.envelope());
    render_curve(chunks[0], app, f, &envelope);
    render_matrix(chunks[1], app, f, &envelope);
    render_envelope(chunks[2], app, f, &envelope);
}

/// Top panel: aggregate throughput vs concurrency — one block bar per
/// sweep level (x = `log2(concurrency)`, y ∝ aggregate t/s), with the
/// sweet spot (green `●`) and the saturation knee (red `▲`) marked.
fn render_curve(area: Rect, app: &App, f: &mut Frame, envelope: &Option<Envelope>) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("THROUGHPUT VS CONCURRENCY (t/s)  ● sweet spot  ▲ knee");

    let sweep = app.sweep.load();
    let Some(result) = sweep.as_ref().as_ref().filter(|r| !r.levels.is_empty()) else {
        f.render_widget(
            Paragraph::new(Line::from(Span::styled(
                "Run a sweep (Engine B in the Config view, or [r]) to plot the curve.",
                style::footer(),
            )))
            .block(block),
            area,
        );
        return;
    };

    let lines = build_curve_lines(
        &result.levels,
        area.width.saturating_sub(2),
        area.height.saturating_sub(2),
        envelope,
    );
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// One plotted point on the throughput-vs-concurrency curve.
struct CurvePoint {
    x: u16,
    y: u16,
    color: Color,
    marker: char,
    label: String,
}

/// Build the block-based curve as lines of styled single-character
/// spans.
///
/// Pure function over the sweep levels (unit-testable, no terminal): one
/// vertical `│` bar per level from the baseline up to its height, capped
/// by the level's marker — green `●` (sweet spot), red `▲` (knee), cyan
/// `•` (other). x is positioned on a `log2(concurrency)` scale (the
/// ladder is a power-of-two series), y is proportional to aggregate
/// t/s, and the bottom row carries the concurrency x-labels.
pub(crate) fn build_curve_lines(
    levels: &[SweepLevel],
    w: u16,
    h: u16,
    envelope: &Option<Envelope>,
) -> Vec<Line<'static>> {
    if w < 6 || h < 4 {
        return vec![Line::from("plot area too small")];
    }
    let plot_h = h - 1; // bottom row reserved for x labels
    let mut sorted: Vec<&SweepLevel> = levels.iter().collect();
    sorted.sort_by_key(|l| l.concurrency);

    let max_tps = sorted
        .iter()
        .map(|l| l.aggregate_tps)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let cmax = sorted.last().map(|l| l.concurrency).unwrap_or(1);

    let mut pts: Vec<CurvePoint> = Vec::with_capacity(sorted.len());
    let mut prev_x: i64 = -1;
    for l in &sorted {
        let x_raw: i64 = if cmax <= 1 {
            (w / 2) as i64
        } else {
            let t = (l.concurrency as f64).log2() / (cmax as f64).log2();
            1 + (t * (w as f64 - 2.0)).round() as i64
        };
        // Strictly increasing x (a bar must never overlap the previous
        // one), clamped inside the plot.
        let x = x_raw.max(prev_x + 1).min((w - 1) as i64).max(1) as u16;
        prev_x = x as i64;
        let y = ((plot_h - 1) as i64
            - (l.aggregate_tps / max_tps * (plot_h - 1) as f64).round() as i64)
            .clamp(0, (plot_h - 1) as i64) as u16;
        let (color, marker) = if envelope
            .as_ref()
            .is_some_and(|e| e.sweet_spot == l.concurrency)
        {
            (palette::OK, '●')
        } else if envelope
            .as_ref()
            .and_then(|e| e.knee)
            .is_some_and(|k| k.concurrency == l.concurrency)
        {
            (palette::ERR, '▲')
        } else {
            (palette::ACCENT, '•')
        };
        pts.push(CurvePoint {
            x,
            y,
            color,
            marker,
            label: l.concurrency.to_string(),
        });
    }

    // Cell grid: (char, optional fg color).
    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w as usize]; h as usize];
    for p in &pts {
        for row in p.y..plot_h {
            grid[row as usize][p.x as usize] = ('│', Some(p.color));
        }
        grid[p.y as usize][p.x as usize] = (p.marker, Some(p.color));
    }
    // x labels on the bottom row, skipping any that would overlap.
    let mut label_end: i64 = -1;
    for p in &pts {
        let start = (p.x as i64 - p.label.len() as i64 / 2).max(0);
        if start > label_end {
            for (i, ch) in p.label.chars().enumerate() {
                let col = (start + i as i64) as usize;
                if col < w as usize {
                    grid[(h - 1) as usize][col] = (ch, Some(palette::MUTED));
                }
            }
            label_end = start + p.label.len() as i64;
        }
    }

    grid.into_iter()
        .map(|row| {
            Line::from(
                row.into_iter()
                    .map(|(ch, c)| match c {
                        Some(color) => Span::styled(ch.to_string(), Style::default().fg(color)),
                        None => Span::raw(ch.to_string()),
                    })
                    .collect::<Vec<Span>>(),
            )
        })
        .collect()
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

    let sweep = app.sweep.load();
    match sweep.as_ref().as_ref() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::concurrency::SweepResult;

    fn lvl(concurrency: usize, tps: f64, p90_ms: f64) -> SweepLevel {
        SweepLevel {
            concurrency,
            aggregate_tps: tps,
            p50_tpot_ns: 0,
            p90_tpot_ns: (p90_ms * 1e6) as u64,
            p99_tpot_ns: 0,
            ttft_p50_ns: 0,
            ttft_p90_ns: 0,
            total_tokens: 0,
            completed_streams: 0,
            failed_streams: 0,
            wall_ns: 0,
            streams: Vec::new(),
        }
    }

    #[test]
    fn curve_lines_guard_degenerate_areas() {
        let lines = build_curve_lines(&[lvl(1, 100.0, 5.0)], 3, 3, &None);
        assert_eq!(lines[0].to_string(), "plot area too small");
    }

    #[test]
    fn curve_marks_sweet_spot_knee_and_other_levels() {
        // Throughput plateaus (350→340) while p90 spikes 8→20 ms: the
        // knee is 4, the sweet spot 2.
        let levels = vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0), lvl(4, 340.0, 20.0)];
        let env = SweepResult {
            levels: levels.clone(),
        }
        .envelope();
        let lines = build_curve_lines(&levels, 40, 10, &env);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        // Sweet spot (2) → green ●, knee (4) → red ▲, other (1) → •.
        assert!(text.contains('●'), "sweet spot marker");
        assert!(text.contains('▲'), "knee marker");
        assert!(text.contains('•'), "regular-level marker");
        assert!(text.contains('│'), "vertical bar");
        // The x labels (1, 2, 4) sit on the bottom row.
        let bottom = lines.last().unwrap().to_string();
        for c in ["1", "2", "4"] {
            assert!(bottom.contains(c), "missing x label {c}");
        }
    }

    #[test]
    fn curve_marker_height_tracks_throughput() {
        // No envelope → both points are `•`; the 400 t/s bar must reach
        // a strictly higher row (smaller index) than the 50 t/s bar.
        let levels = vec![lvl(1, 50.0, 5.0), lvl(2, 400.0, 6.0)];
        let lines = build_curve_lines(&levels, 40, 12, &None);
        let rows: Vec<Vec<char>> = lines
            .iter()
            .map(|l| l.to_string().chars().collect())
            .collect();
        let marker_rows: Vec<usize> = (0..rows[0].len())
            .filter(|&col| rows.iter().any(|r| r[col] == '•'))
            .map(|col| rows.iter().position(|r| r[col] == '•').unwrap())
            .collect();
        assert_eq!(marker_rows.len(), 2, "one marker per level");
        assert_ne!(
            marker_rows[0], marker_rows[1],
            "throughput must change the plotted height"
        );
    }
}
