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
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table, Wrap};
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

/// Top panel: aggregate throughput vs concurrent users — one point per
/// sweep level (x on a `log2(concurrency)` scale, y ∝ aggregate t/s),
/// with each point labelled by its measured t/s, the **sweet spot**
/// (yellow) and the **saturation knee** (red `▲`) marked, the
/// diminishing-returns zone past the knee drawn dim/red, and a
/// plain-language "what to do with this" note underneath.
fn render_curve(area: Rect, app: &App, f: &mut Frame, envelope: &Option<Envelope>) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("CONCURRENCY SWEEP — Aggregate Throughput vs Parallel Users");

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

    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let notes = curve_notes(&result.levels, envelope);
    // Give the plot the space left over for the notes (it self-degrades
    // to a compact form when squeezed by a small terminal).
    let plot_h = (inner_h as i64 - notes.len() as i64).max(4) as u16;
    let lines = build_curve_lines(&result.levels, inner_w as u16, plot_h, envelope);
    let mut all = lines;
    all.extend(notes);
    f.render_widget(Paragraph::new(Text::from(all)).block(block), area);
}

/// One plotted point on the throughput-vs-concurrency curve.
struct CurvePoint {
    x: u16,
    /// The marker's row (0 = top of the grid).
    y: u16,
    color: Color,
    marker: char,
    /// The measured t/s value label.
    value: String,
    /// The concurrency x-label.
    label: String,
    /// The knee annotation (e.g. `KNEE @ 32`), when this point is it.
    knee_note: Option<String>,
}

/// Build the curve as lines of styled single-character spans.
///
/// Pure function over the sweep levels (unit-testable, no terminal):
///
/// * y-axis labels (`max` / `max/2` / `0`) on the left;
/// * one point per level — x on a `log2(concurrency)` scale (the ladder
///   is a power-of-two series), y ∝ aggregate t/s against the curve max;
/// * a vertical `│` stem from each point down to the baseline;
/// * the measured t/s **labelled on every point**;
/// * zone colors: **green** below the sweet spot, **yellow** at the sweet
///   spot, **red** past the knee; the knee itself is an `▲` with a
///   `KNEE @ n` annotation;
/// * a dimmed red `─` connector through the diminishing-returns zone;
/// * an x-axis (`├──┬──`) with the concurrency values and a
///   `concurrent users` axis title;
/// * a legend row (`● measured  ▲ knee  ── diminishing returns`).
///
/// Below 8 rows the axis title / legend drop out (compact form).
pub(crate) fn build_curve_lines(
    levels: &[SweepLevel],
    w: u16,
    h: u16,
    envelope: &Option<Envelope>,
) -> Vec<Line<'static>> {
    if w < 6 || h < 4 {
        return vec![Line::from("plot area too small")];
    }
    let w = w as usize;
    let h = h as usize;
    let compact = h < 8;

    // Vertical layout. Full: unit label, grid, axis, x labels, axis
    // title, legend. Compact: grid, axis, x labels.
    let (unit_row, top_row, axis_row, labels_row, title_row, legend_row) = if compact {
        (None, 0, h - 2, h - 1, None, None)
    } else {
        (Some(0), 1, h - 4, h - 3, Some(h - 2), Some(h - 1))
    };
    let base_row = axis_row.saturating_sub(1); // last grid row
    let grid_rows = base_row.saturating_sub(top_row) + 1;
    if grid_rows < 1 {
        return vec![Line::from("plot area too small")];
    }

    let mut sorted: Vec<&SweepLevel> = levels.iter().collect();
    sorted.sort_by_key(|l| l.concurrency);
    let max_tps = sorted
        .iter()
        .map(|l| l.aggregate_tps)
        .fold(0.0_f64, f64::max)
        .max(1.0);
    let cmax = sorted.last().map(|l| l.concurrency).unwrap_or(1);
    let sweet = envelope.as_ref().map(|e| e.sweet_spot);
    let knee = envelope
        .as_ref()
        .and_then(|e| e.knee)
        .map(|k| k.concurrency);

    // The zone color: green below the sweet spot, yellow at it, red past
    // it (the diminishing-returns zone). No envelope (defensive) → green.
    let zone_color = |c: usize| -> Color {
        match (sweet, knee) {
            (Some(s), Some(k)) => {
                if c > k {
                    palette::ERR
                } else if c == s {
                    palette::WARN
                } else if c > s {
                    palette::ERR
                } else {
                    palette::OK
                }
            }
            (Some(s), None) => {
                if c == s {
                    palette::WARN
                } else if c > s {
                    palette::ERR
                } else {
                    palette::OK
                }
            }
            _ => palette::OK,
        }
    };

    let mut pts: Vec<CurvePoint> = Vec::with_capacity(sorted.len());
    let mut prev_x: i64 = -1;
    for l in &sorted {
        let x_raw: i64 = if cmax <= 1 {
            (w / 2) as i64
        } else {
            let t = (l.concurrency as f64).log2() / (cmax as f64).log2();
            (Y_AXIS_W as f64 + 1.0 + t * (w as f64 - Y_AXIS_W as f64 - 2.0)).round() as i64
        };
        // Strictly increasing x (a point must never overlap the previous
        // one), clamped inside the plot.
        let x = x_raw
            .max(prev_x + 1)
            .min((w - 1) as i64)
            .max(Y_AXIS_W as i64 + 1) as usize;
        prev_x = x as i64;
        let y = (base_row
            - (l.aggregate_tps / max_tps * (base_row - top_row) as f64).round() as usize)
            .clamp(top_row, base_row);
        let is_knee = knee == Some(l.concurrency);
        pts.push(CurvePoint {
            x: x as u16,
            y: y as u16,
            color: zone_color(l.concurrency),
            marker: if is_knee { '▲' } else { '●' },
            value: format!("{:.1} t/s", l.aggregate_tps),
            label: l.concurrency.to_string(),
            knee_note: is_knee.then(|| format!("KNEE @ {}", l.concurrency)),
        });
    }

    // Cell grid: (char, optional fg color).
    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w]; h];

    // The `t/s` unit label above the y-axis (full mode only).
    if let Some(r) = unit_row {
        let text = "t/s";
        for (i, ch) in text.chars().enumerate() {
            if i < w {
                grid[r][i] = (ch, Some(palette::MUTED));
            }
        }
    }
    // y-axis labels: max / max÷2 / 0 (right-aligned in the y-axis column).
    let mut place_y = |row: usize, val: f64| {
        let text = format!("{}", val.round());
        let start = Y_AXIS_W.saturating_sub(text.len());
        for (i, ch) in text.chars().enumerate() {
            let col = start + i;
            if col < w && row < h {
                grid[row][col] = (ch, Some(palette::MUTED));
            }
        }
    };
    place_y(top_row, max_tps);
    place_y((top_row + base_row) / 2, max_tps / 2.0);
    place_y(base_row, 0.0);

    // Stems + markers, and the dimmed `─` connector through the
    // diminishing-returns zone (past the knee / past the sweet spot).
    for (i, p) in pts.iter().enumerate() {
        for row in grid.iter_mut().take(base_row + 1).skip(p.y as usize) {
            row[p.x as usize] = ('│', Some(p.color));
        }
        grid[p.y as usize][p.x as usize] = (p.marker, Some(p.color));
        if i > 0 {
            let prev = &pts[i - 1];
            // Dimmed (MUTED) past the knee — the diminishing-returns zone.
            let line_color = if p.color == palette::ERR {
                palette::MUTED
            } else {
                p.color
            };
            for cell in &mut grid[p.y as usize][(prev.x as usize + 1)..(p.x as usize).min(w)] {
                *cell = ('─', Some(line_color));
            }
        }
    }

    // Value labels to the right of each marker (with the knee note
    // appended), skipping any that would collide or run off the edge.
    // The placed labels are remembered so the line rebuild can restyle
    // them bold (the grid itself only carries char + color).
    let mut label_end: Vec<Option<usize>> = vec![None; h];
    let mut placed: Vec<(usize, usize, String, Color)> = Vec::new();
    for p in &pts {
        let text = match &p.knee_note {
            Some(n) => format!("{} · {n}", p.value),
            None => p.value.clone(),
        };
        // Prefer the right of the marker; a rightmost point (no room)
        // falls back to its left.
        let starts = [
            p.x as usize + 2,
            (p.x as usize).saturating_sub(text.len() + 1).max(Y_AXIS_W),
        ];
        let mut done = false;
        for try_row in [
            p.y as usize,
            (p.y as usize).saturating_sub(1),
            p.y as usize + 1,
        ] {
            if try_row < top_row || try_row > base_row {
                continue;
            }
            for start in starts {
                if start + text.len() > w {
                    continue;
                }
                if let Some(end) = label_end[try_row] {
                    if start <= end + 1 {
                        continue;
                    }
                }
                for (i, ch) in text.chars().enumerate() {
                    grid[try_row][start + i] = (ch, Some(p.color));
                }
                label_end[try_row] = Some(start + text.len());
                placed.push((try_row, start, text.clone(), p.color));
                done = true;
                break;
            }
            if done {
                break;
            }
        }
    }

    // x-axis row: `├──┬──` with a `┬` under each point.
    for cell in &mut grid[axis_row][Y_AXIS_W..] {
        *cell = ('─', Some(palette::MUTED));
    }
    grid[axis_row][Y_AXIS_W] = ('├', Some(palette::MUTED));
    for p in &pts {
        grid[axis_row][p.x as usize] = ('┬', Some(palette::MUTED));
    }

    // x labels (the concurrency values), skipping overlaps.
    let mut x_end: i64 = -1;
    for p in &pts {
        let start = (p.x as i64 - p.label.len() as i64 / 2).max(Y_AXIS_W as i64);
        if start > x_end {
            for (i, ch) in p.label.chars().enumerate() {
                let col = (start + i as i64) as usize;
                if col < w {
                    grid[labels_row][col] = (ch, Some(palette::MUTED));
                }
            }
            x_end = start + p.label.len() as i64;
        }
    }

    // Axis title + legend (full mode only).
    if let Some(r) = title_row {
        write_text(
            &mut grid,
            r,
            Y_AXIS_W,
            "concurrent users",
            palette::MUTED,
            w,
        );
    }
    if let Some(r) = legend_row {
        let legend: [(&str, Color); 3] = [
            ("● measured", palette::OK),
            ("▲ knee (saturation)", palette::ERR),
            ("── diminishing returns", palette::MUTED),
        ];
        let mut col = Y_AXIS_W;
        for (text, color) in legend {
            if col + text.len() > w {
                break;
            }
            write_text(&mut grid, r, col, text, color, w);
            col += text.len() + 3;
        }
    }

    // Rebuild the lines: one span per grid cell, then merge each placed
    // value label's run of single-char spans into one bold span. Per row,
    // the splices run in *descending* start order so an earlier splice
    // never shifts the indices a later one uses.
    let mut lines: Vec<Line> = grid
        .iter()
        .map(|row| {
            Line::from(
                row.iter()
                    .map(|(ch, c)| match c {
                        Some(color) => Span::styled(ch.to_string(), Style::default().fg(*color)),
                        None => Span::raw(ch.to_string()),
                    })
                    .collect::<Vec<Span>>(),
            )
        })
        .collect();
    let mut by_row: std::collections::BTreeMap<usize, Vec<(usize, String, Color)>> =
        Default::default();
    for (row, start, text, color) in placed {
        by_row.entry(row).or_default().push((start, text, color));
    }
    for (row, mut labels) in by_row {
        if row >= lines.len() {
            continue;
        }
        labels.sort_by_key(|&(start, _, _)| std::cmp::Reverse(start));
        for (start, text, color) in labels {
            let end = start + text.len();
            if end > lines[row].spans.len() {
                continue;
            }
            let mut spans: Vec<Span> = lines[row].spans.clone();
            spans.splice(
                start..end,
                [Span::styled(
                    text.clone(),
                    Style::default().fg(color).add_modifier(Modifier::BOLD),
                )],
            );
            lines[row] = Line::from(spans);
        }
    }

    lines
}

/// The y-axis label column width (shared by both chart builders).
const Y_AXIS_W: usize = 5;

/// Write `text` into the grid at `(row, col)` with an fg color
/// (clipped at the grid edge).
fn write_text(
    grid: &mut [Vec<(char, Option<Color>)>],
    row: usize,
    col: usize,
    text: &str,
    color: Color,
    w: usize,
) {
    for (i, ch) in text.chars().enumerate() {
        let c = col + i;
        if c < w && row < grid.len() {
            grid[row][c] = (ch, Some(color));
        }
    }
}

/// The plain-language "what to do with this curve" note under the plot:
/// a dimmed `ℹ` explanation of the knee, plus the actionable sweet spot
/// (or the no-knee fallback: the curve is still climbing).
pub(crate) fn curve_notes(
    levels: &[SweepLevel],
    envelope: &Option<Envelope>,
) -> Vec<Line<'static>> {
    let mut ls: Vec<Line> = vec![Line::from(Span::styled(
        "ℹ Below the knee: more users = more total throughput. Above it, it hurts everyone.",
        style::info(),
    ))];
    match envelope {
        Some(env) => {
            ls.push(Line::from(vec![
                Span::styled("  SWEET SPOT: ", style::value_ok()),
                Span::styled(
                    format!(
                        "{} users — {:.1} t/s. Run at or below it.",
                        env.sweet_spot, env.aggregate_tps
                    ),
                    style::value_ok(),
                ),
            ]));
            match env.knee {
                Some(k) => ls.push(Line::from(vec![
                    Span::styled("  SATURATION:  ", style::value_err()),
                    Span::styled(
                        format!(
                            "{} users — plateau, p90 spikes to {:.1} ms.",
                            k.concurrency,
                            k.p90_tpot_ns as f64 / 1e6
                        ),
                        style::value_err(),
                    ),
                ])),
                None => ls.push(Line::from(Span::styled(
                    "  No knee detected — throughput still climbing.",
                    style::footer(),
                ))),
            }
        }
        None => {
            if let Some(peak) = levels
                .iter()
                .max_by(|a, b| a.aggregate_tps.total_cmp(&b.aggregate_tps))
            {
                ls.push(Line::from(Span::styled(
                    format!(
                        "  No knee detected — still climbing at {} users ({:.1} t/s).",
                        peak.concurrency, peak.aggregate_tps
                    ),
                    style::footer(),
                )));
            }
        }
    }
    ls
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
    // A level with timed-out workers (or a step-abort) ran degraded:
    // its numbers are partial — flag it so the user can see it.
    if level.degraded() {
        state.push_str(" · DEGRADED");
    }
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

/// The dimmed `ℹ` note explaining what the envelope's numbers mean.
const ENVELOPE_INFO: &str = "Finds your server's capacity limit. The sweet spot is where you get \
     the best throughput before latency spikes; the knee is where adding \
     more users starts hurting everyone's response time.";

/// Optimal Operational Envelope (blueprint §6 View 2): the recommended
/// sweet spot, with the detected saturation knee (and the rationale —
/// throughput plateau + p90 spike) when the curve showed a transition,
/// plus a dimmed `ℹ` note explaining the numbers.
fn render_envelope(area: Rect, app: &App, f: &mut Frame, envelope: &Option<Envelope>) {
    let mut lines: Vec<Line> = match envelope {
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
            ls
        }
        None => vec![
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
        ],
    };
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        format!("ℹ {ENVELOPE_INFO}"),
        style::info(),
    )));
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::active_border())
                    .title("OPTIMAL OPERATIONAL ENVELOPE"),
            )
            .wrap(Wrap { trim: true }),
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
            timed_out_streams: 0,
            aborted: false,
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
        let lines = build_curve_lines(&levels, 80, 12, &env);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        // Measured points are ● (sweet spot + regular), the knee is ▲.
        assert!(text.contains('●'), "measured-point marker");
        assert!(text.contains('▲'), "knee marker");
        assert!(text.contains('│'), "vertical stem");
        assert!(text.contains('─'), "axis / connector line");
        // Every point is labelled with its measured t/s.
        assert!(text.contains("100.0 t/s"), "value label at 1: {text}");
        assert!(text.contains("350.0 t/s"), "value label at 2: {text}");
        assert!(text.contains("340.0 t/s"), "value label at 4: {text}");
        // The knee is annotated.
        assert!(text.contains("KNEE @ 4"), "knee annotation: {text}");
        // x labels (1, 2, 4) + the axis title.
        for c in ["1", "2", "4"] {
            assert!(text.contains(c), "missing x label {c}");
        }
        assert!(text.contains("concurrent users"), "axis title: {text}");
        // The legend explains the markers.
        assert!(text.contains("diminishing returns"), "legend: {text}");
        // y-axis max label.
        assert!(text.contains("350"), "y-axis max: {text}");
    }

    #[test]
    fn curve_notes_explain_the_sweet_spot_and_knee() {
        let levels = vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0), lvl(4, 340.0, 20.0)];
        let env = SweepResult {
            levels: levels.clone(),
        }
        .envelope();
        let notes = curve_notes(&levels, &env);
        let text: String = notes
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('ℹ'), "explanation present");
        assert!(text.contains("SWEET SPOT"), "actionable sweet spot: {text}");
        assert!(text.contains("2 users"), "sweet spot value: {text}");
        assert!(text.contains("SATURATION"), "knee line: {text}");
        assert!(text.contains("4 users"), "knee value: {text}");
    }

    #[test]
    fn curve_notes_fallback_when_no_knee() {
        // A strictly climbing curve has no knee.
        let levels = vec![lvl(1, 100.0, 5.0), lvl(2, 200.0, 6.0), lvl(4, 300.0, 7.0)];
        let env = SweepResult {
            levels: levels.clone(),
        }
        .envelope();
        let notes = curve_notes(&levels, &env);
        let text: String = notes
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("still climbing") || text.contains("No knee"),
            "no-knee fallback: {text}"
        );
    }

    #[test]
    fn envelope_panel_shows_the_capacity_info_note() {
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
        assert!(text.contains("OPTIMAL OPERATIONAL ENVELOPE"), "{text}");
        assert!(text.contains('ℹ'), "capacity info note: {text}");
        assert!(text.contains("capacity limit"), "{text}");
    }

    #[test]
    fn curve_marker_height_tracks_throughput() {
        // No envelope → both points are `●`; the 400 t/s marker must sit
        // on a strictly higher row (smaller index) than the 50 t/s one.
        // (h=12 → full layout: grid rows 1..=7; the legend row below is
        // excluded so its `●` is not counted as a data point.)
        let levels = vec![lvl(1, 50.0, 5.0), lvl(2, 400.0, 6.0)];
        let lines = build_curve_lines(&levels, 40, 12, &None);
        let rows: Vec<Vec<char>> = lines
            .iter()
            .map(|l| l.to_string().chars().collect())
            .collect();
        let grid: Vec<&Vec<char>> = rows[1..8].iter().collect();
        let marker_rows: Vec<usize> = (0..rows[0].len())
            .filter(|&col| grid.iter().any(|r| r[col] == '●'))
            .map(|col| grid.iter().position(|r| r[col] == '●').unwrap())
            .collect();
        assert_eq!(marker_rows.len(), 2, "one marker per level");
        assert_ne!(
            marker_rows[0], marker_rows[1],
            "throughput must change the plotted height"
        );
    }
}
