//! View 2 — Concurrency & Saturation matrix (blueprint §6): X = concurrency
//! (1…128), Y = latency vs throughput, with the knee / optimal operational
//! envelope highlighted.
//!
//! Once a [`Sweep`] (plan Chunk 10) has run, the matrix renders the real
//! per-level curve — aggregate tokens/sec and client-perceived p90 TPOT
//! pooled across every concurrent stream of the level — and highlights the
//! **saturation knee** (plan Chunk 11, [`SweepResult::detect_knee`]): the
//! level where aggregate throughput plateaus while p90 TPOT spikes (the
//! memory-bandwidth-bound → compute-bound transition).
//!
//! The bottom panel is the **Concurrency Recommendation** (the practical
//! sweet spot, FIX 3): the knee is *reference only* — the recommendation
//! is per-stream usability, i.e. how many users you can serve while each
//! still gets an acceptable speed (≥40 t/s comfortable, ≥15 t/s usable).
//!
//! The top panel renders the sweep as a block-based throughput-vs-
//! concurrency curve: one vertical bar per ladder level (x on a
//! `log2(concurrency)` scale, y ∝ aggregate t/s), with the knee capped
//! by a red `▲` and the rest marked `●`.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Cell, Paragraph, Row, Table, Wrap};
use ratatui::Frame;

use crate::engines::concurrency::{Envelope, SweepLevel, SweepResult, DEFAULT_LADDER};
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, glyph, style, Theme};

/// Render the Concurrency view into `area`.
///
/// The envelope (knee + sweet spot) is computed once per frame from the
/// last [`SweepResult`] and shared by the matrix and the envelope panel.
/// Reading `app.sweep` is a pure `&` read — no locks, no timing path
/// (measurement isolation, blueprint §4).
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
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
    render_curve(chunks[0], app, th, f, &envelope);
    render_matrix(chunks[1], app, th, f, &envelope);
    render_recommendation(chunks[2], app, th, f);
}

/// Top panel: aggregate throughput vs concurrent users — one point per
/// sweep level (x on a `log2(concurrency)` scale, y ∝ aggregate t/s),
/// with each point labelled by its measured t/s, the **sweet spot**
/// (yellow) and the **saturation knee** (red `▲`) marked, the
/// diminishing-returns zone past the knee drawn dim/red, and a
/// plain-language "what to do with this" note underneath.
fn render_curve(area: Rect, app: &App, th: Theme, f: &mut Frame, envelope: &Option<Envelope>) {
    // The primary title is uppercase + accent; the descriptive subtitle keeps
    // its natural casing (the "… vs Parallel Users" phrasing).
    let title = Line::from(vec![
        Span::styled(format!("{} ", glyph::PREFIX), style::title(th)),
        Span::styled("CONCURRENCY SWEEP — ", style::title(th)),
        Span::styled("Aggregate Throughput vs Parallel Users", style::title(th)),
    ]);
    let block = theme::block(title, style::border(th));

    let sweep = app.sweep.load();
    let Some(result) = sweep.as_ref().as_ref().filter(|r| !r.levels.is_empty()) else {
        // Empty state: no sweep has run yet.
        f.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(Span::styled(
                    "Concurrency sweep will populate this view.",
                    style::info(th),
                )),
                Line::from(Span::styled(
                    "Run a sweep (Engine B in the Config view, or [r]) to plot the curve.",
                    style::footer(th),
                )),
            ]))
            .block(block),
            area,
        );
        return;
    };

    let inner_w = area.width.saturating_sub(2) as usize;
    let inner_h = area.height.saturating_sub(2) as usize;
    let notes = curve_notes(th, &result.levels);
    // Give the plot the space left over for the notes (it self-degrades
    // to a compact form when squeezed by a small terminal).
    let plot_h = (inner_h as i64 - notes.len() as i64).max(4) as u16;
    let lines = build_curve_lines(th, &result.levels, inner_w as u16, plot_h, envelope);
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
    th: Theme,
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
                    th.danger()
                } else if c == s {
                    th.primary()
                } else if c > s {
                    th.accent()
                } else {
                    th.success()
                }
            }
            (Some(s), None) => {
                if c == s {
                    th.primary()
                } else if c > s {
                    th.danger()
                } else {
                    th.success()
                }
            }
            _ => th.success(),
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
            marker: if is_knee { glyph::KNEE } else { glyph::POINT },
            value: fmt::format_rate(l.aggregate_tps),
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
                grid[r][i] = (ch, Some(th.dim()));
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
                grid[row][col] = (ch, Some(th.dim()));
            }
        }
    };
    place_y(top_row, max_tps);
    place_y((top_row + base_row) / 2, max_tps / 2.0);
    place_y(base_row, 0.0);

    // Stems + markers + the dim area fill under the curve, and the
    // dim-cyan `─` connector between points. Past the knee the area turns
    // dim purple — the diminishing-returns / degraded zone.
    for (i, p) in pts.iter().enumerate() {
        // Area fill under this segment: from the base up to the lower of
        // the two endpoints (never above the curve).
        if i > 0 {
            let prev = &pts[i - 1];
            let fill_top = (p.y as usize).min(prev.y as usize);
            let degraded = p.color == th.danger() || p.color == th.accent();
            let fill = if degraded {
                th.secondary()
            } else {
                th.floor()
            };
            for row in grid.iter_mut().take(base_row + 1).skip(fill_top) {
                for cell in row.iter_mut().take(p.x as usize).skip(prev.x as usize + 1) {
                    if cell.0 == ' ' {
                        *cell = (glyph::FLOOR, Some(fill));
                    }
                }
            }
        }
        // Vertical stem (dim cyan).
        for row in grid.iter_mut().take(base_row + 1).skip(p.y as usize) {
            row[p.x as usize] = ('│', Some(th.border_active()));
        }
        // The marker at the top of the stem.
        grid[p.y as usize][p.x as usize] = (p.marker, Some(p.color));
        // The dim-cyan `─` connector through the segment.
        if i > 0 {
            let prev = &pts[i - 1];
            for cell in &mut grid[p.y as usize][(prev.x as usize + 1)..(p.x as usize).min(w)] {
                *cell = ('─', Some(th.border_active()));
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
        *cell = ('─', Some(th.dim()));
    }
    grid[axis_row][Y_AXIS_W] = ('├', Some(th.dim()));
    for p in &pts {
        grid[axis_row][p.x as usize] = ('┬', Some(th.dim()));
    }

    // x labels (the concurrency values), skipping overlaps.
    let mut x_end: i64 = -1;
    for p in &pts {
        let start = (p.x as i64 - p.label.len() as i64 / 2).max(Y_AXIS_W as i64);
        if start > x_end {
            for (i, ch) in p.label.chars().enumerate() {
                let col = (start + i as i64) as usize;
                if col < w {
                    grid[labels_row][col] = (ch, Some(th.dim()));
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
            th.dim(),
            w,
        );
    }
    if let Some(r) = legend_row {
        let legend: [(&str, Color); 3] = [
            ("◆ measured", th.success()),
            ("▲ knee", th.accent()),
            ("── diminishing returns", th.secondary()),
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

    // Rebuild the lines: each row's grid is grouped into runs of identical
    // (char, color) cells — one span per run, not one allocation per cell
    // (a mostly-blank plot row is a handful of spans on the 60 Hz render
    // path) — and each placed value label is emitted directly as one
    // bold span at its column range (the placer already rejected
    // collisions, so the labels never overlap).
    let mut by_row: std::collections::BTreeMap<usize, Vec<(usize, String, Color)>> =
        Default::default();
    for (row, start, text, color) in placed {
        by_row.entry(row).or_default().push((start, text, color));
    }
    let mut lines: Vec<Line> = Vec::with_capacity(grid.len());
    for (row_idx, row) in grid.iter().enumerate() {
        let mut spans: Vec<Span> = Vec::new();
        let mut col = 0usize;
        if let Some(labels) = by_row.get(&row_idx) {
            let mut labels: Vec<&(usize, String, Color)> = labels.iter().collect::<Vec<_>>();
            labels.sort_by_key(|&(start, _, _)| *start);
            for (start, text, color) in labels {
                if *start > col {
                    push_cell_runs(&mut spans, row, col, (*start).min(row.len()));
                }
                spans.push(Span::styled(
                    text.clone(),
                    Style::default().fg(*color).add_modifier(Modifier::BOLD),
                ));
                col = (*start + text.len()).min(row.len());
            }
        }
        if col < row.len() {
            push_cell_runs(&mut spans, row, col, row.len());
        }
        lines.push(Line::from(spans));
    }

    lines
}

/// Append one span per run of identical (char, color) cells in
/// `row[from..to)` (consecutive blanks / same-color glyphs collapse into
/// a single span — the render-path allocation saver).
fn push_cell_runs(spans: &mut Vec<Span>, row: &[(char, Option<Color>)], from: usize, to: usize) {
    let mut run_start = from;
    for i in (from + 1)..=to {
        if i == to || row[i] != row[run_start] {
            let (ch, color) = row[run_start];
            let text: String = std::iter::repeat_n(ch, i - run_start).collect();
            spans.push(match color {
                Some(c) => Span::styled(text, Style::default().fg(c)),
                None => Span::raw(text),
            });
            run_start = i;
        }
    }
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

/// The recommendation data lines (FIX 3): the **practical sweet spot**
/// (the highest concurrency where every user still gets ≥40 t/s — the
/// recommendation), the **maximum usable** level (≥15 t/s each), the
/// **unusable beyond** boundary (<15 t/s each), and the **pure
/// throughput knee** as reference only. Pure over the sweep result
/// (unit-testable, no terminal).
fn recommendation_lines(th: Theme, result: &SweepResult) -> Vec<Line<'static>> {
    let us = result.usability();
    // The reference knee: the detected saturation knee, else the peak-
    // throughput level (where total t/s tops out).
    let knee = result
        .envelope()
        .and_then(|e| e.knee)
        .map(|k| k.concurrency)
        .or_else(|| result.peak_throughput().map(|l| l.concurrency));
    let mut ls: Vec<Line> = Vec::new();
    // Practical sweet spot — the recommendation, prominent.
    match us.practical_sweet_spot {
        Some(n) => {
            ls.push(Line::from(vec![
                Span::styled(
                    "  Practical Sweet Spot: ",
                    Style::default()
                        .fg(th.success())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!("{n} concurrent users"),
                    Style::default()
                        .fg(th.success())
                        .add_modifier(Modifier::BOLD),
                ),
                Span::styled(
                    format!(" — each gets ~{:.0} t/s", us.practical_per_stream),
                    style::value_ok(th),
                ),
            ]));
        }
        None => ls.push(Line::from(Span::styled(
            "  Practical Sweet Spot: — (even 1 user is below 40 t/s)",
            style::value_warn(th),
        ))),
    }
    // Maximum usable.
    match us.max_usable {
        Some(n) => ls.push(Line::from(vec![
            Span::styled("  Maximum Usable:      ", style::label(th)),
            Span::styled(format!("{n} concurrent users"), style::value(th)),
            Span::styled(
                format!(" — each gets ~{:.0} t/s (slower)", us.max_usable_per_stream),
                style::footer(th),
            ),
        ])),
        None => ls.push(Line::from(Span::styled(
            "  Maximum Usable:      — (below 15 t/s per user throughout)",
            style::value_err(th),
        ))),
    }
    // Unusable beyond.
    match us.unusable_from {
        Some(n) => ls.push(Line::from(vec![
            Span::styled("  Unusable Beyond:     ", style::label(th)),
            Span::styled(format!("{n}+ concurrent users"), style::value_err(th)),
            Span::styled(" — each drops below 15 t/s", style::footer(th)),
        ])),
        None => ls.push(Line::from(Span::styled(
            "  Unusable Beyond:     not reached in this sweep",
            style::footer(th),
        ))),
    }
    // Pure throughput knee — reference only.
    if let Some(k) = knee {
        ls.push(Line::from(vec![
            Span::styled("  Pure Throughput Knee: ", style::label(th)),
            Span::styled(format!("{k}"), style::footer(th)),
            Span::styled(" (reference only — total t/s peaks here)", style::footer(th)),
        ]));
    }
    ls
}

/// The plain-language note under the plot (shared with the Live view's
/// concurrency panel, FIX 3): what the recommendation means, the
/// practical-sweet-spot lines, and the aggregate-throughput caveat.
pub(crate) fn curve_notes(th: Theme, levels: &[SweepLevel]) -> Vec<Line<'static>> {
    let result = SweepResult {
        levels: levels.to_vec(),
        matrix: None,
    };
    let mut ls: Vec<Line> = vec![Line::from(Span::styled(
        "ℹ Sweet spot = most users where EACH still gets ≥40 t/s (comfortable).",
        style::info(th),
    ))];
    ls.extend(recommendation_lines(th, &result));
    ls.push(Line::from(Span::styled(
        "ℹ Aggregate t/s alone misleads: many slow users is not fast service.",
        style::info(th),
    )));
    ls
}

/// Sweep matrix: one row per ladder step — real aggregate t/s + p90 TPOT
/// once a sweep has run, `--` until then. The knee row is flagged red
/// (`KNEE`) and the sweet-spot row green (`SWEET`).
fn render_matrix(area: Rect, app: &App, th: Theme, f: &mut Frame, envelope: &Option<Envelope>) {
    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("USERS"),
        Cell::from("AGG T/S"),
        Cell::from("PER-STREAM"),
        Cell::from("P90 TPOT"),
        Cell::from("STATUS"),
    ])
    .style(style::muted_title(th))];

    let sweep = app.sweep.load();
    match sweep.as_ref().as_ref() {
        Some(result) if !result.levels.is_empty() => {
            // The single-user baseline: the min-concurrency level's
            // per-stream rate (aggregate ÷ users). It is the reference the
            // per-stream status is measured against (FIX 3).
            let baseline = result
                .levels
                .iter()
                .min_by_key(|l| l.concurrency)
                .map(|l| l.aggregate_tps / l.concurrency.max(1) as f64)
                .unwrap_or(0.0);
            for level in &result.levels {
                rows.push(sweep_row(th, level, envelope, app.concurrency_target, baseline));
            }
        }
        _ => {
            for &level in &DEFAULT_LADDER {
                let is_target = level == app.concurrency_target;
                rows.push(Row::new(vec![
                    Cell::from(level.to_string()).style(if is_target {
                        style::value_ok(th)
                    } else {
                        style::label(th)
                    }),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from(if is_target {
                        "current target"
                    } else {
                        "not run"
                    })
                    .style(if is_target {
                        style::value_ok(th)
                    } else {
                        style::footer(th)
                    }),
                ]));
            }
        }
    }

    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Percentage(12),
                Constraint::Percentage(18),
                Constraint::Percentage(18),
                Constraint::Percentage(14),
                Constraint::Percentage(38),
            ],
        )
        .block(theme::block(
            theme::panel_title(th, "CONCURRENCY SWEEP — Aggregate vs Per-Stream t/s"),
            style::border(th),
        )),
        area,
    );
}

/// One matrix row for a completed sweep level, flagged when it is the
/// detected knee (red) or the recommended sweet spot (green), and carrying
/// the **Per-Stream t/s** column (FIX 3: aggregate ÷ users) plus a
/// color-coded per-stream status.
fn sweep_row<'a>(
    th: Theme,
    level: &'a SweepLevel,
    envelope: &'a Option<Envelope>,
    target: usize,
    baseline: f64,
) -> Row<'a> {
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

    // FIX 3: per-stream t/s = aggregate ÷ users — what EACH user gets.
    let per = level.aggregate_tps / level.concurrency.max(1) as f64;
    let (ps_label, ps_style) = per_stream_status(th, per, baseline);

    let value_style = if is_knee {
        style::value_err(th)
    } else if is_sweet {
        style::value_ok(th)
    } else if level.concurrency == target {
        style::highlight(th)
    } else {
        style::label(th)
    };
    let state_style = if is_knee {
        style::value_err(th)
    } else if is_sweet {
        style::value_ok(th)
    } else {
        style::footer(th)
    };
    Row::new(vec![
        Cell::from(level.concurrency.to_string()).style(value_style),
        Cell::from(fmt::format_rate(level.aggregate_tps)).style(value_style),
        Cell::from(fmt::format_rate(per)).style(value_style),
        Cell::from(format!("{:.1} ms", level.p90_tpot_ms())).style(value_style),
        Cell::from(Line::from(vec![
            Span::styled(state, state_style),
            Span::styled(format!("  {ps_label}"), ps_style),
        ])),
    ])
}

/// The per-stream status (FIX 3): green `● optimal` → yellow `● good` →
/// `▲ knee` → red `✗ saturated`, measured against the single-user
/// baseline (per-stream = aggregate ÷ users). The "degraded" threshold is
/// when per-stream falls below 50% of the baseline — the point where adding
/// more users starts hurting everyone's individual experience.
fn per_stream_status(th: Theme, per: f64, baseline: f64) -> (String, Style) {
    let ratio = if baseline > 0.0 { per / baseline } else { 1.0 };
    if ratio >= 0.85 {
        ("◆ optimal".to_string(), style::value_ok(th))
    } else if ratio >= 0.50 {
        ("◆ good".to_string(), style::value_warn(th))
    } else if ratio >= 0.25 {
        (
            "▲ knee".to_string(),
            Style::default()
                .fg(th.accent())
                .add_modifier(Modifier::BOLD),
        )
    } else {
        ("✕ saturated".to_string(), style::value_err(th))
    }
}

/// The dimmed `ℹ` notes explaining what the recommendation means (FIX 3).
const RECOMMENDATION_INFO: [&str; 4] = [
    "\"Practical sweet spot\" = most users where each still gets ≥40 t/s.",
    "This is what matters for real use: coding agents, chat, RAG pipelines.",
    "Pure aggregate throughput is misleading — 32 users at 3.8 t/s each",
    "is not \"fast\", it's \"slow for everyone\".",
];

/// The **Concurrency Recommendation** panel (blueprint §6 View 2,
/// reworked in FIX 3): the practical sweet spot (per-stream usability —
/// the recommendation), the maximum usable level, the unusable boundary,
/// and the pure-throughput knee as reference only — plus the dimmed `ℹ`
/// notes explaining why per-stream beats aggregate.
fn render_recommendation(area: Rect, app: &App, th: Theme, f: &mut Frame) {
    let sweep = app.sweep.load();
    let lines: Vec<Line> = sweep
        .as_ref()
        .as_ref()
        .filter(|r| !r.levels.is_empty())
        .map(|r| {
            let mut ls = recommendation_lines(th, r);
            ls.push(Line::raw(""));
            for (i, note) in RECOMMENDATION_INFO.iter().enumerate() {
                // The last line continues the previous sentence (no `ℹ`).
                let prefixed = if i == RECOMMENDATION_INFO.len() - 1 {
                    format!("  {note}")
                } else {
                    format!("ℹ {note}")
                };
                ls.push(Line::from(Span::styled(prefixed, style::info(th))));
            }
            ls
        })
        .unwrap_or_else(|| {
            vec![
                Line::from(vec![
                    Span::styled("Practical Sweet Spot: ", style::label(th)),
                    Span::styled(
                        format!("{} streams", app.concurrency_target),
                        style::highlight(th),
                    ),
                    Span::styled("  (current target — step with [+])", style::footer(th)),
                ]),
                Line::from(Span::styled(
                    "Run a sweep (Engine B) to find how many users you can serve while each stays fast.",
                    style::footer(th),
                )),
                Line::raw(""),
                Line::from(Span::styled(
                    "ℹ \"Practical sweet spot\" = most users where each still gets ≥40 t/s.",
                    style::info(th),
                )),
                Line::from(Span::styled(
                    "ℹ This is what matters for real use: coding agents, chat, RAG pipelines.",
                    style::info(th),
                )),
            ]
        });
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(theme::block(
                theme::panel_title(th, "CONCURRENCY RECOMMENDATION"),
                style::active_border(th),
            ))
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
            context: 0,
            aggregate_tps: tps,
            per_stream_tps: tps / concurrency.max(1) as f64,
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
            loop_excluded_streams: 0,
            loop_excluded_tokens: 0,
            streams: Vec::new(),
        }
    }

    #[test]
    fn curve_lines_guard_degenerate_areas() {
        let lines = build_curve_lines(Theme::default(), &[lvl(1, 100.0, 5.0)], 3, 3, &None);
        assert_eq!(lines[0].to_string(), "plot area too small");
    }

    #[test]
    fn curve_marks_sweet_spot_knee_and_other_levels() {
        // Throughput plateaus (350→340) while p90 spikes 8→20 ms: the
        // knee is 4, the sweet spot 2.
        let levels = vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0), lvl(4, 340.0, 20.0)];
        let env = SweepResult {
            levels: levels.clone(),
            matrix: None,
        }
        .envelope();
        let lines = build_curve_lines(Theme::default(), &levels, 80, 12, &env);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        // Measured points are ◆ (sweet spot + regular), the knee is ▲.
        assert!(text.contains('◆'), "measured-point marker");
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
    fn curve_notes_carry_the_practical_recommendation() {
        // Per-stream: 100, 175, 20 → practical 2, max usable 4, no
        // unusable boundary; the knee (reference) is at 4.
        let levels = vec![lvl(1, 100.0, 5.0), lvl(2, 350.0, 8.0), lvl(4, 80.0, 20.0)];
        let notes = curve_notes(Theme::default(), &levels);
        let text: String = notes
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('ℹ'), "explanation present");
        assert!(
            text.contains("Practical Sweet Spot"),
            "recommendation: {text}"
        );
        assert!(
            text.contains("2 concurrent users"),
            "sweet spot value: {text}"
        );
        assert!(text.contains("Maximum Usable"), "max usable: {text}");
        assert!(
            text.contains("4 concurrent users"),
            "max usable value: {text}"
        );
        assert!(
            text.contains("Pure Throughput Knee"),
            "knee as reference: {text}"
        );
    }

    #[test]
    fn curve_notes_mark_the_unusable_boundary() {
        // Per-stream: 100, 100, 75, 14.7 → unusable from 8 (below 15).
        let levels = vec![
            lvl(1, 100.0, 5.0),
            lvl(2, 200.0, 6.0),
            lvl(4, 300.0, 8.0),
            lvl(8, 117.6, 20.0),
        ];
        let notes = curve_notes(Theme::default(), &levels);
        let text: String = notes
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("Unusable Beyond") && text.contains("8+ concurrent users"),
            "unusable boundary: {text}"
        );
    }

    #[test]
    fn curve_notes_handle_a_slow_curve() {
        // Per-stream: 30, 10 → no practical spot (nothing ≥40), one usable
        // level (30 ≥ 15), unusable from 2.
        let levels = vec![lvl(1, 30.0, 5.0), lvl(2, 20.0, 8.0)];
        let notes = curve_notes(Theme::default(), &levels);
        let text: String = notes
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("even 1 user is below 40 t/s"),
            "no-practical-spot wording: {text}"
        );
        assert!(
            text.contains("1 concurrent users"),
            "partial usable wording: {text}"
        );
        assert!(
            text.contains("2+ concurrent users"),
            "unusable boundary: {text}"
        );
    }

    #[test]
    fn recommendation_panel_shows_the_info_notes() {
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
        assert!(text.contains("CONCURRENCY RECOMMENDATION"), "{text}");
        assert!(text.contains('ℹ'), "info notes: {text}");
        assert!(
            text.contains("coding agents, chat, RAG"),
            "real-use note: {text}"
        );
    }

    #[test]
    fn recommendation_panel_shows_the_sweep_recommendation() {
        let app = crate::ui::app::App::new();
        // Per-stream: 100, 95, 70, 42.5, 21.6, 10.7, 5.3 → practical 8,
        // max usable 16, unusable from 32, knee at 16.
        app.sweep.store(SweepResult {
            levels: vec![
                lvl(1, 100.0, 5.0),
                lvl(2, 190.0, 6.0),
                lvl(4, 280.0, 7.0),
                lvl(8, 340.0, 10.0),
                lvl(16, 345.0, 30.0),
                lvl(32, 342.0, 60.0),
                lvl(64, 338.0, 90.0),
            ],
            matrix: None,
        });
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
        assert!(
            text.contains("Practical Sweet Spot"),
            "recommendation: {text}"
        );
        assert!(text.contains("8 concurrent users"), "{text}");
        assert!(text.contains("Maximum Usable"), "{text}");
        assert!(text.contains("16 concurrent users"), "{text}");
        assert!(text.contains("Unusable Beyond"), "{text}");
        assert!(text.contains("32+ concurrent users"), "{text}");
        assert!(
            text.contains("Pure Throughput Knee"),
            "knee reference: {text}"
        );
    }

    #[test]
    fn curve_marker_height_tracks_throughput() {
        // No envelope → both points are `●`; the 400 t/s marker must sit
        // on a strictly higher row (smaller index) than the 50 t/s one.
        // (h=12 → full layout: grid rows 1..=7; the legend row below is
        // excluded so its `●` is not counted as a data point.)
        let levels = vec![lvl(1, 50.0, 5.0), lvl(2, 400.0, 6.0)];
        let lines = build_curve_lines(Theme::default(), &levels, 40, 12, &None);
        let rows: Vec<Vec<char>> = lines
            .iter()
            .map(|l| l.to_string().chars().collect())
            .collect();
        let grid: Vec<&Vec<char>> = rows[1..8].iter().collect();
        let marker_rows: Vec<usize> = (0..rows[0].len())
            .filter(|&col| grid.iter().any(|r| r[col] == '◆'))
            .map(|col| grid.iter().position(|r| r[col] == '◆').unwrap())
            .collect();
        assert_eq!(marker_rows.len(), 2, "one marker per level");
        assert_ne!(
            marker_rows[0], marker_rows[1],
            "throughput must change the plotted height"
        );
    }

    // ── per-stream status (FIX 3) ───────────────────────────────────────

    #[test]
    fn per_stream_status_tracks_the_baseline() {
        // Baseline 100 t/s (the single-user rate).
        assert_eq!(per_stream_status(Theme::default(), 100.0, 100.0).0, "◆ optimal");
        assert_eq!(per_stream_status(Theme::default(), 90.0, 100.0).0, "◆ optimal"); // 0.90
        assert_eq!(per_stream_status(Theme::default(), 60.0, 100.0).0, "◆ good"); // 0.60
        assert_eq!(per_stream_status(Theme::default(), 40.0, 100.0).0, "▲ knee"); // 0.40
        assert_eq!(per_stream_status(Theme::default(), 20.0, 100.0).0, "✕ saturated"); // 0.20
                                                                     // No baseline → treated as optimal (ratio 1.0).
        assert_eq!(per_stream_status(Theme::default(), 50.0, 0.0).0, "◆ optimal");
    }

    #[test]
    fn matrix_shows_the_per_stream_column() {
        // A two-level sweep: the per-stream column is aggregate ÷ users.
        let app = crate::ui::app::App::new();
        app.sweep.store(SweepResult {
            levels: vec![lvl(1, 100.0, 5.0), lvl(4, 340.0, 20.0)],
            matrix: None,
        });
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
        // The column header is present.
        assert!(text.contains("PER-STREAM"), "per-stream column: {text}");
        // level 1: 100/1 = 100.0; level 4: 340/4 = 85.0.
        assert!(text.contains("100.0 t/s"), "per-stream at 1 user: {text}");
        assert!(text.contains("85.0 t/s"), "per-stream at 4 users: {text}");
    }
}
