//! View 1 — Live Monitor & Telemetry (blueprint §6), redesigned for
//! **remote** users.
//!
//! The remote operator does not sit on the GPU machine, so hardware panels
//! that read `N/A` off-box (the VRAM gauge, GPU clock, power, the per-stream
//! monitor matrix, the ITL braille histogram) are gone. Every pixel now shows
//! what a remote user actually cares about:
//!
//! * **Throughput** (the hero) — a large real-time tokens/sec block chart,
//!   top-left, with y/x axis labels and a `now | peak` value line;
//! * **Key metrics** — aggregate t/s, TTFT, ITL p50/p99, tokens, active
//!   streams, each with a one-line dimmed `ℹ` note;
//! * **Concurrency curve** (Engine B) — aggregate t/s vs parallel users,
//!   reusing View 2's block rendering, with the sweet spot (`●`) and the
//!   saturation knee (`▲`);
//! * **Capability scores** — horizontal bars for Reasoning (C2), NIAH (C1),
//!   Structured (C3), and Energy (D);
//! * **Event log** — a compact stream of the executor's real events.
//!
//! The layout adapts to the running engine: during Engine A only the hero +
//! key metrics show; Engine B adds the concurrency curve; C1–D show the
//! capability scores; after all complete every panel is shown. Panels that
//! don't apply to the current engine are **hidden, never rendered empty**.
//!
//! Every panel reads the shared state lock-free (the [`SeqStateSlot`] for the
//! sequence, the `MetricsState` snapshot for telemetry, the `ResultSlot`s for
//! completed engine results) — the render loop never blocks and never touches
//! the timing path (measurement-isolation invariant, blueprint §4).

use std::sync::Arc;

use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::config::EngineSelection;
use crate::engines::capability::CaseVerdict;
use crate::engines::sequence::{Engine, SeqPhase, SeqState};
use crate::metrics::state::{EngineMarker, MetricsSnapshot};
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, palette, style};
use crate::ui::views::concurrency::{build_curve_lines, curve_notes};

/// Render the Live view into `area`.
///
/// The row plan is built dynamically: the sequence header and the event log
/// are always present; the concurrency curve and the capability scores are
/// included only when they apply to the current engine (or the run is
/// complete). The throughput hero + key metrics row always takes the largest
/// share, so it stays the visual centerpiece.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    // Lock-free read of the latest published snapshot (the `Arc` is bound
    // first so the `&MetricsSnapshot` borrow outlives the frame).
    let snap = app.metrics.load();
    let m = snap.as_ref();
    let seq = app.seq.load();
    // Once the sequence completes the metrics pipeline freezes: the hero
    // shows its final, static state ("final" + a ✓ COMPLETE badge) and the
    // throughput graph stops animating (no more per-tick samples/averages).
    let frozen = app.metrics.is_frozen();

    let show_concurrency = should_show_concurrency(&seq, app);
    let show_capabilities = should_show_capabilities(&seq, app);
    // The structured-output detail sub-section (per-case checks + verdict)
    // needs more vertical room than a plain bar, so the capability panel
    // gets a larger share when C3 has run.
    let structured_detail = app.structured_slot.load().as_ref().is_some();

    // (panel id, constraint). The hero row is the largest slice; the
    // key-metrics panel carries a two-line-per-metric description, so it
    // gets the wider share of the row.
    let mut plan: Vec<(u8, Constraint)> = Vec::new();
    plan.push((0, Constraint::Length(3))); // sequence header + progress bar
    plan.push((1, Constraint::Percentage(38))); // throughput hero | key metrics
    if show_concurrency {
        plan.push((2, Constraint::Percentage(23))); // concurrency curve
    }
    if show_capabilities {
        plan.push((
            3,
            Constraint::Percentage(if structured_detail { 32 } else { 23 }),
        ));
    }
    plan.push((4, Constraint::Length(4))); // compact event log

    let constraints: Vec<Constraint> = plan.iter().map(|(_, c)| *c).collect();
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, (id, _)) in plan.iter().enumerate() {
        match *id {
            0 => render_sequence_header(rects[i], app, f),
            1 => render_throughput_row(rects[i], m, frozen, f),
            2 => render_concurrency_curve(rects[i], app, f),
            3 => render_capability_scores(rects[i], app, m, f),
            4 => render_log(rects[i], app, f),
            _ => {}
        }
    }
}

// ── Engine selection (FIX 1: unselected engines never appear) ─────────────

/// The engines the user selected, as the Live view should treat them:
///
/// * **during / after a run** — the run's own queue is authoritative
///   (it contains exactly the engines selected when the run started);
/// * **before any run** — the current Configuration form.
///
/// Every display panel on this view filters through this: an engine that
/// was not selected does not appear — not in the capability assessment,
/// not in the benchmark queue, not in any results panel (e.g. no
/// `N/A (no GPU telemetry)` line when Energy (D) is off).
fn selected(app: &App) -> EngineSelection {
    if let Some(seq) = app.seq.load().as_ref() {
        return EngineSelection {
            speed: seq.queue.contains(&Engine::Speed),
            concurrency: seq.queue.contains(&Engine::Concurrency),
            niah: seq.queue.contains(&Engine::Niah),
            reasoning: seq.queue.contains(&Engine::Reasoning),
            structured: seq.queue.contains(&Engine::Structured),
            hardware: seq.queue.contains(&Engine::Hardware),
        };
    }
    let c = &app.config;
    EngineSelection {
        speed: c.engine_speed,
        concurrency: c.engine_concurrency,
        niah: c.engine_niah,
        reasoning: c.engine_reasoning,
        structured: c.engine_structured,
        hardware: c.hardware,
    }
}

// ── Engine-adaptive panel visibility ───────────────────────────────────────

/// The concurrency curve shows while Engine B runs, after the whole
/// sequence completes (when B ran in it, or a sweep result is on
/// screen), or whenever a sweep result is already on screen (no sequence
/// in flight) — **and only when Engine B was selected** (FIX 1).
fn should_show_concurrency(seq: &Option<Arc<SeqState>>, app: &App) -> bool {
    if !selected(app).concurrency {
        return false;
    }
    let has_sweep = app
        .sweep
        .load()
        .as_ref()
        .as_ref()
        .is_some_and(|r| !r.levels.is_empty());
    match seq.as_deref() {
        Some(s) if s.phase == SeqPhase::AllComplete => {
            s.completed.iter().any(|(e, _)| *e == Engine::Concurrency) || has_sweep
        }
        Some(s) => s.engine == Engine::Concurrency,
        None => has_sweep,
    }
}

/// The capability scores show while any of the *selected* C/D engines
/// runs, after the whole sequence completes (when a selected capability
/// engine ran in it), or whenever a selected capability's result is
/// already on screen (no sequence in flight) — never for engines the
/// user did not select (FIX 1).
fn should_show_capabilities(seq: &Option<Arc<SeqState>>, app: &App) -> bool {
    let sel = selected(app);
    if !sel.niah && !sel.reasoning && !sel.structured && !sel.hardware {
        return false;
    }
    match seq.as_deref() {
        Some(s) if s.phase == SeqPhase::AllComplete => {
            s.completed.iter().any(|(e, _)| {
                matches!(
                    e,
                    Engine::Niah | Engine::Reasoning | Engine::Structured | Engine::Hardware
                )
            }) || (sel.niah && app.niah.load().as_ref().is_some())
                || (sel.reasoning && app.reasoning_slot.load().as_ref().is_some())
                || (sel.structured && app.structured_slot.load().as_ref().is_some())
        }
        Some(s) => matches!(
            s.engine,
            Engine::Niah | Engine::Reasoning | Engine::Structured | Engine::Hardware
        ),
        None => {
            (sel.niah && app.niah.load().as_ref().is_some())
                || (sel.reasoning && app.reasoning_slot.load().as_ref().is_some())
                || (sel.structured && app.structured_slot.load().as_ref().is_some())
                || (sel.hardware && app.hw.is_some())
        }
    }
}

// ── Throughput hero + key metrics (the top row) ────────────────────────────

/// The top row: the throughput hero chart (left, the *live* rolling
/// window) + the overall-metrics readout (right, *cumulative* across all
/// engines — FIX 1).
fn render_throughput_row(area: Rect, m: &MetricsSnapshot, frozen: bool, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        // 50/50 so the overall panel's `avg │ max │ p5` rows (the widest
        // content) fit without wrapping and clipping the footer (FIX 1).
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    render_throughput_hero(cols[0], m, frozen, f);
    render_overall_metrics(cols[1], m, f);
}

/// The hero: a large **real-time** aggregate tokens/sec block chart
/// (FIX 2). One column per sample (right-aligned, newest at the right
/// edge), each a vertical run of `█` graded green (high) → yellow (medium)
/// → red (low) against the *auto-scaled* window maximum (with headroom),
/// a `now │ peak │ avg` header line, a dashed horizontal line at the
/// window average, and vertical markers at engine transitions. The
/// y-axis auto-scales to the data (never a fixed axis); the x-axis spans
/// the *actual* data window. An empty series shows
/// "Awaiting first tokens…" (never a blank panel).
fn render_throughput_hero(area: Rect, m: &MetricsSnapshot, frozen: bool, f: &mut Frame) {
    // While the run is live the hero is a real-time chart ("now"); once the
    // sequence completes the metrics pipeline freezes and the hero shows its
    // final, static state ("final" + a ✓ COMPLETE badge, green border).
    let title = if frozen {
        "LIVE THROUGHPUT — ✓ COMPLETE (final, frozen)"
    } else {
        "LIVE THROUGHPUT — real-time generation speed (tokens/sec)"
    };
    let block = theme::block(
        theme::panel_title(title),
        if frozen {
            style::value_ok()
        } else {
            style::active_border()
        },
    );
    if area.width < 8 || area.height < 5 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }

    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });

    let series = &m.throughput_series;
    let current = m.aggregate_tps;
    let peak = series.iter().cloned().fold(0.0_f64, f64::max);
    let avg = if series.is_empty() {
        0.0
    } else {
        series.iter().sum::<f64>() / series.len() as f64
    };

    // The header line: the *live* now / peak / avg for the window — or, once
    // frozen, the *final* window average + peak (the run is over, so there is
    // no live "now" and the instantaneous rate would read 0.0).
    // Information hierarchy: the live `now` is the primary (bright) value,
    // `peak` the warning, `avg` the dimmed secondary.
    let value_line = if frozen {
        Line::from(vec![
            Span::styled("final ", style::label()),
            Span::styled(fmt::format_rate(avg), style::value_ok()),
            Span::styled("  │  peak ", style::footer()),
            Span::styled(fmt::format_rate(peak), style::value_warn()),
        ])
    } else {
        Line::from(vec![
            Span::styled("now ", style::label()),
            Span::styled(fmt::format_rate(current), style::value()),
            Span::styled("  │  peak ", style::footer()),
            Span::styled(fmt::format_rate(peak), style::value_warn()),
            Span::styled("  │  avg ", style::footer()),
            Span::styled(fmt::format_rate(avg), style::value_secondary()),
        ])
    };

    // Engine-transition markers positioned in the current window: the
    // window starts `elapsed − (len−1)` seconds ago (one sample/sec).
    let window_start = m.elapsed_sec - (series.len().saturating_sub(1)) as f64;
    let markers: Vec<(f64, String)> = m
        .engine_markers
        .iter()
        .map(|mk: &EngineMarker| (mk.at_sec, mk.label.clone()))
        .collect();

    // The chart fills the inner area below the header line; it carries its
    // own axes, the average line, and the transition markers. Once frozen
    // it renders the *final* state: no new samples ever arrive, and a
    // `✓ COMPLETE` overlay marks it as the finished run's chart.
    let chart = build_throughput_chart(
        series,
        inner.width,
        inner.height.saturating_sub(1),
        avg,
        &markers,
        window_start,
        frozen,
    );
    let mut lines = vec![value_line];
    lines.extend(chart);

    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// Build the throughput block chart as lines of styled single-character
/// spans (FIX 2): a **y-axis that auto-scales to the data** (with 10%
/// headroom — never a fixed axis), one `█` bar per sample column
/// (right-aligned, newest at the right edge, graded green→yellow→red), a
/// dashed horizontal line at the **window average**, **vertical markers at
/// engine transitions**, and an x-axis spanning the *actual* data window
/// (`0 … N−1 s`). An empty series renders "Awaiting first tokens…".
/// When `frozen` (the run is complete), a **`✓ COMPLETE` overlay** is
/// drawn at the top of the plot — the final line is the final line.
/// Pure over its inputs (unit-testable, no terminal).
fn build_throughput_chart(
    series: &[f64],
    w: u16,
    h: u16,
    avg: f64,
    markers: &[(f64, String)],
    window_start_sec: f64,
    frozen: bool,
) -> Vec<Line<'static>> {
    const Y_AXIS_W: usize = 5;
    let w = w as usize;
    let h = h as usize;
    if w < Y_AXIS_W + 3 || h < 3 {
        return vec![Line::from("chart too small")];
    }
    // No samples yet: a friendly prompt, never a blank chart.
    if series.is_empty() {
        return vec![Line::from(Span::styled(
            "Awaiting first tokens…",
            style::info(),
        ))];
    }

    let plot_w = w - Y_AXIS_W;
    let plot_h = h - 1; // bottom row reserved for the x-axis
                        // Auto-scale to the window maximum with 10% headroom (a `max(1.0)`
                        // floor keeps a tiny signal from blowing the axis up to absurdity).
    let max = series.iter().cloned().fold(0.0_f64, f64::max) * 1.1;
    let max = max.max(1.0);

    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w]; h];

    // y-axis labels: top = max, middle = max/2, bottom = 0.
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
    place_y(0, max);
    place_y(plot_h / 2, max / 2.0);
    place_y(plot_h.saturating_sub(1), 0.0);

    // One bar per plot column; right-aligned so the newest sample sits at the
    // right edge. Columns left of the filled region stay blank.
    for col in 0..plot_w {
        let idx = series.len().saturating_sub(plot_w - col);
        if idx >= series.len() {
            continue;
        }
        let v = series[idx];
        let ratio = (v / max).clamp(0.0, 1.0);
        let height = (ratio * plot_h as f64).round() as usize;
        let color = throughput_color(ratio);
        for row in 0..height {
            let grid_row = plot_h.saturating_sub(1).saturating_sub(row);
            let grid_col = Y_AXIS_W + col;
            if grid_row < h && grid_col < w {
                grid[grid_row][grid_col] = ('█', Some(color));
            }
        }
    }

    // Engine-transition markers: a vertical `┊` line at the marker's
    // position in the current window, with a short label at the top.
    for (at_sec, label) in markers {
        let col = (*at_sec - window_start_sec).round() as i64;
        if col < 0 || (col as usize) >= plot_w {
            continue;
        }
        let c = Y_AXIS_W + col as usize;
        if c < w {
            for cell in grid.iter_mut().take(plot_h) {
                cell[c] = ('┊', Some(palette::ACCENT));
            }
        }
        for (i, ch) in label.chars().take(4).enumerate() {
            let cc = c + 1 + i;
            if cc < w {
                grid[0][cc] = (ch, Some(palette::ACCENT));
            }
        }
    }

    // A dashed horizontal line at the window average, tagged in the gutter.
    if avg > 0.0 {
        let ratio = (avg / max).clamp(0.0, 1.0);
        let row = plot_h
            .saturating_sub(1)
            .saturating_sub((ratio * plot_h as f64).round() as usize)
            .min(plot_h.saturating_sub(1));
        for cell in &mut grid[row][Y_AXIS_W..] {
            *cell = ('┄', Some(palette::HIGHLIGHT));
        }
        for (i, ch) in "avg".chars().enumerate() {
            let cc = Y_AXIS_W.saturating_sub(3) + i;
            if cc < Y_AXIS_W {
                grid[row][cc] = (ch, Some(palette::HIGHLIGHT));
            }
        }
    }

    // x-axis: a baseline + time labels spanning the *actual* window
    // (0 … N−1 s, where N = series.len()), skipping overlaps.
    let span = (series.len() - 1).max(1);
    for cell in &mut grid[h - 1][Y_AXIS_W..] {
        *cell = ('─', Some(palette::MUTED));
    }
    grid[h - 1][Y_AXIS_W] = ('├', Some(palette::MUTED));
    let mut label_end: i64 = -1;
    for frac in [0.0, 0.25, 0.5, 0.75, 1.0] {
        let text = format!("{}s", (frac * span as f64).round() as i64);
        let center = Y_AXIS_W as i64 + (frac * (plot_w as f64 - 1.0)).round() as i64;
        let mut start = (center - text.len() as i64 / 2).max(Y_AXIS_W as i64);
        start = start.min((w as i64) - (text.len() as i64));
        if start > label_end {
            for (i, ch) in text.chars().enumerate() {
                let col = (start + i as i64) as usize;
                if col < w {
                    grid[h - 1][col] = (ch, Some(palette::MUTED));
                }
            }
            label_end = start + text.len() as i64;
        }
    }

    // The frozen overlay (FIX 2): `✓ COMPLETE` at the top of the plot —
    // the chart is the run's final state and will not change again.
    if frozen {
        let text = "✓ COMPLETE";
        let start = (w as i64 - text.len() as i64).max(Y_AXIS_W as i64) as usize;
        for (i, ch) in text.chars().enumerate() {
            let col = start + i;
            if col < w {
                grid[0][col] = (ch, Some(palette::OK));
            }
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

/// The **OVERALL** metrics panel (FIX 1): cumulative statistics across
/// *every* benchmark engine that has run — `max` / `avg` / `p5` per metric,
/// the total tokens generated (the sum of server-reported
/// `usage.completion_tokens`, **not** the SSE frame count), the
/// active-stream average/peak, and the total elapsed time. This contrasts
/// with the hero chart, which shows the *live* rolling window. Every number
/// is read lock-free from the snapshot's [`OverallStats`]
/// (measurement-isolation invariant, blueprint §4).
///
/// **Label spacing** (user feedback 6): every row is `Label: value` — the
/// label (colon included) is padded to a fixed width so all values start in
/// the same column. No more `Total Tokens33,108` or `Streamavg` collisions.
fn render_overall_metrics(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let o = &m.overall;
    let block = theme::block(
        theme::panel_title("OVERALL METRICS (all engines)"),
        style::border(),
    );
    if area.width < 10 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    // Throughput rows use the compact stat format (one decimal under 1000,
    // none above); latency rows (TTFT / ITL) format each value with its own
    // unit (`312ms` / `1.2s`), so they carry no trailing unit column.
    let gen = stat_triple(o.gen.avg, o.gen.max, o.gen.p5, fmt_stat);
    let prompt = stat_triple(o.prompt.avg, o.prompt.max, o.prompt.p5, fmt_stat);
    let ttft = stat_triple(
        o.ttft.avg * 1000.0,
        o.ttft.max * 1000.0,
        o.ttft.p5 * 1000.0,
        fmt_latency,
    );
    let itl50 = stat_triple(o.itl_p50.avg, o.itl_p50.max, o.itl_p50.p5, fmt_latency);
    let itl99 = stat_triple(o.itl_p99.avg, o.itl_p99.max, o.itl_p99.p5, fmt_latency);
    let lines = vec![
        stat_row("Gen Throughput", &gen.0, &gen.1, &gen.2, Some("t/s")),
        stat_row(
            "Prompt Throughput",
            &prompt.0,
            &prompt.1,
            &prompt.2,
            Some("t/s"),
        ),
        stat_row("TTFT", &ttft.0, &ttft.1, &ttft.2, None),
        stat_row("ITL p50", &itl50.0, &itl50.1, &itl50.2, None),
        stat_row("ITL p99", &itl99.0, &itl99.1, &itl99.2, None),
        simple_row("Total Tokens", &fmt::format_tokens(o.total_tokens)),
        simple_row(
            "Streams",
            &format!(
                "avg {a:<5} │ max {b}",
                a = format!("{:.1}", o.active_avg),
                b = o.active_max
            ),
        ),
        simple_row("Total Duration", &fmt::format_duration(o.duration_sec)),
        Line::from(Span::styled(
            "ℹ All engines. p5 = 5th percentile (worst 5%).",
            style::info(),
        )),
    ];
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The label cell: `{label}:` padded to 20 columns (the longest label,
/// `Prompt Throughput:`, is 18) so every value in the panel starts in the
/// same column — the user-reported `Total Tokens33,108` / `Streamavg`
/// collisions are gone.
fn label_cell(label: &str) -> Span<'static> {
    Span::styled(format!("{:<20}", format!("{label}:")), style::label())
}

/// One `Label:   avg X │ max Y │ p5 Z [unit]` row for the overall panel
/// (the pre-formatted value strings come from [`stat_triple`], which
/// yields `--` across the board when the metric has no samples yet).
fn stat_row(label: &str, a: &str, m: &str, p: &str, unit: Option<&str>) -> Line<'static> {
    let mut spans = vec![
        label_cell(label),
        Span::styled(format!("avg {a:<5} │ max {m:<5} │ p5 {p}"), style::value()),
    ];
    if let Some(u) = unit {
        spans.push(Span::styled(format!("  {u}"), style::footer()));
    }
    Line::from(spans)
}

/// One `Label:   value` row (total tokens, streams, duration).
fn simple_row(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        label_cell(label),
        Span::styled(value.to_string(), style::value()),
    ])
}

/// Format `(avg, max, p5)` with `fmt`; all `--` when the metric has no
/// samples yet.
fn stat_triple(avg: f64, max: f64, p5: f64, fmt: fn(f64) -> String) -> (String, String, String) {
    let none = avg <= 0.0 && max <= 0.0;
    if none {
        ("--".to_string(), "--".to_string(), "--".to_string())
    } else {
        (fmt(avg), fmt(max), fmt(p5))
    }
}

/// Adaptive latency format: `12.1ms` under 100 ms, `312ms` under 1 s,
/// `1.2s` at/above 1 s — each value carries its own unit, so the latency
/// rows (TTFT / ITL) need no trailing unit column.
fn fmt_latency(v_ms: f64) -> String {
    if v_ms >= 1000.0 {
        format!("{:.1}s", v_ms / 1000.0)
    } else if v_ms >= 100.0 {
        format!("{:.0}ms", v_ms)
    } else {
        format!("{v_ms:.1}ms")
    }
}

/// A stat value: one decimal under 1000, none above (keeps rows compact).
fn fmt_stat(v: f64) -> String {
    if v >= 1000.0 {
        format!("{:.0}", v)
    } else {
        format!("{v:.1}")
    }
}

// ── Concurrency curve (Engine B) ───────────────────────────────────────────

/// The prominent concurrency panel: aggregate t/s vs parallel users —
/// View 2's labelled curve (green below the sweet spot, yellow at it,
/// red past the knee, `▲ KNEE @ n` annotated) plus the plain-language
/// "what to do with this" note. While Engine B is mid-sweep (no result
/// published yet) it shows an in-progress note; with no sweep at all it
/// shows the run-a-sweep hint.
fn render_concurrency_curve(area: Rect, app: &App, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title("CONCURRENCY CURVE — t/s vs parallel users"),
        style::border(),
    );
    if area.width < 8 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });

    let sweep = app.sweep.load();
    let lines = sweep
        .as_ref()
        .as_ref()
        .filter(|r| !r.levels.is_empty())
        .map(|r| {
            let env = r.envelope();
            let notes = curve_notes(&r.levels);
            // The plot gets whatever height the notes leave (it
            // self-degrades to a compact form in the small Live panel).
            let plot_h = (inner.height as i64 - notes.len() as i64).max(4) as u16;
            let mut ls = build_curve_lines(&r.levels, inner.width, plot_h, &env);
            ls.extend(notes);
            ls
        })
        .unwrap_or_else(|| {
            let running_b =
                app.seq.load().as_deref().is_some_and(|s| {
                    s.engine == Engine::Concurrency && s.phase == SeqPhase::Running
                });
            vec![Line::from(Span::styled(
                if running_b {
                    "Engine B sweep in progress — the curve plots when it completes."
                } else {
                    "Run a sweep (Engine B) to plot the curve."
                }
                .to_string(),
                style::footer(),
            ))]
        });
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

// ── Capability scores (C1 / C2 / C3 / D) ───────────────────────────────────

/// One capability score: a label, a horizontal `[████░░]` bar (0–100%), a
/// value detail, a one-line dimmed `ℹ` explanation, and (for poor scores)
/// a `⚠` warning. Only engines that have *actually run* appear.
struct CapScore {
    label: &'static str,
    /// `0.0..=100.0` bar fill, or `None` for a non-percentage metric (the
    /// bar stays empty and the detail carries the value).
    pct: Option<f64>,
    detail: String,
    detail_style: Style,
    color: Color,
    info: &'static str,
    /// A `⚠` warning for a score that means something is wrong.
    warn: Option<&'static str>,
}

/// Score → color: green (>80%), yellow (50–80%), red (<50%), gray (N/A /
/// non-percentage).
fn score_color(pct: Option<f64>) -> Color {
    match pct {
        Some(p) if p > 80.0 => palette::OK,
        Some(p) if p >= 50.0 => palette::WARN,
        Some(_) => palette::ERR,
        None => palette::MUTED,
    }
}

/// The capability scores panel: color-coded horizontal bars for every
/// capability engine that has run (Reasoning C2, Long Context C1,
/// Structured C3, Energy D), each with a `ℹ` explanation, a `⚠` warning
/// when the score is poor, and an **OVERALL** practical summary line at
/// the bottom. Engines that haven't run are omitted (never shown empty).
fn render_capability_scores(area: Rect, app: &App, m: &MetricsSnapshot, f: &mut Frame) {
    let block = theme::block(theme::panel_title("CAPABILITY ASSESSMENT"), style::border());
    if area.width < 10 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let sel = selected(app);
    let scores = build_capability_scores(app, m, &sel);
    let overall = Line::from(Span::styled(capability_overall(&scores), style::value()));
    let mut lines: Vec<Line> = Vec::new();
    // The OVERALL summary sits at the bottom (the verdict after the
    // evidence) — except in a squeezed panel, where it moves up front so
    // it is never clipped away.
    let tall = area.height >= 12;
    if !tall {
        lines.push(overall.clone());
        lines.push(Line::raw(""));
    }
    for s in &scores {
        lines.extend(build_cap_lines(s));
    }
    // The structured-output detail sub-section (per-case checks, the
    // truncated actual output for failures, speed impact, and the
    // practical verdict) — appended only when C3 has run *and was
    // selected* (FIX 1).
    let detail = build_structured_detail_lines(app, &sel);
    if !detail.is_empty() {
        lines.push(Line::raw(""));
        lines.extend(detail);
    }
    if tall {
        let divider = "─".repeat(area.width.saturating_sub(4) as usize);
        lines.push(Line::from(Span::styled(divider, style::border())));
        lines.push(overall);
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// Gather the capability scores from their lock-free result slots — only
/// the engines that were **selected and have actually run** (FIX 1: an
/// unselected engine never appears, even if a stale result is in its
/// slot). Pure over its inputs, unit-testable.
fn build_capability_scores(app: &App, m: &MetricsSnapshot, sel: &EngineSelection) -> Vec<CapScore> {
    let mut v: Vec<CapScore> = Vec::with_capacity(4);

    // Reasoning (C2) — N/M solved as a percentage.
    if sel.reasoning {
        if let Some(r) = app
            .reasoning_slot
            .load()
            .as_ref()
            .as_ref()
            .filter(|r| r.score.total > 0)
        {
            let pct = r.score.solved as f64 / r.score.total as f64 * 100.0;
            v.push(CapScore {
                label: "Reasoning",
                pct: Some(pct),
                detail: format!(
                    "{}  ({}/{} solved)",
                    fmt::format_pct(pct),
                    r.score.solved,
                    r.score.total
                ),
                detail_style: style::value(),
                color: score_color(Some(pct)),
                info: "Math, logic, code problems. Measures analytical ability.",
                warn: (pct < 50.0).then_some("LOW: weak analytical problem-solving."),
            });
        }
    }

    // Long Context (C1, NIAH) — retrieved/total cells as a percentage.
    if sel.niah {
        if let Some(r) = app.niah.load().as_ref() {
            let (retrieved, total) = r.accuracy();
            let pct = if total > 0 {
                retrieved as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            v.push(CapScore {
                label: "Long Context",
                pct: Some(pct),
                detail: format!("{}  ({retrieved}/{total})", fmt::format_pct(pct)),
                detail_style: style::value(),
                color: score_color(Some(pct)),
                info: "Retrieval from large documents. Critical for RAG / chat history.",
                warn: (pct < 50.0).then_some("LOW: the model loses information in long contexts."),
            });
        }
    }

    // Structured Out (C3) — the score across the three schema cases
    // (compliant / partial / failed), with the practical verdict carried by
    // the detail sub-section below.
    if sel.structured {
        if let Some(r) = app.structured_slot.load().as_ref() {
            let (c, _p, _f) = r.score();
            let total = r.cases.len();
            let pct = if total > 0 {
                c as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            v.push(CapScore {
                label: "Structured Out",
                pct: Some(pct),
                detail: r.score_label(),
                detail_style: if c == total {
                    style::value_ok()
                } else if c == 0 {
                    style::value_err()
                } else {
                    style::value_warn()
                },
                color: score_color(Some(pct)),
                info: "JSON schema adherence (3 cases). Required for API / agent tool-calling.",
                warn: (c < total).then_some("See the detail below for which cases fail."),
            });
        }
    }

    // Energy Efficiency (D) — J/token; N/A without local GPU telemetry.
    // `energy_line` returns `None` when D was not selected (FIX 1).
    if let Some(line) = energy_line(m, app, sel) {
        let na = line.starts_with("N/A");
        v.push(CapScore {
            label: "Energy Efficiency",
            pct: None,
            detail: line,
            detail_style: if na { style::footer() } else { style::value() },
            color: palette::MUTED,
            info: "Joules per token. Requires a local GPU with driver support.",
            warn: None,
        });
    }

    v
}

/// Render one capability score as two lines:
/// `Label   [████████████████████]  detail  ⚠ warning`
/// `        ℹ what it measures`
fn build_cap_lines(s: &CapScore) -> Vec<Line<'static>> {
    const BAR_W: usize = 20;
    let filled = s
        .pct
        .map(|p| (p.clamp(0.0, 100.0) / 100.0 * BAR_W as f64).round() as usize)
        .unwrap_or(0);
    let mut bar = String::with_capacity(BAR_W);
    for i in 0..BAR_W {
        bar.push(if i < filled { '█' } else { '░' });
    }
    let mut line1: Vec<Span> = vec![
        Span::styled(format!("{:<16}", s.label), style::label()),
        Span::styled(format!("[{bar}] "), Style::default().fg(s.color)),
        Span::styled(s.detail.clone(), s.detail_style),
    ];
    if let Some(w) = s.warn {
        line1.push(Span::styled(format!("  ⚠ {w}"), style::value_warn()));
    }
    vec![
        Line::from(line1),
        Line::from(Span::styled(
            format!("                ℹ {}", s.info),
            style::info(),
        )),
    ]
}

/// The `✓ label  ✗ label  …` check spans for one structured case.
fn case_check_spans(case: &crate::engines::capability::StructuredCaseResult) -> Vec<Span<'static>> {
    case.checks
        .iter()
        .map(|c| {
            let glyph = if c.passed { '✓' } else { '✗' };
            let color = if c.passed { palette::OK } else { palette::ERR };
            Span::styled(format!(" {glyph} {} ", c.label), Style::default().fg(color))
        })
        .collect()
}

/// The structured-output detail sub-section for the Live capability panel:
/// a header, one line per case (its checks + verdict glyph), the
/// (truncated) actual output for any non-compliant case, the speed-impact
/// line, and the auto-generated practical verdict. `empty` when C3 has not
/// run **or was not selected** (FIX 1: the render path never shows an
/// empty box, and never shows an unselected engine's detail).
fn build_structured_detail_lines(app: &App, sel: &EngineSelection) -> Vec<Line<'static>> {
    if !sel.structured {
        return Vec::new();
    }
    let binding = app.structured_slot.load();
    let Some(r) = binding.as_ref() else {
        return Vec::new();
    };
    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::from(Span::styled(
        "STRUCTURED DETAIL (C3) — schema adherence",
        style::muted_title(),
    )));
    for (i, case) in r.cases.iter().enumerate() {
        let (glyph, color) = match case.verdict {
            CaseVerdict::Compliant => ("✓", palette::OK),
            CaseVerdict::Partial => ("⚠", palette::WARN),
            CaseVerdict::Failed => ("✗", palette::ERR),
        };
        let mut spans = vec![
            Span::styled(
                format!("  Case {} ({}):  ", i + 1, case.name),
                style::label(),
            ),
            Span::styled(glyph.to_string(), Style::default().fg(color)),
        ];
        spans.extend(case_check_spans(case));
        lines.push(Line::from(spans));
        // Show the actual output (truncated to 3 lines) for any case that is
        // not fully compliant, so the user sees *what* went wrong.
        if !case.is_compliant() && !case.output.trim().is_empty() {
            lines.push(Line::from(Span::styled(
                "    actual output:",
                style::info(),
            )));
            for ol in case.output.split('\n').take(3) {
                let trimmed = ol.trim();
                if trimmed.is_empty() {
                    continue;
                }
                lines.push(Line::from(Span::styled(
                    format!("    │ {trimmed}"),
                    style::info(),
                )));
            }
        }
    }
    // Speed impact: free-form vs constrained decode.
    lines.push(Line::from(vec![
        Span::styled("  Speed: ", style::label()),
        Span::styled(
            format!(
                "free-form {:.1} → constrained {:.1} t/s",
                r.free_tps, r.constrained_tps
            ),
            style::value(),
        ),
        Span::styled(format!(" ({:+.1}%)", r.penalty_pct), style::footer()),
    ]));
    lines.push(Line::from(Span::styled(
        "  ℹ JSON mode adds slight overhead due to format constraints.",
        style::info(),
    )));
    // The auto-generated practical verdict.
    let verdict = r.verdict_line();
    let vcolor = if verdict.starts_with('✓') {
        palette::OK
    } else if verdict.starts_with('✗') {
        palette::ERR
    } else {
        palette::WARN
    };
    lines.push(Line::from(Span::styled(
        format!("  VERDICT: {verdict}"),
        Style::default().fg(vcolor).add_modifier(Modifier::BOLD),
    )));
    lines
}

/// The **OVERALL** practical summary under the capability bars: what the
/// scores mean for actually using the model (pure — unit-testable).
fn capability_overall(scores: &[CapScore]) -> String {
    if scores.is_empty() {
        return "Run benchmarks to see capability scores".to_string();
    }
    let mut parts: Vec<String> = Vec::new();
    let mut caveats: Vec<String> = Vec::new();
    for s in scores {
        match s.label {
            "Reasoning" => {
                if let Some(p) = s.pct {
                    parts.push(if p >= 80.0 {
                        "strong reasoning".to_string()
                    } else if p >= 50.0 {
                        "solid reasoning".to_string()
                    } else {
                        "weak reasoning".to_string()
                    });
                }
            }
            "Long Context" => {
                if let Some(p) = s.pct {
                    if p >= 80.0 {
                        parts.push("good long-context retention".to_string());
                    } else if p >= 50.0 {
                        parts.push("moderate long-context retention".to_string());
                    } else {
                        parts.push("weak long-context".to_string());
                        caveats.push("Not suitable for RAG / long chat history".to_string());
                    }
                }
            }
            "Structured Out" => {
                if let Some(p) = s.pct {
                    if p >= 100.0 {
                        parts.push("reliable JSON output".to_string());
                    } else if p >= 66.0 {
                        parts.push("simple-only JSON output".to_string());
                        caveats.push("Unreliable for complex schemas".to_string());
                    } else {
                        parts.push("unreliable JSON output".to_string());
                        caveats.push(
                            "Not safe for API / agent tool-calling without prompt workarounds"
                                .to_string(),
                        );
                    }
                }
            }
            _ => {}
        }
    }
    let mut out = if parts.is_empty() {
        "OVERALL: no scored capabilities yet.".to_string()
    } else {
        format!("OVERALL: {}. ", parts.join(", "))
    };
    if !caveats.is_empty() {
        out.push_str(&caveats.join("; "));
        out.push('.');
    }
    out
}

// ── Engine D energy line (shared by capability scores) ─────────────────────

/// The Engine D data line, when one should be shown: the sequence's
/// sampling summary wins; otherwise the live hardware-poller telemetry
/// (`N/A` without GPU telemetry). `None` when Engine D was **not
/// selected** (FIX 1) — an unselected engine never appears, so no
/// `N/A (no GPU telemetry)` line on a run without D.
fn energy_line(m: &MetricsSnapshot, app: &App, sel: &EngineSelection) -> Option<String> {
    if !sel.hardware {
        return None;
    }
    if let Some(st) = app.seq.load().as_ref() {
        if let Some((_, summary)) = st.completed.iter().find(|(e, _)| *e == Engine::Hardware) {
            return Some(summary.clone());
        }
    }
    if app.hw.is_some() {
        return Some(if m.joules_per_token > 0.0 {
            format!("{:.3} J/token · {:.0} W", m.joules_per_token, m.power_w)
        } else {
            "N/A (no GPU telemetry)".to_string()
        });
    }
    // D was selected but no telemetry source is live (e.g. the poller was
    // not started on this host, or D is still running): show the N/A line
    // so the user knows D was requested.
    Some("N/A (no GPU telemetry)".to_string())
}

// ── Benchmark Sequence: header + progress bar ──────────────────────────────

/// The 10-frame braille spinner (the "alive" pulse of the running state).
const SPINNERS: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

/// Top: the sequence header — current engine + phase + progress line,
/// with a visual progress bar. The border pulses (accent) while an
/// engine runs, turns green on completion, and magenta when the whole
/// sequence is done.
fn render_sequence_header(area: Rect, app: &App, f: &mut Frame) {
    if area.width < 12 || area.height < 3 {
        return;
    }

    // (marker, marker style, text, text style, bar ratio, border style)
    let (marker, marker_style, text, text_style, ratio, border_style) = match app.seq.load() {
        None => (
            "○".to_string(),
            style::footer(),
            "No benchmark running — launch from Setup or press r".to_string(),
            style::footer(),
            0.0,
            style::border(),
        ),
        Some(state) => seq_header_parts(&state, app.tick),
    };

    // The progress bar: `[████████░░░░]  40%` — sized to the remaining
    // width after the text (never negative; small terminals drop it).
    let inner_width = (area.width - 2) as usize;
    let bar_width = inner_width
        .saturating_sub(text.chars().count() + marker.len() + 8)
        .clamp(0, 40);

    let mut spans: Vec<Span> = vec![
        Span::raw(" "),
        Span::styled(marker, marker_style),
        Span::raw(" "),
        Span::styled(text, text_style),
    ];
    if bar_width > 0 {
        let filled = (ratio.clamp(0.0, 1.0) * bar_width as f64).round() as usize;
        let bar_color = match ratio {
            1.0 => palette::HIGHLIGHT,
            _ => palette::ACCENT,
        };
        let mut bar = String::with_capacity(bar_width + 3);
        bar.push('[');
        for i in 0..bar_width {
            bar.push(if i < filled { '█' } else { '░' });
        }
        bar.push(']');
        spans.push(Span::raw("  "));
        spans.push(Span::styled(bar, Style::default().fg(bar_color)));
        spans.push(Span::styled(
            format!("{:4.0}%", ratio * 100.0),
            style::value(),
        ));
    }

    f.render_widget(
        Paragraph::new(Line::from(spans))
            .block(theme::block(
                theme::panel_title("BENCHMARK SEQUENCE"),
                border_style,
            ))
            .style(Style::default().bg(Color::Black)),
        area,
    );
}

/// The header's (marker, marker style, text, text style, bar ratio,
/// border style) for a [`SeqState`].
fn seq_header_parts(state: &SeqState, tick: u64) -> (String, Style, String, Style, f64, Style) {
    match state.phase {
        SeqPhase::Idle => (
            "○".to_string(),
            style::footer(),
            "Idle — press r to run the selected engines".to_string(),
            style::footer(),
            0.0,
            style::border(),
        ),
        SeqPhase::Running => {
            let spinner = SPINNERS[(tick as usize / 6) % SPINNERS.len()].to_string();
            let progress_text = state
                .progress
                .as_ref()
                .map(|p| p.label())
                .unwrap_or_else(|| "starting…".to_string());
            let ratio = state.progress.as_ref().map(|p| p.fraction()).unwrap_or(0.0);
            (
                spinner,
                Style::default()
                    .fg(palette::ACCENT)
                    .add_modifier(ratatui::style::Modifier::BOLD),
                format!(
                    "{} — {} — {progress_text}",
                    state.engine.title(),
                    SeqPhase::Running.label()
                ),
                style::value(),
                ratio,
                style::active_border(),
            )
        }
        SeqPhase::Complete => (
            "✓".to_string(),
            style::value_ok(),
            format!(
                "{} — {} — {}",
                state.engine.title(),
                SeqPhase::Complete.label(),
                state.summary
            ),
            style::value(),
            1.0,
            style::value_ok(),
        ),
        SeqPhase::AllComplete => (
            "✓".to_string(),
            style::highlight(),
            format!("ALL BENCHMARKS COMPLETE — {}", state.summary),
            style::value(),
            1.0,
            style::highlight(),
        ),
    }
}

// ── Event log (bottom) ─────────────────────────────────────────────────────

/// Bottom: the scrolling log / event stream — the executor's *real* events
/// (engine starts, completions, summaries), drained from the mpsc pipe on
/// the tick path (never pre-generated). Kept compact (a few lines).
fn render_log(area: Rect, app: &App, f: &mut Frame) {
    f.render_widget(
        Paragraph::new(Text::from(app.log.clone()))
            .block(theme::block(
                theme::panel_title("EVENT LOG"),
                style::border(),
            ))
            .wrap(Wrap { trim: true }),
        area,
    );
}

// ── Shared formatting helpers ──────────────────────────────────────────────

/// Color gradient for the throughput chart: green (high) → yellow (medium)
/// → red (low), relative to the rolling-window maximum.
pub(crate) fn throughput_color(ratio: f64) -> Color {
    if ratio >= 0.75 {
        palette::OK
    } else if ratio >= 0.40 {
        palette::WARN
    } else {
        palette::ERR
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engines::sequence::EngineProgress;

    // ── throughput chart ─────────────────────────────────────────────────

    #[test]
    fn throughput_chart_guard_degenerate_areas() {
        let lines = build_throughput_chart(&[1.0, 2.0], 3, 2, 1.0, &[], 0.0, false);
        assert_eq!(lines[0].to_string(), "chart too small");
    }

    #[test]
    fn throughput_chart_plots_bars_and_axes() {
        // A 60-sample window → the x-axis spans 0…59s (the *actual* data
        // window, not a fixed 60s).
        let series: Vec<f64> = (0..60)
            .map(|i| 100.0 + 200.0 * ((i as f64) * 0.3).sin())
            .collect();
        let lines = build_throughput_chart(&series, 40, 10, 150.0, &[], 0.0, false);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('█'), "bars rendered: {text}");
        // x-axis time labels span the real window.
        assert!(text.contains("0s"), "x-axis start: {text}");
        assert!(text.contains("59s"), "x-axis end: {text}");
        // y-axis auto-scales to the data with headroom (300 → 330).
        assert!(text.contains("330"), "y-axis max: {text}");
    }

    #[test]
    fn throughput_chart_empty_series_shows_awaiting() {
        // No samples yet → "Awaiting first tokens…", never a blank chart.
        let lines = build_throughput_chart(&[], 30, 8, 0.0, &[], 0.0, false);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            text.contains("Awaiting first tokens"),
            "awaiting prompt: {text}"
        );
    }

    #[test]
    fn overall_metrics_panel_labels_every_metric() {
        // Rendered tall (120x52) so every overall row fits; on a short
        // terminal the least critical rows clip gracefully.
        let app = App::new();
        let text = render_live_text(&app, 120, 52);
        for label in [
            "Gen Throughput",
            "Prompt Throughput",
            "TTFT",
            "ITL p50",
            "ITL p99",
            "Total Tokens",
            "Streams",
            "Total Duration",
        ] {
            assert!(text.contains(label), "missing metric {label}: {text}");
        }
        // The panel is clearly *cumulative*, with the avg/max/p5 legend.
        assert!(text.contains("OVERALL METRICS"), "overall title: {text}");
        assert!(text.contains("percentile"), "p5 legend: {text}");
        assert!(text.contains('ℹ'), "info note: {text}");
    }

    #[test]
    fn overall_metric_label_cells_pad_to_20_with_trailing_spaces() {
        // The user-reported bug: labels ran straight into their values
        // ("Total Tokens33,108", "Streamavg"). Every label cell must be
        // exactly 20 columns — the label, a colon directly after it, and
        // at least two spaces of padding before the value column starts.
        for label in [
            "Gen Throughput",
            "Prompt Throughput",
            "TTFT",
            "ITL p50",
            "ITL p99",
            "Total Tokens",
            "Streams",
            "Total Duration",
        ] {
            let text = label_cell(label).content.as_ref().to_string();
            assert_eq!(text.len(), 20, "{label}: {text:?}");
            let colon = text.find(':').expect("the label cell carries a colon");
            assert_eq!(&text[..colon], label, "{label}: {text:?}");
            assert!(
                text[colon + 1..].starts_with("  "),
                "{label}: at least two spaces after the colon: {text:?}"
            );
        }
    }

    #[test]
    fn latency_format_switches_units_at_one_second() {
        assert_eq!(fmt_latency(12.1), "12.1ms");
        assert_eq!(fmt_latency(41.2), "41.2ms");
        assert_eq!(fmt_latency(145.0), "145ms");
        assert_eq!(fmt_latency(200.0), "200ms");
        assert_eq!(fmt_latency(1200.0), "1.2s");
    }

    #[test]
    fn throughput_chart_right_aligns_newest_sample() {
        // A single sample must plot at the rightmost plot column, not the
        // left.
        let lines = build_throughput_chart(&[100.0], 20, 5, 100.0, &[], 0.0, false);
        let rows: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
        // Find a row containing a bar; its rightmost █ should be near the
        // right edge.
        let bar_row = rows.iter().find(|r| r.contains('█')).expect("a bar row");
        let last_block = bar_row.rfind('█').unwrap();
        assert!(
            last_block > 10,
            "newest sample sits at the right edge: {bar_row}"
        );
    }

    #[test]
    fn throughput_chart_frozen_shows_the_complete_overlay() {
        // FIX 2: once the run completes (frozen), the chart carries a
        // `✓ COMPLETE` overlay and stays the final state.
        let lines = build_throughput_chart(&[100.0, 120.0, 90.0], 40, 8, 100.0, &[], 0.0, true);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains("✓ COMPLETE"), "frozen overlay: {text}");
        // The bars are still the final data (frozen, not cleared).
        assert!(text.contains('█'), "final bars rendered: {text}");
    }

    // ── engine-adaptive visibility ───────────────────────────────────────

    fn running_seq(engine: Engine) -> App {
        let app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::Running,
            queue: Engine::ALL.to_vec(),
            engine,
            progress: None,
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        });
        app
    }

    #[test]
    fn concurrency_and_capabilities_hidden_during_engine_a() {
        let app = running_seq(Engine::Speed);
        let seq = app.seq.load();
        assert!(!should_show_concurrency(&seq, &app));
        assert!(!should_show_capabilities(&seq, &app));
    }

    #[test]
    fn concurrency_shown_during_engine_b() {
        let app = running_seq(Engine::Concurrency);
        let seq = app.seq.load();
        assert!(should_show_concurrency(&seq, &app));
        assert!(!should_show_capabilities(&seq, &app));
    }

    #[test]
    fn capabilities_shown_for_each_c_and_d_engine() {
        for engine in [
            Engine::Niah,
            Engine::Reasoning,
            Engine::Structured,
            Engine::Hardware,
        ] {
            let app = running_seq(engine);
            let seq = app.seq.load();
            assert!(should_show_capabilities(&seq, &app), "{engine:?}");
        }
    }

    #[test]
    fn all_complete_shows_every_panel() {
        let app = App::new();
        // A full run: everything selected, everything completed, and the
        // results are on screen (the sweep slot + a capability slot).
        app.sweep.store(crate::engines::SweepResult {
            levels: vec![crate::engines::SweepLevel {
                concurrency: 1,
                aggregate_tps: 100.0,
                p50_tpot_ns: 0,
                p90_tpot_ns: 0,
                p99_tpot_ns: 0,
                ttft_p50_ns: 0,
                ttft_p90_ns: 0,
                total_tokens: 0,
                completed_streams: 1,
                failed_streams: 0,
                timed_out_streams: 0,
                aborted: false,
                wall_ns: 0,
                streams: Vec::new(),

                context: 0,
                per_stream_tps: 0.0,
                loop_excluded_streams: 0,
                loop_excluded_tokens: 0,
            }],
            matrix: None,
        });
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: Engine::ALL.to_vec(),
            engine: Engine::Hardware,
            progress: None,
            summary: "6 of 6 engines complete".to_string(),
            completed: vec![
                (
                    Engine::Concurrency,
                    "practical sweet spot 1 users".to_string(),
                ),
                (Engine::Hardware, "0.33 J/token".to_string()),
            ],
            engine_started_ms: 0,
        });
        let seq = app.seq.load();
        assert!(should_show_concurrency(&seq, &app));
        assert!(should_show_capabilities(&seq, &app));
    }

    #[test]
    fn all_complete_hides_panels_for_unselected_engines() {
        // FIX 1: a run of only A + B — after AllComplete the capability
        // panel must stay hidden (no C/D engines were selected), and the
        // concurrency panel shows only because B ran in the run.
        let app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: vec![Engine::Speed, Engine::Concurrency],
            engine: Engine::Concurrency,
            progress: None,
            summary: "2 of 2 engines complete".to_string(),
            completed: vec![(Engine::Concurrency, "sweep".to_string())],
            engine_started_ms: 0,
        });
        let seq = app.seq.load();
        assert!(should_show_concurrency(&seq, &app));
        assert!(!should_show_capabilities(&seq, &app));
    }

    #[test]
    fn selected_comes_from_the_run_queue_during_a_run() {
        // The run's queue is the authoritative selection while/after a run.
        let app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::Running,
            queue: vec![Engine::Speed, Engine::Niah],
            engine: Engine::Speed,
            progress: None,
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        });
        let sel = selected(&app);
        assert!(sel.speed && sel.niah);
        assert!(!sel.concurrency && !sel.hardware);
        // Before any run, the config form is the source.
        let app = App::new();
        let sel = selected(&app);
        assert_eq!(
            sel,
            EngineSelection {
                speed: true,
                concurrency: true,
                niah: true,
                reasoning: true,
                structured: true,
                hardware: false, // FIX 4: D is off by default
            }
        );
    }

    // ── capability scores ────────────────────────────────────────────────

    #[test]
    fn capability_scores_reflect_stored_results() {
        let app = App::new();
        app.reasoning_slot
            .store(crate::engines::capability::ReasoningResult {
                responses: vec![],
                ttfts: vec![],
                tg_speeds: vec![85.0],
                score: crate::engines::capability::ReasoningScore {
                    total: 13,
                    solved: 12,
                    by_category: [(5, 5), (4, 4), (3, 4)],
                },
            });
        app.structured_slot
            .store(crate::engines::capability::StructuredResult {
                free_tps: 100.0,
                constrained_tps: 98.0,
                penalty_pct: -0.3,
                free_ttft: 0.1,
                constrained_ttft: 0.101,
                cases: vec![
                    crate::engines::capability::evaluate_case(
                        "Simple",
                        r#"{"name": "Ada", "age": 36}"#,
                    ),
                    crate::engines::capability::evaluate_case("Medium", "not json"),
                    crate::engines::capability::evaluate_case("Complex", "also not json"),
                ],
                constrained_body: "{}".into(),
                free_body: "hi".into(),
            });
        let m = crate::metrics::state::MetricsSnapshot::default();
        let scores = build_capability_scores(&app, &m, &selected(&app));
        // Only the two engines that ran appear (no empty/zero bars).
        assert_eq!(scores.len(), 2, "one bar per *run* capability");
        let details: String = scores
            .iter()
            .map(|s| s.detail.clone())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(details.contains("12/13"), "reasoning detail: {details}");
        assert!(
            details.contains("compliant"),
            "structured detail: {details}"
        );
        // The non-fully-compliant structured score carries a ⚠ warning.
        let structured = scores
            .iter()
            .find(|s| s.label == "Structured Out")
            .expect("structured score");
        assert!(structured.warn.is_some(), "poor score warns");
    }

    #[test]
    fn capability_scores_hide_unrun_engines() {
        let app = App::new();
        let m = crate::metrics::state::MetricsSnapshot::default();
        // Nothing ran, no hw poller → no bars at all.
        assert!(build_capability_scores(&app, &m, &selected(&app)).is_empty());
    }

    #[test]
    fn capability_scores_hide_unselected_engines() {
        // FIX 1: a result in a slot is not enough — the engine must also
        // have been *selected*. (No seq → the config form is the source:
        // D is off by default, so its stale/absent data never shows.)
        let mut app = App::new();
        app.config.hardware = true; // D selected…
        app.hw = Some(Arc::new(std::sync::Mutex::new(crate::hw::HwPoller::new())));
        let m = crate::metrics::state::MetricsSnapshot::default();
        let sel = selected(&app);
        assert!(sel.hardware);
        let scores = build_capability_scores(&app, &m, &sel);
        assert!(
            scores.iter().any(|s| s.label == "Energy Efficiency"),
            "selected + poller live → the Energy line shows"
        );

        // … and when D is NOT selected, the same data yields no line.
        app.config.hardware = false;
        let sel = selected(&app);
        assert!(!sel.hardware);
        let scores = build_capability_scores(&app, &m, &sel);
        assert!(
            scores.iter().all(|s| s.label != "Energy Efficiency"),
            "unselected D never appears"
        );
    }

    #[test]
    fn capability_bar_fills_proportionally() {
        let s = CapScore {
            label: "Reasoning",
            pct: Some(50.0),
            detail: "50%".into(),
            detail_style: style::value(),
            color: palette::OK,
            info: "note",
            warn: None,
        };
        let lines = build_cap_lines(&s);
        assert_eq!(lines.len(), 2, "value line + ℹ line");
        let text: String = lines[0]
            .spans
            .iter()
            .map(|sp| sp.content.as_ref().to_string())
            .collect();
        // 20-wide bar at 50% → 10 filled, 10 empty.
        assert!(text.contains("[██████████░░░░░░░░░░]"), "{text}");
        // The second line is the dimmed explanation.
        let info: String = lines[1]
            .spans
            .iter()
            .map(|sp| sp.content.as_ref().to_string())
            .collect();
        assert!(info.contains('ℹ'), "{info}");
        assert!(info.contains("note"), "{info}");
    }

    #[test]
    fn capability_overall_composes_the_practical_summary() {
        // Strong reasoning, weak long-context, non-compliant JSON.
        let scores = vec![
            CapScore {
                label: "Reasoning",
                pct: Some(92.0),
                detail: "92% (12/13)".into(),
                detail_style: style::value(),
                color: palette::OK,
                info: "",
                warn: None,
            },
            CapScore {
                label: "Long Context",
                pct: Some(29.9),
                detail: "29.9% (23/77)".into(),
                detail_style: style::value(),
                color: palette::ERR,
                info: "",
                warn: Some("LOW"),
            },
            CapScore {
                label: "Structured Out",
                pct: Some(0.0),
                detail: "FAIL".into(),
                detail_style: style::value_err(),
                color: palette::ERR,
                info: "",
                warn: Some("non-compliant"),
            },
        ];
        let overall = capability_overall(&scores);
        assert!(overall.starts_with("OVERALL:"), "{overall}");
        assert!(overall.contains("strong reasoning"), "{overall}");
        assert!(overall.contains("weak long-context"), "{overall}");
        assert!(overall.contains("RAG"), "{overall}");
        assert!(overall.contains("tool-calling"), "{overall}");

        // No scores at all → the empty-state hint.
        assert!(capability_overall(&[]).contains("Run benchmarks to see capability scores"));
    }

    // ── energy line (Engine D) ───────────────────────────────────────────

    #[test]
    fn energy_line_tracks_summary_poller_and_telemetry() {
        use crate::metrics::state::MetricsSnapshot;
        // Nothing (no poller, no sequence summary) and D not selected →
        // omitted.
        let app = App::new();
        assert!(
            energy_line(&MetricsSnapshot::default(), &app, &selected(&app)).is_none(),
            "no Engine D request → no line"
        );
        // A completed D run in the sequence: its summary wins (the run's
        // queue selects D).
        let app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: vec![Engine::Hardware],
            engine: Engine::Hardware,
            progress: None,
            summary: String::new(),
            completed: vec![(Engine::Hardware, "0.338 J/token · peak 285 W".to_string())],
            engine_started_ms: 0,
        });
        assert_eq!(
            energy_line(&MetricsSnapshot::default(), &app, &selected(&app)).as_deref(),
            Some("0.338 J/token · peak 285 W")
        );
        // FIX 1: D *not* selected in the run → no line at all, even with
        // a live poller (the user's reported bug: "N/A (no GPU telemetry)"
        // showing for an unselected engine).
        let mut app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: vec![Engine::Speed],
            engine: Engine::Speed,
            progress: None,
            summary: String::new(),
            completed: vec![(Engine::Speed, "65 t/s".to_string())],
            engine_started_ms: 0,
        });
        app.hw = Some(Arc::new(std::sync::Mutex::new(crate::hw::HwPoller::new())));
        assert!(
            energy_line(&MetricsSnapshot::default(), &app, &selected(&app)).is_none(),
            "unselected D never shows the N/A line"
        );
        // A live poller without GPU telemetry (D selected): N/A.
        let mut app = App::new();
        app.config.hardware = true;
        app.hw = Some(Arc::new(std::sync::Mutex::new(crate::hw::HwPoller::new())));
        assert_eq!(
            energy_line(&MetricsSnapshot::default(), &app, &selected(&app)).as_deref(),
            Some("N/A (no GPU telemetry)")
        );
        // … and with telemetry: the live J/token value.
        let m = MetricsSnapshot {
            joules_per_token: 0.338,
            power_w: 285.0,
            ..MetricsSnapshot::default()
        };
        assert_eq!(
            energy_line(&m, &app, &selected(&app)).as_deref(),
            Some("0.338 J/token · 285 W")
        );
    }

    // ── sequence header parts (pure logic) ───────────────────────────────

    #[test]
    fn header_running_state_shows_engine_phase_and_progress() {
        let state = SeqState {
            phase: SeqPhase::Running,
            queue: vec![Engine::Speed],
            engine: Engine::Speed,
            progress: Some(EngineProgress::Speed {
                iteration: 2,
                total: 5,
                tokens: 248,
            }),
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        };
        let (_, _, text, _, ratio, _) = seq_header_parts(&state, 0);
        assert!(text.contains("ENGINE A: SPEED"), "{text}");
        assert!(text.contains("Running"), "{text}");
        assert!(text.contains("Iteration 2/5 · 248 tok"), "{text}");
        assert!((ratio - 0.2).abs() < 1e-9);
    }

    #[test]
    fn header_complete_state_carries_the_summary() {
        let state = SeqState {
            phase: SeqPhase::Complete,
            queue: vec![Engine::Speed, Engine::Concurrency],
            engine: Engine::Speed,
            progress: None,
            summary: "100.0 t/s decode".to_string(),
            completed: vec![(Engine::Speed, "100.0 t/s decode".to_string())],
            engine_started_ms: 0,
        };
        let (_, _, text, _, ratio, _) = seq_header_parts(&state, 0);
        assert!(text.contains("ENGINE A: SPEED"), "{text}");
        assert!(text.contains("Complete"), "{text}");
        assert!(text.contains("100.0 t/s decode"), "{text}");
        assert_eq!(ratio, 1.0);
    }

    #[test]
    fn header_all_complete_marks_the_whole_run() {
        let state = SeqState {
            phase: SeqPhase::AllComplete,
            queue: vec![Engine::Speed],
            engine: Engine::Speed,
            progress: None,
            summary: String::new(),
            completed: Vec::new(),
            engine_started_ms: 0,
        };
        let (_, _, text, _, ratio, _) = seq_header_parts(&state, 0);
        assert!(text.contains("ALL BENCHMARKS COMPLETE"), "{text}");
        assert_eq!(ratio, 1.0);
    }

    // ── full-render robustness ───────────────────────────────────────────

    fn render_live_text(app: &crate::ui::app::App, w: u16, h: u16) -> String {
        let backend = ratatui::backend::TestBackend::new(w, h);
        let mut terminal = ratatui::Terminal::new(backend).expect("TestBackend terminal");
        terminal
            .draw(|f| render(f.area(), app, f))
            .expect("render frame");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect()
    }

    #[test]
    fn live_view_survives_tiny_terminals() {
        let app = App::new();
        for (w, h) in [(40, 10), (20, 6), (80, 24)] {
            let _ = render_live_text(&app, w, h);
        }
    }

    #[test]
    fn live_view_default_shows_hero_key_metrics_and_log_only() {
        let app = App::new();
        let text = render_live_text(&app, 120, 40);
        assert!(text.contains("BENCHMARK SEQUENCE"), "{text}");
        assert!(text.contains("THROUGHPUT"), "{text}");
        assert!(text.contains("OVERALL METRICS"), "{text}");
        assert!(text.contains("EVENT LOG"), "{text}");
        // The removed hardware panels are gone.
        assert!(!text.contains("Target GPU VRAM"), "{text}");
        assert!(!text.contains("ACTIVE STREAMS MONITOR"), "{text}");
        assert!(!text.contains("ITL) DISTRIBUTION"), "{text}");
        assert!(!text.contains("BENCHMARK QUEUE"), "{text}");
        // No engine running → the adaptive panels stay hidden.
        assert!(!text.contains("CONCURRENCY CURVE"), "{text}");
        assert!(!text.contains("CAPABILITY SCORES"), "{text}");
    }

    // ── frozen (AllComplete) hero ────────────────────────────────────────

    #[test]
    fn live_view_hero_shows_final_frozen_state_when_complete() {
        // A frozen metrics state (the sequence reached AllComplete) → the
        // hero reads "final" + the ✓ COMPLETE badge, the live "now" is gone,
        // and the (frozen) rolling series stops animating.
        let app = App::new();
        app.metrics.update(MetricsSnapshot {
            aggregate_tps: 150.0,
            throughput_series: vec![120.0, 150.0, 180.0],
            ..Default::default()
        });
        app.metrics.freeze();
        let text = render_live_text(&app, 120, 40);
        assert!(text.contains("final"), "final label: {text}");
        assert!(text.contains("COMPLETE"), "✓ COMPLETE badge: {text}");
        assert!(text.contains("peak"), "peak still shown: {text}");
        assert!(
            !text.contains("now "),
            "live 'now' gone when frozen: {text}"
        );
    }

    #[test]
    fn live_view_hero_is_live_when_not_frozen() {
        // Not frozen → the hero keeps the live "now / peak / avg" line and
        // the real-time title (no COMPLETE badge) — behavior unchanged.
        let app = App::new();
        app.metrics.update(MetricsSnapshot {
            aggregate_tps: 150.0,
            throughput_series: vec![120.0, 150.0, 180.0],
            ..Default::default()
        });
        let text = render_live_text(&app, 120, 40);
        assert!(text.contains("now "), "live 'now' present: {text}");
        assert!(text.contains("avg"), "live 'avg' present: {text}");
        assert!(!text.contains("COMPLETE"), "no badge while live: {text}");
    }
}
