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
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Frame;

use crate::engines::sequence::{Engine, SeqPhase, SeqState};
use crate::metrics::state::MetricsSnapshot;
use crate::ui::app::App;
use crate::ui::theme::{palette, style};
use crate::ui::views::concurrency::build_curve_lines;

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

    let show_concurrency = should_show_concurrency(&seq, app);
    let show_capabilities = should_show_capabilities(&seq, app);

    // (panel id, constraint). The hero row is the largest slice.
    let mut plan: Vec<(u8, Constraint)> = Vec::new();
    plan.push((0, Constraint::Length(3))); // sequence header + progress bar
    plan.push((1, Constraint::Percentage(40))); // throughput hero | key metrics
    if show_concurrency {
        plan.push((2, Constraint::Percentage(28))); // concurrency curve
    }
    if show_capabilities {
        plan.push((3, Constraint::Percentage(22))); // capability scores
    }
    plan.push((4, Constraint::Length(5))); // compact event log

    let constraints: Vec<Constraint> = plan.iter().map(|(_, c)| *c).collect();
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    for (i, (id, _)) in plan.iter().enumerate() {
        match *id {
            0 => render_sequence_header(rects[i], app, f),
            1 => render_throughput_row(rects[i], m, f),
            2 => render_concurrency_curve(rects[i], app, f),
            3 => render_capability_scores(rects[i], app, m, f),
            4 => render_log(rects[i], app, f),
            _ => {}
        }
    }
}

// ── Engine-adaptive panel visibility ───────────────────────────────────────

/// The concurrency curve shows while Engine B runs, after the whole sequence
/// completes, or whenever a sweep result is already on screen (no sequence
/// in flight).
fn should_show_concurrency(seq: &Option<Arc<SeqState>>, app: &App) -> bool {
    match seq.as_deref() {
        Some(s) if s.phase == SeqPhase::AllComplete => true,
        Some(s) => s.engine == Engine::Concurrency,
        None => app
            .sweep
            .load()
            .as_ref()
            .as_ref()
            .is_some_and(|r| !r.levels.is_empty()),
    }
}

/// The capability scores show while any of the C/D engines runs, after the
/// whole sequence completes, or whenever a capability result is already on
/// screen (no sequence in flight).
fn should_show_capabilities(seq: &Option<Arc<SeqState>>, app: &App) -> bool {
    match seq.as_deref() {
        Some(s) if s.phase == SeqPhase::AllComplete => true,
        Some(s) => matches!(
            s.engine,
            Engine::Niah | Engine::Reasoning | Engine::Structured | Engine::Hardware
        ),
        None => {
            app.niah.load().as_ref().is_some()
                || app.reasoning_slot.load().as_ref().is_some()
                || app.structured_slot.load().as_ref().is_some()
        }
    }
}

// ── Throughput hero + key metrics (the top row) ────────────────────────────

/// The top row: the throughput hero chart (left, the largest panel) + the
/// key-metrics readout (right).
fn render_throughput_row(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(68), Constraint::Percentage(32)])
        .split(area);
    render_throughput_hero(cols[0], m, f);
    render_key_metrics(cols[1], m, f);
}

/// The hero: a large real-time aggregate tokens/sec block chart. One column
/// per sample (right-aligned, newest at the right edge), each a vertical run
/// of `█` graded green (high) → yellow (medium) → red (low) against the
/// window maximum, with a `now | peak` value line on top and y/x axis labels
/// so it reads at a glance.
fn render_throughput_hero(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::active_border())
        .title(" THROUGHPUT — tokens/sec (last 60s)  ● live ");
    if area.width < 8 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }

    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });

    // The `now | peak` value line.
    let series = &m.throughput_series;
    let current = series.last().copied().unwrap_or(m.aggregate_tps);
    let peak = series.iter().cloned().fold(0.0_f64, f64::max);
    let value_line = Line::from(vec![
        Span::styled("now ", style::label()),
        Span::styled(format!("{current:.1} t/s"), style::value()),
        Span::styled("  |  peak ", style::footer()),
        Span::styled(format!("{peak:.1} t/s"), style::value_warn()),
    ]);

    // The block chart fills the inner area below the value line; both are
    // rendered together inside the hero's bordered block (title + accent
    // border) so the panel reads as the view's centerpiece.
    let chart = build_throughput_chart(series, inner.width, inner.height.saturating_sub(1));
    let mut lines = vec![value_line];
    lines.extend(chart);

    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// Build the throughput block chart as lines of styled single-character
/// spans: a y-axis (max / max÷2 / 0), one `█` bar per sample column
/// (right-aligned), and an x-axis of time labels (0s…60s). Pure over the
/// series (unit-testable, no terminal).
fn build_throughput_chart(series: &[f64], w: u16, h: u16) -> Vec<Line<'static>> {
    const Y_AXIS_W: usize = 5;
    let w = w as usize;
    let h = h as usize;
    if w < Y_AXIS_W + 3 || h < 3 {
        return vec![Line::from("chart too small")];
    }
    let plot_w = w - Y_AXIS_W;
    let plot_h = h - 1; // bottom row reserved for the x-axis
    let max = series.iter().cloned().fold(0.0_f64, f64::max).max(1.0);

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

    // x-axis time labels along the bottom row, skipping overlaps.
    let x_labels = [
        ("0s", 0.0),
        ("15s", 0.25),
        ("30s", 0.5),
        ("45s", 0.75),
        ("60s", 1.0),
    ];
    let mut label_end: i64 = -1;
    for (label, frac) in x_labels {
        let center = Y_AXIS_W as i64 + (frac * (plot_w as f64 - 1.0)).round() as i64;
        let mut start = (center - (label.len() as i64) / 2).max(Y_AXIS_W as i64);
        // Keep the label inside the chart width (right-align at the edge so
        // the final `60s` is never clipped).
        start = start.min((w as i64) - (label.len() as i64));
        if start > label_end {
            for (i, ch) in label.chars().enumerate() {
                let col = (start + i as i64) as usize;
                if col < w {
                    grid[h - 1][col] = (ch, Some(palette::MUTED));
                }
            }
            label_end = start + label.len() as i64;
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

/// The key-metrics readout: the numbers a remote user scans first.
fn render_key_metrics(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title(" KEY METRICS ");
    if area.width < 10 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let ttft = mean_ttft(m);
    let lines =
        vec![
        kv("Throughput", format!("{:.1} t/s", m.aggregate_tps), style::value()),
        kv("TTFT", fmt_ms(ttft), style::value()),
        kv("ITL p50", ms(m.itl_p50_ns), style::value_ok()),
        kv("ITL p99", ms(m.itl_p99_ns), style::value_err()),
        kv("Tokens", grouped(m.completion_tokens), style::value()),
        kv("Streams", format!("{} active", m.active_streams), style::highlight()),
        Line::raw(""),
        Line::from(Span::styled(
            "ℹ Aggregate tokens/sec across all active streams. Higher = more parallel capacity.",
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

/// One `label  value` row for the key-metrics panel (the label column is
/// padded so values line up).
fn kv(label: &str, value: String, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<11}"), style::label()),
        Span::styled(value, value_style),
    ])
}

/// Mean time-to-first-token across the streams that report one (`None` when
/// none do — the panel shows `--`).
fn mean_ttft(m: &MetricsSnapshot) -> Option<f64> {
    let vals: Vec<f64> = m.streams.iter().filter_map(|s| s.ttft_s).collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

/// `250 ms` for a seconds value, `--` when absent.
fn fmt_ms(v: Option<f64>) -> String {
    v.map(|s| format!("{:.0} ms", s * 1000.0))
        .unwrap_or_else(|| "--".to_string())
}

/// `33.0 ms` for a nanosecond value, `--` at the `0` sentinel.
fn ms(ns: u64) -> String {
    if ns > 0 {
        format!("{:.1} ms", ns as f64 / 1_000_000.0)
    } else {
        "--".to_string()
    }
}

// ── Concurrency curve (Engine B) ───────────────────────────────────────────

/// The prominent concurrency panel: aggregate t/s vs parallel users, reusing
/// View 2's block-based curve (the sweet spot `●` and the knee `▲`). While
/// Engine B is mid-sweep (no result published yet) it shows an in-progress
/// note; with no sweep at all it shows the run-a-sweep hint.
fn render_concurrency_curve(area: Rect, app: &App, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title(" CONCURRENCY CURVE — t/s vs parallel users  ● sweet spot  ▲ knee ");
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
            build_curve_lines(&r.levels, inner.width, inner.height, &env)
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

/// One capability bar: a label, a horizontal `[████░░]` bar (0–100%), a value
/// detail, and a one-line dimmed `ℹ` note.
struct CapScore {
    label: &'static str,
    /// `0.0..=100.0` bar fill, or `None` for a non-percentage metric (the bar
    /// stays empty and the detail carries the value).
    pct: Option<f64>,
    detail: String,
    detail_style: Style,
    color: Color,
    info: &'static str,
}

/// The capability scores panel: horizontal bars for Reasoning (C2), NIAH
/// (C1), Structured (C3), and Energy (D). Each reads its lock-free result
/// slot; a metric that hasn't run shows an empty bar with a `—` detail.
fn render_capability_scores(area: Rect, app: &App, m: &MetricsSnapshot, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title(" CAPABILITY SCORES ");
    if area.width < 10 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let scores = build_capability_scores(app, m);
    let lines: Vec<Line> = scores.iter().map(|s| build_cap_bar(s)).collect();
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// Gather the four capability scores from their lock-free result slots (pure
/// over `&App` — unit-testable without a terminal).
fn build_capability_scores(app: &App, m: &MetricsSnapshot) -> Vec<CapScore> {
    let mut v: Vec<CapScore> = Vec::with_capacity(4);

    // Reasoning (C2) — N/M solved as a percentage.
    match app.reasoning_slot.load().as_ref() {
        Some(r) if r.score.total > 0 => {
            let pct = r.score.solved as f64 / r.score.total as f64 * 100.0;
            v.push(CapScore {
                label: "Reasoning",
                pct: Some(pct),
                detail: format!(
                    "{}%  ({}/{})",
                    pct.round() as i64,
                    r.score.solved,
                    r.score.total
                ),
                detail_style: style::value(),
                color: palette::OK,
                info: "Logic, math, code problem-solving",
            });
        }
        _ => v.push(CapScore {
            label: "Reasoning",
            pct: None,
            detail: "—".to_string(),
            detail_style: style::footer(),
            color: palette::MUTED,
            info: "Logic, math, code problem-solving",
        }),
    }

    // NIAH (C1) — retrieved/total cells as a percentage.
    match app.niah.load().as_ref() {
        Some(r) => {
            let (retrieved, total) = r.accuracy();
            let pct = if total > 0 {
                retrieved as f64 / total as f64 * 100.0
            } else {
                0.0
            };
            v.push(CapScore {
                label: "NIAH",
                pct: Some(pct),
                detail: format!("{}%  ({}/{})", pct.round() as i64, retrieved, total),
                detail_style: style::value(),
                color: palette::OK,
                info: "Long-context retrieval (RAG readiness)",
            });
        }
        None => v.push(CapScore {
            label: "NIAH",
            pct: None,
            detail: "—".to_string(),
            detail_style: style::footer(),
            color: palette::MUTED,
            info: "Long-context retrieval (RAG readiness)",
        }),
    }

    // Structured (C3) — compliant (100%) or not (0%).
    match app.structured_slot.load().as_ref() {
        Some(r) => {
            let ok = r.compliant;
            v.push(CapScore {
                label: "Structured",
                pct: Some(if ok { 100.0 } else { 0.0 }),
                detail: if ok {
                    "COMPLIANT".to_string()
                } else {
                    "NON-COMPLIANT".to_string()
                },
                detail_style: if ok {
                    style::value_ok()
                } else {
                    style::value_err()
                },
                color: if ok { palette::OK } else { palette::ERR },
                info: "JSON/API instruction following",
            });
        }
        None => v.push(CapScore {
            label: "Structured",
            pct: None,
            detail: "—".to_string(),
            detail_style: style::footer(),
            color: palette::MUTED,
            info: "JSON/API instruction following",
        }),
    }

    // Energy (D) — J/token (not a percentage); `N/A` without GPU telemetry.
    match energy_line(m, app) {
        Some(line) => v.push(CapScore {
            label: "Energy",
            pct: None,
            detail: line,
            detail_style: style::value(),
            color: palette::WARN,
            info: "Requires local GPU access",
        }),
        None => v.push(CapScore {
            label: "Energy",
            pct: None,
            detail: "N/A".to_string(),
            detail_style: style::footer(),
            color: palette::MUTED,
            info: "Requires local GPU access",
        }),
    }

    v
}

/// Render one capability score as a single line:
/// `Label   [████████████░░]  detail  ℹ info`.
fn build_cap_bar(s: &CapScore) -> Line<'static> {
    const BAR_W: usize = 14;
    let filled = s
        .pct
        .map(|p| (p.clamp(0.0, 100.0) / 100.0 * BAR_W as f64).round() as usize)
        .unwrap_or(0);
    let mut bar = String::with_capacity(BAR_W);
    for i in 0..BAR_W {
        bar.push(if i < filled { '█' } else { '░' });
    }
    Line::from(vec![
        Span::styled(format!("{:<11}", s.label), style::label()),
        Span::styled(format!("[{bar}] "), Style::default().fg(s.color)),
        Span::styled(s.detail.clone(), s.detail_style),
        Span::styled(format!("  ℹ {}", s.info), style::info()),
    ])
}

// ── Engine D energy line (shared by capability scores) ─────────────────────

/// The Engine D data line, when one should be shown: the sequence's
/// sampling summary wins; otherwise the live hardware-poller telemetry
/// (`N/A` without GPU telemetry). `None` when Engine D was never
/// requested (no poller, no summary).
fn energy_line(m: &MetricsSnapshot, app: &App) -> Option<String> {
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
    None
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
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(border_style)
                    .title(" BENCHMARK SEQUENCE "),
            )
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
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title(" EVENT LOG "),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

// ── Shared formatting helpers ──────────────────────────────────────────────

/// Format an integer with thousands separators (`1332 → "1,332"`) for the
/// token readout.
fn grouped(v: u64) -> String {
    let s = v.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

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
        let lines = build_throughput_chart(&[1.0, 2.0], 3, 2);
        assert_eq!(lines[0].to_string(), "chart too small");
    }

    #[test]
    fn throughput_chart_plots_bars_and_axes() {
        let series = vec![100.0, 200.0, 150.0, 300.0, 50.0, 250.0];
        let lines = build_throughput_chart(&series, 40, 10);
        let text: String = lines
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(text.contains('█'), "bars rendered: {text}");
        // x-axis time labels.
        assert!(text.contains("0s"), "x-axis start: {text}");
        assert!(text.contains("60s"), "x-axis end: {text}");
        // y-axis maximum label.
        assert!(text.contains("300"), "y-axis max: {text}");
    }

    #[test]
    fn throughput_chart_right_aligns_newest_sample() {
        // A single sample must plot at the rightmost plot column, not the
        // left.
        let lines = build_throughput_chart(&[100.0], 20, 5);
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
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: Engine::ALL.to_vec(),
            engine: Engine::Hardware,
            progress: None,
            summary: "6 of 6 engines complete".to_string(),
            completed: Vec::new(),
        });
        let seq = app.seq.load();
        assert!(should_show_concurrency(&seq, &app));
        assert!(should_show_capabilities(&seq, &app));
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
                compliant: false,
                constrained_body: "{}".into(),
                free_body: "hi".into(),
            });
        let m = crate::metrics::state::MetricsSnapshot::default();
        let scores = build_capability_scores(&app, &m);
        assert_eq!(scores.len(), 4, "one bar per capability");
        let details: String = scores
            .iter()
            .map(|s| s.detail.clone())
            .collect::<Vec<_>>()
            .join(" | ");
        assert!(details.contains("12/13"), "reasoning detail: {details}");
        assert!(
            details.contains("NON-COMPLIANT"),
            "structured detail: {details}"
        );
        // NIAH + Energy not run → `—` / `N/A`.
        assert!(details.contains('—'), "unrun metric placeholder: {details}");
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
        };
        let line = build_cap_bar(&s);
        let text: String = line.spans.iter().map(|sp| sp.content.as_ref()).collect();
        // 14-wide bar at 50% → 7 filled, 7 empty.
        assert!(text.contains("[███████░░░░░░░]"), "{text}");
    }

    // ── energy line (Engine D) ───────────────────────────────────────────

    #[test]
    fn energy_line_tracks_summary_poller_and_telemetry() {
        use crate::metrics::state::MetricsSnapshot;
        // Nothing (no poller, no sequence summary) → omitted.
        let app = App::new();
        assert!(
            energy_line(&MetricsSnapshot::default(), &app).is_none(),
            "no Engine D request → no line"
        );
        // A completed D run in the sequence: its summary wins.
        let app = App::new();
        app.seq.store(SeqState {
            phase: SeqPhase::AllComplete,
            queue: vec![Engine::Hardware],
            engine: Engine::Hardware,
            progress: None,
            summary: String::new(),
            completed: vec![(Engine::Hardware, "0.338 J/token · peak 285 W".to_string())],
        });
        assert_eq!(
            energy_line(&MetricsSnapshot::default(), &app).as_deref(),
            Some("0.338 J/token · peak 285 W")
        );
        // A live poller without GPU telemetry: N/A.
        let mut app = App::new();
        app.hw = Some(Arc::new(std::sync::Mutex::new(crate::hw::HwPoller::new())));
        assert_eq!(
            energy_line(&MetricsSnapshot::default(), &app).as_deref(),
            Some("N/A (no GPU telemetry)")
        );
        // … and with telemetry: the live J/token value.
        let m = MetricsSnapshot {
            joules_per_token: 0.338,
            power_w: 285.0,
            ..MetricsSnapshot::default()
        };
        assert_eq!(
            energy_line(&m, &app).as_deref(),
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
        assert!(text.contains("KEY METRICS"), "{text}");
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
}
