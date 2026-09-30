//! View 1 — Live Monitor & Telemetry (blueprint §6) + the **Benchmark
//! Sequence** panel.
//!
//! The top of the view is the sequence header: which engine is running,
//! its phase, a live progress bar, and the current engine's progress
//! line (e.g. `ENGINE A: SPEED — Running — Iteration 2/5`). The queue
//! panel lists every engine in the run with `✓` (complete, with its
//! summary), `▶` (running) or `○` (queued) markers. Below, the classic
//! telemetry panels show **real** data from the *currently running*
//! engine only: the engines publish live [`MetricsSnapshot`]s to the
//! shared `ArcSwap` seam (Chunk 6) and the log stream carries the
//! executor's real events. The **ENGINE RESULTS** panel lists each
//! completed engine (A, C2, C3, D) with its headline number and a
//! dimmed `ℹ` note explaining what that number means.
//!
//! Every panel reads the shared state lock-free (the
//! [`crate::engines::sequence::SeqStateSlot`] for the sequence, the
//! `MetricsState` for the telemetry) — the render loop never blocks and
//! never touches the timing path (measurement-isolation invariant,
//! blueprint §4).

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table, Wrap,
};
use ratatui::Frame;

use crate::engines::sequence::{summarize_speed, Engine, SeqPhase, SeqState};
use crate::metrics::state::{MetricsSnapshot, StreamMetric, StreamStatus};
use crate::ui::app::{fmt, App};
use crate::ui::theme::{palette, style};

/// Render the Live Monitor view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    // Lock-free read of the latest published snapshot (Chunk 6). Bind the
    // owned `Arc` first so the `&MetricsSnapshot` borrow outlives the call.
    let snap = app.metrics.load();
    let m = snap.as_ref();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(3),      // sequence header + progress bar
            Constraint::Percentage(26), // gauges | ITL | benchmark queue
            Constraint::Percentage(18), // active streams monitor
            Constraint::Percentage(13), // throughput sparkline | token counter
            Constraint::Percentage(26), // engine results + ℹ info
            Constraint::Percentage(14), // log & event stream
        ])
        .split(area);

    render_sequence_header(rows[0], app, f);
    render_top(rows[1], m, app, f);
    render_streams(rows[2], m, f);
    render_bottom(rows[3], m, f);
    render_results(rows[4], m, app, f);
    render_log(rows[5], app, f);
}

/// Engine results panel: one data line per completed engine (A, C2, C3, D)
/// from the lock-free result slots, each followed by a dimmed `ℹ` line
/// explaining what the numbers mean. Engines that have not run yet are
/// omitted — the panel fills in as the benchmark sequence progresses.
fn render_results(area: Rect, m: &MetricsSnapshot, app: &App, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("ENGINE RESULTS");
    if area.width < 12 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }

    let mut lines: Vec<Line> = Vec::new();

    // Engine A — single-stream speed.
    if let Some(results) = app.speed_slot.load().as_ref().as_ref() {
        lines.push(result_line("A: Speed", &summarize_speed(results)));
        lines.push(info_line(Engine::Speed));
    }
    // Engine C2 — reasoning accuracy.
    if let Some(r) = app.reasoning_slot.load().as_ref().as_ref() {
        lines.push(result_line(
            "C2: Reasoning",
            &format!("{} · {:.1} t/s", r.score.label(), r.avg_tg_speed()),
        ));
        lines.push(info_line(Engine::Reasoning));
    }
    // Engine C3 — structured-output compliance.
    if let Some(s) = app.structured_slot.load().as_ref().as_ref() {
        lines.push(result_line(
            "C3: Structured",
            &format!(
                "{:+.1}% penalty · compliant: {}",
                s.penalty_pct,
                if s.compliant { "YES" } else { "NO" }
            ),
        ));
        lines.push(info_line(Engine::Structured));
    }
    // Engine D — energy (only when requested: a sequence summary or a
    // live hardware poller exists).
    if let Some(energy) = energy_line(m, app) {
        lines.push(result_line("D: Energy", &energy));
        lines.push(info_line(Engine::Hardware));
    }

    if lines.is_empty() {
        lines.push(Line::from(Span::styled(
            "No engines completed yet — results appear here as each engine finishes.",
            style::info(),
        )));
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The Engine D data line, when one should be shown: the sequence's
/// sampling summary wins; otherwise the live hardware-poller telemetry
/// (`N/A` without GPU telemetry). `None` when Engine D was never
/// requested (no poller, no summary) — the line is omitted entirely.
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

/// One results data line: the engine label + its value (the label column
/// is padded so a 14-char label like `C3: Structured` keeps a gap).
fn result_line(label: &str, value: &str) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<14}  "), style::label()),
        Span::styled(value.to_string(), style::value()),
    ])
}

/// The dimmed `ℹ` note explaining an engine's numbers (the description
/// joined to one line; the panel wraps it to fit).
fn info_line(engine: Engine) -> Line<'static> {
    let desc = engine
        .description()
        .lines()
        .map(str::trim)
        .collect::<Vec<_>>()
        .join(" ");
    Line::from(Span::styled(format!("ℹ {desc}"), style::info()))
}

// ── Benchmark Sequence: header, progress bar, queue ──────────────────────

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
        Span::styled(marker.to_string(), marker_style),
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

/// The queue panel: one line per engine in the run — `✓` completed
/// (with its summary), `▶` the running one (with its live progress),
/// `○` the rest (queued). Before a run starts, the full default queue is
/// shown, dimmed.
fn render_queue(area: Rect, app: &App, f: &mut Frame) {
    if area.width < 12 || area.height < 3 {
        return;
    }
    let state = app.seq.load();
    let queue = state
        .as_ref()
        .map(|s| s.queue.clone())
        .unwrap_or_else(|| Engine::ALL.to_vec());
    let lines: Vec<Line> = queue
        .iter()
        .map(|engine| queue_line(state.as_deref(), engine))
        .collect();
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title(" BENCHMARK QUEUE "),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// One queue line for `engine`, styled by its state in `state`.
fn queue_line(state: Option<&SeqState>, engine: &Engine) -> Line<'static> {
    match state {
        None => Line::from(vec![
            Span::styled("○ ", style::footer()),
            Span::styled(engine.label().to_string(), style::footer()),
            Span::styled("  (not started)", style::footer()),
        ]),
        Some(state) => {
            if let Some((_, summary)) = state.completed.iter().find(|(e, _)| e == engine) {
                // Completed: green check + the engine's summary.
                Line::from(vec![
                    Span::styled("✓ ", style::value_ok()),
                    Span::styled(engine.label().to_string(), style::value_ok()),
                    Span::styled(format!("  {summary}"), style::footer()),
                ])
            } else if state.engine == *engine && state.phase != SeqPhase::Idle {
                // The current engine (running, or briefly on its summary
                // hold before the queue advances).
                let progress = state
                    .progress
                    .as_ref()
                    .map(|p| p.label())
                    .unwrap_or_else(|| state.summary.clone());
                Line::from(vec![
                    Span::styled(
                        "▶ ",
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(ratatui::style::Modifier::BOLD),
                    ),
                    Span::styled(
                        engine.label().to_string(),
                        Style::default()
                            .fg(palette::ACCENT)
                            .add_modifier(ratatui::style::Modifier::BOLD),
                    ),
                    Span::styled(format!("  {progress}"), style::value_warn()),
                ])
            } else {
                // Queued (or the Idle state's placeholder queue).
                Line::from(vec![
                    Span::styled("○ ", style::footer()),
                    Span::styled(engine.label().to_string(), style::footer()),
                    Span::styled("  (queued)", style::footer()),
                ])
            }
        }
    }
}

// ── Telemetry panels (real data from the current engine) ─────────────────

/// Top row: telemetry gauges (left) + ITL distribution (middle) + the
/// benchmark queue (right).
fn render_top(area: Rect, m: &MetricsSnapshot, app: &App, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(32), // gauges
            Constraint::Percentage(34), // ITL (the 38-char title needs this)
            Constraint::Fill(1),        // queue (the rest)
        ])
        .split(area);
    render_gauges(cols[0], m, f);
    render_itl(cols[1], m, f);
    render_queue(cols[2], app, f);
}

fn kv(label: &str, value: String, value_style: Style) -> Line<'static> {
    // The gauges column is ~38 wide (the 3-column top row), so the label
    // carries its own two-space gap — no fixed padding column.
    Line::from(vec![
        Span::styled(format!("{label}  "), style::label()),
        Span::styled(value, value_style),
    ])
}

/// Top-left: aggregate throughput, active streams, GPU clock, power
/// (with J/token), and the VRAM capacity bar (blueprint §6 View 1
/// "Key Metrics"). All values are from the *currently running* engine's
/// live snapshots (plus the continuous hardware poller's telemetry).
fn render_gauges(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    // Guard the 0-total case (no GPU telemetry) so the ratio is a real
    // number: graceful degradation to a 0% bar, never a NaN ratio
    // (blueprint §5D: report N/A, never panic).
    let vram_ratio = if m.vram_total_gb > 0.0 {
        (m.vram_used_gb / m.vram_total_gb).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let vram_style = if vram_ratio < 0.75 {
        style::value_ok()
    } else if vram_ratio < 0.90 {
        style::value_warn()
    } else {
        style::value_err()
    };
    let lines = vec![
        kv(
            "Total Aggregate",
            format!("{:.1} t/s", m.aggregate_tps),
            style::value(),
        ),
        kv(
            "Active Streams",
            format!("{:2} / {:2}", m.active_streams, m.total_streams),
            style::value(),
        ),
        kv("GPU Clock", m.gpu_clock_label(), style::value()),
        kv(
            "Current Power",
            format!("{:.0} W ({:.3} J/token)", m.power_w, m.joules_per_token),
            style::value(),
        ),
    ];

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(area);

    f.render_widget(
        Paragraph::new(Text::from(lines)).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("TELEMETRY GAUGES"),
        ),
        chunks[0],
    );
    f.render_widget(
        Gauge::default()
            .ratio(vram_ratio)
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title("Target GPU VRAM"),
            )
            .gauge_style(Style::default().fg(vram_style.fg.unwrap_or(palette::OK)))
            .label(format!(
                "{:.1} / {:.1} GB ({:.0}%)",
                m.vram_used_gb,
                m.vram_total_gb,
                vram_ratio * 100.0
            )),
        chunks[1],
    );
}

/// Top-middle: ITL percentile gauge bars (p50/p90/p99) + histogram.
///
/// The three one-row [`Gauge`] bars are scaled against the widest
/// percentile (p99.9) and carry the green (p50) → yellow (p90) → red
/// (p99) gradient; each bar's centered label shows the value in ms.
fn render_itl(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(3), Constraint::Min(0)])
        .split(area);

    let scale = m
        .itl_p999_ns
        .max(m.itl_p99_ns)
        .max(m.itl_p90_ns)
        .max(m.itl_p50_ns)
        .max(1) as f64;
    let gauges = [
        (m.itl_p50_ns, "p50", palette::OK),
        (m.itl_p90_ns, "p90", palette::WARN),
        (m.itl_p99_ns, "p99", palette::ERR),
    ];
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1); 3])
        .split(chunks[0]);
    for (rect, (ns, name, color)) in rows.iter().zip(gauges) {
        f.render_widget(
            Gauge::default()
                .ratio((ns as f64 / scale).clamp(0.0, 1.0))
                .gauge_style(Style::default().fg(color))
                .label(format!("{name}  {:.1} ms", ns as f64 / 1_000_000.0)),
            *rect,
        );
    }

    let points: Vec<(f64, f64)> = m
        .itl_bins
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();
    let data = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Scatter)
        .style(palette::WARN)
        .data(&points);
    f.render_widget(
        Chart::new(vec![data])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title("INTER-TOKEN LATENCY (ITL) DISTRIBUTION"),
            )
            .x_axis(
                Axis::default()
                    .style(style::footer())
                    .bounds([0.0, 24.0])
                    .labels(vec![
                        Line::from("0ms"),
                        Line::from("25ms"),
                        Line::from("50ms"),
                        Line::from("100ms"),
                        Line::from("200ms"),
                    ]),
            )
            .y_axis(Axis::default().style(style::footer()).bounds([0.0, 1.05])),
        chunks[1],
    );
}

/// Mid-panel: per-stream matrix (ID, type, state, PP/TG, TTFT, gen speed,
/// MTP rate, progress) — the *real* streams of the currently running
/// engine (the engines publish their live stream rows to the snapshot).
fn render_streams(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("ID"),
        Cell::from("TYPE"),
        Cell::from("STATE"),
        Cell::from("TOKENS (PP/TG)"),
        Cell::from("TTFT"),
        Cell::from("GEN SPEED"),
        Cell::from("MTP RATE"),
        Cell::from("PROGRESS"),
    ])
    .style(style::muted_title())];
    for s in &m.streams {
        rows.push(stream_row(s));
    }
    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Percentage(8),
                Constraint::Percentage(13),
                Constraint::Percentage(12),
                Constraint::Percentage(17),
                Constraint::Percentage(10),
                Constraint::Percentage(12),
                Constraint::Percentage(10),
                Constraint::Percentage(18),
            ],
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title(format!(
                    "ACTIVE STREAMS MONITOR (Top {} of {} Active)",
                    m.streams.len(),
                    m.active_streams
                )),
        ),
        area,
    );
}

fn stream_row(s: &StreamMetric) -> Row<'_> {
    let state_style = match s.state {
        StreamStatus::Streaming => style::value_ok(),
        StreamStatus::Waiting => style::value_warn(),
        StreamStatus::Done => style::highlight(),
        StreamStatus::Error => style::value_err(),
    };
    Row::new(vec![
        Cell::from(format!("#{:02}", s.id)),
        Cell::from(s.kind.as_str()),
        Cell::from(s.state.label()).style(state_style),
        Cell::from(format!(
            "{}/{}",
            fmt::tokens(s.pp_tokens),
            fmt::tokens(s.tg_tokens)
        )),
        Cell::from(fmt::sec(s.ttft_s)),
        Cell::from(fmt::tps(s.gen_tps)),
        Cell::from(fmt::mult(s.mtp)).style(style::highlight()),
        Cell::from(format!("[{}]", fmt::progress_bar(s.progress, 14))),
    ])
}

/// Bottom chart row: rolling throughput sparkline (left) + token
/// counter (right).
fn render_bottom(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(70), Constraint::Percentage(30)])
        .split(area);
    render_throughput(cols[0], m, f);
    render_tokens(cols[1], m, f);
}

/// Bottom-left: rolling aggregate tokens/sec over the last 60 seconds as
/// a unicode-block sparkline (`▁▂▃▄▅▆▇█`, one sample per column,
/// right-aligned so the newest sample sits at the right edge), color
/// graded green (high) → yellow (medium) → red (low) against the
/// window maximum, with the current value labeled.
fn render_throughput(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("REAL-TIME SYSTEM PERFORMANCE — TOKENS/SEC (LAST 60s)");
    if area.width < 6 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }

    let series = &m.throughput_series;
    let max = series.iter().cloned().fold(0.0_f64, f64::max).max(1.0);
    let current = series.last().copied().unwrap_or(m.aggregate_tps);
    let peak = series.iter().cloned().fold(0.0_f64, f64::max);

    // One sample per column; left-pad so the window stays right-aligned
    // while it fills.
    let width = (area.width - 2) as usize;
    let n = width.min(series.len());
    let mut spans: Vec<Span> = Vec::with_capacity(width);
    for _ in 0..width - n {
        spans.push(Span::raw(" "));
    }
    for &v in &series[series.len() - n..] {
        let ratio = v / max;
        spans.push(Span::styled(
            block_char(ratio).to_string(),
            Style::default().fg(throughput_color(ratio)),
        ));
    }

    let lines = vec![
        Line::from(vec![
            Span::styled("now ", style::label()),
            Span::styled(format!("{current:.1} t/s"), style::value()),
            Span::styled("  |  peak ", style::footer()),
            Span::styled(format!("{peak:.1} t/s"), style::value_warn()),
        ]),
        Line::from(spans),
    ];
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// Bottom-right: total tokens generated — large number display.
///
/// The counter reads `1,332` at a glance (thousands-separated); the
/// reasoning and prompt splits sit beneath it.
fn render_tokens(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("TOKENS GENERATED");
    if area.width < 6 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let lines = vec![
        Line::from(Span::styled(grouped(m.completion_tokens), style::value())),
        Line::from(Span::styled(
            format!("reasoning {}", grouped(m.reasoning_tokens)),
            style::highlight(),
        )),
        Line::from(Span::styled(
            format!("prompt {}", grouped(m.prompt_tokens)),
            style::footer(),
        )),
    ];
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// Format an integer with thousands separators (`1332 → "1,332"`) for the
/// large-number token display.
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

/// Bottom: scrolling log / event stream — the executor's *real* events
/// (engine starts, completions, summaries), drained from the mpsc pipe
/// on the tick path (never pre-generated).
fn render_log(area: Rect, app: &App, f: &mut Frame) {
    f.render_widget(
        Paragraph::new(Text::from(app.log.clone()))
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title("LOG & EVENT STREAM"),
            )
            .wrap(Wrap { trim: true }),
        area,
    );
}

/// The 8-step unicode block ramp used by the sparklines.
const BLOCK_RAMP: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Map a `0.0..=1.0` ratio to the nearest block-ramp character
/// (`0.0 → ▁`, `1.0 → █`); out-of-range ratios clamp to the ramp ends.
pub(crate) fn block_char(ratio: f64) -> char {
    let idx = (ratio.clamp(0.0, 1.0) * (BLOCK_RAMP.len() - 1) as f64).round() as usize;
    BLOCK_RAMP[idx]
}

/// Color gradient for the throughput sparkline: green (high) → yellow
/// (medium) → red (low), relative to the rolling-window maximum.
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
    use std::sync::{Arc, Mutex};

    #[test]
    fn block_char_maps_ratio_onto_the_ramp() {
        assert_eq!(block_char(0.0), '▁');
        assert_eq!(block_char(0.5), '▅');
        assert_eq!(block_char(1.0), '█');
        // Out-of-range ratios clamp to the ramp ends.
        assert_eq!(block_char(-3.0), '▁');
        assert_eq!(block_char(4.0), '█');
    }

    #[test]
    fn throughput_color_is_a_green_yellow_red_gradient() {
        assert_eq!(throughput_color(1.0), palette::OK);
        assert_eq!(throughput_color(0.75), palette::OK);
        assert_eq!(throughput_color(0.5), palette::WARN);
        assert_eq!(throughput_color(0.4), palette::WARN);
        assert_eq!(throughput_color(0.2), palette::ERR);
        assert_eq!(throughput_color(0.0), palette::ERR);
    }

    // ── sequence header / queue parts (pure logic) ──────────────────────

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

    #[test]
    fn queue_line_marks_completed_running_and_queued() {
        let state = SeqState {
            phase: SeqPhase::Running,
            queue: vec![Engine::Speed, Engine::Concurrency, Engine::Niah],
            engine: Engine::Concurrency,
            progress: Some(EngineProgress::Concurrency {
                level: 8,
                step: 2,
                total_steps: 7,
                active: 8,
            }),
            summary: String::new(),
            completed: vec![(Engine::Speed, "100.0 t/s decode".to_string())],
        };
        let done = queue_line(Some(&state), &Engine::Speed);
        let running = queue_line(Some(&state), &Engine::Concurrency);
        let queued = queue_line(Some(&state), &Engine::Niah);

        let done_text: String = done.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(done_text.starts_with("✓ "), "{done_text}");
        assert!(done_text.contains("A: Speed"), "{done_text}");
        assert!(done_text.contains("100.0 t/s decode"), "{done_text}");

        let running_text: String = running.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(running_text.starts_with("▶ "), "{running_text}");
        assert!(running_text.contains("B: Concurrency"), "{running_text}");
        assert!(
            running_text.contains("Concurrency level 8 (step 2/7)"),
            "{running_text}"
        );

        let queued_text: String = queued.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(queued_text.starts_with("○ "), "{queued_text}");
        assert!(queued_text.contains("C1: NIAH"), "{queued_text}");
        assert!(queued_text.contains("(queued)"), "{queued_text}");
    }

    #[test]
    fn queue_line_before_any_run_is_dimmed_not_started() {
        let line = queue_line(None, &Engine::Speed);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.starts_with("○ "), "{text}");
        assert!(text.contains("(not started)"), "{text}");
    }

    // ── engine results panel (data + dimmed ℹ info) ─────────────────────

    fn speed_result(tg: f64, ttft: f64, tokens: u64) -> crate::engines::speed::SpeedResult {
        crate::engines::speed::SpeedResult {
            ttft,
            prompt_tokens: 100,
            completion_tokens: tokens,
            pp_speed: 1000.0,
            tg_speed: tg,
            mtp_efficiency: 1.0,
            stream_time: 1.0,
            total_chunks: 10,
            content_chunks: 10,
            reasoning_chunks: 0,
            other_chunks: 0,
            estimated: false,
            model: "m".into(),
            mode: "short".into(),
            error: None,
        }
    }

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
    fn results_panel_lists_completed_engines_with_info_notes() {
        let app = crate::ui::app::App::new();
        app.speed_slot.store(vec![speed_result(60.6, 0.25, 256)]);
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

        let text = render_live_text(&app, 120, 40);
        assert!(text.contains("ENGINE RESULTS"), "results panel: {text}");
        assert!(text.contains("A: Speed"), "{text}");
        assert!(text.contains("60.6 t/s decode"), "{text}");
        assert!(text.contains("C2: Reasoning"), "{text}");
        assert!(text.contains("12/13 solved"), "{text}");
        assert!(text.contains("C3: Structured"), "{text}");
        assert!(text.contains("compliant: NO"), "{text}");
        // The dimmed ℹ notes explain each engine's numbers.
        assert!(text.contains('ℹ'), "{text}");
        assert!(text.contains("Single-stream throughput"), "{text}");
        assert!(text.contains("Logical reasoning accuracy"), "{text}");
        assert!(text.contains("JSON compliance"), "{text}");
        // Engine D was never requested (no poller): the results panel
        // carries no energy line ("D: Energy" also appears in the queue
        // panel's default list, so assert on the data value instead).
        assert!(!text.contains("N/A (no GPU telemetry)"), "{text}");
    }

    #[test]
    fn results_panel_placeholder_before_any_engine_completes() {
        let app = crate::ui::app::App::new();
        let text = render_live_text(&app, 120, 40);
        assert!(text.contains("ENGINE RESULTS"), "{text}");
        assert!(text.contains("No engines completed yet"), "{text}");
    }

    #[test]
    fn energy_line_tracks_summary_poller_and_telemetry() {
        use crate::metrics::state::MetricsSnapshot;
        // Nothing (no poller, no sequence summary) → omitted.
        let app = crate::ui::app::App::new();
        assert!(
            energy_line(&MetricsSnapshot::default(), &app).is_none(),
            "no Engine D request → no line"
        );
        // A completed D run in the sequence: its summary wins.
        let app = crate::ui::app::App::new();
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
        let mut app = crate::ui::app::App::new();
        app.hw = Some(Arc::new(Mutex::new(crate::hw::HwPoller::new())));
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

    #[test]
    fn results_panel_survives_tiny_terminals() {
        let app = crate::ui::app::App::new();
        app.speed_slot.store(vec![speed_result(60.6, 0.25, 256)]);
        for (w, h) in [(40, 10), (20, 6), (80, 24)] {
            let _ = render_live_text(&app, w, h);
        }
    }
}
