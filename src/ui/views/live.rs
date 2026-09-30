//! View 1 — Live Monitor & Telemetry (blueprint §6):
//! telemetry gauges, ITL percentile gauge bars + latency distribution,
//! active-streams matrix, a rolling throughput sparkline (unicode block
//! ramp with a green/yellow/red gradient), a large token counter, and the
//! log/event stream.
//!
//! Every panel reads the shared `ArcSwap<MetricsSnapshot>` (Chunk 6) lock-free
//! via `MetricsState::load()`; the panel geometry matches the blueprint
//! mockup.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Style};
use ratatui::symbols::Marker;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Axis, Block, Borders, Cell, Chart, Dataset, Gauge, GraphType, Paragraph, Row, Table, Wrap,
};
use ratatui::Frame;

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
            Constraint::Percentage(30), // telemetry gauges | ITL gauge bars + distribution
            Constraint::Percentage(30), // active streams monitor
            Constraint::Percentage(25), // throughput sparkline | token counter
            Constraint::Percentage(15), // log & event stream
        ])
        .split(area);

    render_top(rows[0], m, f);
    render_streams(rows[1], m, f);
    render_bottom(rows[2], m, f);
    render_log(rows[3], app, f);
}

/// Top row: telemetry gauges (left) + ITL distribution (right).
fn render_top(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(area);
    render_gauges(cols[0], m, f);
    render_itl(cols[1], m, f);
}

fn kv(label: &str, value: String, value_style: Style) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{label:<22} "), style::label()),
        Span::styled(value, value_style),
    ])
}

/// Top-left: aggregate throughput, active streams, GPU clock, power
/// (with J/token), and the VRAM capacity bar (blueprint §6 View 1
/// "Key Metrics": aggregate t/s, VRAM bar, GPU core frequency,
/// energy efficiency).
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

/// Top-right: ITL percentile gauge bars (p50/p90/p99) + histogram.
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
/// MTP rate, progress).
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

/// Bottom: scrolling log / event stream.
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
}
