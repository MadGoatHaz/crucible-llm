//! View 1 — Live Monitor & Telemetry (blueprint §6):
//! telemetry gauges, ITL latency distribution, active-streams matrix,
//! rolling throughput chart, and the log/event stream.
//!
//! Every panel reads the shared `ArcSwap<MetricsSnapshot>` (Chunk 6) lock-free
//! via `MetricsState::load()`; the panel geometry matches the blueprint
//! mockup.

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
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
            Constraint::Percentage(30), // telemetry gauges | ITL distribution
            Constraint::Percentage(30), // active streams monitor
            Constraint::Percentage(25), // rolling throughput chart
            Constraint::Percentage(15), // log & event stream
        ])
        .split(area);

    render_top(rows[0], m, f);
    render_streams(rows[1], m, f);
    render_throughput(rows[2], m, f);
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

/// Top-left: aggregate throughput, active streams, power, VRAM bar.
fn render_gauges(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let vram_ratio = (m.vram_used_gb / m.vram_total_gb).clamp(0.0, 1.0);
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

/// Top-right: ITL percentiles + histogram sparkline.
fn render_itl(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(1), Constraint::Min(0)])
        .split(area);

    let line = Line::from(vec![
        Span::styled(" p50: ", style::label()),
        Span::styled(format!("{:.1} ms", m.itl_p50_ms()), style::value()),
        Span::styled("  |  p90: ", style::label()),
        Span::styled(format!("{:.1} ms", m.itl_p90_ms()), style::value()),
        Span::styled("  |  p99: ", style::label()),
        Span::styled(format!("{:.1} ms", m.itl_p99_ms()), style::value_warn()),
    ]);
    f.render_widget(Paragraph::new(line), chunks[0]);

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

/// Bottom chart: rolling aggregate tokens/sec over the test epoch.
fn render_throughput(area: Rect, m: &MetricsSnapshot, f: &mut Frame) {
    let max = m
        .throughput_series
        .iter()
        .cloned()
        .fold(0.0_f64, f64::max)
        .max(100.0);
    let points: Vec<(f64, f64)> = m
        .throughput_series
        .iter()
        .enumerate()
        .map(|(i, &v)| (i as f64, v))
        .collect();
    let data = Dataset::default()
        .marker(Marker::Braille)
        .graph_type(GraphType::Line)
        .style(palette::ACCENT)
        .data(&points);
    f.render_widget(
        Chart::new(vec![data])
            .block(
                Block::default()
                    .borders(Borders::ALL)
                    .border_style(style::border())
                    .title("REAL-TIME SYSTEM PERFORMANCE (TOKENS/SEC OVER TIME)"),
            )
            .x_axis(
                Axis::default()
                    .style(style::footer())
                    .bounds([0.0, 60.0])
                    .labels(vec![
                        Line::from("0s"),
                        Line::from("15s"),
                        Line::from("30s"),
                        Line::from("45s"),
                        Line::from("60s"),
                    ]),
            )
            .y_axis(
                Axis::default()
                    .style(style::footer())
                    .bounds([0.0, max * 1.1])
                    .labels(vec![
                        Line::from("0"),
                        Line::from(format!("{:.0}", (max * 0.55).round())),
                        Line::from(format!("{:.0}", (max * 1.1).round())),
                    ]),
            ),
        area,
    );
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
