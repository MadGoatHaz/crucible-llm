//! View 6 — GPU & Power Monitor (the dedicated hardware / energy panel).
//!
//! The "wow" panel for AI-server operators: everything that matters about
//! the *power and energy* side of a run, on one screen. It reads the
//! [`crate::hw::GpuPowerMonitor`] the 100 ms poller maintains (copied into
//! the lock-free [`crate::metrics::state::MetricsSnapshot`] — the render
//! loop never blocks and never touches the timing path, blueprint §4).
//!
//! Layout (top → bottom):
//!
//! * **System Power** — total / idle / compute / peak draw, energy (kWh),
//!   estimated cost (`$/kWh`), and the run duration.
//! * **Power over time** — the 1 Hz aggregate power trace as a bar chart
//!   (the "wow" factor), beside an **Efficiency** readout (J/token,
//!   J/ktoken, tokens/W, $/1M tokens, avg/peak/idle/compute power).
//! * **Per-GPU table** — one row per device: power, utilization,
//!   temperature, VRAM, core/mem clock, throttle (the multi-GPU showcase).
//! * **Utilization + Temperature over time** — two more 1 Hz charts.
//!
//! **No GPU** (a driver-less host) → a single "no telemetry" placeholder,
//! never a broken frame (the N/A rule, blueprint §5D).
//!
//! Every panel is a pure function of `&App` state and degrades gracefully
//! to "awaiting samples…" when the history is still empty.

use ratatui::layout::{Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::hw::GpuPowerMonitor;
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, style, Theme};

/// Render the GPU & Power view (View 6) into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
    let snap = app.metrics.load();
    let m = snap.as_ref();

    // No GPU telemetry (a driver-less host) → a clean placeholder, never a
    // broken frame (the N/A rule, blueprint §5D).
    let Some(mon) = &m.gpu_monitor else {
        render_no_gpu(area, th, f);
        return;
    };

    // The title: the vendor + model, with the device count when multi-GPU.
    let title = match &app.gpu {
        Some(be) => {
            let name = crate::hw::gpu_display_name(be.as_ref());
            if mon.gpus.len() > 1 {
                format!("GPU & POWER — {}× {}", mon.gpus.len(), name)
            } else {
                format!("GPU & POWER — {name}")
            }
        }
        None => "GPU & POWER MONITOR".to_string(),
    };

    // The vertical plan: all four sections are percentage-sized (plus a small
    // fixed header) so they never overflow the 80×24 minimum terminal — a
    // long per-GPU table (8+ cards) clips its lowest rows rather than pushing
    // the charts off-screen.
    let plan: Vec<(u8, Constraint)> = vec![
        (0, Constraint::Length(4)),      // system power
        (1, Constraint::Percentage(38)), // power over time | efficiency
        (2, Constraint::Percentage(36)), // per-GPU table
        (3, Constraint::Percentage(22)), // utilization + temperature charts
    ];
    let constraints: Vec<Constraint> = plan.iter().map(|(_, c)| *c).collect();
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_system_power(rects[0], mon, &title, th, f);
    render_power_efficiency_row(rects[1], mon, th, f);
    render_gpu_table(rects[2], mon, th, f);
    render_util_temp_row(rects[3], mon, th, f);
}

/// The "no GPU telemetry" placeholder (a driver-less host).
fn render_no_gpu(area: Rect, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "GPU & POWER MONITOR"),
        style::border(th),
    );
    let text = Text::from(vec![
        Line::from(Span::styled(
            "No GPU telemetry available.",
            style::value(th),
        )),
        Line::from(Span::styled(
            "This panel needs a local GPU with driver support (NVIDIA / AMD / Intel).",
            style::info(th),
        )),
        Line::from(Span::styled(
            "On a remote machine the hardware fields report N/A — run on the GPU box to see power, energy, and cost.",
            style::info(th),
        )),
    ]);
    f.render_widget(Paragraph::new(text).block(block), area);
}

// ── System power summary ───────────────────────────────────────────────────

/// The top **System Power** panel: total / idle / compute / peak draw, the
/// energy consumed (kWh), the estimated cost, and the run duration.
fn render_system_power(area: Rect, mon: &GpuPowerMonitor, title: &str, th: Theme, f: &mut Frame) {
    let block = theme::block(theme::panel_title(th, title), style::active_border(th));
    if area.width < 40 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let total = mon.total_power_w;
    let idle = mon.idle_power_w;
    let compute = mon.compute_power_w();
    let peak = mon.peak_power_w;
    let kwh = mon.energy_kwh();
    let cost = mon.cost_usd();
    let dur = mon.duration_sec();
    let lines = vec![
        Line::from(vec![
            Span::styled("  Total: ", style::label(th)),
            Span::styled(format_w(total), style::value(th)),
            Span::styled("   Idle: ", style::footer(th)),
            Span::styled(format_w(idle), style::value_secondary(th)),
            Span::styled("   Compute: ", style::footer(th)),
            Span::styled(format_w(compute), style::value_ok(th)),
            Span::styled("   Peak: ", style::footer(th)),
            Span::styled(format_w(peak), style::value_warn(th)),
        ]),
        Line::from(vec![
            Span::styled("  Energy: ", style::label(th)),
            Span::styled(format!("{kwh:.3} kWh"), style::value(th)),
            Span::styled("   Est. Cost: ", style::footer(th)),
            Span::styled(format_money(cost), style::value_ok(th)),
            Span::styled(
                format!("   @ ${:.2}/kWh", mon.rate_per_kwh),
                style::info(th),
            ),
            Span::styled("   Duration: ", style::footer(th)),
            Span::styled(fmt::format_duration(dur), style::value(th)),
        ]),
        Line::from(vec![
            Span::styled("  Avg: ", style::label(th)),
            Span::styled(
                mon.avg_power_w()
                    .map(format_w)
                    .unwrap_or_else(|| "N/A".to_string()),
                style::value(th),
            ),
            Span::styled("   Max Temp: ", style::footer(th)),
            Span::styled(
                if mon.max_temp_c > 0.0 {
                    format!("{:.0}°C", mon.max_temp_c)
                } else {
                    "N/A".to_string()
                },
                style::value(th),
            ),
            Span::styled("   Throttle: ", style::footer(th)),
            Span::styled(
                if mon.throttle_events > 0 {
                    format!("{} event(s)", mon.throttle_events)
                } else {
                    "none".to_string()
                },
                if mon.throttle_events > 0 {
                    style::value_warn(th)
                } else {
                    style::value_ok(th)
                },
            ),
        ]),
    ];
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true }),
        area,
    );
}

// ── Power-over-time graph + efficiency readout ─────────────────────────────

/// The middle row: the **power-over-time** bar chart (left) and the
/// **efficiency** readout (right).
fn render_power_efficiency_row(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(area);
    render_power_graph(cols[0], mon, th, f);
    render_efficiency(cols[1], mon, th, f);
}

/// The **power-over-time** bar chart: the 1 Hz aggregate power trace.
fn render_power_graph(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "POWER OVER TIME (W)"),
        style::border(th),
    );
    if area.width < 12 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    let values: Vec<f64> = mon.history.iter().map(|s| s.power_w).collect();
    let lines = build_bar_chart(&values, inner.width, inner.height, "W", th);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

/// The **efficiency** readout: J/token, J/ktoken, tokens/W, $/1M tokens, and
/// the avg / peak / idle / compute power figures.
fn render_efficiency(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(theme::panel_title(th, "EFFICIENCY"), style::border(th));
    if area.width < 12 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let jpt = mon.joules_per_token();
    let jkt = mon.joules_per_ktoken();
    let tpw = mon.tokens_per_watt();
    let cmt = mon.cost_per_million_tokens();
    let na = "N/A".to_string();
    let lines = vec![
        eff_row(
            th,
            "J/token",
            jpt.map(|v| format!("{v:.3}")).unwrap_or(na.clone()),
        ),
        eff_row(th, "J/ktoken", jkt.map(format_int).unwrap_or(na.clone())),
        eff_row(
            th,
            "Tokens/W",
            tpw.map(|v| format!("{v:.1}")).unwrap_or(na.clone()),
        ),
        eff_row(
            th,
            "$/1M tokens",
            cmt.map(format_money).unwrap_or(na.clone()),
        ),
        eff_row(th, "Total tokens", fmt::format_tokens(mon.total_tokens)),
        Line::raw(""),
        eff_row(
            th,
            "Avg power",
            mon.avg_power_w()
                .map(format_w)
                .unwrap_or_else(|| "N/A".to_string()),
        ),
        eff_row(th, "Peak power", format_w(mon.peak_power_w)),
        eff_row(th, "Idle power", format_w(mon.idle_power_w)),
        eff_row(th, "Compute power", format_w(mon.compute_power_w())),
    ];
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true }),
        area,
    );
}

fn eff_row(th: Theme, label: &str, value: String) -> Line<'static> {
    Line::from(vec![
        Span::styled(format!("{:<14}", format!("{label}:")), style::label(th)),
        Span::styled(value, style::value(th)),
    ])
}

// ── Per-GPU table ──────────────────────────────────────────────────────────

/// The **per-GPU** table: one row per device (power, utilization,
/// temperature, VRAM, core/mem clock, throttle). The multi-GPU showcase.
fn render_gpu_table(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(
            th,
            format!(
                "PER-GPU ({} device{})",
                mon.gpus.len(),
                if mon.gpus.len() == 1 { "" } else { "s" }
            ),
        ),
        style::border(th),
    );
    if area.width < 60 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    if mon.gpus.is_empty() {
        f.render_widget(
            Paragraph::new(Text::from(vec![Line::from(Span::styled(
                "  Awaiting GPU samples…",
                style::info(th),
            ))]))
            .block(block),
            area,
        );
        return;
    }
    let header = Line::from(vec![
        Span::styled(format!("{:<4}", "GPU"), style::muted_title(th)),
        Span::styled(format!("{:<16}", "Name"), style::muted_title(th)),
        Span::styled(format!("{:>6}", "Power"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "Util"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "Temp"), style::muted_title(th)),
        Span::styled(format!("{:<13}", "VRAM"), style::muted_title(th)),
        Span::styled(format!("{:<12}", "Clock(C/M)"), style::muted_title(th)),
        Span::styled("Throttle".to_string(), style::muted_title(th)),
    ]);
    let mut lines = vec![header];
    for (i, g) in mon.gpus.iter().enumerate() {
        let name = mon
            .gpu_names
            .get(i)
            .map(|s| fmt::truncate(s, 16))
            .unwrap_or_else(|| format!("GPU {i}"));
        lines.push(gpu_row(th, i, &name, g));
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true }),
        area,
    );
}

fn gpu_row(th: Theme, i: usize, name: &str, g: &crate::hw::GpuSample) -> Line<'static> {
    let power = g
        .power_watts
        .map(|w| format!("{w:.0}W"))
        .unwrap_or_else(|| "N/A".to_string());
    let util = g
        .utilization_pct
        .map(|u| format!("{u}%"))
        .unwrap_or_else(|| "N/A".to_string());
    let temp = g
        .temperature_c
        .map(|t| format!("{t}°C"))
        .unwrap_or_else(|| "N/A".to_string());
    let vram = match (g.memory_used_mb, g.memory_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            format!("{:.1}/{:.1}GB", used as f64 / 1024.0, total as f64 / 1024.0)
        }
        (Some(used), None) => format!("{:.1}GB", used as f64 / 1024.0),
        _ => "N/A".to_string(),
    };
    let clock = match (g.core_clock_mhz, g.memory_clock_mhz) {
        (Some(c), Some(m)) => format!("{c}/{m}"),
        (Some(c), None) => format!("{c}/-"),
        (None, Some(m)) => format!("-/{m}"),
        _ => "N/A".to_string(),
    };
    let throttle = g.throttle_reasons.as_deref().unwrap_or("None");
    let throttle_style = if throttle == "None" {
        style::value_ok(th)
    } else {
        style::value_warn(th)
    };
    Line::from(vec![
        Span::styled(format!("{i:<3} "), style::label(th)),
        Span::styled(format!("{:<16}", name), style::value_secondary(th)),
        Span::styled(format!("{power:>6} "), style::value(th)),
        Span::styled(format!("{util:>5} "), style::value(th)),
        Span::styled(format!("{temp:>5} "), style::value(th)),
        Span::styled(format!("{vram:<13} "), style::value_secondary(th)),
        Span::styled(format!("{clock:<12} "), style::value_secondary(th)),
        Span::styled(throttle.to_string(), throttle_style),
    ])
}

// ── Utilization + temperature over time ────────────────────────────────────

/// The bottom row: the **utilization** (left) and **temperature** (right)
/// 1 Hz bar charts.
fn render_util_temp_row(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    render_util_chart(cols[0], mon, th, f);
    render_temp_chart(cols[1], mon, th, f);
}

fn render_util_chart(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "UTILIZATION OVER TIME (%)"),
        style::border(th),
    );
    if area.width < 10 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    let values: Vec<f64> = mon.history.iter().map(|s| s.util_pct).collect();
    let lines = build_bar_chart(&values, inner.width, inner.height, "%", th);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

fn render_temp_chart(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "TEMPERATURE OVER TIME (°C)"),
        style::border(th),
    );
    if area.width < 10 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let inner = area.inner(Margin {
        horizontal: 1,
        vertical: 1,
    });
    let values: Vec<f64> = mon.history.iter().map(|s| s.temp_c).collect();
    let lines = build_bar_chart(&values, inner.width, inner.height, "°C", th);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

// ── Generic auto-scaled bar chart ──────────────────────────────────────────

/// A generic **auto-scaled** vertical bar chart (one `█` column per sample,
/// newest at the right edge), with a y-axis (max / mid / 0 + unit) and an
/// x-axis time scale. The "wow" visual for the power / utilization /
/// temperature traces. Pure over its inputs (unit-testable, no terminal).
///
/// An empty series renders "Awaiting samples…" (never a blank panel).
fn build_bar_chart(values: &[f64], w: u16, h: u16, unit: &str, th: Theme) -> Vec<Line<'static>> {
    const Y_AXIS_W: usize = 6;
    let w = w as usize;
    let h = h as usize;
    if w < Y_AXIS_W + 4 || h < 3 {
        return vec![Line::from("chart too small")];
    }
    if values.is_empty() {
        return vec![Line::from(Span::styled(
            "Awaiting samples…",
            style::info(th),
        ))];
    }

    let plot_w = w - Y_AXIS_W;
    let plot_h = h - 1; // bottom row reserved for the x-axis
                        // Auto-scale to the data maximum with 10% headroom (a `max(1.0)` floor
                        // keeps a tiny signal from blowing the axis up to absurdity).
    let max = values.iter().cloned().fold(0.0_f64, f64::max) * 1.1;
    let max = max.max(1.0);

    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w]; h];

    // One gradient bar per plot column; right-aligned so the newest sample
    // sits at the right edge. Each bar ramps dim → mid → bright so it glows.
    for col in 0..plot_w {
        let idx = values.len().saturating_sub(plot_w - col);
        if idx >= values.len() {
            continue;
        }
        let v = values[idx];
        let ratio = (v / max).clamp(0.0, 1.0);
        let height = (ratio * plot_h as f64).round() as usize;
        if height == 0 {
            continue;
        }
        let is_now = col == plot_w - 1; // the live sample (right edge)
        let grid_col = Y_AXIS_W + col;
        for r in 0..height {
            let grid_row = plot_h.saturating_sub(1).saturating_sub(r);
            if grid_row < h && grid_col < w {
                let frac = (r as f64 + 0.5) / height as f64;
                let color = if is_now {
                    th.bright()
                } else if frac > 0.66 {
                    th.primary()
                } else if frac > 0.33 {
                    th.tertiary()
                } else {
                    th.floor()
                };
                grid[grid_row][grid_col] = ('█', Some(color));
            }
        }
    }

    // The y-axis numbers (top = max, middle = max/2, bottom = 0) + unit,
    // drawn last so they win over any bar reaching the gutter.
    let mut write_y = |row: usize, value: f64| {
        if row >= h {
            return;
        }
        let num = if value >= 1000.0 {
            format!("{:.0}k", value / 1000.0)
        } else {
            format!("{value:.0}")
        };
        let text = format!("{num}{unit}");
        let start = Y_AXIS_W.saturating_sub(text.len());
        for (i, ch) in text.chars().enumerate() {
            let col = start + i;
            if col < Y_AXIS_W {
                grid[row][col] = (ch, Some(th.dim()));
            }
        }
    };
    write_y(0, max);
    write_y(plot_h / 2, max / 2.0);
    write_y(plot_h.saturating_sub(1), 0.0);

    // x-axis: a baseline + a time scale (0 … N−1 s, one sample/sec).
    let span = (values.len() - 1).max(1);
    for cell in &mut grid[h - 1][Y_AXIS_W..] {
        *cell = ('─', Some(th.dim()));
    }
    grid[h - 1][Y_AXIS_W] = ('├', Some(th.dim()));
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
                    grid[h - 1][col] = (ch, Some(th.dim()));
                }
            }
            label_end = start + text.len() as i64;
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

/// Format a power value in watts: `1,234 W` (thousands separator) or `N/A`
/// for a non-positive reading.
fn format_w(watts: f64) -> String {
    if watts <= 0.0 {
        return "N/A".to_string();
    }
    let s = format!("{:.0}", watts);
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(bytes.len() + bytes.len() / 3 + 2);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*b as char);
    }
    out.push_str(" W");
    out
}

/// Format a money value: `$0.07` (two decimals) or `N/A` for non-positive.
fn format_money(v: f64) -> String {
    if v <= 0.0 {
        return "N/A".to_string();
    }
    format!("${v:.2}")
}

/// Format a large integer with thousands separators (J/ktoken).
fn format_int(v: f64) -> String {
    let n = v as u64;
    let s = n.to_string();
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hw::GpuSample;
    use crate::metrics::state::MetricsSnapshot;
    use ratatui::backend::TestBackend;
    use ratatui::Terminal;

    fn render_text(app: &App, w: u16, h: u16) -> String {
        let backend = TestBackend::new(w, h);
        let mut terminal = Terminal::new(backend).expect("terminal");
        terminal.draw(|f| app.render(f)).expect("frame");
        terminal
            .backend()
            .buffer()
            .content()
            .iter()
            .map(|c| c.symbol())
            .collect::<String>()
    }

    fn app_with_monitor(mon: GpuPowerMonitor) -> App {
        let app = App::new();
        app.metrics.update(MetricsSnapshot {
            gpu_monitor: Some(mon),
            ..Default::default()
        });
        app
    }

    /// A monitor with a small power history + two GPUs, for the render tests.
    fn sample_monitor() -> GpuPowerMonitor {
        let mut mon = GpuPowerMonitor::default().with_rate(0.15);
        mon.begin_run();
        mon.record_idle(150.0);
        mon.start_load();
        mon.has_power = true;
        mon.total_power_w = 1200.0;
        mon.peak_power_w = 1400.0;
        mon.max_temp_c = 72.0;
        mon.avg_util_pct = 91.0;
        mon.total_tokens = 20000;
        mon.gpus = vec![
            GpuSample {
                power_watts: Some(600.0),
                utilization_pct: Some(92),
                temperature_c: Some(70),
                memory_used_mb: Some(17000),
                memory_total_mb: Some(16384),
                core_clock_mhz: Some(555),
                memory_clock_mhz: Some(1219),
                throttle_reasons: None,
            },
            GpuSample {
                power_watts: Some(600.0),
                utilization_pct: Some(90),
                temperature_c: Some(69),
                memory_used_mb: Some(17000),
                memory_total_mb: Some(16384),
                core_clock_mhz: Some(550),
                memory_clock_mhz: Some(1215),
                throttle_reasons: Some("thermal".into()),
            },
        ];
        mon.gpu_names = vec!["NVIDIA A4000".into(), "NVIDIA A4000".into()];
        // A small power history (a few seconds of the run).
        for i in 0..10 {
            mon.history.push(crate::hw::PowerSample {
                t: crate::timing::MonotonicInstant::now(),
                power_w: 1000.0 + (i as f64) * 20.0,
                util_pct: 90.0,
                temp_c: 70.0,
                vram_gb: 32.0,
            });
        }
        mon
    }

    #[test]
    fn view6_renders_system_power_and_per_gpu() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("GPU & POWER"), "title: {text}");
        assert!(text.contains("Total:"), "system power total: {text}");
        assert!(text.contains("PER-GPU"), "per-GPU table: {text}");
        assert!(text.contains("A4000"), "GPU name: {text}");
        assert!(text.contains("thermal"), "throttle reason: {text}");
    }

    #[test]
    fn view6_renders_efficiency_and_graphs() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("EFFICIENCY"), "efficiency panel: {text}");
        assert!(text.contains("POWER OVER TIME"), "power graph: {text}");
        assert!(text.contains("UTILIZATION OVER TIME"), "util graph: {text}");
        assert!(text.contains("TEMPERATURE OVER TIME"), "temp graph: {text}");
    }

    #[test]
    fn view6_no_gpu_shows_the_placeholder() {
        // A monitor is present but empty (no GPUs) → the per-GPU table shows
        // its "awaiting samples" state, and the panel still renders.
        let mut app = App::new();
        app.view = crate::ui::app::View::Gpu;
        // No gpu_monitor set → the "no GPU telemetry" placeholder.
        let text = render_text(&app, 120, 40);
        assert!(text.contains("No GPU telemetry"), "placeholder: {text}");
    }

    #[test]
    fn format_w_uses_thousands_separator() {
        assert_eq!(format_w(1234.0), "1,234 W");
        assert_eq!(format_w(0.0), "N/A");
        assert_eq!(format_w(95.0), "95 W");
    }

    #[test]
    fn format_money_two_decimals() {
        assert_eq!(format_money(0.07), "$0.07");
        assert_eq!(format_money(0.0), "N/A");
        assert_eq!(format_money(12.345), "$12.35");
    }

    #[test]
    fn build_bar_chart_empty_shows_awaiting() {
        let lines = build_bar_chart(&[], 40, 10, "W", Theme::Cyberpunk);
        assert!(lines[0].to_string().contains("Awaiting"));
    }

    #[test]
    fn build_bar_chart_renders_bars_and_axis() {
        let vals = vec![100.0, 200.0, 300.0, 400.0];
        let lines = build_bar_chart(&vals, 40, 10, "W", Theme::Cyberpunk);
        assert_eq!(lines.len(), 10);
        // The plot area contains bar blocks.
        let joined: String = lines.iter().map(|l| l.to_string()).collect();
        assert!(joined.contains('█'), "bars present: {joined}");
        // The y-axis shows the unit.
        assert!(joined.contains('W'), "unit on y-axis: {joined}");
    }
}
