//! View 4 — GPU & Power Monitor (the dedicated hardware / energy panel).
//!
//! The "wow" panel for AI-server operators: everything that matters about
//! the *power and energy* side of a run, on one screen. It reads the
//! [`crate::hw::GpuPowerMonitor`] the 100 ms poller maintains (copied into
//! the lock-free [`crate::metrics::state::MetricsSnapshot`] — the render
//! loop never blocks and never touches the timing path, blueprint §4).
//!
//! Layout (top → bottom):
//!
//! * **System Power** — total / idle / compute / peak draw, energy (kWh)
//!   at the user's `$/kWh` rate, and the run duration (frozen at its
//!   final value once the run completes).
//! * **Power over time** — the 1 Hz aggregate power trace as a bar chart
//!   (the "wow" factor), beside an **Efficiency** readout (J/token,
//!   J/ktoken, tokens/W, $/1M tokens, avg/peak/idle/compute power).
//! * **Per-GPU table** — one row per device: power, current *and*
//!   whole-run average utilization, current *and* whole-run average
//!   temperature, peak temperature, VRAM, core/mem clock, throttle (the
//!   multi-GPU showcase). The Name column is a **fixed 24-char** width
//!   (pad/truncate) so all subsequent columns align perfectly.
//! * **Cost analysis** — the headline **$/1M-token** rates: separate
//!   *input* (prefill) and *output* (decode) measured from the phase
//!   power draws, a blended rate, and the run's total cost — all at the
//!   user's live `$/kWh` rate from the Config form.
//! * **Utilization + Temperature over time** — two 1 Hz **line** charts
//!   whose y-axis **auto-scales to the actual data range** (with padding):
//!   a signal hovering 85–97% renders as a visible curve, not a flat line
//!   at the top of a 0–100 axis. Each carries grid lines, a dim area fill
//!   under the curve, a dashed `avg:` line, a `peak:` marker, and time
//!   markers on the x-axis.
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

/// Render the GPU & Power view (View 4) into `area`.
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

    // The live `$/kWh` rate from the Config form (the user may have changed
    // it since startup; the monitor's own `rate_per_kwh` is a startup
    // snapshot and can be stale).
    let rate = crate::ui::views::config::current_rate(app);

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

    // The vertical plan: all five sections are percentage-sized (plus a
    // small fixed header) so they never overflow the 80×24 minimum
    // terminal — a long per-GPU table (8+ cards) clips its lowest rows
    // rather than pushing the charts off-screen.
    let plan: Vec<(u8, Constraint)> = vec![
        (0, Constraint::Length(6)),      // system power
        (1, Constraint::Percentage(32)), // power over time | efficiency
        (2, Constraint::Percentage(28)), // per-GPU table
        (3, Constraint::Percentage(20)), // cost analysis ($/1M tokens)
        (4, Constraint::Percentage(16)), // utilization + temperature charts
    ];
    let constraints: Vec<Constraint> = plan.iter().map(|(_, c)| *c).collect();
    let rects = Layout::default()
        .direction(Direction::Vertical)
        .constraints(constraints)
        .split(area);

    render_system_power(rects[0], mon, &title, rate, th, f);
    render_power_efficiency_row(rects[1], mon, rate, th, f);
    render_gpu_table(rects[2], mon, th, f);
    render_cost_analysis(rects[3], mon, rate, th, f);
    render_util_temp_row(rects[4], mon, th, f);
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
/// energy consumed (kWh) at the user's `$/kWh` rate, and the run duration.
/// (The per-1M-token *cost* lives in the dedicated COST ANALYSIS panel.)
fn render_system_power(
    area: Rect,
    mon: &GpuPowerMonitor,
    title: &str,
    rate: f64,
    th: Theme,
    f: &mut Frame,
) {
    let block = theme::block(theme::panel_title(th, title), style::active_border(th));
    if area.width < 40 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let idle = mon.idle_power_w;
    let compute = mon.compute_power_w();
    let peak = mon.peak_power_w;
    let kwh = mon.energy_kwh();
    let dur = mon.duration_sec();
    let lines = vec![
        // The total *system* draw, split into its GPU and CPU components
        // (the CPU power is the draw the GPU-only figure used to miss).
        Line::from(vec![
            Span::styled("  System: ", style::label(th)),
            Span::styled(format_w(mon.system_power_w()), style::value(th)),
            Span::styled(
                format!(
                    "  (GPU {} + CPU {})",
                    format_w(mon.total_power_w),
                    format_w(mon.cpu_power_w)
                ),
                style::footer(th),
            ),
        ]),
        Line::from(vec![
            Span::styled("  Idle: ", style::label(th)),
            Span::styled(format_w(idle), style::value_secondary(th)),
            Span::styled("   Compute: ", style::footer(th)),
            Span::styled(format_w(compute), style::value_ok(th)),
            Span::styled("   Peak: ", style::footer(th)),
            Span::styled(format_w(peak), style::value_warn(th)),
        ]),
        Line::from(vec![
            Span::styled("  Energy: ", style::label(th)),
            Span::styled(format!("{kwh:.3} kWh"), style::value(th)),
            Span::styled(format!("   @ ${:.2}/kWh", rate), style::info(th)),
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
fn render_power_efficiency_row(
    area: Rect,
    mon: &GpuPowerMonitor,
    rate: f64,
    th: Theme,
    f: &mut Frame,
) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(62), Constraint::Percentage(38)])
        .split(area);
    render_power_graph(cols[0], mon, th, f);
    render_efficiency(cols[1], mon, rate, th, f);
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
fn render_efficiency(area: Rect, mon: &GpuPowerMonitor, rate: f64, th: Theme, f: &mut Frame) {
    let block = theme::block(theme::panel_title(th, "EFFICIENCY"), style::border(th));
    if area.width < 12 || area.height < 4 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let jpt = mon.joules_per_token();
    let jkt = mon.joules_per_ktoken();
    let tpw = mon.tokens_per_watt();
    // The blended $/1M rate from the phase-aware cost math (the same
    // number the COST ANALYSIS panel shows), at the live config rate.
    let cmt = mon.token_costs_at_rate(rate).map(|c| c.blended);
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

/// The **per-GPU** table: one row per device (power, current *and*
/// whole-run average utilization, current *and* whole-run average
/// temperature, peak temperature, VRAM, core/mem clock, throttle). The
/// multi-GPU showcase — the `AvgU` / `AvgT` / `MaxT` columns give the full
/// run picture, not just "right now".
///
/// The **Name** column is a **fixed 24-char** width (pad with spaces,
/// truncate with `…` when longer) so all subsequent columns align
/// perfectly regardless of the card name length.
fn render_gpu_table(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    const NAME_W: usize = 24;
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
    let name_w = NAME_W;
    let header = Line::from(vec![
        Span::styled(format!("{:<3}", "GPU"), style::muted_title(th)),
        Span::styled(format!("{:<name_w$}", "Name"), style::muted_title(th)),
        Span::styled(format!("{:>6}", "Power"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "Util"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "AvgU"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "Temp"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "AvgT"), style::muted_title(th)),
        Span::styled(format!("{:>5}", "MaxT"), style::muted_title(th)),
        Span::styled(format!("{:>8}", "VRAM"), style::muted_title(th)),
        Span::styled(format!("{:>8}", "Clock"), style::muted_title(th)),
        Span::styled("Thr", style::muted_title(th)),
    ]);
    let mut lines = vec![header];
    for (i, g) in mon.gpus.iter().enumerate() {
        let name = mon
            .gpu_names
            .get(i)
            .map(|s| fmt::truncate(s, name_w))
            .unwrap_or_else(|| format!("GPU {i}"));
        lines.push(gpu_row(th, i, &name, g, mon, name_w));
    }
    // The column legend (whole-run figures explained in one line).
    lines.push(Line::from(Span::styled(
        "  ℹ AvgU/AvgT = run average · MaxT = peak",
        style::info(th),
    )));
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true }),
        area,
    );
}

fn gpu_row(
    th: Theme,
    i: usize,
    name: &str,
    g: &crate::hw::GpuSample,
    mon: &GpuPowerMonitor,
    name_w: usize,
) -> Line<'static> {
    let power = g
        .power_watts
        .map(|w| format!("{w:.0}W"))
        .unwrap_or_else(|| "N/A".to_string());
    let util = g
        .utilization_pct
        .map(|u| format!("{u}%"))
        .unwrap_or_else(|| "N/A".to_string());
    // Whole-run per-device statistics (the monitor's running averages and
    // peak — `N/A` until the device has reported the metric).
    let avg_util = mon
        .avg_util_per_gpu
        .get(i)
        .copied()
        .filter(|v| *v > 0.0)
        .map(|v| format!("{v:.0}%"))
        .unwrap_or_else(|| "N/A".to_string());
    let temp = g
        .temperature_c
        .map(|t| format!("{t}°C"))
        .unwrap_or_else(|| "N/A".to_string());
    let avg_temp = mon
        .avg_temp_per_gpu
        .get(i)
        .copied()
        .filter(|v| *v > 0.0)
        .map(|v| format!("{v:.0}°C"))
        .unwrap_or_else(|| "N/A".to_string());
    let max_temp = mon
        .max_temp_per_gpu
        .get(i)
        .copied()
        .filter(|v| *v > 0)
        .map(|v| format!("{v}°C"))
        .unwrap_or_else(|| "N/A".to_string());
    let vram = match (g.memory_used_mb, g.memory_total_mb) {
        (Some(used), Some(total)) if total > 0 => {
            format!("{:.1}/{:.0}G", used as f64 / 1024.0, total as f64 / 1024.0)
        }
        (Some(used), None) => format!("{:.1}G", used as f64 / 1024.0),
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
        Span::styled(format!("{i:<3}"), style::label(th)),
        Span::styled(format!(" {name:<name_w$}"), style::value_secondary(th)),
        Span::styled(format!(" {power:>6}"), style::value(th)),
        Span::styled(format!(" {util:>5}"), style::value(th)),
        Span::styled(format!(" {avg_util:>5}"), style::value_secondary(th)),
        Span::styled(format!(" {temp:>5}"), style::value(th)),
        Span::styled(format!(" {avg_temp:>5}"), style::value_secondary(th)),
        Span::styled(format!(" {max_temp:>5}"), style::value_warn(th)),
        Span::styled(format!(" {vram:>8}"), style::value_secondary(th)),
        Span::styled(format!(" {clock:>8}"), style::value_secondary(th)),
        Span::styled(format!(" {throttle}"), throttle_style),
    ])
}

// ── Cost analysis ($/1M tokens) ───────────────────────────────────────────

/// The **COST ANALYSIS** panel: **$/1M tokens** for *input* (prefill) and
/// *output* (decode) **separately**, a blended rate, and the run's total
/// cost — all driven by the user's live `$/kWh` rate (read from the Config
/// form at render time) and the measured prefill / decode power draws.
fn render_cost_analysis(area: Rect, mon: &GpuPowerMonitor, rate: f64, th: Theme, f: &mut Frame) {
    let block = theme::block(theme::panel_title(th, "COST ANALYSIS"), style::border(th));
    if area.width < 40 || area.height < 3 {
        f.render_widget(Paragraph::new("").block(block), area);
        return;
    }
    let mut lines = vec![Line::from(vec![
        Span::styled("  Electricity Rate: ", style::label(th)),
        Span::styled(format!("${:.2}/kWh", rate), style::value(th)),
    ])];
    match mon.token_costs_at_rate(rate) {
        Some(c) => {
            lines.push(Line::raw(""));
            let rows: [(&str, f64); 3] = [
                ("$/1M Input (prefill):", c.cost_per_1m_input),
                ("$/1M Output (decode):", c.cost_per_1m_output),
                ("$/1M Blended:         ", c.blended),
            ];
            for (label, val) in rows {
                lines.push(Line::from(vec![
                    Span::styled(format!("  {label}"), style::label(th)),
                    Span::styled(format!(" ${}/1M", format_rate(val)), style::value(th)),
                ]));
            }
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![Span::styled(
                format!(
                    "  This run: {} input + {} output tokens = {} total",
                    fmt::format_tokens(c.prompt_tokens),
                    fmt::format_tokens(c.completion_tokens),
                    format_cost(c.total_cost)
                ),
                style::value_ok(th),
            )]));
            lines.push(Line::raw(""));
            lines.push(Line::from(vec![Span::styled(
                "  ℹ Input = energy during prefill ÷ prompt tokens.",
                style::info(th),
            )]));
            lines.push(Line::from(vec![Span::styled(
                "    Output = energy during decode ÷ completion tokens.",
                style::info(th),
            )]));
            // The prefill fallback warning (Issue D): when the TTFT window
            // had fewer than 2 power samples, we used the overall average.
            if mon.prefill_uses_fallback() {
                lines.push(Line::from(vec![Span::styled(
                    "  ⚠ Prefill too short for separate power measurement — using overall average.",
                    style::value_warn(th),
                )]));
            }
        }
        None => {
            lines.push(Line::from(Span::styled(
                "  Awaiting token + power data…",
                style::info(th),
            )));
            lines.push(Line::from(Span::styled(
                "  The $/1M rates appear once a run has input and output",
                style::info(th),
            )));
            lines.push(Line::from(Span::styled(
                "  tokens plus measured power in each phase.",
                style::info(th),
            )));
        }
    }
    f.render_widget(
        Paragraph::new(Text::from(lines))
            .block(block)
            .wrap(ratatui::widgets::Wrap { trim: true }),
        area,
    );
}

// ── Utilization + temperature over time ────────────────────────────────────

/// The bottom row: the **utilization** (left) and **temperature** (right)
/// 1 Hz line charts — auto-scaled to the data's real range (a flat 90–95%
/// signal shows its variation instead of sitting as a flat line at the top
/// of a 0–100 axis).
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
        theme::panel_title(th, "UTILIZATION OVER TIME (%) — auto-scaled"),
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
    let lines = build_line_chart(&values, inner.width, inner.height, "%", th);
    f.render_widget(Paragraph::new(Text::from(lines)).block(block), area);
}

fn render_temp_chart(area: Rect, mon: &GpuPowerMonitor, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "TEMPERATURE OVER TIME (°C) — auto-scaled"),
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
    let lines = build_line_chart(&values, inner.width, inner.height, "°C", th);
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

// ── Generic auto-scaled line chart ─────────────────────────────────────────

/// Auto-scale a y-axis to the data's *actual* range: `(min − pad, max +
/// pad)` where `pad = range × padding` (at least 1.0 unit, so a flat
/// signal never collapses to a zero-height axis).
///
/// This is the fix for the "flat line straight across" symptom: a signal
/// that hovers in a narrow band (utilization 85–97%, temperature 62–74°C)
/// renders against a scaled axis and its small variations become
/// visible — instead of sitting as an invisible line at the top of a
/// 0–100 scale.
#[must_use]
pub fn auto_scale_axis(values: &[f64], padding: f64) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }
    let min = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max - min;
    let pad = (range * padding).max(1.0);
    (min - pad, max + pad)
}

/// A "nice" x-axis time step (seconds) for a 1 Hz series spanning `span`
/// seconds: the smallest of 1/2/5/10/15/30/60/120/300/600 that yields at
/// most five labels.
fn nice_time_step(span: f64) -> f64 {
    for s in [1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0] {
        if span / s <= 5.0 {
            return s;
        }
    }
    600.0
}

/// The x-axis time label for `t` seconds: whole minutes (`2m`) once a
/// minute or more, else whole seconds (`45s`).
fn format_time_label(t: f64) -> String {
    if t >= 60.0 {
        format!("{:.0}m", t / 60.0)
    } else {
        format!("{t:.0}s")
    }
}

/// The **auto-scaled line chart** (the utilization / temperature graphs).
///
/// * the y-axis spans the data's *real* range (via [`auto_scale_axis`],
///   15% padding) — labelled at the scaled max / mid / min;
/// * subtle horizontal grid lines at the quarter marks (dim `·`);
/// * the curve in the theme's **primary** color (cyan in Cyberpunk), with
///   a very dim **area fill** (the theme's floor color) under it — the
///   fill bridges the steps between columns, so the trace is connected;
/// * a dashed **mean line** tagged `avg: …` (top-left of the plot);
/// * a **peak marker** (▲) tagged `peak: …` (top-right);
/// * an x-axis baseline with **time markers** (1 Hz samples: `0s`, `5s`,
///   `2m`, `4m`, … at a "nice" step).
///
/// Pure over its inputs (unit-testable, no terminal). An empty series
/// renders "Awaiting samples…".
fn build_line_chart(values: &[f64], w: u16, h: u16, unit: &str, th: Theme) -> Vec<Line<'static>> {
    const Y_AXIS_W: usize = 7;
    let w = w as usize;
    let h = h as usize;
    if w < Y_AXIS_W + 8 || h < 4 {
        return vec![Line::from("chart too small")];
    }
    if values.is_empty() {
        return vec![Line::from(Span::styled(
            "Awaiting samples…",
            style::info(th),
        ))];
    }

    let (min, max) = auto_scale_axis(values, 0.15);
    let plot_w = w - Y_AXIS_W;
    let plot_h = h - 1; // bottom row reserved for the x-axis

    // Map a value to its grid row (0 = top of the plot, plot_h−1 = the
    // baseline just above the x-axis).
    let row = |v: f64| {
        let ratio = ((v - min) / (max - min)).clamp(0.0, 1.0);
        (plot_h - 1).saturating_sub((ratio * plot_h as f64).round() as usize)
    };

    let mut grid: Vec<Vec<(char, Option<Color>)>> = vec![vec![(' ', None); w]; h];

    // Horizontal grid lines at the quarter marks (dim, every 2 columns —
    // subtle enough not to fight the curve).
    for frac in [0.25, 0.5, 0.75] {
        let r = (frac * plot_h as f64).round() as usize;
        if r < plot_h {
            for c in (Y_AXIS_W..w).step_by(2) {
                if grid[r][c].0 == ' ' {
                    grid[r][c] = ('·', Some(th.dim()));
                }
            }
        }
    }

    // The curve: one column per sample (right-aligned, newest at the right
    // edge, like every other chart in the app). Each column carries a very
    // dim area fill from the baseline up to the value, and the primary
    // color (cyan) runs along the tops.
    for col in 0..plot_w {
        let idx = values.len().saturating_sub(plot_w - col);
        if idx >= values.len() {
            continue;
        }
        let top = row(values[idx]);
        let grid_col = Y_AXIS_W + col;
        if grid_col >= w {
            continue;
        }
        for (r, row_cells) in grid.iter_mut().enumerate().take(plot_h).skip(top) {
            row_cells[grid_col] = if r == top {
                ('▓', Some(th.primary()))
            } else {
                ('░', Some(th.floor()))
            };
        }
    }

    // The mean line: dashed at the data's average, in the theme's tertiary
    // color (deep blue in Cyberpunk).
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    for (i, cell) in grid[row(mean)][Y_AXIS_W..].iter_mut().enumerate() {
        if cell.0 == ' ' && i % 2 == 1 {
            *cell = ('┄', Some(th.tertiary()));
        }
    }

    // The y-axis labels (scaled max / mid / min + unit), drawn before the
    // tags so the gutter stays readable; one decimal for narrow ranges.
    let decimals = if max - min < 10.0 { 1 } else { 0 };
    let mut write_y = |r: usize, v: f64| {
        if r >= h {
            return;
        }
        let text = format!("{v:.prec$}{unit}", prec = decimals);
        let start = Y_AXIS_W.saturating_sub(text.len());
        for (i, ch) in text.chars().enumerate() {
            let col = start + i;
            if col < Y_AXIS_W {
                grid[r][col] = (ch, Some(th.dim()));
            }
        }
    };
    write_y(0, max);
    write_y(plot_h / 2, (min + max) / 2.0);
    write_y(plot_h.saturating_sub(1), min);

    // The avg / peak tags (top row of the plot).
    let avg_tag = format!("avg: {mean:.1}{unit}");
    for (i, ch) in avg_tag.chars().enumerate() {
        let col = Y_AXIS_W + i;
        if col < w {
            grid[0][col] = (ch, Some(th.tertiary()));
        }
    }
    let peak = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let peak_tag = format!("peak: {peak:.1}{unit}");
    let tag_start = (w as i64 - peak_tag.len() as i64).max(Y_AXIS_W as i64) as usize;
    for (i, ch) in peak_tag.chars().enumerate() {
        let col = tag_start + i;
        if col < w {
            grid[0][col] = (ch, Some(th.accent()));
        }
    }

    // The peak marker (▲, drawn last): one row above the highest sample's
    // top cell (on the top row itself when the peak already reaches it).
    // If that cell is occupied (e.g. by the `peak:` tag) it steps down
    // until it finds a free cell (max 3 tries) — the marker then rides on
    // the area fill, which reads fine.
    if let Some(idx) = values.iter().rposition(|v| *v == peak) {
        // Plot only when the peak falls in the visible window (the newest
        // `plot_w` samples — older ones are not drawn, like the curve).
        let offset = values.len() - idx; // 1..=values.len() (rposition)
        if offset <= plot_w {
            let col = Y_AXIS_W + plot_w - offset;
            let mut marker_row = row(peak).max(1) - 1;
            for _ in 0..3 {
                if grid[marker_row][col].0 == ' ' {
                    break;
                }
                if marker_row + 1 < plot_h {
                    marker_row += 1;
                }
            }
            grid[marker_row][col] = ('▲', Some(th.accent()));
        }
    }

    // The x-axis: a baseline + time markers (1 Hz: sample index = seconds).
    let span = (values.len() - 1).max(1) as f64;
    for cell in &mut grid[h - 1][Y_AXIS_W..] {
        *cell = ('─', Some(th.dim()));
    }
    grid[h - 1][Y_AXIS_W] = ('├', Some(th.dim()));
    let step = nice_time_step(span);
    let mut label_end: i64 = -1;
    let mut t = 0.0;
    while t <= span {
        let center = Y_AXIS_W as i64 + ((t / span) * (plot_w as f64 - 1.0)).round() as i64;
        let text = format_time_label(t);
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
        t += step;
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

/// Format a **$/1M-token rate**: sub-dollar rates get three decimals
/// (`$0.031` — the precision that separates local from cloud),
/// dollar-and-up rates get two (`$2.50`).
fn format_rate(v: f64) -> String {
    if v < 1.0 {
        format!("{v:.3}")
    } else {
        format!("{v:.2}")
    }
}

/// Format a run's **total cost**: small amounts (under a cent) get four
/// decimals (`$0.0011`), larger ones two. `N/A` for non-positive.
fn format_cost(v: f64) -> String {
    if v <= 0.0 {
        return "N/A".to_string();
    }
    if v < 0.01 {
        format!("${v:.4}")
    } else {
        format!("${v:.2}")
    }
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
        let mut mon = GpuPowerMonitor::default().with_rate(0.16);
        mon.begin_run();
        mon.record_idle(150.0);
        mon.start_load();
        mon.has_power = true;
        mon.total_power_w = 1200.0;
        mon.cpu_power_w = 73.0;
        mon.peak_power_w = 1400.0;
        mon.max_temp_c = 72.0;
        mon.avg_util_pct = 91.0;
        mon.total_tokens = 20000;
        // The $/1M cost panel's phase context: input tokens, a real
        // prefill window (load start → first token), then decode.
        mon.prompt_tokens = 5000;
        std::thread::sleep(std::time::Duration::from_millis(250));
        mon.first_token_at = Some(crate::timing::MonotonicInstant::now());
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
        // Whole-run per-GPU statistics (the table's AvgU/AvgT/MaxT columns).
        mon.avg_util_per_gpu = vec![91.0, 88.0];
        mon.avg_temp_per_gpu = vec![64.0, 62.0];
        mon.max_temp_per_gpu = vec![72, 70];
        // A small power history (a few seconds of the run) with a narrow,
        // *varying* band — the test data for the auto-scaled line charts.
        for i in 0..10 {
            mon.history.push(crate::hw::PowerSample {
                t: crate::timing::MonotonicInstant::now(),
                power_w: 1000.0 + (i as f64) * 20.0,
                cpu_power_w: 0.0,
                util_pct: 85.0 + (i as f64) * 1.2, // 85 → 95.8
                temp_c: 62.0 + (i as f64) * 1.1,   // 62 → 71.9
                vram_gb: 32.0,
            });
        }
        // Decode runs until "now" (the last-token clock).
        std::thread::sleep(std::time::Duration::from_millis(250));
        mon.last_token_at = Some(crate::timing::MonotonicInstant::now());
        mon
    }

    #[test]
    fn view4_renders_system_power_and_per_gpu() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("GPU & POWER"), "title: {text}");
        assert!(text.contains("System:"), "system power total: {text}");
        assert!(text.contains("CPU"), "CPU power breakdown: {text}");
        assert!(text.contains("PER-GPU"), "per-GPU table: {text}");
        assert!(text.contains("A4000"), "GPU name: {text}");
        assert!(text.contains("thermal"), "throttle reason: {text}");
        // The whole-run per-GPU columns.
        assert!(text.contains("AvgU"), "avg util column: {text}");
        assert!(text.contains("AvgT"), "avg temp column: {text}");
        assert!(text.contains("MaxT"), "max temp column: {text}");
        assert!(text.contains("72°C"), "peak temp value: {text}");
    }

    // ── cost analysis ($/1M tokens) ──────────────────────────────────────

    #[test]
    fn view4_cost_analysis_shows_input_output_and_blended() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 50);
        assert!(text.contains("COST ANALYSIS"), "cost panel: {text}");
        assert!(text.contains("$0.16/kWh"), "the user's rate: {text}");
        assert!(text.contains("Input (prefill)"), "input row: {text}");
        assert!(text.contains("Output (decode)"), "output row: {text}");
        assert!(text.contains("Blended"), "blended row: {text}");
        assert!(text.contains("This run:"), "run total: {text}");
        // No cloud references.
        assert!(!text.contains("GPT-4o"), "no cloud refs: {text}");
        assert!(!text.contains("Claude"), "no cloud refs: {text}");
    }

    #[test]
    fn view4_cost_panel_shows_awaiting_without_token_data() {
        // A fresh monitor (no tokens, no phase latch) → the N/A state,
        // never a spurious $0.00.
        let mut app = App::new();
        app.metrics.update(MetricsSnapshot {
            gpu_monitor: Some(GpuPowerMonitor::default().with_rate(0.16)),
            ..Default::default()
        });
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("Awaiting token"), "N/A cost state: {text}");
    }

    #[test]
    fn view4_table_name_column_fixed_width() {
        // The Name column is a fixed 24 chars: short names are padded,
        // long names are truncated with "…". All subsequent columns
        // align perfectly.
        let mut mon = sample_monitor();
        mon.gpu_names = vec!["Radeon RX 9700".into(), "NVIDIA A4000".into()];
        let mut app = app_with_monitor(mon);
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        // Both names fit in 24 chars → shown in full.
        assert!(text.contains("Radeon RX 9700"), "full AMD name: {text}");
        assert!(text.contains("NVIDIA A4000"), "full NVIDIA name: {text}");
    }

    #[test]
    fn view4_table_truncates_names_over_24_chars() {
        // A name longer than 24 chars is truncated with "…".
        let mut mon = sample_monitor();
        mon.gpu_names = vec!["NVIDIA GeForce RTX 4090 Titan X Ultra".into()];
        mon.gpus = vec![mon.gpus[0].clone()];
        let mut app = app_with_monitor(mon);
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        // The truncated name should appear (with the ellipsis).
        assert!(text.contains('…'), "truncation ellipsis: {text}");
    }

    #[test]
    fn format_rate_and_cost_adapt_precision() {
        // Sub-dollar rates keep 3 decimals (the local-vs-cloud delta).
        assert_eq!(format_rate(0.031), "0.031");
        assert_eq!(format_rate(0.0), "0.000");
        // Dollar-and-up rates use 2.
        assert_eq!(format_rate(2.5), "2.50");
        // Tiny run totals keep 4; normal totals 2; non-positive is N/A.
        assert_eq!(format_cost(0.0011), "$0.0011");
        assert_eq!(format_cost(0.0), "N/A");
        assert_eq!(format_cost(12.345), "$12.35");
    }

    #[test]
    fn view4_table_shows_run_averages_and_peak() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("91%"), "avg util 91%: {text}");
        assert!(text.contains("88%"), "avg util 88%: {text}");
        assert!(text.contains("64°C"), "avg temp 64°C: {text}");
        assert!(text.contains("62°C"), "avg temp 62°C: {text}");
        // The legend explains the run-average columns.
        assert!(text.contains("run average"), "legend: {text}");
    }

    #[test]
    fn view4_renders_efficiency_and_graphs() {
        let mut app = app_with_monitor(sample_monitor());
        app.view = crate::ui::app::View::Gpu;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("EFFICIENCY"), "efficiency panel: {text}");
        assert!(text.contains("POWER OVER TIME"), "power graph: {text}");
        assert!(text.contains("UTILIZATION OVER TIME"), "util graph: {text}");
        assert!(text.contains("TEMPERATURE OVER TIME"), "temp graph: {text}");
        // The line charts carry their avg / peak tags and the scaled axis.
        assert!(text.contains("avg:"), "mean tag: {text}");
        assert!(text.contains("peak:"), "peak tag: {text}");
    }

    #[test]
    fn view4_no_gpu_shows_the_placeholder() {
        // A monitor is present but empty (no GPUs) → the per-GPU table shows
        // its "awaiting samples" state, and the panel still renders.
        let mut app = App::new();
        app.view = crate::ui::app::View::Gpu;
        // No gpu_monitor set → the "no GPU telemetry" placeholder.
        let text = render_text(&app, 120, 40);
        assert!(text.contains("No GPU telemetry"), "placeholder: {text}");
    }

    // ── auto-scaled line chart ─────────────────────────────────────────────

    #[test]
    fn auto_scale_axis_pads_a_narrow_band() {
        let vals: Vec<f64> = (0..12).map(|i| 85.0 + i as f64).collect(); // 85…96
        let (min, max) = auto_scale_axis(&vals, 0.1);
        // range 11 → pad 1.1: (83.9, 97.1)
        assert!((min - 83.9).abs() < 1e-9, "min: {min}");
        assert!((max - 97.1).abs() < 1e-9, "max: {max}");
    }

    #[test]
    fn auto_scale_axis_flat_signal_gets_one_unit_of_padding() {
        // A perfectly flat signal (the "flat line" case) still gets a
        // visible 2-unit-tall axis.
        let (min, max) = auto_scale_axis(&[90.0, 90.0, 90.0], 0.1);
        assert!((min - 89.0).abs() < 1e-9);
        assert!((max - 91.0).abs() < 1e-9);
    }

    #[test]
    fn auto_scale_axis_empty_falls_back_to_a_unit_range() {
        assert_eq!(auto_scale_axis(&[], 0.1), (0.0, 1.0));
    }

    #[test]
    fn build_line_chart_empty_shows_awaiting() {
        let lines = build_line_chart(&[], 40, 10, "%", Theme::Cyberpunk);
        assert!(lines[0].to_string().contains("Awaiting"));
    }

    #[test]
    fn build_line_chart_shows_scaled_axis_mean_and_peak() {
        // A narrow band (85 → 96): the axis must span *that* band (with
        // padding) — not 0–100 — and the mean / peak tags must appear.
        let vals: Vec<f64> = (0..12).map(|i| 85.0 + i as f64).collect();
        let lines = build_line_chart(&vals, 60, 12, "%", Theme::Cyberpunk);
        let joined: String = lines.iter().map(|l| l.to_string()).collect();
        // The scaled y-axis labels — 15% padding on the 85…96 band gives
        // (83.35, 97.65) → "98%" / "90%" / "83%": the axis spans the
        // data's real band, never 0–100.
        assert!(joined.contains("98%"), "scaled max label: {joined}");
        assert!(joined.contains("90%"), "scaled mid label: {joined}");
        assert!(joined.contains("83%"), "scaled min label: {joined}");
        // The mean + peak tags.
        assert!(joined.contains("avg: 90.5%"), "mean tag: {joined}");
        assert!(joined.contains("peak: 96.0%"), "peak tag: {joined}");
        // The curve (primary line + floor area fill) and the peak marker.
        assert!(joined.contains('▓'), "line present: {joined}");
        assert!(joined.contains('░'), "area fill present: {joined}");
        assert!(joined.contains('▲'), "peak marker present: {joined}");
    }

    #[test]
    fn build_line_chart_labels_the_time_axis_in_minutes() {
        // 15 minutes of 1 Hz samples: the x-axis steps to whole minutes
        // (0s / 5m / 10m / 15m — ≤ 5 labels).
        let vals: Vec<f64> = (0..901).map(|i| 90.0 + (i % 7) as f64).collect();
        let lines = build_line_chart(&vals, 60, 12, "%", Theme::Cyberpunk);
        let joined: String = lines.iter().map(|l| l.to_string()).collect();
        assert!(joined.contains("0s"), "start label: {joined}");
        assert!(joined.contains("5m"), "minute label: {joined}");
        assert!(joined.contains("10m"), "minute label: {joined}");
        assert!(joined.contains("15m"), "end label: {joined}");
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
