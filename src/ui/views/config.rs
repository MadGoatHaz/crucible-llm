//! View 5 — Configuration (blueprint §6 / plan Chunk 18): target endpoint,
//! model, mode, tokens, concurrency ladder, tokenizer path, feature and
//! engine toggles.
//!
//! Read-only placeholder display; interactive editing + persistence lands
//! in Chunk 18.

use ratatui::layout::Rect;
use ratatui::style::Style;
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph};
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::style;

/// Render the Config view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let m = &app.metrics;
    let kv = |k: &str, v: String, vs: Style| -> Line<'static> {
        Line::from(vec![
            Span::styled(format!("{k:<28} "), style::label()),
            Span::styled(v, vs),
        ])
    };

    let on = style::value_ok();
    let off = style::footer();
    let lines = Text::from(vec![
        kv("Target URL", m.endpoint.to_string(), style::value()),
        kv("Model", m.model.to_string(), style::value()),
        kv("Mode", m.mode.to_string(), style::value()),
        kv(
            "Concurrency target",
            format!("{} streams", app.concurrency_target),
            style::highlight(),
        ),
        kv("Concurrency ladder", "1 → 2 → 4 → 8 → 16 → 32 → 64".to_string(), style::label()),
        kv("Prompt mode", "long".to_string(), style::label()),
        kv("Target tokens", "2048".to_string(), style::label()),
        kv("Iterations", "4".to_string(), style::label()),
        kv("Tokenizer", "(not set — chars/4 fallback, estimated)".to_string(), style::value_warn()),
        kv("NVML telemetry", "off (feature-gated, default)".to_string(), off),
        kv("Cache bypass (--nocache)", "off".to_string(), off),
        Line::from(Span::raw("")),
        kv("Engine A — Speed & Latency", "on".to_string(), on),
        kv("Engine B — Concurrency", "on".to_string(), on),
        kv("Engine C — Capability", "off".to_string(), off),
        kv("Engine D — Hardware & Energy", "off".to_string(), off),
        Line::from(Span::raw("")),
        Line::from(Span::styled(
            "Read-only display — interactive editing + persistence lands in Chunk 18.",
            style::footer(),
        )),
    ]);

    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title("CONFIGURATION"),
        ),
        area,
    );
}
