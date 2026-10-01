//! Cohesive color palette + shared style constructors for the TUI.
//!
//! Every view draws from `palette` + `style` + the `block` / `panel_title`
//! helpers so all five views (and the Setup takeover) stay visually
//! consistent: one accent (cyan), a semantic success/warning/danger set,
//! blue for informational text, dark-gray for chrome, bright white for the
//! primary data, and rounded borders on every panel.
//!
//! **Palette roles**
//! * `ACCENT`  (cyan)   — active elements, panel titles, selected items.
//! * `OK`      (green)  — passing tests, complete states, gains.
//! * `WARN`    (yellow) — degraded performance, cautions.
//! * `ERR`     (red)    — failures, errors, critical warnings.
//! * `INFO`    (blue)   — descriptions, help text (the `ℹ` notes).
//! * `MUTED`   (dark)   — secondary text, labels, chrome, inactive hints.
//! * `TEXT`    (white)  — primary data values.
//! * `HIGHLIGHT`(magenta)— MTP / speculative-decoding values, active tab.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

/// Canonical palette.
pub mod palette {
    use ratatui::style::Color;

    /// Accent (cyan): titles, active borders, selected items.
    pub const ACCENT: Color = Color::Cyan;
    /// Success (green): passing tests, complete states, gains.
    pub const OK: Color = Color::Green;
    /// Warning (yellow): degraded performance, cautions.
    pub const WARN: Color = Color::Yellow;
    /// Danger (red): failures, errors, critical warnings.
    pub const ERR: Color = Color::Red;
    /// Info (blue): descriptions, help text (the `ℹ` notes).
    pub const INFO: Color = Color::Blue;
    /// Highlight (magenta): MTP / speculative-decoding values, active tab.
    pub const HIGHLIGHT: Color = Color::Magenta;
    /// Muted chrome (dark gray): borders, inactive hints, secondary labels.
    pub const MUTED: Color = Color::DarkGray;
    /// Bright body text (white): primary data values.
    pub const TEXT: Color = Color::White;
}

/// A bordered panel: **rounded** corners, the given border style, and title.
///
/// Every view builds its panels through this so the border treatment is
/// uniform (the "all panels use `BorderType::Rounded`" rule).
pub fn block(title: impl Into<Line<'static>>, border_style: Style) -> Block<'static> {
    Block::default()
        .borders(Borders::ALL)
        .border_type(BorderType::Rounded)
        .border_style(border_style)
        .title(title)
}

/// A styled panel title line: bold + accent, uppercased (the "panel titles:
/// bold + accent, uppercase" rule).
pub fn panel_title(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        text.into().to_ascii_uppercase(),
        style::title(),
    ))
}

/// Shared style constructors.
pub mod style {
    use ratatui::style::{Modifier, Style};

    use super::palette;

    /// Panel border (inactive chrome).
    pub fn border() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Panel border for the focused / active panel.
    pub fn active_border() -> Style {
        Style::default().fg(palette::ACCENT)
    }

    /// Panel title (e.g. "TELEMETRY GAUGES").
    pub fn title() -> Style {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    /// Muted title / header row.
    pub fn muted_title() -> Style {
        Style::default()
            .fg(palette::MUTED)
            .add_modifier(Modifier::BOLD)
    }

    /// Field label.
    pub fn label() -> Style {
        Style::default().fg(palette::TEXT)
    }

    /// Primary value (bright white, bold) — the headline number.
    pub fn value() -> Style {
        Style::default()
            .fg(palette::TEXT)
            .add_modifier(Modifier::BOLD)
    }

    /// Secondary value (bright white, normal weight, slightly dimmed) —
    /// the avg / p5 companions that support a primary value without
    /// competing with it.
    pub fn value_secondary() -> Style {
        Style::default()
            .fg(palette::TEXT)
            .add_modifier(Modifier::DIM)
    }

    /// Healthy value (green).
    pub fn value_ok() -> Style {
        Style::default()
            .fg(palette::OK)
            .add_modifier(Modifier::BOLD)
    }

    /// Warning value (yellow).
    pub fn value_warn() -> Style {
        Style::default()
            .fg(palette::WARN)
            .add_modifier(Modifier::BOLD)
    }

    /// Error / regression value (red).
    pub fn value_err() -> Style {
        Style::default()
            .fg(palette::ERR)
            .add_modifier(Modifier::BOLD)
    }

    /// MTP / speculative-decoding multiplier (magenta).
    pub fn highlight() -> Style {
        Style::default()
            .fg(palette::HIGHLIGHT)
            .add_modifier(Modifier::BOLD)
    }

    /// Top status bar (endpoint / model / mode).
    pub fn status_bar() -> Style {
        Style::default().fg(palette::ACCENT)
    }

    /// Bottom key-hint footer (blueprint §6).
    pub fn footer() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Informational text (the `ℹ` notes that explain what a panel's numbers
    /// mean) — **blue**, dimmed so it reads as "help" and never competes
    /// with the bright primary data.
    pub fn info() -> Style {
        Style::default()
            .fg(palette::INFO)
            .add_modifier(Modifier::DIM)
    }

    /// Active tab in the tab bar (accent + bold — the current view).
    pub fn tab_active() -> Style {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    /// Inactive tab in the tab bar.
    pub fn tab_inactive() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Separator between tabs / status-bar fields.
    pub fn tab_separator() -> Style {
        Style::default().fg(palette::MUTED)
    }
}
