//! ANSI color palette / styles for the dashboard (green/yellow/red/cyan/
//! magenta per the blueprint §6 mockups).
//!
//! Every view draws from `palette` + `style` so all five views stay
//! visually consistent.

/// Canonical palette.
pub mod palette {
    use ratatui::style::Color;

    /// Info / accent: titles, active borders, primary values.
    pub const ACCENT: Color = Color::Cyan;
    /// Healthy / success / improvement (green NIAH cells, gains).
    pub const OK: Color = Color::Green;
    /// Warning / throttled (yellow NIAH cells, approaching limits).
    pub const WARN: Color = Color::Yellow;
    /// Error / regression / failure (red NIAH cells, regressions).
    pub const ERR: Color = Color::Red;
    /// Highlight: MTP / speculative-decoding values, active tab.
    pub const HIGHLIGHT: Color = Color::Magenta;
    /// Muted chrome: borders, inactive hints.
    pub const MUTED: Color = Color::DarkGray;
    /// Body text.
    pub const TEXT: Color = Color::White;
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

    /// Primary value.
    pub fn value() -> Style {
        Style::default()
            .fg(palette::TEXT)
            .add_modifier(Modifier::BOLD)
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

    /// Dimmed informational text (the `ℹ` notes that explain what a panel's
    /// numbers mean — muted + dim so they never compete with the data).
    pub fn info() -> Style {
        Style::default()
            .fg(palette::MUTED)
            .add_modifier(Modifier::DIM)
    }

    /// Active tab in the tab bar.
    pub fn tab_active() -> Style {
        Style::default()
            .fg(palette::HIGHLIGHT)
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
