//! The **cyberpunk** palette + shared style constructors for the TUI.
//!
//! Neon-on-dark: a cyan primary, electric-purple + deep-blue secondaries,
//! a hot-pink danger color, and a mint success color, all floating on a dark
//! background. Every view draws from `palette` + `style` + the `block` /
//! `panel_title` / `glyph` helpers so the five views (and the Setup
//! takeover) stay visually cohesive: Blade-Runner-meets-terminal, Tron-meets
//! data-viz.
//!
//! **Palette roles**
//! * `ACCENT`   (cyan `#00FFFF`)     — primary: titles, active borders,
//!   selected items, the "now" glow.
//! * `SECONDARY`(electric `#BF00FF`) — highlights, selected rows, warnings.
//! * `DATA`     (deep blue `#0066FF`)— data values, graph bodies, `ℹ` notes.
//! * `CALLOUT`  (pink `#FF0066`)     — knee markers, important callouts.
//! * `OK`       (mint `#00FFAA`)     — passing, complete, good.
//! * `ERR`      (hot pink `#FF0044`) — errors, failures, critical.
//! * `MUTED`    (blue-gray `#4488AA`)- secondary info, labels, chrome.
//! * `TEXT`     (white `#CCFFFF`)    — primary data values (the bright top).
//! * `BORDER_DEFAULT` / `BORDER_ACTIVE` — the dim / bright cyan panel frames.
//! * `FLOOR` / `BRIGHT` — the dim-blue base and hot-cyan top of a gradient.
//!
//! The old semantic names (`ACCENT`/`OK`/`WARN`/`ERR`/`INFO`/`HIGHLIGHT`/
//! `MUTED`/`TEXT`) are kept so every existing call site keeps compiling;
//! only their RGB values (and a few new constants) change.

use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

/// Canonical cyberpunk palette (all `Color::Rgb` — 256-color-safe neon).
pub mod palette {
    use ratatui::style::Color;

    // ── Primary / secondary / tertiary ────────────────────────────────
    /// Primary (cyan `#00FFFF`): titles, active borders, selected, the glow.
    pub const ACCENT: Color = Color::Rgb(0, 255, 255);
    /// Secondary (electric purple `#BF00FF`): highlights, selected rows.
    pub const SECONDARY: Color = Color::Rgb(191, 0, 255);
    /// Tertiary (deep blue `#0066FF`): data values, graph bodies, `ℹ` notes.
    pub const DATA: Color = Color::Rgb(0, 102, 255);
    /// Accent callout (magenta/pink `#FF0066`): knee markers, important text.
    pub const CALLOUT: Color = Color::Rgb(255, 0, 102);

    // ── Semantic (kept names; cyberpunk RGB values) ────────────────────
    /// Success (teal/mint `#00FFAA`): passing, complete, good.
    pub const OK: Color = Color::Rgb(0, 255, 170);
    /// Warning (electric purple `#BF00FF`): degraded, cautions.
    pub const WARN: Color = Color::Rgb(191, 0, 255);
    /// Danger (hot pink `#FF0044`): failures, errors, critical.
    pub const ERR: Color = Color::Rgb(255, 0, 68);
    /// Info (deep blue `#0066FF`): descriptions, `ℹ` help text.
    pub const INFO: Color = Color::Rgb(0, 102, 255);
    /// Highlight (magenta/pink `#FF0066`): MTP values, active tab, callouts.
    pub const HIGHLIGHT: Color = Color::Rgb(255, 0, 102);
    /// Muted chrome (dim blue-gray `#4488AA`): labels, secondary hints.
    pub const MUTED: Color = Color::Rgb(68, 136, 170);
    /// Bright body text (cyan-tinted white `#CCFFFF`): primary data values.
    pub const TEXT: Color = Color::Rgb(204, 255, 255);

    // ── Borders ────────────────────────────────────────────────────────
    /// Default panel border (dark cyan `#004455`).
    pub const BORDER_DEFAULT: Color = Color::Rgb(0, 68, 85);
    /// Active / focused panel border (bright cyan `#00CCCC`).
    pub const BORDER_ACTIVE: Color = Color::Rgb(0, 204, 204);

    // ── Gradient layers (the "glowing from within" texture) ────────────
    /// Dim blue floor (the base of a gradient bar / area fill).
    pub const FLOOR: Color = Color::Rgb(16, 52, 74);
    /// Hot cyan (the bright body just under the white cap).
    pub const BRIGHT: Color = Color::Rgb(140, 255, 255);
    /// Near-black background (the dark base the neon floats on).
    pub const BG: Color = Color::Rgb(5, 10, 18);
}

/// The "digital" glyph set — the layered textures that give the graphs
/// their cyberpunk depth (flat single-color bars are gone).
pub mod glyph {
    // Gradient layers, bottom → top (the 3-layer "glow from within").
    /// Light shade — the dim floor / area fill.
    pub const FLOOR: char = '░';
    /// Medium shade — the low band of a gradient.
    pub const LOW: char = '▒';
    /// Dense shade — the mid (cyan) band.
    pub const MID: char = '▓';
    /// Full block — the bright top.
    pub const HIGH: char = '█';
    // Segmented capability / progress bars.
    /// Filled segment.
    pub const SEG_ON: char = '▰';
    /// Empty segment.
    pub const SEG_OFF: char = '▱';
    // Curve markers (angular, cyberpunk).
    /// A measured data point (diamond).
    pub const POINT: char = '◆';
    /// The saturation knee (triangle).
    pub const KNEE: char = '▲';
    // Status indicators.
    /// Running (the alive pulse).
    pub const RUN: char = '◉';
    /// Complete / passing.
    pub const DONE: char = '✓';
    /// Warning.
    pub const WARN: char = '⚡';
    /// Error / failure.
    pub const ERR: char = '✕';
    /// Idle.
    pub const IDLE: char = '○';
    // Panel-title / active-view prefix.
    /// The `▸` caret that leads every panel title and the active tab.
    pub const PREFIX: char = '▸';
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

/// A styled panel title line: a `▸` caret prefix, bold + accent, uppercased
/// (the "panel titles: `▸` + bold + accent, uppercase" rule).
pub fn panel_title(text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        format!("{} {}", glyph::PREFIX, text.into().to_ascii_uppercase()),
        style::title(),
    ))
}

/// A border that **pulses** between two cyan shades on successive frame
/// groups — the "alive" signal for a running / focused panel. `tick` is the
/// 60 Hz render counter; every ~10 frames the shade flips.
pub fn pulsing_border(tick: u64) -> Style {
    let on = (tick / 10).is_multiple_of(2);
    Style::default().fg(if on {
        palette::BORDER_ACTIVE
    } else {
        palette::ACCENT
    })
}

/// Shared style constructors.
pub mod style {
    use ratatui::style::{Modifier, Style};

    use super::palette;

    /// Panel border (inactive chrome) — dark cyan.
    pub fn border() -> Style {
        Style::default().fg(palette::BORDER_DEFAULT)
    }

    /// Panel border for the focused / active panel — bright cyan.
    pub fn active_border() -> Style {
        Style::default().fg(palette::BORDER_ACTIVE)
    }

    /// Panel title (e.g. "LIVE THROUGHPUT") — bright cyan, bold.
    pub fn title() -> Style {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    /// Muted title / header row — dim blue-gray, bold.
    pub fn muted_title() -> Style {
        Style::default()
            .fg(palette::MUTED)
            .add_modifier(Modifier::BOLD)
    }

    /// Field label — dim blue-gray (secondary info).
    pub fn label() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Primary value (cyan-tinted white, bold) — the headline number.
    pub fn value() -> Style {
        Style::default()
            .fg(palette::TEXT)
            .add_modifier(Modifier::BOLD)
    }

    /// Secondary value (cyan-tinted white, dimmed) — the avg / p5
    /// companions that support a primary value without competing with it.
    pub fn value_secondary() -> Style {
        Style::default()
            .fg(palette::TEXT)
            .add_modifier(Modifier::DIM)
    }

    /// Healthy value (teal/mint, bold).
    pub fn value_ok() -> Style {
        Style::default()
            .fg(palette::OK)
            .add_modifier(Modifier::BOLD)
    }

    /// Warning value (electric purple, bold).
    pub fn value_warn() -> Style {
        Style::default()
            .fg(palette::WARN)
            .add_modifier(Modifier::BOLD)
    }

    /// Error / regression value (hot pink, bold).
    pub fn value_err() -> Style {
        Style::default()
            .fg(palette::ERR)
            .add_modifier(Modifier::BOLD)
    }

    /// MTP / speculative-decoding / callout value (magenta/pink, bold).
    pub fn highlight() -> Style {
        Style::default()
            .fg(palette::HIGHLIGHT)
            .add_modifier(Modifier::BOLD)
    }

    /// Top status bar (endpoint / model / mode) — cyan.
    pub fn status_bar() -> Style {
        Style::default().fg(palette::ACCENT)
    }

    /// Bottom key-hint footer — dim blue-gray.
    pub fn footer() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Informational text (the `ℹ` notes) — deep blue, dimmed, so it reads
    /// as "help" and never competes with the bright primary data.
    pub fn info() -> Style {
        Style::default()
            .fg(palette::INFO)
            .add_modifier(Modifier::DIM)
    }

    /// Active tab in the tab bar — bright cyan, bold (the current view).
    pub fn tab_active() -> Style {
        Style::default()
            .fg(palette::ACCENT)
            .add_modifier(Modifier::BOLD)
    }

    /// Inactive tab in the tab bar — dim blue-gray.
    pub fn tab_inactive() -> Style {
        Style::default().fg(palette::MUTED)
    }

    /// Separator between tabs / status-bar fields — dim cyan.
    pub fn tab_separator() -> Style {
        Style::default().fg(palette::BORDER_DEFAULT)
    }
}
