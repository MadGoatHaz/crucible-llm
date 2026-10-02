//! The TUI theme system: a `Theme` enum with three palettes
//! (Cyberpunk, Vampire, Monochrome Pastel) plus the shared,
//! theme-aware style constructors every view draws from.
//!
//! **Design** — the render path is a pure `&App` read
//! (measurement-isolation invariant, blueprint §4). Every color a view
//! needs comes from the active theme: `let th = app.active_theme;` and
//! then `th.primary()` / `style::title(th)` / `theme::block(...,
//! style::border(th))`. No view hardcodes a `Color::Rgb` — the whole
//! screen re-skins when the user switches themes (first-run picker or
//! the Config view's Theme field).
//!
//! **Theme roles** (each theme supplies all of them):
//! * `primary`        — titles, active borders, selected items, the glow.
//! * `secondary`      — highlights, selected rows, warnings.
//! * `tertiary`       — data values, graph bodies, `ℹ` notes.
//! * `accent`         — knee markers, important callouts.
//! * `success`        — passing, complete, good.
//! * `danger`         — errors, failures, critical.
//! * `dim`            — secondary info, labels, chrome.
//! * `bright`         — primary data values (the bright top).
//! * `border` / `border_active` — the dim / bright panel frames.
//! * plus the gradient texture colors (`floor`, `bright_gradient`, `bg`)
//!   that give the charts their depth.
//!
//! `Cyberpunk` is the default and preserves the original neon palette
//! exactly, so switching to it is a no-op visual change.

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Borders};

/// The available color themes. `Cyberpunk` (the default) keeps the original
/// neon palette; `Vampire` is a dark-red gothic skin; `Monochrome` is a
/// clean pastel skin. Each is a complete palette — switching re-colors the
/// entire TUI.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Theme {
    /// Neon cyan / electric purple / deep blue on dark (the original skin).
    #[default]
    Cyberpunk,
    /// Crimson / gold / dark purple — dark, gothic.
    Vampire,
    /// Soft blue / lavender / clean pastels.
    Monochrome,
}

impl Theme {
    /// All themes in picker / cycle order.
    pub const ALL: [Theme; 3] = [Theme::Cyberpunk, Theme::Vampire, Theme::Monochrome];

    /// The full theme name (shown in the picker + Config view).
    pub fn name(self) -> &'static str {
        match self {
            Theme::Cyberpunk => "Cyberpunk",
            Theme::Vampire => "Vampire",
            Theme::Monochrome => "Monochrome Pastel",
        }
    }

    /// The short config-file value (`"cyberpunk"` / `"vampire"` /
    /// `"monochrome"`).
    pub fn id(self) -> &'static str {
        match self {
            Theme::Cyberpunk => "cyberpunk",
            Theme::Vampire => "vampire",
            Theme::Monochrome => "monochrome",
        }
    }

    /// The one-line palette blurb shown under the name in the picker.
    pub fn tagline(self) -> &'static str {
        match self {
            Theme::Cyberpunk => "Neon cyan \u{00b7} Electric purple \u{00b7} Digital glow",
            Theme::Vampire => "Crimson \u{00b7} Gold \u{00b7} Dark purple \u{00b7} Gothic",
            Theme::Monochrome => "Soft blue \u{00b7} Lavender \u{00b7} Clean \u{00b7} Minimal",
        }
    }

    /// Parse a config-file / env value into a theme (`None` on unknown).
    pub fn from_id(s: &str) -> Option<Theme> {
        match s.to_ascii_lowercase().as_str() {
            "cyberpunk" => Some(Theme::Cyberpunk),
            "vampire" => Some(Theme::Vampire),
            "monochrome" => Some(Theme::Monochrome),
            _ => None,
        }
    }

    // ── Palette roles ────────────────────────────────────────────────────

    /// Primary: titles, active borders, selected items, the glow.
    pub fn primary(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(0, 255, 255),
            Theme::Vampire => Color::Rgb(220, 20, 60),
            Theme::Monochrome => Color::Rgb(126, 182, 232),
        }
    }

    /// Secondary: highlights, selected rows.
    pub fn secondary(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(191, 0, 255),
            Theme::Vampire => Color::Rgb(75, 0, 130),
            Theme::Monochrome => Color::Rgb(200, 162, 200),
        }
    }

    /// Tertiary: data values, graph bodies, `ℹ` notes.
    pub fn tertiary(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(0, 102, 255),
            Theme::Vampire => Color::Rgb(90, 50, 130),
            Theme::Monochrome => Color::Rgb(160, 185, 215),
        }
    }

    /// Accent callout: knee markers, important text.
    pub fn accent(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(255, 0, 102),
            Theme::Vampire => Color::Rgb(255, 215, 0),
            Theme::Monochrome => Color::Rgb(255, 218, 185),
        }
    }

    /// Success: passing, complete, good.
    pub fn success(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(0, 255, 170),
            Theme::Vampire => Color::Rgb(144, 238, 144),
            Theme::Monochrome => Color::Rgb(184, 230, 208),
        }
    }

    /// Danger: failures, errors, critical.
    pub fn danger(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(255, 0, 68),
            Theme::Vampire => Color::Rgb(255, 0, 0),
            Theme::Monochrome => Color::Rgb(240, 128, 128),
        }
    }

    /// Dim: secondary info, labels, chrome.
    pub fn dim(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(68, 136, 170),
            Theme::Vampire => Color::Rgb(92, 48, 48),
            Theme::Monochrome => Color::Rgb(153, 153, 153),
        }
    }

    /// Bright: primary data values (the bright top).
    pub fn bright(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(204, 255, 255),
            Theme::Vampire => Color::Rgb(255, 238, 236),
            Theme::Monochrome => Color::Rgb(255, 255, 255),
        }
    }

    /// Default panel border (inactive chrome).
    pub fn border(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(0, 68, 85),
            Theme::Vampire => Color::Rgb(74, 0, 0),
            Theme::Monochrome => Color::Rgb(204, 204, 204),
        }
    }

    /// Active / focused panel border.
    pub fn border_active(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(0, 204, 204),
            Theme::Vampire => Color::Rgb(220, 20, 60),
            Theme::Monochrome => Color::Rgb(126, 182, 232),
        }
    }

    // ── Extra palette roles (kept from the original cyberpunk set) ──────

    /// Warning color (degraded, cautions) — mirrors `secondary`.
    pub fn warn(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(191, 0, 255),
            Theme::Vampire => Color::Rgb(255, 215, 0),
            Theme::Monochrome => Color::Rgb(255, 218, 185),
        }
    }

    /// Info color (`ℹ` notes) — mirrors `tertiary`.
    pub fn info(self) -> Color {
        self.tertiary()
    }

    /// Highlight / callout color (MTP values, active tab) — mirrors `accent`.
    pub fn highlight(self) -> Color {
        self.accent()
    }

    /// Dim floor: the base of a gradient bar / area fill.
    pub fn floor(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(16, 52, 74),
            Theme::Vampire => Color::Rgb(40, 12, 14),
            Theme::Monochrome => Color::Rgb(55, 65, 85),
        }
    }

    /// Hot top of a gradient bar (just under the bright cap).
    pub fn bright_gradient(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(140, 255, 255),
            Theme::Vampire => Color::Rgb(255, 170, 170),
            Theme::Monochrome => Color::Rgb(220, 230, 245),
        }
    }

    /// Panel background (the base a bright theme floats on — kept dark for
    /// every theme so bright/white text stays readable on a dark terminal).
    pub fn bg(self) -> Color {
        match self {
            Theme::Cyberpunk => Color::Rgb(5, 10, 18),
            Theme::Vampire => Color::Rgb(15, 5, 7),
            Theme::Monochrome => Color::Rgb(16, 18, 24),
        }
    }
}

/// The "digital" glyph set — the layered textures that give the graphs
/// their depth (flat single-color bars are gone).
pub mod glyph {
    // Gradient layers, bottom → top (the 3-layer "glow from within").
    /// Light shade — the dim floor / area fill.
    pub const FLOOR: char = '░';
    /// Medium shade — the low band of a gradient.
    pub const LOW: char = '▒';
    /// Dense shade — the mid (primary) band.
    pub const MID: char = '▓';
    /// Full block — the bright top.
    pub const HIGH: char = '█';
    // Segmented capability / progress bars.
    /// Filled segment.
    pub const SEG_ON: char = '▰';
    /// Empty segment.
    pub const SEG_OFF: char = '▱';
    // Curve markers (angular).
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

/// A styled panel title line: a `▸` caret prefix, bold + primary, uppercased
/// (the "panel titles: `▸` + bold + primary, uppercase" rule).
pub fn panel_title(th: Theme, text: impl Into<String>) -> Line<'static> {
    Line::from(Span::styled(
        format!("{} {}", glyph::PREFIX, text.into().to_ascii_uppercase()),
        style::title(th),
    ))
}

/// A border that **pulses** between two shades on successive frame groups —
/// the "alive" signal for a running / focused panel. `tick` is the 60 Hz
/// render counter; every ~10 frames the shade flips.
pub fn pulsing_border(th: Theme, tick: u64) -> Style {
    let on = (tick / 10).is_multiple_of(2);
    Style::default().fg(if on {
        th.border_active()
    } else {
        th.primary()
    })
}

/// Shared style constructors — every one takes the active [`Theme`] so the
/// whole TUI re-skins from a single source of truth.
pub mod style {
    use ratatui::style::{Modifier, Style};

    use super::Theme;

    /// Panel border (inactive chrome).
    pub fn border(th: Theme) -> Style {
        Style::default().fg(th.border())
    }

    /// Panel border for the focused / active panel.
    pub fn active_border(th: Theme) -> Style {
        Style::default().fg(th.border_active())
    }

    /// Panel title — primary, bold.
    pub fn title(th: Theme) -> Style {
        Style::default()
            .fg(th.primary())
            .add_modifier(Modifier::BOLD)
    }

    /// Muted title / header row — dim, bold.
    pub fn muted_title(th: Theme) -> Style {
        Style::default()
            .fg(th.dim())
            .add_modifier(Modifier::BOLD)
    }

    /// Field label — dim (secondary info).
    pub fn label(th: Theme) -> Style {
        Style::default().fg(th.dim())
    }

    /// Primary value (bright, bold) — the headline number.
    pub fn value(th: Theme) -> Style {
        Style::default()
            .fg(th.bright())
            .add_modifier(Modifier::BOLD)
    }

    /// Secondary value (bright, dimmed) — the avg / p5 companions.
    pub fn value_secondary(th: Theme) -> Style {
        Style::default()
            .fg(th.bright())
            .add_modifier(Modifier::DIM)
    }

    /// Healthy value (success, bold).
    pub fn value_ok(th: Theme) -> Style {
        Style::default()
            .fg(th.success())
            .add_modifier(Modifier::BOLD)
    }

    /// Warning value (warn, bold).
    pub fn value_warn(th: Theme) -> Style {
        Style::default()
            .fg(th.warn())
            .add_modifier(Modifier::BOLD)
    }

    /// Error / regression value (danger, bold).
    pub fn value_err(th: Theme) -> Style {
        Style::default()
            .fg(th.danger())
            .add_modifier(Modifier::BOLD)
    }

    /// Callout / highlight value (accent, bold).
    pub fn highlight(th: Theme) -> Style {
        Style::default()
            .fg(th.highlight())
            .add_modifier(Modifier::BOLD)
    }

    /// Top status bar (endpoint / model / mode).
    pub fn status_bar(th: Theme) -> Style {
        Style::default().fg(th.primary())
    }

    /// Bottom key-hint footer — dim.
    pub fn footer(th: Theme) -> Style {
        Style::default().fg(th.dim())
    }

    /// Informational text (the `ℹ` notes) — info color, dimmed.
    pub fn info(th: Theme) -> Style {
        Style::default()
            .fg(th.info())
            .add_modifier(Modifier::DIM)
    }

    /// Active tab in the tab bar — primary, bold.
    pub fn tab_active(th: Theme) -> Style {
        Style::default()
            .fg(th.primary())
            .add_modifier(Modifier::BOLD)
    }

    /// Inactive tab in the tab bar — dim.
    pub fn tab_inactive(th: Theme) -> Style {
        Style::default().fg(th.dim())
    }

    /// Separator between tabs / status-bar fields.
    pub fn tab_separator(th: Theme) -> Style {
        Style::default().fg(th.border())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_id_parses_all_three_themes() {
        assert_eq!(Theme::from_id("cyberpunk"), Some(Theme::Cyberpunk));
        assert_eq!(Theme::from_id("vampire"), Some(Theme::Vampire));
        assert_eq!(Theme::from_id("monochrome"), Some(Theme::Monochrome));
        // Case-insensitive.
        assert_eq!(Theme::from_id("VAMPIRE"), Some(Theme::Vampire));
        assert_eq!(Theme::from_id("nonsense"), None);
    }

    #[test]
    fn id_and_name_round_trip() {
        for t in Theme::ALL {
            assert_eq!(Theme::from_id(t.id()), Some(t));
            assert!(!t.name().is_empty());
            assert!(!t.tagline().is_empty());
        }
    }

    #[test]
    fn cyberpunk_preserves_the_original_palette() {
        let c = Theme::Cyberpunk;
        assert_eq!(c.primary(), Color::Rgb(0, 255, 255));
        assert_eq!(c.secondary(), Color::Rgb(191, 0, 255));
        assert_eq!(c.accent(), Color::Rgb(255, 0, 102));
        assert_eq!(c.success(), Color::Rgb(0, 255, 170));
        assert_eq!(c.danger(), Color::Rgb(255, 0, 68));
        assert_eq!(c.dim(), Color::Rgb(68, 136, 170));
        assert_eq!(c.bright(), Color::Rgb(204, 255, 255));
        assert_eq!(c.border(), Color::Rgb(0, 68, 85));
        assert_eq!(c.border_active(), Color::Rgb(0, 204, 204));
    }

    #[test]
    fn each_theme_is_a_complete_distinct_palette() {
        for t in Theme::ALL {
            // Every role returns a valid (non-Reset) color.
            assert_ne!(t.primary(), Color::Reset);
            assert_ne!(t.secondary(), Color::Reset);
            assert_ne!(t.tertiary(), Color::Reset);
            assert_ne!(t.accent(), Color::Reset);
            assert_ne!(t.success(), Color::Reset);
            assert_ne!(t.danger(), Color::Reset);
            assert_ne!(t.dim(), Color::Reset);
            assert_ne!(t.bright(), Color::Reset);
            assert_ne!(t.border(), Color::Reset);
            assert_ne!(t.border_active(), Color::Reset);
            assert_ne!(t.floor(), Color::Reset);
            assert_ne!(t.bg(), Color::Reset);
        }
        // The three themes differ in their primary color.
        assert_ne!(Theme::Cyberpunk.primary(), Theme::Vampire.primary());
        assert_ne!(Theme::Vampire.primary(), Theme::Monochrome.primary());
        assert_ne!(Theme::Cyberpunk.primary(), Theme::Monochrome.primary());
    }

    #[test]
    fn vampire_uses_crimson_and_gold() {
        let v = Theme::Vampire;
        assert_eq!(v.primary(), Color::Rgb(220, 20, 60));
        assert_eq!(v.accent(), Color::Rgb(255, 215, 0));
        assert_eq!(v.danger(), Color::Rgb(255, 0, 0));
        assert_eq!(v.border_active(), Color::Rgb(220, 20, 60));
    }

    #[test]
    fn monochrome_uses_pastel_colors() {
        let m = Theme::Monochrome;
        assert_eq!(m.primary(), Color::Rgb(126, 182, 232));
        assert_eq!(m.secondary(), Color::Rgb(200, 162, 200));
        assert_eq!(m.bright(), Color::Rgb(255, 255, 255));
        assert_eq!(m.dim(), Color::Rgb(153, 153, 153));
    }

    #[test]
    fn default_theme_is_cyberpunk() {
        assert_eq!(Theme::default(), Theme::Cyberpunk);
    }
}
