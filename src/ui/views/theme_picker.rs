//! The first-run **theme picker** — a full-screen takeover shown *before*
//! the Setup flow when no color theme has ever been chosen.
//!
//! Three themes are offered (Cyberpunk, Vampire, Monochrome Pastel). The
//! user moves the cursor with `↑`/`↓` and confirms with `Enter`. The key
//! feature is the **live preview**: as the cursor moves, the *entire*
//! screen (outer frame, title, footer, and the highlighted option box)
//! re-colors in the hovered theme, while each option box always shows its
//! own palette so the user can see all three at a glance.
//!
//! `Enter` applies the theme to `app.active_theme`, persists it to the
//! config file (so subsequent runs skip the picker), and falls through to
//! the Setup takeover (fresh target) or the Dashboard (target already
//! given) — see [`App::confirm_theme`](crate::ui::app::App::confirm_theme).
//!
//! **Measurement isolation** (blueprint §4): rendering is a pure `&App`
//! read; the only mutation is the key path (cursor movement / confirm).

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Margin, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::{self, style, Theme};

/// The theme picker's interactive state: the cursor (index into
/// [`Theme::ALL`]) and whether, once a theme is confirmed, the flow should
/// fall through to Setup (fresh target) or the Dashboard (target given).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThemePickerState {
    /// The cursor position (0 = Cyberpunk, 1 = Vampire, 2 = Monochrome).
    pub cursor: usize,
    /// `true` → after confirming, enter the Setup takeover; `false` → go
    /// straight to the Dashboard.
    pub needs_setup: bool,
}

/// Render the full-screen theme picker into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let hovered = Theme::ALL[app.theme_picker.cursor.min(Theme::ALL.len() - 1)];

    // The whole screen previews in the hovered theme (the live preview).
    let outer = theme::block(
        theme::panel_title(hovered, "CRUCIBLE LLM — Select Theme"),
        style::active_border(hovered),
    );
    f.render_widget(Paragraph::new(Line::raw("")).block(outer), area);

    let inner = area.inner(Margin {
        horizontal: 2,
        vertical: 2,
    });
    if inner.width < 20 || inner.height < 8 {
        return;
    }

    // Vertical plan: subtitle, three option boxes (each 4 rows: top border
    // + name + tagline + bottom border), gaps, and the footer.
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // subtitle
            Constraint::Length(1), // gap
            Constraint::Length(4), // box 1
            Constraint::Length(1), // gap
            Constraint::Length(4), // box 2
            Constraint::Length(1), // gap
            Constraint::Length(4), // box 3
            Constraint::Length(1), // gap
            Constraint::Length(1), // footer
        ])
        .split(inner);

    // Subtitle (hovered theme's dim color).
    f.render_widget(
        Paragraph::new(
            Line::from(Span::styled(
                "Choose a color theme — the preview updates as you move",
                style::footer(hovered),
            ))
            .alignment(Alignment::Center),
        ),
        rows[0],
    );

    // The three option boxes, each rendered in its OWN theme so all three
    // palettes are visible at once; the selected one gets the active border.
    let box_w = inner.width.clamp(24, 56);
    let x = inner.x + (inner.width.saturating_sub(box_w)) / 2;
    for i in 0..Theme::ALL.len() {
        let t = Theme::ALL[i];
        let opt = match i {
            0 => rows[2],
            1 => rows[4],
            _ => rows[6],
        };
        let selected = i == app.theme_picker.cursor;
        let box_rect = Rect::new(x, opt.y, box_w, opt.height);
        let border = if selected {
            style::active_border(t)
        } else {
            style::border(t)
        };
        let block = theme::block("", border);
        let marker = if selected { "> " } else { "  " };
        let lines = vec![
            Line::from(vec![
                Span::raw(marker),
                Span::styled(
                    t.name().to_ascii_uppercase(),
                    Style::default()
                        .fg(t.primary())
                        .add_modifier(Modifier::BOLD),
                ),
            ]),
            Line::from(Span::styled(t.tagline().to_string(), style::label(t))),
        ];
        f.render_widget(Paragraph::new(lines).block(block), box_rect);
    }

    // Footer (hovered theme).
    f.render_widget(
        Paragraph::new(
            Line::from(Span::styled(
                "↑↓ Select   ·   Enter Confirm",
                style::footer(hovered),
            ))
            .alignment(Alignment::Center),
        ),
        rows[8],
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::app::Phase;

    fn render_text(app: &App, w: u16, h: u16) -> String {
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
    fn picker_shows_all_three_themes() {
        let mut app = App::new();
        app.phase = Phase::ThemePicker;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("SELECT THEME"), "{text}");
        assert!(text.contains("CYBERPUNK"), "{text}");
        assert!(text.contains("VAMPIRE"), "{text}");
        assert!(text.contains("MONOCHROME PASTEL"), "{text}");
        assert!(text.contains("Enter Confirm"), "{text}");
    }

    #[test]
    fn picker_marks_the_selected_theme() {
        let mut app = App::new();
        app.phase = Phase::ThemePicker;
        // Default cursor is 0 (Cyberpunk) → the `> ` marker leads it.
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("> CYBERPUNK"),
            "cursor 0 marks Cyberpunk: {text}"
        );

        // Move to Vampire (cursor 1) → the marker follows.
        app.theme_picker.cursor = 1;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("> VAMPIRE"), "cursor 1 marks Vampire: {text}");
    }
}
