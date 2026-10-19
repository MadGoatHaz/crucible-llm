//! The **run-log viewer** takeover (design: `work/plans/LOG_ACCESS.md`,
//! §A.2).
//!
//! The `l` key opens this full-screen view with **no completion gate**
//! (the run log exists from process start): a scrollable, **read-only**
//! snapshot of the current run's `latest.log` — or, from the History
//! view, a stored session's archive log (`run-<stamp>-<session8>.log`).
//! `c` copies the full log to the terminal's clipboard via OSC 52 (the
//! zero-dep `crate::clipboard` path), `s` saves it to the pre-filled
//! save path, and `Esc` returns to the dashboard.
//!
//! **Measurement isolation** (blueprint §4): [`LogState`] is mutated
//! only in the key path; [`render`] is a pure `&App` read.
//!
//! **N/A-never-fail**: when the log cannot be read (disabled logger, a
//! missing/unreadable file), the view shows a single dim placeholder
//! line instead of crashing — Copy/Save are omitted from the footer
//! (they would operate on nothing) and `Esc` still closes.
//!
//! The content is a **snapshot** taken at open (the file keeps growing
//! while a run is in progress; re-`l` re-reads it — there is no
//! tail-follow mode, which keeps the view stateless).

use crossterm::event::{KeyCode, KeyEvent};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Paragraph, Wrap};
use ratatui::Frame;

use crate::ui::app::App;
use crate::ui::theme::{self, style, Theme};

/// The run-log viewer takeover state. Owned by [`App`]; mutated only in
/// the key path (measurement-isolation, blueprint §4 — rendering is a
/// pure `&App` read). The view is **read-only** (no editable fields):
/// the log lines are a frozen snapshot and the save path is pre-filled
/// at open.
#[derive(Debug, Clone, Default)]
pub struct LogState {
    /// The full log text, split on newlines (the snapshot taken at
    /// open; the file keeps growing — re-`l` to refresh).
    pub lines: Vec<String>,
    /// First visible line (clamped to the content in render — the key
    /// handler receives no `Rect`).
    pub scroll: usize,
    /// The displayed file path (top bar + footer).
    pub path: String,
    /// The `s` key's save destination (pre-filled at open; not
    /// user-editable — the view is read-only).
    pub save_path: String,
    /// N/A-never-fail placeholder mode: `lines` carries the single
    /// placeholder line and the footer omits Copy/Save.
    pub unavailable: bool,
}

impl LogState {
    /// Fresh takeover: the log lines + the displayed path + the
    /// pre-filled save destination.
    pub fn new(lines: Vec<String>, path: String, save_path: String) -> Self {
        Self {
            lines,
            path,
            save_path,
            ..Default::default()
        }
    }

    /// N/A-never-fail placeholder (a single dim line — e.g. `run log
    /// unavailable (logger disabled)` / `cannot read …: {e}` / `no
    /// archive log found for session …`).
    pub fn unavailable(msg: String) -> Self {
        Self {
            lines: vec![msg],
            unavailable: true,
            ..Default::default()
        }
    }

    /// The first visible line index, clamped to the content (a stale
    /// over-end scroll can never panic).
    fn clamped_scroll(&self) -> usize {
        self.scroll.min(self.lines.len().saturating_sub(1))
    }

    /// Handle one key press in the takeover.
    ///
    /// | Key | Action |
    /// |---|---|
    /// | `Esc` | close (the App returns to the Dashboard) |
    /// | `c` / `C` | copy the full log to the clipboard (OSC 52) |
    /// | `s` / `S` | save the full log to the save path |
    /// | `j` / `↓` | scroll down one line |
    /// | `k` / `↑` | scroll up one line |
    /// | `PageDown` / `PageUp` | scroll ±10 lines (fixed page quantum —
    /// | | the view handler receives no `Rect`; clamping happens in render) |
    /// | `Home` / `End` | scroll to the first / last line |
    ///
    /// Everything else (including `q`) is **inert**: the `Phase::Log`
    /// block in `App::handle_key` returns before the `q`-quit arm, so
    /// `q` can never open the quit overlay from the log view (the same
    /// rule as the report).
    pub fn handle_key(&mut self, key: &KeyEvent) -> LogKeyResult {
        match key.code {
            KeyCode::Esc => LogKeyResult::Closed,
            KeyCode::Char('c') | KeyCode::Char('C') => LogKeyResult::Copied,
            KeyCode::Char('s') | KeyCode::Char('S') => LogKeyResult::Saved,
            KeyCode::Char('j') | KeyCode::Down => {
                self.scroll += 1;
                LogKeyResult::Inert
            }
            KeyCode::Char('k') | KeyCode::Up => {
                self.scroll = self.scroll.saturating_sub(1);
                LogKeyResult::Inert
            }
            KeyCode::PageDown => {
                self.scroll += 10;
                LogKeyResult::Inert
            }
            KeyCode::PageUp => {
                self.scroll = self.scroll.saturating_sub(10);
                LogKeyResult::Inert
            }
            KeyCode::Home => {
                self.scroll = 0;
                LogKeyResult::Inert
            }
            KeyCode::End => {
                self.scroll = self.lines.len().saturating_sub(1);
                LogKeyResult::Inert
            }
            _ => LogKeyResult::Inert,
        }
    }
}

/// The outcome of a log-view key press (consumed by `App::handle_key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogKeyResult {
    /// The key scrolled the view; nothing special to do.
    Inert,
    /// `c` — the App writes the OSC 52 sequence + logs the outcome.
    Copied,
    /// `s` — the App writes the file + logs the outcome.
    Saved,
    /// `Esc` — the App returns to [`crate::ui::app::Phase::Dashboard`].
    Closed,
}

// ── Rendering (pure `&App` reads) ─────────────────────────────────────────

/// Render the full-screen run-log viewer into `area`: the top bar
/// (brand + the log file path), the scrollable wrapped log lines, and
/// the key-hint footer with the visible line range.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
    let s = &app.log_view;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // top bar (title + path)
            Constraint::Min(3),    // body panel
            Constraint::Length(1), // key-hint footer
        ])
        .split(area);

    render_top_bar(chunks[0], th, s, f);

    // Center the body panel (50% width) — the `report.rs`
    // body-centering pattern (a very long path does not stretch the
    // box).
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(50),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);
    render_body(mid[1], th, s, f);

    render_footer(chunks[2], th, s, chunks[1].height as usize, f);
}

/// Top bar: brand + the log file path being viewed.
fn render_top_bar(area: Rect, th: Theme, s: &LogState, f: &mut Frame) {
    let spans = vec![
        Span::styled(" CRUCIBLE", style::title(th)),
        Span::styled(
            "·LLM",
            Style::default()
                .fg(th.secondary())
                .add_modifier(ratatui::style::Modifier::BOLD),
        ),
        Span::styled(" — RUN LOG ", style::title(th)),
        Span::styled(" │ ", style::tab_separator(th)),
        Span::styled(s.path.clone(), style::info(th)),
    ];
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// The body panel: the scrollable, wrapped log lines (borrowed spans —
/// no per-frame string clones; the snapshot lives in [`LogState`]).
/// In `unavailable` mode the single placeholder line renders dim and
/// the panel is unbordered-active (there is nothing to act on).
fn render_body(area: Rect, th: Theme, s: &LogState, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "RUN LOG"),
        if s.unavailable {
            style::border(th)
        } else {
            style::active_border(th)
        },
    );
    let text: Vec<Line<'_>> = if s.unavailable {
        vec![Line::from(Span::styled(
            s.lines.first().cloned().unwrap_or_default(),
            style::info(th),
        ))]
    } else {
        s.lines
            .iter()
            .map(|l| Line::from(Span::styled(l, style::value(th))))
            .collect()
    };
    f.render_widget(
        Paragraph::new(text)
            .scroll((s.clamped_scroll() as u16, 0))
            .wrap(Wrap { trim: true })
            .block(block),
        area,
    );
}

/// The key-hint footer: the actions (omitted in `unavailable` mode —
/// they would operate on nothing) + the visible line range + the path.
fn render_footer(area: Rect, th: Theme, s: &LogState, visible: usize, f: &mut Frame) {
    let spans = if s.unavailable {
        vec![Span::styled("[Esc]lose", style::footer(th))]
    } else {
        let n = s.lines.len();
        let first = if n == 0 { 0 } else { s.clamped_scroll() + 1 };
        let last = if n == 0 {
            0
        } else {
            (s.clamped_scroll() + visible).min(n)
        };
        vec![
            Span::styled("[C]opy", style::footer(th)),
            Span::raw("  "),
            Span::styled("[S]ave", style::footer(th)),
            Span::raw("  "),
            Span::styled("[Esc]lose", style::footer(th)),
            Span::raw("   │  "),
            Span::styled(format!("line {first}–{last} of {n}"), style::info(th)),
            Span::raw("   │  "),
            Span::styled(s.path.clone(), style::value_secondary(th)),
        ]
    };
    f.render_widget(
        Paragraph::new(Line::from(spans)).alignment(ratatui::layout::Alignment::Left),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::app::Phase;
    use crossterm::event::KeyModifiers;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn state() -> LogState {
        LogState::new(
            vec![
                "=== crucible-llm run log — 2026-10-19 01:00:00 (session 12345678) ===".to_string(),
                "[01:00:00.000] [INFO] [SETUP] run started".to_string(),
                "[01:00:01.000] [INFO] [SSE] token #1 received".to_string(),
            ],
            "/tmp/logs/latest.log".to_string(),
            "/tmp/exports/run-log-1.log".to_string(),
        )
    }

    // ── the action keys ──────────────────────────────────────────────────

    #[test]
    fn c_is_copied_s_is_saved_and_esc_closes() {
        let mut s = state();
        assert_eq!(s.handle_key(&key(KeyCode::Char('c'))), LogKeyResult::Copied);
        assert_eq!(s.handle_key(&key(KeyCode::Char('C'))), LogKeyResult::Copied);
        assert_eq!(s.handle_key(&key(KeyCode::Char('s'))), LogKeyResult::Saved);
        assert_eq!(s.handle_key(&key(KeyCode::Char('S'))), LogKeyResult::Saved);
        assert_eq!(s.handle_key(&key(KeyCode::Esc)), LogKeyResult::Closed);
    }

    #[test]
    fn q_and_other_keys_are_inert_in_the_log_view() {
        // `q` is inert (the App's Log branch returns before the q-quit
        // arm — the quit overlay can never open from the log view), and
        // the content is never mutated by it.
        let mut s = state();
        assert_eq!(s.handle_key(&key(KeyCode::Char('q'))), LogKeyResult::Inert);
        assert_eq!(s.handle_key(&key(KeyCode::Char('m'))), LogKeyResult::Inert);
        assert_eq!(s.handle_key(&key(KeyCode::Char('l'))), LogKeyResult::Inert);
        assert_eq!(s.lines.len(), 3, "inert keys never touch the content");
    }

    // ── scrolling ────────────────────────────────────────────────────────

    #[test]
    fn scroll_j_k_page_home_end() {
        let mut s = state();
        s.handle_key(&key(KeyCode::Char('j')));
        assert_eq!(s.scroll, 1);
        s.handle_key(&key(KeyCode::Down));
        assert_eq!(s.scroll, 2);
        s.handle_key(&key(KeyCode::Char('k')));
        assert_eq!(s.scroll, 1);
        s.handle_key(&key(KeyCode::Up));
        assert_eq!(s.scroll, 0);
        s.handle_key(&key(KeyCode::PageDown));
        assert_eq!(s.scroll, 10);
        s.handle_key(&key(KeyCode::PageUp));
        assert_eq!(s.scroll, 0);
        s.handle_key(&key(KeyCode::End));
        assert_eq!(s.scroll, 2, "End = the last line index");
        s.handle_key(&key(KeyCode::Home));
        assert_eq!(s.scroll, 0);
        // `k` at the top saturates (no underflow).
        s.handle_key(&key(KeyCode::Char('k')));
        assert_eq!(s.scroll, 0);
    }

    #[test]
    fn clamped_scroll_never_exceeds_the_content() {
        let mut s = state();
        s.scroll = 1000; // a stale over-end scroll
        assert_eq!(s.clamped_scroll(), 2);
        let empty = LogState::default();
        assert_eq!(empty.clamped_scroll(), 0, "empty content clamps to 0");
    }

    // ── rendering (ratatui::TestBackend) ─────────────────────────────────

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
    fn render_shows_the_lines_path_and_counts() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = state();
        let text = render_text(&app, 120, 40);
        assert!(text.contains("RUN LOG"), "{text}");
        assert!(
            text.contains("/tmp/logs/latest.log"),
            "path in the bar: {text}"
        );
        assert!(text.contains("run started"), "log line 2: {text}");
        assert!(text.contains("token #1 received"), "log line 3: {text}");
        // The footer: the three actions + the visible range + the path.
        assert!(text.contains("[C]opy"), "{text}");
        assert!(text.contains("[S]ave"), "{text}");
        assert!(text.contains("[Esc]lose"), "{text}");
        assert!(text.contains("line 1–3 of 3"), "{text}");
    }

    #[test]
    fn render_clamps_an_over_end_scroll() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = state();
        app.log_view.scroll = 999;
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("line 3–3 of 3"),
            "the range is clamped to the content: {text}"
        );
    }

    #[test]
    fn render_unavailable_shows_the_placeholder_and_omits_copy_save() {
        let mut app = App::new();
        app.phase = Phase::Log;
        app.log_view = LogState::unavailable("run log unavailable (logger disabled)".to_string());
        let text = render_text(&app, 120, 40);
        assert!(
            text.contains("run log unavailable (logger disabled)"),
            "the placeholder renders: {text}"
        );
        assert!(!text.contains("[C]opy"), "copy omitted: {text}");
        assert!(!text.contains("[S]ave"), "save omitted: {text}");
        assert!(text.contains("[Esc]lose"), "Esc still closes: {text}");
    }
}
