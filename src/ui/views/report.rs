//! The post-run **Matrix Report** takeover (design:
//! `work/plans/MATRIX_REPORT.md`, §3).
//!
//! After a completed benchmark run, the `m` key opens this full-screen
//! view: the assembled `[COMMUNITY-SUBMISSION]` block (the auto fields —
//! hardware / VRAM / model / speed — frozen at open) plus the five
//! user-fill fields (backend / quant / use case / notes / save path) as
//! editable text buffers. `c` copies the block to the terminal's
//! clipboard via OSC 52 (the zero-dep `crate::clipboard` path), `s`
//! saves it to the (editable) save path, and `Esc` returns to the
//! dashboard.
//!
//! **Measurement isolation** (blueprint §4): [`ReportState`] is mutated
//! only in the key path; [`render`] is a pure `&App` read.

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use crate::report::MatrixReport;
use crate::ui::app::App;
use crate::ui::theme::{self, style, Theme};

/// The five editable fields, in focus order (`Tab` / `→` cycle forward,
/// `Shift+Tab` / `←` wrap backward).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportField {
    /// The inference backend (`vLLM` / `llama.cpp` / …).
    Backend,
    /// The quantization (e.g. `Q4_K_M`).
    Quant,
    /// The use case (e.g. `daily chat`).
    UseCase,
    /// Free notes (launch flags, quirks).
    Notes,
    /// The destination path for `s` (save).
    SavePath,
}

impl ReportField {
    /// All fields in focus order.
    pub const ALL: [ReportField; 5] = [
        ReportField::Backend,
        ReportField::Quant,
        ReportField::UseCase,
        ReportField::Notes,
        ReportField::SavePath,
    ];

    /// The field's display label (the form rows / the footer).
    pub fn label(self) -> &'static str {
        match self {
            ReportField::Backend => "backend",
            ReportField::Quant => "quant",
            ReportField::UseCase => "use case",
            ReportField::Notes => "notes",
            ReportField::SavePath => "save path",
        }
    }
}

/// The Matrix Report takeover state. Owned by [`App`]; mutated only in
/// the key path (measurement-isolation, blueprint §4 — rendering is a
/// pure `&App` read).
#[derive(Debug, Clone, Default)]
pub struct ReportState {
    /// The assembled report: the auto fields (hardware / vram / model /
    /// speed) are frozen at open; the user fields are synced in from the
    /// buffers as the user types (and again, idempotently, before
    /// copy/save) so `render()` is always what the user sees.
    pub report: MatrixReport,
    /// The user-fill buffers + character cursors (the `setup.rs`
    /// buffer-based text pattern).
    pub backend: String,
    pub backend_cursor: usize,
    pub quant: String,
    pub quant_cursor: usize,
    pub use_case: String,
    pub use_case_cursor: usize,
    pub notes: String,
    pub notes_cursor: usize,
    pub save_path: String,
    pub save_path_cursor: usize,
    /// The focused field (index into [`ReportField::ALL`]).
    pub focus: usize,
}

impl ReportState {
    /// Fresh takeover: the four text fields start empty (focus on
    /// `backend`), the save path pre-filled by the caller.
    pub fn new(report: MatrixReport, default_save_path: String) -> Self {
        Self {
            report,
            save_path: default_save_path,
            ..Default::default()
        }
    }

    /// The focused field (clamped — a stale index can never panic).
    #[must_use]
    pub fn field(&self) -> ReportField {
        ReportField::ALL[self.focus.min(ReportField::ALL.len() - 1)]
    }

    /// The focused buffer + cursor pair (the char-index editing seam).
    fn buffer(&mut self) -> (&mut String, &mut usize) {
        match self.field() {
            ReportField::Backend => (&mut self.backend, &mut self.backend_cursor),
            ReportField::Quant => (&mut self.quant, &mut self.quant_cursor),
            ReportField::UseCase => (&mut self.use_case, &mut self.use_case_cursor),
            ReportField::Notes => (&mut self.notes, &mut self.notes_cursor),
            ReportField::SavePath => (&mut self.save_path, &mut self.save_path_cursor),
        }
    }

    /// Copy the four user buffers (trimmed) into [`report`](Self::report)
    /// — idempotent. Called as the user types (write-back) and again
    /// before copy/save so the emitted block is exactly what the user
    /// sees. The auto fields (hardware / vram / model / speed) are never
    /// touched here.
    pub fn sync_report(&mut self) {
        self.report.backend = self.backend.trim().to_string();
        self.report.quant = self.quant.trim().to_string();
        self.report.use_case = self.use_case.trim().to_string();
        self.report.notes = self.notes.trim().to_string();
    }

    /// Handle one key press in the takeover.
    ///
    /// | Key | Action |
    /// |---|---|
    /// | `Esc` | close (the App returns to the Dashboard) |
    /// | `c` / `C` | copy the block to the clipboard (OSC 52) |
    /// | `s` / `S` | save the block to the save path |
    /// | `Tab` / `→` / `↓` | focus next field (wraps) |
    /// | `Shift+Tab` / `←` / `↑` | focus previous field (wraps) |
    /// | printable `Char` (incl. space) | insert at the cursor |
    /// | `Backspace` / `Delete` | delete before / at the cursor |
    /// | `Home` / `End` | cursor to start / end |
    ///
    /// `q` is a **typeable character** here (exactly as in Setup): the
    /// `Phase::Report` block in `App::handle_key` returns before the
    /// `q`-quit arm, so it can never open the quit overlay from the
    /// report.
    pub fn handle_key(&mut self, key: &KeyEvent) -> ReportKeyResult {
        match key.code {
            KeyCode::Esc => ReportKeyResult::Closed,
            KeyCode::Char('c') | KeyCode::Char('C') => {
                self.sync_report();
                ReportKeyResult::Copied
            }
            KeyCode::Char('s') | KeyCode::Char('S') => {
                self.sync_report();
                ReportKeyResult::Saved
            }
            // Shift+Tab arrives as a `Tab` with the SHIFT modifier — it
            // must be matched before the plain-Tab arm below.
            KeyCode::Tab if key.modifiers.contains(KeyModifiers::SHIFT) => {
                self.focus = (self.focus + ReportField::ALL.len() - 1) % ReportField::ALL.len();
                ReportKeyResult::Inert
            }
            KeyCode::Tab | KeyCode::Right | KeyCode::Down => {
                self.focus = (self.focus + 1) % ReportField::ALL.len();
                ReportKeyResult::Inert
            }
            KeyCode::Left | KeyCode::Up => {
                self.focus = (self.focus + ReportField::ALL.len() - 1) % ReportField::ALL.len();
                ReportKeyResult::Inert
            }
            // Space is allowed deliberately (notes need it): the Setup
            // `is_ascii_graphic()` guard excludes it.
            KeyCode::Char(c) if c.is_ascii_graphic() || c == ' ' => {
                let (buf, cursor) = self.buffer();
                type_at(buf, cursor, c);
                // Write-back: the report tracks the buffers as the user
                // types, so `render()` (copy/save) is always current.
                self.sync_report();
                ReportKeyResult::Inert
            }
            KeyCode::Backspace => {
                let (buf, cursor) = self.buffer();
                delete_before(buf, cursor);
                self.sync_report();
                ReportKeyResult::Inert
            }
            KeyCode::Delete => {
                let (buf, cursor) = self.buffer();
                delete_at(buf, cursor);
                self.sync_report();
                ReportKeyResult::Inert
            }
            KeyCode::Home => {
                let (_, cursor) = self.buffer();
                *cursor = 0;
                ReportKeyResult::Inert
            }
            KeyCode::End => {
                let (buf, cursor) = self.buffer();
                *cursor = buf.chars().count();
                ReportKeyResult::Inert
            }
            _ => ReportKeyResult::Inert,
        }
    }
}

/// The outcome of a report key press (consumed by `App::handle_key`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReportKeyResult {
    /// The key edited/navigated the form; nothing special to do.
    Inert,
    /// `c` — the App writes the OSC 52 sequence + logs the outcome.
    Copied,
    /// `s` — the App writes the file + logs the outcome.
    Saved,
    /// `Esc` — the App returns to [`crate::ui::app::Phase::Dashboard`].
    Closed,
}

// ── Buffer-based text editing helpers (char-index cursors) ────────────────
// Duplicated from `setup.rs` (they are private there): the codebase
// already duplicates small view helpers per view, and this keeps the diff
// out of a previously-merged file.

/// Insert `c` at the cursor.
fn type_at(s: &mut String, cursor: &mut usize, c: char) {
    let i = (*cursor).min(s.chars().count());
    s.insert(i, c);
    *cursor = i + 1;
}

/// Delete the character before the cursor (`Backspace`).
fn delete_before(s: &mut String, cursor: &mut usize) {
    if *cursor == 0 {
        return;
    }
    *cursor -= 1;
    s.remove(*cursor);
}

/// Delete the character at the cursor (`Delete`).
fn delete_at(s: &mut String, cursor: &mut usize) {
    if *cursor >= s.chars().count() {
        return;
    }
    s.remove(*cursor);
}

// ── Rendering (pure `&App` reads) ─────────────────────────────────────────

/// Render the full-screen Matrix Report takeover into `area`: the
/// top bar (brand + the auto-parser URL), the centered `[COMMUNITY-
/// SUBMISSION]` block (auto fields as-is; the focused field marked `▸`),
/// and the key-hint footer with the current save path.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let th = app.active_theme;
    let s = &app.report;

    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1), // top bar (title + URL)
            Constraint::Min(3),    // body panel
            Constraint::Length(1), // key-hint footer
        ])
        .split(area);

    render_top_bar(chunks[0], th, f);

    // Center the body panel (50% width) so a long save path does not
    // stretch the box (the `setup.rs` body-centering pattern).
    let mid = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([
            Constraint::Percentage(25),
            Constraint::Percentage(50),
            Constraint::Percentage(25),
        ])
        .split(chunks[1]);
    render_body(mid[1], s, th, f);

    render_footer(chunks[2], s, th, f);
}

/// Top bar: brand + the auto-parser URL the block is submitted to.
fn render_top_bar(area: Rect, th: Theme, f: &mut Frame) {
    let spans = vec![
        Span::styled(" CRUCIBLE", style::title(th)),
        Span::styled(
            "·LLM",
            Style::default()
                .fg(th.secondary())
                .add_modifier(Modifier::BOLD),
        ),
        Span::styled(" — MATRIX REPORT ", style::title(th)),
        Span::styled(" │ ", style::tab_separator(th)),
        Span::styled(" madgoathaz.github.io/llm-matrix ", style::info(th)),
    ];
    f.render_widget(Paragraph::new(Line::from(spans)), area);
}

/// One block row: the `▸` focus marker (or two spaces) + the
/// `key: value` line aligned **exactly** as the parser contract
/// (`MatrixReport::render`): `key:` followed by spaces so the value
/// starts at column 11 (`hardware: v`, `vram:     v`, …). Styled by
/// auto/user/editable.
fn block_line(th: Theme, key: &str, value: &str, editable: bool, focused: bool) -> Line<'static> {
    let marker = if focused {
        Span::styled("▸ ", style::highlight(th))
    } else {
        Span::raw("  ")
    };
    let prefix = format!("{key}:");
    let text = format!("{prefix:<10}{value}");
    let st = if focused {
        style::value(th)
    } else if editable {
        // The user-fill fields: highlighted so they read as "fill me in".
        style::highlight(th)
    } else {
        style::value_secondary(th)
    };
    Line::from(vec![marker, Span::styled(text, st)])
}

/// The body panel: the full `[COMMUNITY-SUBMISSION]` block in the
/// parser's fixed line order — the auto fields (hardware / vram / model /
/// speed) as assembled, and the user fields (backend / quant / use case /
/// notes) from the live buffers (the focused one marked `▸`).
fn render_body(area: Rect, s: &ReportState, th: Theme, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title(th, "[COMMUNITY-SUBMISSION]"),
        style::active_border(th),
    );
    let lines = vec![
        Line::from(Span::styled("[COMMUNITY-SUBMISSION]", style::label(th))),
        block_line(th, "hardware", &s.report.hardware, false, false),
        block_line(th, "vram", &s.report.vram, false, false),
        block_line(th, "model", &s.report.model, false, false),
        block_line(
            th,
            "backend",
            &s.backend,
            true,
            s.field() == ReportField::Backend,
        ),
        block_line(th, "quant", &s.quant, true, s.field() == ReportField::Quant),
        block_line(th, "speed", &s.report.speed, false, false),
        block_line(
            th,
            "use_case",
            &s.use_case,
            true,
            s.field() == ReportField::UseCase,
        ),
        block_line(th, "notes", &s.notes, true, s.field() == ReportField::Notes),
        Line::from(Span::styled("[/COMMUNITY-SUBMISSION]", style::label(th))),
    ];
    f.render_widget(
        Paragraph::new(lines).block(block).style(Style::default()),
        area,
    );
}

/// The key-hint footer: the three actions + the current save path (the
/// path row carries the `▸` marker when its field is focused).
fn render_footer(area: Rect, s: &ReportState, th: Theme, f: &mut Frame) {
    let path_marker = if s.field() == ReportField::SavePath {
        Span::styled("▸ ", style::highlight(th))
    } else {
        Span::raw("  ")
    };
    let spans = vec![
        Span::styled("[C]opy", style::footer(th)),
        Span::raw("  "),
        Span::styled("[S]ave", style::footer(th)),
        Span::raw("  "),
        Span::styled("[Esc]lose", style::footer(th)),
        Span::raw("   →  "),
        path_marker,
        Span::styled(s.save_path.clone(), style::value_secondary(th)),
    ];
    f.render_widget(
        Paragraph::new(Line::from(spans)).alignment(ratatui::layout::Alignment::Left),
        area,
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::app::Phase;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn shift_key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::SHIFT)
    }

    fn state() -> ReportState {
        ReportState::new(
            MatrixReport {
                hardware: "1x Mock GPU 24G 24GB + 64GB RAM".to_string(),
                vram: "24GB — 16-24".to_string(),
                model: "test-model".to_string(),
                speed: "45 tok/s".to_string(),
                ..Default::default()
            },
            "/tmp/exports/matrix-1.md".to_string(),
        )
    }

    // ── focus cycling ────────────────────────────────────────────────────

    #[test]
    fn tab_cycles_focus_forward_and_wraps() {
        let mut s = state();
        assert_eq!(s.field(), ReportField::Backend);
        for expected in [
            ReportField::Quant,
            ReportField::UseCase,
            ReportField::Notes,
            ReportField::SavePath,
            ReportField::Backend, // wrap
        ] {
            s.handle_key(&key(KeyCode::Tab));
            assert_eq!(s.field(), expected, "Tab → {expected:?}");
        }
    }

    #[test]
    fn shift_tab_cycles_focus_backward_and_wraps() {
        let mut s = state();
        s.focus = 1; // Quant
        s.handle_key(&shift_key(KeyCode::Tab));
        assert_eq!(s.field(), ReportField::Backend, "Shift+Tab wraps back");
        s.handle_key(&key(KeyCode::Left));
        assert_eq!(s.field(), ReportField::SavePath, "← wraps forward-to-last");
    }

    // ── text editing ─────────────────────────────────────────────────────

    #[test]
    fn typing_updates_the_buffer_and_the_report() {
        let mut s = state();
        for c in "vLLM".chars() {
            s.handle_key(&key(KeyCode::Char(c)));
        }
        assert_eq!(s.backend, "vLLM");
        // Write-back: the report's user field tracks the buffer.
        assert_eq!(s.report.backend, "vLLM");
    }

    #[test]
    fn backspace_and_delete_edit_the_focused_buffer() {
        let mut s = state();
        // `c`/`s` are the copy/save **action** keys in this phase (not
        // typeable) — use a string without them.
        for c in "vLLM".chars() {
            s.handle_key(&key(KeyCode::Char(c)));
        }
        s.handle_key(&key(KeyCode::Backspace)); // → "vLL"
        assert_eq!(s.backend, "vLL");
        s.handle_key(&key(KeyCode::Home));
        s.handle_key(&key(KeyCode::Delete)); // → "LL"
        assert_eq!(s.backend, "LL");
        assert_eq!(s.report.backend, "LL", "write-back after delete");
    }

    #[test]
    fn c_and_s_are_action_keys_not_typeable() {
        // The `c` (copy) / `s` (save) keys are consumed by the action
        // arms, never typed into the focused buffer (plan §3.2).
        let mut s = state();
        s.handle_key(&key(KeyCode::Char('c')));
        assert_eq!(s.backend, "", "c is the copy action, not a typed char");
        s.handle_key(&key(KeyCode::Char('s')));
        assert_eq!(s.backend, "", "s is the save action, not a typed char");
        // `q`, by contrast, **is** typeable here (it can never open the
        // quit overlay from the report — the App's Report branch returns
        // first).
        s.handle_key(&key(KeyCode::Char('q')));
        assert_eq!(s.backend, "q");
    }

    #[test]
    fn space_types_into_the_focused_field() {
        // The widened guard (notes need spaces).
        let mut s = state();
        s.focus = 3; // Notes
        s.handle_key(&key(KeyCode::Char('d')));
        s.handle_key(&key(KeyCode::Char(' ')));
        s.handle_key(&key(KeyCode::Char('a')));
        assert_eq!(s.notes, "d a");
        assert_eq!(s.report.notes, "d a");
    }

    #[test]
    fn sync_trims_trailing_whitespace() {
        let mut s = state();
        s.handle_key(&key(KeyCode::Char('v')));
        s.handle_key(&key(KeyCode::Char(' ')));
        assert_eq!(s.backend, "v ");
        assert_eq!(s.report.backend, "v", "trimmed on write-back");
    }

    // ── the action keys ──────────────────────────────────────────────────

    #[test]
    fn c_is_copied_and_s_is_saved_and_esc_closes() {
        let mut s = state();
        assert_eq!(
            s.handle_key(&key(KeyCode::Char('c'))),
            ReportKeyResult::Copied
        );
        assert_eq!(
            s.handle_key(&key(KeyCode::Char('C'))),
            ReportKeyResult::Copied
        );
        assert_eq!(
            s.handle_key(&key(KeyCode::Char('s'))),
            ReportKeyResult::Saved
        );
        assert_eq!(s.handle_key(&key(KeyCode::Esc)), ReportKeyResult::Closed);
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
    fn render_shows_the_full_block_and_footer() {
        let mut app = App::new();
        app.phase = Phase::Report;
        app.report = state();
        let text = render_text(&app, 120, 40);
        // The two markers + every key line.
        assert!(text.contains("[COMMUNITY-SUBMISSION]"), "{text}");
        assert!(text.contains("hardware:"), "{text}");
        assert!(text.contains("vram:"), "{text}");
        assert!(text.contains("model:"), "{text}");
        assert!(text.contains("backend:"), "{text}");
        assert!(text.contains("quant:"), "{text}");
        assert!(text.contains("speed:"), "{text}");
        assert!(text.contains("use_case:"), "{text}");
        assert!(text.contains("notes:"), "{text}");
        // The auto values.
        assert!(text.contains("test-model"), "{text}");
        assert!(text.contains("45 tok/s"), "{text}");
        assert!(text.contains("24GB — 16-24"), "{text}");
        // The footer: the three actions + the current save path.
        assert!(text.contains("[C]opy"), "{text}");
        assert!(text.contains("[S]ave"), "{text}");
        assert!(text.contains("[Esc]lose"), "{text}");
        assert!(text.contains("/tmp/exports/matrix-1.md"), "{text}");
    }

    #[test]
    fn render_marks_the_focused_field() {
        let mut app = App::new();
        app.phase = Phase::Report;
        app.report = state();
        // Focus 0 (backend) → the `▸` marker leads the backend row.
        let text = render_text(&app, 120, 40);
        assert!(text.contains("▸ backend:"), "focus 0 marks backend: {text}");
        // Move to Notes → the marker follows.
        app.report.focus = 3;
        let text = render_text(&app, 120, 40);
        assert!(text.contains("▸ notes:"), "focus 3 marks notes: {text}");
    }

    #[test]
    fn render_shows_the_typed_values() {
        let mut app = App::new();
        app.phase = Phase::Report;
        app.report = state();
        app.report.backend = "vLLM".to_string();
        app.report.sync_report();
        let text = render_text(&app, 120, 40);
        assert!(text.contains("vLLM"), "the typed backend shows: {text}");
    }
}
