//! `crossterm` event loop with a 60Hz `tokio` tick that drives `ratatui`
//! rendering.
//!
//! **Strictly decoupled from the worker pool** (blueprint §4): this loop
//! only reads `App` state and draws to the terminal. A dropped frame or a
//! terminal resize never touches the quanta timing path. The metric snapshot
//! is the shared `ArcSwap<MetricsSnapshot>` pipeline (Chunk 6): the stream
//! worker publishes it via `MetricsState::update()`, and the render path
//! (`App::render` → views → `MetricsState::load()`) reads it lock-free.

use std::io::{self, Stdout};
use std::time::Duration;

use crossterm::event::{self, Event, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{self, EnterAlternateScreen, LeaveAlternateScreen};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;
use tokio::time::{interval, Interval};

use crate::ui::app::{App, KeyAction};
use crate::ui::theme::style;

/// One render frame: 1000 / 60 ≈ 16 ms.
const TICK: Duration = Duration::from_millis(16);

/// Owns the terminal lifecycle (raw mode + alternate screen) and the 60Hz
/// render tick.
pub struct EventLoop {
    terminal: Terminal<CrosstermBackend<Stdout>>,
    tick: Interval,
}

impl EventLoop {
    /// Enter raw mode + the alternate screen and create the terminal.
    pub fn new() -> io::Result<Self> {
        terminal::enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;
        let terminal = Terminal::new(CrosstermBackend::new(stdout))?;
        Ok(Self {
            terminal,
            tick: interval(TICK),
        })
    }

    /// Leave the alternate screen and disable raw mode (clean exit path).
    pub fn teardown(mut self) -> io::Result<()> {
        let _ = self.terminal.clear();
        terminal::disable_raw_mode()?;
        execute!(io::stdout(), LeaveAlternateScreen)?;
        Ok(())
    }

    /// Best-effort terminal restore for the panic path.
    pub fn restore() {
        let _ = terminal::disable_raw_mode();
        let _ = execute!(io::stdout(), LeaveAlternateScreen);
    }

    /// Run the event loop until `app.running` is false.
    ///
    /// Each iteration: (1) wait on the 60Hz tokio tick, (2) drain terminal
    /// input (key / resize) that arrived since the last frame, (3) advance
    /// app state, (4) draw one frame.
    pub async fn run(&mut self, app: &mut App) -> io::Result<()> {
        // First frame immediately so the dashboard appears before the
        // first tick elapses.
        self.draw(app)?;
        while app.running {
            self.tick.tick().await;

            // Poll input on a blocking thread so the async tick stays
            // responsive. `None` = no event within the poll window.
            let polled = tokio::task::spawn_blocking(|| match event::poll(TICK) {
                Ok(true) => Some(event::read()),
                Ok(false) => None,
                Err(e) => Some(Err(e)),
            })
            .await
            .map_err(io::Error::other)?;

            if let Some(Ok(ev)) = polled {
                match ev {
                    // Act on key presses only (ignore release/repeat).
                    Event::Key(key) if key.kind == KeyEventKind::Press => {
                        match app.handle_key(&key) {
                            KeyAction::Quit => app.running = false,
                            // `e` (Chunk 13): export the current snapshot to
                            // the data dir. A failure only logs a warning —
                            // it never disturbs the render/timing loop.
                            KeyAction::Export => {
                                if let Err(e) = app.export() {
                                    app.push_log(
                                        format!("[export] failed: {e}"),
                                        style::value_warn(),
                                    );
                                }
                            }
                            KeyAction::Continue
                            | KeyAction::PauseResume
                            | KeyAction::StepConcurrency
                            | KeyAction::NewNeedle => {}
                        }
                    }
                    // Resize: the backend re-queries the terminal size on
                    // the next draw — nothing to update here, and the
                    // layout re-solves without panicking.
                    Event::Resize(_, _) => {}
                    _ => {}
                }
            }

            // Advance the render clock. The lock-free
            // `ArcSwap<MetricsSnapshot>` read happens in `draw()` (the views
            // call `MetricsState::load()`); the render loop never writes it.
            app.on_tick();
            self.draw(app)?;
        }
        Ok(())
    }

    fn draw(&mut self, app: &App) -> io::Result<()> {
        self.terminal.draw(|f| app.render(f))?;
        Ok(())
    }
}
