//! View 4 — History: list, detail, compare, and delete of stored benchmark
//! runs (blueprint §6).
//!
//! **Modes** (driven by the key path, rendered pure from `&App`):
//!
//! * [`HistoryMode::List`] — the session table with a cursor;
//! * [`HistoryMode::Detail`] — one run's per-engine summary;
//! * [`HistoryMode::Compare`] — two runs side-by-side with signed deltas;
//! * [`HistoryMode::DeleteConfirm`] — a `[y/N]` prompt before deleting.
//!
//! **Keys** (active in this view only; `1`–`5` and `q` are global):
//!
//! * `j` / `↓` and `k` / `↑` — move the cursor (List / Compare-select);
//! * `Enter` — view details (List) / select the second run (Compare);
//! * `C` — enter Compare mode (List only);
//! * `D` — delete the selected run (List only, with confirmation);
//! * `Esc` — back to the List (from Detail / Compare / DeleteConfirm);
//! * `y` / `n` — confirm / cancel the delete prompt.
//!
//! **Empty state**: when no runs are stored the view shows a centered
//! placeholder explaining that results will appear after the first
//! benchmark.
//!
//! **Measurement isolation** (blueprint §4): the render path is a pure
//! `&App` read — all DB I/O happens in the key path (user-driven, rare)
//! and the results are cached in [`HistoryState`] so the 60 Hz frame loop
//! never touches the DB.

use std::path::Path;

use ratatui::layout::{Alignment, Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Cell, Paragraph, Row, Table, Wrap};
use ratatui::Frame;

use crate::storage::db::{Database, StorageError};
use crate::storage::models::{BenchmarkSession, StreamMetricRow};
use crate::ui::app::{fmt, App};
use crate::ui::theme::{self, palette, style};

// ── mode ─────────────────────────────────────────────────────────────────

/// The History view's interactive mode.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum HistoryMode {
    /// The session list with a cursor.
    #[default]
    List,
    /// Detailed results for one run (index into `sessions`).
    Detail(usize),
    /// Compare two runs: `first` is set, `second` is selected with `Enter`.
    Compare { first: usize, second: Option<usize> },
    /// Delete confirmation for the run at `idx`.
    DeleteConfirm(usize),
}

// ── pure diff computation (no locks, no I/O) ────────────────────────────

/// Per-metric summary of one stored session: the mean over its
/// `stream_metrics` rows (rows lacking a value are skipped, so a NULL
/// `joules_per_token` before hardware telemetry never skews the others).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionSummary {
    /// Mean TTFT, milliseconds.
    pub ttft_ms: Option<f64>,
    /// Mean decode speed, tokens/sec (derived from `tpot_ms`).
    pub tokens_per_sec: Option<f64>,
    /// Mean prompt throughput, tokens/sec (derived from `ttft_ms` + `prompt_tokens`).
    pub prompt_tps: Option<f64>,
    /// Mean MTP η (tokens per content chunk).
    pub mtp: Option<f64>,
    /// Mean silicon efficiency (Chunk 17; `None` without GPU telemetry).
    pub joules_per_token: Option<f64>,
}

/// Aggregate a session's `stream_metrics` rows into its summary.
pub fn session_summary(metrics: &[StreamMetricRow]) -> SessionSummary {
    let ttft: Vec<f64> = metrics.iter().filter_map(|m| m.ttft_ms).collect();
    let tps: Vec<f64> = metrics
        .iter()
        .filter_map(|m| m.tpot_ms)
        .filter(|v| *v > 0.0)
        .map(|v| 1000.0 / v)
        .collect();
    // Prompt throughput: prompt_tokens / (ttft_ms / 1000) per row.
    let pps: Vec<f64> = metrics
        .iter()
        .filter_map(|m| match (m.prompt_tokens, m.ttft_ms) {
            (Some(pt), Some(ttft)) if ttft > 0.0 => Some(pt as f64 / (ttft / 1000.0)),
            _ => None,
        })
        .collect();
    let mtp: Vec<f64> = metrics.iter().filter_map(|m| m.mtp_efficiency).collect();
    let joules: Vec<f64> = metrics.iter().filter_map(|m| m.joules_per_token).collect();
    SessionSummary {
        ttft_ms: mean(&ttft),
        tokens_per_sec: mean(&tps),
        prompt_tps: mean(&pps),
        mtp: mean(&mtp),
        joules_per_token: mean(&joules),
    }
}

fn mean(values: &[f64]) -> Option<f64> {
    values
        .first()
        .map(|_| values.iter().sum::<f64>() / values.len() as f64)
}

/// One side-by-side delta row of the diff table.
#[derive(Debug, Clone)]
pub struct DiffRow {
    pub label: &'static str,
    /// Run A's value (mean over its stored rows).
    pub a: Option<f64>,
    /// Run B's value.
    pub b: Option<f64>,
    /// Signed % change of B relative to A: `(b − a) / a × 100`
    /// (`None` when not computable).
    pub delta_pct: Option<f64>,
    /// `Some(true)` = improvement (green), `Some(false)` = regression
    /// (red), `None` = neutral / not comparable.
    pub improved: Option<bool>,
}

/// Compute one metric's delta row.
pub fn diff_row(
    label: &'static str,
    a: Option<f64>,
    b: Option<f64>,
    lower_is_better: bool,
) -> DiffRow {
    let delta_pct = match (a, b) {
        (Some(a), Some(b)) if a.abs() > f64::EPSILON => Some((b - a) / a * 100.0),
        _ => None,
    };
    let improved = match (a, b, delta_pct) {
        (Some(a), Some(b), Some(d)) if d.abs() > 1e-9 => {
            Some(if lower_is_better { b < a } else { b > a })
        }
        _ => None,
    };
    DiffRow {
        label,
        a,
        b,
        delta_pct,
        improved,
    }
}

/// The computed diff between two stored runs.
#[derive(Debug, Clone)]
pub struct DiffReport {
    pub a: BenchmarkSession,
    pub b: BenchmarkSession,
    pub summary_a: SessionSummary,
    pub summary_b: SessionSummary,
    pub rows: Vec<DiffRow>,
}

impl DiffReport {
    /// Compare run A vs run B from their stored metric rows (pure).
    pub fn from_sessions(
        a: &BenchmarkSession,
        a_metrics: &[StreamMetricRow],
        b: &BenchmarkSession,
        b_metrics: &[StreamMetricRow],
    ) -> Self {
        let summary_a = session_summary(a_metrics);
        let summary_b = session_summary(b_metrics);
        let rows = vec![
            diff_row("TTFT (ms)", summary_a.ttft_ms, summary_b.ttft_ms, true),
            diff_row(
                "Gen t/s",
                summary_a.tokens_per_sec,
                summary_b.tokens_per_sec,
                false,
            ),
            diff_row(
                "Prompt t/s",
                summary_a.prompt_tps,
                summary_b.prompt_tps,
                false,
            ),
            diff_row("MTP rate", summary_a.mtp, summary_b.mtp, false),
            diff_row(
                "J/token",
                summary_a.joules_per_token,
                summary_b.joules_per_token,
                true,
            ),
        ];
        Self {
            a: a.clone(),
            b: b.clone(),
            summary_a,
            summary_b,
            rows,
        }
    }
}

// ── interactive state (key path) ────────────────────────────────────────

/// The History view's interactive state: the stored session list (newest
/// first), the cursor, the current mode, and pre-computed data for the
/// active mode (so the render path stays a pure `&App` read).
#[derive(Debug, Default)]
pub struct HistoryState {
    pub db_path: std::path::PathBuf,
    pub sessions: Vec<BenchmarkSession>,
    pub cursor: usize,
    pub mode: HistoryMode,
    /// Pre-computed detail data (set when entering `Detail` mode).
    pub detail_summary: Option<SessionSummary>,
    /// Pre-computed NIAH count (retrieved, total) for the detail view.
    pub detail_needle: Option<(usize, usize)>,
    /// Pre-computed comparison (set when both runs are selected).
    pub compare_diff: Option<DiffReport>,
}

impl HistoryState {
    /// Open the DB at `path` and load all stored sessions (newest first).
    pub fn load(path: &Path) -> Result<Self, StorageError> {
        let db = Database::open(path)?;
        let sessions = db.list_sessions()?;
        Ok(Self {
            db_path: path.to_path_buf(),
            sessions,
            cursor: 0,
            mode: HistoryMode::List,
            detail_summary: None,
            detail_needle: None,
            compare_diff: None,
        })
    }

    /// Move the cursor by `delta` (clamped to the list bounds).
    pub fn move_cursor(&mut self, delta: isize) {
        if self.sessions.is_empty() {
            return;
        }
        let len = self.sessions.len() as isize;
        self.cursor = (self.cursor as isize + delta).clamp(0, len - 1) as usize;
    }

    /// Enter Detail mode for the cursor's session (fetches metrics).
    pub fn enter_detail(&mut self) {
        if self.cursor >= self.sessions.len() {
            return;
        }
        let session_id = self.sessions[self.cursor].session_id.clone();
        let db = match Database::open(&self.db_path) {
            Ok(db) => db,
            Err(_) => return,
        };
        self.detail_summary = db
            .get_stream_metrics(&session_id)
            .ok()
            .map(|m| session_summary(&m));
        self.detail_needle = db.get_needle_evaluations(&session_id).ok().map(|evals| {
            (
                evals
                    .iter()
                    .filter(|e| e.retrieved_successfully == Some(true))
                    .count(),
                evals.len(),
            )
        });
        self.mode = HistoryMode::Detail(self.cursor);
    }

    /// Enter Compare mode with the cursor's session as the first run.
    pub fn enter_compare(&mut self) {
        if self.cursor >= self.sessions.len() {
            return;
        }
        self.compare_diff = None;
        self.mode = HistoryMode::Compare {
            first: self.cursor,
            second: None,
        };
    }

    /// Select the second run for the comparison (Enter in Compare mode).
    pub fn select_compare_second(&mut self) {
        let HistoryMode::Compare { first, .. } = &self.mode else {
            return;
        };
        if self.cursor == *first || self.cursor >= self.sessions.len() {
            return; // can't compare a run with itself
        }
        let db = match Database::open(&self.db_path) {
            Ok(db) => db,
            Err(_) => return,
        };
        let a_id = self.sessions[*first].session_id.clone();
        let b_id = self.sessions[self.cursor].session_id.clone();
        let ma = db.get_stream_metrics(&a_id).ok();
        let mb = db.get_stream_metrics(&b_id).ok();
        if let (Some(ma), Some(mb)) = (ma, mb) {
            self.compare_diff = Some(DiffReport::from_sessions(
                &self.sessions[*first],
                &ma,
                &self.sessions[self.cursor],
                &mb,
            ));
        }
        self.mode = HistoryMode::Compare {
            first: *first,
            second: Some(self.cursor),
        };
    }

    /// Enter the delete-confirmation prompt for the cursor's session.
    pub fn enter_delete_confirm(&mut self) {
        if self.cursor >= self.sessions.len() {
            return;
        }
        self.mode = HistoryMode::DeleteConfirm(self.cursor);
    }

    /// Confirm the delete: remove the session from the DB and the list.
    pub fn confirm_delete(&mut self) {
        let HistoryMode::DeleteConfirm(idx) = &self.mode else {
            return;
        };
        if *idx >= self.sessions.len() {
            self.mode = HistoryMode::List;
            return;
        }
        let session_id = self.sessions[*idx].session_id.clone();
        let mut db = match Database::open(&self.db_path) {
            Ok(db) => db,
            Err(_) => {
                self.mode = HistoryMode::List;
                return;
            }
        };
        if db.delete_session(&session_id).is_ok() {
            self.sessions.remove(*idx);
            // Clamp the cursor to the new list bounds.
            if self.cursor >= self.sessions.len() && !self.sessions.is_empty() {
                self.cursor = self.sessions.len() - 1;
            }
            if self.sessions.is_empty() {
                self.cursor = 0;
            }
        }
        self.mode = HistoryMode::List;
        self.detail_summary = None;
        self.detail_needle = None;
        self.compare_diff = None;
    }

    /// Cancel back to the List mode (Esc / n).
    pub fn back_to_list(&mut self) {
        self.mode = HistoryMode::List;
        self.detail_summary = None;
        self.detail_needle = None;
        self.compare_diff = None;
    }

    /// Re-load the session list from the DB (after a run completes).
    pub fn reload(&mut self) {
        let db = match Database::open(&self.db_path) {
            Ok(db) => db,
            Err(_) => return,
        };
        self.sessions = db.list_sessions().unwrap_or_default();
        if self.cursor >= self.sessions.len() {
            self.cursor = self.sessions.len().saturating_sub(1);
        }
    }
}

// ── rendering (pure `&App` read) ────────────────────────────────────────

/// Render the History view into `area`. Dispatches on the current mode.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let Some(h) = app.history.as_ref() else {
        // No DB loaded yet — show the empty placeholder.
        render_empty(area, f);
        return;
    };
    if h.sessions.is_empty() {
        render_empty(area, f);
        return;
    }
    match &h.mode {
        HistoryMode::List => render_list(area, h, f),
        HistoryMode::Detail(idx) => render_detail(area, h, *idx, f),
        HistoryMode::Compare { first, second } => {
            if second.is_some() {
                render_compare(area, h, *first, second.unwrap(), f);
            } else {
                render_compare_select(area, h, *first, f);
            }
        }
        HistoryMode::DeleteConfirm(idx) => render_delete_confirm(area, h, *idx, f),
    }
}

/// The empty-state placeholder.
fn render_empty(area: Rect, f: &mut Frame) {
    let block = theme::block(theme::panel_title("HISTORY"), style::border());
    let lines = vec![
        Line::raw(""),
        Line::from(Span::styled("No benchmark runs saved yet.", style::value())),
        Line::from(Span::styled(
            "Complete a run and results will appear here.",
            style::info(),
        )),
        Line::from(Span::styled(
            "You can then compare runs over time to track improvements.",
            style::info(),
        )),
    ];
    f.render_widget(
        Paragraph::new(lines)
            .block(block)
            .wrap(Wrap { trim: true })
            .alignment(Alignment::Center),
        area,
    );
}

/// The session list with cursor highlighting and key-hint subtitle.
fn render_list(area: Rect, h: &HistoryState, f: &mut Frame) {
    let title_line = Line::from(vec![Span::styled(
        format!(
            "HISTORY — Past Benchmark Runs  ({} total)",
            h.sessions.len()
        ),
        style::title(),
    )]);
    let hints_line = Line::from(Span::styled(
        "  ↑↓/jk scroll │ Enter details │ C compare │ D delete │ 1-5 views │ q quit",
        style::footer(),
    ));
    let block = theme::block(title_line, style::border());

    let rows: Vec<Row> = h
        .sessions
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ts = s.timestamp.as_deref().unwrap_or("--");
            let ts_short = if ts.len() >= 16 { &ts[..16] } else { ts };
            let dur = s
                .total_duration_sec
                .map(fmt::format_duration)
                .unwrap_or_else(|| "--".to_string());
            let is_cursor = i == h.cursor;
            let row_style = if is_cursor {
                Style::default().fg(palette::ACCENT)
            } else {
                Style::default()
            };
            let prefix = if is_cursor { "> " } else { "  " };
            Row::new(vec![
                Cell::from(format!("{prefix}[{}]", i + 1)).style(row_style),
                Cell::from(ts_short.to_string()).style(row_style),
                Cell::from(fmt::truncate(&s.model_name, 20)).style(row_style),
                Cell::from(fmt::truncate(&s.target_url, 28)).style(row_style),
                Cell::from(dur).style(row_style),
            ])
        })
        .collect();

    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(6),
                Constraint::Length(17),
                Constraint::Percentage(25),
                Constraint::Percentage(38),
                Constraint::Length(10),
            ],
        )
        .header(
            Row::new(vec![
                Cell::from(""),
                Cell::from("Date"),
                Cell::from("Model"),
                Cell::from("Endpoint"),
                Cell::from("Duration"),
            ])
            .style(style::muted_title()),
        )
        .block(block),
        sub[0],
    );
    f.render_widget(Paragraph::new(hints_line), sub[1]);
}

/// Detail view for one run.
fn render_detail(area: Rect, h: &HistoryState, idx: usize, f: &mut Frame) {
    let Some(s) = h.sessions.get(idx) else {
        render_list(area, h, f);
        return;
    };
    let ts = s.timestamp.as_deref().unwrap_or("--");
    let title = format!(
        "RUN DETAILS — {} ({})",
        if ts.len() >= 16 { &ts[..16] } else { ts },
        fmt::truncate(&s.model_name, 24)
    );
    let block = theme::block(
        Line::from(Span::styled(title, style::title())),
        style::active_border(),
    );

    let mut lines: Vec<Line> = Vec::new();
    lines.push(Line::raw(""));

    // Session metadata.
    let dur = s
        .total_duration_sec
        .map(fmt::format_duration)
        .unwrap_or_else(|| "--".to_string());
    lines.push(Line::from(vec![
        Span::styled("  Endpoint:  ", style::label()),
        Span::styled(fmt::truncate(&s.target_url, 50), style::value()),
    ]));
    lines.push(Line::from(vec![
        Span::styled("  Duration:  ", style::label()),
        Span::styled(dur, style::value()),
    ]));
    if let Some(gpu) = &s.system_gpu {
        lines.push(Line::from(vec![
            Span::styled("  GPU:       ", style::label()),
            Span::styled(gpu.clone(), style::value()),
        ]));
    }
    lines.push(Line::raw(""));

    // Speed metrics (Engine A).
    lines.push(Line::from(Span::styled(
        "  Engine A: Speed",
        style::value_ok(),
    )));
    match &h.detail_summary {
        Some(sum) => {
            if let Some(ttft) = sum.ttft_ms {
                lines.push(Line::from(vec![
                    Span::styled("    TTFT:        ", style::label()),
                    Span::styled(format!("{ttft:.1} ms"), style::value()),
                ]));
            }
            if let Some(tps) = sum.tokens_per_sec {
                lines.push(Line::from(vec![
                    Span::styled("    Gen speed:   ", style::label()),
                    Span::styled(fmt::format_rate(tps), style::value()),
                ]));
            }
            if let Some(pps) = sum.prompt_tps {
                lines.push(Line::from(vec![
                    Span::styled("    Prompt t/s:  ", style::label()),
                    Span::styled(fmt::format_rate(pps), style::value()),
                ]));
            }
            if let Some(mtp) = sum.mtp {
                lines.push(Line::from(vec![
                    Span::styled("    MTP rate:    ", style::label()),
                    Span::styled(format!("{mtp:.2} x"), style::highlight()),
                ]));
            }
            if let Some(jpt) = sum.joules_per_token {
                lines.push(Line::from(vec![
                    Span::styled("    J/token:     ", style::label()),
                    Span::styled(format!("{jpt:.3}"), style::value()),
                ]));
            }
            if sum.ttft_ms.is_none() && sum.tokens_per_sec.is_none() {
                lines.push(Line::from(Span::styled(
                    "    (no stream metrics stored)",
                    style::footer(),
                )));
            }
        }
        None => {
            lines.push(Line::from(Span::styled(
                "    (no metrics available)",
                style::footer(),
            )));
        }
    }
    lines.push(Line::raw(""));

    // NIAH metrics (Engine C1).
    lines.push(Line::from(Span::styled(
        "  Engine C1: NIAH",
        style::value_ok(),
    )));
    match &h.detail_needle {
        Some((retrieved, total)) if *total > 0 => {
            let pct = *retrieved as f64 / *total as f64 * 100.0;
            lines.push(Line::from(vec![
                Span::styled("    Overall:     ", style::label()),
                Span::styled(
                    format!("{pct:.1}%  ({retrieved}/{total} retrieved)"),
                    style::value(),
                ),
            ]));
        }
        _ => {
            lines.push(Line::from(Span::styled(
                "    (no NIAH evaluations stored)",
                style::footer(),
            )));
        }
    }
    lines.push(Line::raw(""));

    // Reasoning / Structured (not yet in the DB schema).
    lines.push(Line::from(Span::styled(
        "  Engine C2: Reasoning",
        style::value_ok(),
    )));
    lines.push(Line::from(Span::styled(
        "    (not stored in this version)",
        style::footer(),
    )));
    lines.push(Line::from(Span::styled(
        "  Engine C3: Structured",
        style::value_ok(),
    )));
    lines.push(Line::from(Span::styled(
        "    (not stored in this version)",
        style::footer(),
    )));
    lines.push(Line::raw(""));
    lines.push(Line::from(Span::styled(
        "  [Esc] Back to list",
        style::footer(),
    )));

    f.render_widget(
        Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
        area,
    );
}

/// Compare mode, selecting the second run.
fn render_compare_select(area: Rect, h: &HistoryState, first: usize, f: &mut Frame) {
    let title = format!("COMPARE — Run [{}] selected, pick a second run", first + 1);
    let block = theme::block(
        Line::from(Span::styled(title, style::title())),
        style::active_border(),
    );

    let rows: Vec<Row> = h
        .sessions
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let ts = s.timestamp.as_deref().unwrap_or("--");
            let ts_short = if ts.len() >= 16 { &ts[..16] } else { ts };
            let mut marks = String::new();
            if i == first {
                marks.push_str(" A");
            }
            if i == h.cursor && i != first {
                marks.push_str(" >");
            }
            let is_cursor = i == h.cursor && i != first;
            let is_first = i == first;
            let row_style = if is_cursor {
                Style::default().fg(palette::ACCENT)
            } else if is_first {
                Style::default().fg(palette::OK)
            } else {
                Style::default()
            };
            let prefix = if is_cursor { "> " } else { "  " };
            Row::new(vec![
                Cell::from(format!("{prefix}[{}]{}", i + 1, marks)).style(row_style),
                Cell::from(ts_short.to_string()).style(row_style),
                Cell::from(fmt::truncate(&s.model_name, 20)).style(row_style),
                Cell::from(fmt::truncate(&s.target_url, 28)).style(row_style),
            ])
        })
        .collect();

    let hints = Line::from(Span::styled(
        "  ↑↓/jk navigate │ Enter select │ Esc cancel",
        style::footer(),
    ));

    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);

    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Length(10),
                Constraint::Length(17),
                Constraint::Percentage(30),
                Constraint::Percentage(43),
            ],
        )
        .header(
            Row::new(vec![
                Cell::from(""),
                Cell::from("Date"),
                Cell::from("Model"),
                Cell::from("Endpoint"),
            ])
            .style(style::muted_title()),
        )
        .block(block),
        sub[0],
    );
    f.render_widget(Paragraph::new(hints), sub[1]);
}

/// Compare mode, both runs selected — show the diff table.
fn render_compare(area: Rect, h: &HistoryState, first: usize, second: usize, f: &mut Frame) {
    let sa = h.sessions.get(first);
    let sb = h.sessions.get(second);
    let title = match (sa, sb) {
        (Some(a), Some(b)) => {
            let ta = a.timestamp.as_deref().unwrap_or("--");
            let tb = b.timestamp.as_deref().unwrap_or("--");
            format!(
                "COMPARING RUN [{}] {}  vs  RUN [{}] {}",
                first + 1,
                if ta.len() >= 16 { &ta[..16] } else { ta },
                second + 1,
                if tb.len() >= 16 { &tb[..16] } else { tb },
            )
        }
        _ => "COMPARING RUNS".to_string(),
    };
    let block = theme::block(
        Line::from(Span::styled(title, style::title())),
        style::active_border(),
    );

    let Some(diff) = &h.compare_diff else {
        // No diff data (shouldn't happen if second is Some).
        let lines = vec![Line::from(Span::styled(
            "No comparison data available.",
            style::footer(),
        ))];
        f.render_widget(
            Paragraph::new(lines).block(block).wrap(Wrap { trim: true }),
            area,
        );
        return;
    };

    let mut rows: Vec<Row> = vec![Row::new(vec![
        Cell::from("Metric"),
        Cell::from(format!("Run [{}] (A)", first + 1)),
        Cell::from(format!("Run [{}] (B)", second + 1)),
        Cell::from("Change"),
    ])
    .style(style::muted_title())];

    for r in &diff.rows {
        let delta_style = match r.improved {
            Some(true) => style::value_ok(),
            Some(false) => style::value_err(),
            None => style::footer(),
        };
        let delta_text = match r.improved {
            Some(true) => format!("▲ {}", format_delta(r.delta_pct)),
            Some(false) => format!("▼ {}", format_delta(r.delta_pct)),
            None => format!("— {}", format_delta(r.delta_pct)),
        };
        rows.push(Row::new(vec![
            Cell::from(r.label).style(style::label()),
            Cell::from(format_metric(r.label, r.a)).style(style::value()),
            Cell::from(format_metric(r.label, r.b)).style(style::value()),
            Cell::from(delta_text).style(delta_style),
        ]));
    }

    let info_note = Line::from(Span::styled(
        "  ℹ Green = improvement, Red = regression, — = no change. TTFT & J/token improve when lower.",
        style::info(),
    ));
    let back_hint = Line::from(Span::styled("  [Esc] Back to list", style::footer()));

    let sub = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .split(area);

    f.render_widget(
        Table::new(
            rows,
            [
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
                Constraint::Percentage(25),
            ],
        )
        .block(block),
        sub[0],
    );
    f.render_widget(Paragraph::new(info_note), sub[1]);
    f.render_widget(Paragraph::new(back_hint), sub[2]);
}

/// Delete confirmation overlay.
fn render_delete_confirm(area: Rect, h: &HistoryState, idx: usize, f: &mut Frame) {
    let block = theme::block(
        theme::panel_title("DELETE RUN?"),
        Style::default()
            .fg(palette::WARN)
            .add_modifier(ratatui::style::Modifier::BOLD),
    );
    let s = h.sessions.get(idx);
    let desc = match s {
        Some(s) => {
            let ts = s.timestamp.as_deref().unwrap_or("--");
            format!(
                "  Delete run [{}]  {}  {}  ?",
                idx + 1,
                if ts.len() >= 16 { &ts[..16] } else { ts },
                fmt::truncate(&s.model_name, 24),
            )
        }
        None => "  Delete this run?".to_string(),
    };
    let lines = vec![
        Line::raw(""),
        Line::from(Span::styled(desc, Style::default().fg(palette::WARN))),
        Line::raw(""),
        Line::from(vec![
            Span::styled("  [y] Delete", style::value_err()),
            Span::styled("   [n/Esc] Cancel", style::value()),
        ]),
    ];
    // Center the confirmation box.
    const W: u16 = 50;
    const H: u16 = 8;
    let w = W.min(area.width);
    let hgt = H.min(area.height);
    let x = area.x + (area.width.saturating_sub(w)) / 2;
    let y = area.y + (area.height.saturating_sub(hgt)) / 2;
    let overlay = Rect::new(x, y, w, hgt);
    f.render_widget(Paragraph::new(lines).block(block), overlay);
}

// ── formatting helpers ──────────────────────────────────────────────────

/// Per-metric value formatting (the dashboard's `--` N/A marker).
fn format_metric(label: &str, v: Option<f64>) -> String {
    let Some(v) = v else {
        return "--".to_string();
    };
    match label {
        "TTFT (ms)" => format!("{v:.1} ms"),
        "Gen t/s" => fmt::format_rate(v),
        "Prompt t/s" => fmt::format_rate(v),
        "MTP rate" => format!("{v:.2} x"),
        "J/token" => format!("{v:.3} J/tok"),
        _ => format!("{v:.3}"),
    }
}

/// Signed % change: `+12.5%` / `-20.0%` / `--` when not computable.
fn format_delta(d: Option<f64>) -> String {
    d.map(|d| format!("{d:+.1}%"))
        .unwrap_or_else(|| "--".to_string())
}

// ── tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn session(id: &str, model: &str) -> BenchmarkSession {
        BenchmarkSession {
            session_id: id.to_string(),
            timestamp: Some("2026-09-29 22:00:00".to_string()),
            target_url: "http://127.0.0.1:8000/v1/chat/completions".to_string(),
            model_name: model.to_string(),
            backend_type: Some("vllm".to_string()),
            quantization: None,
            system_gpu: None,
            total_duration_sec: Some(10.0),
        }
    }

    fn metric(
        session_id: &str,
        ttft_ms: Option<f64>,
        tpot_ms: Option<f64>,
        mtp: Option<f64>,
        joules: Option<f64>,
    ) -> StreamMetricRow {
        StreamMetricRow {
            metric_id: None,
            session_id: session_id.to_string(),
            concurrency_level: Some(1),
            prompt_tokens: None,
            completion_tokens: None,
            reasoning_tokens: None,
            ttft_ms,
            tpot_ms,
            mtp_efficiency: mtp,
            joules_per_token: joules,
            cache_hit: None,
        }
    }

    fn metric_with_prompt(
        session_id: &str,
        prompt_tokens: Option<i64>,
        ttft_ms: Option<f64>,
        tpot_ms: Option<f64>,
        mtp: Option<f64>,
    ) -> StreamMetricRow {
        StreamMetricRow {
            metric_id: None,
            session_id: session_id.to_string(),
            concurrency_level: Some(1),
            prompt_tokens,
            completion_tokens: None,
            reasoning_tokens: None,
            ttft_ms,
            tpot_ms,
            mtp_efficiency: mtp,
            joules_per_token: None,
            cache_hit: None,
        }
    }

    // ---- session_summary ----

    #[test]
    fn summary_averages_rows_and_derives_tps_from_tpot() {
        let rows = vec![
            metric("s", Some(200.0), Some(20.0), Some(1.0), None),
            metric("s", Some(100.0), Some(10.0), Some(3.0), None),
        ];
        let sum = session_summary(&rows);
        assert_eq!(sum.ttft_ms, Some(150.0));
        // tps = mean(1000/20, 1000/10) = mean(50, 100) = 75.
        assert_eq!(sum.tokens_per_sec, Some(75.0));
        assert_eq!(sum.mtp, Some(2.0));
        assert_eq!(sum.joules_per_token, None); // NULLs skip, don't zero
    }

    #[test]
    fn summary_derives_prompt_tps() {
        // 2000 prompt tokens in 200 ms → 10,000 t/s.
        let rows = vec![metric_with_prompt(
            "s",
            Some(2000),
            Some(200.0),
            Some(20.0),
            None,
        )];
        let sum = session_summary(&rows);
        assert_eq!(sum.prompt_tps, Some(10_000.0));
    }

    #[test]
    fn summary_of_no_rows_is_all_none() {
        let sum = session_summary(&[]);
        assert_eq!(sum, SessionSummary::default());
    }

    #[test]
    fn summary_ignores_nonpositive_tpot() {
        let rows = vec![metric("s", Some(50.0), Some(0.0), Some(1.0), None)];
        assert_eq!(session_summary(&rows).tokens_per_sec, None);
    }

    // ---- diff_row ----

    #[test]
    fn lower_ttft_is_an_improvement_with_negative_delta() {
        let r = diff_row("TTFT (ms)", Some(200.0), Some(160.0), true);
        assert_eq!(r.delta_pct, Some(-20.0));
        assert_eq!(r.improved, Some(true));
    }

    #[test]
    fn higher_tps_and_mtp_are_improvements() {
        let t = diff_row("Gen t/s", Some(50.0), Some(62.5), false);
        assert_eq!(t.delta_pct, Some(25.0));
        assert_eq!(t.improved, Some(true));
        let m = diff_row("MTP rate", Some(1.0), Some(1.25), false);
        assert_eq!(m.delta_pct, Some(25.0));
        assert_eq!(m.improved, Some(true));
    }

    #[test]
    fn slower_run_is_a_regression() {
        let r = diff_row("TTFT (ms)", Some(200.0), Some(240.0), true);
        assert_eq!(r.delta_pct, Some(20.0));
        assert_eq!(r.improved, Some(false));
    }

    #[test]
    fn equal_values_are_neutral() {
        let r = diff_row("MTP rate", Some(1.0), Some(1.0), false);
        assert_eq!(r.delta_pct, Some(0.0));
        assert_eq!(r.improved, None);
    }

    #[test]
    fn missing_or_zero_baselines_yield_no_delta() {
        assert_eq!(diff_row("J/token", None, Some(0.3), true).delta_pct, None);
        assert_eq!(diff_row("J/token", Some(0.3), None, true).delta_pct, None);
        assert_eq!(
            diff_row("Gen t/s", Some(0.0), Some(5.0), false).delta_pct,
            None
        );
        assert_eq!(diff_row("MTP rate", None, None, false).improved, None);
    }

    // ---- DiffReport::from_sessions ----

    #[test]
    fn from_sessions_produces_the_blueprint_delta_set() {
        let a = session("aaaa", "vllm-0.6");
        let b = session("bbbb", "vllm-0.7");
        let report = DiffReport::from_sessions(
            &a,
            &[metric("aaaa", Some(200.0), Some(20.0), Some(1.0), None)],
            &b,
            &[metric("bbbb", Some(160.0), Some(16.0), Some(1.25), None)],
        );
        assert_eq!(report.rows.len(), 5);
        assert_eq!(report.rows[0].label, "TTFT (ms)");
        assert_eq!(report.rows[0].delta_pct, Some(-20.0));
        assert_eq!(report.rows[0].improved, Some(true));
        assert_eq!(report.rows[1].delta_pct, Some(25.0)); // 50 → 62.5 t/s
        assert_eq!(report.rows[1].improved, Some(true));
        assert_eq!(report.rows[3].delta_pct, Some(25.0)); // 1.0 → 1.25 x
        assert_eq!(report.rows[3].improved, Some(true));
        // J/token: no hardware telemetry on either side → not comparable.
        assert_eq!(report.rows[4].delta_pct, None);
        assert_eq!(report.rows[4].improved, None);
    }

    // ---- formatting ----

    #[test]
    fn delta_formats_signed() {
        assert_eq!(format_delta(Some(12.5)), "+12.5%");
        assert_eq!(format_delta(Some(-20.0)), "-20.0%");
        assert_eq!(format_delta(None), "--");
    }

    // ---- HistoryMode ----

    #[test]
    fn mode_defaults_to_list() {
        assert_eq!(HistoryMode::default(), HistoryMode::List);
    }

    #[test]
    fn mode_clone_and_compare() {
        let m1 = HistoryMode::Detail(3);
        let m2 = HistoryMode::Detail(3);
        assert_eq!(m1, m2);
        let m3 = HistoryMode::Compare {
            first: 0,
            second: Some(1),
        };
        assert_ne!(m1, m3);
    }

    // ---- HistoryState navigation ----

    fn state_with_sessions(n: usize) -> HistoryState {
        let sessions: Vec<BenchmarkSession> = (0..n)
            .map(|i| session(&format!("s{i:04}"), &format!("model-{i}")))
            .collect();
        HistoryState {
            db_path: std::path::PathBuf::from("/nonexistent/test.db"),
            sessions,
            cursor: 0,
            mode: HistoryMode::List,
            ..Default::default()
        }
    }

    #[test]
    fn cursor_moves_down_and_up_clamped() {
        let mut h = state_with_sessions(3);
        h.move_cursor(1);
        assert_eq!(h.cursor, 1);
        h.move_cursor(1);
        assert_eq!(h.cursor, 2);
        h.move_cursor(1); // clamp at end
        assert_eq!(h.cursor, 2);
        h.move_cursor(-1);
        assert_eq!(h.cursor, 1);
        h.move_cursor(-1);
        assert_eq!(h.cursor, 0);
        h.move_cursor(-1); // clamp at start
        assert_eq!(h.cursor, 0);
    }

    #[test]
    fn cursor_noop_on_empty_list() {
        let mut h = HistoryState::default();
        h.move_cursor(1);
        assert_eq!(h.cursor, 0);
    }

    #[test]
    fn enter_detail_sets_mode() {
        let mut h = state_with_sessions(2);
        h.cursor = 1;
        // Can't actually open the DB in a unit test, so just verify
        // the mode transition logic by checking the guard.
        // (The real DB call would fail silently and leave mode unchanged.)
        h.enter_detail();
        // With a non-existent DB path, the mode should stay List
        // because the DB open fails.
        assert_eq!(h.mode, HistoryMode::List);
    }

    #[test]
    fn enter_compare_sets_mode() {
        let mut h = state_with_sessions(2);
        h.cursor = 0;
        h.enter_compare();
        assert_eq!(
            h.mode,
            HistoryMode::Compare {
                first: 0,
                second: None
            }
        );
    }

    #[test]
    fn enter_compare_noop_on_empty() {
        let mut h = HistoryState::default();
        h.enter_compare();
        assert_eq!(h.mode, HistoryMode::List);
    }

    #[test]
    fn select_compare_second_skips_same_run() {
        let mut h = state_with_sessions(2);
        h.mode = HistoryMode::Compare {
            first: 0,
            second: None,
        };
        h.cursor = 0; // same as first
        h.select_compare_second();
        // Should not have changed (can't compare with self).
        assert_eq!(
            h.mode,
            HistoryMode::Compare {
                first: 0,
                second: None
            }
        );
    }

    #[test]
    fn back_to_list_resets_mode_and_caches() {
        let mut h = state_with_sessions(2);
        h.mode = HistoryMode::Detail(0);
        h.detail_summary = Some(SessionSummary::default());
        h.back_to_list();
        assert_eq!(h.mode, HistoryMode::List);
        assert!(h.detail_summary.is_none());
    }

    #[test]
    fn enter_delete_confirm_sets_mode() {
        let mut h = state_with_sessions(2);
        h.cursor = 1;
        h.enter_delete_confirm();
        assert_eq!(h.mode, HistoryMode::DeleteConfirm(1));
    }

    #[test]
    fn delete_confirm_noop_on_empty() {
        let mut h = HistoryState::default();
        h.enter_delete_confirm();
        assert_eq!(h.mode, HistoryMode::List);
    }

    // ---- rendering: empty state ----

    fn render_history_text(app: &crate::ui::app::App, w: u16, h: u16) -> String {
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
    fn empty_state_when_no_history_loaded() {
        let app = crate::ui::app::App::new();
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("No benchmark runs saved yet"), "{text}");
    }

    #[test]
    fn empty_state_when_sessions_empty() {
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![],
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("No benchmark runs saved yet"), "{text}");
    }

    #[test]
    fn list_mode_renders_sessions() {
        let a = session("aaaa", "model-a");
        let b = session("bbbb", "model-b");
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a, b],
            cursor: 0,
            mode: HistoryMode::List,
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("HISTORY"), "{text}");
        assert!(text.contains("model-a"), "{text}");
        assert!(text.contains("model-b"), "{text}");
        assert!(text.contains("2 total"), "{text}");
    }

    #[test]
    fn list_mode_shows_cursor_marker() {
        let a = session("aaaa", "model-a");
        let b = session("bbbb", "model-b");
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a, b],
            cursor: 1,
            mode: HistoryMode::List,
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("> [2]"), "cursor on row 2: {text}");
    }

    #[test]
    fn detail_mode_renders() {
        let a = session("aaaa", "model-a");
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a],
            cursor: 0,
            mode: HistoryMode::Detail(0),
            detail_summary: Some(SessionSummary {
                ttft_ms: Some(250.0),
                tokens_per_sec: Some(65.1),
                prompt_tps: Some(2400.0),
                mtp: Some(1.5),
                joules_per_token: None,
            }),
            detail_needle: Some((36, 77)),
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("RUN DETAILS"), "{text}");
        assert!(text.contains("model-a"), "{text}");
        assert!(text.contains("250.0 ms"), "{text}");
        assert!(text.contains("65.1 t/s"), "{text}");
        assert!(text.contains("46.8%"), "NIAH 36/77: {text}");
        assert!(text.contains("[Esc] Back to list"), "{text}");
    }

    #[test]
    fn compare_mode_renders_diff_table() {
        let a = session("aaaa", "model-a");
        let b = session("bbbb", "model-b");
        let diff = DiffReport::from_sessions(
            &a,
            &[metric("aaaa", Some(200.0), Some(20.0), Some(1.0), None)],
            &b,
            &[metric("bbbb", Some(160.0), Some(16.0), Some(1.25), None)],
        );
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a, b],
            cursor: 1,
            mode: HistoryMode::Compare {
                first: 0,
                second: Some(1),
            },
            compare_diff: Some(diff),
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("COMPARING RUN"), "{text}");
        assert!(text.contains("-20.0%"), "TTFT delta: {text}");
        assert!(text.contains("[Esc] Back to list"), "{text}");
    }

    #[test]
    fn compare_select_mode_renders() {
        let a = session("aaaa", "model-a");
        let b = session("bbbb", "model-b");
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a, b],
            cursor: 1,
            mode: HistoryMode::Compare {
                first: 0,
                second: None,
            },
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("COMPARE"), "{text}");
        assert!(text.contains("pick a second run"), "{text}");
    }

    #[test]
    fn delete_confirm_renders() {
        let a = session("aaaa", "model-a");
        let mut app = crate::ui::app::App::new();
        app.history = Some(HistoryState {
            sessions: vec![a],
            cursor: 0,
            mode: HistoryMode::DeleteConfirm(0),
            ..Default::default()
        });
        let text = render_history_text(&app, 120, 40);
        assert!(text.contains("DELETE RUN?"), "{text}");
        assert!(text.contains("[y] Delete"), "{text}");
        assert!(text.contains("[n/Esc] Cancel"), "{text}");
    }
}
