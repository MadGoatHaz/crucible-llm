//! View 4 — Historical Comparison & Regression Diffing (blueprint §6):
//! side-by-side comparison of two stored runs (e.g. `vLLM v0.6.x` vs
//! `vLLM v0.7.x`, `llama.cpp b3200` vs `b3300`) with signed, color-coded
//! delta metrics — % change in TTFT, tokens/sec, and speculative
//! verification (MTP) rate, plus J/token once hardware telemetry lands
//! (Chunk 17). Gains render green, regressions red.
//!
//! Data source: the SQLite storage layer (Chunk 12) — stored
//! `benchmark_sessions` and their `stream_metrics` rows. The session
//! list is loaded once per selection change (user-driven, on the key
//! path) and the diff is precomputed into [`HistoryState`]; the render
//! itself is a pure function of `&App` state, so the 60 Hz render path
//! never touches the DB (measurement-isolation invariant, blueprint §4).
//!
//! Keys (active in this view only): `j` / `↓` and `k` / `↑` move the
//! session cursor; `a` sets the cursor's session as run A; `b` sets it as
//! run B. With both selected, the delta panel shows the signed % changes.

use std::path::Path;

use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::Style;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Cell, Paragraph, Row, Table};
use ratatui::Frame;

use crate::storage::db::{Database, StorageError};
use crate::storage::models::{BenchmarkSession, StreamMetricRow};
use crate::ui::app::App;
use crate::ui::theme::{palette, style};

// ── pure diff computation (no locks, no I/O) ────────────────────────────

/// Per-metric summary of one stored session: the mean over its
/// `stream_metrics` rows (rows lacking a value are skipped, so a NULL
/// `joules_per_token` before Chunk 17 never skews the others).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SessionSummary {
    /// Mean TTFT, milliseconds.
    pub ttft_ms: Option<f64>,
    /// Mean decode speed, tokens/sec (derived from `tpot_ms`).
    pub tokens_per_sec: Option<f64>,
    /// Mean MTP η (tokens per content chunk).
    pub mtp: Option<f64>,
    /// Mean silicon efficiency (Chunk 17; `None` without GPU telemetry).
    pub joules_per_token: Option<f64>,
}

/// Aggregate a session's `stream_metrics` rows into its summary.
pub fn session_summary(metrics: &[StreamMetricRow]) -> SessionSummary {
    let ttft: Vec<f64> = metrics.iter().filter_map(|m| m.ttft_ms).collect();
    // Decode speed from the stored TPOT: `1000 / tpot_ms` per row, then
    // mean (a row with `tpot_ms <= 0` carries no speed information).
    let tps: Vec<f64> = metrics
        .iter()
        .filter_map(|m| m.tpot_ms)
        .filter(|v| *v > 0.0)
        .map(|v| 1000.0 / v)
        .collect();
    let mtp: Vec<f64> = metrics.iter().filter_map(|m| m.mtp_efficiency).collect();
    let joules: Vec<f64> = metrics.iter().filter_map(|m| m.joules_per_token).collect();
    SessionSummary {
        ttft_ms: mean(&ttft),
        tokens_per_sec: mean(&tps),
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
    /// (`None` when not computable — missing on either side, or a
    /// zero baseline).
    pub delta_pct: Option<f64>,
    /// `Some(true)` = improvement (green), `Some(false)` = regression
    /// (red), `None` = neutral / not comparable.
    pub improved: Option<bool>,
}

/// Compute one metric's delta row.
///
/// `lower_is_better` flips the improvement test: TTFT and J/token improve
/// when they go *down*; tokens/sec and MTP rate improve when they go *up*.
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
                "Tokens/s",
                summary_a.tokens_per_sec,
                summary_b.tokens_per_sec,
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
/// first), the cursor, and the A/B selection with its precomputed diff.
///
/// `db_path` is remembered from the load; each selection change re-opens
/// the DB briefly to fetch the selected runs' `stream_metrics` rows —
/// user-driven and rare, never on the render path.
#[derive(Debug, Default)]
pub struct HistoryState {
    pub db_path: std::path::PathBuf,
    pub sessions: Vec<BenchmarkSession>,
    pub cursor: usize,
    pub run_a: Option<usize>,
    pub run_b: Option<usize>,
    pub summary_a: Option<SessionSummary>,
    pub summary_b: Option<SessionSummary>,
    pub diff: Option<DiffReport>,
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
            run_a: None,
            run_b: None,
            summary_a: None,
            summary_b: None,
            diff: None,
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

    /// Set the cursor's session as run A and refresh the diff.
    pub fn select_a(&mut self) {
        if self.cursor < self.sessions.len() {
            self.run_a = Some(self.cursor);
            self.refresh();
        }
    }

    /// Set the cursor's session as run B and refresh the diff.
    pub fn select_b(&mut self) {
        if self.cursor < self.sessions.len() {
            self.run_b = Some(self.cursor);
            self.refresh();
        }
    }

    /// Re-fetch the selected runs' stored metrics and recompute the diff.
    ///
    /// A storage failure clears the summaries/diff — the view falls back
    /// to its placeholder text and never panics the render loop.
    fn refresh(&mut self) {
        self.summary_a = None;
        self.summary_b = None;
        self.diff = None;
        let db = match Database::open(&self.db_path) {
            Ok(db) => db,
            Err(_) => return,
        };
        let mut metrics_a: Option<Vec<StreamMetricRow>> = None;
        let mut metrics_b: Option<Vec<StreamMetricRow>> = None;
        if let Some(i) = self.run_a.filter(|&i| i < self.sessions.len()) {
            metrics_a = db.get_stream_metrics(&self.sessions[i].session_id).ok();
            self.summary_a = metrics_a.as_ref().map(|m| session_summary(m));
        }
        if let Some(i) = self.run_b.filter(|&i| i < self.sessions.len()) {
            metrics_b = db.get_stream_metrics(&self.sessions[i].session_id).ok();
            self.summary_b = metrics_b.as_ref().map(|m| session_summary(m));
        }
        if let (Some(a), Some(b), Some(ma), Some(mb)) =
            (self.run_a, self.run_b, metrics_a, metrics_b)
        {
            self.diff = Some(DiffReport::from_sessions(
                &self.sessions[a],
                &ma,
                &self.sessions[b],
                &mb,
            ));
        }
    }
}

// ── rendering (pure `&App` read) ────────────────────────────────────────

/// Render the History Diff view into `area`.
pub fn render(area: Rect, app: &App, f: &mut Frame) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Percentage(45), Constraint::Percentage(55)])
        .split(area);
    render_runs(chunks[0], app, f);
    let bottom = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(55), Constraint::Percentage(45)])
        .split(chunks[1]);
    render_session_list(bottom[0], app, f);
    render_delta(bottom[1], app, f);
}

/// Side-by-side run panels (A left, B right).
fn render_runs(area: Rect, app: &App, f: &mut Frame) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(50), Constraint::Percentage(50)])
        .split(area);
    let h = app.history.as_ref();
    let (a_idx, b_idx) = h.map(|h| (h.run_a, h.run_b)).unwrap_or((None, None));
    let (a_sum, b_sum) = h
        .map(|h| (h.summary_a.clone(), h.summary_b.clone()))
        .unwrap_or((None, None));
    render_run_panel(cols[0], "RUN A", h, a_idx, a_sum, f);
    render_run_panel(cols[1], "RUN B", h, b_idx, b_sum, f);
}

fn render_run_panel(
    area: Rect,
    title: &str,
    h: Option<&HistoryState>,
    idx: Option<usize>,
    summary: Option<SessionSummary>,
    f: &mut Frame,
) {
    let subtitle = match (h, idx) {
        (Some(h), Some(i)) if i < h.sessions.len() => {
            let s = &h.sessions[i];
            format!(
                "{}  ·  {}  ·  {}",
                &s.session_id[..s.session_id.len().min(8)],
                s.timestamp.as_deref().unwrap_or("--"),
                s.model_name
            )
        }
        _ => "no run selected".to_string(),
    };
    let mut rows: Vec<Row> =
        vec![Row::new(vec![Cell::from("METRIC"), Cell::from("VALUE")]).style(style::muted_title())];
    for (label, value, value_style) in [
        (
            "TTFT",
            format_metric("TTFT (ms)", summary.as_ref().and_then(|s| s.ttft_ms)),
            style::value(),
        ),
        (
            "Tokens/s",
            format_metric("Tokens/s", summary.as_ref().and_then(|s| s.tokens_per_sec)),
            style::value(),
        ),
        (
            "MTP rate",
            format_metric("MTP rate", summary.as_ref().and_then(|s| s.mtp)),
            style::highlight(),
        ),
        (
            "J/token",
            format_metric("J/token", summary.as_ref().and_then(|s| s.joules_per_token)),
            style::value(),
        ),
    ] {
        rows.push(Row::new(vec![
            Cell::from(label).style(style::label()),
            Cell::from(value).style(value_style),
        ]));
    }
    f.render_widget(
        Table::new(
            rows,
            [Constraint::Percentage(50), Constraint::Percentage(50)],
        )
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(style::border())
                .title(Line::from(vec![
                    Span::styled(format!("{title}  "), style::title()),
                    Span::styled(subtitle, style::footer()),
                ])),
        ),
        area,
    );
}

/// The stored-session list with the A/B selection markers and cursor.
fn render_session_list(area: Rect, app: &App, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("STORED SESSIONS (newest first)   [j/k] move · [a]=Run A · [b]=Run B");
    match app.history.as_ref() {
        None => {
            let lines = vec![
                Line::from(Span::styled("No stored runs found.", style::footer())),
                Line::from(Span::styled(
                    "Run a headless benchmark (--url …) to populate the history.",
                    style::footer(),
                )),
            ];
            f.render_widget(Paragraph::new(lines).block(block), area);
        }
        Some(h) if h.sessions.is_empty() => {
            let lines = vec![Line::from(Span::styled(
                "No stored sessions yet — run a headless benchmark to populate the history.",
                style::footer(),
            ))];
            f.render_widget(Paragraph::new(lines).block(block), area);
        }
        Some(h) => {
            let rows: Vec<Row> = h
                .sessions
                .iter()
                .enumerate()
                .map(|(i, s)| {
                    let mut mark_spans: Vec<Span> = Vec::new();
                    if h.run_a == Some(i) {
                        mark_spans.push(Span::styled("A ", style::value_ok()));
                    }
                    if h.run_b == Some(i) {
                        mark_spans.push(Span::styled("B", style::highlight()));
                    }
                    let marks_cell = if mark_spans.is_empty() {
                        Cell::from("  ").style(Style::default().fg(palette::MUTED))
                    } else {
                        Cell::from(Line::from(mark_spans))
                    };
                    Row::new(vec![
                        Cell::from(format!("#{}", i + 1)).style(style::footer()),
                        marks_cell,
                        Cell::from(s.timestamp.as_deref().unwrap_or("--")).style(style::label()),
                        Cell::from(s.model_name.clone()).style(style::value()),
                        Cell::from(shorten(&s.target_url, 24)).style(style::footer()),
                    ])
                })
                .collect();
            f.render_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Length(4),
                        Constraint::Length(4),
                        Constraint::Length(20),
                        Constraint::Percentage(35),
                        Constraint::Percentage(37),
                    ],
                )
                .highlight_symbol("▶ ")
                .row_highlight_style(Style::default().fg(palette::ACCENT))
                .block(block),
                area,
            );
        }
    }
}

/// Delta panel: signed % change per metric, green = gain, red = regression.
fn render_delta(area: Rect, app: &App, f: &mut Frame) {
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(style::border())
        .title("DELTA (signed % change, B vs A)");
    match app.history.as_ref().and_then(|h| h.diff.as_ref()) {
        None => {
            let lines = vec![
                Line::from(Span::styled(
                    "Select two stored sessions to diff (e.g. vLLM v0.6.x vs v0.7.x).",
                    style::footer(),
                )),
                Line::from(Span::styled(
                    "Gains render green, regressions red.",
                    style::footer(),
                )),
            ];
            f.render_widget(Paragraph::new(lines).block(block), area);
        }
        Some(diff) => {
            let mut rows: Vec<Row> = vec![Row::new(vec![
                Cell::from("METRIC"),
                Cell::from("RUN A"),
                Cell::from("RUN B"),
                Cell::from("Δ %"),
            ])
            .style(style::muted_title())];
            for r in &diff.rows {
                let delta_style = match r.improved {
                    Some(true) => style::value_ok(),
                    Some(false) => style::value_err(),
                    None => style::footer(),
                };
                rows.push(Row::new(vec![
                    Cell::from(r.label).style(style::label()),
                    Cell::from(format_metric(r.label, r.a)).style(style::value()),
                    Cell::from(format_metric(r.label, r.b)).style(style::value()),
                    Cell::from(format_delta(r.delta_pct)).style(delta_style),
                ]));
            }
            f.render_widget(
                Table::new(
                    rows,
                    [
                        Constraint::Percentage(35),
                        Constraint::Percentage(25),
                        Constraint::Percentage(25),
                        Constraint::Percentage(15),
                    ],
                )
                .block(block),
                area,
            );
        }
    }
}

// ── formatting helpers ──────────────────────────────────────────────────

/// Per-metric value formatting (the dashboard's `--` N/A marker).
fn format_metric(label: &str, v: Option<f64>) -> String {
    let Some(v) = v else {
        return "--".to_string();
    };
    match label {
        "TTFT (ms)" => format!("{v:.1} ms"),
        "Tokens/s" => format!("{v:.1} t/s"),
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

/// Truncate a URL to `max` chars (byte-safe: back off to a char boundary).
fn shorten(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}…", &s[..end])
}

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
        let t = diff_row("Tokens/s", Some(50.0), Some(62.5), false);
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
            diff_row("Tokens/s", Some(0.0), Some(5.0), false).delta_pct,
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
        assert_eq!(report.rows.len(), 4);
        assert_eq!(report.rows[0].label, "TTFT (ms)");
        assert_eq!(report.rows[0].delta_pct, Some(-20.0));
        assert_eq!(report.rows[0].improved, Some(true));
        assert_eq!(report.rows[1].delta_pct, Some(25.0)); // 50 → 62.5 t/s
        assert_eq!(report.rows[1].improved, Some(true));
        assert_eq!(report.rows[2].delta_pct, Some(25.0)); // 1.0 → 1.25 x
        assert_eq!(report.rows[2].improved, Some(true));
        // J/token: no hardware telemetry on either side → not comparable.
        assert_eq!(report.rows[3].delta_pct, None);
        assert_eq!(report.rows[3].improved, None);
    }

    // ---- formatting ----

    #[test]
    fn delta_formats_signed() {
        assert_eq!(format_delta(Some(12.5)), "+12.5%");
        assert_eq!(format_delta(Some(-20.0)), "-20.0%");
        assert_eq!(format_delta(None), "--");
    }

    #[test]
    fn shorten_respects_char_boundaries() {
        assert_eq!(shorten("abc", 10), "abc");
        assert_eq!(shorten("abcdefghijklmnop", 5), "abcde…");
        // Multi-byte boundary (each é is 2 bytes): back off to a valid
        // boundary — 5 bytes lands mid-é, so the cut is at 4.
        let s = "ééééé";
        assert_eq!(shorten(s, 5), "éé…");
    }
}
