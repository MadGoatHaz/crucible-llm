//! The **Run Logger**: a file-based, timestamped diagnostic log of every
//! benchmark run.
//!
//! The log is written to two files in the run-log directory (default
//! `~/.local/share/crucible/logs/`, configurable via `--log-dir` /
//! `CRUCIBLE_LOG_DIR` / the config file):
//!
//! * `latest.log` — **overwritten on every run** (the file to review after
//!   the app runs);
//! * `run-<YYYYmmdd-HHMMSS>.log` — an **archive** copy for this run.
//!
//! Line format: `[HH:MM:SS.mmm] [LEVEL] [CONTEXT] message`
//!
//! Design constraints (the log must never slow down the benchmark):
//!
//! * **thread-safe** — the logger is an `Arc<RunLogger>`; every
//!   `info`/`warn`/`error`/`debug` call is a non-blocking `send` on an
//!   unbounded channel from any thread / task;
//! * **non-blocking** — a dedicated writer thread formats and writes the
//!   records (to both files, flushing on every newline). A logger whose
//!   files cannot be created degrades to a silent no-op
//!   ([`RunLogger::disabled`]) — logging never breaks a run;
//! * **simple** — one struct, four level methods, a [`Context`] tag, and
//!   a [`Level`] (INFO / WARN / ERROR / DEBUG).
//!
//! Callers are expected to keep the rate sane: the stream worker logs the
//! first token, every 50th token, and the completion — not every frame —
//! so a 64-stream level produces at most a few dozen lines in a minute.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::config::Config;
use crate::engines::human_size;

/// A log level (the `[LEVEL]` field of each line).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Level {
    /// Diagnostic detail (rarely used outside of `--verbose`-style runs).
    Debug,
    /// Normal operational events (request sent, stream complete, …).
    Info,
    /// A degraded but recoverable condition (stall, partial timeout, …).
    Warn,
    /// A failure (HTTP error, worker killed, run aborted, …).
    Error,
}

impl Level {
    /// The `[LEVEL]` label.
    pub fn label(self) -> &'static str {
        match self {
            Level::Debug => "DEBUG",
            Level::Info => "INFO",
            Level::Warn => "WARN",
            Level::Error => "ERROR",
        }
    }
}

/// The subsystem that produced a line (the `[CONTEXT]` field).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Context {
    /// Setup / connection phase (URL entered, model selected, config).
    Setup,
    /// Model discovery (`GET {base}/models`).
    Discovery,
    /// Engine A — Speed & Latency.
    EngineA,
    /// Engine B — Concurrency & Saturation sweep.
    EngineB,
    /// Engine C1 — Needle-in-a-Haystack.
    EngineC1,
    /// Engine C2 — Deterministic reasoning.
    EngineC2,
    /// Engine C3 — Structured output.
    EngineC3,
    /// Engine D — Hardware & Energy.
    EngineD,
    /// Engine F — Flat Out (sustained max-speed, one continuous 60s
    /// stream).
    EngineF,
    /// The Benchmark Sequence executor (engine transitions).
    Sequence,
    /// HTTP request/response lifecycle (the stream worker).
    Http,
    /// SSE stream lifecycle (token reception, stream completion).
    Sse,
    /// The TUI (app lifecycle events).
    Tui,
    /// Unstructured / catch-all errors.
    Error,
}

impl Context {
    /// The `[CONTEXT]` label.
    pub fn label(self) -> &'static str {
        match self {
            Context::Setup => "SETUP",
            Context::Discovery => "DISCOVERY",
            Context::EngineA => "ENGINE_A",
            Context::EngineB => "ENGINE_B",
            Context::EngineC1 => "ENGINE_C1",
            Context::EngineC2 => "ENGINE_C2",
            Context::EngineC3 => "ENGINE_C3",
            Context::EngineD => "ENGINE_D",
            Context::EngineF => "ENGINE_F",
            Context::Sequence => "SEQUENCE",
            Context::Http => "HTTP",
            Context::Sse => "SSE",
            Context::Tui => "TUI",
            Context::Error => "ERROR",
        }
    }
}

impl std::fmt::Display for Context {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

/// One item on the logger's channel.
enum Msg {
    /// A log record. `at_ms` is the wall-clock time (ms since the Unix
    /// epoch) captured at the *send* site, so queued records keep their
    /// original timestamps even if the writer is briefly busy.
    Record {
        at_ms: u64,
        level: Level,
        ctx: Context,
        msg: String,
    },
    /// Shut the writer down (flush both files and exit the thread).
    Shutdown,
}

/// The shared run logger (see the module docs).
///
/// Cheap to clone-share as `Arc<RunLogger>`; every method is a
/// non-blocking channel send. A `disabled` logger (no files) silently
/// drops all records.
pub struct RunLogger {
    tx: mpsc::Sender<Msg>,
    /// `true` while the writer thread is alive and accepting records.
    active: AtomicBool,
    /// The writer thread (taken + joined on [`finish`](Self::finish) /
    /// drop; `finish` takes only `&self`, so the handle sits behind a
    /// `Mutex` that is locked once per run, never on the hot path).
    handle: std::sync::Mutex<Option<std::thread::JoinHandle<()>>>,
    /// The `latest.log` path (`None` when disabled).
    path: Option<PathBuf>,
    /// The archive path (`run-<stamp>-<session8>.log`, `None` when
    /// disabled) — the run's permanent log copy, known at start (the
    /// name is computed before the writer thread spawns).
    archive_path: Option<PathBuf>,
    /// R2 (traceability): the run's session UUID, stamped into the log
    /// header + archive name so the log joins deterministically to the
    /// `benchmark_sessions` row. `None` when the caller did not supply one.
    session: Option<String>,
}

impl std::fmt::Debug for RunLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunLogger")
            .field("active", &self.active.load(Ordering::Relaxed))
            .field("path", &self.path)
            .field("archive_path", &self.archive_path)
            .field("session", &self.session)
            .finish()
    }
}

impl RunLogger {
    /// Start a fresh run log (no session id): a thin wrapper over
    /// [`Self::start_with_session`] for callers that have no session to
    /// join the log to.
    pub fn start(log_dir: Option<&Path>) -> Arc<Self> {
        Self::start_with_session(log_dir, None)
    }

    /// Start a fresh run log: open `latest.log` (overwritten) and an
    /// archived `run-<timestamp>.log` in `log_dir`, and spawn the writer
    /// thread.
    ///
    /// R2 (traceability): when `session` is `Some(id)` (the run's UUID),
    /// it is stamped into the **header** (`(session {id})`) and the
    /// **archive file name** (`run-<timestamp>-<id[:8]>.log`), so the log
    /// joins deterministically to the `benchmark_sessions` row that
    /// carries the same id.
    ///
    /// `log_dir` defaults to `data_dir()/crucible/logs` (i.e.
    /// `~/.local/share/crucible/logs` on Linux) when `None`. Any failure
    /// (missing home dir, permissions, …) degrades to a
    /// [`disabled`](Self::disabled) logger — logging never breaks a run.
    pub fn start_with_session(log_dir: Option<&Path>, session: Option<&str>) -> Arc<Self> {
        let session = session.map(str::to_string);
        let dir = log_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| crate::config::data_dir().join("logs"));
        let now_ms = now_ms();
        // The archive name is second-granular; with a session id it is
        // `run-<timestamp>-<id[:8]>.log` (the deterministic log↔DB join),
        // else `run-<timestamp>.log`. Collisions (two runs in the same
        // second) are disambiguated with a `-2`, `-3`, … suffix.
        let stamp = archive_stamp(now_ms);
        let base = match &session {
            Some(id) => format!("run-{stamp}-{}", short_id(id)),
            None => format!("run-{stamp}"),
        };
        let mut archive_name = format!("{base}.log");
        let mut counter = 1;
        while dir.join(&archive_name).exists() {
            counter += 1;
            archive_name = format!("{base}-{counter}.log");
        }
        let (latest, archive) = match (
            std::fs::create_dir_all(&dir),
            std::fs::OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .open(dir.join("latest.log")),
        ) {
            (Ok(()), Ok(f)) => (
                f,
                match std::fs::OpenOptions::new()
                    .create(true)
                    .truncate(true)
                    .write(true)
                    .open(dir.join(&archive_name))
                {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("crucible-llm: run log archive {archive_name} unavailable: {e}");
                        return Self::disabled();
                    }
                },
            ),
            (Ok(()), Err(e)) => {
                eprintln!(
                    "crucible-llm: cannot open {}latest.log: {e} — logging disabled",
                    dir.display()
                );
                return Self::disabled();
            }
            (Err(e), _) => {
                eprintln!(
                    "crucible-llm: run log directory {} unavailable: {e} — logging disabled",
                    dir.display()
                );
                return Self::disabled();
            }
        };

        let (tx, rx) = mpsc::channel::<Msg>();
        let header_at = now_ms;
        // R2: the header carries the session id (when present) so the log
        // is joinable to its DB row even without reading the archive name.
        let header = match &session {
            Some(id) => format!(
                "=== crucible-llm run log — {} (session {}) ===",
                format_timestamp(header_at, true),
                id
            ),
            None => format!(
                "=== crucible-llm run log — {} ===",
                format_timestamp(header_at, true)
            ),
        };
        let handle = std::thread::Builder::new()
            .name("crucible-runlog".to_string())
            .spawn(move || writer_thread(rx, latest, archive, header))
            .ok();
        let logger = Arc::new(Self {
            tx,
            active: AtomicBool::new(true),
            handle: std::sync::Mutex::new(handle),
            path: Some(dir.join("latest.log")),
            archive_path: Some(dir.join(&archive_name)),
            session,
        });
        // The header is the first record the writer emits (it is not
        // sent through the channel: the writer owns it and prints it
        // before draining).
        logger
    }

    /// A no-op logger (records are silently dropped). Used when the log
    /// files cannot be created and by unit tests that never want to touch
    /// the filesystem.
    pub fn disabled() -> Arc<Self> {
        let (tx, rx) = mpsc::channel::<Msg>();
        drop(rx); // sends fail immediately → every method is a no-op
        Arc::new(Self {
            tx,
            active: AtomicBool::new(false),
            handle: std::sync::Mutex::new(None),
            path: None,
            archive_path: None,
            session: None,
        })
    }

    /// The run's session id (`None` when the logger was started without
    /// one) — exposed so callers can verify the log↔DB join.
    #[must_use]
    pub fn session_id(&self) -> Option<&str> {
        self.session.as_deref()
    }

    /// `true` while the logger is writing to its files.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// The `latest.log` path (`None` when disabled).
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// The archive path (`run-<stamp>-<session8>.log`, `None` when
    /// disabled) — the run's permanent log copy.
    pub fn archive_path(&self) -> Option<&Path> {
        self.archive_path.as_deref()
    }

    /// Log at [`Level::Debug`].
    pub fn debug(&self, ctx: Context, msg: impl Into<String>) {
        self.send(Level::Debug, ctx, msg);
    }

    /// Log at [`Level::Info`].
    pub fn info(&self, ctx: Context, msg: impl Into<String>) {
        self.send(Level::Info, ctx, msg);
    }

    /// Log at [`Level::Warn`].
    pub fn warn(&self, ctx: Context, msg: impl Into<String>) {
        self.send(Level::Warn, ctx, msg);
    }

    /// Log at [`Level::Error`].
    pub fn error(&self, ctx: Context, msg: impl Into<String>) {
        self.send(Level::Error, ctx, msg);
    }

    /// Emit the run-start `[CONFIG]` block (C1): the resolved inputs
    /// (url / model / mode / tokens / iterations / timeout / nocache /
    /// ladder / NIAH ladder / engines / hardware) as one compact,
    /// greppable line. One channel send — never blocks the caller.
    pub fn log_config(&self, cfg: &Config) {
        self.info(Context::Setup, config_block_line(cfg));
    }

    fn send(&self, level: Level, ctx: Context, msg: impl Into<String>) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let _ = self.tx.send(Msg::Record {
            at_ms: now_ms(),
            level,
            ctx,
            msg: msg.into(),
        });
    }

    /// Flush both files and stop the writer thread. Idempotent: a second
    /// call (and the `Drop` impl) is a no-op.
    pub fn finish(&self) {
        if self.active.swap(false, Ordering::SeqCst) {
            let _ = self.tx.send(Msg::Shutdown);
            let mut guard = self
                .handle
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            if let Some(h) = guard.take() {
                // The writer only blocks on `recv`, so it exits as soon as
                // the shutdown marker arrives; the join cannot hang.
                let _ = h.join();
            }
        }
    }
}

impl Drop for RunLogger {
    fn drop(&mut self) {
        self.finish();
    }
}

/// Read a run-log file into lines (the **N/A-never-fail** seam for the
/// log-view / `--show-log` readers): `Ok(lines)` on success; `Err(String)`
/// when the file is missing or unreadable (disabled logger, a
/// `--log-dir` elsewhere, permissions). Pure `std::fs` — the only file
/// reads in the log module live on key-press / run-end paths, never the
/// non-blocking writer path.
pub fn read_log_file(path: &Path) -> Result<Vec<String>, String> {
    std::fs::read_to_string(path)
        .map_err(|e| e.to_string())
        .map(|s| s.lines().map(String::from).collect())
}

/// The writer thread: drain the channel, format each record, and write it
/// to *both* files (flushing on every newline so a crash loses nothing).
fn writer_thread(
    rx: mpsc::Receiver<Msg>,
    latest: std::fs::File,
    archive: std::fs::File,
    header: String,
) {
    let mut w_latest = std::io::LineWriter::new(latest);
    let mut w_archive = std::io::LineWriter::new(archive);
    let mut dead = false;

    // The run header first (both files).
    if !write_line(&mut w_latest, &mut w_archive, &header) {
        dead = true;
    }

    for m in rx {
        let Msg::Record {
            at_ms,
            level,
            ctx,
            msg,
        } = m
        else {
            break; // Shutdown
        };
        if dead {
            continue;
        }
        let line = format_line(at_ms, level, ctx, &msg);
        if !write_line(&mut w_latest, &mut w_archive, &line) {
            dead = true;
            eprintln!(
                "crucible-llm: run log write failed — logging disabled for the rest of this run"
            );
        }
    }
    let _ = w_latest.flush();
    let _ = w_archive.flush();
}

/// Write one line to both files. `false` on any write error.
fn write_line(
    w_latest: &mut std::io::LineWriter<std::fs::File>,
    w_archive: &mut std::io::LineWriter<std::fs::File>,
    line: &str,
) -> bool {
    let ok = w_latest
        .write_all(line.as_bytes())
        .and(w_latest.write_all(b"\n"))
        .and(w_archive.write_all(line.as_bytes()))
        .and(w_archive.write_all(b"\n"))
        .is_ok();
    ok
}

/// The run-start `[CONFIG]` block (C1): the resolved inputs of the run —
/// url / model / mode / tokens / iterations / timeout / nocache / the
/// concurrency ladder / the NIAH ladder / the selected engines / the
/// hardware toggle — as **one** compact line (a multi-line block would
/// break the `[ts] [LEVEL] [CTX] msg` line contract). Pure: no I/O.
#[must_use]
pub fn config_block_line(cfg: &Config) -> String {
    format!(
        "[CONFIG] url={} model={} mode={} tokens={} iterations={} timeout={}s nocache={} ladder=[{}] niah={} engines=[{}] hardware={}",
        cfg.url,
        cfg.model,
        cfg.mode.label(),
        cfg.tokens,
        cfg.iterations,
        cfg.timeout,
        cfg.nocache,
        cfg.ladder
            .iter()
            .map(|n| n.to_string())
            .collect::<Vec<_>>()
            .join(","),
        niah_ladder(cfg),
        cfg.engines.iter_labels().collect::<Vec<_>>().join(","),
        if cfg.hardware { "on" } else { "off" },
    )
}

/// The NIAH (C1) part of the `[CONFIG]` line: the configured ladder
/// (`2k..128k × 0..100`, or `custom 256k` for the count-test mode).
///
/// C3 (qualify once): without a `--tokenizer` the sizes are `chars/4`
/// estimates, and the one-time `(approx. token counts — no --tokenizer)`
/// qualifier lands **here** (plus the C1 start line in `sequence.rs`)
/// instead of repeating on every one of the per-cell log lines.
fn niah_ladder(cfg: &Config) -> String {
    let base = match cfg.niah_custom {
        Some(s) => format!("custom {}", human_size(s)),
        None => {
            let (Some(first), Some(last)) =
                (cfg.niah_contexts.first(), cfg.niah_contexts.last())
            else {
                return String::new();
            };
            let (d0, d1) = match (cfg.niah_depths.first(), cfg.niah_depths.last()) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    return format!("{}..{}", human_size(*first), human_size(*last))
                }
            };
            format!(
                "{}..{} × {}..{}",
                human_size(*first),
                human_size(*last),
                d0,
                d1
            )
        }
    };
    if cfg.tokenizer.is_none() && !base.is_empty() {
        format!("{base} (approx. token counts — no --tokenizer)")
    } else {
        base
    }
}

/// Format one log line: `[HH:MM:SS.mmm] [LEVEL] [CONTEXT] message`.
pub(crate) fn format_line(at_ms: u64, level: Level, ctx: Context, msg: &str) -> String {
    format!(
        "[{}] [{}] [{}] {}",
        format_timestamp(at_ms, false),
        level.label(),
        ctx.label(),
        msg
    )
}

/// The wall clock in milliseconds since the Unix epoch.
pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

/// Format a millisecond epoch timestamp.
///
/// * `date_part == false` → `HH:MM:SS.mmm` (the line prefix, per the
///   spec);
/// * `date_part == true` → `YYYY-MM-DD HH:MM:SS` (the run header and the
///   archive file name use `YYYYmmdd-HHMMSS` — see [`archive_stamp`]).
pub(crate) fn format_timestamp(at_ms: u64, date_part: bool) -> String {
    let secs = (at_ms / 1000) as i64;
    let millis = (at_ms % 1000) as u32;
    let (y, mo, d) = civil_from_days(secs / 86_400);
    let sod = (secs % 86_400).unsigned_abs();
    let h = sod / 3600;
    let mi = (sod % 3600) / 60;
    let s = sod % 60;
    if date_part {
        format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}")
    } else {
        format!("{h:02}:{mi:02}:{s:02}.{millis:03}")
    }
}

/// The first `n` characters of an id (the archive-name session suffix).
fn short_id(id: &str) -> String {
    id.chars().take(8).collect()
}

/// The archive file-name stamp: `YYYYmmdd-HHMMSS`.
pub(crate) fn archive_stamp(at_ms: u64) -> String {
    let secs = (at_ms / 1000) as i64;
    let (y, mo, d) = civil_from_days(secs / 86_400);
    let sod = (secs % 86_400).unsigned_abs();
    format!(
        "{y:04}{mo:02}{d:02}-{h:02}{m:02}{s:02}",
        h = sod / 3600,
        m = (sod % 3600) / 60,
        s = sod % 60
    )
}

/// Days-since-civil-epoch → `(year, month, day)` (Howard Hinnant's
/// `civil_from_days` algorithm; no date crate needed).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = ((5 * doy + 2) / 153) as u32; // [0, 11]
    let d = (doy - (153 * mp as u64 + 2) / 5 + 1) as u32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    (if m <= 2 { y + 1 } else { y }, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── timestamp / line formatting (pure) ──────────────────────────────

    #[test]
    fn civil_from_days_known_dates() {
        // 1970-01-01 (day 0)
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        // 1970-01-02
        assert_eq!(civil_from_days(1), (1970, 1, 2));
        // 2000-01-01 (day 10957) — the century boundary
        assert_eq!(civil_from_days(10_957), (2000, 1, 1));
        // 2024-02-29 (leap day; day 19782)
        assert_eq!(civil_from_days(19_782), (2024, 2, 29));
        // 1969-12-31 (day -1)
        assert_eq!(civil_from_days(-1), (1969, 12, 31));
    }

    #[test]
    fn format_timestamp_time_only_and_with_date() {
        // 2026-09-30 is day 20726 since the epoch (56 years + 14 leap
        // days to 2026-01-01, then Jan..Aug = 243 days + 29 into Sep).
        let day = 20_726i64;
        assert_eq!(civil_from_days(day), (2026, 9, 30));
        let ms = ((day * 86_400 + 20 * 3600 + 54 * 60 + 2) * 1000 + 123) as u64;
        assert_eq!(format_timestamp(ms, false), "20:54:02.123");
        assert_eq!(format_timestamp(ms, true), "2026-09-30 20:54:02");
        assert_eq!(archive_stamp(ms), "20260930-205402");
    }

    #[test]
    fn format_line_matches_the_spec() {
        let ms = ((20_726i64 * 86_400 + 20 * 3600 + 54 * 60 + 2) * 1000 + 123) as u64;
        let line = format_line(
            ms,
            Level::Warn,
            Context::EngineB,
            "Step 5/7: worker #3 no tokens for 30s",
        );
        assert_eq!(
            line,
            "[20:54:02.123] [WARN] [ENGINE_B] Step 5/7: worker #3 no tokens for 30s"
        );
        let line = format_line(ms, Level::Info, Context::Http, "→ POST x");
        assert_eq!(line, "[20:54:02.123] [INFO] [HTTP] → POST x");
    }

    // ── the logger itself (temp dir, no real $HOME touched) ─────────────

    fn temp_log_dir(name: &str) -> PathBuf {
        let dir =
            std::env::temp_dir().join(format!("crucible-runlog-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn start_creates_latest_and_archive_and_writes_records() {
        let dir = temp_log_dir("files");
        let logger = RunLogger::start(Some(&dir));
        assert!(logger.is_active());
        assert_eq!(logger.path(), Some(dir.join("latest.log").as_path()));

        logger.info(Context::Setup, "run started — url=http://x");
        logger.warn(Context::EngineB, "step 2 slow");
        logger.error(Context::Error, "boom");
        logger.debug(Context::Sse, "token #1");
        logger.finish();
        assert!(!logger.is_active());

        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        let archives: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("run-") && n.ends_with(".log"))
            .collect();
        assert_eq!(archives.len(), 1, "exactly one archive: {archives:?}");
        let archive = std::fs::read_to_string(dir.join(&archives[0])).unwrap();

        for content in [&latest, &archive] {
            assert!(
                content.contains("=== crucible-llm run log"),
                "header: {content}"
            );
            assert!(
                content.contains("[INFO] [SETUP] run started — url=http://x"),
                "{content}"
            );
            assert!(
                content.contains("[WARN] [ENGINE_B] step 2 slow"),
                "{content}"
            );
            assert!(content.contains("[ERROR] [ERROR] boom"), "{content}");
            assert!(content.contains("[DEBUG] [SSE] token #1"), "{content}");
        }
        // Both files carry the same records.
        assert_eq!(latest, archive);
        let _ = std::fs::remove_dir_all(&dir);
    }

    // R2 (traceability): the session id is stamped into the log header and
    // the archive file name, so the log joins deterministically to the
    // `benchmark_sessions` row that carries the same UUID.
    #[test]
    fn start_with_session_stamps_the_session_id() {
        let dir = temp_log_dir("session");
        let session_id = "12345678-aaaa-bbbb-cccc-dddddddddddd".to_string();
        let logger = RunLogger::start_with_session(Some(&dir), Some(&session_id));
        logger.info(Context::Setup, "run started");
        logger.finish();

        // The header in latest.log carries the full session id.
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        let needle = format!("(session {session_id})");
        assert!(
            latest.contains(&needle),
            "header carries the session id: {latest}"
        );

        // The archive file name carries the short (8-char) session id.
        let archives: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("run-") && n.ends_with(".log"))
            .collect();
        assert_eq!(archives.len(), 1, "one archive: {archives:?}");
        assert!(
            archives[0].contains("12345678"),
            "archive name carries the short session id: {archives:?}"
        );
        // The archive carries the same stamped header.
        let archive = std::fs::read_to_string(dir.join(&archives[0])).unwrap();
        assert!(archive.contains(&needle), "{archive}");

        // The logger exposes the session id for a programmatic join.
        assert_eq!(logger.session_id(), Some(session_id.as_str()));
        // The archive path accessor points at the on-disk archive.
        assert_eq!(
            logger.archive_path(),
            Some(dir.join(&archives[0]).as_path()),
            "archive_path() is the created archive file"
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn latest_is_overwritten_on_each_run() {
        let dir = temp_log_dir("overwrite");
        let a = RunLogger::start(Some(&dir));
        a.info(Context::Tui, "first run");
        a.finish();
        drop(a);
        let b = RunLogger::start(Some(&dir));
        b.info(Context::Tui, "second run");
        b.finish();
        drop(b);

        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        assert!(!latest.contains("first run"), "latest.log must be fresh");
        assert!(latest.contains("second run"));
        // Two archives exist now (one per run).
        let archives: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.starts_with("run-"))
            .collect();
        assert_eq!(archives.len(), 2, "{archives:?}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn disabled_logger_drops_everything_silently() {
        let logger = RunLogger::disabled();
        assert!(!logger.is_active());
        assert!(logger.path().is_none());
        assert!(logger.archive_path().is_none());
        // All four levels are no-ops (no panic, no channel).
        logger.debug(Context::Setup, "x");
        logger.info(Context::Setup, "x");
        logger.warn(Context::Setup, "x");
        logger.error(Context::Setup, "x");
        // C1: the config block is the same no-op channel send (the
        // non-blocking invariant — no file I/O, no blocking, nothing to
        // drop).
        logger.log_config(&Config::default());
        logger.finish(); // idempotent no-op
    }

    // ── read_log_file (the N/A-never-fail read seam) ────────────────────

    #[test]
    fn read_log_file_returns_the_lines() {
        let dir = temp_log_dir("read-log-file");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("latest.log");
        std::fs::write(
            &p,
            "[10:00:00.000] [INFO] [SETUP] one\n[10:00:01.000] [INFO] [SSE] two\n",
        )
        .unwrap();
        let lines = read_log_file(&p).expect("readable file");
        assert_eq!(
            lines,
            vec![
                "[10:00:00.000] [INFO] [SETUP] one".to_string(),
                "[10:00:01.000] [INFO] [SSE] two".to_string(),
            ]
        );
        // A trailing newline does not produce a phantom empty line.
        let p2 = dir.join("trailing.log");
        std::fs::write(&p2, "only\n").unwrap();
        assert_eq!(read_log_file(&p2).unwrap(), vec!["only".to_string()]);
        // An empty file is zero lines (not an error).
        let p3 = dir.join("empty.log");
        std::fs::write(&p3, "").unwrap();
        assert!(read_log_file(&p3).unwrap().is_empty());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_log_file_missing_is_an_err_not_a_panic() {
        let missing = std::env::temp_dir()
            .join(format!("crucible-runlog-missing-{}", std::process::id()))
            .join("never-created.log");
        let e = read_log_file(&missing).unwrap_err();
        assert!(!e.is_empty(), "the error carries the io reason");
    }

    #[test]
    fn finish_is_idempotent() {
        let dir = temp_log_dir("idempotent");
        let logger = RunLogger::start(Some(&dir));
        logger.info(Context::Setup, "once");
        logger.finish();
        logger.finish(); // second call: no panic, no hang
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        assert!(latest.contains("once"));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn concurrent_senders_are_safe() {
        let dir = temp_log_dir("concurrent");
        let logger = RunLogger::start(Some(&dir));
        let mut handles = Vec::new();
        for t in 0..4 {
            let l = logger.clone();
            handles.push(std::thread::spawn(move || {
                for i in 0..50 {
                    l.info(Context::EngineB, format!("thread {t} record {i}"));
                }
            }));
        }
        for h in handles {
            h.join().unwrap();
        }
        logger.finish();
        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        let n = latest.lines().filter(|l| l.contains("record ")).count();
        assert_eq!(n, 200, "every record from every thread landed");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ── C1: the run-start [CONFIG] block ───────────────────────────────

    #[test]
    fn config_block_line_carries_the_resolved_inputs() {
        use std::path::PathBuf;
        let mut cfg = Config::default();
        cfg.url = "http://192.168.1.10:8000".to_string();
        cfg.model = "qwen3.8-27b".to_string();
        cfg.mode = crate::config::Mode::Long;
        cfg.tokens = 32_000;
        cfg.iterations = 6;
        cfg.timeout = 120;
        cfg.nocache = true;
        cfg.ladder = vec![1, 2, 4, 8];
        cfg.niah_contexts = vec![2_000, 4_000, 8_000];
        cfg.niah_depths = vec![0, 50, 100];
        cfg.engines = crate::config::EngineSelection::all();
        cfg.hardware = true;
        cfg.tokenizer = Some(PathBuf::from("/tmp/tokenizer.json"));
        let line = config_block_line(&cfg);
        assert!(line.starts_with("[CONFIG] "), "{line}");
        assert!(line.contains("url=http://192.168.1.10:8000"), "{line}");
        assert!(line.contains("model=qwen3.8-27b"), "{line}");
        assert!(line.contains("mode=long"), "{line}");
        assert!(line.contains("tokens=32000"), "{line}");
        assert!(line.contains("iterations=6"), "{line}");
        assert!(line.contains("timeout=120s"), "{line}");
        assert!(line.contains("nocache=true"), "{line}");
        assert!(line.contains("ladder=[1,2,4,8]"), "{line}");
        assert!(line.contains("niah=2k..8k × 0..100"), "{line}");
        assert!(line.contains(
            "engines=[A (speed),B (concurrency),C1 (niah),C2 (reasoning),C3 (structured),D (hardware),F (flat out)]"
        ),
            "{line}"
        );
        assert!(line.contains("hardware=on"), "{line}");
        // A tokenizer is set → the NIAH sizes are exact → no qualifier.
        assert!(!line.contains("approx"), "{line}");
        // One line (the `[ts] [LEVEL] [CTX] msg` contract).
        assert!(!line.contains('\n'));
    }

    #[test]
    fn config_block_line_default_config_qualifies_niah_once() {
        // The default config has no `--tokenizer`: the one-time
        // `(approx.)` qualifier lands in the NIAH part (C3 — it no
        // longer repeats on every per-cell line).
        let line = config_block_line(&Config::default());
        assert!(
            line.contains("niah=2k..128k × 0..100 (approx. token counts — no --tokenizer)"),
            "{line}"
        );
        // Exactly one occurrence.
        assert_eq!(line.matches("approx").count(), 1, "{line}");
    }

    #[test]
    fn niah_ladder_custom_mode_and_tokenizer() {
        let mut cfg = Config::default();
        cfg.niah_custom = Some(256_000);
        assert_eq!(
            niah_ladder(&cfg),
            "custom 256k (approx. token counts — no --tokenizer)"
        );
        cfg.tokenizer = Some(std::path::PathBuf::from("/tmp/tok.json"));
        assert_eq!(niah_ladder(&cfg), "custom 256k");
    }

    #[test]
    fn log_config_is_the_first_record_after_the_header() {
        let dir = temp_log_dir("config-block");
        let logger = RunLogger::start(Some(&dir));
        logger.log_config(&Config::default());
        logger.info(Context::EngineA, "Engine A started");
        logger.finish();

        let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
        let lines: Vec<&str> = latest.lines().collect();
        // Line 0 is the run header; the [CONFIG] block is line 1 — near
        // the top of the run log, one compact line.
        assert!(lines[0].starts_with("=== crucible-llm run log"), "{latest}");
        assert!(
            lines[1].contains("[SETUP] [CONFIG] url="),
            "the [CONFIG] block is the first record: {latest}"
        );
        assert!(lines[1].contains("niah="), "{latest}");
        assert!(lines[1].contains("engines="), "{latest}");
        assert!(lines[1].contains("hardware="), "{latest}");
        assert_eq!(lines.len(), 3, "header + config + one record: {latest}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn uncreatable_dir_degrades_to_disabled() {
        // A path whose parent cannot exist (file in the way): on a normal
        // user account this fails → the logger degrades, never panics.
        let dir =
            std::env::temp_dir().join(format!("crucible-runlog-blocked-{}", std::process::id()));
        std::fs::write(&dir, "i am a file, not a dir").unwrap();
        let logger = RunLogger::start(Some(&dir.join("impossible")));
        assert!(!logger.is_active());
        logger.info(Context::Setup, "dropped");
        logger.finish();
        let _ = std::fs::remove_file(&dir);
    }
}
