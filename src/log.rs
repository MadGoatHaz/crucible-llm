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
}

impl std::fmt::Debug for RunLogger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunLogger")
            .field("active", &self.active.load(Ordering::Relaxed))
            .field("path", &self.path)
            .finish()
    }
}

impl RunLogger {
    /// Start a fresh run log: open `latest.log` (overwritten) and an
    /// archived `run-<timestamp>.log` in `log_dir`, and spawn the writer
    /// thread.
    ///
    /// `log_dir` defaults to `data_dir()/crucible/logs` (i.e.
    /// `~/.local/share/crucible/logs` on Linux) when `None`. Any failure
    /// (missing home dir, permissions, …) degrades to a
    /// [`disabled`](Self::disabled) logger — logging never breaks a run.
    pub fn start(log_dir: Option<&Path>) -> Arc<Self> {
        let dir = log_dir
            .map(PathBuf::from)
            .unwrap_or_else(|| crate::config::data_dir().join("logs"));
        let now_ms = now_ms();
        // The archive name is second-granular; disambiguate collisions
        // (two runs in the same second) with a `-2`, `-3`, … suffix.
        let stamp = archive_stamp(now_ms);
        let mut archive_name = format!("run-{stamp}.log");
        let mut counter = 1;
        while dir.join(&archive_name).exists() {
            counter += 1;
            archive_name = format!("run-{stamp}-{counter}.log");
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
        let header = format!(
            "=== crucible-llm run log — {} ===",
            format_timestamp(header_at, true)
        );
        let handle = std::thread::Builder::new()
            .name("crucible-runlog".to_string())
            .spawn(move || writer_thread(rx, latest, archive, header))
            .ok();
        let logger = Arc::new(Self {
            tx,
            active: AtomicBool::new(true),
            handle: std::sync::Mutex::new(handle),
            path: Some(dir.join("latest.log")),
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
        })
    }

    /// `true` while the logger is writing to its files.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// The `latest.log` path (`None` when disabled).
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
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
        // All four levels are no-ops (no panic, no channel).
        logger.debug(Context::Setup, "x");
        logger.info(Context::Setup, "x");
        logger.warn(Context::Setup, "x");
        logger.error(Context::Setup, "x");
        logger.finish(); // idempotent no-op
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
