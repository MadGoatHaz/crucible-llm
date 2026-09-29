//! `rusqlite` open/migrate (from `schema.sql`) and insert/query helpers,
//! storing to the platform data dir (`~/.local/share/crucible/benchmarks.db`).
//!
//! `Database` owns one `Connection`. Writes take `&mut self`; reads take
//! `&self`. Every open runs the schema migration (blueprint §8 DDL,
//! `IF NOT EXISTS` → idempotent), so a first run creates the file and a
//! later run reopens the existing one.
//!
//! Measurement-isolation note (blueprint §4): persistence is a
//! background concern — nothing in the quanta timing path calls into
//! this module; runs are persisted only *after* they complete.

use std::path::{Path, PathBuf};

use rusqlite::{params, Connection, Row};
use thiserror::Error;

use crate::config::data_dir;

use super::models::{BenchmarkSession, NeedleEvaluation, StreamMetricRow};

/// The DB file name inside the platform data dir.
pub const DB_FILE_NAME: &str = "benchmarks.db";

/// The schema batch (blueprint §8), embedded at compile time.
pub const SCHEMA: &str = include_str!("schema.sql");

/// Storage errors.
#[derive(Debug, Error)]
pub enum StorageError {
    /// The data dir / DB file could not be created.
    #[error("storage I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// A SQLite failure (constraint violation, malformed statement, …).
    #[error("sqlite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
}

/// An open `benchmarks.db` (migrated to the blueprint §8 schema).
pub struct Database {
    conn: Connection,
    path: PathBuf,
}

impl Database {
    /// The default location: `dirs::data_dir()/crucible/benchmarks.db`
    /// (Linux `~/.local/share/crucible/benchmarks.db`, Windows
    /// `%APPDATA%\crucible\benchmarks.db`).
    pub fn default_path() -> PathBuf {
        data_dir().join(DB_FILE_NAME)
    }

    /// Open (creating the file and parent dirs if needed) the database at
    /// `path` and run the schema migration.
    pub fn open(path: &Path) -> Result<Self, StorageError> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let conn = Connection::open(path)?;
        let db = Self {
            conn,
            path: path.to_path_buf(),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Open (creating if needed) the database at [`Self::default_path`].
    pub fn open_default() -> Result<Self, StorageError> {
        Self::open(&Self::default_path())
    }

    /// Apply the schema (blueprint §8). Idempotent — safe to run on every
    /// open, which is how migrations get applied.
    pub fn migrate(&self) -> Result<(), StorageError> {
        self.conn.execute_batch(SCHEMA)?;
        Ok(())
    }

    /// The on-disk location of this database.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The raw `Connection` (escape hatch for ad-hoc queries, e.g. the
    /// Chunk 13 exporters and the Chunk 14 history diff).
    pub fn conn(&self) -> &Connection {
        &self.conn
    }

    // ── inserts ─────────────────────────────────────────────────────────

    /// Insert a `benchmark_sessions` row.
    ///
    /// `timestamp == None` → the DDL's `DEFAULT CURRENT_TIMESTAMP` fills
    /// it. Fails with a constraint error when the `session_id` already
    /// exists (primary key).
    pub fn insert_session(&mut self, s: &BenchmarkSession) -> Result<(), StorageError> {
        insert_session_conn(&self.conn, s)?;
        Ok(())
    }

    /// Insert a `stream_metrics` row; returns the new `metric_id`.
    pub fn insert_stream_metric(&mut self, m: &StreamMetricRow) -> Result<i64, StorageError> {
        Ok(insert_stream_metric_conn(&self.conn, m)?)
    }

    /// Insert a `needle_evaluations` row; returns the new `eval_id`.
    pub fn insert_needle_evaluation(&mut self, e: &NeedleEvaluation) -> Result<i64, StorageError> {
        Ok(insert_needle_conn(&self.conn, e)?)
    }

    /// Persist a completed run: the session row plus one `stream_metrics`
    /// row per iteration, atomically in a single transaction.
    ///
    /// Any failure (e.g. a duplicate `session_id`) rolls the whole batch
    /// back — a run is recorded whole or not at all.
    pub fn persist_run(
        &mut self,
        session: &BenchmarkSession,
        metrics: &[StreamMetricRow],
    ) -> Result<(), StorageError> {
        let tx = self.conn.transaction()?;
        insert_session_conn(&tx, session)?;
        for m in metrics {
            insert_stream_metric_conn(&tx, m)?;
        }
        tx.commit()?;
        Ok(())
    }

    // ── queries ─────────────────────────────────────────────────────────

    /// Fetch one session by id.
    pub fn get_session(&self, session_id: &str) -> Result<Option<BenchmarkSession>, StorageError> {
        let mut stmt = self.conn.prepare(
            "SELECT session_id, timestamp, target_url, model_name,
                    backend_type, quantization, system_gpu, total_duration_sec
             FROM benchmark_sessions
             WHERE session_id = ?1",
        )?;
        let mut rows = stmt.query_map(params![session_id], row_to_session)?;
        match rows.next() {
            Some(Ok(s)) => Ok(Some(s)),
            Some(Err(e)) => Err(e.into()),
            None => Ok(None),
        }
    }

    /// All sessions, newest first.
    pub fn list_sessions(&self) -> Result<Vec<BenchmarkSession>, StorageError> {
        let mut stmt = self.conn.prepare(
            "SELECT session_id, timestamp, target_url, model_name,
                    backend_type, quantization, system_gpu, total_duration_sec
             FROM benchmark_sessions
             ORDER BY timestamp DESC, session_id DESC",
        )?;
        let out = stmt
            .query_map([], row_to_session)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(out)
    }

    /// All `stream_metrics` rows for a session (in `metric_id` order).
    pub fn get_stream_metrics(
        &self,
        session_id: &str,
    ) -> Result<Vec<StreamMetricRow>, StorageError> {
        let mut stmt = self.conn.prepare(
            "SELECT metric_id, session_id, concurrency_level, prompt_tokens,
                    completion_tokens, reasoning_tokens, ttft_ms, tpot_ms,
                    mtp_efficiency, joules_per_token, cache_hit
             FROM stream_metrics
             WHERE session_id = ?1
             ORDER BY metric_id",
        )?;
        let out = stmt
            .query_map(params![session_id], row_to_stream_metric)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(out)
    }

    /// All `needle_evaluations` rows for a session (in `eval_id` order).
    pub fn get_needle_evaluations(
        &self,
        session_id: &str,
    ) -> Result<Vec<NeedleEvaluation>, StorageError> {
        let mut stmt = self.conn.prepare(
            "SELECT eval_id, session_id, context_length, depth_percent,
                    retrieved_successfully, latency_ms
             FROM needle_evaluations
             WHERE session_id = ?1
             ORDER BY eval_id",
        )?;
        let out = stmt
            .query_map(params![session_id], row_to_needle)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        Ok(out)
    }
}

// ── shared insert statements (plain connection; `Transaction` derefs to
//    `Connection`, so the same functions serve `persist_run`'s tx) ─────

fn insert_session_conn(conn: &Connection, s: &BenchmarkSession) -> rusqlite::Result<()> {
    // `timestamp == None` → the column is *omitted* from the INSERT so the
    // DDL's `DEFAULT CURRENT_TIMESTAMP` applies. (An explicit `NULL` would
    // override the default.)
    match &s.timestamp {
        Some(ts) => {
            conn.execute(
                "INSERT INTO benchmark_sessions
                    (session_id, timestamp, target_url, model_name,
                     backend_type, quantization, system_gpu, total_duration_sec)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    s.session_id,
                    ts,
                    s.target_url,
                    s.model_name,
                    s.backend_type,
                    s.quantization,
                    s.system_gpu,
                    s.total_duration_sec,
                ],
            )?;
        }
        None => {
            conn.execute(
                "INSERT INTO benchmark_sessions
                    (session_id, target_url, model_name,
                     backend_type, quantization, system_gpu, total_duration_sec)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                params![
                    s.session_id,
                    s.target_url,
                    s.model_name,
                    s.backend_type,
                    s.quantization,
                    s.system_gpu,
                    s.total_duration_sec,
                ],
            )?;
        }
    }
    Ok(())
}

fn insert_stream_metric_conn(conn: &Connection, m: &StreamMetricRow) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO stream_metrics
            (session_id, concurrency_level, prompt_tokens, completion_tokens,
             reasoning_tokens, ttft_ms, tpot_ms, mtp_efficiency,
             joules_per_token, cache_hit)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
        params![
            m.session_id,
            m.concurrency_level,
            m.prompt_tokens,
            m.completion_tokens,
            m.reasoning_tokens,
            m.ttft_ms,
            m.tpot_ms,
            m.mtp_efficiency,
            m.joules_per_token,
            m.cache_hit,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

fn insert_needle_conn(conn: &Connection, e: &NeedleEvaluation) -> rusqlite::Result<i64> {
    conn.execute(
        "INSERT INTO needle_evaluations
            (session_id, context_length, depth_percent,
             retrieved_successfully, latency_ms)
         VALUES (?1, ?2, ?3, ?4, ?5)",
        params![
            e.session_id,
            e.context_length,
            e.depth_percent,
            e.retrieved_successfully,
            e.latency_ms,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

// ── row decoders ────────────────────────────────────────────────────────

fn row_to_session(row: &Row) -> rusqlite::Result<BenchmarkSession> {
    Ok(BenchmarkSession {
        session_id: row.get(0)?,
        timestamp: row.get(1)?,
        target_url: row.get(2)?,
        model_name: row.get(3)?,
        backend_type: row.get(4)?,
        quantization: row.get(5)?,
        system_gpu: row.get(6)?,
        total_duration_sec: row.get(7)?,
    })
}

fn row_to_stream_metric(row: &Row) -> rusqlite::Result<StreamMetricRow> {
    Ok(StreamMetricRow {
        metric_id: row.get(0)?,
        session_id: row.get(1)?,
        concurrency_level: row.get(2)?,
        prompt_tokens: row.get(3)?,
        completion_tokens: row.get(4)?,
        reasoning_tokens: row.get(5)?,
        ttft_ms: row.get(6)?,
        tpot_ms: row.get(7)?,
        mtp_efficiency: row.get(8)?,
        joules_per_token: row.get(9)?,
        cache_hit: row.get(10)?,
    })
}

fn row_to_needle(row: &Row) -> rusqlite::Result<NeedleEvaluation> {
    Ok(NeedleEvaluation {
        eval_id: row.get(0)?,
        session_id: row.get(1)?,
        context_length: row.get(2)?,
        depth_percent: row.get(3)?,
        retrieved_successfully: row.get(4)?,
        latency_ms: row.get(5)?,
    })
}
