//! Chunk 12 acceptance tests: the SQLite storage engine.
//!
//! Covers the plan's acceptance criteria:
//! * a completed run creates a `benchmark_sessions` row + `stream_metrics`
//!   rows matching the blueprint §8 schema;
//! * reopening the DB and querying returns the stored rows;
//! * the file lands in the correct platform data dir
//!   (`dirs::data_dir()/crucible/benchmarks.db`).
//!
//! All tests use throwaway databases under the system temp dir — never the
//! real user data dir.

use std::path::Path;

use crucible_llm::config::data_dir;
use crucible_llm::engines::speed::SpeedResult;
use crucible_llm::storage::{
    BenchmarkSession, Database, NeedleEvaluation, StreamMetricRow, DB_FILE_NAME,
};

/// A unique temp directory for one test (process id + name → parallel
/// test binaries never collide).
fn temp_db_dir(test: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("crucible-test-{}-{test}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn cleanup(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
}

fn sample_session(id: &str) -> BenchmarkSession {
    BenchmarkSession {
        session_id: id.to_string(),
        timestamp: None,
        target_url: "http://127.0.0.1:8000/v1".to_string(),
        model_name: "Qwen3.6-35B-A3B".to_string(),
        backend_type: Some("vllm".to_string()),
        quantization: Some("Q4_K_XL".to_string()),
        system_gpu: Some("RTX 4090".to_string()),
        total_duration_sec: Some(12.5),
    }
}

// ── schema / migrations ─────────────────────────────────────────────────

#[test]
fn schema_has_all_three_tables() {
    let dir = temp_db_dir("schema");
    let path = dir.join("benchmarks.db");
    let db = Database::open(&path).unwrap();

    let mut stmt = db
        .conn()
        .prepare("SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
        .unwrap();
    let names: Vec<String> = stmt
        .query_map([], |r| r.get(0))
        .unwrap()
        .collect::<Result<_, _>>()
        .unwrap();

    assert!(
        names.contains(&"benchmark_sessions".to_string()),
        "missing benchmark_sessions: {names:?}"
    );
    assert!(
        names.contains(&"stream_metrics".to_string()),
        "missing stream_metrics: {names:?}"
    );
    assert!(
        names.contains(&"needle_evaluations".to_string()),
        "missing needle_evaluations: {names:?}"
    );
    cleanup(&dir);
}

#[test]
fn migrations_are_idempotent() {
    let dir = temp_db_dir("migrate");
    let path = dir.join("benchmarks.db");
    // Three successive opens (each runs the migration) must all succeed.
    for _ in 0..3 {
        let db = Database::open(&path).unwrap();
        db.migrate().unwrap();
        drop(db);
    }
    cleanup(&dir);
}

// ── location ────────────────────────────────────────────────────────────

#[test]
fn default_path_is_platform_data_dir() {
    let p = Database::default_path();
    // `…/crucible/benchmarks.db` (Linux `~/.local/share/crucible`,
    // Windows `%APPDATA%\crucible`).
    assert_eq!(p.file_name(), Some(std::ffi::OsStr::new(DB_FILE_NAME)));
    assert_eq!(
        p.parent().and_then(|d| d.file_name()),
        Some(std::ffi::OsStr::new("crucible"))
    );
    // And it is rooted at the platform data dir (config.rs source of
    // truth for the location).
    assert_eq!(p.parent().unwrap(), &data_dir());
}

// ── session round-trip ──────────────────────────────────────────────────

#[test]
fn session_round_trips_through_reopen() {
    let dir = temp_db_dir("session-rt");
    let path = dir.join("benchmarks.db");

    {
        let mut db = Database::open(&path).unwrap();
        db.insert_session(&sample_session("sess-1")).unwrap();
    } // drop → close

    // Reopen: the stored row comes back, and the NULL timestamp was
    // filled by the DDL's `DEFAULT CURRENT_TIMESTAMP`.
    let db = Database::open(&path).unwrap();
    let s = db.get_session("sess-1").unwrap().expect("session missing");
    assert_eq!(s.target_url, "http://127.0.0.1:8000/v1");
    assert_eq!(s.model_name, "Qwen3.6-35B-A3B");
    assert_eq!(s.backend_type.as_deref(), Some("vllm"));
    assert_eq!(s.quantization.as_deref(), Some("Q4_K_XL"));
    assert_eq!(s.system_gpu.as_deref(), Some("RTX 4090"));
    assert!((s.total_duration_sec.unwrap() - 12.5).abs() < 1e-9);
    let ts = s.timestamp.as_deref().expect("CURRENT_TIMESTAMP default");
    assert!(
        ts.len() == 19 && ts.as_bytes()[4] == b'-' && ts.as_bytes()[10] == b' ',
        "unexpected timestamp format: {ts}"
    );
    assert!(db.get_session("nope").unwrap().is_none());
    cleanup(&dir);
}

// ── stream_metrics round-trip ───────────────────────────────────────────

#[test]
fn stream_metrics_round_trip_preserves_order_and_values() {
    let dir = temp_db_dir("metrics-rt");
    let path = dir.join("benchmarks.db");

    {
        let mut db = Database::open(&path).unwrap();
        db.insert_session(&sample_session("sess-2")).unwrap();
        let a = StreamMetricRow {
            metric_id: None,
            session_id: "sess-2".into(),
            concurrency_level: Some(1),
            prompt_tokens: Some(2048),
            completion_tokens: Some(312),
            reasoning_tokens: Some(200),
            ttft_ms: Some(182.0),
            tpot_ms: Some(13.8),
            mtp_efficiency: Some(1.84),
            joules_per_token: Some(0.338),
            cache_hit: Some(true),
        };
        let b = StreamMetricRow {
            metric_id: None,
            session_id: "sess-2".into(),
            concurrency_level: Some(4),
            prompt_tokens: Some(512),
            completion_tokens: Some(180),
            reasoning_tokens: None,
            ttft_ms: Some(45.0),
            tpot_ms: None,
            mtp_efficiency: Some(1.02),
            joules_per_token: None,
            cache_hit: None,
        };
        let id_a = db.insert_stream_metric(&a).unwrap();
        let id_b = db.insert_stream_metric(&b).unwrap();
        assert!(id_b > id_a, "autoincrement must advance");
    }

    let db = Database::open(&path).unwrap();
    let rows = db.get_stream_metrics("sess-2").unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].concurrency_level, Some(1));
    assert_eq!(rows[0].prompt_tokens, Some(2048));
    assert_eq!(rows[0].completion_tokens, Some(312));
    assert_eq!(rows[0].reasoning_tokens, Some(200));
    assert!((rows[0].ttft_ms.unwrap() - 182.0).abs() < 1e-9);
    assert!((rows[0].tpot_ms.unwrap() - 13.8).abs() < 1e-9);
    assert!((rows[0].mtp_efficiency.unwrap() - 1.84).abs() < 1e-9);
    assert!((rows[0].joules_per_token.unwrap() - 0.338).abs() < 1e-9);
    assert_eq!(rows[0].cache_hit, Some(true));

    // NULLs stay NULL.
    assert_eq!(rows[1].concurrency_level, Some(4));
    assert_eq!(rows[1].reasoning_tokens, None);
    assert_eq!(rows[1].tpot_ms, None);
    assert_eq!(rows[1].joules_per_token, None);
    assert_eq!(rows[1].cache_hit, None);
    cleanup(&dir);
}

// ── needle_evaluations round-trip ───────────────────────────────────────

#[test]
fn needle_evaluations_round_trip() {
    let dir = temp_db_dir("needle-rt");
    let path = dir.join("benchmarks.db");

    {
        let mut db = Database::open(&path).unwrap();
        db.insert_session(&sample_session("sess-3")).unwrap();
        db.insert_needle_evaluation(&NeedleEvaluation {
            eval_id: None,
            session_id: "sess-3".into(),
            context_length: Some(8192),
            depth_percent: Some(30.0),
            retrieved_successfully: Some(true),
            latency_ms: Some(950.0),
        })
        .unwrap();
        db.insert_needle_evaluation(&NeedleEvaluation {
            eval_id: None,
            session_id: "sess-3".into(),
            context_length: Some(32768),
            depth_percent: Some(90.0),
            retrieved_successfully: Some(false),
            latency_ms: Some(4100.0),
        })
        .unwrap();
    }

    let db = Database::open(&path).unwrap();
    let rows = db.get_needle_evaluations("sess-3").unwrap();
    assert_eq!(rows.len(), 2);
    assert_eq!(rows[0].context_length, Some(8192));
    assert!((rows[0].depth_percent.unwrap() - 30.0).abs() < 1e-9);
    assert_eq!(rows[0].retrieved_successfully, Some(true));
    assert!((rows[0].latency_ms.unwrap() - 950.0).abs() < 1e-9);
    assert_eq!(rows[1].retrieved_successfully, Some(false));
    cleanup(&dir);
}

// ── SpeedResult → row mapping ───────────────────────────────────────────

fn speed_result(tg_speed: f64) -> SpeedResult {
    SpeedResult {
        ttft: 0.125,
        prompt_tokens: 128,
        completion_tokens: 64,
        pp_speed: 1024.0,
        tg_speed,
        mtp_efficiency: 1.84,
        stream_time: 2.0,
        total_chunks: 40,
        content_chunks: 35,
        reasoning_chunks: 4,
        other_chunks: 1,
        estimated: false,
        model: "test-model".into(),
        mode: "short".into(),
        error: None,
        looping: false,
    }
}

#[test]
fn from_speed_result_maps_the_schema_fields() {
    let row = StreamMetricRow::from_speed_result(&speed_result(80.0), "sess-x", 1);
    assert_eq!(row.session_id, "sess-x");
    assert_eq!(row.concurrency_level, Some(1));
    assert_eq!(row.prompt_tokens, Some(128));
    assert_eq!(row.completion_tokens, Some(64));
    assert!(
        (row.ttft_ms.unwrap() - 125.0).abs() < 1e-9,
        "0.125s → 125ms"
    );
    assert!(
        (row.tpot_ms.unwrap() - 12.5).abs() < 1e-9,
        "tpot = 1000 / 80 t/s"
    );
    assert!((row.mtp_efficiency.unwrap() - 1.84).abs() < 1e-9);
    // Not produced by the single-stream engine yet.
    assert_eq!(row.reasoning_tokens, None);
    assert_eq!(row.joules_per_token, None);
    assert_eq!(row.cache_hit, None);
}

#[test]
fn from_speed_result_zero_decode_speed_yields_null_tpot() {
    let row = StreamMetricRow::from_speed_result(&speed_result(0.0), "sess-x", 1);
    assert_eq!(row.tpot_ms, None, "no decode → no TPOT");
    assert_eq!(row.ttft_ms, Some(125.0));
}

// ── completed-run flow (the chunk's core acceptance) ────────────────────

#[test]
fn completed_run_persists_session_and_stream_rows() {
    let dir = temp_db_dir("run-flow");
    let path = dir.join("benchmarks.db");

    // A completed run: 3 iterations (2 ok, 1 failed — still persisted,
    // matching "after every completed run").
    let r1 = speed_result(80.0);
    let r2 = speed_result(75.0);
    let mut r3 = speed_result(0.0);
    r3.completion_tokens = 0;
    r3.error = Some("Connection refused — is the server running?".into());

    {
        let mut db = Database::open(&path).unwrap();
        let session = BenchmarkSession {
            session_id: "flow-1".into(),
            timestamp: None,
            target_url: "http://127.0.0.1:8000/v1".into(),
            model_name: "test-model".into(),
            backend_type: None,
            quantization: None,
            system_gpu: None,
            total_duration_sec: Some(r1.stream_time + r2.stream_time + r3.stream_time),
        };
        let rows = [
            StreamMetricRow::from_speed_result(&r1, "flow-1", 1),
            StreamMetricRow::from_speed_result(&r2, "flow-1", 1),
            StreamMetricRow::from_speed_result(&r3, "flow-1", 1),
        ];
        db.persist_run(&session, &rows).unwrap();
    }

    // Reopen and query: the stored rows come back.
    let db = Database::open(&path).unwrap();
    let s = db.get_session("flow-1").unwrap().expect("session missing");
    assert_eq!(s.model_name, "test-model");
    assert!(s.timestamp.is_some());

    let rows = db.get_stream_metrics("flow-1").unwrap();
    assert_eq!(rows.len(), 3);
    assert_eq!(rows[0].prompt_tokens, Some(128));
    assert_eq!(rows[0].completion_tokens, Some(64));
    assert!((rows[0].tpot_ms.unwrap() - 12.5).abs() < 1e-9);
    assert!((rows[1].tpot_ms.unwrap() - 1000.0 / 75.0).abs() < 1e-9);
    assert_eq!(rows[2].completion_tokens, Some(0));
    assert_eq!(rows[2].tpot_ms, None);

    // list_sessions sees it too.
    let all = db.list_sessions().unwrap();
    assert_eq!(all.len(), 1);
    assert_eq!(all[0].session_id, "flow-1");
    cleanup(&dir);
}

// ── atomicity ───────────────────────────────────────────────────────────

#[test]
fn persist_run_rolls_back_the_whole_batch_on_duplicate_session() {
    let dir = temp_db_dir("rollback");
    let path = dir.join("benchmarks.db");

    let db_owned = Database::open(&path).unwrap();
    // Pre-seed the session id so the batch's INSERT hits the PK.
    let mut db = db_owned;
    db.insert_session(&sample_session("dup-1")).unwrap();

    let session = BenchmarkSession {
        session_id: "dup-1".into(),
        ..sample_session("dup-1")
    };
    let row = StreamMetricRow::from_speed_result(&speed_result(80.0), "dup-1", 1);
    let err = db
        .persist_run(&session, std::slice::from_ref(&row))
        .unwrap_err();
    assert!(
        err.to_string().contains("sqlite"),
        "expected a constraint error, got: {err}"
    );

    // Rollback: the pre-seeded session is untouched and NO metric rows
    // from the failed batch were written.
    assert_eq!(db.get_stream_metrics("dup-1").unwrap().len(), 0);
    assert!(db.get_session("dup-1").unwrap().is_some());
    cleanup(&dir);
}
