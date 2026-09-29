//! Chunk 13 integration tests: the JSON / Markdown / CSV exporters over
//! the SQLite storage layer (blueprint §8 "Export Formats").

use crucible_llm::client::StreamEvent;
use crucible_llm::config::ExportFormat;
use crucible_llm::sse::{Chunk, ParsedFrame, Usage};
use crucible_llm::storage::export::{
    default_path, extension, samples_from_events, write, ExportError, ExportPayload, PacketSample,
};
use crucible_llm::storage::models::{BenchmarkSession, NeedleEvaluation, StreamMetricRow};
use crucible_llm::storage::Database;
use crucible_llm::timing::{MonotonicInstant, StreamTimestamps};

fn sample_session() -> BenchmarkSession {
    BenchmarkSession {
        session_id: "aaaabbbb-cccc-dddd-eeee-ffff00001111".to_string(),
        timestamp: None,
        target_url: "http://127.0.0.1:8000/v1/chat/completions".to_string(),
        model_name: "qwen3-8b".to_string(),
        backend_type: Some("vllm".to_string()),
        quantization: None,
        system_gpu: None,
        total_duration_sec: Some(3.5),
    }
}

fn sample_metrics(session_id: &str) -> Vec<StreamMetricRow> {
    vec![
        StreamMetricRow {
            metric_id: None,
            session_id: session_id.to_string(),
            concurrency_level: Some(1),
            prompt_tokens: Some(256),
            completion_tokens: Some(128),
            reasoning_tokens: Some(64),
            ttft_ms: Some(150.25),
            tpot_ms: Some(12.5),
            mtp_efficiency: Some(2.0),
            joules_per_token: None,
            cache_hit: Some(true),
        },
        StreamMetricRow {
            metric_id: None,
            session_id: session_id.to_string(),
            concurrency_level: Some(1),
            prompt_tokens: Some(256),
            completion_tokens: Some(130),
            reasoning_tokens: None,
            ttft_ms: Some(148.0),
            tpot_ms: Some(12.7),
            mtp_efficiency: Some(1.98),
            joules_per_token: None,
            cache_hit: Some(false),
        },
    ]
}

fn temp_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("crucible-export-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

/// A synthetic worker event stream (Chunk 5 shape): reasoning → content →
/// content → usage → `[DONE]`, with known `t_nanos` stamps since `T0`.
fn synthetic_events() -> Vec<StreamEvent> {
    let t0 = MonotonicInstant::now();
    let ts = StreamTimestamps {
        t0: Some(t0),
        ..StreamTimestamps::default()
    };
    let mk = |chunk: Chunk, t_nanos: u64, done: bool| StreamEvent::Frame {
        frame: ParsedFrame {
            chunk,
            t_nanos,
            done,
        },
        at: t0,
        timestamps: ts,
    };
    vec![
        mk(Chunk::Reasoning("think".into()), 1_000_000, false),
        mk(Chunk::Content("a".into()), 1_012_000_000, false),
        mk(Chunk::Content("b".into()), 1_030_000_000, false),
        mk(
            Chunk::Usage(Usage {
                prompt_tokens: 10,
                completion_tokens: 4,
            }),
            1_031_000_000,
            false,
        ),
        mk(Chunk::Control, 1_031_500_000, true),
    ]
}

#[test]
fn persist_then_export_json_round_trips_through_sqlite() {
    let dir = temp_dir("json");
    let db_path = dir.join("benchmarks.db");
    let mut db = Database::open(&db_path).unwrap();
    let session = sample_session();
    db.persist_run(&session, &sample_metrics(&session.session_id))
        .unwrap();
    let _ = db.insert_needle_evaluation(&NeedleEvaluation {
        eval_id: None,
        session_id: session.session_id.clone(),
        context_length: Some(8192),
        depth_percent: Some(50.0),
        retrieved_successfully: Some(true),
        latency_ms: Some(210.0),
    });

    // Consume the run back from the storage layer.
    let payload = ExportPayload::from_stored(&db, &session.session_id).unwrap();
    assert_eq!(payload.metrics.len(), 2);
    assert_eq!(payload.needles.len(), 1);

    let json = payload.to_json();
    // Acceptance: `--export json` produces valid, parseable JSON.
    let back: ExportPayload = serde_json::from_str(&json).unwrap();
    assert_eq!(back.session.session_id, session.session_id);
    assert_eq!(back.session.model_name, "qwen3-8b");
    assert_eq!(back.session.target_url, session.target_url);
    assert_eq!(back.metrics.len(), 2);
    assert_eq!(back.metrics[0].ttft_ms, Some(150.25));
    assert_eq!(back.metrics[0].cache_hit, Some(true));
    assert_eq!(back.metrics[1].cache_hit, Some(false));
    assert_eq!(back.needles[0].context_length, Some(8192));
    // `timestamp` was `None` on insert → SQLite's `DEFAULT
    // CURRENT_TIMESTAMP` filled it → it comes back set.
    assert!(back.session.timestamp.is_some());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn persist_then_export_markdown_renders_gfm_tables() {
    let dir = temp_dir("md");
    let db_path = dir.join("benchmarks.db");
    let mut db = Database::open(&db_path).unwrap();
    let session = sample_session();
    db.persist_run(&session, &sample_metrics(&session.session_id))
        .unwrap();
    db.insert_needle_evaluation(&NeedleEvaluation {
        eval_id: None,
        session_id: session.session_id.clone(),
        context_length: Some(8192),
        depth_percent: Some(50.0),
        retrieved_successfully: Some(true),
        latency_ms: Some(210.0),
    })
    .unwrap();

    let payload = ExportPayload::from_stored(&db, &session.session_id).unwrap();
    let md = payload.to_markdown();

    // Acceptance: `--export md` produces a GFM table with the key metrics.
    assert!(md.contains("# Crucible-LLM Benchmark Report"));
    assert!(md.contains("**Model:** qwen3-8b"));
    assert!(md.contains("**Duration:** 3.50 s"));
    assert!(md.contains("| # | Concurrency | Prompt Tokens | Completion Tokens"));
    assert!(
        md.contains("| 1 | 1 | 256 | 128 | 64 | 150.250 | 12.500 | 2.000 | -- | true |"),
        "metric row 1 missing: {md}"
    );
    assert!(
        md.contains("| 2 | 1 | 256 | 130 | -- | 148.000 | 12.700 | 1.980 | -- | false |"),
        "metric row 2 missing: {md}"
    );
    assert!(md.contains("## Needle Evaluations"));
    assert!(md.contains("| 1 | 8192 | 50.0 | true | 210.000 |"));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn csv_export_carries_per_packet_timestamps_and_itl() {
    // The stored path carries no raw packets (header only)…
    let dir = temp_dir("csv-empty");
    let db_path = dir.join("benchmarks.db");
    let mut db = Database::open(&db_path).unwrap();
    let session = sample_session();
    db.persist_run(&session, &sample_metrics(&session.session_id))
        .unwrap();
    let payload = ExportPayload::from_stored(&db, &session.session_id).unwrap();
    assert_eq!(payload.to_csv(), "stream,arrival_ns,itl_ns,kind\n");
    let _ = std::fs::remove_dir_all(&dir);

    // …and the live path attaches the captured per-packet samples.
    let events = synthetic_events();
    let packets = samples_from_events(&events, 1);
    let payload = ExportPayload::from_live(
        sample_session(),
        sample_metrics("aaaabbbb-cccc-dddd-eeee-ffff00001111"),
        packets,
        Vec::new(),
    );
    assert_eq!(payload.packets.len(), 5);

    // Acceptance: `--export csv` includes per-packet timestamps and ITL values.
    let csv = payload.to_csv();
    let lines: Vec<&str> = csv.lines().collect();
    assert_eq!(lines[0], "stream,arrival_ns,itl_ns,kind");
    assert_eq!(lines.len(), 6);
    assert_eq!(lines[1], "1,1000000,,reasoning");
    // ITL = 1_012_000_000 − 1_000_000 (previous *token* frame).
    assert_eq!(lines[2], "1,1012000000,1011000000,content");
    assert_eq!(lines[3], "1,1030000000,18000000,content");
    assert_eq!(lines[4], "1,1031000000,,usage");
    assert_eq!(lines[5], "1,1031500000,,done");
    drop(payload);
}

#[test]
fn all_three_formats_write_to_nested_paths_and_read_back() {
    let dir = temp_dir("write");
    let payload = {
        let packets: Vec<PacketSample> = samples_from_events(&synthetic_events(), 7);
        ExportPayload::from_live(sample_session(), Vec::new(), packets, Vec::new())
    };
    for format in [ExportFormat::Json, ExportFormat::Md, ExportFormat::Csv] {
        let path = write(
            &dir.join("deep/nested")
                .join(format!("out.{}", extension(format))),
            format,
            &payload,
        )
        .unwrap();
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, payload.render(format), "{format:?} mismatch");
    }
    // The JSON file on disk is independently parseable (CI/CD gate shape).
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("deep/nested/out.json")).unwrap())
            .unwrap();
    assert_eq!(v["session"]["model_name"], "qwen3-8b");
    assert_eq!(v["packets"].as_array().unwrap().len(), 5);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn missing_session_is_a_typed_error() {
    let dir = temp_dir("missing");
    let db = Database::open(&dir.join("benchmarks.db")).unwrap();
    let e = ExportPayload::from_stored(&db, "does-not-exist").unwrap_err();
    assert!(matches!(e, ExportError::MissingSession(_)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn write_failure_is_an_io_error() {
    let dir = temp_dir("iofail");
    // Make the "parent directory" a *file*: create_dir_all must fail.
    std::fs::create_dir_all(&dir).unwrap();
    let blocker = dir.join("blocker");
    std::fs::write(&blocker, "i am a file").unwrap();
    let payload = ExportPayload::from_live(sample_session(), Vec::new(), Vec::new(), Vec::new());
    let e = write(
        &blocker.join("impossible.json"),
        ExportFormat::Json,
        &payload,
    )
    .unwrap_err();
    assert!(matches!(e, ExportError::Io(_)));
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn default_path_is_under_the_platform_data_dir() {
    let p = default_path("aaaabbbb-cccc-dddd-eeee-ffff00001111", ExportFormat::Csv);
    assert!(p.starts_with(crucible_llm::config::data_dir()));
    assert!(p.to_string_lossy().ends_with("crucible-aaaabbbb.csv"));
    let j = default_path("short", ExportFormat::Json);
    assert!(j.to_string_lossy().ends_with("crucible-short.json"));
}
