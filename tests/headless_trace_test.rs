//! Offline traceability-completion tests (Wave 6a gaps).
//!
//! The headline test runs the **real headless path** (`crucible_llm::
//! run_headless`) against an in-process mock SSE server and asserts the DB
//! + `--export` file now persist **everything the TUI path does**:
//!
//! 1. **C1 NIAH** → `needle_evaluations` rows are written (the headless path
//!    used to run C1 and drop the result — 0 rows).
//! 2. **Engine A** → `stream_metrics.reasoning_tokens` is populated from the
//!    CF-02 `UsageDetails` seam (used to be `NULL`).
//! 3. **Engine F** → the persisted / exported `f_flatout` carries the CF-04
//!    dual denominators (`tps_per_completed_stream` /
//!    `tps_per_requested_stream` / `failed_stream_ratio`) that were only in
//!    the `--json` stdout.
//!
//! Everything is hermetic: the mock server is in-process, and the database
//! is redirected to a temp `XDG_DATA_HOME` (the env-modifying test is
//! serialized on a mutex so it can never race a sibling test's data dir).
//! `run_headless` is a *sync* fn that builds its own `current_thread`
//! runtime, so the test is a plain `#[test]` (not `#[tokio::test]`) and the
//! mock server runs on a separate `Runtime` — a `block_on` from inside an
//! async context would panic.
//!
//! A few fast, dependency-free unit tests pin the individual model seams
//! (no server, no DB, no env).

use std::sync::{Mutex, PoisonError};

use crucible_llm::config::{Config, EngineSelection, ExportFormat};
use crucible_llm::engines::flatout::FlatOutResult;
use crucible_llm::engines::hardware::EnergyResult;
use crucible_llm::log::RunLogger;
use crucible_llm::run_headless;
use crucible_llm::sse::UsageDetails;
use crucible_llm::storage::models::{EnergySummary, EngineResults, FlatOutSummary};
use crucible_llm::storage::{Database, StreamMetricRow};

mod common;
use common::mock::*;

/// Serializes the one test that mutates `XDG_DATA_HOME` (process-global).
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn lock_env() -> std::sync::MutexGuard<'static, ()> {
    ENV_LOCK
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
}

// ── Mock server: routes by prompt content ─────────────────────────────────

/// The Engine A / prefill stream: a vLLM-style SSE with a `usage` block that
/// carries the CF-02 **detail** fields (`completion_tokens_details.
/// reasoning_tokens`), so the closed seam populates `reasoning_tokens`.
fn speed_frames() -> Vec<String> {
    vec![
        FRAME_ROLE.to_string(),
        "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"Hello, \"}}]}\n\n"
            .to_string(),
        "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"world!\"}}]}\n\n"
            .to_string(),
        "data: {\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":34,\"completion_tokens_details\":{\"reasoning_tokens\":20}}}\n\n"
            .to_string(),
        FRAME_DONE.to_string(),
    ]
}

/// The Engine C1 (NIAH) response: the mock extracts the hidden needle value
/// from the prompt and echoes it as content, so the cell is *retrieved*.
fn niah_frames(value: &str) -> Vec<String> {
    vec![
        FRAME_ROLE.to_string(),
        format!(
            "data: {{\"id\":\"cmpl-1\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"{value}\"}}}}]}}\n\n"
        ),
        FRAME_DONE.to_string(),
    ]
}

/// The Engine F (Flat Out) stream: a short, complete (terminating) stream so
/// the 60 s window closes early (all streams end on their own) and the run
/// stays fast.
fn flatout_frames() -> Vec<String> {
    vec![
        FRAME_ROLE.to_string(),
        "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The keeper \"}}]}\n\n"
            .to_string(),
        "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"descended the stairs.\"}}]}\n\n"
            .to_string(),
        FRAME_DONE.to_string(),
    ]
}

/// The hidden needle value: the sentence is `IMPORTANT: the secret code is
/// <VALUE>. Remember …` — read `<VALUE>` (up to the next space / period).
fn extract_needle(prompt: &str) -> String {
    prompt
        .find("the secret code is ")
        .map(|i| {
            let rest = &prompt[i + "the secret code is ".len()..];
            rest.split(|c| c == ' ' || c == '.')
                .next()
                .unwrap_or("")
                .to_string()
        })
        .unwrap_or_default()
}

async fn start_trace_mock() -> (String, tokio::task::JoinHandle<()>) {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let (_headers, body) = read_request(&mut sock).await;
            let prompt = prompt_from_body(&body).unwrap_or_default();
            if prompt.contains("the secret code is") {
                serve_sse(&mut sock, niah_frames(&extract_needle(&prompt)), std::time::Duration::ZERO)
                    .await;
            } else if prompt.contains("lighthouse keeper") {
                serve_sse(&mut sock, flatout_frames(), std::time::Duration::ZERO).await;
            } else {
                serve_sse(&mut sock, speed_frames(), std::time::Duration::ZERO).await;
            }
        }
    });
    (format!("http://{addr}"), handle)
}

// ── The headline integration test ─────────────────────────────────────────

#[test]
fn headless_run_persists_needles_reasoning_tokens_and_f_dual_denominators() {
    let _env = lock_env();

    // Hermetic data dir: `data_dir()` = `$XDG_DATA_HOME/crucible`.
    let base =
        std::env::temp_dir().join(format!("crucible-trace-test-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&base);
    let data_home = base.join("data");
    let export_dir = base.join("exports");
    std::fs::create_dir_all(&data_home).unwrap();
    std::fs::create_dir_all(&export_dir).unwrap();
    std::env::set_var("XDG_DATA_HOME", &data_home);

    // The mock server runs on its own runtime (run_headless builds a nested
    // current_thread runtime, so this test stays a sync #[test]).
    let rt = tokio::runtime::Runtime::new().unwrap();
    let (url, server) = rt.block_on(start_trace_mock());

    let session_id = "trace-test-session-0001";
    let cfg = Config {
        url,
        model: "test-model".to_string(),
        iterations: 1,
        timeout: 10,
        hardware: false, // no poller thread (the energy N/A path is unit-tested)
        json: false,
        no_color: true,
        export: Some(ExportFormat::Json),
        export_path: Some(export_dir.join("trace-export.json")),
        engines: EngineSelection {
            speed: true,
            concurrency: false,
            niah: true,
            reasoning: false,
            structured: false,
            hardware: false,
            flatout: true,
        },
        ..Config::default()
    };

    let logger = RunLogger::start(None);
    let code = run_headless(&cfg, logger, None, None, session_id.to_string());

    server.abort();
    drop(rt);
    std::env::remove_var("XDG_DATA_HOME");

    assert_eq!(code, 0, "the headless run must succeed (exit 0)");

    // Open the hermetic DB and assert the three traceability fixes.
    let db_path = data_home.join("crucible").join("benchmarks.db");
    let db = Database::open(&db_path).unwrap();

    // 1. C1 NIAH: all 77 cells (7 sizes × 11 depths) are persisted.
    let needles = db.get_needle_evaluations(session_id).unwrap();
    assert_eq!(
        needles.len(),
        77,
        "the headless path must persist every NIAH cell to needle_evaluations"
    );
    assert!(
        needles.iter().all(|n| n.context_length.is_some() && n.depth_percent.is_some()),
        "each needle row carries its context length + depth"
    );

    // 2. Engine A: reasoning_tokens is populated (the CF-02 seam), not NULL.
    let metrics = db.get_stream_metrics(session_id).unwrap();
    assert!(!metrics.is_empty(), "the speed run persisted stream_metrics rows");
    assert!(
        metrics.iter().any(|m| m.reasoning_tokens == Some(20)),
        "a stream_metrics row must carry the reported reasoning_tokens (20), got: {:?}",
        metrics
            .iter()
            .map(|m| m.reasoning_tokens)
            .collect::<Vec<_>>()
    );

    // 3. Engine F: the persisted f_flatout carries the dual denominators.
    let session = db.get_session(session_id).unwrap().expect("session row");
    let er: EngineResults =
        serde_json::from_str(session.engine_results.as_deref().unwrap())
            .expect("engine_results parses");
    let f = er.f_flatout.expect("f_flatout must be persisted");
    assert!(
        f.tps_per_completed_stream > 0.0,
        "capacity view (÷ completed) present: {}",
        f.tps_per_completed_stream
    );
    assert!(
        f.tps_per_requested_stream > 0.0,
        "user-experience view (÷ requested) present: {}",
        f.tps_per_requested_stream
    );
    assert!(
        f.failed_stream_ratio >= 0.0,
        "failed-stream ratio present: {}",
        f.failed_stream_ratio
    );

    // 4. The --export JSON (built *from the DB*) carries the same detail.
    let export =
        std::fs::read_to_string(export_dir.join("trace-export.json")).expect("export file");
    let v: serde_json::Value = serde_json::from_str(&export).expect("valid export JSON");
    let f_export = v["engine_results"]["f_flatout"]
        .as_object()
        .expect("export f_flatout object");
    assert!(
        f_export.get("tps_per_completed_stream").is_some(),
        "export F carries tps_per_completed_stream"
    );
    assert!(
        f_export.get("tps_per_requested_stream").is_some(),
        "export F carries tps_per_requested_stream"
    );
    assert!(
        f_export.get("failed_stream_ratio").is_some(),
        "export F carries failed_stream_ratio"
    );
    assert_eq!(
        v["needles"].as_array().map(|a| a.len()),
        Some(77),
        "the DB-built export carries the 77 needle rows"
    );

    let _ = std::fs::remove_dir_all(&base);
}

// ── Fast, dependency-free unit tests for the new model seams ──────────────

#[test]
fn aggregate_populates_reasoning_tokens_from_the_cf02_detail() {
    // A minimal completed run whose usage block reported 20 reasoning tokens.
    use crucible_llm::engines::speed::aggregate;
    use crucible_llm::prompt::GeneratedPrompt;
    use crucible_llm::client::StreamOutcome;
    use crucible_llm::timing::StreamTimestamps;

    let prompt = GeneratedPrompt {
        text: "hi".into(),
        token_count: 2,
        estimated: true,
        nocache: false,
    };
    let outcome = StreamOutcome {
        timestamps: StreamTimestamps::default(),
        usage: Some(crucible_llm::sse::Usage {
            prompt_tokens: 10,
            completion_tokens: 30,
        }),
        usage_details: Some(UsageDetails {
            reasoning_tokens: 20,
            cached_tokens: 0,
        }),
        premature: false,
        malformed_frames: 0,
        error: None,
        looping: false,
        loop_excluded_tokens: 0,
        frames: 2,
    };
    let r = aggregate(&Config::default(), &prompt, &outcome, &[], None, outcome.usage_details);
    assert_eq!(
        r.reasoning_tokens,
        Some(20),
        "aggregate must publish the reported reasoning token count"
    );
    // A run whose backend reported no detail stays `None` (honest NULL).
    let no_detail = StreamOutcome {
        usage_details: None,
        ..outcome
    };
    let r2 = aggregate(&Config::default(), &prompt, &no_detail, &[], None, None);
    assert_eq!(r2.reasoning_tokens, None, "no detail → NULL, not a fabricated 0");
}

#[test]
fn stream_metric_row_carries_the_reasoning_tokens() {
    // The persistence builder wires the (now-populated) SpeedResult field
    // into the `stream_metrics.reasoning_tokens` column.
    use crucible_llm::engines::speed::SpeedResult;
    let r = SpeedResult {
        ttft: 0.1,
        prompt_tokens: 100,
        completion_tokens: 30,
        reasoning_tokens: Some(20),
        pp_speed: 1000.0,
        tg_speed: 50.0,
        mtp_efficiency: 1.0,
        stream_time: 1.0,
        total_chunks: 30,
        content_chunks: 10,
        reasoning_chunks: 20,
        other_chunks: 0,
        estimated: false,
        model: "m".into(),
        mode: "short".into(),
        error: None,
        looping: false,
        usage_discrepancy: false,
        e2e_only: false,
        itl_clamped: 0,
    };
    let mut row = StreamMetricRow::from_speed_result(&r, "sess", 1);
    assert_eq!(row.reasoning_tokens, None, "from_speed_result leaves it NULL");
    if let Some(t) = r.reasoning_tokens {
        row = row.with_reasoning_tokens(t);
    }
    assert_eq!(row.reasoning_tokens, Some(20), "the builder populates it");
}

#[test]
fn flatout_summary_carries_the_dual_denominators() {
    let r = FlatOutResult {
        stream_count: 4,
        total_tokens: 12_828,
        duration_secs: 60.0,
        aggregate_tps: 213.8,
        per_stream_tps: 53.4,
        tps_per_completed_stream: 71.27,
        tps_per_requested_stream: 53.4,
        failed_stream_ratio: 0.25,
        ..minimal_flatout()
    };
    let s = FlatOutSummary::from_flatout(&r);
    assert!((s.tps_per_completed_stream - 71.27).abs() < 1e-9);
    assert!((s.tps_per_requested_stream - 53.4).abs() < 1e-9);
    assert!((s.failed_stream_ratio - 0.25).abs() < 1e-9);
    // And the JSON round-trips them (the persisted / exported shape).
    let json = serde_json::to_string(&s).unwrap();
    let back: FlatOutSummary = serde_json::from_str(&json).unwrap();
    assert!((back.tps_per_completed_stream - 71.27).abs() < 1e-9);
    assert!((back.failed_stream_ratio - 0.25).abs() < 1e-9);
    // Pre-existing stored JSON without the fields still deserializes.
    let legacy: FlatOutSummary =
        serde_json::from_str(r#"{"stream_count":4,"total_tokens":1,"duration_secs":1.0,"aggregate_tps":1.0,"per_stream_tps":1.0}"#)
            .unwrap();
    assert_eq!(legacy.tps_per_completed_stream, 0.0);
}

fn minimal_flatout() -> FlatOutResult {
    FlatOutResult {
        stream_count: 1,
        total_tokens: 0,
        duration_secs: 0.0,
        aggregate_tps: 0.0,
        per_stream_tps: 0.0,
        tps_per_completed_stream: 0.0,
        tps_per_requested_stream: 0.0,
        failed_stream_ratio: 0.0,
        config_error: None,
        max_tokens_used: 0,
        ttft_avg_ms: 0.0,
        itl_p50_ms: 0.0,
        itl_p99_ms: 0.0,
        tokens_estimated: false,
        counting_method: "none".into(),
        stream_count_source: crucible_llm::engines::StreamCountSource::Default,
    }
}

#[test]
fn energy_summary_labels_the_na_case_clearly() {
    // No power telemetry (the RAPL-quirk / driverless case): an honest,
    // labeled N/A — never a silent gap, never a spurious 0.0 J/token.
    let na = EnergyResult {
        joules: 0.0,
        joules_per_token: None,
        avg_power_w: None,
        peak_power_w: None,
        peak_vram_bytes: None,
        peak_vram_ratio: None,
        fragmentation_warning: None,
        avg_utilization_pct: None,
        vendor: None,
        model: None,
        cpu_power_estimated: false,
    };
    assert!(na.is_na(), "no power readings → N/A");
    let s = EnergySummary::from_energy(&na);
    assert_eq!(s.na_reason.as_deref(), Some("N/A — no power telemetry"));

    // A profile with power is not N/A (no reason label).
    let powered = EnergyResult {
        joules: 100.0,
        avg_power_w: Some(200.0),
        peak_power_w: Some(300.0),
        ..na
    };
    assert!(!powered.is_na());
    assert_eq!(EnergySummary::from_energy(&powered).na_reason, None);
}
