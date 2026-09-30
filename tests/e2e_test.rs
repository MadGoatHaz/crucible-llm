//! Chunk 19 — end-to-end smoke test (plan Chunk 19 acceptance):
//!
//! 1. **In-process full pipeline** against an in-process mock SSE server:
//!    single-stream (Engine A) → small ladder sweep (Engine B) →
//!    needle-in-a-haystack (Engine C1) → SQLite persistence → all three
//!    export formats (JSON / Markdown / CSV, blueprint §8).
//! 2. **The real binary as a subprocess**: headless `--json` + `--export
//!    json` against the mock, with the platform data dir redirected into a
//!    throwaway temp dir (no user data is touched) — verifying the
//!    prototype's JSON field set, the persisted session/metric rows, and
//!    the written export file.
//! 3. **Bare invocation** prints the banner and exits 0.
//!
//! The mock simulates a *well-behaved* endpoint without any LLM in the
//! loop: a NIAH-style prompt (one that embeds `the secret code is …`)
//! gets the needle value echoed back; any other prompt gets a short
//! two-frame completion with a `usage` block and `[DONE]`.

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;

use crucible_llm::client::pool::WorkerPool;
use crucible_llm::config::{Config, ExportFormat};
use crucible_llm::engines::{NiahEngine, SpeedEngine, Sweep};
use crucible_llm::storage::export::{self, ExportPayload};
use crucible_llm::storage::{BenchmarkSession, Database, StreamMetricRow};

// ── Mock SSE server (multi-connection) ───────────────────────────────────

/// Read the full HTTP request (headers + body) and return
/// `messages[0].content` from the JSON body.
async fn read_request(sock: &mut tokio::net::TcpStream) -> String {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 8192];
    loop {
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => {
                buf.extend_from_slice(&tmp[..n]);
                if buf.windows(4).any(|w| w == b"\r\n\r\n") {
                    break;
                }
            }
        }
    }
    let header_end = buf
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .map(|p| p + 4)
        .unwrap_or(buf.len());
    let headers = String::from_utf8_lossy(&buf[..header_end]).to_ascii_lowercase();
    let mut content_length = 0usize;
    for line in headers.lines() {
        if let Some(rest) = line.strip_prefix("content-length:") {
            content_length = rest.trim().parse().unwrap_or(0);
        }
    }
    let mut body = buf[header_end..].to_vec();
    while body.len() < content_length {
        match sock.read(&mut tmp).await {
            Ok(0) | Err(_) => break,
            Ok(n) => body.extend_from_slice(&tmp[..n]),
        }
    }
    let body = body[..content_length.min(body.len())].to_vec();
    serde_json::from_slice::<serde_json::Value>(&body)
        .ok()
        .and_then(|v| v["messages"][0]["content"].as_str().map(String::from))
        .unwrap_or_default()
}

/// The needle value as the NIAH engine injects it into the prompt
/// ("IMPORTANT: the secret code is {value}. …").
fn extract_needle_value(prompt: &str) -> Option<String> {
    let marker = "the secret code is ";
    let start = prompt.find(marker)? + marker.len();
    let rest = &prompt[start..];
    let end = rest.find(['.', ' '])?;
    Some(rest[..end].to_string())
}

/// A well-behaved mock endpoint: it accepts connections in a loop, reads
/// the request, and serves one SSE stream per connection
/// (`Connection: close`).
async fn start_mock() -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let prompt = read_request(&mut sock).await;
            respond(&mut sock, prompt).await;
        }
    });
    (format!("http://{addr}"), handle)
}

/// Serve one completion for `prompt`:
/// * NIAH-style prompt → the model "retrieves" the needle value;
/// * anything else → a short two-content-frame answer.
///
/// Frame order: role opener, content deltas, `usage`, `[DONE]` — the
/// vLLM shape the SSE parser (Chunk 3) consumes.
async fn respond(sock: &mut tokio::net::TcpStream, prompt: String) {
    write_all(
        sock,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )
    .await;

    let (answer, completion_tokens) = match extract_needle_value(&prompt) {
        Some(value) => {
            let tokens = value.chars().count() as u64;
            (value, tokens)
        }
        None => ("The answer is 42.".to_string(), 6),
    };
    let prompt_tokens = (prompt.chars().count() / 4).max(1) as u64;

    // Split the answer across two content frames (a realistic decode).
    let mid = answer
        .char_indices()
        .nth(1)
        .map(|(i, _)| i)
        .unwrap_or(answer.len());
    let (part1, part2) = answer.split_at(mid);

    let role =
        "data: {\"id\":\"cmpl-e2e\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n"
            .to_string();
    let c1 = format!(
        "data: {{\"id\":\"cmpl-e2e\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(part1).unwrap()
    );
    let c2 = format!(
        "data: {{\"id\":\"cmpl-e2e\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(part2).unwrap()
    );
    let usage = format!(
        "data: {{\"id\":\"cmpl-e2e\",\"choices\":[],\"usage\":{{\"prompt_tokens\":{prompt_tokens},\"completion_tokens\":{completion_tokens}}}}}\n\n"
    );
    let done = "data: [DONE]\n\n".to_string();
    for frame in [role, c1, c2, usage, done] {
        write_chunk(sock, frame.as_bytes()).await;
    }
    write_chunk(sock, b"").await; // terminating chunk
    let _ = sock.shutdown().await;
}

async fn write_all(sock: &mut tokio::net::TcpStream, bytes: &[u8]) {
    let _ = sock.write_all(bytes).await;
    let _ = sock.flush().await;
}

/// Write one `Transfer-Encoding: chunked` frame.
async fn write_chunk(sock: &mut tokio::net::TcpStream, payload: &[u8]) {
    let mut msg = format!("{:x}\r\n", payload.len());
    msg.push_str(&String::from_utf8_lossy(payload));
    msg.push_str("\r\n");
    write_all(sock, msg.as_bytes()).await;
}

// ── Helpers ───────────────────────────────────────────────────────────────

/// A throwaway temp dir unique to this test binary invocation.
fn temp_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "crucible-e2e-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn test_config(url: &str) -> Config {
    Config {
        url: url.to_string(),
        model: "e2e-model".to_string(),
        timeout: 15,
        ..Config::default()
    }
}

// ── 1. In-process full pipeline ──────────────────────────────────────────

#[tokio::test]
async fn full_pipeline_single_stream_sweep_niah_persist_and_export() {
    let (url, _server) = start_mock().await;
    let cfg = test_config(&url);

    // ── Engine A: single-stream (TTFT / PP / TG / MTP, blueprint §7) ──
    let engine = SpeedEngine::new(&cfg).unwrap();
    let prompt = engine.generate_prompt();
    let (result, events) = engine.run_iteration_events(&prompt).await;
    assert!(
        !result.is_failed(),
        "Engine A run failed: {:?}",
        result.error
    );
    assert!(result.ttft > 0.0, "TTFT measured");
    assert!(result.pp_speed > 0.0, "prefill throughput measured");
    assert!(result.tg_speed > 0.0, "generation throughput measured");
    assert!(result.mtp_efficiency > 0.0, "MTP ratio measured");
    assert!(result.completion_tokens > 0, "usage captured");
    // The raw per-packet samples the CSV export dumps:
    let packets = export::samples_from_events(&events, 1);
    assert!(!packets.is_empty(), "packet samples captured");

    // ── Engine B: small ladder sweep (1 → 2 → 4) ──
    let client = reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .build()
        .unwrap();
    let pool = WorkerPool::new(client, &url, "e2e-model", &prompt.text, 16)
        .read_timeout(Duration::from_secs(10));
    let sweep = Sweep::new(pool, vec![1, 2, 4]);
    let sweep_result = sweep.run().await;

    assert_eq!(sweep_result.levels.len(), 3, "one level per ladder step");
    for level in &sweep_result.levels {
        assert_eq!(
            level.failed_streams, 0,
            "level {} had no failures",
            level.concurrency
        );
        assert!(level.total_tokens > 0, "level {} tokens", level.concurrency);
        assert!(
            level.aggregate_tps > 0.0,
            "level {} aggregate throughput",
            level.concurrency
        );
        assert!(
            level.p90_tpot_ns > 0,
            "level {} p90 TPOT",
            level.concurrency
        );
    }
    // The Curve analysis seam (Chunk 11) sees a non-empty result:
    assert!(sweep_result.peak_throughput().is_some());

    // ── Engine C1: NIAH (2k context × 3 depths) ──
    let niah = NiahEngine::new(&cfg)
        .unwrap()
        .sizes(vec![2000])
        .depths(vec![0, 50, 100]);
    let niah_result = niah.run().await;
    assert_eq!(niah_result.cells.len(), 3, "one cell per depth");
    assert_eq!(niah_result.accuracy(), (3, 3), "the mock always retrieves");

    // ── Persistence: SQLite in a throwaway temp dir ──
    let tmp = temp_dir();
    let db_path = tmp.join("crucible").join("benchmarks.db");
    let mut db = Database::open(&db_path).unwrap();
    let session = BenchmarkSession {
        session_id: uuid::Uuid::new_v4().to_string(),
        timestamp: None, // the DDL default fills it
        target_url: url.clone(),
        model_name: "e2e-model".to_string(),
        backend_type: None,
        quantization: None,
        system_gpu: None,
        total_duration_sec: Some(result.stream_time),
    };
    let rows = vec![StreamMetricRow::from_speed_result(
        &result,
        &session.session_id,
        1,
    )];
    db.persist_run(&session, &rows).unwrap();
    for row in niah_result.to_needle_rows(&session.session_id) {
        db.insert_needle_evaluation(&row).unwrap();
    }

    // Read the rows back (the same seam the History view / exporters use):
    let stored = db
        .get_session(&session.session_id)
        .unwrap()
        .expect("session row stored");
    assert_eq!(stored.model_name, "e2e-model");
    assert_eq!(stored.target_url, url);
    assert_eq!(db.get_stream_metrics(&session.session_id).unwrap().len(), 1);
    assert_eq!(
        db.get_needle_evaluations(&session.session_id)
            .unwrap()
            .len(),
        3
    );

    // ── Export: all three formats (blueprint §8) ──
    let mut payload = ExportPayload::from_stored(&db, &session.session_id).unwrap();
    payload.packets = packets;

    let json_path = export::write(
        &tmp.join("exports").join("e2e.json"),
        ExportFormat::Json,
        &payload,
    )
    .unwrap();
    let md_path = export::write(
        &tmp.join("exports").join("e2e.md"),
        ExportFormat::Md,
        &payload,
    )
    .unwrap();
    let csv_path = export::write(
        &tmp.join("exports").join("e2e.csv"),
        ExportFormat::Csv,
        &payload,
    )
    .unwrap();

    // JSON: valid, parseable, and referencing the stored session (the
    // CI/CD regression-gating shape).
    let json = std::fs::read_to_string(&json_path).unwrap();
    let v: serde_json::Value = serde_json::from_str(&json).expect("export must be valid JSON");
    assert_eq!(v["session"]["session_id"], session.session_id);
    assert_eq!(v["metrics"].as_array().unwrap().len(), 1);
    assert_eq!(v["needles"].as_array().unwrap().len(), 3);
    assert!(!v["packets"].as_array().unwrap().is_empty());

    // Markdown: a GFM document with the model name and table structure.
    let md = std::fs::read_to_string(&md_path).unwrap();
    assert!(md.contains("e2e-model"), "metadata carries the model");
    assert!(md.contains('|'), "GFM table present");

    // CSV: the header + at least one per-packet row.
    let csv = std::fs::read_to_string(&csv_path).unwrap();
    assert!(
        csv.starts_with("stream,arrival_ns,itl_ns,kind"),
        "CSV header: {csv}"
    );
    assert!(csv.lines().count() > 1, "CSV has data rows");

    std::fs::remove_dir_all(&tmp).ok();
}

// ── 2. The real binary, as a subprocess ──────────────────────────────────

/// The test binary path cargo provides for the package's `crucible-llm`
/// bin target (built by `cargo test` before the integration tests run).
fn binary() -> PathBuf {
    PathBuf::from(
        std::env::var("CARGO_BIN_EXE_crucible-llm")
            .expect("CARGO_BIN_EXE_crucible-llm must be provided by cargo"),
    )
}

#[test]
fn binary_headless_json_and_export_end_to_end() {
    // A multi-thread runtime: the mock server must keep accepting while
    // this test thread blocks on the subprocess.
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_io()
        .enable_time()
        .worker_threads(2)
        .build()
        .unwrap();
    let (url, _server) = rt.block_on(start_mock());

    // Isolate the platform data dir: `dirs::data_dir()` on Linux honors
    // `XDG_DATA_HOME`, so the binary's SQLite + exports land in a
    // throwaway dir — the user's real data is never touched.
    let tmp = temp_dir();
    let data_home = tmp.join("xdg-data");
    let export_path = tmp.join("exports").join("binary-e2e.json");

    let out = Command::new(binary())
        .args([
            "--url",
            url.as_str(),
            "--model",
            "e2e-binary-model",
            "--mode",
            "short",
            "--iterations",
            "2",
            "--timeout",
            "15",
            "--json",
            "--no-color",
            "--export",
            "json",
            "--export-path",
            export_path.to_str().unwrap(),
        ])
        .env_clear()
        .env("HOME", &tmp)
        .env("XDG_DATA_HOME", &data_home)
        .output()
        .expect("spawn the crucible-llm binary");

    assert!(
        out.status.success(),
        "exit: {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );

    // stdout is pure JSON with the prototype's `output_json` field set:
    let report: serde_json::Value =
        serde_json::from_slice(&out.stdout).expect("stdout must be valid JSON");
    assert_eq!(report["url"], url);
    assert_eq!(report["model"], "e2e-binary-model");
    assert_eq!(report["iterations"], 2);
    let results = report["results"].as_array().expect("results array");
    assert_eq!(results.len(), 2);
    for r in results {
        assert!(r["error"].is_null(), "no run failed: {:?}", r["error"]);
        assert!((r["ttft_s"].as_f64().unwrap_or(0.0)) > 0.0, "TTFT > 0");
        assert!(
            (r["tg_speed_tok_s"].as_f64().unwrap_or(0.0)) > 0.0,
            "TG speed > 0"
        );
        assert!(
            r["completion_tokens"].as_u64().unwrap_or(0) > 0,
            "usage captured"
        );
    }
    // Two valid iterations → the prototype's summary block:
    assert!(
        report["summary"]["avg_tg_speed"].is_number(),
        "summary present"
    );

    // SQLite: the run persisted into the isolated data dir —
    // one session row, one metric row per iteration.
    let db = Database::open(&data_home.join("crucible").join("benchmarks.db")).unwrap();
    let sessions = db.list_sessions().unwrap();
    assert_eq!(sessions.len(), 1, "one session persisted");
    assert_eq!(sessions[0].model_name, "e2e-binary-model");
    assert_eq!(
        db.get_stream_metrics(&sessions[0].session_id)
            .unwrap()
            .len(),
        2,
        "one metric row per iteration"
    );

    // `--export json`: the file exists, is valid JSON, and references the
    // persisted session.
    let export = std::fs::read_to_string(&export_path).expect("export file written");
    let v: serde_json::Value = serde_json::from_str(&export).expect("export is valid JSON");
    assert_eq!(v["session"]["session_id"], sessions[0].session_id);

    std::fs::remove_dir_all(&tmp).ok();
}

#[test]
fn binary_bare_invocation_prints_banner_and_exits_zero() {
    let tmp = temp_dir();
    let out = Command::new(binary())
        .env_clear()
        .env("HOME", &tmp)
        .output()
        .expect("spawn the crucible-llm binary");

    assert!(
        out.status.success(),
        "exit: {:?}\nstderr: {}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8(out.stdout).expect("utf-8 banner");
    assert!(stdout.contains("Crucible-LLM"), "banner: {stdout}");
    assert!(stdout.contains("--help"), "banner points at --help");

    std::fs::remove_dir_all(&tmp).ok();
}
