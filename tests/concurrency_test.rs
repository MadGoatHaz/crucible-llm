//! Chunk 10 — multi-stream pool + concurrency ladder sweep acceptance
//! tests.
//!
//! Acceptance (plan Chunk 10):
//!
//! * a sweep against the mock server produces **one (concurrency,
//!   aggregate_tps, p90_tpot) record per ladder step**;
//! * **no connection leaks** — the fd count is stable after shutdown;
//! * **aggregate throughput and p90 are computed across all concurrent
//!   streams** of the level (not just one).
//!
//! Plus Chunk 11 acceptance: knee-point detection on synthetic sweep data
//! (the detected knee matches the injected inflection) and View 2 render
//! checks — the placeholder matrix before a sweep, and the real curve with
//! the knee / optimal-operational-envelope highlighted after one.
//!
//! The mock is an in-process `tokio` `TcpListener` that accepts *many*
//! concurrent connections (one task per connection, vLLM-style SSE with a
//! small per-frame delay so inter-token latencies are real), keeping the
//! suite fully offline.

use std::time::Duration;

use ratatui::backend::TestBackend;
use ratatui::Terminal;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;

use crucible_llm::client::pool::WorkerPool;
use crucible_llm::engines::concurrency::{Sweep, SweepLevel, SweepResult, DEFAULT_LADDER};
use crucible_llm::metrics::state::{StreamMetric, StreamStatus};
use crucible_llm::ui::app::{App, View};
use crucible_llm::ui::views::concurrency;

// ── Mock SSE payloads (vLLM-style) ────────────────────────────────────────

fn role_frame() -> String {
    r#"data: {"id":"cmpl-1","choices":[{"index":0,"delta":{"role":"assistant","content":""}}]}"#
        .to_string()
}

fn content_frame(i: usize) -> String {
    format!(
        "data: {{\"id\":\"cmpl-1\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":\"tok{i} \"}}}}]}}"
    )
}

fn usage_frame(completion_tokens: u64) -> String {
    format!(
        "data: {{\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{{\"prompt_tokens\":64,\"completion_tokens\":{completion_tokens}}}}}"
    )
}

const DONE_FRAME: &str = r#"data: [DONE]"#;

// ── Multi-connection mock SSE server ──────────────────────────────────────

/// Start a mock that accepts an unbounded number of *concurrent*
/// connections, serving each one a vLLM-style SSE stream: role opener,
/// `content_frames` token frames (spaced by `frame_delay`), usage, `[DONE]`.
async fn start_mock_sse(
    content_frames: usize,
    frame_delay: Duration,
) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        loop {
            let (sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            tokio::spawn(async move {
                let mut sock = sock;
                drain_request(&mut sock).await;
                serve_stream(&mut sock, content_frames, frame_delay).await;
            });
        }
    });
    (format!("http://{addr}"), handle)
}

/// Read the request until the header block is complete.
async fn drain_request(sock: &mut tokio::net::TcpStream) {
    let mut buf = [0u8; 8192];
    let mut acc = Vec::new();
    use tokio::io::AsyncReadExt;
    loop {
        match sock.read(&mut buf).await {
            Ok(0) => break,
            Ok(n) => {
                acc.extend_from_slice(&buf[..n]);
                if acc.windows(4).any(|w| w == b"\r\n\r\n") || acc.len() > 65536 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
}

async fn serve_stream(
    sock: &mut tokio::net::TcpStream,
    content_frames: usize,
    frame_delay: Duration,
) {
    write_all(
        sock,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
    )
    .await;
    let mut frames = vec![role_frame()];
    for i in 0..content_frames {
        frames.push(content_frame(i));
    }
    frames.push(usage_frame(content_frames as u64));
    frames.push(DONE_FRAME.to_string());
    for frame in &frames {
        if frame_delay > Duration::ZERO {
            tokio::time::sleep(frame_delay).await;
        }
        write_chunk(sock, format!("{frame}\n\n").as_bytes()).await;
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

fn test_pool(url: &str) -> WorkerPool {
    WorkerPool::new(reqwest::Client::new(), url, "test-model", "Hello there", 16)
        .read_timeout(Duration::from_secs(10))
}

/// Count open `socket:` file descriptors in this process (Linux `/proc`).
/// `None` when the platform does not expose `/proc/self/fd`.
fn socket_fd_count() -> Option<usize> {
    let mut count = 0;
    let entries = std::fs::read_dir("/proc/self/fd").ok()?;
    for entry in entries.flatten() {
        let target = std::fs::read_link(entry.path()).ok()?;
        if target.to_string_lossy().starts_with("socket:") {
            count += 1;
        }
    }
    Some(count)
}

/// Render the Concurrency view at 120x40 and return the flat buffer text.
fn render_concurrency(app: &App) -> String {
    let backend = TestBackend::new(120, 40);
    let mut terminal = Terminal::new(backend).expect("TestBackend terminal");
    terminal
        .draw(|f| concurrency::render(f.area(), app, f))
        .expect("render frame");
    terminal
        .backend()
        .buffer()
        .content()
        .iter()
        .map(|c| c.symbol())
        .collect()
}

fn level(concurrency: usize, tps: f64, p90_ms: f64) -> SweepLevel {
    SweepLevel {
        concurrency,
        aggregate_tps: tps,
        p50_tpot_ns: (p90_ms * 0.8e6) as u64,
        p90_tpot_ns: (p90_ms * 1e6) as u64,
        p99_tpot_ns: (p90_ms * 2e6) as u64,
        ttft_p50_ns: 10_000_000,
        ttft_p90_ns: 20_000_000,
        total_tokens: 100,
        completed_streams: concurrency,
        failed_streams: 0,
        wall_ns: 1_000_000_000,
        streams: (0..concurrency as u32)
            .map(|id| StreamMetric {
                id,
                kind: "Content".into(),
                state: StreamStatus::Done,
                pp_tokens: Some(64),
                tg_tokens: Some(25),
                ttft_s: Some(0.01),
                gen_tps: Some(25.0),
                mtp: Some(1.0),
                progress: 1.0,
            })
            .collect(),
    }
}

// ── Acceptance: one record per ladder step ────────────────────────────────

#[tokio::test]
async fn sweep_produces_one_record_per_ladder_step() {
    let (url, server) = start_mock_sse(3, Duration::from_millis(5)).await;
    let sweep = Sweep::new(test_pool(&url), [1, 2, 4]);
    let result = sweep.run().await;
    server.abort();

    // One (concurrency, aggregate_tps, p90_tpot) record per ladder step.
    assert_eq!(result.levels.len(), 3);
    assert_eq!(
        result
            .levels
            .iter()
            .map(|l| l.concurrency)
            .collect::<Vec<_>>(),
        vec![1, 2, 4]
    );
    for lvl in &result.levels {
        assert!(
            lvl.aggregate_tps > 0.0,
            "level {} has positive aggregate t/s",
            lvl.concurrency
        );
        assert!(
            lvl.p90_tpot_ns > 0,
            "level {} has a p90 TPOT",
            lvl.concurrency
        );
        // Every stream of the level completes.
        assert_eq!(lvl.completed_streams, lvl.concurrency);
        assert_eq!(lvl.failed_streams, 0);
        // usage reports 3 completion tokens per stream.
        assert_eq!(lvl.total_tokens, 3 * lvl.concurrency as u64);
        // Per-stream detail rows exist for every worker.
        assert_eq!(lvl.streams.len(), lvl.concurrency);
    }
}

// ── Acceptance: aggregate metrics span ALL concurrent streams ─────────────

#[tokio::test]
async fn aggregate_metrics_span_all_concurrent_streams() {
    let (url, server) = start_mock_sse(4, Duration::from_millis(10)).await;
    let sweep = Sweep::new(test_pool(&url), [4]);
    let result = sweep.run().await;
    server.abort();

    let lvl = &result.levels[0];
    assert_eq!(lvl.concurrency, 4);

    // Total tokens = 4 streams × 4 tokens (usage is the source of truth).
    assert_eq!(lvl.total_tokens, 16);

    // The p90 TPOT is the pooled inter-token latency across all four
    // streams: each stream paces its frames ~10 ms apart, so the pooled
    // p90 must sit near that interval (scheduling jitter tolerated).
    assert!(
        (5_000_000..60_000_000).contains(&lvl.p90_tpot_ns),
        "pooled p90 TPOT out of range: {} ns",
        lvl.p90_tpot_ns
    );

    // Aggregate throughput ≈ 16 tokens over the level wall time — and the
    // wall time reflects the *slowest* of the four concurrent streams, not
    // four sequential runs (all four finished within ~4 frame delays).
    assert!(lvl.wall_ns < 2_000_000_000, "wall: {} ns", lvl.wall_ns);
    assert!((lvl.aggregate_tps - 16.0 / lvl.wall_secs()).abs() < 0.01);
}

// ── Acceptance: no connection leaks (fd count stable after shutdown) ──────

#[tokio::test]
async fn no_fd_leaks_after_sweep_shutdown() {
    let (url, server) = start_mock_sse(2, Duration::from_millis(5)).await;
    // Baseline *after* the mock's listener exists, *before* any sweep.
    let baseline = socket_fd_count();

    let sweep = Sweep::new(test_pool(&url), [2, 4]);
    let result = sweep.run().await;
    assert_eq!(result.levels.len(), 2);

    // Tear everything down: the sweep (owns the pooled client), then the
    // mock server.
    drop(sweep);
    server.abort();
    // Give the pool teardown a moment to close sockets.
    tokio::time::sleep(Duration::from_millis(300)).await;

    if let (Some(before), Some(after)) = (baseline, socket_fd_count()) {
        assert!(
            after <= before + 2,
            "fd leak: {before} sockets before sweep, {after} after shutdown"
        );
    }
}

// ── Default ladder (blueprint §5 Engine B) ────────────────────────────────

#[tokio::test]
async fn default_ladder_is_the_blueprint_sweep() {
    assert_eq!(DEFAULT_LADDER, [1, 2, 4, 8, 16, 32, 64]);
    let sweep = Sweep::new(test_pool("http://127.0.0.1:1"), DEFAULT_LADDER);
    assert_eq!(sweep.ladder(), &[1, 2, 4, 8, 16, 32, 64]);
}

// ── View 2: Concurrency Matrix rendering ──────────────────────────────────

#[test]
fn view2_shows_placeholder_matrix_before_a_sweep() {
    let mut app = App::new();
    app.view = View::Concurrency;
    let text = render_concurrency(&app);
    // The full default ladder is listed, unrun.
    for lvl in [1usize, 2, 4, 8, 16, 32, 64] {
        assert!(text.contains(&lvl.to_string()), "missing ladder row {lvl}");
    }
    assert!(text.contains("not run"));
    assert!(text.contains("CONCURRENCY SWEEP"));
    assert!(text.contains("OPTIMAL OPERATIONAL ENVELOPE"));
    assert!(text.contains("Run a sweep to detect the saturation knee"));
}

#[test]
fn view2_renders_sweep_curve_knee_and_envelope() {
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep = Some(std::sync::Arc::new(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 350.0, 8.0),
            level(4, 340.0, 20.0),
        ],
    }));
    let text = render_concurrency(&app);

    // Real curve rows: aggregate t/s + p90 TPOT per level.
    assert!(text.contains("100.0 t/s"));
    assert!(text.contains("350.0 t/s"));
    assert!(text.contains("340.0 t/s"));
    assert!(text.contains("5.0 ms"));
    assert!(text.contains("8.0 ms"));
    assert!(text.contains("20.0 ms"));
    // The knee row (c=4: throughput plateaus 350→340, p90 spikes 8→20 ms)
    // is flagged in the matrix.
    assert!(text.contains("KNEE"));
    // The sweet-spot row (c=2, the last healthy level) is flagged.
    assert!(text.contains("SWEET"));
    // The envelope panel reports the recommended sweet spot and the
    // detected knee.
    assert!(text.contains("Recommended sweet spot"));
    assert!(text.contains("2 streams"));
    assert!(text.contains("Saturation knee"));
    assert!(text.contains("4 streams"));
    // The placeholder copy is gone.
    assert!(!text.contains("not run"));
}

#[test]
fn view2_highlights_knee_on_the_full_ladder_curve() {
    // A full 7-level curve with a clear saturation knee at 16: throughput
    // plateaus (340→345, +1.5%) while p90 TPOT spikes 3× (10→30 ms).
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep = Some(std::sync::Arc::new(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 190.0, 6.0),
            level(4, 280.0, 7.0),
            level(8, 340.0, 10.0),
            level(16, 345.0, 30.0),
            level(32, 342.0, 60.0),
            level(64, 338.0, 90.0),
        ],
    }));
    let text = render_concurrency(&app);

    // The detected knee matches the injected inflection (acceptance:
    // "the detected knee matches the injected level").
    let knee = app.sweep.as_ref().unwrap().detect_knee().expect("knee");
    assert_eq!(knee.concurrency, 16);
    assert_eq!(knee.sweet_spot, 8);

    // The matrix flags the knee and sweet-spot rows.
    assert!(text.contains("KNEE"));
    assert!(text.contains("SWEET"));
    // The envelope panel reports the sweet spot (8) and the knee (16).
    assert!(text.contains("Recommended sweet spot"));
    assert!(text.contains("8 streams"));
    assert!(text.contains("Saturation knee"));
    assert!(text.contains("16 streams"));
}

#[test]
fn view2_survives_a_partial_failure_level() {
    let mut app = App::new();
    app.view = View::Concurrency;
    let mut lvl = level(8, 300.0, 12.0);
    lvl.completed_streams = 6;
    lvl.failed_streams = 2;
    lvl.streams = lvl
        .streams
        .into_iter()
        .enumerate()
        .map(|(i, mut m)| {
            if i >= 6 {
                m.state = StreamStatus::Error;
                m.gen_tps = None;
                m.tg_tokens = None;
            }
            m
        })
        .collect();
    app.sweep = Some(std::sync::Arc::new(SweepResult { levels: vec![lvl] }));
    let text = render_concurrency(&app);
    assert!(text.contains("6/8 ok"));
    assert!(text.contains("300.0 t/s"));
}

// ── Pool fan-in sanity (all workers complete, channel closes) ─────────────

#[tokio::test]
async fn pool_fan_in_delivers_every_worker_and_closes() {
    let (url, server) = start_mock_sse(2, Duration::from_millis(2)).await;
    let (mut rx, supervisor) = test_pool(&url).spawn(3);

    // Drain to closure: the channel must close (None) once all 3 workers
    // finish, and every worker must have delivered a terminal event.
    let mut terminals = 0usize;
    while let Some(event) = rx.recv().await {
        if event.is_terminal() {
            terminals += 1;
        }
    }
    let outcomes = supervisor.await.unwrap();
    server.abort();

    assert_eq!(terminals, 3, "one terminal event per worker");
    assert_eq!(outcomes.len(), 3);
    assert!(outcomes.iter().all(|o| o.is_ok()));
}
