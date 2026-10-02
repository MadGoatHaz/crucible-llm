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

use ratatui::buffer::Buffer;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

use crucible_llm::client::pool::WorkerPool;
use crucible_llm::engines::concurrency::{Sweep, SweepLevel, SweepResult, DEFAULT_LADDER};
use crucible_llm::metrics::state::{StreamMetric, StreamStatus};
use crucible_llm::ui::app::{App, View};
use crucible_llm::ui::theme::Theme;
use crucible_llm::ui::views::concurrency;

mod common;
use common::mock::*;
use common::render::{render_buffer, render_text};

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

/// Serve a ladder-level stream: role opener, `content_frames` token
/// frames (spaced by `frame_delay`), usage, `[DONE]`.
async fn serve_stream(sock: &mut TcpStream, content_frames: usize, frame_delay: Duration) {
    write_all(sock, SSE_HEADERS).await;
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

// ── Helpers ───────────────────────────────────────────────────────────────

fn test_pool(url: &str) -> WorkerPool {
    WorkerPool::new(reqwest::Client::new(), url, "test-model", "Hello there", 16)
        .read_timeout(Duration::from_secs(10))
}

/// Count TCP sockets to remote port `port` that are stuck in a *leak*
/// state (ESTABLISHED / CLOSING / CLOSE_WAIT), by parsing
/// `/proc/net/tcp{,6}` (Linux).
///
/// This isolates *this* test's mock connections from sibling tests (which
/// bind their own ports), so the leak check is stable under parallel test
/// execution. A clean close never leaves these states: the leaked-fd
/// signature is CLOSE_WAIT (the peer closed, we never did). Transient
/// TIME_WAIT / FIN_WAIT sockets from a normal shutdown are intentionally
/// not counted.
/// `None` when the platform does not expose `/proc/net/tcp`.
fn stuck_sockets_to_port(port: u16) -> Option<usize> {
    const LEAK_STATES: &[&str] = &["01", "07", "08"]; // ESTABLISHED, CLOSING, CLOSE_WAIT
    let mut count = 0;
    for file in ["/proc/net/tcp", "/proc/net/tcp6"] {
        let Ok(data) = std::fs::read_to_string(file) else {
            continue;
        };
        for line in data.lines().skip(1) {
            let mut fields = line.split_whitespace();
            // sl, local_address, rem_address, st, …
            fields.next()?;
            fields.next()?;
            let rem = fields.next()?;
            let state = fields.next()?;
            if !LEAK_STATES.contains(&state) {
                continue;
            }
            // `rem` is `HEXIP:HEXPORT`.
            let rport = rem.rsplit(':').next()?.parse::<u16>().ok()?;
            if rport == port {
                count += 1;
            }
        }
    }
    Some(count)
}

/// Sample `stuck_sockets_to_port` `samples` times at `interval` spacing and
/// return the minimum — a socket still tearing down must not masquerade as
/// a leak.
async fn min_stuck_sockets(port: u16, samples: usize, interval: Duration) -> Option<usize> {
    let mut min: Option<usize> = None;
    for _ in 0..samples {
        if let Some(c) = stuck_sockets_to_port(port) {
            min = Some(min.map(|m| m.min(c)).unwrap_or(c));
        }
        tokio::time::sleep(interval).await;
    }
    min
}

/// Render the Concurrency view at `w`x`h` and return the resulting buffer.
fn render_concurrency_at(app: &App, w: u16, h: u16) -> Buffer {
    render_buffer(concurrency::render, app, w, h)
}

/// Render the Concurrency view at 120x40 and return the flat buffer text.
fn render_concurrency(app: &App) -> String {
    render_text(concurrency::render, app, 120, 40)
}

fn level(concurrency: usize, tps: f64, p90_ms: f64) -> SweepLevel {
    SweepLevel {
        concurrency,
        context: 0,
        aggregate_tps: tps,
        per_stream_tps: tps / concurrency.max(1) as f64,
        p50_tpot_ns: (p90_ms * 0.8e6) as u64,
        p90_tpot_ns: (p90_ms * 1e6) as u64,
        p99_tpot_ns: (p90_ms * 2e6) as u64,
        ttft_p50_ns: 10_000_000,
        ttft_p90_ns: 20_000_000,
        total_tokens: 100,
        completed_streams: concurrency,
        failed_streams: 0,
        timed_out_streams: 0,
        aborted: false,
        wall_ns: 1_000_000_000,
        loop_excluded_streams: 0,
        loop_excluded_tokens: 0,
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
                looping: false,
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
    // Our mock's port — the leak check is scoped to it, so sibling tests
    // (each binding their own port) can never contaminate the count.
    let port: u16 = url
        .rsplit(':')
        .next()
        .and_then(|p| p.parse().ok())
        .expect("mock url must carry a port");

    let sweep = Sweep::new(test_pool(&url), [2, 4]);
    let result = sweep.run().await;
    assert_eq!(result.levels.len(), 2);

    // Tear everything down: the sweep (owns the pooled client), then the
    // mock server.
    drop(sweep);
    server.abort();
    // Give the pool teardown a moment to close sockets, then sample: a
    // leaked worker fd would linger in CLOSE_WAIT against our port.
    tokio::time::sleep(Duration::from_millis(300)).await;
    let after = min_stuck_sockets(port, 10, Duration::from_millis(100)).await;

    if let Some(after) = after {
        assert_eq!(
            after, 0,
            "socket leak: {after} connection(s) to the mock still open after shutdown"
        );
    }
}

// ── Default ladder (blueprint §5 Engine B) ────────────────────────────────

#[tokio::test]
async fn default_ladder_is_the_blueprint_sweep() {
    // FIX 3: the new default ladder — granular at the low end where home
    // users operate, capped at 32.
    assert_eq!(DEFAULT_LADDER, [1, 2, 3, 4, 8, 12, 16, 24, 32]);
    let sweep = Sweep::new(test_pool("http://127.0.0.1:1"), DEFAULT_LADDER);
    assert_eq!(sweep.ladder(), &[1, 2, 3, 4, 8, 12, 16, 24, 32]);
}

// ── View 2: Concurrency Matrix rendering ──────────────────────────────────

#[test]
fn view2_shows_placeholder_matrix_before_a_sweep() {
    let mut app = App::new();
    app.view = View::Concurrency;
    let text = render_concurrency(&app);
    // The full default ladder is listed, unrun.
    for lvl in [1usize, 2, 3, 4, 8, 12, 16, 24, 32] {
        assert!(text.contains(&lvl.to_string()), "missing ladder row {lvl}");
    }
    assert!(text.contains("not run"));
    assert!(text.contains("CONCURRENCY SWEEP"));
    // FIX 3: the bottom panel is the practical recommendation.
    assert!(text.contains("CONCURRENCY RECOMMENDATION"));
    assert!(text.contains("Run a sweep (Engine B)"));
}

#[test]
fn view2_renders_sweep_curve_knee_and_envelope() {
    let mut app = App::new();
    app.view = View::Concurrency;
    // Per-stream: 100 / 175 / 20 — the knee (c=4: throughput collapses
    // 350→80, p90 spikes 8→20 ms) is past the point where each user
    // still gets a comfortable rate.
    app.sweep.store(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 350.0, 8.0),
            level(4, 80.0, 20.0),
        ],
        matrix: None,
    });
    let text = render_concurrency(&app);

    // Real curve rows: aggregate t/s + p90 TPOT per level.
    assert!(text.contains("100.0 t/s"));
    assert!(text.contains("350.0 t/s"));
    assert!(text.contains("80.0 t/s"));
    assert!(text.contains("5.0 ms"));
    assert!(text.contains("8.0 ms"));
    assert!(text.contains("20.0 ms"));
    // The knee row (c=4) is flagged in the matrix.
    assert!(text.contains("KNEE"));
    // The sweet-spot row (c=2, the last healthy level) is flagged.
    assert!(text.contains("SWEET"));
    // The recommendation panel (FIX 3): per-stream = 100 / 175 / 20 →
    // practical sweet spot 2, max usable 4, no unusable boundary, knee
    // 4 as reference.
    assert!(text.contains("CONCURRENCY RECOMMENDATION"));
    assert!(text.contains("Practical Sweet Spot"));
    assert!(text.contains("2 concurrent users"));
    assert!(text.contains("Maximum Usable"));
    assert!(text.contains("4 concurrent users"));
    assert!(text.contains("Pure Throughput Knee"));
    // The placeholder copy is gone.
    assert!(!text.contains("not run"));
}

#[test]
fn view2_highlights_knee_on_the_full_ladder_curve() {
    // A full 7-level curve with a clear saturation knee at 16: throughput
    // plateaus (340→345, +1.5%) while p90 TPOT spikes 3× (10→30 ms).
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep.store(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 190.0, 6.0),
            level(4, 280.0, 7.0),
            level(8, 340.0, 10.0),
            level(16, 345.0, 30.0),
            level(32, 342.0, 60.0),
            level(64, 338.0, 90.0),
        ],
        matrix: None,
    });
    let text = render_concurrency(&app);

    // The detected knee matches the injected inflection (acceptance:
    // "the detected knee matches the injected level").
    let knee = app
        .sweep
        .load()
        .as_ref()
        .as_ref()
        .unwrap()
        .detect_knee()
        .expect("knee");
    assert_eq!(knee.concurrency, 16);
    assert_eq!(knee.sweet_spot, 8);

    // The matrix flags the knee and sweet-spot rows.
    assert!(text.contains("KNEE"));
    assert!(text.contains("SWEET"));
    // The recommendation panel (FIX 3): per-stream = 100 / 95 / 70 /
    // 42.5 / 21.6 / 10.7 / 5.3 → practical 8, max usable 16, unusable
    // from 32, knee (reference) 16.
    assert!(text.contains("Practical Sweet Spot"));
    assert!(text.contains("8 concurrent users"));
    assert!(text.contains("Maximum Usable"));
    assert!(text.contains("16 concurrent users"));
    assert!(text.contains("Unusable Beyond"));
    assert!(text.contains("32+ concurrent users"));
    assert!(text.contains("Pure Throughput Knee"));
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
    app.sweep.store(SweepResult {
        levels: vec![lvl],
        matrix: None,
    });
    let text = render_concurrency(&app);
    assert!(text.contains("6/8 ok"));
    assert!(text.contains("300.0 t/s"));
}

// ── Pool fan-in sanity (all workers complete, channel closes) ─────────────

// ── View 2: throughput vs concurrency curve (block plot) ────────────────

#[test]
fn view2_curve_shows_placeholder_before_a_sweep() {
    let mut app = App::new();
    app.view = View::Concurrency;
    let text = render_concurrency(&app);
    assert!(text.contains("Parallel Users"), "curve panel title");
    assert!(text.contains("Run a sweep"));
}

#[test]
fn view2_curve_marks_sweet_spot_and_knee() {
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep.store(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 350.0, 8.0),
            level(4, 340.0, 20.0),
        ],
        matrix: None,
    });
    let buf = render_concurrency_at(&app, 120, 40);
    let text: String = buf.content().iter().map(|c| c.symbol()).collect();

    assert!(text.contains("Parallel Users"), "curve panel title");
    // Sweet spot (2): a bright-cyan ◆ marker. Knee (4): a hot-pink ▲ marker.
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "◆" && c.fg == Theme::Cyberpunk.primary()),
        "sweet spot is a bright-cyan ◆"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "▲" && c.fg == Theme::Cyberpunk.accent()),
        "knee is a hot-pink ▲"
    );
    // The level below the sweet spot (1): a mint ◆ marker.
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "◆" && c.fg == Theme::Cyberpunk.success()),
        "levels below the sweet spot are mint ◆"
    );
    // Stems, the knee annotation, and the value labels on every point.
    assert!(text.contains('│'));
    assert!(text.contains("KNEE @ 4"));
    assert!(text.contains("350.0 t/s"), "value label: {text}");
    assert!(text.contains("concurrent users"), "axis title: {text}");
    // The actionable note under the plot (FIX 3: the recommendation is
    // per-stream usability, with the knee as reference).
    assert!(text.contains("Practical Sweet Spot"), "note: {text}");
    assert!(text.contains("Pure Throughput Knee"), "note: {text}");
    // x labels: every ladder step appears on the plot.
    for c in ["1", "2", "4"] {
        assert!(text.contains(c));
    }
}

#[test]
fn view2_curve_renders_the_full_ladder() {
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep.store(SweepResult {
        levels: vec![
            level(1, 100.0, 5.0),
            level(2, 190.0, 6.0),
            level(4, 280.0, 7.0),
            level(8, 340.0, 10.0),
            level(16, 345.0, 30.0),
            level(32, 342.0, 60.0),
            level(64, 338.0, 90.0),
        ],
        matrix: None,
    });
    let buf = render_concurrency_at(&app, 120, 40);
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "◆" && c.fg == Theme::Cyberpunk.primary()),
        "sweet spot (8) is a bright-cyan ◆"
    );
    assert!(
        buf.content()
            .iter()
            .any(|c| c.symbol() == "▲" && c.fg == Theme::Cyberpunk.accent()),
        "knee (16) is a hot-pink ▲"
    );
    assert!(
        buf.content().iter().filter(|c| c.symbol() == "◆").count() >= 5,
        "the five non-knee levels get ◆ markers"
    );
}

#[test]
fn view2_curve_survives_small_terminals() {
    let mut app = App::new();
    app.view = View::Concurrency;
    app.sweep.store(SweepResult {
        levels: vec![level(1, 100.0, 5.0), level(2, 350.0, 8.0)],
        matrix: None,
    });
    for (w, h) in [(40, 10), (20, 6), (80, 24)] {
        let _ = render_concurrency_at(&app, w, h);
    }
}

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

// ── The freeze fix: hung servers must not hang the sweep ─────────────────

/// A mock that accepts *many* concurrent connections, serves each one a
/// vLLM-style SSE header + `frames` token frames, and then **hangs**: it
/// holds the connection open and sends nothing more (no `[DONE]`, no
/// close) — the user's "froze at level 16" scenario.
///
/// The hang ends as soon as the *client* disconnects (an aborted /
/// timed-out worker drops its socket → the server read sees EOF), so no
/// server-side fd outlives the test (fd-leak test isolation).
async fn start_hang_mock(frames: usize) -> (String, tokio::task::JoinHandle<()>) {
    use tokio::io::AsyncReadExt;
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
                write_all(
                    &mut sock,
                    b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n",
                )
                .await;
                for i in 0..frames {
                    write_chunk(
                        &mut sock,
                        format!("{content}\n\n", content = content_frame(i)).as_bytes(),
                    )
                    .await;
                }
                // The hang: send nothing, but release the socket as soon
                // as the client goes away (or after 300 s at the latest).
                let mut buf = [0u8; 64];
                tokio::select! {
                    _ = tokio::time::sleep(Duration::from_secs(300)) => {}
                    _ = async {
                        loop {
                            match sock.read(&mut buf).await {
                                Ok(0) | Err(_) => return,
                                Ok(_) => continue,
                            }
                        }
                    } => {}
                }
                let _ = sock.shutdown().await;
            });
        }
    });
    (format!("http://{addr}"), handle)
}

#[tokio::test]
async fn hung_workers_are_killed_by_the_per_worker_timeout() {
    // 2 workers against a server that goes silent after 3 frames. With a
    // 2 s per-worker cap (and a 30 s stall window that would never fire
    // first), both workers must be killed, the level must collect their
    // partial tokens, and the sweep must complete — never hang.
    let (url, server) = start_hang_mock(3).await;
    let pool = WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 64)
        .read_timeout(Duration::from_secs(30))
        .worker_timeout(Duration::from_secs(2));
    let sweep = Sweep::new(pool, [2]);
    let result = tokio::time::timeout(Duration::from_secs(20), sweep.run())
        .await
        .expect("the sweep must not hang on hung workers");
    server.abort();

    assert_eq!(result.levels.len(), 1);
    let lvl = &result.levels[0];
    assert!(lvl.degraded(), "a level with killed workers is degraded");
    assert_eq!(lvl.timed_out_streams, 2, "both workers hit the cap");
    assert_eq!(lvl.failed_streams, 2);
    assert_eq!(lvl.completed_streams, 0);
    // Partial results were collected before the kill (3 frames each).
    assert_eq!(lvl.total_tokens, 6);
}

#[tokio::test]
async fn hung_step_is_aborted_by_the_step_budget_and_sweep_continues() {
    // A 60 s per-worker cap (never reached), but a 1.5 s step-budget
    // override: the watchdog (5 s ticks) sees the level past 2× budget
    // and aborts it, and the sweep must proceed to the next level — the
    // whole run is bounded (no hang, ever).
    let (url, server) = start_hang_mock(2).await;
    let pool = WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 64)
        .read_timeout(Duration::from_secs(60))
        .worker_timeout(Duration::from_secs(60));
    let sweep = Sweep::new(pool, [2, 1]).step_budget(Duration::from_millis(1500));
    let result = tokio::time::timeout(Duration::from_secs(30), sweep.run())
        .await
        .expect("the sweep must not hang on a hung step");
    server.abort();

    assert_eq!(
        result.levels.len(),
        2,
        "the sweep continued past the bad step"
    );
    // The ladder normalizes to [1, 2]: level 0 has one worker, level 1 two.
    let bad = &result.levels[0];
    assert!(bad.aborted, "the over-budget step was aborted");
    assert!(bad.degraded());
    // The stranded workers were finalized as failures.
    assert_eq!(bad.failed_streams, 1);
    // The second step ran too (and was aborted the same way — the mock
    // hangs every connection).
    assert!(result.levels[1].aborted);
    assert_eq!(result.levels[1].failed_streams, 2);
}

#[tokio::test]
async fn sweep_with_logger_writes_step_lifecycle_to_the_run_log() {
    // The run log the user reviews after a hang: step starts, the
    // worker-kill warnings, and the per-step summary must all land in
    // `latest.log`.
    let dir = std::env::temp_dir().join(format!("crucible-runlog-sweep-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let logger = crucible_llm::log::RunLogger::start(Some(&dir));

    let (url, server) = start_hang_mock(3).await;
    let pool = WorkerPool::new(reqwest::Client::new(), &url, "m", "p", 64)
        .read_timeout(Duration::from_secs(30))
        .worker_timeout(Duration::from_secs(2))
        .logger(logger.clone());
    let sweep = Sweep::new(pool, [2]).logger(logger.clone());
    let result = tokio::time::timeout(Duration::from_secs(20), sweep.run())
        .await
        .expect("bounded");
    server.abort();
    logger.finish();

    let latest = std::fs::read_to_string(dir.join("latest.log")).unwrap();
    assert!(
        latest.contains("Engine B (Concurrency) started — ladder [2]"),
        "{latest}"
    );
    assert!(
        latest.contains("Step 1/1: concurrency=2, spawning 2 worker(s)"),
        "{latest}"
    );
    // Per-worker HTTP lifecycle from the tagged workers (B1-0, B1-1).
    assert!(
        latest.contains("B1-0"),
        "worker tags land in the log: {latest}"
    );
    assert!(latest.contains("B1-1"), "{latest}");
    assert!(latest.contains("→ POST"), "{latest}");
    assert!(
        latest.contains("TIMEOUT"),
        "the worker kills are logged: {latest}"
    );
    assert!(
        latest.contains("[DEGRADED]"),
        "the step summary flags the degradation"
    );
    assert_eq!(result.levels[0].timed_out_streams, 2);
    let _ = std::fs::remove_dir_all(&dir);
}
