//! Chunk 15 — Engine C1 (Needle-In-A-Haystack) integration tests against
//! an in-process mock SSE server (plan testing strategy: the suite runs
//! offline).
//!
//! Acceptance (plan Chunk 15): the NIAH runner executes a size×depth
//! matrix against the endpoint; retrieval is verified by checking the
//! needle value is present/extractable in the response; prefill speed is
//! recorded per cell; View 3 renders the color-coded grid.
//!
//! The mock server reads the request body, extracts the needle value the
//! engine injected into the prompt, and (for the `Retrieves` mode)
//! streams it back as the completion — so a *well-behaved* model is
//! simulated without any LLM in the loop.

use std::time::Duration;

use ratatui::buffer::Buffer;
use ratatui::style::Color;
use tokio::io::AsyncWriteExt;
use tokio::net::TcpListener;
use tokio::net::TcpStream;

use crucible_llm::config::Config;
use crucible_llm::engines::capability::{NiahCell, NiahCellState, NiahResult, NIAH_DEPTHS};
use crucible_llm::ui::app::{App, View};
use crucible_llm::ui::theme::Theme;
use crucible_llm::ui::views::needle;

mod common;
use common::mock::{prompt_from_body, read_request, write_all, write_chunk, SSE_HEADERS};
use common::render::render_buffer;

// ── Mock server ───────────────────────────────────────────────────────────

/// What the mock server does for each connection.
#[derive(Clone, Copy, PartialEq)]
enum Mock {
    /// Read the prompt, extract the needle value, echo it back.
    Retrieves,
    /// Answer with a fixed non-retrieval (the model hallucinates).
    Hallucinates,
    /// Like `Retrieves`, but sleep before the first byte: short for
    /// small contexts, super-linearly longer for large ones (simulates
    /// prefill degradation → the throttled classification).
    SlowPrefill,
}

async fn start_mock(mock: Mock) -> (String, tokio::task::JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        loop {
            let (mut sock, _) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => return,
            };
            let prompt = read_prompt(&mut sock).await;
            respond(&mut sock, mock, prompt).await;
        }
    });
    (format!("http://{addr}"), handle)
}

/// Read the full HTTP request (headers + body) and return
/// `messages[0].content` from the JSON body.
async fn read_prompt(sock: &mut TcpStream) -> String {
    let (_, body) = read_request(sock).await;
    prompt_from_body(&body).unwrap_or_default()
}

/// The needle value as the engine injected it into the prompt
/// ("IMPORTANT: the secret code is {value}. …").
fn extract_needle_value(prompt: &str) -> Option<String> {
    let marker = "the secret code is ";
    let start = prompt.find(marker)? + marker.len();
    let rest = &prompt[start..];
    let end = rest.find(['.', ' '])?;
    Some(rest[..end].to_string())
}

async fn respond(sock: &mut TcpStream, mock: Mock, prompt: String) {
    // Headers go out immediately; for SlowPrefill the *first body byte*
    // is what arrives late — TTFT is `T3 − T1` (T1 = headers received),
    // so only a post-header delay simulates prefill degradation.
    write_all(sock, SSE_HEADERS).await;
    if mock == Mock::SlowPrefill {
        let approx_tokens = (prompt.chars().count() / 4) as u64;
        let ms = if approx_tokens < 3000 { 40 } else { 600 };
        tokio::time::sleep(Duration::from_millis(ms)).await;
    }

    let answer = match mock {
        Mock::Retrieves | Mock::SlowPrefill => {
            // A well-behaved model answers with the value it found.
            extract_needle_value(&prompt).unwrap_or_else(|| "I don't know.".to_string())
        }
        Mock::Hallucinates => "I don't know.".to_string(),
    };

    let role = "data: {\"id\":\"cmpl-n\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n".to_string();
    let content = format!(
        "data: {{\"id\":\"cmpl-n\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
        serde_json::to_string(&answer).unwrap()
    );
    let usage = "data: {\"id\":\"cmpl-n\",\"choices\":[],\"usage\":{\"prompt_tokens\":4096,\"completion_tokens\":8}}\n\n".to_string();
    let done = "data: [DONE]\n\n".to_string();
    for frame in [role, content, usage, done] {
        write_chunk(sock, frame.as_bytes()).await;
    }
    write_chunk(sock, b"").await; // terminating chunk
    let _ = sock.shutdown().await;
}

// ── Test harness ──────────────────────────────────────────────────────────

fn test_config(url: &str) -> Config {
    Config {
        url: url.to_string(),
        model: "niah-test-model".to_string(),
        timeout: 15,
        ..Config::default()
    }
}

fn cell(size: u32, depth: u8, retrieved: bool, ttft: f64) -> NiahCell {
    NiahCell {
        target_tokens: size,
        depth_percent: depth,
        retrieved,
        ttft_s: ttft,
        prefill_tps: if ttft > 0.0 { size as f64 / ttft } else { 0.0 },
        state: NiahCellState::Nominal,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[tokio::test]
async fn matrix_run_retrieves_the_needle_at_every_depth() {
    let (url, _server) = start_mock(Mock::Retrieves).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000])
        .depths(NIAH_DEPTHS.to_vec());
    let result = engine.run().await;

    // One cell per depth: 11 cells, all retrieved.
    assert_eq!(result.cells.len(), NIAH_DEPTHS.len());
    assert_eq!(result.accuracy(), (NIAH_DEPTHS.len(), NIAH_DEPTHS.len()));
    for (i, depth) in NIAH_DEPTHS.iter().enumerate() {
        let c = &result.cells[i];
        assert!(c.retrieved, "depth {depth}% must retrieve");
        assert!(c.ttft_s > 0.0, "TTFT recorded per cell (depth {depth}%)");
        assert!(
            c.prefill_tps > 0.0,
            "prefill speed recorded per cell (depth {depth}%)"
        );
        // Every cell shares the (only) size = the baseline → nominal.
        assert_eq!(c.state, NiahCellState::Nominal, "depth {depth}% state");
    }
}

#[tokio::test]
async fn hallucinating_endpoint_marks_cells_failed() {
    let (url, _server) = start_mock(Mock::Hallucinates).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000])
        .depths(vec![0, 50]);
    let result = engine.run().await;

    assert_eq!(result.accuracy(), (0, 2));
    assert!(
        result
            .cells
            .iter()
            .all(|c| c.state == NiahCellState::Failed),
        "unretrieved cells must be classified failed (red)"
    );
}

#[tokio::test]
async fn superlinear_prefill_is_classified_throttled() {
    let (url, _server) = start_mock(Mock::SlowPrefill).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000, 4000])
        .depths(vec![0, 100]);
    let result = engine.run().await;

    assert_eq!(result.cells.len(), 4);
    assert_eq!(result.accuracy(), (4, 4), "the mock always retrieves");

    // Baseline row (smallest size): nominal.
    for c in &result.cells[0..2] {
        assert_eq!(c.target_tokens, 2000);
        assert_eq!(
            c.state,
            NiahCellState::Nominal,
            "baseline cell must be nominal"
        );
    }
    // 4k row: ~15× the linear expectation → throttled (yellow).
    for c in &result.cells[2..4] {
        assert_eq!(c.target_tokens, 4000);
        assert_eq!(
            c.state,
            NiahCellState::Throttled,
            "super-linear prefill must throttle"
        );
    }
}

// ── View 3 (blueprint §6) ────────────────────────────────────────────────

fn render_needle(app: &App, w: u16, h: u16) -> Buffer {
    render_buffer(needle::render, app, w, h)
}

fn buf_text(buf: &Buffer) -> String {
    buf.content().iter().map(|c| c.symbol()).collect()
}

/// The (symbol, foreground-color) pairs present in the buffer.
fn buf_colors(buf: &Buffer) -> Vec<(String, Color)> {
    buf.content()
        .iter()
        .filter(|c| c.symbol() != " ")
        .map(|c| (c.symbol().to_string(), c.fg))
        .collect()
}

#[test]
fn view3_renders_color_coded_grid_from_published_result() {
    let mut app = App::new();
    app.view = View::Needle;

    // A synthetic matrix: 2k@0 nominal, 4k@0 throttled, 4k@100 failed.
    let mut r = NiahResult::new(vec![2000, 4000], vec![0, 100]);
    r.cells[0] = cell(2000, 0, true, 0.1);
    r.cells[0].state = NiahCellState::Nominal;
    r.cells[1] = cell(2000, 100, true, 0.1);
    r.cells[1].state = NiahCellState::Nominal;
    r.cells[2] = cell(4000, 0, true, 0.5);
    r.cells[2].state = NiahCellState::Throttled;
    r.cells[3] = cell(4000, 100, false, 0.0);
    r.cells[3].state = NiahCellState::Failed;
    app.niah.store(r);

    let buf = render_needle(&app, 120, 40);
    let text = buf_text(&buf);

    // The grid title and the full size/depth axis labels.
    assert!(text.contains("NEEDLE-IN-A-HAYSTACK MATRIX"));
    for label in ["2k", "4k", "8k", "16k", "32k", "64k", "128k"] {
        assert!(text.contains(label), "missing size row {label}");
    }
    assert!(text.contains("100%"));

    // The three color classes are present in the grid (cyberpunk:
    // nominal = mint, throttled = electric purple, failed = hot pink).
    let colors = buf_colors(&buf);
    assert!(
        colors.iter().any(|(s, c)| s == "●" && *c == Theme::Cyberpunk.success()),
        "a nominal (mint) cell must render"
    );
    assert!(
        colors.iter().any(|(s, c)| s == "●" && *c == Theme::Cyberpunk.warn()),
        "a throttled (purple) cell must render"
    );
    assert!(
        colors.iter().any(|(s, c)| s == "✗" && *c == Theme::Cyberpunk.danger()),
        "a failed (hot-pink) cell must render"
    );

    // The legend reports the headline accuracy (cells 0, 1, 2 retrieved;
    // cell 3 failed).
    assert!(text.contains("Accuracy: 3/4 retrieved (75.0%)"));
    assert!(text.contains("LEGEND"));
}

#[test]
fn view3_shows_running_status_and_empty_grid_before_a_run() {
    let mut app = App::new();
    app.view = View::Needle;

    // Nothing run yet: every cell is the dimmed placeholder, and the
    // legend advertises the request count + confirmation.
    let buf = render_needle(&app, 120, 40);
    let text = buf_text(&buf);
    assert!(text.contains("···"));
    assert!(text.contains("[N] runs a new NIAH test"));
    assert!(text.contains("77 requests (7 sizes × 11 depths)"));

    // A run in progress: the status line switches.
    app.niah.set_running(true);
    let buf = render_needle(&app, 120, 40);
    assert!(buf_text(&buf).contains("Running NIAH matrix…"));
}
