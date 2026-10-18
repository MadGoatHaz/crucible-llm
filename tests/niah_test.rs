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
    /// A reasoning model: a chain-of-thought monologue (no needle) followed
    /// by the visible answer (the needle). Content-only matching (CF-01)
    /// must retrieve it — a model that thinks *then* answers is judged on
    /// its answer (closes CF-23: the suite never exercised a reasoning
    /// delta before).
    Reasons,
    /// A reasoning model that mentions the needle value *only* in its
    /// chain-of-thought and emits no visible answer. Content-only matching
    /// (CF-01 / CF-15) must NOT pass this cell — the old combined
    /// reasoning+content match would have false-passed it.
    ReasonsOnly,
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

    let role = "data: {\"id\":\"cmpl-n\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n".to_string();
    let usage = "data: {\"id\":\"cmpl-n\",\"choices\":[],\"usage\":{\"prompt_tokens\":4096,\"completion_tokens\":8}}\n\n".to_string();
    let done = "data: [DONE]\n\n".to_string();
    let content_frame = |text: &str| {
        format!(
            "data: {{\"id\":\"cmpl-n\",\"choices\":[{{\"index\":0,\"delta\":{{\"content\":{}}}}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    };
    let reasoning_frame = |text: &str| {
        format!(
            "data: {{\"id\":\"cmpl-n\",\"choices\":[{{\"index\":0,\"delta\":{{\"reasoning_content\":{}}}}}]}}\n\n",
            serde_json::to_string(text).unwrap()
        )
    };

    let mut frames: Vec<String> = vec![role.clone()];
    match mock {
        Mock::Retrieves | Mock::SlowPrefill => {
            // A well-behaved model answers with the value it found.
            let answer =
                extract_needle_value(&prompt).unwrap_or_else(|| "I don't know.".to_string());
            frames.push(content_frame(&answer));
        }
        Mock::Hallucinates => {
            frames.push(content_frame("I don't know."));
        }
        // CF-23: think (CoT, no needle) → then answer (content, the needle).
        Mock::Reasons => {
            let answer =
                extract_needle_value(&prompt).unwrap_or_else(|| "I don't know.".to_string());
            frames.push(reasoning_frame(
                "Let me think about where the secret code is…",
            ));
            frames.push(reasoning_frame(
                "Scanning the document… I believe I found it.",
            ));
            frames.push(content_frame(&answer));
        }
        // CF-15: the needle appears ONLY in the CoT; no visible answer.
        Mock::ReasonsOnly => {
            if let Some(value) = extract_needle_value(&prompt) {
                frames.push(reasoning_frame(&format!("The secret code is {value}.")));
                frames.push(reasoning_frame("…but I'm not sure, let me stop."));
            } else {
                frames.push(reasoning_frame("I don't know."));
            }
        }
    }
    frames.push(usage);
    frames.push(done);
    for frame in &frames {
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
        truncated_by_reasoning: false,
        reasoning_tokens: None,
        prompt_tokens: size as u64,
        completion_tokens: None,
        ttft_s: ttft,
        prefill_tps: if ttft > 0.0 { size as f64 / ttft } else { 0.0 },
        state: NiahCellState::Nominal,
        no_baseline: false,
        approximate: false,
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

#[tokio::test]
async fn cells_are_flagged_approximate_without_a_tokenizer_cf46() {
    // The mock config has no `--tokenizer`, so every document is sized by the
    // `chars/4` estimate; `run_cell` must propagate that to each cell (CF-46)
    // so the context size is labeled approximate, never presented as exact.
    let (url, _server) = start_mock(Mock::Retrieves).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000])
        .depths(vec![0, 50]);
    let result = engine.run().await;
    assert!(
        result.cells.iter().all(|c| c.approximate),
        "no-tokenizer cells must be flagged approximate (CF-46)"
    );
    // And the cell's context label surfaces the approximation.
    assert_eq!(result.cells[0].context_label(), "2k (approx.)");
}

// ── CF-01 / CF-15 / CF-23: reasoning-model matching ──────────────────────

#[tokio::test]
async fn reasoning_then_content_is_retrieved_content_only() {
    // CF-23 (closes the suite gap): the mock emits `reasoning` deltas (no
    // needle) THEN a `content` delta (the needle). A reasoning model that
    // thinks and then answers must be judged on its *answer* — content-only
    // matching (CF-01) retrieves the needle. Before CF-01 the suite never
    // exercised a reasoning delta at all.
    let (url, _server) = start_mock(Mock::Reasons).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000])
        .depths(vec![0, 100]);
    let result = engine.run().await;
    assert_eq!(
        result.accuracy(),
        (2, 2),
        "reasoning-then-content must retrieve (content-only verdict)"
    );
}

#[tokio::test]
async fn needle_only_in_cot_does_not_false_pass() {
    // CF-15 (the latent false-positive): the mock mentions the needle value
    // *only* in its chain-of-thought and emits no visible answer. Content-only
    // matching must NOT pass the cell — the old combined reasoning+content
    // substring match would have false-passed a model that never answered.
    let (url, _server) = start_mock(Mock::ReasonsOnly).await;
    let engine = crucible_llm::engines::NiahEngine::new(&test_config(&url))
        .unwrap()
        .sizes(vec![2000])
        .depths(vec![0, 100]);
    let result = engine.run().await;
    assert_eq!(
        result.accuracy(),
        (0, 2),
        "a needle mentioned only in the CoT is not a retrieval"
    );
    assert!(
        result
            .cells
            .iter()
            .all(|c| c.state == NiahCellState::Failed),
        "CoT-only needle cells must be failed (red)"
    );
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

    // The grid title. F3: the grid renders the *published result's* own
    // axes — this 2×2 custom-ladder result shows exactly its two size
    // rows; the default const ladder no longer pads the grid with unrun
    // 8k…128k rows (before F3 the matrix was 7×11 and every lookup on the
    // non-2k/4k rows missed → `···`).
    assert!(text.contains("NEEDLE-IN-A-HAYSTACK MATRIX"));
    assert!(text.contains("2k"), "missing the 2k size row");
    assert!(text.contains("4k"), "missing the 4k size row");
    assert!(text.contains("100%"));
    // Every cell of the published result ran → no unrun placeholders, and
    // no default-ladder size rows (16k/32k/64k appear nowhere else in the
    // view, so their absence proves the axes come from the result).
    assert_eq!(
        text.matches("···").count(),
        0,
        "a fully-run 2×2 grid has no unrun cells: {text}"
    );
    for label in ["16k", "32k", "64k"] {
        assert!(
            !text.contains(label),
            "default-ladder row {label} must not pad a custom-ladder result"
        );
    }

    // The three color classes are present in the grid (cyberpunk:
    // nominal = mint, throttled = electric purple, failed = hot pink).
    let colors = buf_colors(&buf);
    assert!(
        colors
            .iter()
            .any(|(s, c)| s == "●" && *c == Theme::Cyberpunk.success()),
        "a nominal (mint) cell must render"
    );
    assert!(
        colors
            .iter()
            .any(|(s, c)| s == "●" && *c == Theme::Cyberpunk.warn()),
        "a throttled (purple) cell must render"
    );
    assert!(
        colors
            .iter()
            .any(|(s, c)| s == "✗" && *c == Theme::Cyberpunk.danger()),
        "a failed (hot-pink) cell must render"
    );

    // The legend reports the headline accuracy (cells 0, 1, 2 retrieved;
    // cell 3 is a genuine fail — not truncated — so the Option C label
    // breaks the fail count out).
    assert!(text.contains("Accuracy: 3/4 retrieved, 1 failed (75.0%)"));
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
