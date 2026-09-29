//! Integration tests for the Chunk 3 SSE parser (`src/sse/`).
//!
//! Acceptance (plan Chunk 3): a captured/representative vLLM-style SSE byte
//! stream containing `reasoning_content` deltas, `content` deltas, a final
//! `usage` chunk, and `[DONE]` yields exact counts of each category in the
//! correct order; a malformed frame injected mid-stream is skipped without
//! panic.

use crucible_llm::sse::{classify, Chunk, SseParser, Usage};
use serde_json::json;

/// A representative vLLM-style streaming response:
/// role-only opener, three `reasoning_content` deltas, two `content`
/// deltas, an empty-delta keepalive, a final `usage` chunk, then `[DONE]`.
const VLLM_STREAM: &str = concat!(
    "data: {\"id\":\"cmpl-1\",\"object\":\"text_completion\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Let me think.\"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Step by step.\"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"Now I answer.\"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"The answer \"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"is 42.\"}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[{\"index\":0,\"delta\":{}}]}\n",
    "\n",
    "data: {\"id\":\"cmpl-1\",\"choices\":[],\"usage\":{\"prompt_tokens\":128,\"completion_tokens\":34}}\n",
    "\n",
    "data: [DONE]\n",
    "\n",
);

fn count(frames: &[crucible_llm::sse::ParsedFrame], pred: impl Fn(&Chunk) -> bool) -> usize {
    frames.iter().filter(|f| pred(&f.chunk)).count()
}

#[test]
fn full_stream_exact_counts_and_order() {
    let mut parser = SseParser::new();
    let frames = parser.feed(VLLM_STREAM.as_bytes()).to_vec();

    assert_eq!(frames.len(), 9, "one frame per data block");
    assert_eq!(
        count(&frames, |c| matches!(c, Chunk::Reasoning(_))),
        3,
        "three reasoning deltas"
    );
    assert_eq!(
        count(&frames, |c| matches!(c, Chunk::Content(_))),
        2,
        "two content deltas"
    );
    assert_eq!(
        count(&frames, |c| matches!(c, Chunk::Usage(_))),
        1,
        "one usage chunk"
    );
    assert_eq!(
        count(&frames, |c| matches!(c, Chunk::Control)),
        3,
        "role-only opener + empty-delta keepalive + [DONE]"
    );

    // Ordering: all reasoning frames precede all content frames.
    let last_reasoning = frames
        .iter()
        .position(|f| matches!(f.chunk, Chunk::Reasoning(_)))
        .and_then(|i| {
            (0..=i)
                .rev()
                .find(|&j| matches!(frames[j].chunk, Chunk::Reasoning(_)))
        })
        .unwrap();
    let first_content = frames
        .iter()
        .position(|f| matches!(f.chunk, Chunk::Content(_)))
        .unwrap();
    assert!(
        last_reasoning < first_content,
        "reasoning phase before writing phase"
    );

    // Usage payload exact.
    assert_eq!(
        frames[7].chunk,
        Chunk::Usage(Usage {
            prompt_tokens: 128,
            completion_tokens: 34
        })
    );

    // [DONE] terminator: last frame, flagged done.
    assert!(frames.last().unwrap().done);
    assert!(frames[..frames.len() - 1].iter().all(|f| !f.done));

    // Parser captured usage for the worker to read at stream end.
    assert_eq!(
        parser.usage(),
        Some(Usage {
            prompt_tokens: 128,
            completion_tokens: 34
        })
    );
    assert_eq!(parser.malformed_frames(), 0);
}

#[test]
fn chunked_feeds_match_whole_feed() {
    // Feed the same stream in awkward 7-byte slices: lines are split
    // mid-prefix, mid-JSON, and mid-blank-line. Result must be identical.
    let mut whole = SseParser::new();
    let whole_frames = whole.feed(VLLM_STREAM.as_bytes()).to_vec();

    let mut sliced = SseParser::new();
    let bytes = VLLM_STREAM.as_bytes();
    let mut sliced_frames = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        let end = (i + 7).min(bytes.len());
        sliced_frames.extend_from_slice(sliced.feed(&bytes[i..end]));
        i = end;
    }
    sliced_frames.extend_from_slice(sliced.finish());

    assert_eq!(sliced_frames.len(), whole_frames.len());
    for (a, b) in sliced_frames.iter().zip(whole_frames.iter()) {
        assert_eq!(a.chunk, b.chunk, "sliced feed must classify identically");
        assert_eq!(a.done, b.done);
    }
    assert_eq!(sliced.usage(), whole.usage());
}

#[test]
fn malformed_frame_skipped_without_panic() {
    // A broken JSON frame injected mid-stream (e.g. a truncated transfer).
    let stream = concat!(
        "data: {\"choices\":[{\"delta\":{\"reasoning_content\":\"thinking\"}}]}\n",
        "\n",
        "data: {\"choices\":[{\"delta\":{\"cont\n", // <- malformed (truncated)
        "\n",
        "data: {\"choices\":[{\"delta\":{\"content\":\"hello\"}}]}\n",
        "\n",
        "data: [DONE]\n",
        "\n",
    );
    let mut parser = SseParser::new();
    let frames = parser.feed(stream.as_bytes()).to_vec();

    assert_eq!(
        parser.malformed_frames(),
        1,
        "exactly one malformed frame counted"
    );
    assert_eq!(
        frames.len(),
        3,
        "malformed frame dropped, neighbors survive"
    );
    assert_eq!(frames[0].chunk, Chunk::Reasoning("thinking".to_string()));
    assert_eq!(frames[1].chunk, Chunk::Content("hello".to_string()));
    assert!(frames[2].done);
}

#[test]
fn multi_line_data_frames_are_joined() {
    // One JSON payload split across two `data:` lines (spec: join with \n).
    // The split point sits outside any string literal, so the joined
    // payload is valid JSON (a raw newline inside a string is not).
    let stream = concat!(
        "data: {\"choices\":[{\"delta\":\n",
        "data: {\"content\":\"split across lines\"}}]}\n",
        "\n",
    );
    let mut parser = SseParser::new();
    let frames = parser.feed(stream.as_bytes()).to_vec();

    assert_eq!(frames.len(), 1);
    assert_eq!(
        frames[0].chunk,
        Chunk::Content("split across lines".to_string())
    );
}

#[test]
fn bom_crlf_no_space_and_ignored_fields() {
    // BOM at stream start, CRLF line endings, `data:` without the optional
    // space, and `event:`/`id:`/`retry:` lines that must be ignored.
    let stream = concat!(
        "\u{feff}event: message\r\n",
        "id: 42\r\n",
        "retry: 3000\r\n",
        "data:{\"choices\":[{\"delta\":{\"content\":\"no space\"}}]}\r\n",
        "\r\n",
    );
    let mut parser = SseParser::new();
    let frames = parser.feed(stream.as_bytes()).to_vec();

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].chunk, Chunk::Content("no space".to_string()));
}

#[test]
fn finish_flushes_trailing_frame_without_blank_line() {
    // Stream closes after a `data:` line with no terminating blank line
    // (premature-close tolerance).
    let stream = "data: {\"choices\":[{\"delta\":{\"content\":\"tail\"}}]}";
    let mut parser = SseParser::new();
    assert_eq!(
        parser.feed(stream.as_bytes()).len(),
        0,
        "no frame before stream end"
    );
    let frames = parser.finish().to_vec();

    assert_eq!(frames.len(), 1);
    assert_eq!(frames[0].chunk, Chunk::Content("tail".to_string()));
}

#[test]
fn done_without_space_is_recognized() {
    let mut parser = SseParser::new();
    let frames = parser.feed(b"data:[DONE]\n\n").to_vec();
    assert_eq!(frames.len(), 1);
    assert!(frames[0].done);
}

#[test]
fn usage_captured_even_on_content_frame() {
    // vLLM can attach `usage` to a chunk that also carries a delta; the
    // frame classifies as Content (priority) but the parser must still
    // capture the usage for end-of-stream readers.
    let stream = concat!(
        "data: {\"choices\":[{\"delta\":{\"content\":\"last\"}}],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":5}}\n",
        "\n",
    );
    let mut parser = SseParser::new();
    let frames = parser.feed(stream.as_bytes()).to_vec();

    assert_eq!(frames[0].chunk, Chunk::Content("last".to_string()));
    assert_eq!(
        parser.usage(),
        Some(Usage {
            prompt_tokens: 10,
            completion_tokens: 5
        })
    );
}

#[test]
fn classify_priority_reasoning_over_content_over_usage() {
    // Reasoning beats content when both are present.
    let both = json!({
        "choices": [{ "delta": {
            "reasoning_content": "inner monologue",
            "content": "visible text"
        } }]
    });
    assert_eq!(
        classify(&both),
        Chunk::Reasoning("inner monologue".to_string())
    );

    // `reasoning` (not just `reasoning_content`) is honored.
    let alt_key = json!({
        "choices": [{ "delta": { "reasoning": "alt key" } }]
    });
    assert_eq!(classify(&alt_key), Chunk::Reasoning("alt key".to_string()));

    // Content when no reasoning.
    let content = json!({
        "choices": [{ "delta": { "content": "hello" } }]
    });
    assert_eq!(classify(&content), Chunk::Content("hello".to_string()));

    // Usage when neither delta key is present.
    let usage = json!({
        "choices": [],
        "usage": { "prompt_tokens": 7, "completion_tokens": 9 }
    });
    assert_eq!(
        classify(&usage),
        Chunk::Usage(Usage {
            prompt_tokens: 7,
            completion_tokens: 9
        })
    );

    // Role-only / empty delta is control.
    let role_only = json!({
        "choices": [{ "delta": { "role": "assistant" } }]
    });
    assert_eq!(classify(&role_only), Chunk::Control);
}
