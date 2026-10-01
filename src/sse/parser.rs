//! Incremental, low-allocation SSE frame parser.
//!
//! The parser is a manual line-buffer state machine (the plan allows
//! `eventsource-stream` *or* a manual line buffer; the manual variant keeps
//! zero extra dependencies on the hot path and gives full control over
//! `[DONE]`, BOM stripping, and malformed-JSON tolerance).
//!
//! Per-frame contract handled by [`SseParser::feed`]:
//!
//! * `data:` prefix, with or without the single leading space the SSE spec
//!   allows;
//! * multiple `data:` lines inside one frame are joined with `\n`
//!   (spec-compliant multi-line payloads);
//! * a blank line terminates a frame;
//! * `event:` / `id:` / `retry:` / `:`-comment lines are ignored;
//! * a UTF-8 BOM is stripped once, at stream start;
//! * `[DONE]` marks end-of-stream (emitted as a frame with `done: true`);
//! * malformed JSON is counted and skipped — the parser never panics.
//!
//! Frames are returned through the parser's internal ring
//! ([`SseParser::feed`] / [`SseParser::finish`] yield `&[ParsedFrame]`), so a
//! steady stream causes no per-frame allocation. `t_nanos` is left `0` by the
//! parser: per the measurement-isolation invariant, the timing owner (the
//! Chunk 5 `StreamWorker`, via `quanta`) stamps arrival times after each
//! `feed` call, not inside the parse path.

use crate::sse::chunk::{classify, Chunk, Usage};
use serde_json::Value;

/// UTF-8 byte-order mark (stripped once at stream start).
const BOM: &[u8] = &[0xEF, 0xBB, 0xBF];

/// One parsed SSE frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedFrame {
    /// Classified payload of the frame.
    pub chunk: Chunk,
    /// Arrival timestamp in nanoseconds on the worker's monotonic clock.
    /// The parser leaves this `0`; the `StreamWorker` (Chunk 5) stamps it
    /// with the `quanta` clock after each `feed`.
    pub t_nanos: u64,
    /// `true` for the terminal `[DONE]` frame.
    pub done: bool,
}

/// Incremental SSE parser over raw bytes from an LLM streaming endpoint.
pub struct SseParser {
    /// Tail bytes from the last `feed` that did not form a complete line.
    line_buf: Vec<u8>,
    /// `data:` lines accumulated for the in-progress frame.
    data_lines: Vec<Vec<u8>>,
    /// Reusable output ring — no per-frame allocation on the hot path.
    out: Vec<ParsedFrame>,
    /// BOM stripped once at stream start.
    bom_seen: bool,
    /// Count of malformed-JSON frames skipped (tolerance counter).
    malformed: u64,
    /// Last `usage` object seen anywhere in the stream, captured even on
    /// frames that also carry a content delta (matches the Python prototype).
    usage: Option<Usage>,
}

impl Default for SseParser {
    fn default() -> Self {
        Self::new()
    }
}

impl SseParser {
    /// Create a parser in its initial (BOM-not-yet-stripped) state.
    pub fn new() -> Self {
        Self {
            line_buf: Vec::new(),
            data_lines: Vec::new(),
            out: Vec::new(),
            bom_seen: false,
            malformed: 0,
            usage: None,
        }
    }

    /// Consume a slice of raw stream bytes.
    ///
    /// Returns the frames completed by this feed (possibly empty). The caller
    /// must consume/forward the frames before the next `feed` or `finish`
    /// call. `t_nanos` on returned frames is `0` — stamp it with the
    /// monotonic clock in the worker.
    pub fn feed(&mut self, bytes: &[u8]) -> &[ParsedFrame] {
        self.out.clear();
        let mut rest = bytes;
        if !self.bom_seen {
            self.bom_seen = true;
            if let Some(stripped) = rest.strip_prefix(BOM) {
                rest = stripped;
            }
        }
        self.line_buf.extend_from_slice(rest);
        while let Some(pos) = self.line_buf.iter().position(|&b| b == b'\n') {
            // Drain the completed line (incl. newline) out of the buffer so
            // the immutable borrow ends before the mutable `consume_line`.
            let line: Vec<u8> = self.line_buf.drain(0..=pos).collect();
            self.consume_line(&line);
        }
        &self.out
    }

    /// Flush any pending frame at end-of-stream.
    ///
    /// Covers streams that close without a trailing blank line (the
    /// "premature close" case the Chunk 5 worker must tolerate).
    pub fn finish(&mut self) -> &[ParsedFrame] {
        self.out.clear();
        if !self.line_buf.is_empty() {
            let pending: Vec<u8> = std::mem::take(&mut self.line_buf);
            self.consume_line(&pending);
        }
        self.flush_frame();
        &self.out
    }

    /// Reset all state so the parser can be reused for the next stream.
    pub fn reset(&mut self) {
        self.line_buf.clear();
        self.data_lines.clear();
        self.out.clear();
        self.bom_seen = false;
        self.malformed = 0;
        self.usage = None;
    }

    /// Number of malformed-JSON frames skipped so far.
    #[must_use]
    pub fn malformed_frames(&self) -> u64 {
        self.malformed
    }

    /// Last `usage` object seen in the stream, if any.
    pub fn usage(&self) -> Option<Usage> {
        self.usage
    }

    /// Handle one complete line. Drained lines carry their trailing `\n`
    /// (removed here, plus an optional `\r` for CRLF streams); the pending
    /// tail handed in by `finish` has no newline and passes through as-is.
    fn consume_line(&mut self, line: &[u8]) {
        let line = line.strip_suffix(b"\n").unwrap_or(line);
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if line.is_empty() {
            self.flush_frame();
            return;
        }
        if let Some(payload) = line.strip_prefix(b"data:") {
            // SSE spec: strip at most one leading space after the colon.
            let payload = payload.strip_prefix(b" ").unwrap_or(payload);
            self.data_lines.push(payload.to_vec());
        }
        // `event:` / `id:` / `retry:` / `:`-comment / unknown lines: ignored.
    }

    /// Dispatch the accumulated `data:` lines as one frame.
    fn flush_frame(&mut self) {
        if self.data_lines.is_empty() {
            return; // keepalive / control-only frame: nothing to classify
        }
        let mut joined = String::new();
        for (i, line) in self.data_lines.iter().enumerate() {
            if i > 0 {
                joined.push('\n');
            }
            joined.push_str(&String::from_utf8_lossy(line));
        }
        self.data_lines.clear();

        let trimmed = joined.trim();
        if trimmed == "[DONE]" {
            self.out.push(ParsedFrame {
                chunk: Chunk::Control,
                t_nanos: 0,
                done: true,
            });
            return;
        }

        match serde_json::from_str::<Value>(trimmed) {
            Ok(value) => {
                if let Some(usage) = Usage::from_value(&value) {
                    self.usage = Some(usage);
                }
                self.out.push(ParsedFrame {
                    chunk: classify(&value),
                    t_nanos: 0,
                    done: false,
                });
            }
            Err(_) => {
                // Malformed frame: count + skip, never panic.
                self.malformed += 1;
            }
        }
    }
}
