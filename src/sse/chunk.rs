//! `Chunk` model for parsed SSE frames.
//!
//! Every `data:` frame from an OpenAI-compatible streaming endpoint is
//! classified into one of four categories (blueprint §4.2 — the Engine Core
//! consumes these to split reasoning-phase vs. writing-phase telemetry):
//!
//! * [`Chunk::Reasoning`] — chain-of-thought / inner-monologue delta
//!   (`choices[0].delta.reasoning` or `choices[0].delta.reasoning_content`).
//! * [`Chunk::Content`] — visible output delta (`choices[0].delta.content`).
//! * [`Chunk::Control`] — bookkeeping frames (role-only deltas, empty
//!   keepalives, `[DONE]`).
//! * [`Chunk::Usage`] — server-reported token usage (typically the final
//!   chunk, as sent by vLLM/llama.cpp with `include_usage: true`).
//!
//! [`classify`] priority: reasoning first, then content, then usage, else
//! control. This mirrors the Python prototype (`llmspeedtest2.py`), which
//! checks `reasoning`/`reasoning_content` before `content` so CoT tokens
//! never skew writing-phase throughput.

use serde_json::Value;

/// Server-reported token usage, typically carried by the final chunk.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Usage {
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
}

impl Usage {
    /// Extract the top-level `usage` object from a streaming chunk, if present.
    pub fn from_value(value: &Value) -> Option<Usage> {
        let usage = value.get("usage")?.as_object()?;
        let field = |key: &str| usage.get(key).and_then(|v| v.as_u64()).unwrap_or(0);
        Some(Usage {
            prompt_tokens: field("prompt_tokens"),
            completion_tokens: field("completion_tokens"),
        })
    }
}

/// Category of one parsed SSE data frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Chunk {
    /// Reasoning / chain-of-thought delta text.
    Reasoning(String),
    /// Visible content delta text.
    Content(String),
    /// Bookkeeping frame with no token payload (role, keepalive, `[DONE]`).
    Control,
    /// Token usage report.
    Usage(Usage),
}

impl Chunk {
    /// `true` when this chunk carries generated tokens (a
    /// [`Reasoning`](Chunk::Reasoning) or
    /// [`Content`](Chunk::Content) delta). A [`Usage`](Chunk::Usage) /
    /// [`Control`](Chunk::Control) frame is *not* a token arrival — the
    /// workers and engines use this to latch the first-token timestamp
    /// (T3) and to count token frames.
    #[must_use]
    pub fn is_token(&self) -> bool {
        matches!(self, Self::Reasoning(_) | Self::Content(_))
    }

    /// The decoded token text of a token-carrying chunk (`Reasoning` /
    /// `Content`), for the decode-loop guard's rolling token buffer.
    /// `None` for `Usage` / `Control` frames.
    #[must_use]
    pub fn token_text(&self) -> Option<&str> {
        match self {
            Self::Reasoning(s) | Self::Content(s) => Some(s.as_str()),
            Self::Usage(_) | Self::Control => None,
        }
    }
}

/// Classify one parsed streaming chunk.
///
/// Reads the first choice's `delta` first: `reasoning`/`reasoning_content`
/// wins, then `content`; a top-level `usage` object on a chunk with no token
/// delta yields [`Chunk::Usage`]; anything else is [`Chunk::Control`].
#[must_use]
pub fn classify(value: &Value) -> Chunk {
    if let Some(text) = delta_string(value, &["reasoning", "reasoning_content"]) {
        return Chunk::Reasoning(text);
    }
    if let Some(text) = delta_string(value, &["content"]) {
        return Chunk::Content(text);
    }
    if let Some(usage) = Usage::from_value(value) {
        return Chunk::Usage(usage);
    }
    Chunk::Control
}

/// First non-empty string among `keys` in `choices[0].delta`, if any.
fn delta_string(value: &Value, keys: &[&str]) -> Option<String> {
    let delta = value.get("choices")?.get(0)?.get("delta")?.as_object()?;
    keys.iter().find_map(|key| {
        delta
            .get(*key)?
            .as_str()
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
    })
}
