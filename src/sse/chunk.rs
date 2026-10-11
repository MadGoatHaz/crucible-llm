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

/// The reasoning / cache **detail** fields of a server `usage` block, beyond
/// the core prompt/completion counts carried by [`Usage`].
///
/// These are the fields the research doc shows each backend emits for
/// reasoning-model accounting (URC-1 — the reasoning-model blind spot):
///
/// * `reasoning_tokens` — chain-of-thought tokens. vLLM and the OpenAI API
///   put it in `completion_tokens_details.reasoning_tokens`; SGLang in
///   `output_tokens_details.reasoning_tokens`. `0` when the backend does not
///   report it.
/// * `cached_tokens` — prompt tokens served from the KV / prefix cache
///   (`prompt_tokens_details.cached_tokens`). `0` when absent.
///
/// Kept separate from [`Usage`] (which stays a 2-field core so existing
/// construction sites are untouched): this is the *detection* half of
/// CF-02 — it lets a run be flagged reasoning-dominated and lets us verify
/// that thinking suppression worked (`reasoning_tokens ≈ 0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct UsageDetails {
    pub reasoning_tokens: u64,
    pub cached_tokens: u64,
}

impl UsageDetails {
    /// Extract the detail fields from a streaming chunk's `usage` object.
    ///
    /// Returns [`UsageDetails::default`] (all zeros) when the block is absent
    /// — graceful degradation for backends that do not report the details.
    #[must_use]
    pub fn from_value(value: &Value) -> UsageDetails {
        let Some(usage) = value.get("usage").and_then(|u| u.as_object()) else {
            return UsageDetails::default();
        };
        // `reasoning_tokens` lives under `completion_tokens_details`
        // (vLLM / OpenAI) or `output_tokens_details` (SGLang); read whichever
        // is present.
        let reasoning_tokens = usage
            .get("completion_tokens_details")
            .or_else(|| usage.get("output_tokens_details"))
            .and_then(|d| d.get("reasoning_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        // `cached_tokens` lives under `prompt_tokens_details`.
        let cached_tokens = usage
            .get("prompt_tokens_details")
            .and_then(|d| d.get("cached_tokens"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        UsageDetails {
            reasoning_tokens,
            cached_tokens,
        }
    }

    /// `true` when the completion budget was (nearly) all chain-of-thought:
    /// the reasoning token count equals the total completion count. This is
    /// the exact "thinking ate the whole budget" condition that false-fails
    /// the 64-token NIAH probe (Strawberry Issue 1).
    #[must_use]
    pub fn is_reasoning_dominated(&self, completion_tokens: u64) -> bool {
        completion_tokens > 0 && self.reasoning_tokens >= completion_tokens
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
