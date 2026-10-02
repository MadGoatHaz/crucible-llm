//! Prompt `generator`: `short` (~50 tok), `long` (pad to `target_tokens`),
//! and `nocache` (unique random prefix to bust the KV cache). Port of both
//! Python prototypes (`llmspeedtest.py` / `llmspeedtest2.py`).
//!
//! - **short** — the fixed ~50-token prompt (both prototypes' short prompts
//!   combined, `llmspeedtest2.py`'s Rust-macro prompt + `llmspeedtest.py`'s
//!   B-tree prompt).
//! - **long** — with a tokenizer: pad to `target_tokens` using the
//!   token-stable `" apple"` filler from `llmspeedtest2.py` (`" apple"` is
//!   almost universally exactly 1 BPE token); without one: pad to
//!   `target_tokens * 4` chars using the sentence-rotation filler from
//!   `llmspeedtest.py`, which the `chars/4` heuristic reads as
//!   `target_tokens`.
//! - **nocache** — prepend a unique 32-char hex (uuid v4) prefix, matching
//!   `llmspeedtest.py`'s `[{uuid4().hex}]` scheme, so the server's KV cache
//!   cannot be reused.

use std::sync::Arc;

use uuid::Uuid;

use crate::prompt::tokenizer::{count_tokens, Tokenizer};

/// The fixed short prompt (~50 tokens).
pub const SHORT_PROMPT: &str = "Write a complex Rust macro to measure function execution time. \
Then write a brief explanation of how a B-tree database index works, including when it's \
preferred over a hash index.";

/// The `llmspeedtest2.py` long-prompt base (the `" apple"` padding anchor).
pub const LONG_BASE: &str =
    "Repeat the following sequence back to me briefly, then explain the importance of context \
window size: ";

/// Token-stable filler: `" apple"` is (almost) universally exactly 1 BPE token.
pub const FILLER: &str = " apple";

/// The `llmspeedtest.py` sentence-rotation filler (chars/4 fallback path).
pub const BASE_SENTENCES: [&str; 10] = [
    "The quick brown fox jumps over the lazy dog.",
    "Python is a high-level interpreted programming language.",
    "Machine learning models require careful data preprocessing.",
    "Distributed systems must handle network partitions gracefully.",
    "The compiler optimizes intermediate representation for speed.",
    "Async I/O allows concurrent non-blocking operations.",
    "Vector databases store embeddings for similarity search.",
    "Kubernetes orchestrates containerized microservices at scale.",
    "The kernel manages memory, processes, and device drivers.",
    "Type systems prevent entire classes of runtime errors.",
];

/// A generated prompt plus its (possibly estimated) token count.
#[derive(Debug, Clone)]
pub struct GeneratedPrompt {
    pub text: String,
    pub token_count: u32,
    /// `true` when `token_count` came from the `chars/4` heuristic.
    pub estimated: bool,
    /// `true` when a unique KV-cache-busting prefix was prepended.
    pub nocache: bool,
}

/// Generates benchmark prompts. Holds an optional `Tokenizer` (shared via
/// `Arc` so it can be cloned cheaply into workers).
#[derive(Debug, Clone, Default)]
pub struct PromptGenerator {
    tokenizer: Option<Arc<Tokenizer>>,
}

impl PromptGenerator {
    /// Create a generator. Pass `None` to use the `chars/4` fallback for all
    /// token counts (results flagged `estimated`).
    pub fn new(tokenizer: Option<Tokenizer>) -> Self {
        Self {
            tokenizer: tokenizer.map(Arc::new),
        }
    }

    /// Access to the underlying tokenizer (if one was loaded).
    #[must_use]
    pub fn tokenizer(&self) -> Option<Arc<Tokenizer>> {
        self.tokenizer.clone()
    }

    /// The fixed ~50-token short prompt.
    pub fn short(&self) -> GeneratedPrompt {
        let c = count_tokens(SHORT_PROMPT, self.tokenizer.as_deref());
        GeneratedPrompt {
            text: SHORT_PROMPT.to_string(),
            token_count: c.tokens,
            estimated: c.estimated,
            nocache: false,
        }
    }

    /// A prompt padded to `target_tokens` (see module docs for the two
    /// padding strategies). Targets below the base size return the base as-is.
    pub fn long(&self, target_tokens: u32) -> GeneratedPrompt {
        let target = target_tokens.max(1);
        match self.tokenizer.as_deref() {
            Some(tok) => self.long_tokenized(tok, target),
            None => self.long_fallback(target),
        }
    }

    /// `short` with a unique KV-cache-busting prefix.
    pub fn nocache_short(&self) -> GeneratedPrompt {
        self.apply_nocache(self.short())
    }

    /// `long` with a unique KV-cache-busting prefix.
    pub fn nocache_long(&self, target_tokens: u32) -> GeneratedPrompt {
        self.apply_nocache(self.long(target_tokens))
    }

    /// Prepend a fresh 32-char hex uuid prefix and re-count tokens.
    fn apply_nocache(&self, mut prompt: GeneratedPrompt) -> GeneratedPrompt {
        prompt.text = format!("[{}] {}", Uuid::new_v4().simple(), prompt.text);
        let c = count_tokens(&prompt.text, self.tokenizer.as_deref());
        prompt.token_count = c.tokens;
        prompt.estimated = c.estimated;
        prompt.nocache = true;
        prompt
    }

    /// Tokenized long prompt: base + `" apple"` repeats, corrected so the
    /// final count lands on (or within one filler unit of) `target`.
    fn long_tokenized(&self, tok: &Tokenizer, target: u32) -> GeneratedPrompt {
        // Target below the base size: the base is the smallest unit.
        let base_n = match tok.try_count(LONG_BASE) {
            Some(n) if n >= target => {
                return GeneratedPrompt {
                    text: LONG_BASE.to_string(),
                    token_count: n,
                    estimated: false,
                    nocache: false,
                };
            }
            Some(n) => n,
            // The tokenizer cannot encode the base — degrade to the chars/4
            // sentence-rotation strategy (graceful, never panic).
            None => return self.long_fallback(target),
        };

        let unit = tok.try_count(FILLER).map(|u| u.max(1)).unwrap_or(1);
        let pad = target - base_n;
        let mut text = String::with_capacity(LONG_BASE.len() + pad as usize * FILLER.len());
        text.push_str(LONG_BASE);
        text.push_str(&FILLER.repeat(pad as usize));

        // Correct for fillers that are not exactly 1 token (the common case
        // is 1; this keeps the guarantee for exotic vocabularies).
        match tok.try_count(&text) {
            Some(n) if n < target => text.push_str(&FILLER.repeat((target - n) as usize)),
            Some(n) if n > target => {
                let excess_units = (n - target) / unit;
                text.truncate(text.len() - excess_units as usize * FILLER.len());
            }
            _ => {}
        }
        let c = count_tokens(&text, Some(tok));
        GeneratedPrompt {
            text,
            token_count: c.tokens,
            estimated: c.estimated,
            nocache: false,
        }
    }

    /// The `chars/4` sentence-rotation strategy (`llmspeedtest.py`).
    fn long_fallback(&self, target: u32) -> GeneratedPrompt {
        let text = long_sentence_rotation(target);
        let c = count_tokens(&text, None);
        GeneratedPrompt {
            text,
            token_count: c.tokens,
            estimated: true,
            nocache: false,
        }
    }
}

/// The `llmspeedtest.py` long-prompt strategy: rotate the sentence bank until
/// `target_tokens * 4` chars are reached (1 token ≈ 4 chars for English).
fn long_sentence_rotation(target_tokens: u32) -> String {
    let target_chars = (target_tokens * 4) as usize;
    let mut parts: Vec<&str> = Vec::new();
    let mut total: usize = 0;
    let mut i = 0;
    while total < target_chars {
        let sentence = BASE_SENTENCES[i % BASE_SENTENCES.len()];
        parts.push(sentence);
        total += sentence.len() + 1;
        i += 1;
    }
    parts.join(" ")
}
