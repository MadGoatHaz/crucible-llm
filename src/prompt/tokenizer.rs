//! Client-side `tokenizer`: optional HF `tokenizer.json` via the `tokenizers`
//! crate; falls back to the `chars/4` heuristic (marked `estimated`) when no
//! file is supplied.
//!
//! The blueprint makes client-side tokenization a first-class feature: it
//! eliminates reliance on server `usage` chunks and guarantees deterministic
//! input token counts (e.g. for pre-computing needle positions). Loading is
//! optional at runtime (`--tokenizer <path>`); when absent, every count is
//! flagged `estimated` so downstream metrics can be labelled as such.

use std::path::{Path, PathBuf};

use thiserror::Error;
use tokenizers::Tokenizer as HfTokenizer;

/// Errors from loading/parsing a `tokenizer.json` file.
#[derive(Debug, Error)]
pub enum TokenizerError {
    #[error("failed to read tokenizer file `{path}`: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to parse tokenizer JSON: {0}")]
    Parse(String),
}

/// A loaded Hugging Face `tokenizer.json` (BPE / WordPiece / WordLevel / ...).
#[derive(Debug)]
pub struct Tokenizer {
    inner: HfTokenizer,
}

impl Tokenizer {
    /// Load a `tokenizer.json` from disk.
    pub fn from_file(path: &Path) -> Result<Self, TokenizerError> {
        let raw = std::fs::read_to_string(path).map_err(|e| TokenizerError::Io {
            path: path.to_path_buf(),
            source: e,
        })?;
        Self::from_json(&raw)
    }

    /// Parse a `tokenizer.json` document from a string.
    pub fn from_json(raw: &str) -> Result<Self, TokenizerError> {
        // `Tokenizer` implements `Deserialize` (this is what `from_file` does
        // internally in the `tokenizers` crate).
        let inner = serde_json::from_str::<HfTokenizer>(raw)
            .map_err(|e| TokenizerError::Parse(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Count the tokens in `text` (no special tokens added). Returns `None`
    /// if this tokenizer cannot encode the input (exotic vocabularies);
    /// callers should then fall back to the `chars/4` estimate.
    pub fn try_count(&self, text: &str) -> Option<u32> {
        self.inner
            .encode(text, false)
            .ok()
            .map(|e| e.get_ids().len() as u32)
    }
}

/// A token count plus a flag for whether it is an estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenCount {
    pub tokens: u32,
    /// `true` when the count came from the `chars/4` heuristic (no tokenizer file).
    pub estimated: bool,
}

/// Count `text` with `tokenizer` when provided, else fall back to `chars/4`.
/// If the tokenizer fails to encode the input, the `chars/4` estimate is used
/// and the result is flagged `estimated` (graceful degradation, never panic).
pub fn count_tokens(text: &str, tokenizer: Option<&Tokenizer>) -> TokenCount {
    match tokenizer.and_then(|t| t.try_count(text)) {
        Some(tokens) => TokenCount {
            tokens,
            estimated: false,
        },
        None => TokenCount {
            tokens: (text.chars().count() / 4) as u32,
            estimated: true,
        },
    }
}
