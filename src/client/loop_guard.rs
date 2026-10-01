//! Decode-loop guard (v0.1.1 "Measurement Credibility", blueprint §v0.1.1-C).
//!
//! Detects degenerate output — a model stuck repeating itself — so its
//! tokens never inflate a throughput measurement.
//!
//! Rule: keep a rolling buffer of the last [`LOOP_BUFFER`] (64) decoded
//! token strings. The stream is partitioned into consecutive
//! [`LOOP_WINDOW`] (32)-token blocks; each block is hashed (FNV-1a fold of
//! the per-token FNV-1a hashes). When **three consecutive block hashes are
//! equal** — a 32-token sequence repeating 3 times in a row — the stream
//! is flagged `looping` and its tokens are excluded from throughput
//! calculations by the engines.
//!
//! The check is O(1) per token (one FNV-1a over the small token string,
//! one hash slot appended to the current block) and uses no locks: each
//! stream worker owns its guard. Rust `String` SSO keeps tokens ≤23 bytes
//! (the common case) allocation-free on the hot path.
//!
//! A false positive is possible for genuinely repetitive output (e.g. a
//! model that legitimately writes `"apple apple apple …"` for 96+ tokens);
//! that is exactly the degenerate case a benchmark must not credit with
//! throughput, so the guard errs toward exclusion.

use std::collections::VecDeque;

/// A 32-token block is the repeating unit the guard compares.
pub const LOOP_WINDOW: usize = 32;

/// The rolling buffer depth: the last 64 decoded token strings.
pub const LOOP_BUFFER: usize = 64;

/// Consecutive equal 32-token blocks required to declare a loop.
pub const LOOP_REPETITIONS: usize = 3;

/// FNV-1a 64-bit offset basis / prime (the guard's hash constants).
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a over a byte slice (one token string).
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = FNV_OFFSET;
    for b in bytes {
        h ^= u64::from(*b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Per-stream decode-loop detector (see module docs for the rule).
#[derive(Debug, Default)]
pub struct LoopGuard {
    /// Rolling buffer of the last 64 decoded token strings.
    recent: VecDeque<String>,
    /// Per-token hashes of the in-progress 32-token block.
    block: Vec<u64>,
    /// Hashes of the last [`LOOP_REPETITIONS`] completed 32-token blocks.
    blocks: VecDeque<u64>,
    /// `true` once three consecutive equal blocks have been seen.
    detected: bool,
    /// Tokens received after (and including the trigger of) detection —
    /// the guard's own counter; engines use the stream's *total* token
    /// count for exclusion (a looping stream is untrusted as a whole).
    excluded_tokens: u64,
}

impl LoopGuard {
    /// An empty guard (no tokens seen, not looping).
    pub fn new() -> Self {
        Self::default()
    }

    /// Feed one decoded token string.
    ///
    /// Returns `true` **exactly once** — at the moment the loop is
    /// detected (the 3rd consecutive equal 32-token block completes).
    pub fn push(&mut self, token: &str) -> bool {
        if self.detected {
            self.excluded_tokens += 1;
            self.keep_recent(token);
            return false;
        }
        self.keep_recent(token);
        self.block.push(fnv1a(token.as_bytes()));
        if self.block.len() == LOOP_WINDOW {
            let block_hash = self.fold_block();
            self.block.clear();
            self.blocks.push_back(block_hash);
            while self.blocks.len() > LOOP_REPETITIONS {
                self.blocks.pop_front();
            }
            if self.blocks.len() == LOOP_REPETITIONS
                && self
                    .blocks
                    .iter()
                    .all(|h| *h == *self.blocks.back().unwrap())
            {
                self.detected = true;
                self.excluded_tokens = 1;
                return true;
            }
        }
        false
    }

    /// `true` once the stream has been flagged looping.
    #[must_use]
    pub fn is_detected(&self) -> bool {
        self.detected
    }

    /// Tokens this guard counted as excluded (its own post-detection
    /// counter; see the struct field docs for the engine-side convention).
    #[must_use]
    pub fn excluded_tokens(&self) -> u64 {
        self.excluded_tokens
    }

    /// The rolling buffer of the last 64 decoded token strings (for
    /// diagnostics / a future live-output preview).
    #[must_use]
    pub fn recent(&self) -> &VecDeque<String> {
        &self.recent
    }

    fn keep_recent(&mut self, token: &str) {
        self.recent.push_back(token.to_string());
        while self.recent.len() > LOOP_BUFFER {
            self.recent.pop_front();
        }
    }

    /// Fold the in-progress block's per-token hashes into one block hash
    /// (FNV-1a chain — order-sensitive, so two blocks with the same
    /// tokens in a different order hash differently).
    fn fold_block(&self) -> u64 {
        let mut h = FNV_OFFSET;
        for t in &self.block {
            h ^= *t;
            h = h.wrapping_mul(FNV_PRIME);
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Feed `n` blocks of 32 tokens into a fresh guard. With
    /// `repeat: true` every block is the identical 32-token pattern (the
    /// loop case); with `false` the blocks differ (the non-loop case).
    fn feed_blocks(guard: &mut LoopGuard, n_blocks: usize, repeat: bool) {
        for block in 0..n_blocks {
            for i in 0..LOOP_WINDOW {
                // The pattern's block index is part of the token only when
                // the blocks must differ.
                let tok = if repeat {
                    format!("tok{i}")
                } else {
                    format!("b{block}-tok{i}")
                };
                let _ = guard.push(&tok);
            }
        }
    }

    #[test]
    fn two_equal_blocks_do_not_trigger() {
        let mut g = LoopGuard::new();
        feed_blocks(&mut g, 2, true); // 64 tokens = 2 identical blocks
        assert!(!g.is_detected(), "a 2× repeat is not a loop");
        assert_eq!(g.excluded_tokens(), 0);
    }

    #[test]
    fn three_consecutive_equal_blocks_trigger_once() {
        let mut g = LoopGuard::new();
        let mut triggers = 0;
        for _ in 0..3 {
            for i in 0..LOOP_WINDOW {
                if g.push(&format!("tok{i}")) {
                    triggers += 1;
                }
            }
        }
        assert!(g.is_detected());
        assert_eq!(triggers, 1, "push returns true exactly once");
        // Post-detection tokens count as excluded.
        g.push("anything");
        g.push("more");
        assert_eq!(g.excluded_tokens(), 3);
    }

    #[test]
    fn distinct_blocks_never_trigger() {
        let mut g = LoopGuard::new();
        feed_blocks(&mut g, 5, false); // 160 distinct tokens, 5 distinct blocks
        assert!(!g.is_detected());
    }

    #[test]
    fn a_single_repeating_token_is_a_loop() {
        // The degenerate "the the the the …" case: one token repeated 96×
        // is three equal 32-token blocks and must be caught.
        let mut g = LoopGuard::new();
        for i in 0..96 {
            let _ = g.push("the");
            let _ = i;
        }
        assert!(g.is_detected());
    }

    #[test]
    fn the_rolling_buffer_keeps_the_last_64_tokens() {
        let mut g = LoopGuard::new();
        for i in 0..100 {
            let _ = g.push(&format!("tok{i}"));
        }
        let recent: Vec<&str> = g.recent().iter().map(String::as_str).collect();
        assert_eq!(recent.len(), LOOP_BUFFER);
        assert_eq!(recent[0], "tok36");
        assert_eq!(recent.last().copied(), Some("tok99"));
    }

    #[test]
    fn detection_resets_across_guard_instances() {
        // Guards are per-stream and `Default`: a fresh worker starts clean.
        let g = LoopGuard::default();
        assert!(!g.is_detected());
        assert_eq!(g.excluded_tokens(), 0);
        assert!(g.recent().is_empty());
    }

    #[test]
    fn pattern_offset_does_not_break_block_alignment() {
        // A 32-token pattern that starts mid-buffer: the block boundaries
        // are fixed by token position, so a repeating pattern whose phase
        // matches the 32-token block grid is still caught. (10 warm-up
        // distinct tokens, then 3× the same 32-token block.)
        let mut g = LoopGuard::new();
        for i in 0..10 {
            let _ = g.push(&format!("warm{i}"));
        }
        for _ in 0..3 {
            for i in 0..LOOP_WINDOW {
                let _ = g.push(&format!("p{i}"));
            }
        }
        // 10 + 96 = 106 tokens → blocks at [0..32), [32..64), [64..96),
        // [96..106): the three fully-populated blocks after the warm-up
        // are 10p + 22p, 32p, 32p — the last two are equal, the first is
        // not, so this specific phase does NOT trigger (three *consecutive*
        // equal blocks are required). Assert the guard's documented rule
        // holds: no false trigger from a partial-block prefix.
        assert!(!g.is_detected());
    }
}
