//! Chunk 4 acceptance tests: prompt generator + client-side tokenizer.
//!
//! Uses a hand-crafted WordLevel `tokenizer.json` fixture so the suite runs
//! offline with a deterministic, hand-countable reference.

use std::path::Path;

use crucible_llm::prompt::{count_tokens, PromptGenerator, Tokenizer, LONG_BASE, SHORT_PROMPT};

fn fixture_tokenizer() -> Tokenizer {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/mini_tokenizer.json");
    Tokenizer::from_file(&path).expect("mini tokenizer fixture must parse")
}

// ── tokenizer ────────────────────────────────────────────────────────────────

#[test]
fn tokenizer_counts_known_string_against_hand_reference() {
    let tok = fixture_tokenizer();
    // WordLevel: one token per whitespace-separated word.
    assert_eq!(tok.try_count("apple apple apple"), Some(3));
    assert_eq!(tok.try_count("context window"), Some(2));
    assert_eq!(tok.try_count("the quick brown fox"), Some(4));
    // An out-of-vocabulary word collapses to a single UNK token.
    assert_eq!(tok.try_count("apple zebra apple"), Some(3));
    assert_eq!(tok.try_count(""), Some(0));
}

#[test]
fn tokenizer_long_base_is_18_tokens() {
    let tok = fixture_tokenizer();
    // Hand count: the Whitespace pre-tokenizer splits on `\w+|[^\w\s]+`, so
    // "briefly," → ["briefly", ","] and "size:" → ["size", ":"]:
    // Repeat | the | following | sequence | back | to | me | briefly | ,
    // then | explain | the | importance | of | context | window | size | :
    assert_eq!(tok.try_count(LONG_BASE), Some(18));
}

#[test]
fn count_tokens_falls_back_to_chars_over_4() {
    let c = count_tokens(&"abcd".repeat(10), None); // 40 chars
    assert!(c.estimated);
    assert_eq!(c.tokens, 10);
}

#[test]
fn count_tokens_exact_with_tokenizer() {
    let tok = fixture_tokenizer();
    let c = count_tokens("apple apple", Some(&tok));
    assert!(!c.estimated);
    assert_eq!(c.tokens, 2);
}

// ── short ────────────────────────────────────────────────────────────────────

#[test]
fn short_prompt_is_fixed_and_about_50_tokens() {
    let g = PromptGenerator::new(None);
    let p = g.short();
    assert_eq!(p.text, SHORT_PROMPT);
    assert!(!p.nocache);
    // chars/4 estimate of the fixed ~50-token prompt.
    assert!((40..=60).contains(&p.token_count), "got {}", p.token_count);
    assert!(p.estimated);
}

#[test]
fn short_prompt_counted_exactly_with_tokenizer() {
    let g = PromptGenerator::new(Some(fixture_tokenizer()));
    let p = g.short();
    assert!(!p.estimated);
    // Hand count: every chunk of the short prompt is out of the mini vocab →
    // one UNK token each. Under `\w+|[^\w\s]+` the prompt splits into 37
    // chunks (30 words, with "time." / "works," / "it's" / "index." /
    // "B-tree" each splitting into 2-3 punctuation-aware chunks).
    assert_eq!(p.token_count, 37);
}

// ── long (chars/4 fallback) ──────────────────────────────────────────────────

#[test]
fn long_prompt_without_tokenizer_is_within_10pct() {
    let g = PromptGenerator::new(None);
    let p = g.long(2000);
    assert!(p.estimated);
    let err = (p.token_count as i64 - 2000).abs() as f64 / 2000.0;
    assert!(
        err <= 0.10,
        "estimated {} tokens, {}% off target",
        p.token_count,
        err * 100.0
    );
    // Built from the llmspeedtest.py rotating sentence bank.
    assert!(p
        .text
        .contains("The quick brown fox jumps over the lazy dog."));
    assert!(p
        .text
        .contains("Type systems prevent entire classes of runtime errors."));
}

#[test]
fn long_prompt_without_tokenizer_is_deterministic() {
    let g = PromptGenerator::new(None);
    assert_eq!(g.long(500).text, g.long(500).text);
}

#[test]
fn long_prompt_target_below_base_returns_base() {
    let g = PromptGenerator::new(Some(fixture_tokenizer()));
    let p = g.long(5); // base is 18 tokens in the mini tokenizer
    assert_eq!(p.text, LONG_BASE);
    assert_eq!(p.token_count, 18);
}

// ── long (tokenized) ─────────────────────────────────────────────────────────

#[test]
fn long_prompt_with_tokenizer_hits_target_exactly() {
    let g = PromptGenerator::new(Some(fixture_tokenizer()));
    let p = g.long(2000);
    assert!(!p.estimated);
    // " apple" is exactly 1 token in the mini vocab → exact hit.
    assert_eq!(p.token_count, 2000);
    assert!(p.text.starts_with(LONG_BASE));
    assert!(p.text.ends_with('e')); // ...padding ends with " apple"
}

#[test]
fn long_prompt_with_tokenizer_scales() {
    let g = PromptGenerator::new(Some(fixture_tokenizer()));
    for target in [100u32, 500, 2000, 8000] {
        let p = g.long(target);
        let err = (p.token_count as i64 - target as i64).abs() as f64 / target as f64;
        assert!(
            err <= 0.05,
            "target {target}: got {} ({}% off)",
            p.token_count,
            err * 100.0
        );
    }
}

// ── nocache ──────────────────────────────────────────────────────────────────

#[test]
fn nocache_prefix_is_unique_each_call() {
    let g = PromptGenerator::new(None);
    let a = g.nocache_short();
    let b = g.nocache_short();
    assert_ne!(a.text, b.text);
    assert!(a.nocache && b.nocache);
    // [32-char hex uuid] + space: `[` at 0, hex at 1..33, `]` at 33, ` ` at 34
    let close = a.text.find(']').expect("prefix must be bracketed");
    assert_eq!(&a.text[..1], "[");
    assert_eq!(close, 33);
    let hex = &a.text[1..33];
    assert!(hex.bytes().all(|b| b.is_ascii_hexdigit()));
    assert_eq!(&a.text[34..35], " ");
}

#[test]
fn nocache_long_changes_prefix_and_keeps_body() {
    let g = PromptGenerator::new(None);
    let plain = g.long(300);
    let a = g.nocache_long(300);
    let b = g.nocache_long(300);
    assert_ne!(a.text, b.text);
    // Body after the prefix is identical to the non-nocache long prompt.
    assert!(a.text.ends_with(plain.text.as_str()));
    assert!(a.nocache);
    assert!(!plain.nocache);
}
