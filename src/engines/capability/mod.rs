//! Engine C — Capability & Fidelity: needle-in-a-haystack, deterministic
//! reasoning/code verification, and structured-output/JSON-grammar compliance.
//!
//! * [`niah`] — (Chunk 15) Engine C1: the N×M context-size × depth
//!   matrix measuring retrieval accuracy and prefill degradation.
//! * [`reasoning`] — (Chunk 16) Engine C2: a standardized bank of
//!   deterministic logic/math/code challenges validated by strict
//!   checkers, yielding a reproducible pass/fail accuracy score.
//! * [`structured`] — (Chunk 16) Engine C3: the speed penalty of
//!   grammar-constrained (`response_format`) generation vs free-form,
//!   plus JSON compliance.

pub mod niah;
pub mod reasoning;
pub mod structured;

pub use niah::{
    build_document, classify, Needle, NiahCell, NiahCellState, NiahDocument, NiahEngine,
    NiahEngineConfig, NiahResult, NiahSlot, NIAH_DEPTHS, NIAH_MAX_GEN_TOKENS, NIAH_SIZES,
    PREFILL_THROTTLE_FACTOR,
};
pub use reasoning::{
    score_responses, Challenge, Checker, ReasoningEngine, ReasoningResult, ReasoningScore,
    REASONING_BANK, REASONING_MAX_GEN_TOKENS,
};
pub use structured::{
    evaluate_case, is_json_compliant, CaseCheck, CaseVerdict, StructuredCase, StructuredCaseResult,
    StructuredEngine, StructuredResult, STRUCTURED_CASES, STRUCTURED_MAX_GEN_TOKENS,
};
