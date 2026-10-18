//! Engine C3 — Structured Output & JSON Grammar Compliance (plan Chunk 16,
//! blueprint §5 Engine C3).
//!
//! C3 answers two questions for **API / agent** use:
//!
//! 1. **How well does the model follow a JSON schema?** A *ladder* of three
//!    cases of increasing complexity (Simple flat object → Medium
//!    fixed-length array → Complex nested object) is each run with a
//!    `response_format` directive and scored on a small set of checks
//!    (valid JSON, schema shape, all fields, correct types). Each case is
//!    classified [`CaseVerdict`]:
//!    * **Compliant** — every check passes;
//!    * **Partial** — valid JSON, but one or more checks fail;
//!    * **Failed** — the output is not valid JSON (cannot be evaluated);
//!    * **Truncated** (R2, `truncated_by_reasoning`) — the stream completed
//!      OK but the budget was exhausted in chain-of-thought: all-CoT, zero
//!      content. A distinct "inconclusive (budget)" outcome, reported
//!      separately from the compliance score (never a capability failure).
//!
//!    **What the directive actually is (CF-08).** Each constrained run
//!    sends a real `response_format: { "type": "json_schema",
//!    "json_schema": {…} }` carrying the case's strict schema, which
//!    engages server-side **constrained decoding** (vLLM XGrammar, Outlines,
//!    GBNF). The shared `StreamWorker` transport encodes the object verbatim
//!    (`response_format_value`); backends that do not support
//!    `response_format` (llama.cpp, TGI) *drop* the field per the OpenAI
//!    spec — a graceful no-op. If a transport cannot encode the object, C3
//!    falls back to the softer `response_format: { "type": "json_object" }`
//!    **hint** — a directive that biases the model toward JSON but is *not*
//!    a grammar constraint. The [`ConstraintMode`] of each run is recorded
//!    and **flagged** in the result and summary line, so a ~0% "penalty" on
//!    a hint-ignoring backend is never silently misread as "constrained
//!    decoding is free".
//! 2. **What does grammar validation cost in speed?** One free-form
//!    baseline run (case 1, unconstrained) vs the constrained run (case 1)
//!    measures `penalty% = (free_tps − constrained_tps) / free_tps × 100`
//!    (positive = the constraint costs throughput). The penalty is only a
//!    valid *constrained-decoding* measurement in
//!    [`ConstraintMode::JsonSchema`]; in the `JsonObjectHint` fallback it
//!    measures the cost of a prompt hint (≈0 where the backend drops the
//!    field).
//!
//! **Reasoning-model handling (hybrid, R2 — Option C).** Every run sets
//! `disable_thinking` on the shared worker (per-backend and graceful —
//! `StreamWorker::disable_thinking`, CF-01/CF-18) as a **best-effort speed
//! optimization, not a correctness dependency**: on servers that honor it
//! (vLLM/SGLang/LM Studio/Ollama) the model answers without CoT and the
//! budget is a rarely-hit ceiling. A *config-only* server silently drops the
//! per-request fields (OpenAI-spec graceful degradation) and the model
//! thinks by default — which is why **the 4096-token budget is the
//! correctness guarantee** (`max_tokens` is honored by every server). It
//! leaves headroom for chain-of-thought *before* the JSON, so a
//! suppression-ignoring model can finish its CoT and still commit the
//! answer, while the content-only accumulation keeps judging the visible
//! span. A stream that exhausts the budget in CoT with zero content is the
//! distinct [`CaseVerdict::Truncated`] (`truncated_by_reasoning`) outcome —
//! an honest "inconclusive (budget)", never a false capability failure
//! (`work/plans/REASONING_DESIGN.md` §5).
//!
//! **Checker hardening (Phase 3, CF-17/36/50/51).** With a real
//! `json_schema` in place (CF-08), the scoring was hardened to report what it
//! actually measures:
//! * **CF-17 — valid-JSON vs. schema-correct, split.** A response can parse
//!   as JSON yet violate the schema; [`StructuredResult::compliance_rates`] /
//!   [`compliance_label`] report the two signals separately instead of one
//!   blended number.
//! * **CF-36 — the penalty only when the constraint is effective.**
//!   [`StructuredResult::constraint_effective`] is true only when the
//!   constrained case-0 run produced valid JSON; otherwise the summary flags
//!   "constraint not effective" instead of reading a +100% / ~0% number.
//! * **CF-50 — integer-valued floats.** `is_int` accepts a finite float with
//!   `fract() == 0.0`, so a model emitting `36.0` for an `integer` field is
//!   not false-failed to Partial.
//! * **CF-51 — repeat-for-stability.** Each constrained case is repeated
//!   `self.repeats`× (default 1, single-shot; [`STABILITY_REPEATS`] = 3 is
//!   the recommended reliability measurement); [`fold_case_repeats`] folds
//!   the repeats and [`StructuredResult::stability_rate`] reports the
//!   fraction that were fully compliant — the data prerequisite for the
//!   planned McNemar/Wilson A/B comparator.
//!
//! The [`StructuredResult`] carries the per-case checks, the overall score
//! (N/3 compliant / partial / failed, plus the distinct R2
//! `truncated_by_reasoning` bucket via [`StructuredResult::truncated_count`]),
//! the split compliance rates, the speed penalty (gated on the constraint
//! being effective — a truncated case is never penalized as a compliance
//! failure), the stability rate, the constraint mode, and an
//! auto-generated practical **verdict** — the TUI renders the full
//! compliance detail in the Live capability panel and a summary line in the
//! sequence header / headless output.
//!
//! Measurement-isolation note: every run reuses the Chunk 5 `StreamWorker`
//! (quanta `T0..Tn` live in the worker); this module only takes deltas of
//! those records.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::client::{join_worker, spawn_worker, token_window, StreamEvent, StreamWorker};
use crate::config::Config;
use crate::engines::sequence::{EngineProgress, ProgressBus, RunPause};
use crate::engines::speed::{EngineError, SpeedEngine};
use crate::metrics::state::MetricsState;
use crate::sse::Chunk;
use crate::timing::StreamTimestamps;

/// Bounded worker→engine channel capacity (same as Engine A).
const CHANNEL_CAPACITY: usize = 256;

/// Max generation tokens per structured run (the default; overridable at
/// runtime via `--structured-max-tokens` / `CRUCIBLE_STRUCTURED_MAX_TOKENS`
/// / the config file — R3).
///
/// **Hybrid reasoning (R2, Option C):** 4096 is the *correctness
/// guarantee*. The JSON answer itself is small (the largest case, Complex,
/// needs well under 200 tokens) and is further constrained by the case's
/// strict `json_schema`; 4096 is headroom for a reasoning model's
/// chain-of-thought *before* the JSON on a server that ignores our
/// best-effort `disable_thinking` suppression (R1: the budget carries
/// correctness, suppression is only a speed optimization). A stream that
/// exhausts the budget in CoT with zero content is the distinct
/// `truncated_by_reasoning` outcome ([`CaseVerdict::Truncated`]), not a
/// capability failure.
///
/// **R3 calibration (live, 2026-10-11, qwen3.8-27b finetune @
/// 192.168.51.163):** the token-efficient finetune used **0** CoT tokens
/// per case (the vLLM server honors `enable_thinking=false`); forced to
/// think (`--no-disable-thinking`) it spent 57–64+ CoT on the C3 prompts
/// (the 64-token force-test budget was exhausted mid-CoT on 2 of 3 cases).
/// 4096 leaves **large headroom** above that: a verbose reasoner (several×
/// the finetune's ~60) can spend ~3800 tokens on CoT and still commit the
/// ≤200-token JSON answer.
pub const STRUCTURED_MAX_GEN_TOKENS: u32 = 4096;

/// CF-51: how many times each constrained case is repeated. The stability
/// rate (the fraction of repeats that are fully compliant) turns C3's old
/// single-shot score into a reliability measurement — the data prerequisite
/// for the planned McNemar/Wilson A/B comparator.
pub const STABILITY_REPEATS: usize = 3;

/// The constraint mechanism a C3 "constrained" run actually requests (CF-08).
///
/// The docs once claimed C3 engages server-side *constrained decoding*
/// (XGrammar/Outlines/GBNF) via `json_object` — that is false. `json_object`
/// is a soft *hint* that most backends ignore and that never drives a grammar
/// stack. Recording the mode makes the "grammar speed penalty" honest: a
/// constrained run is flagged as real `json_schema` constrained decoding,
/// or — in the fallback — as a *hint* measurement (not a
/// constrained-decoding one).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ConstraintMode {
    /// The free-form baseline run (no `response_format` at all).
    FreeForm,
    /// A soft `response_format: { "type": "json_object" }` *hint* — biases the
    /// model toward JSON but does **not** engage a grammar/decoder constraint.
    /// The flagged fallback for a transport that cannot encode the schema
    /// object.
    JsonObjectHint,
    /// A real `response_format: { "type": "json_schema", "json_schema": {…} }`
    /// that engages server-side constrained decoding (XGrammar/Outlines).
    /// The active mode: the `StreamWorker` transport encodes the schema
    /// object verbatim (the CF-08 handoff is closed).
    JsonSchema,
}

impl ConstraintMode {
    /// The short label used in summaries / the UI.
    pub fn label(self) -> &'static str {
        match self {
            ConstraintMode::FreeForm => "free-form",
            ConstraintMode::JsonObjectHint => "json_object hint (not schema-constrained)",
            ConstraintMode::JsonSchema => "json_schema (constrained decoding)",
        }
    }
}

/// Whether the shared `StreamWorker` transport can encode a full `json_schema`
/// `response_format` object.
///
/// **`true` — the CF-08 handoff is closed.** `client/stream.rs` carries
/// `response_format` as a verbatim `serde_json::Value`
/// (`response_format_value`), so the full `{ "type": "json_schema",
/// "json_schema": {…} }` object C3 builds reaches the wire; backends that do
/// not support `response_format` (llama.cpp, TGI) drop the field per the
/// OpenAI spec — a graceful no-op. C3 therefore runs in
/// [`ConstraintMode::JsonSchema`]; [`ConstraintMode::JsonObjectHint`]
/// remains the flagged fallback for a transport that cannot encode the
/// object.
const TRANSPORT_SUPPORTS_JSON_SCHEMA: bool = true;

/// Choose the constraint mode for a C3 "constrained" run (CF-08).
///
/// [`ConstraintMode::JsonSchema`] is the target (real constrained decoding).
/// It is selected only when the transport can encode the schema object;
/// otherwise we gracefully fall back to the [`ConstraintMode::JsonObjectHint`]
/// — and the result flags which mode was actually used, so a ~0% penalty on a
/// hint-ignoring backend is never a silent false-negative.
#[must_use]
pub fn select_constraint_mode(transport_supports_schema: bool) -> ConstraintMode {
    if transport_supports_schema {
        ConstraintMode::JsonSchema
    } else {
        ConstraintMode::JsonObjectHint
    }
}

/// The real OpenAI `response_format` object for constrained decoding (CF-08):
/// `{"type": "json_schema", "json_schema": {"name", "schema", "strict"}}`.
///
/// This is the target artifact that engages the server-side grammar stack
/// (XGrammar/Outlines). C3 owns the per-case schemas and builds the object
/// here; wiring it into the HTTP body is the `client/stream.rs` enhancement
/// (sibling-owned). `strict: true` is the OpenAI recommended flag for
/// schema-constrained generation.
#[must_use]
pub fn json_schema_response_format(name: &str, schema: &serde_json::Value) -> serde_json::Value {
    serde_json::json!({
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": schema,
            "strict": true,
        }
    })
}

/// The complexity ladder: three cases of increasing schema depth. Each
/// prompt asks for a *specific* JSON shape so the constrained output can be
/// checked field-by-field.
#[derive(Debug, Clone, Copy)]
pub struct StructuredCase {
    /// The display name (`Simple` / `Medium` / `Complex`).
    pub name: &'static str,
    /// The generation prompt (the *only* thing that differs between runs is
    /// the `response_format` constraint — the prompt is identical for the
    /// free-form baseline and its constrained pair).
    pub prompt: &'static str,
}

/// The three C3 cases (blueprint §5 C3: simple → complex schema adherence).
pub const STRUCTURED_CASES: [StructuredCase; 3] = [
    StructuredCase {
        name: "Simple",
        prompt: "Return a JSON object with fields: \"name\" (a string), \
                 \"age\" (an integer). Respond with the JSON object only.",
    },
    StructuredCase {
        name: "Medium",
        prompt: "Return a JSON array of exactly 3 objects, each with: \
                 \"id\" (an integer), \"label\" (a string), \"active\" (a \
                 boolean). Respond with the JSON array only.",
    },
    StructuredCase {
        name: "Complex",
        prompt: "Return a JSON object matching this schema: \
                  {{\"user\": {{\"name\": string, \"email\": string}}, \
                  \"orders\": [{{\"id\": int, \"total\": number, \
                  \"items\": [string]}}]}}. Respond with the JSON object only.",
    },
];

/// The per-case strict JSON schemas (CF-08), index-aligned with
/// [`STRUCTURED_CASES`]: the `schema` each constrained run sends inside
/// `response_format: { "type": "json_schema", "json_schema": {…} }`.
///
/// They are **strict-mode** shapes: OpenAI `strict: true` requires
/// `additionalProperties: false` on every object, and a schema-capable
/// backend (vLLM XGrammar) rejects non-strict schemas outright. Each
/// mirrors its case prompt exactly (Simple flat object → Medium
/// fixed-length array → Complex nested object).
///
/// Stored as JSON strings — `serde_json::Value` is not `Copy`, so it cannot
/// live in a `const` array — and parsed on demand (three parses per engine
/// run; negligible).
pub const STRUCTURED_CASE_SCHEMAS: [&str; 3] = [
    r#"{
        "type": "object",
        "properties": {
            "name": { "type": "string" },
            "age": { "type": "integer" }
        },
        "required": ["name", "age"],
        "additionalProperties": false
    }"#,
    r#"{
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "id": { "type": "integer" },
                "label": { "type": "string" },
                "active": { "type": "boolean" }
            },
            "required": ["id", "label", "active"],
            "additionalProperties": false
        },
        "minItems": 3,
        "maxItems": 3
    }"#,
    r#"{
        "type": "object",
        "properties": {
            "user": {
                "type": "object",
                "properties": {
                    "name": { "type": "string" },
                    "email": { "type": "string" }
                },
                "required": ["name", "email"],
                "additionalProperties": false
            },
            "orders": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer" },
                        "total": { "type": "number" },
                        "items": {
                            "type": "array",
                            "items": { "type": "string" }
                        }
                    },
                    "required": ["id", "total", "items"],
                    "additionalProperties": false
                }
            }
        },
        "required": ["user", "orders"],
        "additionalProperties": false
    }"#,
];

/// Case `i`'s strict schema parsed to a `Value` (index-aligned with
/// [`STRUCTURED_CASES`]).
fn case_schema(i: usize) -> serde_json::Value {
    serde_json::from_str(STRUCTURED_CASE_SCHEMAS[i])
        .expect("C3 case schemas are valid JSON (checked in tests)")
}

/// The per-case compliance verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CaseVerdict {
    /// Every check passed.
    Compliant,
    /// Valid JSON, but one or more checks failed.
    Partial,
    /// Not valid JSON — cannot be evaluated. The conservative `Default`
    /// (a missing/unknown verdict in a persisted summary must never read
    /// as compliant).
    #[default]
    Failed,
    /// **R2 hybrid reasoning (`truncated_by_reasoning`).** The stream
    /// completed OK but the model spent the entire generation budget on
    /// chain-of-thought — all-CoT, zero content committed. A distinct
    /// "inconclusive (budget)" outcome: *not* a schema/compliance failure
    /// (there is no content to fail a check against) and *not* a pass (we
    /// cannot verify a capability we never saw committed).
    Truncated,
}

/// One compliance check within a case (the `✓`/`✗` items in the UI).
#[derive(Debug, Clone)]
pub struct CaseCheck {
    /// The check label (`Valid JSON`, `Schema match`, `All fields`,
    /// `Types`, `Array length`).
    pub label: String,
    pub passed: bool,
}

/// The scored result of one case (its checks + verdict + raw output).
#[derive(Debug, Clone)]
pub struct StructuredCaseResult {
    /// The case name (`Simple` / `Medium` / `Complex`).
    pub name: String,
    /// The per-check results, in display order.
    pub checks: Vec<CaseCheck>,
    /// The classified outcome.
    pub verdict: CaseVerdict,
    /// The raw constrained output (shown, truncated, for non-compliant cases).
    pub output: String,
    /// CF-51: how many times this case was run (1 for a single-shot
    /// [`evaluate_case`]; [`STABILITY_REPEATS`] for the repeated `run()` path).
    pub repeats: usize,
    /// CF-51: how many of those repeats were fully compliant.
    pub compliant_repeats: usize,
    /// **R2 hybrid reasoning:** `true` when a repeat's stream completed OK
    /// but committed no content — the budget was exhausted in
    /// chain-of-thought (`truncated_by_reasoning`, the distinct non-fail
    /// outcome). For a single-shot case this equals
    /// [`verdict`](Self::verdict) == [`CaseVerdict::Truncated`]; for a
    /// repeated case it marks that *any* repeat was truncated (the folded
    /// verdict is `Truncated` only when *all* repeats were).
    pub truncated_by_reasoning: bool,
    /// The server-reported chain-of-thought tokens for this case (R3
    /// calibration measurement): `usage.completion_tokens_details.
    /// reasoning_tokens`. For a repeated case this is the **max** over its
    /// repeats (the headroom-relevant worst case). `None` when the backend
    /// did not report a usage block (the honest "not reported" case);
    /// `Some(0)` when it reported zero CoT (thinking suppression worked).
    pub reasoning_tokens: Option<u64>,
}

impl StructuredCaseResult {
    /// `true` when the `Valid JSON` check (the first check) passed.
    pub fn is_valid_json(&self) -> bool {
        self.checks
            .first()
            .is_some_and(|c| c.label == "Valid JSON" && c.passed)
    }

    /// `true` when the case is fully compliant.
    pub fn is_compliant(&self) -> bool {
        self.verdict == CaseVerdict::Compliant
    }

    /// CF-51: the case's stability rate — the fraction of its repeats that
    /// were fully compliant (`[0.0, 1.0]`). For a single-shot case this is
    /// 1.0 when compliant, 0.0 otherwise.
    #[must_use]
    pub fn stability_rate(&self) -> f64 {
        if self.repeats == 0 {
            0.0
        } else {
            self.compliant_repeats as f64 / self.repeats as f64
        }
    }
}

/// The full structured-output evaluation result.
#[derive(Debug, Clone)]
pub struct StructuredResult {
    /// Free-form (unconstrained) decode speed, tokens/s (case 1 baseline).
    pub free_tps: f64,
    /// Constrained (grammar-validated) decode speed, tokens/s (case 1).
    pub constrained_tps: f64,
    /// `(free − constrained) / free × 100` — positive = the grammar
    /// constraint costs throughput (the quantified penalty).
    pub penalty_pct: f64,
    pub free_ttft: f64,
    pub constrained_ttft: f64,
    /// The three scored cases (Simple → Complex).
    pub cases: Vec<StructuredCaseResult>,
    /// The constrained case-1 output (kept for headless / export detail).
    pub constrained_body: String,
    /// The free-form case-1 output.
    pub free_body: String,
}

impl StructuredResult {
    /// CF-08: the constraint mode this result's "constrained" runs used.
    ///
    /// Exposed as a method (not a stored field) so the struct-literal
    /// construction sites in the UI / tests are unaffected. The transport
    /// encodes the `json_schema` object (the CF-08 handoff is closed), so
    /// constrained runs use [`ConstraintMode::JsonSchema`]; the
    /// [`ConstraintMode::JsonObjectHint`] fallback remains for a transport
    /// that cannot.
    pub fn constraint_mode(&self) -> ConstraintMode {
        select_constraint_mode(TRANSPORT_SUPPORTS_JSON_SCHEMA)
    }

    /// `(compliant, partial, failed)` counts over the cases.
    ///
    /// **R2:** the distinct [`CaseVerdict::Truncated`] bucket
    /// (`truncated_by_reasoning`) is *not* counted in any of these three —
    /// a case whose budget was exhausted in CoT is neither compliant,
    /// partially compliant, nor a compliance failure. It is reported
    /// separately via [`truncated_count`](Self::truncated_count).
    pub fn score(&self) -> (usize, usize, usize) {
        let (mut c, mut p, mut f) = (0, 0, 0);
        for case in &self.cases {
            match case.verdict {
                CaseVerdict::Compliant => c += 1,
                CaseVerdict::Partial => p += 1,
                CaseVerdict::Failed => f += 1,
                CaseVerdict::Truncated => {}
            }
        }
        (c, p, f)
    }

    /// **R2 hybrid reasoning:** the number of cases whose folded verdict is
    /// [`CaseVerdict::Truncated`] — the stream completed OK but the budget
    /// was exhausted in chain-of-thought (all-CoT, zero content). The
    /// distinct "inconclusive (budget)" bucket, kept out of
    /// [`score`](Self::score) and out of every compliance penalty.
    #[must_use]
    pub fn truncated_count(&self) -> usize {
        self.cases
            .iter()
            .filter(|c| c.verdict == CaseVerdict::Truncated)
            .count()
    }

    /// `true` when *every* case is fully compliant (the old boolean flag).
    pub fn fully_compliant(&self) -> bool {
        self.cases
            .iter()
            .all(|c| c.verdict == CaseVerdict::Compliant)
    }

    /// A compact score label (`2/3 compliant, 1 partial, 0 failed`, with
    /// the distinct R2 truncated bucket appended when present:
    /// `2/3 compliant, 0 partial, 0 failed, 1 truncated`).
    pub fn score_label(&self) -> String {
        let (c, p, f) = self.score();
        let t = self.truncated_count();
        let total = self.cases.len();
        if p == 0 && f == 0 && t == 0 {
            format!("{c}/{total} compliant")
        } else if f == 0 && t == 0 {
            format!("{c}/{total} compliant, {p} partial")
        } else {
            format!("{c}/{total} compliant, {p} partial, {f} failed, {t} truncated")
        }
    }

    /// The auto-generated practical verdict (based on the compliant count):
    /// 3/3 → API/agent-ready; 2/3 → simple-only; 1/3 → trivial only;
    /// 0/3 → not suitable.
    ///
    /// **R2:** a case that was [`CaseVerdict::Truncated`] (budget exhausted
    /// in CoT, no content committed) is *inconclusive*, not a failure — so
    /// when every case is truncated the verdict says "inconclusive (budget)"
    /// instead of "not suitable", and a zero-compliant result whose
    /// non-compliant cases are all truncated (no real schema failures) reads
    /// as inconclusive too.
    pub fn verdict_line(&self) -> String {
        let (c, _p, f) = self.score();
        let t = self.truncated_count();
        let total = self.cases.len();
        if t == total {
            return format!(
                "⚠ Inconclusive — all {total} cases truncated by reasoning (budget exhausted in CoT; no content committed to score)"
            );
        }
        match c {
            3 => "✓ Fully suitable for API/agent use".to_string(),
            2 => "⚠ Suitable for simple structures, unreliable for complex schemas".to_string(),
            1 => "⚠ Limited use — only trivial key-value extraction".to_string(),
            _ => {
                if f == 0 && t > 0 {
                    format!(
                        "⚠ Inconclusive — {c} compliant, {t} case(s) truncated by reasoning (no content committed to score)"
                    )
                } else {
                    "✗ Not suitable for structured output. Requires output parsing/fallback."
                        .to_string()
                }
            }
        }
    }

    /// CF-17: the two distinct compliance signals, kept separate because a
    /// response can parse as JSON yet violate the schema.
    ///
    /// Returns `(valid_json_rate, schema_correct_rate)` as fractions in
    /// `[0.0, 1.0]` over the scored cases:
    /// * `valid_json_rate` — the fraction whose output parses as JSON
    ///   (Compliant + Partial);
    /// * `schema_correct_rate` — the fraction that are fully Compliant (valid
    ///   JSON *and* every schema / field / type check passes).
    ///
    /// The old single "compliance" number conflated these; with a real
    /// `json_schema` in place (CF-08) they are two different capabilities
    /// (a decoder/model that emits JSON vs. one that emits *schema-correct*
    /// JSON) and are now reported separately.
    #[must_use]
    pub fn compliance_rates(&self) -> (f64, f64) {
        let total = self.cases.len();
        if total == 0 {
            return (0.0, 0.0);
        }
        let valid_json = self.cases.iter().filter(|c| c.is_valid_json()).count();
        let schema_correct = self.cases.iter().filter(|c| c.is_compliant()).count();
        (
            valid_json as f64 / total as f64,
            schema_correct as f64 / total as f64,
        )
    }

    /// CF-17: a two-part compliance label that keeps "valid JSON" and
    /// "schema-correct" distinct (e.g. `valid JSON 3/3 · schema-correct
    /// 2/3`), so a response that parses as JSON yet violates the schema is
    /// reported as valid-JSON but *not* schema-correct — never one blended
    /// number.
    #[must_use]
    pub fn compliance_label(&self) -> String {
        let total = self.cases.len();
        let valid_json = self.cases.iter().filter(|c| c.is_valid_json()).count();
        let schema_correct = self.cases.iter().filter(|c| c.is_compliant()).count();
        format!("valid JSON {valid_json}/{total} · schema-correct {schema_correct}/{total}")
    }

    /// CF-36: `true` when the constrained case-0 run produced valid JSON, so
    /// the `penalty_pct` is a genuine constrained-decoding measurement.
    ///
    /// The old code computed `(free − constrained) / free` unconditionally, so
    /// a *crashed/empty* constrained run read as +100% (indistinguishable from
    /// "total failure") and a *dropped* schema (the backend ignored the
    /// constraint) read as ~0% (indistinguishable from "the constraint is
    /// free"). Both are meaningless. The penalty is only *effective* when the
    /// constrained run actually emitted valid JSON; otherwise it is flagged
    /// "constraint not effective" and never read as a number.
    ///
    /// Derived from the stored cases (not a new field) so the external
    /// `StructuredResult` construction sites are unaffected.
    #[must_use]
    pub fn constraint_effective(&self) -> bool {
        self.cases.first().is_some_and(|c| c.is_valid_json())
    }

    /// CF-51: `true` when any case was run more than once (the stability
    /// signal is meaningful).
    fn has_repeats(&self) -> bool {
        self.cases.iter().any(|c| c.repeats > 1)
    }

    /// CF-51: the overall stability rate — the mean of the per-case stability
    /// rates (the fraction of repeated runs that were fully compliant). This
    /// is the data prerequisite for the planned McNemar/Wilson A/B comparator.
    #[must_use]
    pub fn stability_rate(&self) -> f64 {
        if self.cases.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.cases.iter().map(|c| c.stability_rate()).sum();
        sum / self.cases.len() as f64
    }

    /// CF-51: a compact stability label (e.g. `stability 83%`), or empty when
    /// no case was repeated (single-shot → nothing extra to say).
    #[must_use]
    pub fn stability_label(&self) -> String {
        if !self.has_repeats() {
            return String::new();
        }
        format!("stability {:.0}%", self.stability_rate() * 100.0)
    }

    /// One-line summary for the sequence header / headless output.
    ///
    /// CF-17: the compliance part is the *split* `valid JSON … / schema-
    /// correct …` label (a response can parse as JSON yet violate the
    /// schema). CF-36: the penalty is reported only when the constraint was
    /// *effective* (the constrained run produced valid JSON); otherwise it is
    /// flagged "constraint not effective" — and, **R2**, when the
    /// case-0 run was `truncated_by_reasoning` (no content to measure) it is
    /// flagged as such: a truncated case is never penalized as a compliance
    /// failure. CF-51: a `stability …` figure is added when the cases were
    /// repeated. CF-08: the constraint mode is flagged inline so a ~0%
    /// penalty on a backend that drops `response_format` is never misread as
    /// "constrained decoding is free". **R2:** the schema score and the
    /// distinct truncated bucket are broken out (`2/3 schema, 1
    /// truncated_by_reasoning`).
    #[must_use]
    pub fn summary_line(&self) -> String {
        let (c, _p, _f) = self.score();
        let t = self.truncated_count();
        let total = self.cases.len();
        let penalty = if self.constraint_effective() {
            format!("{:+.1}% penalty", self.penalty_pct)
        } else if self
            .cases
            .first()
            .is_some_and(|first| first.truncated_by_reasoning)
        {
            // CF-36 (R2): the speed pair has no content to measure because
            // the constrained case-0 run exhausted its budget in CoT.
            "penalty N/A (case-0 truncated by reasoning)".to_string()
        } else {
            "penalty N/A (constraint not effective)".to_string()
        };
        let mut parts = vec![
            penalty,
            self.compliance_label(),
            format!("{c}/{total} schema, {t} truncated_by_reasoning"),
        ];
        let stability = self.stability_label();
        if !stability.is_empty() {
            parts.push(stability);
        }
        parts.push(format!("[{}]", self.constraint_mode().label()));
        parts.join(" · ")
    }

    /// The `--json` `structured` section (R2; the C3 face of the
    /// traceability pattern the `niah` / `flat_out` sections use): the
    /// score split — compliant / partial / failed plus the distinct
    /// `truncated_by_reasoning` count — the CF-17 compliance label, the
    /// CF-36-gated penalty, the constraint mode, the stability rate, and
    /// the **per-case** detail with each case's `truncated_by_reasoning`
    /// flag. Additive — the Engine-A `--json` shape is untouched.
    #[must_use]
    pub fn to_dict(&self) -> serde_json::Value {
        let (c, p, f) = self.score();
        let t = self.truncated_count();
        let cases: Vec<serde_json::Value> = self
            .cases
            .iter()
            .map(|case| {
                serde_json::json!({
                    "name": case.name,
                    "verdict": case.verdict,
                    "truncated_by_reasoning": case.truncated_by_reasoning,
                    // R3: the server-reported CoT tokens for the case
                    // (`null` when the backend reported no usage block) —
                    // the calibration measurement of a model's reasoning
                    // usage.
                    "reasoning_tokens": case.reasoning_tokens,
                    "valid_json": case.is_valid_json(),
                    "compliant": case.is_compliant(),
                    "repeats": case.repeats,
                    "compliant_repeats": case.compliant_repeats,
                })
            })
            .collect();
        serde_json::json!({
            "score": {
                "compliant": c,
                "partial": p,
                "failed": f,
                "truncated_by_reasoning": t,
                "total": self.cases.len(),
            },
            "compliance_label": self.compliance_label(),
            "score_label": self.score_label(),
            "penalty_pct": self.penalty_pct,
            "constraint_effective": self.constraint_effective(),
            "constraint_mode": self.constraint_mode().label(),
            "stability_rate": self.stability_rate(),
            "free_tps": self.free_tps,
            "constrained_tps": self.constrained_tps,
            "cases": cases,
            "summary": self.summary_line(),
            "verdict": self.verdict_line(),
        })
    }
}

/// F6 (CF-25): one run's TTFT in seconds — `T3 − T1`, or the **`0.0`
/// N/A sentinel** when no token frame ever arrived (a usage-only stream,
/// a failure). Engine A's convention: the total stream time is NOT a
/// TTFT (the old fallback mislabeled total latency as first-token time),
/// and summary consumers filter `ttft > 0.0`.
fn run_ttft(ts: &StreamTimestamps) -> f64 {
    ts.ttft_nanos().map(|n| n as f64 / 1e9).unwrap_or(0.0)
}

/// One `run_once` result: the judged body, the §7 timing metrics, and the
/// hybrid `truncated_by_reasoning` flag (R2).
struct RunOutput {
    /// The content-only accumulated body (the judged span — `Chunk::Content`
    /// deltas only; empty when nothing was committed).
    body: String,
    /// Time to first token, seconds.
    ttft: f64,
    /// Decode speed, tokens/s (the authoritative count over the decode
    /// window).
    tps: f64,
    /// `true` when the stream completed OK but the model spent the entire
    /// budget on chain-of-thought (all-CoT, zero content) — the distinct
    /// `truncated_by_reasoning` outcome (R1 seam:
    /// `StreamOutcome::is_truncated_by_reasoning`).
    truncated_by_reasoning: bool,
    /// The server-reported CoT tokens for this run (R3 calibration
    /// measurement). `None` when the backend reported no usage block.
    reasoning_tokens: Option<u64>,
}

/// The live runner: a free-form baseline plus the three constrained cases.
#[derive(Debug)]
pub struct StructuredEngine {
    client: reqwest::Client,
    url: String,
    model: String,
    api_key: Option<String>,
    timeout: u64,
    /// CF-51: how many times each constrained case is repeated (the stability
    /// rate). Defaults to 1 (single-shot, backward-compatible with the
    /// existing 4-request run); set to [`STABILITY_REPEATS`] (3) for the full
    /// reliability measurement.
    repeats: usize,
    /// Optional HF tokenizer for re-tokenizing output when a proxy omits
    /// `usage` (the research's fallback counting method).
    tokenizer: Option<Arc<crate::prompt::Tokenizer>>,
    /// The per-run generation budget in tokens (R3). The resolved config
    /// value (`--structured-max-tokens` / `CRUCIBLE_STRUCTURED_MAX_TOKENS` /
    /// the config file > the built-in [`STRUCTURED_MAX_GEN_TOKENS`]
    /// hybrid-reasoning safety net).
    max_gen_tokens: u32,
    /// The best-effort `disable_thinking` suppression (R3). On by default
    /// (a speed optimization); `--no-disable-thinking` turns it off so the
    /// model thinks by default — the force-test path for the
    /// `truncated_by_reasoning` safety net.
    disable_thinking: bool,
    /// Optional sequence seam: publish `Run {n}/{total}` to the
    /// [`ProgressBus`] the Benchmark Sequence mirrors into the TUI.
    progress: Option<Arc<ProgressBus>>,
    /// Optional TUI seam: publish live single-stream
    /// [`MetricsState`] snapshots while a run, so the Live Monitor shows
    /// the current run's real stream data.
    metrics: Option<Arc<MetricsState>>,
    /// Optional `Space`-key pause gate: each run waits on it before its
    /// stream is spawned (in-flight runs complete).
    pause: Option<Arc<RunPause>>,
    /// Optional run logger (each run's worker records its HTTP / SSE
    /// lifecycle; the engine records the evaluation start / result).
    logger: Option<Arc<crate::log::RunLogger>>,
}

impl StructuredEngine {
    /// Build from the resolved config (same client policy as Engine A).
    pub fn new(cfg: &Config) -> Result<Self, EngineError> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(cfg.timeout.max(1)))
            .build()?;
        let tokenizer = match &cfg.tokenizer {
            Some(path) => Some(Arc::new(crate::prompt::Tokenizer::from_file(path)?)),
            None => None,
        };
        Ok(Self {
            client,
            url: cfg.url.clone(),
            model: cfg.model.clone(),
            api_key: cfg.api_key.clone(),
            timeout: cfg.timeout,
            repeats: 1,
            tokenizer,
            max_gen_tokens: cfg.structured_max_tokens,
            disable_thinking: cfg.disable_thinking,
            progress: None,
            metrics: None,
            pause: None,
            logger: None,
        })
    }

    /// The configured per-run generation budget in tokens (R3).
    #[must_use]
    pub fn max_gen_tokens(&self) -> u32 {
        self.max_gen_tokens
    }

    /// The configured `disable_thinking` suppression (R3; on by default).
    #[must_use]
    pub fn disable_thinking(&self) -> bool {
        self.disable_thinking
    }

    /// CF-51: set how many times each constrained case is repeated. The
    /// stability rate (the fraction of repeats that are fully compliant) is
    /// then reported per case and overall. Defaults to 1 (single-shot); use
    /// [`STABILITY_REPEATS`] (3) for the full reliability measurement that is
    /// the data prerequisite for the planned McNemar/Wilson A/B comparator.
    ///
    /// The free-form baseline always stays a single shot (it only feeds the
    /// speed pair), so the total request count is `1 + 3 × n`.
    #[must_use]
    pub fn repeats(mut self, n: usize) -> Self {
        self.repeats = n.max(1);
        self
    }

    /// Attach a [`ProgressBus`] so the runs report `Run {n}/{total}` to the
    /// Benchmark Sequence (the TUI's progress bar).
    pub fn progress(mut self, bus: Arc<ProgressBus>) -> Self {
        self.progress = Some(bus);
        self
    }

    /// Attach a [`MetricsState`] publisher so each run pushes live
    /// single-stream snapshots to the TUI (the Live Monitor's stream
    /// matrix / gauges show the current run's real data).
    pub fn metrics(mut self, state: Arc<MetricsState>) -> Self {
        self.metrics = Some(state);
        self
    }

    /// Attach the `Space`-key pause gate: each run waits on it before
    /// its stream is spawned (the headless path leaves it `None`).
    pub fn pause(mut self, gate: Arc<RunPause>) -> Self {
        self.pause = Some(gate);
        self
    }

    /// Attach the run logger (each run's stream then logs its HTTP /
    /// SSE lifecycle).
    pub fn logger(mut self, logger: Arc<crate::log::RunLogger>) -> Self {
        self.logger = Some(logger);
        self
    }

    /// One run: spawn a worker (optionally with the case's `response_format`
    /// constraint), drain its channel, and synthesize the §7 metrics.
    /// Thinking is suppressed on every run (CF-18, best-effort speed) so the
    /// budget mostly goes to the JSON answer, and only `Chunk::Content`
    /// deltas are accumulated into the judged body. The run also reports the
    /// hybrid `truncated_by_reasoning` flag (R2) via [`RunOutput`].
    async fn run_once(&self, case_idx: usize, constrained: bool) -> RunOutput {
        let case = &STRUCTURED_CASES[case_idx];
        let (tx, mut rx) = mpsc::channel(CHANNEL_CAPACITY);
        let mut worker = StreamWorker::new(
            self.client.clone(),
            &self.url,
            &self.model,
            case.prompt,
            self.max_gen_tokens,
        )
        .read_timeout(Duration::from_secs(self.timeout.max(1)))
        .tag(if constrained {
            "C3:constrained"
        } else {
            "C3:free"
        });
        if let Some(key) = &self.api_key {
            worker = worker.api_key(key);
        }
        if let Some(logger) = &self.logger {
            worker = worker.logger(logger.clone());
        }
        // CF-18 (hybrid, R2): suppress the model's chain-of-thought so the
        // budget goes to the JSON answer. This is a **best-effort speed
        // optimization, not a correctness dependency** (R1): a config-only
        // server silently drops the per-request fields and the model thinks
        // by default — the *raised budget* is what then lets it finish its
        // CoT and still commit content, and a stream that exhausts the
        // budget in CoT with zero content is reported as the distinct
        // `truncated_by_reasoning` outcome (not a false "not valid JSON"
        // failure of a structured capability the model has). The suppression
        // is per-backend and graceful (see `StreamWorker::disable_thinking`),
        // so non-reasoning models and backends without a switch are
        // unaffected. Set on the free-form baseline too, so both legs of
        // the penalty measurement judge content on equal footing.
        // `--no-disable-thinking` (R3) turns this off to force-test the
        // safety net: with thinking on and a tiny budget, the model runs
        // out mid-CoT and the run is reported `truncated_by_reasoning`.
        worker = worker.disable_thinking(self.disable_thinking);
        if constrained {
            // CF-08 (activated): the transport encodes the full object, so a
            // constrained run sends the case's strict `json_schema`
            // (`json_schema_response_format`) — real server-side constrained
            // decoding. The `json_object` hint is the flagged fallback for a
            // transport that cannot encode the object; the result records
            // which mode ran (`constraint_mode()`).
            match select_constraint_mode(TRANSPORT_SUPPORTS_JSON_SCHEMA) {
                ConstraintMode::JsonSchema => {
                    worker = worker.response_format_value(json_schema_response_format(
                        case.name,
                        &case_schema(case_idx),
                    ))
                }
                ConstraintMode::JsonObjectHint => {
                    worker = worker.response_format("json_object");
                }
                ConstraintMode::FreeForm => {}
            }
        }

        // Spawn without awaiting: drain the channel concurrently to avoid
        // the >capacity deadlock (server-agnostic fix).
        let handle = spawn_worker(worker, tx);
        // Content-only accumulation (verified, CF-18): the judged body is
        // the model's *visible JSON answer*; `Chunk::Reasoning` deltas are
        // deliberately excluded (they are forwarded to the live snapshot /
        // metrics only). With thinking suppressed above, a reasoning model
        // no longer starves this span.
        let mut body = String::new();
        let mut events = Vec::new();
        let mut batch = 0u32;
        while let Some(event) = rx.recv().await {
            if let StreamEvent::Frame { frame, .. } = &event {
                if !frame.done {
                    if let Chunk::Content(text) = &frame.chunk {
                        body.push_str(text);
                    }
                }
            }
            // Publish a live snapshot every 8 events, and immediately on
            // a terminal event, so the TUI's Live Monitor shows the
            // current run's real stream data (measurement-isolation
            // invariant: the render loop only reads the snapshot).
            let is_terminal = matches!(
                &event,
                StreamEvent::Complete { .. } | StreamEvent::Failed { .. }
            );
            events.push(event);
            batch += 1;
            if let Some(state) = &self.metrics {
                if batch >= 8 || is_terminal {
                    state.update(SpeedEngine::single_stream_snapshot(
                        &self.url,
                        &self.model,
                        "Structured",
                        &events,
                        self.max_gen_tokens,
                        self.tokenizer.as_deref(),
                    ));
                    batch = 0;
                }
            }
        }
        // Final publish (the terminal event may have been batched out).
        if let Some(state) = &self.metrics {
            state.update(SpeedEngine::single_stream_snapshot(
                &self.url,
                &self.model,
                "Structured",
                &events,
                self.max_gen_tokens,
                self.tokenizer.as_deref(),
            ));
        }

        // Channel closed: collect the outcome.
        let outcome = join_worker(handle).await;

        // R3: the server-reported CoT token count for this run (the
        // calibration measurement). `None` when the backend reported no
        // usage block.
        let reasoning_tokens = outcome.usage_details.map(|d| d.reasoning_tokens);

        // R2 hybrid reasoning: a *completed* (OK) stream that spent the
        // whole budget on chain-of-thought — all-CoT, zero content — is the
        // distinct `truncated_by_reasoning` outcome (R1 seam:
        // `StreamOutcome::is_truncated_by_reasoning`, the frame-based wire
        // truth). A transport failure is NOT truncated (the seam's
        // `is_ok()` gate) — it stays a plain run failure.
        let truncated_by_reasoning = outcome.is_truncated_by_reasoning(&events);

        // §7 timing deltas (the worker owns the quanta stamps).
        let ts: StreamTimestamps = outcome.timestamps;
        let stream_time = ts.total_nanos().unwrap_or(0) as f64 / 1e9;
        let ttft = run_ttft(&ts);
        // Authoritative count: the server's `usage.completion_tokens`
        // first; re-tokenize the full output with the model's exact
        // tokenizer when a proxy omits usage; the observed frame count as
        // a last-resort estimate.
        let completion = match outcome
            .usage
            .map(|u| u.completion_tokens)
            .filter(|&n| n > 0)
        {
            Some(n) => n,
            None => match self
                .tokenizer
                .as_deref()
                .and_then(|t| t.try_count(&body))
                .filter(|&n| n > 0)
            {
                Some(n) => n as u64,
                None => outcome.frames.max(1),
            },
        };
        // Decode window: first → last content/reasoning token (prefill /
        // TTFT excluded) — the research's Timing Boundary Rule.
        let (t_first, t_last) = token_window(&events);
        let generation_time = match (t_first, t_last) {
            (Some(f), Some(l)) => (f.delta_nanos(&l) as f64 / 1e9).max(0.001),
            _ => (stream_time - ttft).max(0.001),
        };
        let tps = completion as f64 / generation_time;
        RunOutput {
            body,
            ttft,
            tps,
            truncated_by_reasoning,
            reasoning_tokens,
        }
    }

    /// The full evaluation: a free-form baseline (case 0), then the three
    /// constrained cases. The speed penalty compares case 0 free-form vs
    /// constrained; each case is scored on its own schema.
    ///
    /// CF-51: each constrained case is repeated [`STABILITY_REPEATS`]× so a
    /// single flake no longer silently drops the score; the stability rate
    /// (the fraction of repeats that are fully compliant) is the data
    /// prerequisite for the planned McNemar/Wilson A/B comparator. The
    /// free-form baseline stays a single shot — it only feeds the speed pair.
    ///
    /// While a [`ProgressBus`] is attached, each run publishes
    /// `Run {n}/{total}` — the Benchmark Sequence mirrors it into the TUI's
    /// progress bar.
    pub async fn run(&self) -> StructuredResult {
        let n = self.repeats.max(1);
        // 1 free-form baseline (the speed pair) + N repeats per constrained case.
        let total_runs = 1 + STRUCTURED_CASES.len() * n;
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC3,
                format!(
                    "Engine C3 (Structured) started — 3 schema cases (Simple → Complex) × {n} repeats + free-form baseline"
                ),
            );
        }

        // Run 1 — free-form baseline (case 0 prompt, *no* constraint).
        if let Some(gate) = &self.pause {
            gate.wait_while_paused().await;
        }
        if let Some(bus) = &self.progress {
            bus.publish(EngineProgress::Structured {
                run: 1,
                total: total_runs,
            });
        }
        let free = self.run_once(0, false).await;
        let (free_body, free_ttft, free_tps) = (free.body, free.ttft, free.tps);

        // Constrained cases, each repeated N× (CF-51). Case 0's *first* repeat
        // carries the speed-pair numbers; every repeat is evaluated for the
        // per-case stability rate.
        let mut constrained_tps = 0.0;
        let mut constrained_ttft = 0.0;
        let mut constrained_body = String::new();
        let mut case_evals: Vec<Vec<StructuredCaseResult>> =
            Vec::with_capacity(STRUCTURED_CASES.len());
        let mut run_no = 1usize;
        for (case_idx, case) in STRUCTURED_CASES.iter().enumerate() {
            let mut evals = Vec::with_capacity(n);
            for rep in 0..n {
                run_no += 1;
                if let Some(gate) = &self.pause {
                    gate.wait_while_paused().await;
                }
                if let Some(bus) = &self.progress {
                    bus.publish(EngineProgress::Structured {
                        run: run_no,
                        total: total_runs,
                    });
                }
                let out = self.run_once(case_idx, true).await;
                if case_idx == 0 && rep == 0 {
                    constrained_tps = out.tps;
                    constrained_ttft = out.ttft;
                    constrained_body = out.body.clone();
                }
                // R2: the distinct `truncated_by_reasoning` outcome (all-CoT,
                // zero content on a completed stream) is marked on the case
                // — not folded into the compliance failure buckets. R3: the
                // run's server-reported CoT count rides along for the
                // `--json` per-case detail.
                evals.push(evaluate_case_with(
                    case.name,
                    &out.body,
                    out.truncated_by_reasoning,
                    out.reasoning_tokens,
                ));
            }
            case_evals.push(evals);
        }

        // Fold each case's repeats into its representative (CF-51).
        let cases: Vec<StructuredCaseResult> = case_evals
            .into_iter()
            .map(|evals| {
                let name = evals.first().map(|e| e.name.clone()).unwrap_or_default();
                fold_case_repeats(&name, evals)
            })
            .collect();

        let penalty_pct = if free_tps > 0.0 {
            (free_tps - constrained_tps) / free_tps * 100.0
        } else {
            0.0
        };

        let result = StructuredResult {
            free_tps,
            constrained_tps,
            penalty_pct,
            free_ttft,
            constrained_ttft,
            cases,
            constrained_body,
            free_body,
        };
        if let Some(l) = &self.logger {
            l.info(
                crate::log::Context::EngineC3,
                format!("Engine C3 complete — {}", result.summary_line()),
            );
        }
        result
    }
}

// ── Per-case evaluation ────────────────────────────────────────────────────

/// `true` when `v` is a JSON integer — including *integer-valued floats*
/// (CF-50). Models frequently emit `36.0` for an `integer` field; that parses
/// to a `serde_json` f64, so the old `as_i64()`-only check false-failed a
/// semantically-correct integer and downgraded the case to Partial. A number
/// counts as an integer when it is a native int **or** a finite float with no
/// fractional part (`3.0` → integer; `3.5` → not).
fn is_int(v: &serde_json::Value) -> bool {
    v.as_i64().is_some()
        || v.as_f64()
            .is_some_and(|f| f.is_finite() && f.fract() == 0.0)
}

/// Score one case's constrained output against its checks and classify it.
///
/// Pure over its inputs (unit-testable, no network): the output is parsed
/// (a ` ```json ` fence is stripped first), and each case checks its own
/// schema shape, fields, and types.
pub fn evaluate_case(name: &str, body: &str) -> StructuredCaseResult {
    evaluate_case_with(name, body, false, None)
}

/// `evaluate_case` with the hybrid-reasoning flag (R2) and the
/// server-reported CoT token count (R3).
///
/// `truncated` is `true` when the run's stream **completed OK** but the
/// model spent the entire budget on chain-of-thought — all-CoT, zero
/// content (the R1 seam `StreamOutcome::is_truncated_by_reasoning`). Such a
/// case has no committed span to judge, so it is the distinct
/// [`CaseVerdict::Truncated`] (`truncated_by_reasoning`) outcome: *not* a
/// schema/compliance failure (there is nothing to fail a check against) and
/// *not* a pass (the capability was never demonstrated). A stream that
/// *failed* (transport / HTTP / read) with an empty body is **not**
/// truncated — the seam's `is_ok()` gate keeps it a plain [`Failed`]
/// verdict, and `truncated` must only ever be `true` for a completed
/// all-CoT stream.
///
/// `reasoning_tokens` is the run's server-reported CoT count (R3
/// calibration) — carried on the result for the `--json` per-case detail.
fn evaluate_case_with(
    name: &str,
    body: &str,
    truncated: bool,
    reasoning_tokens: Option<u64>,
) -> StructuredCaseResult {
    if truncated {
        return StructuredCaseResult {
            name: name.to_string(),
            checks: vec![CaseCheck {
                label: "Valid JSON".to_string(),
                passed: false,
            }],
            verdict: CaseVerdict::Truncated,
            output: String::new(),
            repeats: 1,
            compliant_repeats: 0,
            truncated_by_reasoning: true,
            reasoning_tokens,
        };
    }
    let normalized = strip_json_fence(body);
    let value = serde_json::from_str::<serde_json::Value>(normalized).ok();
    let valid = value.is_some();

    let mut checks: Vec<CaseCheck> = vec![CaseCheck {
        label: "Valid JSON".to_string(),
        passed: valid,
    }];

    // Only the non-JSON checks run when the body parsed; otherwise the case
    // is `Failed` and there is nothing further to evaluate.
    if let Some(v) = &value {
        match name {
            "Simple" => {
                let schema = v.is_object();
                let (fields, types) = if let Some(o) = v.as_object() {
                    (
                        o.contains_key("name") && o.contains_key("age"),
                        o.get("name").map(|x| x.is_string()).unwrap_or(false)
                            && o.get("age").map(is_int).unwrap_or(false),
                    )
                } else {
                    (false, false)
                };
                checks.push(CaseCheck {
                    label: "Schema match".into(),
                    passed: schema,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
            }
            "Medium" => {
                let is_array = v.is_array();
                let (len_ok, fields, types) = if let Some(a) = v.as_array() {
                    (
                        a.len() == 3,
                        a.iter().all(|o| {
                            o.is_object()
                                && o.get("id").is_some()
                                && o.get("label").is_some()
                                && o.get("active").is_some()
                        }),
                        a.iter().all(|o| {
                            o.get("id").map(is_int).unwrap_or(false)
                                && o.get("label").map(|x| x.is_string()).unwrap_or(false)
                                && o.get("active").map(|x| x.is_boolean()).unwrap_or(false)
                        }),
                    )
                } else {
                    (false, false, false)
                };
                checks.push(CaseCheck {
                    label: "Array length".into(),
                    passed: len_ok,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
                // The "Schema match" for the array case is carried by the
                // Array-length check; report shape separately.
                checks.insert(
                    1,
                    CaseCheck {
                        label: "Schema match".into(),
                        passed: is_array,
                    },
                );
            }
            "Complex" => {
                let schema = v.is_object();
                let (fields, types) = if let Some(o) = v.as_object() {
                    let user_fields = o
                        .get("user")
                        .map(|u| {
                            u.is_object() && u.get("name").is_some() && u.get("email").is_some()
                        })
                        .unwrap_or(false);
                    let orders_fields = o
                        .get("orders")
                        .and_then(|ord| ord.as_array())
                        .map(|items| {
                            items.iter().all(|it| {
                                it.is_object()
                                    && it.get("id").is_some()
                                    && it.get("total").is_some()
                                    && it.get("items").is_some()
                            })
                        })
                        .unwrap_or(false);
                    let user_types = o
                        .get("user")
                        .map(|u| {
                            u.get("name").map(|x| x.is_string()).unwrap_or(false)
                                && u.get("email").map(|x| x.is_string()).unwrap_or(false)
                        })
                        .unwrap_or(false);
                    let orders_types = o
                        .get("orders")
                        .and_then(|ord| ord.as_array())
                        .map(|items| {
                            items.iter().all(|it| {
                                it.get("id").map(is_int).unwrap_or(false)
                                    && it.get("total").map(|x| x.is_number()).unwrap_or(false)
                                    && it
                                        .get("items")
                                        .and_then(|i| i.as_array())
                                        .map(|arr| arr.iter().all(|s| s.is_string()))
                                        .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                    (user_fields && orders_fields, user_types && orders_types)
                } else {
                    (false, false)
                };
                checks.push(CaseCheck {
                    label: "Schema match".into(),
                    passed: schema,
                });
                checks.push(CaseCheck {
                    label: "All fields".into(),
                    passed: fields,
                });
                checks.push(CaseCheck {
                    label: "Types".into(),
                    passed: types,
                });
            }
            // Unknown case name: only the Valid-JSON check was added.
            _ => {}
        }
    }

    let verdict = if !valid {
        CaseVerdict::Failed
    } else if checks.iter().all(|c| c.passed) {
        CaseVerdict::Compliant
    } else {
        CaseVerdict::Partial
    };

    StructuredCaseResult {
        name: name.to_string(),
        checks,
        verdict,
        output: body.to_string(),
        repeats: 1,
        compliant_repeats: if verdict == CaseVerdict::Compliant {
            1
        } else {
            0
        },
        truncated_by_reasoning: false,
        reasoning_tokens,
    }
}

/// CF-51: fold the N per-repeat evaluations of one case into the single
/// representative [`StructuredCaseResult`] shown in the UI, carrying the
/// stability signal.
///
/// The representative's detail (checks / output) is the *first* repeat's, but
/// its `verdict` is derived from the stability across all repeats so the
/// headline reflects reliability, not a single flake:
/// * all N repeats compliant → `Compliant`;
/// * no repeat compliant, **all** repeats truncated (R2) → `Truncated`
///   (the distinct `truncated_by_reasoning` outcome — a budget
///   exhaustion, not a compliance failure);
/// * no repeat compliant (and not all truncated) → the first repeat's
///   verdict (`Failed` / `Partial`);
/// * a mixed result → `Partial` (unstable — a truncated repeat among
///   compliant ones marks the case, but never reads as a pass).
///
/// A dedicated `Stability (x/N)` check is appended (never inserted first, so
/// [`StructuredCaseResult::is_valid_json`] keeps reading the `Valid JSON`
/// check) and shown only when N > 1.
fn fold_case_repeats(name: &str, evals: Vec<StructuredCaseResult>) -> StructuredCaseResult {
    let repeats = evals.len();
    let compliant_repeats = evals.iter().filter(|e| e.is_compliant()).count();
    let truncated_repeats = evals.iter().filter(|e| e.truncated_by_reasoning).count();
    // R3: the case's CoT usage is the **max** over its repeats (the
    // headroom-relevant worst case); `None` when no repeat's backend
    // reported a usage block.
    let reasoning_tokens = evals.iter().filter_map(|e| e.reasoning_tokens).max();
    let mut rep = match evals.into_iter().next() {
        Some(r) => r,
        None => return evaluate_case(name, ""),
    };
    rep.verdict = if repeats > 0 && compliant_repeats == repeats {
        CaseVerdict::Compliant
    } else if compliant_repeats == 0 && truncated_repeats == repeats {
        // R2: every repeat exhausted the budget in CoT (no content
        // committed) — the distinct `truncated_by_reasoning` outcome.
        CaseVerdict::Truncated
    } else if compliant_repeats == 0 {
        rep.verdict
    } else {
        CaseVerdict::Partial
    };
    if repeats > 1 {
        rep.checks.push(CaseCheck {
            label: format!("Stability ({compliant_repeats}/{repeats})"),
            passed: compliant_repeats == repeats,
        });
    }
    rep.repeats = repeats;
    rep.compliant_repeats = compliant_repeats;
    // R2: mark the case when *any* repeat was truncated (the folded verdict
    // above is `Truncated` only when *all* repeats were).
    rep.truncated_by_reasoning = truncated_repeats > 0;
    // R3: the max-CoT figure (the worst case across the repeats).
    rep.reasoning_tokens = reasoning_tokens;
    rep
}

/// `true` when `body` is a JSON object carrying every `required_fields` key.
///
/// A general-purpose compliance helper (kept public for the headless /
/// export paths and the tests). Normalizes the common model habit of
/// wrapping the object in a ` ```json ` fenced block before parsing.
pub fn is_json_compliant(body: &str, required_fields: &[&str]) -> bool {
    let normalized = strip_json_fence(body);
    serde_json::from_str::<serde_json::Value>(normalized)
        .ok()
        .and_then(|v| v.as_object().cloned())
        .map(|o| required_fields.iter().all(|f| o.contains_key(*f)))
        .unwrap_or(false)
}

/// Strip a single leading/trailing ` ``` ` (json) fence, if present.
fn strip_json_fence(body: &str) -> &str {
    let trimmed = body.trim();
    if let Some(inner) = trimmed.strip_prefix("```") {
        // Skip the optional language tag on the fence line.
        let after_lang = inner.split_once('\n').map(|(l, r)| {
            let lang = l.trim();
            if lang.is_empty() || lang.eq_ignore_ascii_case("json") {
                r
            } else {
                inner
            }
        });
        if let Some(r) = after_lang {
            if let Some(closing) = r.rsplit_once("```") {
                return closing.0.trim_end();
            }
        }
    }
    trimmed
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::timing::MonotonicInstant;

    // ── hybrid reasoning (R2): the budget is the correctness guarantee ──

    #[test]
    fn budget_is_4096_for_reasoning_headroom() {
        // R2 (Option C, `work/plans/REASONING_DESIGN.md` §5.1) + R3
        // calibration: the 4096 generation budget is the *correctness
        // guarantee* — headroom for a verbose reasoner's chain-of-thought
        // before the JSON on a server that ignores our best-effort
        // `disable_thinking` suppression. The JSON body itself is still
        // constrained by the case's strict schema (the largest case needs
        // well under 200 tokens). Sized with large headroom above the
        // measured qwen3.8-27b finetune usage (0 CoT suppressed; 57–64+
        // forced — `work/scratch/r3_calibration.md`).
        assert_eq!(STRUCTURED_MAX_GEN_TOKENS, 4096);
    }

    // ── is_json_compliant (general helper) ───────────────────────────────

    #[test]
    fn compliant_object_passes() {
        let body = r#"{"name": "Ada", "age": 36, "city": "London"}"#;
        assert!(is_json_compliant(body, &["name", "age", "city"]));
    }

    #[test]
    fn missing_field_fails() {
        let body = r#"{"name": "Ada", "age": 36}"#;
        assert!(!is_json_compliant(body, &["name", "age", "city"]));
    }

    #[test]
    fn non_json_fails() {
        assert!(!is_json_compliant(
            "Just some prose, no JSON here.",
            &["name", "age", "city"]
        ));
        assert!(!is_json_compliant("", &["name", "age", "city"]));
        // A JSON array is not a JSON object.
        assert!(!is_json_compliant(
            r#"[{"name": "Ada"}]"#,
            &["name", "age", "city"]
        ));
    }

    #[test]
    fn fenced_json_is_normalized() {
        let body = "```json\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &["name", "age", "city"]));
        // Bare fence without a language tag too.
        let body = "```\n{\"name\": \"Ada\", \"age\": 36, \"city\": \"London\"}\n```";
        assert!(is_json_compliant(body, &["name", "age", "city"]));
    }

    #[test]
    fn strip_json_fence_leaves_plain_text_alone() {
        assert_eq!(strip_json_fence("  hello  "), "hello");
    }

    #[test]
    fn penalty_is_positive_when_constrained_is_slower() {
        // Mirrors the production formula.
        let free: f64 = 100.0;
        let constrained: f64 = 70.0;
        let penalty: f64 = (free - constrained) / free * 100.0;
        assert!((penalty - 30.0).abs() < 1e-9);
    }

    // ── evaluate_case (the per-case schema checks) ───────────────────────

    #[test]
    fn simple_case_compliant() {
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#);
        assert_eq!(r.verdict, CaseVerdict::Compliant);
        assert!(r.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn simple_case_partial_on_bad_type() {
        // `age` is a string, not an integer → Types fails, rest pass.
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": "36"}"#);
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(!r.checks.iter().find(|c| c.label == "Types").unwrap().passed);
    }

    #[test]
    fn integer_valued_float_is_accepted_for_int_fields() {
        // CF-50: models often emit `36.0` for an `integer` field. That is a
        // `serde_json` f64, so the old `as_i64()`-only check false-failed it
        // to Partial. An integer-valued float must count as an integer.
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": 36.0}"#);
        assert_eq!(
            r.verdict,
            CaseVerdict::Compliant,
            "36.0 is an integer-valued float and must be compliant"
        );
        assert!(r.checks.iter().find(|c| c.label == "Types").unwrap().passed);

        // A genuinely fractional value is NOT an integer → still Partial.
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": 36.5}"#);
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(!r.checks.iter().find(|c| c.label == "Types").unwrap().passed);
    }

    #[test]
    fn medium_case_compliant_array_of_three() {
        let r = evaluate_case(
            "Medium",
            r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}, {"id": 3, "label": "three", "active": true}]"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Compliant);
    }

    #[test]
    fn medium_case_partial_on_wrong_length() {
        // Two objects instead of three → Array length fails.
        let r = evaluate_case(
            "Medium",
            r#"[{"id": 1, "label": "one", "active": true}, {"id": 2, "label": "two", "active": false}]"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(
            !r.checks
                .iter()
                .find(|c| c.label == "Array length")
                .unwrap()
                .passed
        );
    }

    #[test]
    fn complex_case_compliant_nested() {
        let r = evaluate_case(
            "Complex",
            r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": [{"id": 1, "total": 50.5, "items": ["a", "b"]}, {"id": 2, "total": 10, "items": []}]}"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Compliant);
        assert!(r.checks.iter().all(|c| c.passed));
    }

    #[test]
    fn complex_case_partial_when_orders_is_text() {
        // `orders` is free text, not an array of objects → schema/types fail.
        let r = evaluate_case(
            "Complex",
            r#"{"user": {"name": "Ada", "email": "ada@x.com"}, "orders": "I have 2 orders totaling $5000"}"#,
        );
        assert_eq!(r.verdict, CaseVerdict::Partial);
        assert!(!r.checks.iter().find(|c| c.label == "Types").unwrap().passed);
    }

    #[test]
    fn non_json_is_failed() {
        let r = evaluate_case("Simple", "this is definitely not json");
        assert_eq!(r.verdict, CaseVerdict::Failed);
        assert!(!r.is_valid_json());
        // Only the Valid-JSON check is present (the rest are unevaluable).
        assert_eq!(r.checks.len(), 1);
    }

    #[test]
    fn fenced_output_is_stripped_before_parsing() {
        let r = evaluate_case("Simple", "```json\n{\"name\": \"Ada\", \"age\": 36}\n```");
        assert_eq!(r.verdict, CaseVerdict::Compliant);
    }

    // ── StructuredResult score / verdict ─────────────────────────────────

    fn result_with(c1: &str, c2: &str, c3: &str) -> StructuredResult {
        StructuredResult {
            free_tps: 65.2,
            constrained_tps: 61.8,
            penalty_pct: 5.2,
            free_ttft: 0.2,
            constrained_ttft: 0.21,
            cases: vec![
                evaluate_case("Simple", c1),
                evaluate_case("Medium", c2),
                evaluate_case("Complex", c3),
            ],
            constrained_body: c1.to_string(),
            free_body: "prose".to_string(),
        }
    }

    #[test]
    fn score_and_verdict_track_compliant_count() {
        // 3/3 compliant.
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        assert_eq!(r.score(), (3, 0, 0));
        assert!(r.fully_compliant());
        assert_eq!(r.verdict_line(), "✓ Fully suitable for API/agent use");

        // 1/3 compliant (simple passes; medium wrong length; complex text).
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true}]"#,
            r#"{"user":{"name":"Ada"},"orders":"free text"}"#,
        );
        assert_eq!(r.score(), (1, 2, 0));
        assert_eq!(
            r.verdict_line(),
            "⚠ Limited use — only trivial key-value extraction"
        );

        // 0/3 (all invalid JSON).
        let r = result_with("nope", "nope", "nope");
        assert_eq!(r.score(), (0, 0, 3));
        assert!(!r.fully_compliant());
        assert!(r.verdict_line().starts_with("✗ Not suitable"));
    }

    #[test]
    fn summary_line_combines_penalty_and_score() {
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        // 2 compliant (simple, complex), 1 partial (medium: 2 not 3) — but all
        // three are *valid* JSON, so the CF-17 split reports them distinctly.
        assert_eq!(r.score(), (2, 1, 0));
        assert!(r.summary_line().contains("+5.2% penalty"));
        assert!(r.summary_line().contains("valid JSON 3/3"));
        assert!(r.summary_line().contains("schema-correct 2/3"));
    }

    #[test]
    fn compliance_rates_split_valid_json_from_schema_correct() {
        // CF-17: a response can parse as JSON yet violate the schema. Here all
        // three cases are *valid* JSON, but only one is *schema-correct*
        // (Medium has 2 elements, not 3; Complex's orders is free text).
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":"free text"}"#,
        );
        let (valid, correct) = r.compliance_rates();
        assert!(
            (valid - 1.0).abs() < 1e-9,
            "all three parse as JSON: {valid}"
        );
        assert!(
            (correct - (1.0 / 3.0)).abs() < 1e-9,
            "only Simple is schema-correct: {correct}"
        );
        assert!(valid > correct, "valid-JSON must exceed schema-correct");
        assert!(r.compliance_label().contains("valid JSON 3/3"));
        assert!(r.compliance_label().contains("schema-correct 1/3"));
    }

    #[test]
    fn penalty_flagged_not_effective_when_constrained_run_not_valid_json() {
        // CF-36: when the constrained case-0 run did NOT produce valid JSON
        // (the backend dropped the schema and the model returned prose, or the
        // run crashed/emptied), the penalty must be flagged "not effective"
        // rather than read as a real constrained-decoding cost.
        let r = result_with(
            "this is not json at all",
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        assert!(
            !r.constraint_effective(),
            "case-0 not valid JSON → constraint not effective"
        );
        assert!(
            r.summary_line().contains("constraint not effective"),
            "summary flags the ineffective constraint: {}",
            r.summary_line()
        );
        assert!(
            !r.summary_line().contains("% penalty"),
            "no numeric penalty when the constraint is not effective"
        );
    }

    #[test]
    fn penalty_reported_when_constrained_run_is_valid_json() {
        // The inverse of CF-36: when case-0 IS valid JSON, the numeric
        // penalty is reported.
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        assert!(r.constraint_effective(), "case-0 valid JSON → effective");
        assert!(r.summary_line().contains("+5.2% penalty"));
    }

    // ── CF-51: repeat-for-stability ────────────────────────────────────

    #[test]
    fn fold_case_repeats_reports_stability_rate() {
        // CF-51: repeat a case N× and report the fraction of repeats that
        // pass. 2 of 3 compliant → stability rate 2/3, verdict Partial
        // (unstable, not a single flake).
        let evals = vec![
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#), // compliant
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#), // compliant
            evaluate_case("Simple", "not json"),                      // failed
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(folded.repeats, 3);
        assert_eq!(folded.compliant_repeats, 2);
        assert!((folded.stability_rate() - (2.0 / 3.0)).abs() < 1e-9);
        assert_eq!(
            folded.verdict,
            CaseVerdict::Partial,
            "a mixed result is unstable (Partial), not Compliant"
        );
        // A dedicated stability check row is present (appended, not first).
        assert!(folded
            .checks
            .iter()
            .any(|c| c.label == "Stability (2/3)" && !c.passed));

        // All-compliant → Compliant, rate 1.0.
        let evals = vec![
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(folded.verdict, CaseVerdict::Compliant);
        assert!((folded.stability_rate() - 1.0).abs() < 1e-9);
        assert!(folded
            .checks
            .iter()
            .any(|c| c.label == "Stability (3/3)" && c.passed));

        // All-failed → keeps the first repeat's Failed verdict, rate 0.0.
        let evals = vec![
            evaluate_case("Simple", "nope"),
            evaluate_case("Simple", "nope"),
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(folded.verdict, CaseVerdict::Failed);
        assert!((folded.stability_rate() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn stability_rate_is_the_mean_of_case_rates() {
        // CF-51: the result-level stability is the mean over the cases.
        // `result_with` uses single-shot `evaluate_case` (repeats = 1):
        // compliant → 1.0, partial/failed → 0.0.
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,           // compliant → 1.0
            r#"[{"id":1,"label":"a","active":true}]"#, // partial → 0.0
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":"free text"}"#, // partial → 0.0
        );
        assert!((r.stability_rate() - (1.0 / 3.0)).abs() < 1e-9);
        // No repeats → the stability label is empty (not surfaced).
        assert!(r.stability_label().is_empty());
        assert!(!r.summary_line().contains("stability"));
    }

    #[test]
    fn summary_line_includes_stability_when_cases_are_repeated() {
        // CF-51: when a case carries N > 1 repeats, the summary line shows
        // the stability figure.
        let mut r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        // Simulate the run() path: give every case 3 repeats, all compliant.
        for c in r.cases.iter_mut() {
            c.repeats = 3;
            c.compliant_repeats = 3;
        }
        assert!(r.has_repeats());
        assert!((r.stability_rate() - 1.0).abs() < 1e-9);
        assert!(r.summary_line().contains("stability 100%"));
    }

    // ── CF-08: real constrained decoding (mode selection + flagging) ─────

    #[test]
    fn select_constraint_mode_prefers_schema_when_supported() {
        // When the transport can encode a `json_schema` object, C3 requests
        // real constrained decoding.
        assert_eq!(select_constraint_mode(true), ConstraintMode::JsonSchema);
    }

    #[test]
    fn select_constraint_mode_falls_back_to_hint_when_unsupported() {
        // When the transport can only send a `{"type": …}` string, C3
        // gracefully falls back to the `json_object` hint — and flags it.
        assert_eq!(
            select_constraint_mode(false),
            ConstraintMode::JsonObjectHint
        );
    }

    #[test]
    fn json_schema_response_format_is_the_openai_object() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {
                "name": { "type": "string" },
                "age": { "type": "integer" }
            },
            "required": ["name", "age"]
        });
        let rf = json_schema_response_format("simple", &schema);
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "simple");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(
            rf["json_schema"]["schema"]["properties"]["age"]["type"],
            "integer"
        );
        assert_eq!(
            rf["json_schema"]["schema"]["required"],
            serde_json::json!(["name", "age"])
        );
    }

    #[test]
    fn result_flags_the_constraint_mode_it_used() {
        // The result exposes the mode it actually ran under (the `json_schema`
        // object — constrained decoding — since the transport encodes it).
        let r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true}]"#,
            r#"{"user":{"name":"Ada"},"orders":"free text"}"#,
        );
        assert_eq!(r.constraint_mode(), ConstraintMode::JsonSchema);
        // And the one-line summary carries the flag (no more silent
        // "constrained decoding" claim).
        assert!(
            r.summary_line()
                .contains("json_schema (constrained decoding)"),
            "summary_line flags the mode: {}",
            r.summary_line()
        );
    }

    #[test]
    fn case_schemas_are_strict_and_match_the_cases() {
        // `strict: true` requires `additionalProperties: false` on every
        // object; a schema-capable backend (vLLM XGrammar) rejects
        // non-strict shapes, so each per-case schema must be strict.
        let s0 = case_schema(0);
        assert_eq!(s0["type"], "object");
        assert_eq!(s0["additionalProperties"], false);
        assert_eq!(s0["properties"]["age"]["type"], "integer");

        let s1 = case_schema(1);
        assert_eq!(s1["type"], "array");
        assert_eq!(s1["minItems"], 3);
        assert_eq!(s1["maxItems"], 3);
        assert_eq!(s1["items"]["additionalProperties"], false);

        let s2 = case_schema(2);
        assert_eq!(s2["type"], "object");
        assert_eq!(s2["additionalProperties"], false);
        assert_eq!(s2["properties"]["user"]["additionalProperties"], false);
        assert_eq!(
            s2["properties"]["orders"]["items"]["additionalProperties"],
            false
        );
    }

    #[test]
    fn constrained_run_sends_the_case_json_schema_object() {
        // The activated handoff: a constrained run transmits the full
        // `json_schema` object (name + strict schema) verbatim, not the
        // `{"type": <string>}` hint.
        let rf = json_schema_response_format(STRUCTURED_CASES[0].name, &case_schema(0));
        assert_eq!(rf["type"], "json_schema");
        assert_eq!(rf["json_schema"]["name"], "Simple");
        assert_eq!(rf["json_schema"]["strict"], true);
        assert_eq!(rf["json_schema"]["schema"]["type"], "object");
        assert_eq!(rf["json_schema"]["schema"]["additionalProperties"], false);
    }

    #[test]
    fn constraint_mode_labels_are_distinct() {
        assert_ne!(
            ConstraintMode::JsonObjectHint.label(),
            ConstraintMode::JsonSchema.label()
        );
        assert_eq!(ConstraintMode::FreeForm.label(), "free-form");
    }

    // ── R2 hybrid reasoning: `truncated_by_reasoning` ───────────────────

    /// One `StreamEvent::Frame` carrying a single `chunk` — the wire-truth
    /// fixture for the frame-based `truncated_by_reasoning` detection
    /// (mirrors the R1 seam tests in `client/stream.rs`).
    fn frame_event(chunk: Chunk) -> StreamEvent {
        StreamEvent::Frame {
            frame: crate::sse::ParsedFrame {
                chunk,
                t_nanos: 0,
                done: false,
            },
            at: crate::timing::MonotonicInstant::now(),
            timestamps: StreamTimestamps::default(),
        }
    }

    /// The `StreamOutcome` shape `run_once` hands to
    /// `is_truncated_by_reasoning` after the drain loop.
    fn outcome_with(
        frames: u64,
        error: Option<crate::client::StreamError>,
    ) -> crate::client::StreamOutcome {
        crate::client::StreamOutcome {
            timestamps: StreamTimestamps::default(),
            usage: None,
            usage_details: None,
            premature: false,
            malformed_frames: 0,
            error,
            looping: false,
            loop_excluded_tokens: 0,
            frames,
        }
    }

    #[test]
    fn all_cot_zero_content_ok_stream_is_truncated_not_failed() {
        // (a) A *completed* stream (OK) that spent the whole budget on
        // chain-of-thought — all-CoT, zero content — marks the case
        // `Truncated`: not non-compliant (no content to fail a schema
        // check against) and not penalized as a compliance failure.
        let events = vec![
            frame_event(Chunk::Reasoning("let me think…".into())),
            frame_event(Chunk::Reasoning("…".into())),
        ];
        let outcome = outcome_with(2, None);
        assert!(outcome.is_ok());
        assert!(
            outcome.is_truncated_by_reasoning(&events),
            "reasoning>0 && content==0 && is_ok ⇒ truncated (R1 seam)"
        );
        let r = evaluate_case_with("Simple", "", true, None);
        assert_eq!(
            r.verdict,
            CaseVerdict::Truncated,
            "the distinct outcome — not Failed, not Partial"
        );
        assert!(r.truncated_by_reasoning);
        assert!(!r.is_compliant());
        assert!(!r.is_valid_json());
    }

    #[test]
    fn content_present_bad_schema_is_non_compliant_not_truncated() {
        // (b) The other half of the three-way split: content was committed
        // but violates the schema → non-compliant (Partial). A content
        // frame on the wire means the R1 seam reports *not* truncated.
        let events = vec![
            frame_event(Chunk::Reasoning("thinking…".into())),
            frame_event(Chunk::Content(
                r#"{"name": "Ada", "age": "thirty-six"}"#.into(),
            )),
        ];
        let outcome = outcome_with(2, None);
        assert!(
            !outcome.is_truncated_by_reasoning(&events),
            "content>0 ⇒ not truncated"
        );
        let r = evaluate_case_with(
            "Simple",
            r#"{"name": "Ada", "age": "thirty-six"}"#,
            false,
            None,
        );
        assert_eq!(
            r.verdict,
            CaseVerdict::Partial,
            "content present but schema-wrong ⇒ non-compliant"
        );
        assert!(!r.truncated_by_reasoning);
    }

    #[test]
    fn compliant_case_passes_and_is_not_truncated() {
        // (c) A case with committed, schema-correct content → Compliant,
        // never marked truncated.
        let r = evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, None);
        assert_eq!(r.verdict, CaseVerdict::Compliant);
        assert!(!r.truncated_by_reasoning);
        assert!(r.is_compliant());
    }

    #[test]
    fn transport_failure_with_empty_body_is_failed_not_truncated() {
        // The `is_ok()` gate: a stream that *failed* (connection refused)
        // after emitting only CoT is a genuine failure — the R1 seam
        // reports it as NOT truncated, and the case scores the plain
        // `Failed` verdict (empty body), never `Truncated`.
        let events = vec![frame_event(Chunk::Reasoning("thinking…".into()))];
        let outcome = outcome_with(
            1,
            Some(crate::client::StreamError::Connection("refused".into())),
        );
        assert!(!outcome.is_ok());
        assert!(
            !outcome.is_truncated_by_reasoning(&events),
            "an error is a fail, not a truncation"
        );
        let r = evaluate_case_with("Simple", "", false, None);
        assert_eq!(r.verdict, CaseVerdict::Failed);
        assert!(!r.truncated_by_reasoning);
    }

    #[test]
    fn score_counts_truncated_as_its_own_bucket() {
        // R2: a truncated case sits in its own bucket — out of
        // compliant/partial/failed (the `score()` three-tuple) and out of
        // every compliance penalty.
        let mut r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            "nope",
        );
        r.cases[2] = evaluate_case_with("Complex", "", true, None);
        assert_eq!(
            r.score(),
            (2, 0, 0),
            "truncated is not in the three compliance buckets"
        );
        assert_eq!(r.truncated_count(), 1);
        assert!(!r.fully_compliant());
    }

    #[test]
    fn summary_line_breaks_out_truncated_cases() {
        // R2 terminal surfacing: the C3 summary breaks the truncated cases
        // out of the schema score (`2/3 schema, 1 truncated_by_reasoning`),
        // alongside the intact CF-17 split.
        let mut r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            "nope",
        );
        r.cases[2] = evaluate_case_with("Complex", "", true, None);
        let s = r.summary_line();
        assert!(
            s.contains("2/3 schema, 1 truncated_by_reasoning"),
            "the summary breaks out the truncated case: {s}"
        );
        assert!(s.contains("valid JSON 2/3"), "CF-17 split intact: {s}");
        assert!(s.contains("schema-correct 2/3"), "CF-17 split intact: {s}");
    }

    #[test]
    fn verdict_line_all_truncated_is_inconclusive_not_a_failure() {
        // R2: every case exhausted the budget in CoT (no content
        // committed) → the practical verdict is *inconclusive (budget)*,
        // not "not suitable" — and the penalty is N/A: a truncated case is
        // never penalized as a compliance failure (CF-36, R2 face).
        let r = StructuredResult {
            free_tps: 65.2,
            constrained_tps: 61.8,
            penalty_pct: 5.2,
            free_ttft: 0.2,
            constrained_ttft: 0.21,
            cases: vec![
                evaluate_case_with("Simple", "", true, None),
                evaluate_case_with("Medium", "", true, None),
                evaluate_case_with("Complex", "", true, None),
            ],
            constrained_body: String::new(),
            free_body: String::new(),
        };
        let v = r.verdict_line();
        assert!(
            v.starts_with("⚠"),
            "inconclusive (⚠), not a failure (✗): {v}"
        );
        assert!(v.contains("truncated by reasoning"), "{v}");
        assert!(!v.starts_with('✗'));
        let s = r.summary_line();
        assert!(
            !s.contains("% penalty"),
            "no numeric penalty for an all-truncated run: {s}"
        );
        assert!(
            s.contains("penalty N/A (case-0 truncated by reasoning)"),
            "{s}"
        );
        assert!(s.contains("0/3 schema, 3 truncated_by_reasoning"), "{s}");
    }

    #[test]
    fn penalty_label_names_truncation_when_case_zero_truncated() {
        // CF-36 (R2 face): a truncated case-0 has no content to measure —
        // the penalty is N/A for a *distinct* reason (budget exhaustion in
        // CoT), not "constraint not effective".
        let mut r = result_with(
            "this is not json at all",
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            r#"{"user":{"name":"Ada","email":"e@x"},"orders":[{"id":1,"total":1.0,"items":["a"]}]}"#,
        );
        r.cases[0] = evaluate_case_with("Simple", "", true, None);
        assert!(!r.constraint_effective(), "no content ⇒ not effective");
        let s = r.summary_line();
        assert!(
            s.contains("penalty N/A (case-0 truncated by reasoning)"),
            "the distinct truncation reason is named: {s}"
        );
        assert!(!s.contains("constraint not effective"), "{s}");
    }

    #[test]
    fn fold_case_repeats_all_truncated_is_truncated_verdict() {
        // R2: every repeat all-CoT / zero content → the folded case keeps
        // the distinct `Truncated` verdict (not `Failed`), flagged
        // `truncated_by_reasoning`, and never reads as a pass.
        let evals = vec![
            evaluate_case_with("Simple", "", true, None),
            evaluate_case_with("Simple", "", true, None),
            evaluate_case_with("Simple", "", true, None),
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(
            folded.verdict,
            CaseVerdict::Truncated,
            "all repeats truncated ⇒ the distinct outcome"
        );
        assert!(folded.truncated_by_reasoning);
        assert!(!folded.is_compliant());
        assert_eq!(folded.compliant_repeats, 0);
    }

    #[test]
    fn fold_case_repeats_mixed_truncated_is_partial_and_flagged() {
        // R2: a truncated repeat among compliant ones is *unstable*
        // (Partial) and the case carries the truncation flag — but it never
        // reads as a compliance failure.
        let evals = vec![
            evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, None),
            evaluate_case_with("Simple", "", true, None),
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(
            folded.verdict,
            CaseVerdict::Partial,
            "a mixed result is unstable (Partial) — not a pass, not a fail"
        );
        assert!(
            folded.truncated_by_reasoning,
            "the flag marks that a repeat was truncated"
        );
        assert_eq!(folded.compliant_repeats, 1);
    }

    #[test]
    fn to_dict_carries_the_truncated_flag_for_json() {
        // R2 `--json` surfacing: the `structured` section carries the score
        // split (with the distinct `truncated_by_reasoning` count) and the
        // per-case `truncated_by_reasoning` flag.
        let mut r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            "nope",
        );
        r.cases[2] = evaluate_case_with("Complex", "", true, None);
        let d = r.to_dict();
        assert_eq!(d["score"]["compliant"], 2);
        assert_eq!(d["score"]["failed"], 0);
        assert_eq!(d["score"]["truncated_by_reasoning"], 1);
        assert_eq!(d["score"]["total"], 3);
        // Per-case flags: the truncated case carries it, the others don't.
        assert_eq!(d["cases"][2]["truncated_by_reasoning"], true);
        assert_eq!(d["cases"][2]["verdict"], "truncated");
        assert_eq!(d["cases"][2]["valid_json"], false);
        assert_eq!(d["cases"][0]["truncated_by_reasoning"], false);
        assert_eq!(d["cases"][0]["verdict"], "compliant");
        assert!(d["summary"]
            .as_str()
            .unwrap()
            .contains("truncated_by_reasoning"));
    }

    // ── R3: the configurable generation budget + CoT measurement ────────

    #[test]
    fn engine_honors_the_configured_generation_budget() {
        // The resolved config budget (R3) reaches the engine and is used
        // for the per-run worker / snapshots (not the built-in default).
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            structured_max_tokens: 1024,
            ..Config::default()
        };
        let engine = StructuredEngine::new(&cfg).unwrap();
        assert_eq!(engine.max_gen_tokens(), 1024);
        // The default config carries the built-in safety-net budget.
        let engine = StructuredEngine::new(&Config::default()).unwrap();
        assert_eq!(engine.max_gen_tokens(), STRUCTURED_MAX_GEN_TOKENS);
    }

    #[test]
    fn evaluate_case_carries_the_reasoning_tokens() {
        // R3: the server-reported CoT count rides on the case result (the
        // `--json` per-case detail), for both the truncated and the
        // content-present paths.
        let r = evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, Some(12));
        assert_eq!(r.reasoning_tokens, Some(12));
        let t = evaluate_case_with("Simple", "", true, Some(64));
        assert_eq!(t.reasoning_tokens, Some(64));
        // No usage block reported → `None` (honest "not reported").
        let r = evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#);
        assert_eq!(r.reasoning_tokens, None);
    }

    #[test]
    fn fold_case_repeats_takes_the_max_reasoning_tokens() {
        // R3: the folded case reports the worst-case CoT usage across its
        // repeats (the headroom-relevant number).
        let evals = vec![
            evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, Some(10)),
            evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, Some(25)),
            evaluate_case_with("Simple", "", true, Some(64)),
        ];
        let folded = fold_case_repeats("Simple", evals);
        assert_eq!(folded.reasoning_tokens, Some(64));
        // A `None` among `Some`s is ignored; all-`None` stays `None`.
        let evals = vec![
            evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, None),
            evaluate_case_with("Simple", r#"{"name": "Ada", "age": 36}"#, false, Some(7)),
        ];
        assert_eq!(fold_case_repeats("Simple", evals).reasoning_tokens, Some(7));
        let evals = vec![
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
            evaluate_case("Simple", r#"{"name": "Ada", "age": 36}"#),
        ];
        assert_eq!(fold_case_repeats("Simple", evals).reasoning_tokens, None);
    }

    #[test]
    fn engine_honors_the_disable_thinking_switch() {
        // R3: the resolved suppression switch reaches the engine (on by
        // default; `--no-disable-thinking` → off, the force-test path).
        let cfg = Config {
            url: "http://127.0.0.1:9".to_string(),
            disable_thinking: false,
            ..Config::default()
        };
        let engine = StructuredEngine::new(&cfg).unwrap();
        assert!(!engine.disable_thinking());
        // The default config keeps suppression on.
        let engine = StructuredEngine::new(&Config::default()).unwrap();
        assert!(engine.disable_thinking());
    }

    #[test]
    fn to_dict_carries_the_per_case_reasoning_tokens() {
        // R3: the `--json` structured section carries each case's
        // server-reported CoT tokens (`null` when the backend reported no
        // usage block) — the calibration measurement.
        let mut r = result_with(
            r#"{"name": "Ada", "age": 36}"#,
            r#"[{"id":1,"label":"a","active":true},{"id":2,"label":"b","active":false},{"id":3,"label":"c","active":true}]"#,
            "nope",
        );
        r.cases[0].reasoning_tokens = Some(14);
        let d = r.to_dict();
        assert_eq!(d["cases"][0]["reasoning_tokens"], 14);
        assert_eq!(d["cases"][1]["reasoning_tokens"], serde_json::Value::Null);
        assert_eq!(d["cases"][2]["reasoning_tokens"], serde_json::Value::Null);
    }

    // ── F6: the no-frame TTFT is the N/A sentinel, not total time ────────

    #[test]
    fn f6_no_frame_stream_ttft_is_the_na_sentinel() {
        // A run that never delivered a token frame (a usage-only /
        // failed stream): TTFT is the `0.0` N/A sentinel — NOT the total
        // stream time (the CF-25 anti-pattern Engine A explicitly fixed).
        let t0 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(5));
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: None,
            t3: None,
            t_end: Some(MonotonicInstant::now()),
        };
        assert!(
            ts.total_nanos().unwrap() > 0,
            "the stream did take real total time"
        );
        assert_eq!(
            run_ttft(&ts),
            0.0,
            "no token frame → the 0.0 N/A sentinel, not the total"
        );
        // A summary that filters `ttft > 0.0` (Engine A's convention)
        // excludes it.
        let ttfts = [run_ttft(&ts), 0.3];
        let filtered: Vec<f64> = ttfts.iter().copied().filter(|t| *t > 0.0).collect();
        assert_eq!(filtered, vec![0.3], "the N/A sentinel is filtered out");
    }

    #[test]
    fn f6_token_frame_stream_ttft_is_t3_minus_t1() {
        // The measured case is unchanged: a run with token frames
        // reports `T3 − T1`.
        let t0 = MonotonicInstant::now();
        std::thread::sleep(Duration::from_millis(10));
        let t3 = MonotonicInstant::now();
        let ts = StreamTimestamps {
            t0: Some(t0),
            t1: Some(t0),
            t2: Some(t0),
            t3: Some(t3),
            t_end: Some(MonotonicInstant::now()),
        };
        let expected = t0.delta_nanos(&t3) as f64 / 1e9;
        let got = run_ttft(&ts);
        assert!((got - expected).abs() < 1e-9, "{got} vs {expected}");
        assert!(got > 0.0);
    }
}
