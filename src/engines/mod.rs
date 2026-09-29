//! The four benchmark engines: A (speed), B (concurrency), C (capability),
//! and D (hardware).
//!
//! * [`speed`] — Engine A (Chunk 7): single-stream TTFT / PP / TG / MTP
//!   orchestration; the headless result box / `--json` report / summary
//!   (parity with `llmspeedtest.py`).
//! * [`concurrency`] — Engine B (Chunk 10 + 11): the multi-stream ladder
//!   sweep (`1→2→4→8→16→32→64`), collecting per-level aggregate tokens/sec
//!   and client-perceived p90 TPOT across all concurrent streams, plus
//!   saturation knee-point detection and the "Optimal Operational
//!   Envelope" (the recommended sweet spot).

pub mod capability;
pub mod concurrency;
pub mod hardware;
pub mod speed;

pub use concurrency::{
    normalize_ladder, Envelope, KneePoint, Sweep, SweepLevel, SweepResult, DEFAULT_LADDER,
    KNEE_GAIN_THRESHOLD, KNEE_SPIKE_THRESHOLD,
};
pub use speed::{
    aggregate, all_failed, format_result_box, format_summary, json_report, EngineError,
    SpeedEngine, SpeedResult, MAX_GEN_TOKENS,
};
