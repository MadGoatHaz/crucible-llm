//! Crucible-LLM crate root.
//!
//! Declares the module tree. Chunk 1 (scaffold) creates each module as a stub;
//! subsequent chunks fill them in. See `work/plans/PLAN.md` for the chunk map.

pub mod client;
pub mod clipboard;
pub mod config;
pub mod engines;
pub mod headless;
pub mod hw;
pub mod log;
pub mod metrics;
pub mod prompt;
pub mod report;
pub mod sse;
pub mod storage;
pub mod timing;
pub mod ui;

// CF-42: the headless orchestration is a public lib function so it is
// testable and reusable — `main.rs` is a thin caller. `crucible_llm::
// run_headless` (and `build_energy_json`, which carries the P3-A
// `cpu_power_estimated` flag into the `--json` energy object) are the
// crate-root entry points.
pub use headless::{build_energy_json, run_headless};
