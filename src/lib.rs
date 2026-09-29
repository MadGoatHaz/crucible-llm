//! Crucible-LLM crate root.
//!
//! Declares the module tree. Chunk 1 (scaffold) creates each module as a stub;
//! subsequent chunks fill them in. See `work/plans/PLAN.md` for the chunk map.

pub mod client;
pub mod config;
pub mod engines;
pub mod hw;
pub mod metrics;
pub mod prompt;
pub mod sse;
pub mod storage;
pub mod timing;
pub mod ui;
