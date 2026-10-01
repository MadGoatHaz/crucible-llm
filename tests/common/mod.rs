//! Shared integration-test harness: the in-process mock SSE server
//! primitives and the `ratatui::TestBackend` render helpers that every
//! test file used to re-implement locally.
//!
//! Each test binary compiles this module independently and uses only the
//! subset it needs — the rest is intentionally allowed to be dead.
#![allow(dead_code)]

pub mod mock;
pub mod render;
