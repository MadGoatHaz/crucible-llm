//! Crucible-LLM entry point.
//!
//! Chunk 1 (scaffold): prints the banner and version, then exits 0.
//! Later chunks wire up clap-based CLI parsing and headless/TUI mode dispatch.

fn main() {
    let version = env!("CARGO_PKG_VERSION");
    println!("============================================================");
    println!("  Crucible-LLM v{version}");
    println!("  High-performance terminal LLM benchmark & inference profiler");
    println!("  (scaffold ready — engines arrive in later chunks)");
    println!("============================================================");
}
