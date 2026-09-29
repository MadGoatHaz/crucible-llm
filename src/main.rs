//! Crucible-LLM entry point.
//!
//! Dispatch:
//! - `crucible-llm`      → banner (headless CLI parity arrives in Chunk 7)
//! - `crucible-llm --tui` → the ratatui dashboard (TUI layout & event loop)

use std::env;
use std::panic;
use std::process;

use crucible_llm::ui::app::App;
use crucible_llm::ui::event::EventLoop;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    let tui = args.iter().any(|a| a == "--tui" || a == "tui");

    if tui {
        let ok = run_tui();
        EventLoop::restore(); // best-effort terminal restore on all paths
        if !ok {
            process::exit(1);
        }
        return;
    }

    let version = env!("CARGO_PKG_VERSION");
    println!("============================================================");
    println!("  Crucible-LLM v{version}");
    println!("  High-performance terminal LLM benchmark & inference profiler");
    println!("  (scaffold ready — engines arrive in later chunks)");
    println!("============================================================");
    println!();
    println!("  modes:");
    println!("    --tui   open the interactive ratatui dashboard");
}

/// Run the TUI on a dedicated single-thread runtime, guaranteeing the
/// terminal is restored even on panic.
///
/// The event loop is async (60Hz tokio tick) but the app itself is a
/// foreground terminal program, so a current-thread runtime is all it
/// needs; the worker pool will run on its own multi-thread runtime in
/// later chunks.
fn run_tui() -> bool {
    let result = panic::catch_unwind(panic::AssertUnwindSafe(|| {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .expect("failed to build tokio runtime");
        runtime.block_on(async {
            let mut event_loop = EventLoop::new()?;
            let mut app = App::new();
            event_loop.run(&mut app).await?;
            event_loop.teardown()
        })
    }));
    match result {
        Ok(Ok(())) => true,
        Ok(Err(e)) => {
            eprintln!("crucible-llm: TUI error: {e}");
            false
        }
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .unwrap_or_else(|| "unknown panic".to_string());
            eprintln!("crucible-llm: TUI panicked: {msg}");
            false
        }
    }
}
