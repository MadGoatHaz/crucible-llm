# Contributing to Crucible LLM

## Development Setup

```bash
# Prerequisites: Rust stable, cargo
git clone https://github.com/YOURUSERNAME/crucible-llm.git
cd crucible-llm
cargo build
cargo test
```

## Project Structure

```
src/
├── lib.rs            # Library crate root (modules shared by the binary & tests)
├── main.rs           # Entry point, CLI parsing, TUI/headless dispatch
├── config.rs         # Configuration (CLI > env > file > defaults)
├── timing.rs         # High-resolution timing (quanta)
├── log.rs            # File-based run logger (latest.log + run archives)
├── client/
│   ├── mod.rs
│   ├── stream.rs     # StreamWorker (HTTP + SSE + timing)
│   ├── pool.rs       # WorkerPool (concurrent streams)
│   └── models.rs     # Model discovery (GET /v1/models)
├── sse/
│   ├── mod.rs
│   ├── parser.rs     # SSE stream parser
│   └── chunk.rs      # SSE chunk types
├── prompt/
│   ├── mod.rs
│   ├── generator.rs  # Prompt generation (short/long/nocache)
│   └── tokenizer.rs  # Token counting (HF or chars/4)
├── engines/
│   ├── mod.rs
│   ├── sequence.rs   # Benchmark sequence executor
│   ├── speed.rs      # Engine A
│   ├── concurrency.rs # Engine B
│   ├── hardware.rs   # Engine D
│   └── capability/
│       ├── mod.rs
│       ├── niah.rs       # Engine C1
│       ├── reasoning.rs  # Engine C2
│       └── structured.rs # Engine C3
├── metrics/
│   ├── mod.rs
│   ├── state.rs      # ArcSwap shared metrics
│   └── histogram.rs  # HdrHistogram (ITL distribution)
├── hw/
│   ├── mod.rs        # HwPoller trait
│   └── nvml.rs       # NVIDIA backend
├── storage/
│   ├── mod.rs
│   ├── db.rs         # SQLite persistence
│   ├── models.rs     # Storage data models
│   ├── schema.sql    # Database schema
│   └── export.rs     # JSON/MD/CSV export
└── ui/
    ├── mod.rs
    ├── app.rs        # App state machine
    ├── event.rs      # Event loop (crossterm)
    ├── theme.rs      # Colors and styles
    └── views/
        ├── mod.rs
        ├── setup.rs      # Setup/connection phase
        ├── live.rs       # View 1: Live monitor
        ├── concurrency.rs # View 2: Concurrency
        ├── needle.rs     # View 3: NIAH
        ├── history.rs    # View 4: History
        └── config.rs     # View 5: Config
```

## Coding Standards

- `cargo clippy --all-targets` must be clean
- `cargo fmt --check` must pass
- All public functions need doc comments
- Tests required for new functionality
- No `unwrap()` in production code (use `?` or handle errors)

## Testing

```bash
cargo test              # All tests
cargo test --release    # Release mode
cargo test serial       # Run specific test
```

The test suite includes:
- Unit tests (in-module `#[cfg(test)]`)
- Integration tests (`tests/` directory)
- Mock SSE server for network tests (no real server needed)

## Submitting Changes

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/descriptive-name`
3. Make your changes
4. Ensure `cargo build && cargo test && cargo clippy --all-targets && cargo fmt --check` all pass
5. Submit a pull request

## License

By contributing, you agree that your contributions will be licensed under GNU GPL v3.
