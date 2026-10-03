# Changelog

All notable changes to Crucible LLM.

## [0.1.1] - 2026-10-03

### Added
- **3 Visual Themes**: Cyberpunk (cyan/purple/neon), Vampire (crimson/gold/gothic), Monochrome Pastel (soft/clean). First-run theme picker with live preview. Switchable from Config (tab 5).
- **Flat Out Engine (F)**: 60-second real-world maximum throughput test. Runs at your server's concurrency sweet-spot for full-load measurement. The "one number" to quote when comparing setups.
- **Authoritative Token Counting**: Uses server-reported `usage.completion_tokens` as primary source. Re-tokenization fallback. Accurate across all backends (vLLM batches ~2.4 tokens/frame — we count actual tokens, not frames).
- **Multi-Backend Support**: Verified working with vLLM, llama.cpp, LM Studio, Unsloth Desktop, SGLang, Ollama, TGI. Any OpenAI-compatible `/v1/chat/completions` endpoint.
- **Practical Sweet Spot (30 t/s)**: Concurrency engine now identifies the highest user count where each still gets ≥30 t/s (±2% margin). The recommended deployment number.
- **Methodology Transparency**: Every JSON export includes a `methodology` block with exact formulas, timing resolution, and counting method.
- **Decode Loop Guard**: Detects repeating token patterns. Looping streams excluded from metrics.
- **Labeled Metric Layers**: Prefill / Decode / E2E reported separately. Never blended.
- **2D Concurrency × Context Matrix**: Cross-product scaling surface.
- **History View**: Full list/detail/compare/delete with statistical deltas.
- **Config Persistence**: Settings saved to `~/.config/crucible/config.json`. Setup walkthrough only on first run.
- **Run Logging**: Timestamped logs at `~/.local/share/crucible/logs/latest.log` for diagnostics.
- **Structured Event Log**: Filterable engine/HTTP/SSE events.

### Changed
- **TUI is now the default** (no `--tui` flag needed). Use `--headless` for CLI mode.
- **Sweet spot threshold**: Changed from 40 t/s to 30 t/s per-user (more practical for home setups)
- **Default ladder**: `1,2,3,4,8,12,16,24,32` (more granular at low end)
- **Default tokens**: 10,000 (realistic workload)
- **Default engines**: A, B, C1, C2, C3, F (Energy D is opt-in)
- **Ctrl+C no longer quits** (use `q` with confirmation). Terminal copy works normally.
- **Token counting**: Now uses server-reported usage (was frame-based, caused 3x undercount with vLLM)
- **Live throughput graph**: Now uses same calculation as Overall Metrics (converges to true rate)

### Fixed
- **vLLM token undercount**: Server batches ~2.4 tokens/SSE frame. We now count actual tokens, not frames.
- **Non-vLLM compatibility**: Deadlock fixed for llama.cpp, LM Studio, Unsloth (channel buffer overflow with high-frame-count servers)
- **Tokio IO**: Enabled for TUI runtime (model discovery HTTP)
- **Layout collisions**: Panel border overlaps fixed
- **Config view**: Enter-to-edit gate, number keys type in edit mode, Esc saves+stays
- **Metrics freeze**: All values lock on sequence completion
- **Stall detection**: Phase-aware (90s TTFB, 30s inter-token)

### Architecture
- Single 14MB static Rust binary. Zero runtime dependencies.
- `quanta` ns-resolution timing. `ArcSwap` lock-free metrics. Tokio async.
- Works with any OpenAI-compatible streaming endpoint.
- GPL-3.0 licensed.
