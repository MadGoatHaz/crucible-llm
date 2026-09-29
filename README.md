# Crucible-LLM

A native-Rust, zero-runtime-dependency terminal application that benchmarks
OpenAI-compatible streaming LLM inference endpoints (vLLM, llama.cpp, SGLang,
Ollama). It measures not only raw token speed, but how efficiently an engine
handles compute saturation, speculative decoding, and complex task fidelity.

## Engines

- **A — Speed & Latency:** microsecond TTFT, isolated prefill/decode throughput,
  MTP/speculative-decoding ratio, warm/cold KV-cache detection.
- **B — Concurrency & Saturation:** multi-stream sweeps (1→64), knee-point
  detection, throughput-vs-p90 latency curves.
- **C — Capability & Fidelity:** needle-in-a-haystack context retention,
  deterministic reasoning/code verification, structured-output/JSON-grammar
  compliance.
- **D — Hardware & Energy:** VRAM, GPU wattage, joules-per-token.

Results are presented through a 60 FPS ratatui+crossterm dashboard and persisted
to embedded SQLite (`~/.local/share/crucible/benchmarks.db`) with JSON/Markdown/CSV
export for CI/CD regression gating.

## Build

The toolchain is pinned to `stable` via `rust-toolchain.toml`.

```sh
cargo build
cargo test
```

## Status

Chunk 1 (scaffold + dependency gate) complete: all crates resolve and compile on
the pinned stable toolchain. See `work/plans/PLAN.md` for the full implementation
roadmap.
