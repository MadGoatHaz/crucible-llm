# DEV_LOG — Crucible-LLM

> **Status: IMPLEMENTATION COMPLETE — all 19 plan chunks + Chunk 20 (TUI default + model discovery) delivered and committed. Full suite green (297 tests), `clippy --all-targets` + `fmt --check` clean; Definition of Done items 1–5 satisfied.**
> Authoritative blueprint: `crucible_llm_architecture_blueprint.md`
> Chunked implementation plan: `work/plans/PLAN.md`
> Final QA verdict: `work/scratch/qa_report.txt`
>
> Toolchain note: `rust-toolchain.toml` pinned to `stable` (1.98.1). The old default (1.75) is too old for ratatui/reqwest/quanta/hdrhistogram/sysinfo.

## Active Work

None - implementation complete


## Completed Milestones

- **plan-authoring** — Blueprint distilled + 19-chunk plan authored; toolchain pinned to stable 1.98.1, MSRVs verified.
- **chunk-1** — Dependency gate PASSED: 22 crates resolve+compile on stable 1.98.1, zero warnings; full `src/` module tree stubbed.
- **chunk-2** — MonotonicInstant (quanta) + StreamTimestamps T0..Tn + LatencyHistogram (hdrhistogram, p50–p99.9).
- **chunk-3-sse** — Low-alloc incremental SSE parser (BOM strip, multi-line data join, [DONE] flag, malformed-skip, chunk model).
- **chunk-3** — TUI layout + event loop: 5-view App state machine, crossterm raw/alt-screen lifecycle, 60Hz tick decoupled from timing path.
- **chunk-4** — PromptGenerator (short/long/nocache) + Tokenizer (HF tokenizers, chars/4 graceful fallback).
- **chunk-5** — StreamWorker inference-engine adapter (SSE + plain-JSON fallback, T0..Tn on quanta, retriable-error handling).
- **chunk-6** — ArcSwap<MetricsSnapshot> double-buffered metrics pipeline (lock-free reads, measurement-isolation intact).
- **chunk-7** — Single-stream CLI parity / headless mode (Python-parity defaults, 3-source config layering, exit-1 rule).
- **chunk-9** — View 1 Live Monitor & Telemetry (all five panels from the snapshot).
- **chunk-10** — Multi-stream pool + concurrency ladder sweep (Engine B).
- **chunk-11** — Knee-point detection + View 2 concurrency matrix (optimal operational envelope).
- **chunk-12** — SQLite persistence engine (idempotent schema, row models, atomic `persist_run`).
- **chunk-13** — Export JSON / GFM Markdown / raw CSV.
- **chunk-14** — View 4 Historical Comparison & Regression Diffing.
- **chunk-15** — Engine C1 Needle-In-A-Haystack + View 3 (N×M retrieval grid).
- **chunk-16** — Engines C2 & C3 (deterministic reasoning bank + structured-output/JSON compliance).
- **chunk-17** — Engine D Hardware & Energy Profiler (NVML feature-gated, sysinfo fallback, joules-per-token).
- **chunk-18** — View 5 Config + full engine integration (editable/persistent config form, EngineSelection A–D, `run_selected`).
- **chunk-19** — Release binary hardening + e2e smoke tests + full README.
- **exit-code-fix** — `llmspeedtest.py` exit-code fix (py_compile OK, dead-endpoint run exits 1).
- **chunk-20** — TUI as the default mode (bare invocation launches the dashboard; `--headless`/`--json` force the classic run; non-TTY falls back to headless; `--banner` demoted to a hidden flag) + model discovery client (`src/client/models.rs`: `GET {base}/v1/models`, `ModelInfo`, `ModelError`, id-sorted deduped list) wired into `App` via a lock-free `ResultSlot<Vec<ModelInfo>>` + `start_discovery()` (`tokio::spawn`, measurement-isolation intact). 297 tests green incl. live mock-discovery e2e.
