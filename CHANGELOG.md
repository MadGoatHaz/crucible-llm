# Changelog

All notable changes to Crucible LLM.

## [Unreleased]

### Added
- **NIAH (C1) matrix configurable in the TUI** — `niah_contexts` / `niah_depths` / `niah_custom` are now first-class editable fields in the Config view (View 6) and the Setup flow's stage-3 form, with `k`/`M` suffix support matching the CLI. The ladders display through the engine's shared `human_size` formatter, blank/invalid input falls back to the built-in defaults (lenient, not a hard error), and a single `parsed_niah()` helper keeps the run (`to_config`) and persist (`save`) paths in lockstep.

### Fixed
- **`F2` no longer wipes the NIAH ladders** — `save()` previously hard-coded `niah_contexts` / `niah_depths` / `niah_custom` to `None`, so saving from the TUI reset any file-set ladder back to the built-in defaults. It now writes the **parsed** values, so a TUI-set ladder persists to `~/.config/crucible/config.json` and reloads on the next launch. The R3 budgets (`niah_max_tokens` / `structured_max_tokens`) intentionally remain `None` (out of scope).
- **Energy (D) J/token was ~25× inflated** — three compounding root causes are fixed: (1) an NVML phantom GPU power reading could pass the plausibility clamp **silently** when *both* of the card's power-management-limit (TDP) queries failed (e.g. an A4000 reporting ~989 W on a 200 W-TDP card) — a **per-model TDP table** now engages the >1.5×TDP cap and **logs a WARN**, with a per-model ceiling below the 1000 W global cap, so a known card's phantom reading can never pass unflagged; (2) the energy integral now runs over the **full load window** (first engine to last) instead of the bounded 6-minute display trace, so long runs are measured over their entire duration; (3) the J/token denominator now sums the **real per-cell prompt + completion tokens** (the NIAH C1 cells count their real prompt tokens; the old `cells × gen-cap` product is only a labeled fallback when the server reports no usage). The measured **idle baseline is also subtracted** before integrating (load-attributable energy), and the **`cpu_power_estimated` flag** (fixed 40 W CPU when no RAPL / Super I/O) is now **surfaced in the TUI**. Net measured effect: J/token dropped from ~2.0 to ~0.1 and peak power is now a realistic, clamped value. *Live-validated on a host where NVML's TDP queries succeed (the clamp, the J/token correction, and the full-window integration are confirmed there); the A4000 "queries-fail → per-model TDP table + WARN" branch is covered by offline unit tests, not a live A4000 run.*
- **Flat Out (F) ran at the wrong concurrency** — it previously used Engine B's **practical sweet spot** (the highest level where every user still gets ≥ ~29.4 t/s — a *hosting-comfort* number) instead of the **max-throughput** point. It now runs at the concurrency **max-throughput level** (the sweep *envelope*: the last level before the saturation knee, else the peak aggregate), resolved through the shared **`flatout_streams_for`** helper so the headless and TUI paths always agree. The TUI now keeps the two definitions **distinct**: the recommendation panel shows the **Practical Sweet Spot** (how many users to host comfortably) while the curve marks the **Max Throughput** level (what Flat Out runs at), so the two numbers no longer read as contradictory.

## [0.1.5] - 2026-10-12

The NIAH expansion + SSE keep-alive + hybrid-reasoning cycle: long-context NIAH becomes runtime-configurable, 1M+ scalable, and token-honest (and lands in `--json`); the stream transport stops misrouting valid SSE (Issue #1) and the TUI labels approximate sizes honestly (Issue #4); and the C1/C3 capability engines adopt the **hybrid reasoning design** — the generation budget carries correctness, `disable_thinking` is best-effort, and a new `truncated_by_reasoning` outcome replaces the false capability failures — with live-calibrated defaults (C1 2048 / C3 4096) sized for verbose reasoners.

### Added
- **NIAH (C1) runtime-configurable ladders** — `--niah-contexts <CSV>` (e.g. `16k,64k,256k,1M`), `--niah-depths <CSV>` (e.g. `0,50,100`), and `--niah-custom <SIZE>` (single-size "count test", e.g. `256k`), each layered **CLI > env (`CRUCIBLE_NIAH_CONTEXTS` / `CRUCIBLE_NIAH_DEPTHS` / `CRUCIBLE_NIAH_CUSTOM`) > config file > default** (2k→128k × 11 depths). `--niah-custom` takes precedence and collapses the matrix to one size row — for probing a RoPE/YaRN-extended limit.
- **1M+ haystack scaling** — the document generator is unbounded (a 1M-token haystack builds in a few hundred ms; `target × 4` char math is overflow-safe for any `u32` target). The practical ceiling is the server's max context; an over-limit size degrades to a labeled per-cell HTTP 400 (no crash, never a silent zero).
- **NIAH results in `--json`** — an additive `niah` section (accuracy headline, `retrieved`/`total`, `reliable_up_to`, per-size recall with labels, full per-cell detail incl. `ttft_s` / `prefill_tps` / `state` / `no_baseline` / `approximate`) so a headless `--json` run is self-contained; the Engine-A JSON shape is untouched.
- **Hybrid reasoning design (Option C)** — the generation **budget is the correctness guarantee**: `max_tokens` is the one per-request knob every OpenAI-compatible server honors, so C1/C3 size their budgets to carry correctness end-to-end. `disable_thinking` is **best-effort** — a speed optimization that is fast where a server honors it (vLLM/SGLang/LM Studio/Ollama) and a harmless no-op where a config-only server silently drops the per-request fields (the OpenAI spec's graceful degradation) — so correctness can never depend on suppression. **Content-only scoring is kept** (a value mentioned only in CoT is still not credited), and **C2 is left alone** (its challenges need the chain-of-thought; its 512-token budget is unchanged).
- **`truncated_by_reasoning` outcome** — a stream that completes **all-CoT / zero-content** (the budget exhausted in chain-of-thought, nothing visible committed) is now a distinct **over-thinker** result, not a capability failure: *inconclusive (budget)*, honest about what was observed, and surfaced on every output surface — the terminal/headless one-liner, `--json` (a top-level count plus a per-cell / per-case flag), the SQLite DB (`needle_evaluations` column + `engine_results` JSON), and the TUI (a 4th grid marker and the capability breakdown). A genuine transport failure is still a fail, never a truncation.
- **Configurable C1/C3 generation budgets (R3)** — `--niah-max-tokens <N>` / `CRUCIBLE_NIAH_MAX_TOKENS` (C1, was a fixed 1024) and `--structured-max-tokens <N>` / `CRUCIBLE_STRUCTURED_MAX_TOKENS` (C3, was a fixed 2048), each layered **CLI > env > config file (`niah_max_tokens` / `structured_max_tokens`) > default**, so any model — from a token-efficient finetune to a verbose reasoner — can be accommodated without a recompile.
- **Per-cell / per-case `reasoning_tokens` in `--json` (R3)** — the `niah` cells and `structured` cases now carry the server-reported CoT token count (`null` when the backend reports no usage block), the calibration measurement of a model's reasoning usage.
- **`--no-disable-thinking` force-test switch (R3)** — the C1/C3 best-effort `disable_thinking` suppression (a speed optimization) can be turned off client-side, so the `truncated_by_reasoning` safety net (designed for servers that drop the suppression field) can be exercised live: tiny budget + thinking on → the model runs out mid-CoT and is reported *truncated*, never a capability fail.

### Changed
- **C1 `chars/4` approximate labeling** — without a `--tokenizer`, the haystack is a `chars/4` estimate and every context size now reads `Nk (approx.)` consistently across the headless run log (header + per-cell tags), the TUI NIAH grid, the TUI Long Context bar, and the `--json` `niah` section; with a tokenizer, labels are exact.
- **Headless log size formatting** — NIAH sizes render through the shared `human_size` formatter (`32k` / `256k` / `2M`) in the run log and worker tags (was `128000k`); the `[C1] NIAH (…)` header now shows the *configured* ladders and names the count-test mode.
- **Hybrid-reasoning defaults raised with live-calibrated headroom (R3)** — the C1 per-cell budget goes **1024 → 2048** and the C3 per-run budget **2048 → 4096**. Calibrated live against the token-efficient qwen3.8-27b finetune (0 CoT when the server honors suppression; 57–90+ when forced to think): the new defaults leave large headroom for a verbose reasoner (several× the finetune's usage, e.g. vanilla qwen3.8's 256-CoT small-prompt baseline) to finish its chain-of-thought *and* commit the answer, while staying a rarely-hit ceiling on suppression-honoring servers.
- **Docs updated for the hybrid design** — README, blueprint, architecture, and user guide now describe the hybrid reasoning model (budget = correctness guarantee, `disable_thinking` = best-effort speed, the `truncated_by_reasoning` outcome, and the new C1/C3 budget flags) with every claim verified against the implementation; the v0.1.4 "suppression is load-bearing" narrative is replaced (its 64-token / 256-token budgets kept only as labeled history).

### Fixed
- **SSE keep-alive misroute (Issue #1)** — the stream-vs-JSON route is now decided by the response **`Content-Type` + request intent** (`text/event-stream` → SSE, `application/json` → JSON, absent/other → SSE since we always send `stream: true`), not by sniffing the first body byte; a `:keep-alive` comment preamble can no longer push a valid stream onto the JSON path and poison the result. A stream ending with zero valid `data:` frames first tries a **JSON rescue** on the retained body (1 MiB inspection cap), and a body that is genuinely neither surfaces as a real `StreamError::InvalidTransport` — never a silent retrieval miss. The SSE parser gains `data_frames` / `valid_frames` counters to drive that resolution.
- **TUI token-honest labeling (completes Issue #4)** — the interactive TUI now renders the `(approx.)` qualifier: the NIAH grid (View 3) row labels (`128k (approx.)`) and the Live view's Long Context capability bar (`≤ 128k (approx.) reliable`). Both read through the same shared `context_size_label` the headless run log and the `--json` `niah` section use, so a no-`--tokenizer` run is never presented as exact on any output surface (with a `--tokenizer`, all labels are plain).

### Code Quality
- 967 tests, 0 failures (21 suites: unit + integration + e2e, fully offline via the in-process mock SSE server)
- `cargo build --release` 0 warnings; `cargo clippy --all-targets` 0 warnings

## [0.1.4] - 2026-10-11

The full remediation cycle: metric integrity, reasoning-model correctness, engine selection, and traceability.

### Fixed
- **NIAH (C1) reasoning-model false-fails** — each cell now suppresses the model's chain-of-thought (per-backend: `enable_thinking=false` for vLLM/SGLang/LM Studio, `think=false` for Ollama, `/no_think` for llama.cpp) and checks retrieval on the **visible answer only**. A reasoning model whose 64-token budget used to be all-CoT (≈35 % false-fail) now retrieves ≈100 %; a needle mentioned only in CoT can no longer false-pass.
- **Engine A's conflated "pre-fill speed"** — the single `prompt_tokens / TTFT` number on a ~45–87-token short prompt (a latency floor wearing a bandwidth label) is **split** into two labeled metrics: **first-token latency (ms)** from the short prompt (dual: perceived `T3−T0` + server-side `T3−T1`) and **pre-fill throughput (tok/s)** from a long prompt (default 10k, three averaged runs).
- **C2/C3 checker correctness on reasoning models** — C2 judges the committed content span (first-number/whole-value math match, first whole-word verdict, string/comment-aware delimiter balance) with thinking suppressed; C3 suppresses thinking on all runs.
- **Honest T1** — `T1` is stamped at *response-headers receipt* (not "request write complete"), and both TTFT flavors are published since what falls inside `T1→T3` varies by backend.
- **Methodology block** — every formula string in the JSON `methodology` block now states exactly what the code does (dual TTFT, the prefill/latency split, the canonical MTP η, the dual denominators, the real warmup, the $/kWh rate source).
- **Energy math re-based onto the 100 ms trace** — `∫P(t)dt` is windowed to the active load on the 100 ms hardware trace (the 1 Hz history is graph-only), so a sub-second TTFT yields a *real* prefill-phase power; the J/token denominator spans **every engine that ran**, not just A+B.
- **`--no-hardware` now disables both the poller and the energy engine** consistently in the headless and TUI paths.

### Added
- **All seven engines individually selectable** — A/B/C1/C2/C3/D/F via `--engine <name>` (repeatable, plus the `all` primitive), `CRUCIBLE_ENGINE`, the config file's `engines` object, and the TUI (Setup / View 6). **Headless honors the selection**: Engine A is no longer unconditional (`--engine niah` runs only C1) and Engine D is selectable. An empty selection is an operator error, not a silent full run.
- **Traceability** — every run gets a session UUID that stamps the `benchmark_sessions` row, the run-log archive (`run-<timestamp>-<id[:8]>.log`), and the default export name (`crucible-<id[:8]>.<ext>`). The session row records which engines ran (`engines` column) and the headline result of every non-stream engine (`engine_results`: B/C2/C3/D/F); Engine A persists per-run `stream_metrics` and C1 persists per-cell `needle_evaluations` — in **both** the TUI and headless paths.
- **Complete exports** — JSON/MD/CSV now carry the energy object, Engine A's prefill/latency split, the dual per-stream denominators, NIAH reliability, and `reasoning_tokens`.
- **`reasoning_tokens` tracked end-to-end** — parsed from the server's `usage` detail, persisted, exported, and driving the decode-window split (Engine A's decode t/s is the *answer phase* when the detail is present; pure-CoT streams report 0.0).
- **Canonical MTP η** — `completion_tokens / total_token_frames` (all Reasoning + Content frames; η > 1 legitimate) as the single definition across Engine A, Engine B, and the methodology block.
- **Dual per-stream denominators (B & F)** — `tps_per_completed_stream` (capacity) / `tps_per_requested_stream` (user-experience) / `failed_stream_ratio`, with levels > 20 % failed flagged `high_failure`.
- **Real discarded warmups** — Engine A runs one discarded warmup iteration and Engine B one discarded warmup level before measuring (logged, counted nowhere).
- **C3 real constrained decoding** — constrained runs send `response_format: { "type": "json_schema", "json_schema": { …, "strict": true } }` (strict schemas, `additionalProperties: false`) where the transport supports it — engaging server-side XGrammar/Outlines/GBNF — with a clearly-flagged `json_object`-hint fallback. The speed penalty is reported only when the constraint was effective (`constraint_effective`), plus a per-case `stability_rate` (repeatable cases).
- **Stream-integrity flags surfaced** — `usage_discrepancy` (usage > 4× frames), `e2e_only` (non-streaming result), `itl_clamped` (histogram 60 s cap), and `degraded` (no token frame → TTFT N/A) in the JSON, terminal, and TUI.
- **Engine F honesty & portability** — `max_tokens` clamped to the model's probed max context (`GET /v1/models`); a uniform HTTP-400/zero-token run surfaces as a `config_error`, never a silent 0 t/s; `chars/4` fallback counts are flagged (`tokens_estimated` / `counting_method`, `≈ … (est.)`).
- **Energy N/A rule** — no power telemetry (no GPU, no driver, non-Linux host) degrades every derived metric to a clearly-labeled `N/A`, never a spurious `0.0`; Linux-only by design.
- **NIAH reliability display** — "reliable up to" (largest size with ≥ 80 % pass rate) drives the TUI capability bar; missing-baseline cells are flagged distinctly and non-tokenizer sizes are labeled `(approx.)`.
- **C1 accuracy counts run cells only** — a partial/interrupted matrix no longer dilutes the headline with never-run cells.

### Changed
- Engine A report layout: two clearly labeled sections (first-token latency / pre-fill throughput) plus the intact decode/MTP numbers and a pointer to C1's 2k→128k prefill curve.
- 100 ms hardware poller runs on a **dedicated OS thread** in both paths (decoupled from the runtime that owns the measured workload).
- Docs updated to match the code: README, blueprint, architecture, user guide (all claims verified against the implementation).

### Code Quality
- 893 tests, 0 failures (19 suites: unit + integration + e2e, fully offline via the in-process mock SSE server)
- `cargo build --release` 0 warnings; `cargo clippy --all-targets` 0 warnings

## [0.1.3] - 2026-10-05

### Added
- **View 4: GPU & Power Monitor** — dedicated tab with comprehensive hardware telemetry
- **Multi-GPU support** — per-GPU table showing all cards (power, util, temp, VRAM, clocks, throttle)
- **Power-over-time graph** — 1Hz sampling during benchmark runs, rendered as a live graph
- **Idle power baseline** — 5-second measurement before test starts; compute power = total − idle
- **Energy tracking** — kWh calculated via trapezoidal integration of power samples
- **Cost estimation** — configurable $/kWh rate; shows estimated cost per run
- **Efficiency metrics** — J/token, J/ktoken, tokens/watt, $/1M tokens
- **Peak tracking** — max power, max temperature across the entire run
- **`$/kWh` config field** — set your electricity rate in Config (tab 6)
- **Multi-GPU detection (AMD + Intel)** — scans all DRM cards, not just the first. Quad/octo GPU setups show every card.
- **$/1M token cost analysis** — separate input (prefill) and output (decode) rates, plus a token-weighted blended rate. Throughput-based energy model. User-set $/kWh rate in Config.
- **Duration anchored to test start** — no longer counts from app launch.
- **Full GPU names** — no truncation in per-GPU table.
- **CPU power monitoring** — Intel RAPL, AMD Super I/O, or estimated fallback. Total system draw (GPU + CPU) now included in energy and cost calculations.
- **NVML power plausibility clamp** — anomalous readings (>2× TDP) are detected, logged, and capped. The TDP reference is the card's *default* power-management limit (factory TDP, not user-settable), falling back to the current limit — so the clamp engages even when the current-limit query fails. Prevents impossible values like 974W on a 200W A4000 from inflating the $/1M cost.
- **All-vendor power ceiling** — 1000W global cap on all GPU backends as a last-resort safety net (no single GPU draws more).
- **System power display** — GPU tab now shows "System: X W (GPU Y + CPU Z)" for true total draw.

### Changed
- Tab reorder: GPU & Power → [4], History → [5], Config → [6]
- Auto-scaled utilization/temperature graphs (show variation, not flat lines)
- Per-GPU average and max temperature tracking
- Duration counter freezes when benchmark completes
- Mean and peak markers on all GPU graphs

### Fixed
- **Cost calculation accuracy** — throughput-based energy model correctly attributes power to input vs output tokens across multi-request runs. Blended rate is always between input and output.
- **NVML power clamp fixed** — uses factory TDP (`power_management_limit_default`) instead of user-settable limit. Global ceiling tightened to 1000W. Eliminates phantom power readings (e.g., 974W on a 200W card).
- **Avg power window** — now only averages during active test duration. Freezes when tests complete. No more creeping from app launch.
- **Manual idle measurement** — press `[i]` in GPU tab for a fresh 10-second no-load baseline.
- **Token count clarity** — display shows "N input (M requests)" to distinguish per-request from cumulative.

### Improved
- GPU monitoring now tracks history (time-series) not just current snapshot
- System power summary panel (total, idle, compute, peak)
- Per-GPU names and clocks displayed in table

### Code Quality
- New `src/hw/monitor.rs` — power history tracker with 1Hz sampling
- New `src/ui/views/gpu.rs` — View 4 rendering
- 676 tests, 0 clippy warnings, fmt clean

## [0.1.2] - 2026-10-04

### Added
- **AMD GPU monitoring** — sysfs/hwmon backend: utilization, VRAM, temperature, power, clocks
- **Intel Arc GPU monitoring** — Level Zero Sysman via libloading (no link dependency, graceful fallback)
- **Intel sysfs fallback** — for i915/xe drivers when Level Zero unavailable
- **Vendor-agnostic GpuBackend trait** — unified interface for all GPU monitoring backends
- **GPU auto-detection** — detects NVIDIA/AMD/Intel at startup, activates correct backend
- **NVIDIA by default** — `nvml` feature now enabled by default (was opt-in in 0.1.1)
- **Full GPU telemetry panel in TUI** — power, utilization, temp, clocks, VRAM, throttle, J/token (shown only when GPU active)
- **Engine D auto-select** — enabled by default when GPU detected, disabled when not
- **Config/Setup shows detected GPU** — "D: Energy — [✓] RTX 4090"

### Fixed
- NVIDIA telemetry was silently off in v0.1.1 default builds (feature not enabled)
- Hardware panel was removed in v0.1.1 remote-user redesign; now conditionally shown when GPU data is live

### Improved
- Engine D results now include GPU vendor, model, and mean utilization
- Flat Out added to capability scores display
- Dead dependency removed (eventsource-stream)
- Repo URL corrected in all docs
- ~6 new hardware tests; suite at 652 total

### Code Quality
- `#[must_use]` annotations on pure functions
- Module docs corrected (engine count, ladder defaults)
- Zero clippy warnings, zero fmt issues

## [0.1.1] - 2026-10-03

### Added
- **3 Visual Themes**: Cyberpunk (cyan/purple/neon), Vampire (crimson/gold/gothic), Monochrome Pastel (soft/clean). First-run theme picker with live preview. Switchable from Config (tab 5).
- **Flat Out Engine (F)**: 60-second real-world maximum throughput test. Runs at your server's concurrency sweet-spot for full-load measurement. The "one number" to quote when comparing setups.
- **Authoritative Token Counting**: Uses server-reported `usage.completion_tokens` as primary source. Re-tokenization fallback. Accurate across all backends (vLLM batches ~2.4 tokens/frame — I count actual tokens, not frames).
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
- **vLLM token undercount**: Server batches ~2.4 tokens/SSE frame. I now count actual tokens, not frames.
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
