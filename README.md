<div align="center">

# ⚗️ Crucible LLM

**Terminal-based LLM inference benchmarking suite.**

[![Rust](https://img.shields.io/badge/rust-stable-green?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue)](./LICENSE)
[![Binary](https://img.shields.io/badge/binary-static%20%C2%B7%20zero--deps-green)](https://crates.io/)
[![Tests](https://img.shields.io/badge/tests-676%20green-brightgreen)](#)

</div>

Crucible LLM is a comprehensive benchmarking tool for **any OpenAI-compatible inference server** — vLLM, llama.cpp, LM Studio, Unsloth Desktop, SGLang, Ollama, and TGI. It measures generation speed, concurrency capacity, reasoning ability, long-context retrieval, structured-output compliance, and energy efficiency — all from a single interactive TUI or a headless CLI.

Built in Rust for **zero-dependency deployment**. One static binary. No Python. No JVM. No runtime. SQLite is compiled in; GPU telemetry (NVIDIA, AMD, Intel) is built in. Point it at any `/v1` endpoint and start measuring.

### Live Monitor

<div align="center">

![Live Monitor](<docs/img/Live Monitor.png>)

</div>

Real-time throughput graph with auto-scaling y-axis, engine-transition markers, key metrics, and a live event log.

---

## Why Crucible?

- **TUI-first** — an interactive terminal dashboard with real-time throughput graphs, live model discovery, and per-engine metrics.
- **Headless-ready** — a full CLI mode for CI/CD, scripting, and automated regression testing, with pure-JSON output and meaningful exit codes.
- **Comprehensive** — seven benchmark engines covering speed, concurrency scale, reasoning intelligence, long-context retrieval, structured-output fidelity, real-world maximum throughput, and silicon energy efficiency.
- **Portable** — a single static binary. Run it from any terminal on any Linux box. No install, no venv, no system libraries.
- **Honest metrics** — measures what actually matters for a *user*: per-stream experience and practical capacity, not just aggregate throughput. A server that serves 64 users at 2.6 t/s each is not "fast" — it is slow for everyone. Crucible reports the difference.
- **Rigorous timing** — all latency is captured on a hardware cycle clock (`quanta`) in isolated worker rings and is never perturbed by UI repaints, allocation churn, or database writes (the *measurement-isolation* invariant).

---

## Features

### Benchmark Engines

| Engine | Measures | Use Case |
| :--- | :--- | :--- |
| **A · Speed** | tokens/sec (decode), prefill throughput, TTFT, ITL p50/p99, MTP/speculative ratio | How fast does the model respond to **one** user? |
| **B · Concurrency** | sweep 1→32 parallel streams, per-stream t/s, saturation knee, **practical sweet spot** (highest level where each user still gets ≥ 30 t/s) | How many users can your server **actually** serve? |
| **C1 · NIAH** | needle-in-a-haystack retrieval across 2k→128k contexts × 11 depths | Can the model **find a fact** buried in a long document? |
| **C2 · Reasoning** | 13 deterministic math / logic / code challenges, strictly checked | How **smart** is the model? |
| **C3 · Structured** | JSON compliance across 3 schema-complexity levels + grammar speed penalty | Can you **trust it** for API / agent tool-calling? |
| **D · Energy** | GPU power + utilization monitoring. Auto-detects NVIDIA, AMD, or Intel. Measures watts, Joules/token, temperature, VRAM, clocks | What's the **power cost**? |
| **F · Flat Out** | runs at the server's **sweet-spot concurrency** for 60 s — aggregate + per-stream t/s at real-world full load | What's the **headline number** to quote when comparing setups? |

By default a run executes **A, B, C1, C2, C3, F** — and **Engine D auto-enables when a GPU is detected** (it must run on the machine with the GPU; no GPU → off, no N/A panels). Select any subset with `--engine`; an explicit selection always wins over auto-detection.

### GPU Monitoring (Multi-Vendor)
- NVIDIA (NVML), AMD (sysfs/hwmon), Intel (Level Zero + sysfs)
- Auto-detection at startup
- Live telemetry: power, utilization, VRAM, temperature, clocks, throttle
- Energy efficiency: Joules per token
- Full TUI panel (shown only when GPU is active)
- **Dedicated GPU & Power tab** — multi-GPU table, power-over-time graphs, energy (kWh), cost estimation, efficiency metrics (J/token, $/1M tokens)
- **$/1M token cost** — compare your local server's energy cost directly to cloud API pricing (input and output separately)
- **Multi-GPU** — all cards detected and displayed (NVIDIA, AMD, Intel)

### TUI Interface

The **default mode** — a bare `crucible-llm` opens it, no flag required. A keyboard-first, 60 Hz dashboard (vim-style navigation) with six views (`1`–`6`):

- **Interactive setup** with live model auto-discovery (`GET /v1/models`) and a type-to-filter picker (first run only; a saved config skips straight to the dashboard).
- **View 1 · Live Monitor** — real-time throughput graph with auto-scaling y-axis, engine-transition markers, a live event log, and a per-engine benchmark queue panel.
- **View 2 · Concurrency** — the sweep curve with a **practical sweet-spot** recommendation (per-stream ≥ 30 t/s, 2% margin), not just the aggregate throughput knee.
- **View 3 · NIAH** — the long-context retrieval matrix (7 sizes × 11 depths), color-coded green / yellow / red.
- **View 4 · GPU & Power** — multi-GPU telemetry table, power-over-time graph, auto-scaled utilization/temperature charts, energy (kWh), cost estimation, and efficiency metrics (J/token, tokens/watt, $/1M tokens).
- **View 5 · History** — browse stored runs, drill into a run's detail, **compare** two runs side-by-side with signed deltas, and **delete** old ones.
- **View 6 · Config** — an editable form with **per-field explanations** behind a read-only **edit gate**; save to the config file, reset, or launch a run from it.
- **Color-coded capability assessment** (Reasoning / Long-context / Structured / Energy) with plain-language verdicts.
- **Pause/resume** (`Space`), one-key export (`e`), and `Ctrl+C` left **inert** for the terminal's copy selection — quit with `q`, which asks for confirmation.

More views in action:

<div align="center">

**Concurrency** — per-stream and aggregate throughput across the sweep, with the practical sweet spot.

![Concurrency Matrix](<docs/img/Concurrency Matrix.png>)

**NIAH** — the long-context retrieval matrix (7 sizes × 11 depths), color-coded green / yellow / red.

![Needle (NIAH)](<docs/img/Needle (NIAH).png>)

</div>

### Themes

Three complete color palettes — the entire TUI re-skins from a single source of truth:

| Theme | Palette | Character |
| :--- | :--- | :--- |
| **Cyberpunk** (default) | Neon cyan · electric purple · deep blue | The original digital-glow skin |
| **Vampire** | Crimson · gold · dark purple | Dark, gothic |
| **Monochrome Pastel** | Soft blue · lavender · clean whites | Calm, minimal |

On your **first run** (no theme ever chosen) a full-screen **theme picker** appears before Setup — move with `↑`/`↓` and the whole screen live-previews the hovered theme; `Enter` applies and saves it. Change it any time from **Config (View 6)**'s Theme field (`←`/`→` to cycle, `1`/`2`/`3` to select).

### Headless / CLI Mode

For CI/CD, scripting, and regression gating:

- **Pure JSON** output on stdout for programmatic consumption.
- **Export** to JSON, Markdown (GitHub-Flavored tables), or raw CSV (per-packet timestamps).
- **SQLite persistence** of every run for historical comparison and diffing.
- **Exit codes** for pipeline integration: `0` = pass, `1` = benchmark failed, `2` = configuration error.

---

## Quick Start

### Prerequisites

- A Rust **stable** toolchain (to build from source) — **or** a pre-compiled binary from [Releases].
- A running **OpenAI-compatible inference server** (vLLM, llama.cpp, SGLang, Ollama, …).
- Optional: NVIDIA, AMD, or Intel GPU with drivers for Engine D telemetry (auto-detected; the rest of the suite runs fine without one).

### Build

```bash
git clone https://github.com/MadGoatHaz/crucible-llm.git
cd crucible-llm
cargo build --release          # → target/release/crucible-llm  (static binary, GPU telemetry on by default)

# Build without the NVIDIA (NVML) telemetry dependency, if you prefer:
cargo build --release --no-default-features
```

The release binary is fully self-contained: SQLite is bundled (no system `libsqlite3`), and there are **no** runtime library dependencies. NVIDIA GPU telemetry (`nvml`) is a default feature — a plain `cargo build` gives you full GPU monitoring on NVIDIA systems; AMD (sysfs) and Intel (Level Zero, loaded at runtime) need no extra features at all.

### Run (TUI)

The TUI is the **default** — no flag needed:

```bash
./target/release/crucible-llm
```

On your **first run** (no saved config) the interactive Setup flow walks you through it:

1. Enter your server URL (e.g. `http://localhost:8000/v1`).
2. Pick a model from the auto-discovered list (or type it).
3. Configure the benchmark (or accept the defaults).
4. Press **Enter** to launch — watch the live dashboard run A → B → C1 → C2 → C3 → F.

On **subsequent runs**, a saved config (`~/.config/crucible/config.json`) with a URL + model skips Setup and opens the dashboard directly. Re-open Setup any time with `c`.

> Pass `--url … --model …` on the command line to skip straight to the dashboard.

### Run (Headless)

```bash
# Quick single-stream speed test (Engine A), JSON on stdout
./target/release/crucible-llm --headless --url http://localhost:8000/v1 \
  --model my-model --json

# A full multi-engine run with export
./target/release/crucible-llm --headless --url http://localhost:8000/v1 --model my-model \
  --engine speed --engine concurrency --engine niah \
  --export json --export-path ./results.json

# CI/CD integration (non-zero exit on failure)
./target/release/crucible-llm --headless --url "$SERVER_URL" --model "$MODEL" --json --timeout 60
echo "exit code: $?"   # 0 = success, 1 = benchmark failed, 2 = config error
```

> `--json` implies headless. When stdout is **not** a TTY (a pipe or CI runner), the TUI automatically falls back to a headless run — use `--tui` to force the dashboard.

---

## Configuration

### Priority Order

Every field is resolved from the **first** source that provides it:

```
CLI flag  >  environment variable  >  config file  >  built-in default
```

### Config File

`~/.config/crucible/config.json` (override the path with `--config` / `CRUCIBLE_CONFIG`). All fields are optional; only what you specify overrides the defaults.

```json
{
  "url": "http://localhost:8000/v1",
  "model": "my-model",
  "mode": "long",
  "tokens": 10000,
  "iterations": 3,
  "timeout": 120,
  "ladder": [1, 2, 3, 4, 8, 12, 16, 24, 32],
  "matrix_contexts": [0, 8000, 32000],
  "hardware": true,
  "theme": "cyberpunk",
  "engines": {
    "speed": true, "concurrency": true, "niah": true,
    "reasoning": true, "structured": true, "flatout": true,
    "hardware": false
  }
}
```

### Environment Variables

Every flag has a `CRUCIBLE_`-prefixed equivalent: `CRUCIBLE_URL`, `CRUCIBLE_MODEL`, `CRUCIBLE_MODE`, `CRUCIBLE_TOKENS`, `CRUCIBLE_ITERATIONS`, `CRUCIBLE_API_KEY`, `CRUCIBLE_TIMEOUT`, `CRUCIBLE_NOCACHE`, `CRUCIBLE_TOKENIZER`, `CRUCIBLE_JSON`, `CRUCIBLE_VERBOSE`, `CRUCIBLE_NO_COLOR`, `CRUCIBLE_TUI`, `CRUCIBLE_HEADLESS`, `CRUCIBLE_CONFIG`, `CRUCIBLE_LADDER`, `CRUCIBLE_HARDWARE`, `CRUCIBLE_ENGINE`, `CRUCIBLE_LOG_DIR`, `CRUCIBLE_EXPORT`, `CRUCIBLE_EXPORT_PATH`, `CRUCIBLE_MATRIX_CONTEXT`, `CRUCIBLE_THEME`.

### CLI Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--url <URL>` | *(built-in)* | Endpoint: bare host, base URL, or full completions path. |
| `--model <NAME>` | `default` | Model name sent in the request. |
| `--mode <short\|long>` | `short` | `short` = fixed ~50-token prompt; `long` = padded to `--tokens`. (`base` is an alias for `short`.) |
| `--tokens <N>` | `10000` | Target prompt tokens for `long` mode. |
| `--iterations <N>` | `1` | Number of headless single-stream runs (Engine A). |
| `--api-key <KEY>` | — | Sent as `Authorization: Bearer <KEY>`. |
| `--timeout <SECS>` | `120` | Connection / idle-read timeout. |
| `--nocache` | off | Prepend a unique random prefix to bypass the server KV cache (cold-cache runs). |
| `--json` | off | Emit the result as JSON on stdout (implies headless). |
| `--verbose` | off | Per-run chunk detail. |
| `--no-color` | off | Force-disable ANSI colors (default: on only when stdout is a TTY). |
| `--tokenizer <PATH>` | — | HF `tokenizer.json` for exact token counts; without it, counts are `chars/4` estimates (flagged `estimated`). |
| `--tui` | off | Force the interactive dashboard (default mode on a TTY). |
| `--headless` | off | Run the classic headless benchmark instead of the TUI. |
| `--engine <NAME>` | A,B,C1,C2,C3,F | Select which engines a run orchestrates (repeatable): `speed`/`a`, `concurrency`/`b`, `niah`/`c1`, `reasoning`/`c2`, `structured`/`c3`, `hardware`/`d`, `flatout`/`f`. |
| `--ladder <CSV>` | `1,2,3,4,8,12,16,24,32` | Concurrency ladder for Engine B. |
| `--matrix-context <CSV>` | `0,8k,32k` | Context sizes for Engine B's 2D concurrency × context matrix (`0` = the configured prompt). |
| `--no-hardware` | off | Disable the hardware/energy telemetry poller. |
| `--export <fmt>` | — | Export the run: `json`, `md`, or `csv`. |
| `--export-path <PATH>` | `data_dir()/exports/…` | Destination file for `--export`. |
| `--config <PATH>` | `~/.config/crucible/config.json` | Config file path. |
| `--log-dir <PATH>` | `~/.local/share/crucible/logs` | Run-log directory. |
| `--help` / `--version` | | Standard clap help. |

---

## Understanding the Results

### Token Counting

Token counts use the **server's reported `usage.completion_tokens`** as the authoritative source, with a **re-tokenization fallback** (the exact tokenizer when supplied, else a `chars/4` estimate) when a stream is aborted before a usage frame arrives. Crucible never counts raw SSE frames: batched servers (vLLM's multi-token prediction) pack ~2.4 tokens into one frame, and frame-counting understates true output by 30–40 %. This keeps the counts accurate across vLLM, llama.cpp, LM Studio, SGLang, Ollama, and TGI.

### Speed (Engine A)

- **Tokens/sec (decode / TG)** — how fast the model produces output tokens.
- **Prompt throughput (prefill / PP)** — `prompt_tokens / TTFT`; how fast the server ingests your input.
- **TTFT** — time from sending the request to the first token arriving.
- **ITL p50 / p99** — median and worst-case gap between consecutive output tokens.
- **MTP ratio** — `output_tokens / SSE_packets`; `1.0` = standard auto-regressive, `> 1` quantifies speculative-decoding acceptance.

### Concurrency (Engine B)

- **Practical sweet spot** — the most users where **each** still gets ≥ **30 t/s** (29.4 t/s effective with the 2% margin — comfortable for chat / agents / RAG).
- **Maximum usable** — the most users where each gets ≥ **15 t/s** (workable, noticeably slower).
- **Throughput knee** — where *aggregate* t/s stops increasing (a reference point, not the recommendation).
- **Per-stream t/s** — `aggregate ÷ users`; what each individual user actually experiences.

### NIAH (Engine C1)

Hides a unique random fact in a synthetic document and asks the model to retrieve it, across **7 context sizes (2k → 128k) × 11 depths (0 → 100%)** = 77 cells.

- 🟢 **Green ≥ 80 %** · 🟡 **Yellow 50–79 %** · 🔴 **Red < 50 %** (per context size).
- Tells you the **maximum reliable context window** for RAG / long-document QA, plus how prefill speed degrades as context grows.

### Reasoning (Engine C2)

- **13 deterministic challenges** (5 math, 5 logic, 3 code), each validated by a **strict checker** — no LLM-as-judge, no fuzzy scoring.
- **% solved** is a reproducible, cross-model indicator of raw problem-solving ability, reported alongside average decode speed.

### Structured (Engine C3)

A 3-level complexity ladder — **Simple** (flat object) → **Medium** (fixed-length array) → **Complex** (nested schema) — each run under the `response_format: json_object` constraint and scored field-by-field, plus a free-form baseline to quantify the **grammar speed penalty**.

- **Verdict:** `3/3` fully suitable for API/agent use · `2/3` simple-only · `1/3` trivial key-value only · `0/3` not suitable (needs parsing/fallback).

### Energy (Engine D)

- GPU **power draw** (watts) sampled at 100 ms during inference.
- **Joules/token** = `∫P(t)dt / total_tokens` — the silicon-efficiency metric for comparing quantizations and hardware.
- **Requires a local GPU with driver support** — Crucible auto-detects **NVIDIA (NVML, on by default), AMD (sysfs/hwmon), or Intel (Level Zero, with a sysfs fallback)** at startup and activates the matching backend. On a remote or driverless host it degrades gracefully to **N/A** (never a failure, never a spurious `0.0`).

### Flat Out (Engine F)

- **Real-world maximum throughput** — the server loaded at its **concurrency sweet spot** (Engine B's recommendation) for a full **60 seconds**.
- **Aggregate t/s** is the headline number: `total_tokens / 60 s`. **Per-stream t/s** (= aggregate ÷ streams) confirms each user stays at the sweet spot (~30 t/s).
- Without a prior sweep it falls back to a small default load (3 streams). This is the "one number" to quote when comparing setups.

---

## Data & Storage

| What | Where |
| :--- | :--- |
| Benchmark history (SQLite) | `~/.local/share/crucible/benchmarks.db` |
| Exports (default) | `~/.local/share/crucible/exports/` |
| Run logs | `~/.local/share/crucible/logs/latest.log` (+ timestamped archives) |
| Config file | `~/.config/crucible/config.json` |

Every completed run (headless or TUI) persists automatically across three tables — `benchmark_sessions`, `stream_metrics`, and `needle_evaluations`. A storage failure degrades gracefully and **never** changes the exit code of a benchmark.

---

## Architecture

Crucible is a single static binary organized around one core invariant: **measurement isolation**.

- **Four decoupled execution rings** — a stream-worker pool (network I/O), the engine core (metric synthesis), a 100 ms hardware profiler, and the 60 Hz TUI render loop — connected by lock-free channels.
- **Timing never touches the UI.** All latency is stamped by `quanta` (CPU cycle counters, no syscalls) inside the worker rings. The dashboard reads a lock-free, double-buffered (`ArcSwap`) snapshot; a dropped frame, a resize, or a SQLite flush can never perturb a measurement.
- **High-resolution statistics.** `hdrhistogram` drives the p50/p90/p99/p99.9 latency percentiles; a manual zero-allocation SSE line-buffer state machine + `reqwest` (HTTP/1.1 for local endpoints, HTTP/2 for TLS) drive the stream parsing that separates *reasoning* (chain-of-thought) deltas from *content* deltas.
- **Zero runtime dependencies.** SQLite is compiled in (`rusqlite` bundled); GPU telemetry (NVIDIA NVML on by default, AMD sysfs, Intel Level Zero loaded at runtime — no link dependency) degrades to N/A where a driver is missing.

See [`docs/blueprint.md`](./docs/blueprint.md) for the full system specification, metric formulations, and database schema.

---

## License

[GPL-3.0](./LICENSE). This is a copyleft-licensed tool: you may use, study, modify, and share it, provided any derivative works carry the same license.

---

## Contributing

Contributions are welcome. The bar for a merge is high — the measurement path must stay clean.

1. **Fork** the repository and create a feature branch from `master`.
2. **Build and test** — the suite is fully offline (an in-process mock SSE server stands in for a real inference server):

   ```bash
   cargo build --release
   cargo test                     # unit + integration + e2e
   cargo clippy --all-targets     # must be clean
   cargo fmt --check              # must be clean
   ```

3. **Respect the invariants:**
   - **Measurement isolation** — nothing on the render / storage / hardware path may touch the `quanta` timing path.
   - **Graceful degradation** — a missing GPU, driver, tokenizer, or server `usage` block must yield N/A or a flagged estimate, **never** a panic.
   - **Deterministic checkers** — capability scoring (NIAH / reasoning / structured) must stay strict and reproducible.
4. **Open a pull request** with a concise description of the change and the test coverage it adds.

---

## Roadmap

- [x] **AMD GPU telemetry** — watts / VRAM / clocks via in-tree `amdgpu` sysfs + hwmon (no ROCm stack required). *Shipped in v0.1.2.*
- [x] **Intel GPU telemetry** — Level Zero Sysman, with a sysfs/hwmon fallback for minimal hosts. *Shipped in v0.1.2.*
- [ ] **WebSocket / streaming metrics** — a remote, real-time frontend for the dashboard.
- [ ] **Model comparison mode** — A/B two models (or two quantizations) side-by-side with signed deltas.
- [ ] **Docker image** — one-command deployment of the static binary.
- [ ] **Web dashboard** — an optional browser frontend over the same lock-free metrics pipeline.

---

<div align="center">

**Crucible LLM** — forge your inference stack in fire, and measure what survives.

</div>
