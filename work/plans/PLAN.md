# CRUCIBLE-LLM — Implementation Plan

> **Authoritative spec:** `crucible_llm_architecture_blueprint.md` (all formulas, schemas,
> layouts, and phase requirements live there — consult it before deviating).
> **Recon summary:** `work/scratch/recon_blueprint.md`.
> **Status:** Planning complete. Chunks below are ordered so dependencies flow forward
> (foundation → features → integration). Each chunk is self-contained: an implementation
> agent can pick up any chunk that has its dependencies satisfied and know exactly what to build.

---

## 1. Project Overview (1 paragraph)

Crucible-LLM is a native-Rust, zero-runtime-dependency terminal application that benchmarks
OpenAI-compatible streaming LLM inference endpoints (vLLM, llama.cpp, SGLang, Ollama). It ports
the two-file Python prototype (`llmspeedtest.py` / `llmspeedtest2.py`) into a persistent,
multi-engine benchmarking suite with four engines: **(A) Speed & Latency** (microsecond TTFT,
isolated prefill/decode throughput, MTP/speculative-decoding ratio, warm/cold KV-cache detection),
**(B) Concurrency & Saturation** (multi-stream sweeps 1→64, knee-point detection, throughput-vs-p90
latency curves), **(C) Capability & Fidelity** (needle-in-a-haystack context retention,
deterministic reasoning/code verification, structured-output/JSON-grammar compliance), and
**(D) Hardware & Energy** (VRAM, GPU wattage, joules-per-token). It presents results through a
60 FPS ratatui+crossterm dashboard (five views: Live Monitor, Concurrency Matrix, Needle, History
Diff, Config) with a lock-free, double-buffered metrics pipeline so UI repaints never touch the
timing path. Persistence is embedded SQLite (`~/.local/share/crucible/benchmarks.db`) with
JSON/Markdown/CSV export for CI/CD regression gating.

---

## 2. Toolchain & Environment (READ FIRST)

- **Pin the toolchain.** Create `rust-toolchain.toml` at the repo root:
  ```toml
  [toolchain]
  channel = "stable"
  components = ["clippy", "rustfmt"]
  ```
  The machine's *default* toolchain is **1.75.0** (too old), but the installed **`stable` =
  1.98.1** satisfies every latest-crate MSRV:
  | crate | latest | MSRV | ok on 1.98.1 |
  |---|---|---|---|
  | ratatui | 0.30.2 | 1.88.0 | ✅ |
  | crossterm | 0.29.0 | 1.63.0 | ✅ |
  | tokio | 1.53.1 | 1.71 | ✅ |
  | reqwest | 0.13.5 | 1.85.0 | ✅ |
  | quanta | 0.13.0 | 1.87 | ✅ |
  | hdrhistogram | 7.6.0 | 1.88 | ✅ |
  | sysinfo | 0.39.6 | 1.95 | ✅ |
  | nvml-wrapper | 0.13.0 | 1.60.0 | ✅ |
  | eventsource-stream / rusqlite / tokenizers / serde | latest | (undeclared) | ✅ on 1.98.1 |
- **Feature-gate hardware telemetry.** `nvml-wrapper` needs the NVIDIA driver (`libnvidia-ml`) at
  runtime. Put it behind a cargo feature `nvml` (default-off) and detect availability at runtime;
  report `"N/A"` (never panic) when no GPU/driver is present.
- **Make tokenization optional.** The HF `tokenizers` crate is a heavy native build and needs a
  `tokenizer.json`. Support an optional `--tokenizer <path>`; fall back to the `chars/4`
  heuristic (marking the result `estimated`) when absent.
- **Measurement isolation is the #1 invariant.** Worker-pool timing (quanta) must never be touched
  by UI repaints, GC-like allocation churn, or the SQLite flusher. Enforce via separate rings +
  lock-free channels + a double-buffered (ArcSwap) snapshot.

---

## 3. Rust Project Structure

### 3.1 `Cargo.toml` (dependencies)

```toml
[package]
name = "crucible-llm"
version = "0.1.0"
edition = "2021"
description = "High-performance terminal LLM benchmark & inference profiler"

[dependencies]
# TUI
ratatui = "0.30"
crossterm = "0.29"
# Async engine
tokio = { version = "1.53", features = ["full"] }
# Networking / SSE
reqwest = { version = "0.13", features = ["json", "stream", "http2"] }
eventsource-stream = "0.2"
futures-util = "0.3"
# Tokenization (optional at runtime)
tokenizers = "0.23"
# Timing & statistics
quanta = "0.13"
hdrhistogram = "7.6"
# Hardware monitoring (feature-gated)
sysinfo = "0.39"
nvml-wrapper = { version = "0.13", optional = true }
# Persistence
rusqlite = { version = "0.40", features = ["bundled"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
# Concurrency plumbing
crossbeam-channel = "0.5"
arc-swap = "1.7"
# Utilities
clap = { version = "4", features = ["derive"] }
thiserror = "2"
anyhow = "1"
dirs = "5"
rand = "0.9"
uuid = { version = "1", features = ["v4"] }

[features]
default = []
nvml = ["dep:nvml-wrapper"]

[profile.release]
lto = "thin"
codegen-units = 1
```

> Note: `rusqlite` uses `features = ["bundled"]` so it compiles SQLite from source (zero system
> dependency → true static binary). Verify exact latest patch versions at `cargo add` time.

### 3.2 `src/` layout

```
crucible-llm/
├── Cargo.toml
├── rust-toolchain.toml              # channel = "stable" (1.98.1)
├── .gitignore
├── README.md
├── src/
│   ├── main.rs                      # CLI parse (clap), runtime bootstrap, mode dispatch
│   ├── lib.rs                       # crate root; re-exports modules
│   ├── config.rs                    # Config struct: all CLI flags + defaults + data-dir path
│   ├── timing.rs                    # quanta clock wrapper; MonotonicInstant; T0..Tn stamps
│   ├── sse/
│   │   ├── mod.rs
│   │   ├── parser.rs                # incremental, low-alloc SSE frame parser
│   │   └── chunk.rs                 # Chunk enum: Reasoning/Content/Control/Usage + delta extract
│   ├── client/
│   │   ├── mod.rs
│   │   ├── stream.rs                # StreamWorker: one HTTP/2 SSE stream, timestamps, events
│   │   └── pool.rs                  # spawn/manage N workers; bounded tokio mpsc fan-in
│   ├── metrics/
│   │   ├── mod.rs
│   │   ├── engine.rs                # EngineCore: aggregate, classify, ITL, §7 formulas
│   │   ├── histogram.rs             # HdrHistogram wrapper: p50/p90/p99/p99.9
│   │   └── state.rs                 # ArcSwap double-buffered snapshot for UI
│   ├── prompt/
│   │   ├── mod.rs
│   │   ├── generator.rs             # short/long/nocache prompt gen (port of both .py)
│   │   └── tokenizer.rs             # HF tokenizers (optional) + chars/4 fallback
│   ├── engines/
│   │   ├── mod.rs
│   │   ├── speed.rs                 # Engine A: TTFT/PP/TG/MTP/cache orchestration
│   │   ├── concurrency.rs           # Engine B: ladder sweep + knee detection
│   │   ├── hardware.rs              # Engine D: power↔token correlation, joules/token
│   │   └── capability/
│   │       ├── mod.rs
│   │       ├── niah.rs              # Engine C1: Needle-in-a-Haystack
│   │       ├── reasoning.rs         # Engine C2: deterministic reasoning/code bank
│   │       └── structured.rs        # Engine C3: JSON/grammar compliance
│   ├── hw/
│   │   ├── mod.rs
│   │   └── nvml.rs                  # 100ms poll worker: VRAM/clock/temp/power (feature-gated)
│   ├── storage/
│   │   ├── mod.rs
│   │   ├── db.rs                    # rusqlite open/migrate (schema.sql), insert/query
│   │   ├── schema.sql               # DDL from blueprint §8
│   │   ├── models.rs                # serde row structs
│   │   └── export.rs                # JSON / GFM Markdown / CSV
│   └── ui/
│       ├── mod.rs
│       ├── app.rs                   # App state machine; View enum; key handling
│       ├── event.rs                 # crossterm event loop + 60Hz tick (decoupled)
│       ├── theme.rs                 # colors/styles
│       └── views/
│           ├── mod.rs
│           ├── live.rs              # View 1: Live Monitor & Telemetry
│           ├── concurrency.rs       # View 2: Concurrency & Saturation matrix
│           ├── needle.rs            # View 3: NIAH matrix grid
│           └── history.rs           # View 4: Historical Comparison & Diff
└── tests/
    ├── sse_parser_test.rs
    ├── metrics_test.rs
    ├── prompt_test.rs
    └── storage_test.rs
```


---

## 4. Phased Implementation Plan (Chunks)

**Chunk format:** `ID · Title` → *Build* / *Files* / *Depends* / *Acceptance*.
Order is the build order. A chunk is ready when all its *Depends* are DONE.

---

### PHASE 1 — Rust Core & Single-Stream Parity

#### Chunk 1 · Project scaffold, toolchain pin & dependency gate
- **Build:** Create the full `Cargo.toml` (dependencies from §3.1) and `rust-toolchain.toml`
  (channel `stable`). Stub `src/lib.rs` (module tree declarations) and `src/main.rs` (prints a
  banner + version, exits 0). Add `.gitignore` (`/target`, `benchmarks.db`) and `README.md`.
  This chunk is the **dependency gate**: it proves all crates resolve and compile on 1.98.1.
- **Files:** `Cargo.toml`, `rust-toolchain.toml`, `.gitignore`, `README.md`, `src/lib.rs`, `src/main.rs`
- **Depends:** — (first chunk)
- **Acceptance:** `cargo build` and `cargo test` succeed on the `stable` toolchain with zero
  unresolved deps; `cargo run` prints the banner and exits 0; `cargo clippy` is clean.

#### Chunk 2 · Timing core (quanta) + latency histogram (hdrhistogram)
- **Build:** `timing.rs` — a `MonotonicInstant` wrapper over `quanta`'s cycle clock exposing
  nanosecond deltas with no syscalls; define the `T0..Tn` timestamp record type
  (`T0` socket start, `T1` request write complete, `T2` first byte, `T3` first token frame,
  `Tn` stream close). `metrics/histogram.rs` — a thin `LatencyHistogram` over `hdrhistogram`
  with `record(nanos)` and `percentile(p) -> f64` for p50/p90/p99/p99.9.
- **Files:** `src/timing.rs`, `src/metrics/mod.rs`, `src/metrics/histogram.rs`, `tests/metrics_test.rs`
- **Depends:** Chunk 1
- **Acceptance:** Unit test records a known synthetic distribution (e.g., 10k samples with a
  known p50/p99) and asserts recovered percentiles are within 2% of the true values; delta of
  two instants is monotonic and ≥ 0.

#### Chunk 3 · Low-allocation SSE parser + chunk model
- **Build:** `sse/chunk.rs` — `Chunk` enum `{ Reasoning(String), Content(String), Control,
  Usage{prompt_tokens, completion_tokens} }` + a `classify(delta) -> Chunk` fn that reads
  `reasoning`/`reasoning_content` first, then `content`, else control/usage. `sse/parser.rs` —
  an incremental, low-alloc parser that consumes raw bytes (via `eventsource-stream` or a manual
  line buffer), handles: `data:` prefix (+ optional space), multi-line `data:` join, blank-line
  frame separator, `[DONE]` terminator, BOM strip, `event:`/`id:`/`retry:` ignore, and
  malformed-JSON tolerance (count + skip, never panic). Emits `ParsedFrame { chunk, t_nanos }`.
- **Files:** `src/sse/mod.rs`, `src/sse/chunk.rs`, `src/sse/parser.rs`, `tests/sse_parser_test.rs`
- **Depends:** Chunk 1
- **Acceptance:** Feed a captured/representative vLLM-style SSE byte stream containing
  `reasoning_content` deltas, `content` deltas, a final `usage` chunk, and `[DONE]`; assert exact
  counts of reasoning/content/usage and correct ordering. A second test injects a malformed
  frame mid-stream and asserts it is skipped without panic.

#### Chunk 4 · Prompt generator + client-side tokenizer
- **Build:** `prompt/generator.rs` — port both prototypes: `short` (~50 tok fixed prompt),
  `long` (pad to `target_tokens`; prefer a token-stable filler like repeated `" apple"` from
  `llmspeedtest2.py`, with the sentence-rotation fallback from `llmspeedtest.py`), and `nocache`
  (prepend a unique random hex/uuid prefix to bust the KV cache). `prompt/tokenizer.rs` — load an
  optional HF `tokenizer.json` via the `tokenizers` crate to count prompt tokens exactly; if no
  file is supplied, fall back to `chars/4` and flag the count `estimated`.
- **Files:** `src/prompt/mod.rs`, `src/prompt/generator.rs`, `src/prompt/tokenizer.rs`, `tests/prompt_test.rs`
- **Depends:** Chunk 1
- **Acceptance:** A 2000-token long prompt generates within ±5% of target (when a tokenizer is
  available) or ±10% (chars/4 fallback); `nocache` yields a different prefix each call; tokenizer
  count for a known string matches a hand-counted reference.

#### Chunk 5 · Stream worker (single HTTP/2 SSE client)
- **Build:** `client/stream.rs` — `StreamWorker` that POSTs to `/v1/chat/completions`
  (`stream:true`, `stream_options.include_usage:true`, `temperature:0`, `max_tokens`) via
  `reqwest` (HTTP/2, connection pooling, socket timeout). It records `T0..Tn` via the quanta
  clock, streams response bytes into the Chunk 3 parser, and emits `(ParsedFrame, timestamps)`
  over a bounded `tokio::sync::mpsc` channel. Handle: non-SSE plain-JSON fallback (extract
  usage + single content chunk), HTTP non-200, connection refused, read timeout, and
  premature `ChunkedEncoding`-style close (mark `premature_exit`).
- **Files:** `src/client/mod.rs`, `src/client/stream.rs`
- **Depends:** Chunk 2, Chunk 3
- **Acceptance:** Against a local mock SSE server (a tiny `tokio` test harness or `nginx`-style
  fixture), the worker records T0/T2/T3, delivers correctly-classified frames on the channel,
  captures `usage`, and shuts down cleanly on `[DONE]` and on simulated early close.

#### Chunk 6 · Engine core + metric synthesis (single stream)
- **Build:** `metrics/engine.rs` — `EngineCore` consumes the worker channel: aggregates packet
  deltas, classifies Reasoning/Content/Control, tracks inter-token latencies into the
  `LatencyHistogram`, and computes the blueprint §7 formulas: TTFT, PP throughput
  (`prompt_tokens/TTFT`), TG speed (`(N−1)/(T_end−T_first)`), ITL jitter (std-dev of deltas),
  MTP `η = N_tokens/N_packets`, and the prefix-cache heuristic (`HIT` if
  `TTFT ≤ 0.15×TTFT_cold_baseline`). It also splits a reasoning-phase vs writing-phase
  throughput. `metrics/state.rs` — an `ArcSwap<MetricsSnapshot>` double buffer the UI reads
  lock-free; the engine pushes a unified snapshot per batch.
- **Files:** `src/metrics/engine.rs`, `src/metrics/state.rs`, `tests/metrics_test.rs`
- **Depends:** Chunk 2, Chunk 5
- **Acceptance:** Feed a synthetic frame stream with known timestamps/token counts; assert
  computed TTFT, PP, TG, MTP, and jitter match hand-calculated values within tolerance; the
  cache heuristic returns HIT for a fast TTFT and MISS for a slow one; the ArcSwap snapshot is
  readable without locking the engine.

#### Chunk 7 · Single-stream CLI parity (headless mode)
- **Build:** `config.rs` — a `clap`-derived `Config` that is the **superset** of both Python
  CLIs: `--url --model --mode {short,long} --tokens --iterations --api-key --timeout --nocache
  --json --verbose --no-color --tokenizer <path>` (plus the new `--export` flag (Chunk 13)).
  `main.rs` — a **headless** mode (no TUI) that runs N iterations, prints the result box
  (mirroring `llmspeedtest.py`'s `print_result_box`) and/or `--json` (same fields as
  `llmspeedtest.py`'s `output_json`), a multi-iteration summary, and exits 1 if all runs failed.
- **Files:** `src/config.rs`, `src/main.rs`, `src/engines/mod.rs`, `src/engines/speed.rs`
- **Depends:** Chunk 4, Chunk 5, Chunk 6
- **Acceptance:** `cargo run -- --url <mock> --json` emits JSON with the same metric keys as
  `llmspeedtest.py --json`; a single run against the mock completes; a dead endpoint yields exit
  code 1; `--nocache` changes the sent prefix.

---

### PHASE 2 — Ratatui Interface & Reactive Architecture

#### Chunk 8 · TUI app skeleton + decoupled 60Hz event loop
- **Build:** `ui/app.rs` — `App` state machine with `View` enum
  `{ Live, Concurrency, Needle, History, Config }` and key handling (1-5 switch views, `q`
  quit, `Space` pause/resume, `+` step concurrency, `n` new needle, `e` export — per blueprint
  §6 footer). `ui/event.rs` — a `crossterm` event loop with a 60Hz `tokio` tick that pulls the
  `ArcSwap` snapshot and drives `ratatui` rendering; **strictly decoupled** from the worker pool
  (a dropped frame or resize must not touch timing). `ui/theme.rs` — the ANSI color palette
  (green/yellow/red/cyan/magenta per the mockups).
- **Files:** `src/ui/mod.rs`, `src/ui/app.rs`, `src/ui/event.rs`, `src/ui/theme.rs`, `src/ui/views/mod.rs`
- **Depends:** Chunk 1, Chunk 6
- **Acceptance:** `cargo run` (TUI mode) opens the dashboard; keys 1-5 switch views; `q` exits
  cleanly; window resize re-lays out without panic; running a stream in the background while
  resizing/dropping frames does not alter measured TTFT (verified against the headless numbers).

#### Chunk 9 · View 1 — Live Monitor & Telemetry dashboard
- **Build:** `ui/views/live.rs` implementing the blueprint §6 View 1 layout: status bar
  (endpoint, model, mode); top-left key metrics (aggregate t/s, VRAM capacity bar, GPU clock,
  J/token); top-right ITL latency histogram (p50/p90/p99 sparkline); mid-panel stream matrix
  (per-stream ID, type, state, PP/TG tokens, TTFT, gen speed, MTP rate, progress bar);
  bottom-panel rolling throughput canvas chart; and a scrolling log/event stream.
- **Files:** `src/ui/views/live.rs`
- **Depends:** Chunk 8
- **Acceptance:** With injected/synthetic metrics, all five panels render and update at 60Hz;
  the histogram and rolling chart reflect changing data; the stream matrix shows PP vs TG split
  and MTP ratio per stream.

---

### PHASE 3 — Concurrency Engine & Load Curves

#### Chunk 10 · Multi-stream pool + concurrency ladder sweep
- **Build:** `client/pool.rs` — spawn/manage N `StreamWorker`s (from Chunk 5) with independent
  HTTP/2 streams, fanning their channels into a single bounded `tokio::mpsc` into the
  `EngineCore`. `engines/concurrency.rs` — a `Sweep` that steps a configurable concurrency
  ladder (default `1→2→4→8→16→32→64`), running a fixed workload at each level and collecting
  per-level **aggregate tokens/sec** (all streams) and **client-perceived p90 TPOT**.
- **Files:** `src/client/pool.rs`, `src/engines/concurrency.rs`
- **Depends:** Chunk 5, Chunk 6
- **Acceptance:** A sweep against the mock server produces one (concurrency, aggregate_tps,
  p90_tpot) record per ladder step; no connection leaks (fd count stable after shutdown);
  aggregate throughput and p90 are computed across all concurrent streams.

#### Chunk 11 · Knee-point detection + View 2 (Concurrency Matrix)
- **Build:** Extend `engines/concurrency.rs` with **knee detection**: find the concurrency level
  where aggregate throughput plateaus while p90 TPOT spikes (memory-bandwidth-bound →
  compute-bound transition); emit the "Optimal Operational Envelope" (recommended sweet spot).
  `ui/views/concurrency.rs` — the View 2 heatmap/matrix: X = concurrency (1…128), Y = latency vs
  throughput, with the knee/envelope highlighted (blueprint §6 View 2).
- **Files:** `src/engines/concurrency.rs`, `src/ui/views/concurrency.rs`
- **Depends:** Chunk 8, Chunk 10
- **Acceptance:** Given synthetic sweep data with a clear inflection, the detected knee matches
  the injected level; View 2 renders the matrix and highlights the optimal envelope.

---

### PHASE 4 — Capability Suites, Hardware Telemetry & Persistence

#### Chunk 12 · Persistence — SQLite storage engine
- **Build:** `storage/schema.sql` — the exact DDL from blueprint §8
  (`benchmark_sessions`, `stream_metrics`, `needle_evaluations`). `storage/db.rs` — open/create
  the DB at the platform data dir (`dirs::data_dir()/crucible/benchmarks.db`; Windows
  `%APPDATA%\crucible`), run migrations, and provide `insert_session`, `insert_stream_metrics`,
  `insert_needle`, and query helpers. `storage/models.rs` — `serde` row structs matching the
  schema. After every completed run (headless or TUI), persist a session + its stream metrics.
- **Files:** `src/storage/mod.rs`, `src/storage/schema.sql`, `src/storage/db.rs`, `src/storage/models.rs`, `tests/storage_test.rs`
- **Depends:** Chunk 6
- **Acceptance:** A completed run creates a `benchmark_sessions` row + `stream_metrics` rows
  matching the schema; reopening the DB and querying returns the stored rows; the file lands in
  the correct platform data dir.

#### Chunk 13 · Export — JSON / Markdown / CSV
- **Build:** `storage/export.rs` — three exporters over stored/live data: **zero-alloc JSON**
  (for CI/CD regression gating), **GitHub-Flavored Markdown** tables (for PRs/issues/READMEs),
  and **raw CSV** (all packet arrival times + inter-token intervals for Python/R/Grafana). Wire
  a `--export {json,md,csv} [path]` flag and the TUI `e` key to it.
- **Files:** `src/storage/export.rs`, `src/config.rs`, `src/main.rs`
- **Depends:** Chunk 12
- **Acceptance:** `--export json` produces valid, parseable JSON; `--export md` produces a
  GFM table with the key metrics; `--export csv` includes per-packet timestamps and ITL values.

#### Chunk 14 · View 4 — Historical Comparison & Regression Diffing
- **Build:** `ui/views/history.rs` — side-by-side comparison of two stored runs (e.g.,
  `vLLM v0.6.x` vs `v0.7.x`, `llama.cpp b3200` vs `b3300`); compute and display delta metrics:
  % change in TTFT, tokens/sec, and speculative-verification (MTP) rate; highlight gains
  (green) vs regressions (red).
- **Files:** `src/ui/views/history.rs`
- **Depends:** Chunk 8, Chunk 12
- **Acceptance:** Selecting two stored sessions renders the side-by-side table with correct
  signed deltas and color-coding for improvement/regression.

#### Chunk 15 · Engine C1 — Needle-In-A-Haystack + View 3
- **Build:** `engines/capability/niah.rs` — generate synthetic long-context documents scaled to
  `2k/4k/8k/16k/32k/64k/128k` tokens; insert a unique cryptographically-random key-value "needle"
  at each depth `0%..100%` in `10%` steps; run the retrieval query; measure **(1) accuracy**
  (does the model retrieve the needle cleanly?) and **(2) prefill degradation** (TTFT/prefill
  speed decay as context scales). `ui/views/needle.rs` — the N×M grid (context size × depth)
  with ANSI color-coding: green = accurate + nominal prefill, yellow = accurate + throttled
  prefill, red = retrieval failed/hallucinated (blueprint §6 View 3).
- **Files:** `src/engines/capability/mod.rs`, `src/engines/capability/niah.rs`, `src/ui/views/needle.rs`
- **Depends:** Chunk 4, Chunk 5, Chunk 6
- **Acceptance:** The NIAH runner executes a size×depth matrix against the endpoint; retrieval is
  verified by checking the needle value is present/extractable in the response; prefill speed is
  recorded per cell; View 3 renders the color-coded grid.

#### Chunk 16 · Engines C2 & C3 — Deterministic Reasoning + Structured Output
- **Build:** `engines/capability/reasoning.rs` — a standardized bank of deterministic logic/math/
  code-generation challenges; validate responses against strict checkers (AST parse for code,
  regex / exact-value match for math/logic) to report a deterministic **pass/fail accuracy
  score** alongside speed. `engines/capability/structured.rs` — run generation under a
  constrained JSON schema / grammar (GBNF / tool-call format) and quantify the **speed penalty**
  of grammar validation vs unrestricted generation.
- **Files:** `src/engines/capability/reasoning.rs`, `src/engines/capability/structured.rs`
- **Depends:** Chunk 5, Chunk 6
- **Acceptance:** The reasoning bank yields a pass/fail accuracy score (e.g., N/M solved); a
  known-correct response passes and a known-wrong one fails; the structured runner reports the
  t/s under grammar constraint vs the free-form baseline (penalty %).

#### Chunk 17 · Engine D — Hardware & Energy Efficiency Profiler
- **Build:** `hw/nvml.rs` — a 100ms polling worker (feature-gated `nvml`) reading NVIDIA VRAM
  (weights vs KV-cache), SM clock, temperature, and instantaneous power (mW) via
  `nvml-wrapper`; plus `sysinfo` for cross-platform CPU/RAM. Degrades to `"N/A"` (no panic) when
  no GPU/driver is present. `engines/hardware.rs` — correlate the power trace `P(t)` with active
  token output windows and compute the **Silicon Efficiency Metric**
  `Joules/Token = ∫P(t)dt / Total_Generated_Tokens`; warn when approaching VRAM fragmentation
  thresholds.
- **Files:** `src/hw/mod.rs`, `src/hw/nvml.rs`, `src/engines/hardware.rs`
- **Depends:** Chunk 2, Chunk 6
- **Acceptance:** On a machine with an NVIDIA GPU, VRAM + watts are sampled at 100ms and
  joules/token is computed for a run; on a machine without a GPU, all hardware fields report
  `"N/A"` and the app runs normally; a fragmentation warning fires as VRAM usage approaches the
  threshold.

#### Chunk 18 · View 5 (Config) + full engine integration
- **Build:** `ui/views/` Config view — editable target URL, model, mode, tokens, concurrency
  ladder, tokenizer path, feature toggles (nvml, cache-bypass), and per-engine enable switches.
  Wire **all** engines (A/B/C/D) into the `App` so a single run can orchestrate any combination;
  ensure the TUI, headless, and export paths all draw from the same `EngineCore` + storage.
- **Files:** `src/ui/app.rs`, `src/ui/views/` (new `config.rs`), `src/main.rs`, `src/engines/mod.rs`
- **Depends:** Chunks 9, 11, 14, 15, 16, 17
- **Acceptance:** The Config view edits and persists settings; a single command/run can trigger
  any subset of engines A–D; TUI, headless, and `--export` all reflect the same underlying data.

#### Chunk 19 · Final integration, static binary & docs
- **Build:** Produce the standalone **static release binary** (`cargo build --release`,
  `lto=thin`, `codegen-units=1`); write `README.md` (usage, all CLI flags, TUI key map, engine
  descriptions, export formats, data-dir location); add an end-to-end smoke test that runs a
  full pipeline (single-stream + a small sweep + NIAH + export) against the mock server; final
  `clippy`/`fmt` pass.
- **Files:** `README.md`, `tests/` (e2e), `Cargo.toml` (release profile confirm)
- **Depends:** All prior chunks
- **Acceptance:** `cargo build --release` yields a single static binary that runs headless, TUI,
  and export modes; the e2e smoke test passes; `cargo clippy` and `cargo fmt --check` are clean;
  README documents every flag and view.

---

## 5. Cross-Cutting Concerns

- **Measurement isolation (top priority):** timing lives only in the worker rings + `quanta`;
  the TUI reads an `ArcSwap` snapshot and the SQLite flusher runs in the background. No UI or
  persistence code may call into the timing path.
- **Feature gating:** `nvml` is a cargo feature (default-off) with runtime detection; `tokenizers`
  is optional at runtime via `--tokenizer`. Both must degrade gracefully, never crash.
- **Testing strategy:** unit tests for parser/metrics/prompt/storage (deterministic, no network);
  a local **mock SSE server** (tokio test harness) for stream/concurrency/integration tests so the
  suite runs offline; one e2e smoke test in Chunk 19.
- **Config as single source of truth:** one `clap` `Config` (superset of both Python CLIs) feeds
  headless, TUI, and export paths identically.
- **Data dir:** `dirs::data_dir()/crucible/` (Linux `~/.local/share/crucible`, Windows
  `%APPDATA%\crucible`); DB file `benchmarks.db`.

## 6. Definition of Done (project-level)

1. `cargo build --release` → single static binary; `cargo test` green; `clippy`/`fmt` clean.
2. Headless mode matches `llmspeedtest.py` metrics (TTFT/PP/TG/MTP) against the same endpoint.
3. TUI renders all five views at 60Hz without perturbing measurements.
4. Engines A–D all produce the blueprint §7 metrics; results persist to SQLite and export to
   JSON/MD/CSV.
5. Graceful degradation verified on a machine with no NVIDIA GPU and no tokenizer file.
