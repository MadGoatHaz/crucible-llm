# Crucible LLM — Architecture & Measurement Methodology

This document explains how each benchmark works, what it measures, and why. Understanding the methodology helps you interpret results correctly.

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CRUCIBLE-LLM                                   │
│                                                                             │
│  ┌──────────┐    ┌──────────┐    ┌────────────────────────────────────────┐  │
│  │   User   │───▶│ TUI /    │───▶│         Benchmark Sequence             │  │
│  │  (keys)  │    │   CLI    │    │  (A → B → C1 → C2 → C3 → D → F, one at │  │
│  └──────────┘    └──────────┘    │   a time, tokio::spawned task)         │  │
│                          │       └──────┬───────────────┬────────────────┘  │
│                          ▼              ▼               ▼                     │
│                   ┌────────────┐  ┌──────────┐  ┌──────────────┐            │
│                   │  Config    │  │ Engines  │  │  ProgressBus │            │
│                   │ (URL, model│  │ A,B,C1,  │  │ (10 Hz →     │            │
│                   │  tokens,   │  │ C2,C3,D,F│  │  SeqState)   │            │
│                   │  ladder)   │  └────┬─────┘  └──────────────┘            │
│                   └────────────┘       │                                    │
│                                        ▼                                    │
│                    ┌─────────────────────────────────────┐                  │
│                    │        Stream Worker Pool            │                  │
│                    │  (reqwest, HTTP/1.1 or HTTP/2,      │                  │
│                    │   /v1/chat/completions, stream:true) │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────────────┐                  │
│                    │         SSE Parser                   │                  │
│                    │  (incremental line-buffer state     │                  │
│                    │   machine, zero-alloc hot path)     │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                          │
│                                   ▼                                          │
│                    ┌─────────────────────────────────────┐                  │
│                    │      Timing (quanta)                │                  │
│                    │  T0 → T1 → T2 → T3 → Tn            │                  │
│                    │  (RDTSC / System Counter, ns)       │                  │
│                    └──────────────┬──────────────────────┘                  │
│                                   │                                          │
│                    ┌──────────────┴──────────────────────────┐              │
│                    │                                         │              │
│                    ▼                                         ▼              │
│         ┌───────────────────┐                    ┌───────────────────┐     │
 │         │  Metrics State     │                    │  Hardware Poller  │     │
 │         │  (ArcSwap double-  │                    │  (100 ms GPU      │     │
 │         │   buffered)        │                    │  backend + sysinfo)│     │
│         └────────┬──────────┘                    └────────┬──────────┘     │
│                  │                                         │               │
│                  ▼                                         ▼               │
│         ┌───────────────────┐                    ┌───────────────────┐     │
│         │   TUI Render       │                    │  Energy Profile   │     │
│         │   (60 Hz, lock-   │                    │  (trapezoidal     │     │
│         │    free reads)    │                    │   ∫P(t)dt)        │     │
│         └───────────────────┘                    └────────┬──────────┘     │
│                                                            │               │
│                                                            ▼               │
│         ┌───────────────────┐                    ┌───────────────────┐     │
│         │  SQLite Storage    │◀───────────────────│  Export           │     │
│         │  (benchmarks.db)   │                    │  (JSON / MD / CSV)│     │
│         └───────────────────┘                    └───────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The key architectural principle is **measurement isolation**: the render loop, the hardware poller, and the storage layer never touch the timing path. All timestamp capture happens in the stream workers (quanta RDTSC), and all downstream consumers (TUI, metrics, storage) read from lock-free snapshots. A frame drop in the terminal or a slow disk write cannot perturb a measurement.

## Core Measurement Pipeline

### How I Talk to the Server

All engines use the OpenAI-compatible `/v1/chat/completions` endpoint with `stream: true`. The request body includes:

```json
{
  "model": "<model-name>",
  "messages": [{ "role": "user", "content": "<prompt>" }],
  "stream": true,
  "stream_options": { "include_usage": true },
  "temperature": 0,
  "max_tokens": 256
}
```

Requests are sent via `reqwest` (a connection-pooled HTTP client). For local endpoints (plain HTTP), this is HTTP/1.1 with chunked transfer encoding. For remote endpoints (TLS), this is HTTP/2. The server responds with Server-Sent Events (SSE) — one `data:` frame per token (or token group), terminated by `data: [DONE]`.

The `include_usage: true` flag is critical: it asks the server to append a `usage` object (with `prompt_tokens` and `completion_tokens`) to the final SSE frame, giving me authoritative token counts without needing a client-side tokenizer.

If a server ignores `stream: true` and returns a plain JSON completion object instead, the worker detects this (the first bytes do not start with `data:`) and falls back to parsing the single-object response. The measurement is still valid, but there is only one "token frame" instead of a stream.

### Timing Methodology

I use `quanta` for high-resolution monotonic clock timestamps. `quanta` reads the CPU's Time-Stamp Counter (TSC) directly via RDTSC on x86, falling back to the OS reference clock (`clock_gettime`) where the CPU lacks invariant counters. No syscalls on the fast path — deltas are computed from raw cycle counts converted to nanoseconds.

Each stream worker records five lifecycle milestones:

```
T0 ────────── T1 ──── T2 ──── T3 ──── ... ──── Tn
 │             │       │       │                │
 │             │       │       │                │
 │             │       │       │                └─ Stream close
 │             │       │       │                   ([DONE] or EOF)
 │             │       │       └─ First token frame
 │             │       │          decoded (Content or
 │             │       │          Reasoning chunk)
 │             │       └─ First body byte received
 │             │            (TTFB)
 │             └─ Request write complete
 │                (send() resolved on
 │                 response headers)
 └─ Socket connection start
```

| Timestamp | Meaning | When captured |
|-----------|---------|---------------|
| **T0** | Socket connection start | Just before `reqwest` sends the POST |
| **T1** | Request write complete | `send()` resolves (response headers received) |
| **T2** | First byte received | First body chunk arrives (TTFB) |
| **T3** | First token decoded | First `Content` or `Reasoning` SSE frame parsed |
| **Tn** | Stream close | `[DONE]` frame, clean EOF, or premature close |

From these, I derive:

- **TTFT** = T3 − T1 — time from request dispatched to first token arrival. This is the perceived "responsiveness" of the server.
- **TTFB** = T2 − T1 — time from request dispatched to first byte. This is the network + prefill hand-off window (used for logging, not the primary metric).
- **ITL** (Inter-Token Latency) = T(n) − T(n−1) for each consecutive pair of token frames. This measures generation smoothness.
- **Generation speed (TG)** = `completion_tokens / (Tn − T3)` — tokens/sec during the decode phase only (the prefill is excluded).
- **Prompt throughput (PP)** = `prompt_tokens / TTFT` — tokens/sec during the prefill phase.
- **MTP efficiency (η)** = `completion_tokens / content_chunks` — tokens per SSE data frame. A value of 1.0 means standard single-token generation; values above 1.0 indicate speculative decoding or multi-token prediction is active on the server.

### Token Counting

Every completion-token count flows through one canonical function, `authoritative_tokens()` (`client/stream.rs`), with a strict priority:

1. **Server-reported (primary, authoritative)** — `usage.completion_tokens` from the terminal `usage` frame (requested via `stream_options.include_usage: true`). When the server provides it, it is the count: it reflects the server's own tokenization, so it matches what the backend reports.

2. **Re-tokenization (fallback)** — when no `usage` frame arrived (an aborted stream, or a non-compliant proxy), the full reasoning + content text is re-tokenized: the model's exact HuggingFace `tokenizer.json` when one is supplied, else a `chars/4` estimate (flagged `estimated`).

**The raw SSE frame tally is never used as a token count.** Batched servers (vLLM's multi-token prediction / speculative decoding) pack ~2.4 tokens into a single `data:` frame, so counting frames understates true output by 30–40 %. Frame counts survive only as decode-rate evidence on the wire, never as the token numerator.

For **prompt token counting** (sizing the input), an optional HuggingFace `tokenizer.json` gives exact counts; without it I estimate at `chars/4` (the common BPE heuristic) and flag the result `[ESTIMATED]`. This is sufficient for Engine A's short/long modes and for Engine C1's haystack sizing (which pads with a token-stable `" apple"` filler when a tokenizer is available).

### SSE Parsing

The SSE parser is a manual, incremental line-buffer state machine — zero external dependencies on the hot path. It handles:

- `data:` prefix (with or without the single leading space the SSE spec allows)
- Multi-line `data:` payloads (joined with `\n`)
- Blank-line frame termination
- `event:` / `id:` / `retry:` / `:`-comment lines (ignored)
- UTF-8 BOM stripping (once, at stream start)
- `[DONE]` end-of-stream marker
- Malformed JSON (counted and skipped — the parser never panics)

Each parsed frame is classified into one of four `Chunk` types: `Content(text)`, `Reasoning(text)`, `Usage({prompt_tokens, completion_tokens})`, or `Control`. The parser returns frames through an internal ring buffer, so a steady stream causes no per-frame heap allocation. The parser leaves `t_nanos` at zero by contract — the stream worker stamps arrival times with the quanta clock after each `feed` call (measurement-isolation invariant).

### Run Completion & the Metrics Freeze

The metrics pipeline has a **freeze-on-completion** behavior. While a run is active, the stream workers keep publishing fresh `MetricsSnapshot`s and the hardware poller keeps sampling, so the dashboard shows live, moving numbers. The moment a run **completes**, the `MetricsState` is **frozen** (`MetricsState::freeze()`):

- the render loop keeps showing the **final** snapshot — the numbers stop drifting, so the values you read at the end of a run are stable and reproducible;
- the 100 ms hardware poller **goes idle** (it checks the freeze latch each tick and skips the sample/merge work while frozen), so it stops perturbing the system;
- the frozen state persists until the **next run starts**, which calls `MetricsState::unfreeze()` to re-arm the pipeline.

This applies to every run shape — the full `A → B → C1 → C2 → C3 → D → F` sequence and a standalone `n`-key NIAH run. The freeze is a **read-side** concern (it only gates how often snapshots are refreshed); it never touches the `quanta` timing path, so the measurement-isolation invariant holds.

## Engine A: Speed

### What It Measures

Single-stream generation performance. This answers: "How fast does the model talk to one person?" It isolates the server's prefill and decode performance from concurrency effects.

### Methodology

1. **Generate a prompt** of the target token length:
   - `short` mode: a brief ~100-token prompt (measures TTFT responsiveness for chat-like use)
   - `long` mode: a prompt padded to the configured token target (measures sustained throughput)
   - `nocache` variant: prepends a unique UUID prefix to bust the server's KV-cache, ensuring a cold prefill
2. **Send a single streaming request** to the server (`temperature: 0`, `max_tokens: 256`)
3. **Record T0, T1, T2, T3, and every Tn** via the quanta monotonic clock
4. **Parse all SSE frames** through the incremental parser, classifying each as Content / Reasoning / Usage / Control
5. **Calculate** on stream completion:
   - TTFT (T3 − T1)
   - ITL distribution: p50, p90, p99, p99.9 (HdrHistogram)
   - Generation speed: `completion_tokens / (Tn − T3)`
   - Prompt throughput: `prompt_tokens / TTFT`
   - MTP efficiency: `completion_tokens / content_chunks`
6. **Repeat for N iterations** (default: 1; the same prompt is reused across iterations)
7. **Report**: average, min, and max across all iterations

### What the Numbers Mean

| Metric | Interpretation |
|--------|---------------|
| **TTFT** | Time from request to first token. >1s suggests queuing, cold cache, or slow prefill. <100ms is snappy. |
| **ITL p50** | Typical gap between tokens. 30ms ≈ 33 t/s. This is your "reading speed." |
| **ITL p99** | Worst 1% of token gaps. >100ms means occasional "stuttering" the user will perceive. |
| **Generation t/s** | Raw decode speed. 30+ t/s = comfortable reading. 100+ = fast. |
| **Prompt t/s** | Prefill speed. Critical for long-context use. A 4k-token prompt at 1000 t/s prefills in 4ms. |
| **MTP η** | >1.0 indicates speculative decoding is active. 1.8× means the server is generating ~2 tokens per frame. |

### Why `temperature: 0`

Deterministic output ensures that repeated iterations measure the *server*, not the *model's sampling randomness*. The same prompt produces the same tokens, so variance across iterations reflects scheduling, cache state, and thermal effects — not generation quality.

## Engine B: Concurrency

### What It Measures

How your server scales with multiple simultaneous users. This answers: "How many people/agents can use this model at once before each one notices?"

### Methodology

1. **Build a `WorkerPool`** — a shared connection-pooled `reqwest` client with N `StreamWorker` instances, each on an independent HTTP stream
2. **Step through the concurrency ladder** (default: `1, 2, 3, 4, 8, 12, 16, 24, 32`):
   - Granular at the low end (where home setups operate), capped at 32 (64 concurrent streams is unrealistic for a home rig)
   - Custom ladders are accepted (normalized: positive, deduped, ascending)
3. **At each level**: spawn N identical streams simultaneously, all sending the same prompt with `max_tokens`
4. **Wait for all N streams to complete** (with three safety bounds, see below)
5. **Record per level**:
   - Aggregate t/s = total tokens across all streams ÷ level wall time
   - Per-stream t/s = aggregate t/s ÷ N
   - ITL percentiles (p50, p90, p99) pooled across all streams
   - TTFT percentiles (p50, p90) — one sample per stream
   - Completion / failure / timeout counts
   - Per-stream detail rows (for the TUI stream matrix)
6. **Repeat until all ladder levels complete**

### Safety Bounds (The Freeze Fix)

Each level is bounded three ways so the sweep can never hang on a single bad level:

1. **Per-worker timeout**: a worker that does not finish within the configured timeout is killed (socket closes), and its partial tokens are collected. Reported as a `WorkerTimeout` failure.
2. **Stall detection** (checked every 5 seconds):
   - *TTFB phase* (no token received yet): warned after 90s (or `worker_timeout/2` when larger). At high concurrency, the server batch-schedules — a worker waiting in the queue is *not* stalled.
   - *Inter-token phase* (≥1 token received, then silence): warned after 30s. This *is* a real stall.
   - Each warning fires once per worker. The kill comes from #1.
3. **Step-level timeout**: if the whole level runs past its budget (`max(worker_cap, 3× previous step's wall time)`), it is warned. At 2× the budget, the level is **aborted** — the supervisor task is dropped (killing all its worker children), partial results are collected, the level is marked `degraded`, and the sweep continues to the next step.

### Sweet Spot Detection

I calculate four thresholds from the sweep curve:

| Threshold | Definition | Meaning |
|-----------|-----------|---------|
| **Practical Sweet Spot** | Highest level where per-stream t/s ≥ 30 (29.4 t/s effective, 2% margin) | Comfortable for chat, coding agents, RAG pipelines. **This is what I recommend — and the load Engine F runs at.** |
| **Maximum Usable** | Highest level where per-stream t/s ≥ 15 | Minimal but functional. Noticeably slower, but workable. |
| **Unusable From** | First level where per-stream t/s < 15 | Interactive use becomes impractical. |
| **Throughput Knee** | First level where aggregate t/s gain ≤ 5% AND p90 TPOT grows ≥ 2× | Pure capacity limit: the GPU has transitioned from memory-bandwidth-bound to compute-bound. |

The **practical sweet spot** is the headline recommendation. Aggregate throughput is misleading: 64 users at 2.6 t/s each is not "fast" — it is "slow for everyone." What matters for real-world use is what each individual stream gets.

### What the Numbers Mean

- **If your sweet spot is 8**: your server comfortably serves 8 simultaneous users. A 9th user will notice degradation.
- **Per-stream t/s drops** as you add users because the GPU's compute and memory bandwidth are shared across all active streams.
- **TTFB increases at higher concurrency** because requests queue on the server's batch scheduler. This is normal and expected — it is the server working, not the server failing.
- **The concurrency curve** (View 2) plots aggregate t/s vs. concurrency on a log₂ x-axis, with the sweet spot marked `●` (green) and the knee marked `▲` (red).

## Engine C1: NIAH (Needle In A Haystack)

### What It Measures

Long-context retrieval accuracy. Can the model find specific information buried in a large document? This predicts RAG, chat-history, and long-document QA performance.

### Methodology

The matrix is **7 context sizes × 11 depths = 77 cells**:

- **Sizes**: 2k, 4k, 8k, 16k, 32k, 64k, 128k tokens
- **Depths**: 0%, 10%, 20%, 30%, 40%, 50%, 60%, 70%, 80%, 90%, 100%

For each cell:

1. **Generate a random needle**: a unique, cryptographically random 128-bit value rendered as `{16 hex}-{16 hex}` (e.g., `9F3A21BC44D08E51-7C21E8A3`). The key is always "the secret code." A coincidental occurrence in a model response is effectively impossible — a substring hit is a genuine retrieval.

2. **Build the haystack document**:
   - A synthetic filler text of the target token size, constructed from a rotating sentence bank
   - With a tokenizer: padded with `" apple"` filler (exactly 1 token each for most BPE vocabularies) until the count reaches the target. The deficit is distributed around the needle in proportion to its depth, preserving the insertion position.
   - Without a tokenizer: the sentence-rotation filler's `chars/4` length is used, and the count is flagged `estimated`.
   - The needle sentence is inserted at the depth position: `IMPORTANT: the secret code is {value}. Remember the secret code exactly as written.`
   - The retrieval query is appended: `What is the secret code? Answer with the value only.`

3. **Send the full document as a single streaming request** (`max_tokens: 64` — the answer is the value only, so a short budget keeps the matrix fast)

4. **Check retrieval**: does the model's response contain the exact 33-character needle value? Strict substring match — no fuzzy matching. A one-character difference is a failure.

5. **Record TTFT and prefill throughput** for the cell

6. **Classify the cell** (after the full matrix is complete):
   - **Failed (red)**: retrieval missed, or no TTFT recorded
   - **Nominal (green)**: retrieval succeeded, and TTFT is within 2× the linear expectation (anchored on the smallest context size at the same depth)
   - **Throttled (yellow)**: retrieval succeeded, but TTFT exceeds 2× the linear expectation — the server is struggling with the context length

Cells run **sequentially** (one stream at a time) so each cell measures the endpoint alone — no queueing cross-talk, and the prefill-degradation curve is not polluted by concurrency.

### What the Numbers Mean

- **100% at 2k–8k**: the model handles normal conversation lengths perfectly.
- **Drops at 32k+**: the model starts losing track of information buried in the middle of long documents.
- **The "reliable limit"** is the largest context size where the pass rate across all depths is ≥ 80%.
- **Practical interpretation**: if your RAG system sends 16k-token contexts and the model only reliably retrieves to 8k, you will get hallucinated answers. Limit your context windows to the reliable threshold, or choose a model with better long-context training.
- **Prefill degradation**: the yellow (throttled) cells show where the server's prefill speed falls off as context grows — a separate concern from retrieval accuracy.

## Engine C2: Reasoning

### What It Measures

Raw logical problem-solving ability. Independent of speed — this answers "how smart is the model?" The score is deterministic and reproducible: no LLM-as-judge, no fuzzy scoring.

### Methodology

A bank of **13 self-contained challenges** across three categories:

| Category | Count | Examples |
|----------|-------|----------|
| **Math** | 5 | `27 × 43`, `3/4 of 240`, sequence continuation, percentage increase, sum of 1..100 |
| **Logic** | 5 | Syllogism validity, parity check, knights-and-knaves, boiling eggs (parallel vs. serial), premise-based deduction |
| **Code** | 3 | Fibonacci function, string reversal, GCD (Euclidean algorithm) — all in Rust |

Each challenge has exactly one correct answer, validated by a **strict, deterministic checker**:

- **Math**: the *last* numeric value in the response must equal the expected value (tolerance: 1e-6). The prompts ask for "the number only," so the last number is the answer.
- **Logic**: the *last* whole-word yes/no verdict (case-insensitive) must match the expected verdict. `none`, `normal`, `yeses` never match — only exact `yes`/`no` word boundaries.
- **Code**: the response must contain a fenced code block (``` ``` ```). The block must have balanced, properly nested delimiters (`()` `[]` `{}`) and contain every required structural marker (e.g., `fn fib`, `n: u32`, `-> u32`). This is a *structural* (lexical) check — full AST parsing is a post-v1 extension.

All checkers are pure `std` — no `regex` or parser crates — preserving the zero-dependency design.

The bank runs **sequentially** (one stream at a time, `max_tokens: 512` for code answers), so each challenge measures the endpoint alone.

### What the Numbers Mean

| Score | Interpretation |
|-------|---------------|
| **≥ 90%** (12–13/13) | Strong reasoning. Suitable for complex agent tasks, code generation with logic. |
| **70–89%** (9–11/13) | Adequate. Handles most tasks but will fail edge cases. |
| **< 70%** (≤ 8/13) | Weak. Will struggle with multi-step problems. Consider a larger model or better prompting. |

The per-category breakdown (math N/5, logic N/5, code N/3) shows *where* the model's reasoning breaks down — a model that gets all math right but fails on logic puzzles has a different profile than one that fails at code generation.

## Engine C3: Structured Output

### What It Measures

Can you trust the model to follow format instructions? Critical for API integrations, function calling, and agent tool use. This tests both **compliance** (does the model produce valid, schema-conforming JSON?) and **speed cost** (what does grammar-constrained decoding cost in throughput?).

### Methodology

**4 runs total**: one free-form baseline + three constrained cases of escalating complexity.

Each constrained run sends `response_format: { "type": "json_object" }` in the request body, which server-side constrained-decoding engines (vLLM XGrammar, SGLang, llama.cpp grammar, Outlines) honor to restrict the token space to valid JSON.

| Case | Prompt | Checks |
|------|--------|--------|
| **Simple** | "Return a JSON object with fields: `name` (string), `age` (integer)" | Valid JSON, schema is object, all fields present, types correct |
| **Medium** | "Return a JSON array of exactly 3 objects, each with: `id` (int), `label` (string), `active` (bool)" | Valid JSON, schema is array, array length = 3, all fields present, types correct |
| **Complex** | "Return a JSON object matching: `{user: {name, email}, orders: [{id, total, items: [string]}]}`" | Valid JSON, schema is object, nested structure, all fields present, types correct (including nested array-of-objects) |

The **free-form baseline** (run 1) uses the Simple case's prompt *without* the `response_format` constraint. This gives the unconstrained speed for the penalty calculation.

Each case is classified:

- **Compliant**: every check passes
- **Partial**: valid JSON, but one or more checks fail (e.g., wrong array length, missing field, bad type)
- **Failed**: not valid JSON (cannot be evaluated further)

A ` ```json ` fence is stripped before parsing (models commonly wrap JSON in code fences).

**Speed penalty** = `(free_tps − constrained_tps) / free_tps × 100`. Positive means the grammar constraint costs throughput. This measures the server-side constrained-decoding overhead.

### What the Numbers Mean

| Score | Verdict |
|-------|---------|
| **3/3 compliant** | ✓ Fully suitable for API/agent use. The model reliably follows format instructions. |
| **2/3 compliant** | ⚠ Suitable for simple structures, unreliable for complex schemas. Add output validation. |
| **1/3 compliant** | ⚠ Limited use — only trivial key-value extraction. |
| **0/3 compliant** | ✗ Not suitable for structured output. The model produces free text regardless of instructions. Requires output parsing/fallback. |

The **speed penalty** tells you the cost of constrained decoding: a +5% penalty is negligible; a +30% penalty is significant for high-throughput deployments.

## Engine D: Energy

### What It Measures

Power consumption and energy efficiency of inference. "What does it cost, in joules, to generate a token?"

### Methodology

1. **Hardware poller** (100 ms cadence, runs continuously in the background):
   - **GPU** (auto-detected at startup — NVIDIA NVML / AMD sysfs / Intel Level Zero + sysfs fallback, all through one `GpuBackend` trait): utilization %, instantaneous power (watts), core temperature, VRAM used/total, core + memory clocks, throttle reasons. All visible devices are read and aggregated (VRAM and power summed, clock/temperature/utilization take the max).
   - **Cross-platform CPU/RAM** (`sysinfo`): global CPU usage %, RAM used/total. Available on every host.
   - Each sample is appended to a bounded rolling trace (3600 samples = 6 minutes of power history).

2. **During the benchmark run**, the poller continuously integrates power over time using the **trapezoidal rule**:
   ```
   Energy = Σ (P(i) + P(i+1)) / 2 × Δt(i)
   ```
   where `P(i)` is the instantaneous power in watts and `Δt(i)` is the time between samples. A missing power reading (e.g., `power_usage()` fails on some vGPU hosts) contributes nothing — the integral simply spans from the last known reading to the next.

3. **At run completion**, the energy profile is computed:
   - **Joules/Token** = total energy ÷ total tokens generated
   - **Peak power** (watts)
   - **Average power** (watts) = total energy ÷ span of power-bearing samples
   - **Peak VRAM** (bytes and ratio)
   - **Fragmentation warning** when VRAM occupancy ≥ 90%

### Hardware Support

| Platform | Telemetry | Status |
|----------|-----------|--------|
| **NVIDIA** | NVML API (`nvml-wrapper` crate, `nvml` feature — **on by default**) | Full support: power, VRAM, clocks, temperature, utilization, throttle |
| **AMD** | `amdgpu` sysfs + hwmon (zero-dep, no ROCm stack) | Full support: utilization, VRAM, temperature, power, core + memory clocks |
| **Intel** | Level Zero Sysman (dlopen'd at runtime via `libloading` — no link dependency) with i915/xe sysfs fallback | Full support on Arc: utilization, power, VRAM, clocks, temperature, throttle; sysfs fallback: VRAM, temperature, power (utilization N/A) |
| **Any** | `sysinfo` (CPU/RAM) | Always available |

**Must run ON the machine with the GPU.** Every telemetry source (NVML, sysfs/hwmon, Level Zero) is local — it cannot be queried remotely. Remote users will see `N/A` for all GPU fields.

### The N/A Rule

When no power telemetry is available (no GPU, no driver, feature off), all derived energy metrics are `None` — rendered `N/A`, **never `0.0`**. A `0.0` J/token would claim "free" inference, which is misleading. The `HwSample` struct uses `Option` for every field; a `None` field means "no reading," and the integration and profile functions propagate this correctly.

### What the Numbers Mean

- **Joules/token**: lower = more efficient. A 50W GPU producing 30 t/s uses ~1.67 J/token. A 200W GPU at 120 t/s uses ~1.67 J/token too — the energy efficiency is similar despite very different power and speed.
- **Peak watts**: maximum power draw during the run. Important for cooling and power budget planning.
- **VRAM usage**: the aggregate of model weights + KV cache. A fragmentation warning at 90% means the server may start OOM-killing requests.
- **Comparison**: use J/token to compare quantization levels (Q4 vs. Q8), model sizes (7B vs. 70B), or hardware options (A5000 vs. 4090) on equal footing.

## GPU Monitoring Architecture

All GPU telemetry flows through one vendor-agnostic seam: the **`GpuBackend` trait** (`src/hw/mod.rs`) and a normalized **`GpuSample`** (all-`Option` fields — the N/A rule). NVIDIA, AMD, and Intel each implement the trait; nothing above the seam knows which vendor is present.

### The `GpuBackend` Trait

```
trait GpuBackend: Send + Sync {
    fn vendor(&self) -> &str;              // "NVIDIA" | "AMD" | "Intel"
    fn model(&self) -> String;             // e.g. "GeForce RTX 4090"
    fn poll(&self) -> GpuSample;           // one sample per 100 ms tick
}
```

`GpuSample` carries utilization %, power (W), temperature (°C), VRAM used/total, core + memory clocks, and throttle reasons — every field `Option`, so a backend that can't read a metric emits `None` (rendered `N/A`), never a fake zero and never a panic.

### Detection (NVIDIA → AMD → Intel)

`detect_gpu()` runs at startup and tries the backends in priority order, stopping at the first that initializes:

1. **NVIDIA** — NVML init (the `nvml` feature, on by default; requires the NVIDIA driver).
2. **AMD** — `amdgpu` sysfs probe (`device/vendor == 0x1002`); zero-dependency, no ROCm stack.
3. **Intel** — **Level Zero Sysman first** (`libze_loader.so.1` dlopen'd at runtime via `libloading` — a *runtime* load, never a link dependency, so the binary still runs on machines without the loader), **falling back to the i915/xe sysfs backend** when Level Zero is unavailable.

The single detected backend is shared by the TUI panel, Engine D, and the Config/Setup views; the result is logged to the run log (`[HW] Detected: <name>`). No GPU / no driver → `None`: CPU/RAM-only telemetry, no GPU panel, Engine D auto-off.

### Backends & Data Sources

| Backend | Data source | Metrics |
|---------|-------------|---------|
| NVIDIA (`nvml.rs`) | NVML (`nvml-wrapper` crate) | power, utilization, temperature, VRAM, core clock, throttle |
| AMD (`amd.rs`) | Linux sysfs + hwmon (`amdgpu` driver) | utilization (`gpu_busy_percent`), VRAM, temperature, power, core clock (hwmon, DPM-table fallback), memory clock |
| Intel Arc (`intel_level_zero.rs`) | Level Zero Sysman (`libloading` dlopen, all 16 `zes*` symbols resolved at runtime) | utilization, power (µJ/µs counter deltas), temperature, VRAM, core + memory clocks, throttle reasons |
| Intel fallback (`intel.rs`) | sysfs (i915 / xe drivers) | VRAM (discrete `lmem_*`), temperature, power; utilization N/A (not exposed by Intel sysfs) |

### Integration

- **Hardware poller (100 ms)** polls the detected backend and merges each `GpuSample` into the `MetricsState` snapshot via the same `load → clone → merge → update` path as CPU/RAM — measurement isolation is untouched.
- **Engine D** integrates the power trace (trapezoidal `∫P(t)dt`) into J/token, peak/average power, and peak VRAM; its result now also carries **mean GPU utilization, GPU vendor, and GPU model**.
- **TUI Live view** renders a full GPU telemetry panel — power, utilization, temperature, core/memory clocks, VRAM, throttle reasons, live J/token — **only when a backend is live and publishing**. With no GPU the panel is hidden entirely: no N/A box, no panic.
- **Engine D default**: auto-enabled when a GPU is detected, auto-disabled when not. An explicit user selection (`--engine`, `CRUCIBLE_ENGINE`, or the config file) always wins over the auto-detection.
- **Config / Setup views** show the detected GPU on the Engine D row (e.g. `[✓] Energy — RTX 4090`, or `no GPU`).

## GPU & Power Monitor (View 4)

A dedicated TUI view (`src/ui/views/gpu.rs`) that surfaces the full hardware/energy picture on one screen. It reads a `GpuPowerMonitor` (`src/hw/monitor.rs`) that the 100 ms hardware poller maintains and copies into the lock-free `MetricsSnapshot` — the render loop never blocks and never touches the timing path.

### Power History (1 Hz Sampling)

During a benchmark run, the poller records one **aggregate** `PowerSample` per second (all GPUs summed): timestamp, total power (W), max utilization (%), max temperature (°C), and summed VRAM (GB). The history is bounded to 1800 samples (30 min), with the oldest dropping first. This time-series drives the power-over-time, utilization-over-time, and temperature-over-time charts in View 4.

### Idle Baseline Measurement

Before load begins, the poller collects power samples during a 5-second idle window (`IDLE_WINDOW_SECS`). The mean of those samples becomes the **idle baseline** (`idle_power_w`). Once the benchmark load starts (`start_load()`), the baseline is frozen and the load window opens. **Compute power** is then defined as `total_power_w − idle_power_w` (clamped non-negative) — the power the *work* added, not the machine's floor.

### Energy Integration (Trapezoidal Rule)

Total energy is computed as `∫P(t)dt` over the 1 Hz power history using the trapezoidal rule:

```
Energy (J) = Σ (P(i) + P(i+1)) / 2 × Δt(i)
```

where `Δt(i)` is the time between consecutive samples. This yields:
- **Energy (kWh)** = joules / 3,600,000
- **Cost** = kWh × `rate_per_kwh` (user-configurable, default $0.15)

### Cost Calculation

Energy cost is calculated by separating prefill and decode phases:
- Prefill energy = avg_power(T0→T_first_token) × TTFT_duration
- Decode energy = avg_power(T_first→T_last) × decode_duration
- $/1M input = (prefill_kWh × $/kWh) / (prompt_tokens / 1M)
- $/1M output = (decode_kWh × $/kWh) / (completion_tokens / 1M)

### Multi-GPU Support

The `GpuBackend::poll()` path returns one `GpuSample` per device (NVML's `nvmlDeviceGetCount` + per-device reads; sysfs enumerates all `card*` directories; Level Zero queries all Sysman devices). The monitor stores:
- `gpus: Vec<GpuSample>` — one entry per device (power, utilization, temperature, VRAM, clocks, throttle)
- `gpu_names: Vec<String>` — per-device display names

The View 4 per-GPU table renders one row per device, making multi-GPU servers (e.g. 8× A4000) fully visible.

### View 4 Layout

| Section | Content |
|---------|---------|
| **System Power** (top) | Total / idle / compute / peak draw, energy (kWh), estimated cost, duration (frozen once the run completes), avg power, max temp, throttle events |
| **Power Over Time** (middle-left) | 1 Hz aggregate power bar chart with auto-scaling y-axis |
| **Efficiency** (middle-right) | J/token, J/ktoken, tokens/watt, $/1M tokens, total tokens, avg/peak/idle/compute power |
| **Per-GPU Table** (lower-middle) | One row per device: name, power, utilization, run-average utilization, temperature, run-average temperature, peak temperature, VRAM, core/mem clock, throttle |
| **Utilization + Temperature** (bottom) | Two 1 Hz **auto-scaled line charts** side by side — the y-axis spans the data's real range (with padding) so a narrow band shows its variation, each with grid lines, an area fill, a dashed mean line, and a peak marker |

### N/A Rule

With no GPU telemetry (driver-less host), View 4 renders a single "no GPU telemetry" placeholder — never a broken frame, never fake zeros. Every derived metric is `Option`-gated: no power → all efficiency metrics are `None` (rendered `N/A`).

### CPU Power Monitoring

Cascade detection at startup:
1. Intel RAPL — cumulative `energy_uj` counter, delta at 1Hz → watts. Plausibility-clamped (≤250W).
2. Super I/O — motherboard sensor chip (asusec, nct6775, it8688). Reads `power1_input` or calculates V×I.
3. Estimated — 40W fixed. Labeled as estimate in UI.

Total system power = Σ(GPU power) + CPU power. Feeds the energy integral and $/token cost.

### Power Plausibility

All GPU backends apply a clamp: readings >5000W are treated as errors. NVIDIA additionally clamps to 1.5× TDP (from `nvmlDeviceGetPowerManagementLimit`). Anomalous readings are logged once (warn-once latch) and capped.

## Engine F: Flat Out

### What It Measures

Real-world maximum throughput: the server's **total** tokens/sec when it is loaded at its **concurrency sweet spot** — the user count Engine B recommends. This is the "one number" to quote when comparing servers or configurations, because it measures the deployment at the load you would actually run it.

### Methodology

1. **Read the stream count** from Engine B's stored sweep result: the practical sweet spot (the highest level where every user still gets ≥ 29.4 t/s — 30 ideal with a 2% margin). When Engine B did not run, fall back to a small default load (3 streams, tagged `default`).
2. **Spawn all `n` streams together** via the shared `WorkerPool`, each sending the same minimal (~15-token) open-ended prompt so prefill is <100 ms — negligible against the 60-second window.
3. **`max_tokens = 100,000` + `ignore_eos`** — effectively unlimited, so the *only* stop condition is the **60-second window** (`WINDOW_SECS`). `ignore_eos` keeps llama.cpp servers from ending a stream on the model's own end-token; other backends ignore the field (graceful degradation).
4. **Abort at 60 s** — the supervisor task is dropped, every worker task and its socket closes, and the server stops generating.
5. **Authoritative token counting** per stream (`authoritative_tokens`): the server `usage` when it arrives, else re-tokenized text — never the raw frame tally.

### What the Numbers Mean

- **Aggregate t/s** = `total_tokens / duration` — the headline: what a full-load deployment sustains.
- **Per-stream t/s** = `aggregate ÷ streams` — ≈ 30 t/s confirms the load is at the sweet spot.
- **TTFT avg / ITL p50 / p99** — pooled across all streams, the latency evidence behind the throughput.

The JSON export records the result under `flat_out` with `stream_count`, `stream_count_source` (`concurrency_sweet_spot` / `default`), `total_tokens`, `duration_secs`, `aggregate_tps`, `per_stream_tps`, and the latency percentiles.

## Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW (one stream)                          │
│                                                                          │
│  HTTP POST ──▶ reqwest ──▶ SSE bytes ──▶ SseParser ──▶ ParsedFrame[]   │
│  (T0)        (T1)        (T2)        (line-buffer)  (chunk classified) │
│                                                              │           │
│                                                              ▼           │
│  StreamWorker stamps each frame with quanta T3/Tn         │           │
│  (T3 = first token, Tn = stream close)                    │           │
│                                                              │           │
│                                                              ▼           │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  Bounded mpsc channel (capacity 256)                           │    │
│  │  StreamEvent::Frame { frame, at, timestamps }                  │    │
│  │  StreamEvent::Complete { timestamps, usage, premature }        │    │
│  │  StreamEvent::Failed { timestamps, error }                     │    │
│  └────────────────────────┬────────────────────────────────────────┘    │
│                           │                                              │
│              ┌────────────┴────────────────┐                            │
│              │                             │                            │
│              ▼                             ▼                            │
│  ┌─────────────────────┐     ┌─────────────────────────┐              │
│  │  Engine Core         │     │  Metrics State           │              │
│  │  (consumes events,  │────▶│  (ArcSwap double-buffer) │              │
│  │   computes §7       │     │  .update() per batch    │              │
│  │   metrics)          │     │  (every 8/16 events)    │              │
│  └─────────┬───────────┘     └───────────┬─────────────┘              │
│            │                              │                            │
│            │                              ▼                            │
│            │               ┌──────────────────────────┐               │
│            │               │  TUI Render Loop (60 Hz)  │               │
│            │               │  .load() → Arc<Snapshot>  │               │
│            │               │  (lock-free, wait-free)   │               │
│            │               └──────────────────────────┘               │
│            │                                                           │
│            ▼                                                           │
│  ┌─────────────────────┐     ┌──────────────────────────┐            │
│  │  SQLite Storage      │     │  Export                   │            │
│  │  (background        │     │  (JSON / Markdown / CSV)  │            │
│  │   flusher)          │     │                           │            │
│  └─────────────────────┘     └──────────────────────────┘            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

**Key invariants:**

- The **timing path** (T0→Tn) lives entirely in the `StreamWorker`. No other component touches quanta.
- The **metrics path** is a one-way pipeline: workers → bounded channel → engine core → `MetricsState::update()` → `ArcSwap` → TUI `load()`. The TUI never writes.
- The **hardware poller** merges its samples into the same `MetricsState` via `load() → clone → merge → update()`. It never touches the stream workers.
- The **storage layer** receives completed results from the engine core after the stream finishes. It is not on the timing path.
- The **History view** reads completed runs back from the storage layer on the key path only (cached in view state) — the 60 Hz render loop never opens the database.

## Configuration Persistence

Crucible is a **single-source-of-truth** tool: one `Config` feeds the headless path, the TUI path, and the export path identically. Every field resolves from the first source that provides it:

```
CLI flag  >  environment variable  >  config file  >  built-in default
```

The **config file** (`~/.config/crucible/config.json`) is the persistence seam. It is read on every start, and it is written from the **TUI Config view (View 6)** — `F2` (or `Esc` while editing) persists the current form back to the file through the same serde layer, so a value edited in the UI flows end-to-end into the next run, the SQLite persistence, and the `--export` output.

**Skip-Setup on subsequent runs:** when a config file already carries a `url` **and** a non-placeholder `model`, a bare `crucible-llm` opens straight onto the dashboard — the interactive Setup walkthrough (URL → model discovery → config → launch) is **skipped**. Setup still runs on the first invocation (no file / no explicit target), and can be re-opened any time with `c`. This is what makes the tool "set it up once, then just run it."

## History & Comparison

Every completed run is persisted to the SQLite store (`~/.local/share/crucible/benchmarks.db`), and **View 5 (History)** reads it back. The view has four modes:

- **List** — the stored sessions, newest first, with a cursor (`j`/`↓`, `k`/`↑`).
- **Detail** (`Enter`) — one run's per-engine summary (mean TTFT, gen t/s, prompt t/s, MTP, J/token, plus the NIAH pass count).
- **Compare** (`C`, then `Enter` to pick the second run) — two runs side-by-side with **signed % deltas** for TTFT, gen t/s, prompt t/s, MTP, and J/token. Green = improvement, red = regression (TTFT and J/token improve when *lower*).
- **Delete** (`D`, then `y`/`n`) — remove a stored run, with a confirmation prompt.

**Measurement isolation:** all DB I/O happens on the key path (user-driven, rare) and the results are cached in the view's state, so the 60 Hz render loop never touches the database — it renders a pure `&App` read.

## Themes

The TUI is skinned by a `Theme` enum with three complete palettes (`ui/theme.rs`). Every color a view draws comes from the active theme — no view hardcodes a `Color` — so switching re-colors the entire screen from a single source of truth.

| Theme | Palette | Config id |
|-------|---------|-----------|
| **Cyberpunk** (default) | Neon cyan · electric purple · deep blue · digital glow | `cyberpunk` |
| **Vampire** | Crimson · gold · dark purple · gothic | `vampire` |
| **Monochrome Pastel** | Soft blue · lavender · clean whites · minimal | `monochrome` |

Each theme supplies the full role set — `primary`, `secondary`, `tertiary`, `accent`, `success`, `danger`, `dim`, `bright`, `border`, `border_active`, plus the gradient-texture colors (`floor`, `bright_gradient`, `bg`) that give the charts their depth.

**Selection & switching.** The theme resolves from `CRUCIBLE_THEME` / the config file's `theme` field / the built-in default (`cyberpunk`). On the **first run** (no theme ever chosen — `theme_explicit` is false) a full-screen **theme picker** takes over before Setup: `↑`/`↓` moves the cursor and the *entire* screen live-previews the hovered theme, while each option box shows its own palette; `Enter` applies it to `app.active_theme`, persists it to the config file, and falls through to Setup (fresh target) or the Dashboard (target given). **Subsequent runs** skip the picker. The theme is changeable any time from **Config (View 6)**'s Theme field — `←`/`→` cycles with a live preview, `1`/`2`/`3` select directly, and `Esc` saves.

Rendering stays a pure `&App` read (the `active_theme` field), so the theme never touches the measurement path.

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language | Rust (stable, 1.98.1) | Performance, memory safety, single static binary |
| Async runtime | Tokio | Non-blocking I/O for concurrent streams, work-stealing thread pool |
| HTTP client | Reqwest | Streaming support, connection pooling, HTTP/1.1 + HTTP/2 |
| SSE parsing | Manual line-buffer state machine | Zero external deps on the hot path, full control over `[DONE]`/BOM/malformed frames |
| TUI | Ratatui + Crossterm | Full-screen terminal UI, 60 FPS, no flicker, unicode block charts |
| Timing | Quanta | Nanosecond-resolution monotonic clock (RDTSC), no syscalls on the fast path |
| Latency histograms | HdrHistogram | Lock-free, high-dynamic-range percentiles (p50 through p99.9) |
| Shared state | ArcSwap | Lock-free reads from the TUI render thread; atomic double-buffer swap |
| Storage | SQLite (rusqlite) | Zero-config, single file, ACID, fast for read-heavy workloads |
| Tokenization | HuggingFace `tokenizers` (optional) | Accurate token counts for prompt sizing and NIAH haystack padding |
| GPU telemetry (NVIDIA) | NVML (`nvml-wrapper`, `nvml` feature — on by default) | Power, VRAM, clocks, temperature, utilization, throttle |
| GPU telemetry (AMD) | Linux sysfs + hwmon (`amdgpu` driver, zero-dep) | Utilization, VRAM, temperature, power, core + memory clocks |
| GPU telemetry (Intel) | Level Zero Sysman via `libloading` (runtime dlopen, no link dep) + i915/xe sysfs fallback | Utilization, power, VRAM, clocks, temperature, throttle (Arc); VRAM/temp/power (fallback) |
| System telemetry | `sysinfo` | Cross-platform CPU/RAM counters (always available) |
| Serialization | Serde + `serde_json` | Config, API requests/responses, export formats |
| Randomness | `rand` (thread RNG) | Cryptographically random needles for NIAH |

## Design Principles

### 1. Measurement Isolation

The act of measuring must not perturb the system. This is enforced architecturally:

- All timing is captured in the stream workers (quanta RDTSC) — the render loop, hardware poller, and storage layer never touch the timing path.
- The TUI reads metrics via `ArcSwap::load_full()` — a lock-free, wait-free atomic load. No mutex, no channel, no blocking.
- The hardware poller merges into the metrics snapshot via `load → clone → merge → update` — it reads the current snapshot, modifies a clone, and atomically swaps it back. The stream workers' events are unaffected.
- The 60 Hz render tick and the 100 ms hardware poll tick run on separate Tokio tasks. A frame drop or a slow disk write cannot introduce measurement latency.

### 2. Honest Metrics

I report **per-user experience**, not just aggregate numbers.

- Engine B's headline recommendation is the **practical sweet spot** (the highest level where per-stream t/s ≥ 30 — 29.4 effective with the 2% margin), not the peak aggregate throughput. 64 users at 2.6 t/s each is not "fast."
- ITL percentiles (p50, p90, p99, p99.9) show the *distribution*, not just the mean. A p99 of 500ms means 1% of tokens take a half-second — the user perceives stutter even if the average is 30ms.
- NIAH reports pass rates per context size, not just an overall percentage. The degradation curve is the actionable data.
- Structured output reports per-case checks (valid JSON, schema, fields, types) — not just a boolean. You see *exactly what failed*.

### 3. Zero Dependencies

A single static Rust binary. No Python, no Node, no system packages (beyond optional GPU drivers). The binary is ~14 MB and runs on any Linux/macOS machine with a terminal.

- The SSE parser is a manual state machine — no `eventsource-stream` crate (dependency removed in v0.1.2).
- The reasoning checkers are pure `std` — no `regex` or parser crates.
- NVIDIA NVML is **on by default** (opt out with `--no-default-features`); Intel Level Zero is dlopen'd at runtime via `libloading` — a zero link dependency, so the binary still runs on machines without the loader.
- The AMD backend is pure sysfs/hwmon — no ROCm stack, no extra crates.
- The tokenizer is optional — without it, `chars/4` estimation keeps the binary self-contained.

### 4. Graceful Degradation

The tool always runs, even when optional features are missing:

| Missing feature | Behavior |
|----------------|----------|
| No GPU / no driver (any vendor) | GPU telemetry is `None`: the TUI GPU panel is hidden, Engine D auto-disables (or reports `N/A` if forced on). CPU/RAM telemetry still works. |
| No tokenizer file | Prompt and completion counts use `chars/4` estimation, flagged `[ESTIMATED]`. NIAH haystack sizing uses the estimate. |
| Server omits `usage` | Token counts fall back to SSE frame count (labeled as estimate in the log). |
| Server ignores `stream: true` | Plain-JSON fallback: the whole body is parsed as one completion. |
| Malformed SSE frames | Counted and skipped. The parser never panics. A note is added to the result. |
| Connection refused | Only *connect-phase* failures are retriable. HTTP errors, timeouts, and read errors are final — they are *measurements* of the endpoint, not transient faults. |
| Stream closes without `[DONE]` | Marked as `premature close`. All captured timestamps and partial tokens are preserved. |
| VRAM at 90%+ | Fragmentation warning. The run continues, but the user is alerted. |

### 5. Reproducible

Every run is logged, stored, and exportable:

- **Run log**: every HTTP request, SSE frame milestone, stream completion, and engine transition is written to `latest.log` (overwritten each run) and an archived `run-<timestamp>.log`. Thread-safe, non-blocking (dedicated writer thread), and degrades to a silent no-op when files can't be created.
- **SQLite storage**: every benchmark session is persisted with its stream metrics, NIAH evaluations, and energy profile. Compare across time, across models, across hardware, across server versions.
- **Export**: JSON (for CI/CD pipelines), GitHub Flavored Markdown (for PRs and READMEs), and raw CSV (for external analysis in Python, R, or Grafana).
- **Deterministic**: `temperature: 0` ensures the same prompt produces the same tokens. The 13-challenge reasoning bank has hand-verified canonical answers. NIAH needles are random per cell (so a coincidental match is impossible), but the pass/fail criterion is deterministic.
