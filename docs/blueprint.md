# CRUCIBLE-LLM: High-Performance Terminal Benchmark & Inference Profiler
## Comprehensive Architecture Blueprint & System Specification

---

## 1. Executive Summary & Vision

`Crucible-LLM` is a dedicated, zero-overhead terminal application designed for machine learning engineers, local inference enthusiasts, and infrastructure operators. It transforms ephemeral command-line speed testing into an empirical, continuous benchmarking suite. Built natively in Rust for near-zero runtime latency, lock-free telemetry streaming, and nanosecond-level clock accuracy (a `quanta` TSC-based monotonic clock, one stamp per parsed token frame), `Crucible-LLM` measures not only how fast an inference server produces characters, but also how efficiently it handles compute saturation, memory bandwidth limits, speculative decoding verification, and complex task fidelity.

```
+-----------------------------------------------------------------------------------+
|                                  CRUCIBLE-LLM                                     |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |   Speed & Latency  |   | Concurrency & Load    |   | Capability & Fidelity  |  |
|  | * Dual TTFT / Latency |  |  * Multi-stream Sweeps|   |  * Needle In A Haystack|  |
|  | * Prefill Throughput|   |  * Saturation Knee    |   |  * Reasoning Depth     |  |
|  | * TPOT / TG Speed  |   |  * TTFT vs Load Curve |   |  * Schema Adherence    |  |
|  | * MTP Efficiency (η)|  |  * Practical Sweet Spot|  |  * (reasoning-aware)   |  |
|  +--------------------+   +-----------------------+   +------------------------+  |
|                                       |                                           |
|         +-----------------------------+-----------------------------+             |
|         |                                                           |             |
|  +-----------------------+   +-------------------------+   +-------------------+  |
|  | Hardware Profiling    |   | Reactive TUI Engine     |   | Historical DB     |  |
|  |  * VRAM & GPU Compute |   |  * Ratatui Dashboards   |   |  * SQLite Storage |  |
|  |  * Watts / Token      |   |  * Latency Histograms   |   |  * Model Diffing  |  |
|  +-----------------------+   +-------------------------+   +-------------------+  |
+-----------------------------------------------------------------------------------+
```

---

## 2. Retrospective & Review of the Python Prototype

The initial Python implementation (`llmspeedtest2.py`) established several critical concepts that differentiate modern inference testing from legacy API polling:

### Core Strengths of the Prototype
1. **Separation of Inference Phases:** Accurately isolates Prompt Processing (PP / Prefill) from Token Generation (TG / Decode).
2. **First-Class Reasoning Protocol Detection:** Understands the distinction between raw content stream tokens and inner monologue/chain-of-thought tokens (e.g., `reasoning` and `reasoning_content` deltas), preventing artificial skewing of output metrics.
3. **Multi-Token Prediction (MTP) / Packet Heuristics:** Measures tokens-per-packet ratio to infer speculative decoding efficacy.
4. **Cache State Awareness:** Employs randomized bypass prefixes to distinguish cold-context ingestion from warm KV-cache hits.

### Structural Limitations of the Python Architecture
1. **I/O & Interpreter Jitter:** The Python Global Interpreter Lock (GIL), garbage collection cycles, and non-vectorized JSON string parsing introduce microsecond-to-millisecond measurement noise, skewing TTFT calculations.
2. **Client-Side Tokenization Blindness:** The script relies entirely on post-stream metadata returned in the `usage` block from servers like `llama.cpp` or `vLLM`. If a server omits usage stats or drops mid-stream, throughput calculations fail or fall back to naive character estimates.
3. **Single-Stream Serialization:** Real-world inference servers are designed around continuous batching (vLLM PagedAttention, llama.cpp batched decode). Testing one sequential request does not expose GPU memory bus saturation, scheduling overhead, or multi-tenant degradation.
4. **Ephemeral Data Lifecycle:** Measurements vanish upon exit; there is no ability to contrast quantizations (e.g., Q4_K_M vs. Q8_0), track regression across server versions, or calculate hardware power-efficiency metrics.

---

## 3. Technology Stack & Rust Ecosystem

The application will be compiled as a standalone, zero-dependency static binary using the following Rust library ecosystem:

| Layer | Crate / Library | Purpose & Rationale |
| :--- | :--- | :--- |
| **TUI Interface** | `ratatui` + `crossterm` | Industry-standard terminal rendering with rich widget sets (sparklines, gauges, layout constraints, custom canvas graphs) running at 60 FPS without terminal flicker. |
| **Async Engine** | `tokio` | Multi-threaded runtime with dedicated work-stealing thread pools for network I/O, background hardware telemetry, and UI rendering. |
| **Networking & SSE** | `reqwest` (+ a manual SSE state machine — `eventsource-stream` was evaluated and dropped: the manual parser keeps zero extra dependencies on the hot path and gives full control over `[DONE]`, BOM stripping, and malformed-JSON tolerance; see `ARCHITECTURE.md`) | HTTP/2-enabled asynchronous client supporting connection pooling, byte-level streaming, custom socket-level timeout enforcement, and low-allocation Server-Sent Events (SSE) parsing. |
| **Tokenization** | Hugging Face `tokenizers` | Embedded, client-side BPE/WordPiece tokenization engine (optional at runtime). Supplies exact input token counts and pre-computed needle positions, and the re-tokenization fallback when a stream ends without a server `usage` frame — the server's `usage.completion_tokens` remains the primary, authoritative count. |
| **Timing & Statistics** | `quanta` + `hdrhistogram` | High-precision time measurement using CPU cycle counters (RDTSC) without syscall overhead; generates high-dynamic-range histograms for latency percentiles ($p50, p90, p99, p99.9$). |
| **Hardware Monitoring** | `nvml-wrapper` + `sysinfo` + in-tree sysfs/Level Zero backends | Direct, vendor-level GPU telemetry: NVIDIA via NVML (VRAM, clocks, temperature, power), AMD via `amdgpu` sysfs/hwmon (zero-dependency, no ROCm stack), Intel Arc via Level Zero Sysman (dlopen'd at runtime, no link dependency) with an i915/xe sysfs fallback — plus cross-platform CPU/RAM counters. Energy telemetry is Linux-only by design (all power sources are Linux interfaces) and degrades to a clearly-labeled N/A elsewhere. |
| **Persistence** | `rusqlite` + `serde` | Embedded local relational storage for zero-setup benchmark persistence, cross-run comparisons, and structured export (JSON, Markdown tables, CSV). |

---

## 4. Concurrent Architecture & Data Flow

To ensure that high-frequency UI repaints or heavy network streams never introduce artificial measurement latency, `Crucible-LLM` decouples its workload: all timing lives in the stream workers (nothing on the UI, storage, or hardware path touches the clock), the engines consume tagged events and synthesize metrics, and the TUI reads a lock-free double-buffered snapshot. The hardware poller runs on its own dedicated OS `std::thread` in **both** the TUI and the headless path so it can never contend with the measured workload:

```
+---------------------------------------------------------------------------------+
|                               THREAD TOPOLOGY                                   |
|                                                                                 |
|  [Hardware Telemetry Worker] ---> (crossbeam channel) --->+                     |
|                                                          |                      |
|  [Stream Worker Pool (1..N)] ---> (tokio mpsc channel) -->+--> [Engine Core]    |
|                                                          |        |             |
|  [Clock / Timer (quanta)]   ---> (epoch timestamps) ---->+        |             |
|                                                                   v             |
|  [TUI Render Loop (60Hz)]  <--- (ArcSwap / Double-Buffer) <--- [Metrics State] |
|                                                                   |             |
|  [SQLite Storage Pool]     <--- (Background flusher) <-----------+             |
+---------------------------------------------------------------------------------+
```

### 1. The Stream Worker Pool (I/O Ring)
* Spawns $N$ asynchronous lightweight workers based on the requested concurrency profile.
 * Each worker maintains independent HTTP streams (HTTP/1.1 for local endpoints, HTTP/2 for TLS), recording monotonic timestamps ($T_0$: socket connection start, $T_1$: **response headers received** — the honest T1, not "request write complete", $T_2$: first body byte, $T_3$: first token frame decoded, $T_n$: stream close).
* As chunks arrive, they are decoded without heap re-allocations and pushed to the Engine Core via bounded channels.

### 2. The Engine Core & Metric Synthesizer
* Aggregates packet deltas, classifies them into `Reasoning`, `Content`, `Usage`, or `Control` categories, and tracks Inter-Token Latencies (ITL).
* Calculates rolling statistics and pushes unified snapshots to an atomic, double-buffered state cache read by the UI.

### 3. Hardware Profiler
* Polls system sensors every 100 milliseconds. 
* Collects instantaneous wattage and memory footprint, synchronizing these samples with active prompt and token generation windows.

### 4. The Render Engine
* Consumes metrics from the atomic snapshot on the main thread.
* If a frame drop or terminal resize occurs, it has zero impact on the timing measurements running in the Worker Pool.

---

## 5. Comprehensive Feature & Capability Engines

### Engine A: Advanced Speed & Latency Profiler (Single Stream)
Extending the Python test into deep micro-benchmarks:

* **Dual Time-To-First-Token (TTFT):** two labeled values from the short prompt — *perceived* TTFT ($T_3 - T_0$, the full client round-trip, network included) and *server-side* TTFT ($T_3 - T_1$, where $T_1$ is the response-headers receipt, isolating server queue + prefill + first decode). Published as a **latency** metric (ms), never divided into a token count.
* **Pre-fill (PP) Processing Bandwidth** (the CF-05 split): the old single "PP speed" on a ~45–87-token short prompt was a *latency floor wearing a bandwidth label*. Engine A now measures prefill **throughput** on a **long** prompt (a real context size, `--tokens`, default 10k):
  $$\text{PP Throughput} = \frac{\text{Prompt Tokens (long prompt)}}{\text{Server-side TTFT (}T_3 - T_1\text{)}}$$
  Three clean runs are averaged. The full 2k→128k prefill-degradation curve is measured by Engine C1's NIAH matrix.
* **Time-Per-Output-Token (TPOT / Inter-Token Latency):** one timestamp per parsed token frame (so batched MTP frames keep their real inter-frame gaps). Reports the mean plus the latency *distribution* (p50/p90/p99/p99.9 via `hdrhistogram`), not a single averaged number.
* **Reasoning vs. Content Latency Breakdown:** 
  * Explicit telemetry tracking the transition point where reasoning markers end and output delivery begins; the server-reported `reasoning_tokens` count is tracked, persisted, and exported.
  * When that detail is present, the published decode rate is the **answer phase** — content tokens over the content window, split at the reasoning→content transition (pure-CoT streams report 0.0, not a blended rate); without it, the historical full-window blended rate.
* **MTP (Multi-Token Prediction / Speculative Decoding) Ratio — the canonical η:** 
  $$\eta_{\text{MTP}} = \frac{\text{Total Output Tokens (usage)}}{\text{All Token Frames (Reasoning + Content)}}$$
  * A value of $1.0$ indicates standard single-token auto-regressive generation.
  * Values above $1.0$ are legitimate and quantify multi-token / speculative batching (vLLM MTP, EAGLE), where one SSE frame carries several accepted tokens.
* **Warm vs. Cold KV-Cache State:**
  * `--nocache` prepends a unique random prefix to force a genuinely cold prefill (a cached prefill reads near-zero TTFT and would inflate the bandwidth number).
  * A `cache_hit` flag is derived from the server's `usage.cached_tokens > 0` (a warm KV / prefix-cache hit) — no speculative cold/warm ratio is computed.
* **Discarded warmup:** one warmup iteration is run and thrown away before the measured iterations, so cold-start / KV-cache effects never pollute the first measured sample.

---

### Engine B: Concurrency & Saturation Curve Sweeper
LLM inference engines scale non-linearly. Crucible provides an automated load sweeper:

* **Concurrency Sweep:** steps a customizable ladder (default $1 \to 2 \to 3 \to 4 \to 8 \to 12 \to 16 \to 24 \to 32$ simultaneous sessions — granular at the low end where home setups operate). One **discarded warmup level** (at the first ladder's concurrency) runs before the measured levels.
* **Dual per-stream denominators:** each level reports `tps_per_completed_stream` (aggregate ÷ completed streams — the *capacity* view) and `tps_per_requested_stream` (aggregate ÷ requested streams — the *user-experience* view, where a failed stream dilutes it), plus `failed_stream_ratio`; levels with > 20 % failed streams are flagged `high_failure`.
* **Practical Sweet Spot & Usability:** the headline recommendation is the highest level where every user still gets ≥ 30 t/s (29.4 t/s effective with the 2 % margin); a "Maximum Usable" (≥ 15 t/s) threshold and an "Unusable From" point complete the profile. Aggregate throughput alone is deliberately *not* the recommendation — 64 users at 2.6 t/s each is "slow for everyone".
* **Throughput Knee-Point Detection:**
  * Plots System Generation Throughput (Aggregate Tokens/Sec across all streams) against Client-Perceived Latency (p90 TPOT).
  * Automatically identifies the "Saturation Knee"—the exact concurrency level where the GPU transitions from memory-bandwidth-bound (under-saturated) to compute-bound, beyond which client latency spikes exponentially.
* **Queue Latency Tracking:** Identifies when an inference server (such as vLLM or llama.cpp) runs out of KV cache blocks and begins queuing or pre-empting client requests (visible as TTFB growth at higher concurrency).

---

### Engine C: Capability, Accuracy & Context Degradation Engine
A fast model is useless if speed comes at the expense of context retention or reasoning fidelity. Crucible integrates three automated capability micro-suites:

#### 1. Needle-In-A-Haystack (NIAH) Context Retainer
* Automatically generates synthetic long-context documents dynamically scaled to $2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}, 64\text{k},$ or $128\text{k}$ tokens.
* Inserts a unique, cryptographically random key-value fact ("needle" — a 128-bit value rendered `{16 hex}-{16 hex}`) at customizable depths ($0\%$ to $100\%$ in $10\%$ intervals): 7 sizes × 11 depths = 77 cells, run sequentially.
* **Reasoning-model correct:** each cell suppresses the model's chain-of-thought (a per-backend switch — `chat_template_kwargs.enable_thinking=false` for vLLM/SGLang/LM Studio, `think=false` for Ollama, a trailing `/no_think` for llama.cpp; graceful no-op elsewhere) and checks retrieval **on the visible answer only** (strict substring of the needle value). A model that merely *mentions* the value in its CoT cannot false-pass; a reasoning model whose answer used to be buried in a 64-token CoT budget no longer false-fails (retrieval on reasoning models went from ~35 % false-fail to ~100 %).
* Measures two factors simultaneously:
  1. **Accuracy:** Does the model retrieve the needle cleanly? Per-size pass rates, color-coded green ≥ 80 % / yellow 50–79 % / red < 50 %, and a "reliable up to" threshold (largest size with ≥ 80 % pass rate).
  2. **Prefill Degradation:** Measures the decay curve of prefill processing speed (`prompt_tokens / TTFT` per cell) as context length scales up; a cell is *throttled* (yellow) when its TTFT exceeds 2× the linear expectation anchored on the smallest context size at the same depth.

#### 2. Deterministic Reasoning & Code Verification
* Runs a standardized bank of **13 self-contained challenges** (5 math, 5 logic, 3 code), each with exactly one correct answer.
* Validates responses against **strict, deterministic, pure-`std` checkers** (no regex or parser crates, no LLM-as-judge): math by numeric-value match, logic by whole-word yes/no verdict, code by structural markers and balanced-delimiter checks. The checkers judge the **committed answer span only** (content deltas; chain-of-thought is suppressed per-backend and excluded), so chatty answers cannot flip a verdict and CoT-only output is not credited. A deterministic pass/fail accuracy score is reported alongside the answer-phase decode speed.

#### 3. Structured Output & JSON Grammar Compliance
* One free-form baseline plus three escalating schema cases (flat object → fixed-length array → nested schema), scored field-by-field (valid JSON, schema shape, fields, types).
* Where the transport supports it, each constrained run sends a **real** `response_format: { "type": "json_schema", "json_schema": { …, "strict": true } }` object — engaging the server's constrained-decoding engine (vLLM XGrammar, Outlines, GBNF) to restrict the token space to the schema. Backends that cannot carry the object fall back to the softer `json_object` *hint*, and the result **flags which mode was used** — a ~0 % "penalty" on a hint-dropping backend is reported as "no constraint was applied", never as "constrained decoding is free".
* Quantifies the performance penalty of the constraint in use (free-form vs. constrained decode speed), reported only when the constraint was effective (the constrained run produced valid JSON).

---

### Engine D: Hardware & Energy Efficiency Profiler
Bridges the gap between software performance and physical compute. **Linux-only by design** — every power source (NVML, `amdgpu` sysfs/hwmon, Level Zero, RAPL/Super I/O) is a Linux interface — so on a non-Linux host, or a host with no GPU/driver, Engine D degrades to a clearly-labeled **N/A** (never a failure, never a spurious `0.0`), while the rest of the suite runs unchanged. It must run on the machine with the GPU: all telemetry is local.

* **Real-Time Memory Tracking:** 
  * Displays VRAM used/total per GPU (all detected cards) plus core/memory clocks, temperature, utilization, and throttle reasons.
  * Warns users when VRAM occupancy reaches the 90 % fragmentation threshold.
* **Energy Consumption per Token:**
  * Correlates GPU power telemetry with active token output. Power is sampled at **100 ms** on a dedicated poller thread; energy is the trapezoidal integral over that 100 ms trace, **windowed to the active load** (`[load_start, load_end]`) — the 1 Hz history is graph-only. Phase-aware averages (prefill vs. decode power) sample the same 100 ms trace, so even a sub-second TTFT yields a real phase power.
  * Calculates the **Silicon Efficiency Metric**:
    $$\text{Joules per Token} = \frac{\int P(t) \, dt}{\text{Total Tokens Generated by Every Engine That Ran}}$$
    *(Where $P(t)$ is total system wattage — GPU + CPU. The denominator spans all engines' token counts, so the figure is not inflated by counting whole-run energy against a partial token total.)*
  * Allows users to directly compare the running cost and energy efficiency of different model quantizations (e.g., comparing an unquantized FP16 model against a modern 4-bit or 3-bit quantization).

---

## 6. Terminal User Interface (TUI) Design & Layout

`Crucible-LLM` operates an interactive, multi-view dashboard driven by an intuitive, keyboard-first modal paradigm (`vim`-style navigation).

```
 =====================================================================================================
  Crucible-LLM v0.1.4 [vLLM / llama.cpp]  |  Target: Qwen3.6-35B-A3B-UD-Q4_K_XL  |  Mode: Concurrency
 =====================================================================================================
  [1] Live Monitor  |  [2] Concurrency  |  [3] Needle (NIAH)  |  [4] GPU & Power  |  [5] History  |  [6] Config
 -----------------------------------------------------------------------------------------------------
 TELEMETRY GAUGES                            INTER-TOKEN LATENCY (ITL) DISTRIBUTION
 Total Aggregate:  842.3 t/s                 p50: 12.1 ms  |  p90: 16.4 ms  |  p99: 41.2 ms
 Active Streams:   16 / 16                   50ms |
 Target GPU VRAM:  21.4 GB / 24.0 GB (89%)        |         *
 Current Power:    285 W (0.338 J/token)          |        ***   *
                                             0ms  +------------------------------------
-----------------------------------------------------------------------------------------------------
 ACTIVE STREAMS MONITOR (Top 4 of 16 Active)
 ID   TYPE       STATE      TOKENS (PP/TG)   TTFT      GEN SPEED   MTP RATE   PROGRESS
 #01  Reasoning  Streaming  2048 / 312       0.182 s   72.4 t/s    1.84 x     [============>       ]
 #02  Content    Streaming  512  / 180       0.045 s   88.1 t/s    1.02 x     [==================> ]
 #03  Tool-Call  Waiting    4096 / --        --        --          --         [                    ]
 #04  Reasoning  Done       2048 / 840       0.191 s   68.9 t/s    1.79 x     [====================]
-----------------------------------------------------------------------------------------------------
 REAL-TIME SYSTEM PERFORMANCE (TOKENS/SEC OVER TIME)
 1200 |                                                         ..-***--.
  800 |                                            ..---*******'         '***--.
  400 |                             ...---*********'
    0 +-----------------------------+---------------+---------------+---------------+----------------
      0s                            15s             30s             45s             60s
-----------------------------------------------------------------------------------------------------
 LOG & EVENT STREAM
 [12:44:02] Stream #04 completed: 840 tokens in 12.19s (68.9 t/s). Speculative Acceptance: 79%
 [12:44:03] Warning: Stream #03 prefill context reached 4096 tokens. Server KV memory allocation +400MB
 [12:44:04] Concurrency step up: Spawning batch 17..24.
=====================================================================================================
 [Space] Pause/Resume  |  [+] Step Concurrency  |  [N] New Needle Test  |  [E] Export  |  [Q] Quit
```

### Detailed View Descriptions

#### View 1: Live Monitor & Telemetry (Primary Dashboard)
* **Status Bar:** Displays connected endpoint URL, target model identifier, context parameters, and active profile.
* **Key Metrics:** Aggregate throughput, per-engine and overall statistics (max/avg/p5), and a capability-assessment panel (Reasoning / Long-context / Structured / Energy) with plain-language verdicts — the Long-context bar is the NIAH *reliable-up-to* threshold, not a flat all-cells average.
* **Stream Matrix:** Live progress indicators for every active stream, explicitly splitting prompt prefill from token generation and highlighting speculative decoding / MTP ratios (with estimated-token tagging when a count came from a `chars/4` fallback).
* **Rolling Chart:** Continuous real-time charting of system throughput across the testing epoch, with engine-transition markers and a live event log.

#### View 2: Concurrency & Saturation Curve Analysis
* Displays the sweep curve (aggregate t/s and per-stream t/s per ladder level, log₂ x-axis) with the **practical sweet spot** highlighted (green in the level table; the theme primary color on the curve, where measured levels are `◆` points) and the throughput knee marked `▲` (red).
* Highlights the "Optimal Operational Envelope"—the recommended load for hosting this model on the target hardware without overwhelming latency budgets — plus the dual per-stream denominators and the > 20 % failure flag per level.

#### View 3: Context Needle Matrix (NIAH)
* A 7 × 11 terminal grid displaying context depths ($0\%$ to $100\%$) across context token sizes ($2\text{k} \dots 128\text{k}$).
* Cells render with dynamic ANSI color-coding:
  * Bright Green: Accurate retrieval + nominal prefill speed.
  * Yellow: Accurate retrieval with substantial prefill throttling.
  * Red: Needle retrieval failed (or no valid baseline at a smaller size — flagged distinctly).
* Context sizes measured without a tokenizer are labeled `(approx.)` — never presented as exact.

#### View 4: GPU & Power Monitor
* Multi-GPU telemetry table (full names, power, utilization, temperature, VRAM, clocks, throttle), power-over-time graph (1 Hz, auto-scaled), auto-scaled utilization/temperature charts with mean & peak markers, energy (kWh), $/1M token cost analysis (input/output/blended), efficiency metrics (J/token, tokens/watt), system power (GPU + CPU), and the idle baseline (auto 5 s / manual `[i]` 10 s). Shown only when a GPU backend is live; hidden (not N/A-boxed) when there is none.

#### View 5: History & Regression Diffing
* Enables side-by-side terminal comparisons of prior benchmark runs stored in the local SQLite engine (list / detail / compare / delete modes).
* Highlights differential changes (e.g., comparing `vLLM v0.6.x` vs `vLLM v0.7.x` or `llama.cpp b3200` vs `b3300`).
* Calculates signed delta metrics: percentage gain/loss in TTFT, tokens-per-second, and MTP efficiency, plus J/token (TTFT and J/token improve when *lower*).

#### View 6: Config
* An editable form (behind a read-only edit gate) with per-field explanations: URL, model, mode, tokens, iterations, timeout, API key, cache bypass, tokenizer, ladder, the six engine toggle rows (A/B/C1/C2/C3/F) plus the Hardware telemetry row (which selects D), theme, and $/kWh rate. All seven engines are individually selectable here (as in the Setup flow, which lists all seven). `F2` saves to the config file, `F5` launches a run, `R` resets to defaults.

---

## 7. Mathematical Formulations for Engine Telemetry

The application implements rigorous system definitions for all calculated metrics:

1. **Monotonic High-Resolution TTFT (dual, CF-06):**
   $$\text{TTFT}_{\text{perceived}} = T_{\text{first\_token}} - T_{\text{request\_start}} \qquad\qquad \text{TTFT}_{\text{server}} = T_{\text{first\_token}} - T_{\text{response\_headers}}$$
   *(Timed via hardware cycle registers (quanta TSC), strictly excluding local parsing and buffer setup delays. `T1` is stamped when the server's **response headers arrive** — not when the last request byte is written — so the server-side TTFT excludes the request-send network leg and isolates server queue + prefill + first decode; the perceived TTFT is the full client round-trip. What falls inside `T1→T3` varies by backend (vLLM includes queue + prefill; llama.cpp/Ollama exclude it), which is why both are published.)*

2. **Isolated Token Generation Speed (TG Speed):**
   $$\text{TG Speed} = \frac{N_{\text{completion\_tokens}}}{T_{\text{last\_token}} - T_{\text{first\_token}}}$$
   *(v0.1.1 code formula: the decode window runs from the **first** token frame to the **last** — prefill/TTFT and the trailing gap to `[DONE]` are excluded. The numerator is `N`, not `N−1`: `N` tokens span the window, while `N−1` is the count of inter-token *gaps* — that is the ITL formula, not the rate. When the server reports reasoning-token detail, the window splits at the reasoning→content transition and the rate is content tokens over the content window (the answer phase).)*

3. **Inter-Token Latency (ITL):**
    $$\overline{\text{ITL}} = \frac{T_{\text{last\_token}} - T_{\text{first\_token}}}{N_{\text{completion\_tokens}} - 1} \qquad\text{plus percentiles } p50/p90/p99/p99.9$$
    *(Where $\Delta t_i$ is the arrival delta of token frame $i$ vs. $i-1$. One timestamp is stamped per parsed token frame (CF-09), so a batched MTP frame yields its real inter-frame gap — not a ~1 ns artifact that drags p50 toward zero. The published percentiles come from an `hdrhistogram` over those per-frame deltas; samples above the 60 s bound are clamped and the clamp count is reported (`itl_clamped`), so a saturated histogram is never silent.)*

4. **Speculative Decoding Multi-Token Efficiency Rate (MTP η):**
   $$\eta_{\text{MTP}} = \frac{N_{\text{completion\_tokens}}}{N_{\text{total\_token\_frames}}}$$
   *(v0.1.1 canonical definition (CF-12): the numerator is the server's `usage.completion_tokens`; the denominator is **all** token frames observed in the stream (Reasoning + Content combined) — not content-only frames and not raw SSE packets. η = 1.0 means one token per frame (standard auto-regressive); η > 1 is legitimate and quantifies multi-token / speculative batching (vLLM MTP, EAGLE), where a single SSE frame carries several accepted tokens. 0.0 = N/A when no token frame arrives.)*

5. **KV-Cache State:**
    $$\text{cache\_hit} = \big[\, \text{usage.cached\_tokens} > 0 \,\big] \qquad\text{cold runs: } \texttt{--nocache} \text{ (unique random prefix)}$$
    *(The `cache_hit` flag is derived from the server's own `usage.cached_tokens` detail — a warm KV / prefix-cache hit. `--nocache` prepends a unique random prefix to force a genuinely cold prefill, so bandwidth numbers are not inflated by cached context. No speculative cold/warm TTFT ratio is computed.)*

---

## 8. Persistence Architecture & Export Schema

Benchmarking data is saved to a local SQLite database located in the user's platform-specific data directory (`~/.local/share/crucible/benchmarks.db` or `%APPDATA%\crucible`).

### Database Relational Schema
```sql
CREATE TABLE benchmark_sessions (
    session_id TEXT PRIMARY KEY,   -- the run's UUID; also stamps the log archive + export filename
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_url TEXT NOT NULL,
    model_name TEXT NOT NULL,
    backend_type TEXT,       -- 'vllm', 'llamacpp', 'sglang', 'ollama'
    quantization TEXT,
    system_gpu TEXT,
    total_duration_sec REAL,
    engines TEXT,            -- which engines this run selected, as a JSON name list (e.g. ["speed","niah"])
    engine_results TEXT      -- headline results of the non-stream engines (B/C2/C3/D/F) as a JSON object
);

CREATE TABLE stream_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES benchmark_sessions(session_id),
    concurrency_level INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    reasoning_tokens INTEGER,
    ttft_ms REAL,
    tpot_ms REAL,
    mtp_efficiency REAL,
    joules_per_token REAL,
    cache_hit BOOLEAN
);

CREATE TABLE needle_evaluations (
    eval_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES benchmark_sessions(session_id),
    context_length INTEGER,
    depth_percent REAL,
    retrieved_successfully BOOLEAN,
    latency_ms REAL
);
```

### Export Formats
* **JSON:** For integration with CI/CD regression pipelines (e.g., gating model deploys on prefill or throughput thresholds). Carries the full picture: the `methodology` block (the exact formula behind every number), the energy object, Engine A's prefill/latency split, the dual per-stream denominators, NIAH reliability, and `reasoning_tokens`.
* **GitHub Flavored Markdown:** One-key export generating structured tables formatted for instant posting into pull requests, GitHub issues, or README documentation.
* **Raw CSV Data:** Dumps all raw packet arrival times and inter-token intervals for external analysis in Python, R, or Grafana.

**Traceability:** every run is identified by a session UUID that stamps three artifacts — the `benchmark_sessions` row, the run-log archive (`run-<timestamp>-<id[:8]>.log`), and the default export filename (`crucible-<id[:8]>.<ext>`) — so a stored run, its log, and its export join deterministically. Every completed run (headless or TUI) persists the engines it ran **and** the headline result of every engine that ran: A → `stream_metrics` (per run, including `reasoning_tokens`), C1 → `needle_evaluations` (per cell), B/C2/C3/D/F → the session's `engine_results` JSON column.

---

## 9. Phased Implementation Roadmap

All four phases below are **shipped** (v0.1.1 → v0.1.4): the Rust core with single-stream parity, the reactive Ratatui architecture, the concurrency engine with load curves, and the capability suites with hardware telemetry are all in the codebase. The phased breakdown is retained as the historical build record.

```
+----------------------------------------------------------------------------------+
| PHASE 1: Rust Core & Single-Stream Parity                                        |
|  * Port Python logic to Rust (Tokio + Reqwest).                                  |
|  * Implement low-allocation SSE chunk parser for reasoning and content deltas.   |
|  * Integrate Quanta for RDTSC cycle timing and HdrHistogram for metrics.         |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| PHASE 2: Ratatui Interface & Reactive Architecture                               |
|  * Build the terminal dashboard layout using Ratatui & Crossterm.                |
|  * Implement lock-free messaging between worker threads and UI loop.             |
|  * Build live sparklines, gauges, and scrolling event logs.                      |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| PHASE 3: Concurrency Engine & Load Curves                                        |
|  * Multi-stream connection pooling.                                              |
|  * Concurrency stepping sweeps (1 to 64 streams).                                |
|  * Saturation knee-point detection & throughput/latency matrix visualizer.       |
+----------------------------------------------------------------------------------+
                                        |
                                        v
+----------------------------------------------------------------------------------+
| PHASE 4: Capability Suites & Hardware Telemetry                                  |
|  * Embedded Needle-In-A-Haystack (NIAH) runner with automated verification.      |
|  * NVML / AMD sysfs / Intel Level Zero backends for VRAM & watts (Linux-only).   |
|  * SQLite database engine, historical diffing view, and CI/CD export tooling.   |
+----------------------------------------------------------------------------------+
```

---

## 10. Summary of Architectural Advantages

By shifting your prototype from a Python script to this Rust TUI architecture:
1. **Timing Noise is Eliminated:** Measurement accuracy shifts from coarse millisecond ranges down to hardware-level microsecond precision.
2. **True System Load Testing:** Continuous batching and server scheduling dynamics are fully exposed through concurrent, multi-stream workloads.
3. **Multi-Faceted Evaluation:** Testing expands beyond raw token speed to measure output accuracy, context fidelity, and silicon power efficiency.
4. **Actionable Developer UX:** An interactive, responsive terminal dashboard replaces ephemeral console text with historical tracking and diagnostic insights.