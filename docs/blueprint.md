# CRUCIBLE-LLM: High-Performance Terminal Benchmark & Inference Profiler
## Comprehensive Architecture Blueprint & System Specification

---

## 1. Executive Summary & Vision

`Crucible-LLM` is a dedicated, zero-overhead terminal application designed for machine learning engineers, local inference enthusiasts, and infrastructure operators. It transforms ephemeral command-line speed testing into an empirical, continuous benchmarking suite. Built natively in Rust for near-zero runtime latency, lock-free telemetry streaming, and microsecond-level clock accuracy, `Crucible-LLM` measures not only how fast an inference server produces characters, but also how efficiently it handles compute saturation, memory bandwidth limits, speculative decoding verification, and complex task fidelity.

```
+-----------------------------------------------------------------------------------+
|                                  CRUCIBLE-LLM                                     |
|                                                                                   |
|  +--------------------+   +-----------------------+   +------------------------+  |
|  |   Speed & Latency  |   | Concurrency & Load    |   | Capability & Fidelity  |  |
|  |  * TTFT / PP Speed |   |  * Multi-stream Sweeps|   |  * Needle In A Haystack|  |
|  |  * TPOT / TG Speed |   |  * Saturation Knee    |   |  * Reasoning Depth     |  |
|  |  * MTP Efficiency  |   |  * TTFT vs Load Curve |   |  * Schema Adherence    |  |
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
| **Tokenization** | Hugging Face `tokenizers` | Embedded, client-side BPE/WordPiece tokenization engine. Eliminates reliance on server usage chunks and guarantees deterministic input token counts and pre-computation of needle positions. |
| **Timing & Statistics** | `quanta` + `hdrhistogram` | High-precision time measurement using CPU cycle counters (RDTSC) without syscall overhead; generates high-dynamic-range histograms for latency percentiles ($p50, p90, p99, p99.9$). |
| **Hardware Monitoring** | `nvml-wrapper` + `sysinfo` | Direct, vendor-level telemetry bindings for NVIDIA GPUs (VRAM usage, SM clock, temperature, current power draw in milliwatts), Apple Silicon Unified Memory metrics via macOS private frameworks, and cross-platform CPU/RAM counters. |
| **Persistence** | `rusqlite` + `serde` | Embedded local relational storage for zero-setup benchmark persistence, cross-run comparisons, and structured export (JSON, Markdown tables, CSV). |

---

## 4. Concurrent Architecture & Data Flow

To ensure that high-frequency UI repaints or heavy network streams never introduce artificial measurement latency, `Crucible-LLM` decouples its workload across four independent execution rings connected via lock-free channels:

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
* Each worker maintains independent HTTP/2 streams, recording monotonic timestamps ($T_0$: socket connection start, $T_1$: request write complete, $T_2$: first byte received, $T_3$: first token frame decoded, $T_n$: stream close).
* As chunks arrive, they are decoded without heap re-allocations and pushed to the Engine Core via bounded channels.

### 2. The Engine Core & Metric Synthesizer
* Aggregates packet deltas, classifies them into `Reasoning`, `Content`, or `Control` categories, and tracks Inter-Token Latencies (ITL).
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

* **Microsecond Time-To-First-Token (TTFT):** Measures exact prompt processing latency from request termination to the arrival of the first completion packet.
* **Prefill (PP) Processing Bandwidth:** 
  $$\text{PP Throughput} = \frac{\text{Client Verified Prompt Tokens}}{\text{TTFT}}$$
* **Time-Per-Output-Token (TPOT / Inter-Token Latency):** Tracks delta timing between sequential tokens. Records jitter, variance, and latency distributions rather than merely averaging output over the total run.
* **Reasoning vs. Content Latency Breakdown:** 
  * Explicit telemetry tracking the transition point where reasoning markers end and output delivery begins.
  * Separate throughput numbers for the "thinking phase" versus the "writing phase."
* **MTP (Multi-Token Prediction / Speculative Decoding) Ratio:** 
  Calculates the efficiency factor:
  $$\text{MTP Multiplier} = \frac{\text{Total Output Tokens}}{\text{Received SSE Data Packets}}$$
  * A value of $1.0$ indicates standard single-token auto-regressive generation.
  * Values above $1.0$ quantify speculative draft model acceptance rates and multi-token execution.
* **Warm vs. Cold KV-Cache State Detection:**
  * Tests the engine with randomized anti-cache prefixes versus deterministic identical prefixes.
  * Computes the "Prefix Cache Acceleration Ratio" ($\text{TTFT}_{\text{cold}} / \text{TTFT}_{\text{warm}}$) to evaluate the host server's chunked context caching efficiency.

---

### Engine B: Concurrency & Saturation Curve Sweeper
LLM inference engines scale non-linearly. Crucible provides an automated load sweeper:

* **Concurrency Sweep Modes:** Automates runs across customizable concurrency ladders (e.g., $1 \to 2 \to 4 \to 8 \to 16 \to 32 \to 64$ simultaneous sessions).
* **Throughput Knee-Point Detection:**
  * Plots System Generation Throughput (Aggregate Tokens/Sec across all streams) against Client-Perceived Latency (p90 TPOT).
  * Automatically identifies the "Saturation Knee"—the exact concurrency level where the GPU transitions from memory-bandwidth-bound (under-saturated) to compute-bound, beyond which client latency spikes exponentially.
* **Queue Latency Tracking:** Identifies when an inference server (such as vLLM or llama.cpp) runs out of KV cache blocks and begins queuing or pre-empting client requests.

---

### Engine C: Capability, Accuracy & Context Degradation Engine
A fast model is useless if speed comes at the expense of context retention or reasoning fidelity. Crucible integrates three automated capability micro-suites:

#### 1. Needle-In-A-Haystack (NIAH) Context Retainer
* Automatically generates synthetic long-context documents dynamically scaled to $2\text{k}, 4\text{k}, 8\text{k}, 16\text{k}, 32\text{k}, 64\text{k},$ or $128\text{k}$ tokens.
* Inserts a unique, cryptographically random key-value fact ("needle") at customizable depths ($0\%$ to $100\%$ in $10\%$ intervals).
* Measures two factors simultaneously:
  1. **Accuracy:** Does the model retrieve the needle cleanly?
  2. **Prefill Degradation:** Measures the decay curve of prefill processing speed as context length scales up.

#### 2. Deterministic Reasoning & Code Verification
* Runs a standardized bank of deterministic logic, math, and code generation challenges (e.g., self-contained algorithmic puzzles, schema-constrained outputs).
* Validates responses against strict AST checkers, regular expressions, and formal constraints to report a deterministic pass/fail accuracy score alongside speed.

#### 3. Structured Output & JSON Grammar Compliance
* Tests speed when enforcing structured JSON schemas and tool-call formats via constrained sampling engines (e.g., GBNF grammars, Outlines, or XGrammar).
* Quantifies the performance penalty imposed by grammar validation compared to unrestricted token generation.

---

### Engine D: Hardware & Energy Efficiency Profiler
Bridges the gap between software performance and physical compute:

* **Real-Time Memory Tracking:** 
  * Displays active base model weight allocation vs. dynamic KV-cache reservation in VRAM.
  * Warns users when approaching memory fragmentation thresholds.
* **Energy Consumption per Token:**
  * Correlates GPU power telemetry with active token output.
  * Calculates the **Silicon Efficiency Metric**:
    $$\text{Joules per Token} = \frac{\int_{T_0}^{T_n} P(t) \, dt}{\text{Total Generated Tokens}}$$
    *(Where $P(t)$ is instantaneous GPU wattage).*
  * Allows users to directly compare the running cost and energy efficiency of different model quantizations (e.g., comparing an unquantized FP16 model against a modern 4-bit or 3-bit quantization).

---

## 6. Terminal User Interface (TUI) Design & Layout

`Crucible-LLM` operates an interactive, multi-view dashboard driven by an intuitive, keyboard-first modal paradigm (`vim`-style navigation).

```
=====================================================================================================
 Crucible-LLM v1.0.0 [vLLM / llama.cpp]  |  Target: Qwen3.6-35B-A3B-UD-Q4_K_XL  |  Mode: Concurrency
=====================================================================================================
 [1] Live Monitor  |  [2] Concurrency Matrix  |  [3] Needle (NIAH)  |  [4] History Diff  |  [5] Config
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
* **Top-Left (Key Metrics):** Aggregate throughput, dynamic VRAM capacity bar, GPU core frequency, and instantaneous energy efficiency ($J/\text{token}$).
* **Top-Right (Latency Histogram):** Visual sparkline rendering latency percentiles ($p50, p90, p99$) for token delivery, immediately exposing streaming hiccups, pause states, and GC overhead.
* **Mid-Panel (Stream Matrix):** Live progress indicators for every active stream, explicitly splitting prompt prefill from token generation and highlighting speculative decoding / MTP ratios.
* **Bottom-Panel (Rolling Chart):** Continuous real-time canvas charting system throughput across the testing epoch.

#### View 2: Concurrency & Saturation Curve Analysis
* Displays an aggregate performance heatmap where the X-axis represents Concurrency ($1 \dots 128$) and the Y-axis tracks Latency vs. Throughput.
* Highlights the "Optimal Operational Envelope"—the recommended sweet spot for hosting this model on the target hardware without overwhelming latency budgets.

#### View 3: Context Needle Matrix (NIAH)
* An $N \times M$ terminal grid visually displaying context depths ($0\%$ to $100\%$) across multiple context token sizes ($2\text{k} \dots 128\text{k}$).
* Cells render with dynamic ANSI color-coding:
  * Bright Green: Accurate retrieval + nominal prefill speed.
  * Yellow: Accurate retrieval with substantial prefill throttling.
  * Red: Needle retrieval failed or hallucinated.

#### View 4: Historical Comparison & Regression Diffing
* Enables side-by-side terminal comparisons of prior benchmark runs stored in the local SQLite engine.
* Highlights differential changes (e.g., comparing `vLLM v0.6.x` vs `vLLM v0.7.x` or `llama.cpp b3200` vs `b3300`).
* Calculates delta metrics: percentage gain/loss in TTFT, changes in tokens-per-second, and changes in speculative token verification rates.

---

## 7. Mathematical Formulations for Engine Telemetry

The application implements rigorous system definitions for all calculated metrics:

1. **Monotonic High-Resolution TTFT:**
   $$\text{TTFT} = T_{\text{first\_token\_arrival}} - T_{\text{request\_dispatched}}$$
   *(Timed via hardware cycle registers, strictly excluding local parsing and buffer setup delays).*

2. **Isolated Token Generation Speed (TG Speed):**
   $$\text{TG Speed} = \frac{N_{\text{completion\_tokens}} - 1}{T_{\text{stream\_end}} - T_{\text{first\_token\_arrival}}}$$

3. **Inter-Token Latency (ITL) Jitter:**
   $$\text{Jitter} = \sqrt{\frac{1}{M-1} \sum_{i=1}^{M} (\Delta t_i - \overline{\Delta t})^2}$$
   *(Where $\Delta t_i$ represents the time delta between completion chunk $i$ and $i-1$).*

4. **Speculative Decoding Multi-Token Efficiency Rate:**
   $$\eta_{\text{MTP}} = \frac{N_{\text{total\_tokens}}}{N_{\text{SSE\_packets}}}$$

5. **Prefix Cache Detection Heuristic:**
   $$\text{Cache Status} = \begin{cases} \text{HIT}, & \text{if } \text{TTFT}_{\text{test}} \le 0.15 \times \text{TTFT}_{\text{cold\_baseline}} \\ \text{MISS}, & \text{otherwise} \end{cases}$$

---

## 8. Persistence Architecture & Export Schema

Benchmarking data is saved to a local SQLite database located in the user's platform-specific data directory (`~/.local/share/crucible/benchmarks.db` or `%APPDATA%\crucible`).

### Database Relational Schema
```sql
CREATE TABLE benchmark_sessions (
    session_id TEXT PRIMARY KEY,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    target_url TEXT NOT NULL,
    model_name TEXT NOT NULL,
    backend_type TEXT,       -- 'vllm', 'llamacpp', 'sglang', 'ollama'
    quantization TEXT,
    system_gpu TEXT,
    total_duration_sec REAL
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
* **Zero-Allocation JSON:** For integration with CI/CD regression pipelines (e.g., gating model deploys on prefill or throughput thresholds).
* **GitHub Flavored Markdown:** One-key export generating structured tables formatted for instant posting into pull requests, GitHub issues, or README documentation.
* **Raw CSV Data:** Dumps all raw packet arrival times and inter-token intervals for external analysis in Python, R, or Grafana.

---

## 9. Phased Implementation Roadmap

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
|  * NVML / Apple Silicon hardware bindings for VRAM and wattage profiling.        |
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