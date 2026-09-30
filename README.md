# Crucible-LLM

A native-Rust, zero-runtime-dependency terminal application that benchmarks
OpenAI-compatible streaming LLM inference endpoints (vLLM, llama.cpp, SGLang,
Ollama). It measures not only raw token speed, but how efficiently an engine
handles compute saturation, speculative decoding, and complex task fidelity.

```
crucible-llm --url http://your-inference-host:8080/v1 --model qwen3 --mode long
```

## Engines

| Engine | What it measures |
| :--- | :--- |
| **A — Speed & Latency** | Microsecond TTFT, isolated prefill (PP) / decode (TG) throughput, MTP / speculative-decoding ratio, ITL jitter, warm/cold KV-cache detection |
| **B — Concurrency & Saturation** | Multi-stream sweeps (1→2→4→8→16→32→64), aggregate tokens/sec, client-perceived p50/p90/p99 TPOT, saturation **knee-point** detection and the "Optimal Operational Envelope" |
| **C — Capability & Fidelity** | C1 needle-in-a-haystack context retention (2k–128k × 0–100% depth, with prefill-degradation curves), C2 deterministic reasoning/code verification, C3 structured-output / JSON-grammar compliance and its speed penalty |
| **D — Hardware & Energy** | VRAM usage, GPU clock/temperature, instantaneous wattage, and the Silicon Efficiency Metric `Joules/Token = ∫P(t)dt / Total_Generated_Tokens`; VRAM fragmentation warnings |

Results are presented through a 60 FPS ratatui+crossterm dashboard and
persisted to embedded SQLite (`~/.local/share/crucible/benchmarks.db`) with
JSON / Markdown / CSV export for CI/CD regression gating.

Measurement isolation is the design's top invariant: worker-pool timing
(`quanta` cycle clock) lives in dedicated rings and is never touched by UI
repaints, allocation churn, or the SQLite flusher — the dashboard reads a
lock-free, double-buffered (`ArcSwap`) snapshot.

## Build

The toolchain is pinned to `stable` via `rust-toolchain.toml` (the machine's
*default* toolchain is too old for the latest ratatui/reqwest/quanta).

```sh
cargo build --release          # optimized static binary → target/release/crucible-llm
cargo test                     # unit + integration + end-to-end suites (offline mock server)
cargo clippy --all-targets     # static analysis
cargo fmt --check              # formatting
```

The release binary is fully self-contained: SQLite is compiled in
(`rusqlite` bundled) and there are no system library dependencies at runtime.

**Optional NVIDIA GPU telemetry** (VRAM, wattage, clocks):

```sh
cargo build --release --features nvml
```

`nvml` links against the NVIDIA driver (`libnvidia-ml`). On machines without
the driver (or without an NVIDIA GPU) every hardware field degrades to
`N/A` — the app never panics (CPU/RAM telemetry via `sysinfo` still works).

## Quick start

**Headless single-stream benchmark** (Engine A) — the direct port of
`llmspeedtest.py`:

```sh
crucible-llm --url http://host:8000/v1 --model my-model --mode long --tokens 4096
crucible-llm --url http://host:8000/v1 --model my-model --iterations 5 --json
crucible-llm --url http://host:8000/v1 --model my-model --nocache   # bust the server KV cache
```

**Interactive dashboard** (all engines, 60 FPS):

```sh
crucible-llm --url http://host:8000/v1 --model my-model --tui
```

**Export for CI/CD** (after any headless run):

```sh
crucible-llm --url http://host:8000/v1 --model my-model --json \
  --export json --export-path ./results.json
```

A bare invocation (`crucible-llm` with no target) prints the banner and exits
0.

## CLI reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `--url <URL>` | `http://192.168.51.163:8080/v1/chat/completions` | Endpoint: bare host, base URL, or full completions path |
| `--model <NAME>` | `default` | Model name sent in the request |
| `--mode <short\|long>` | `short` | `short` = fixed ~50-token prompt; `long` = padded to `--tokens` (`base` is an alias for `short`) |
| `--tokens <N>` | `2000` | Target prompt tokens for `long` mode |
| `--iterations <N>` | `1` | Number of headless runs |
| `--api-key <KEY>` | — | Sent as `Authorization: Bearer <KEY>` |
| `--timeout <SECS>` | `120` | Connect / idle-read timeout |
| `--nocache` | off | Prepend a unique random prefix to bypass the server KV cache (cold-cache runs) |
| `--json` | off | Emit the result as JSON on stdout (the prototype's `output_json` field set) |
| `--verbose` | off | Per-run chunk detail on stderr |
| `--no-color` | off | Force-disable ANSI colors (default: on only when stdout is a TTY) |
| `--tokenizer <PATH>` | — | HF `tokenizer.json` for exact prompt token counts; without it, counts are `chars/4` estimates (flagged `estimated`) |
| `--tui` | off | Open the interactive ratatui dashboard instead of the headless run |
| `--export <json\|md\|csv>` | — | Export the completed run (headless: after persisting; TUI: `e` key) |
| `--export-path <PATH>` | `data_dir()/exports/…` | Destination file for `--export` |
| `--config <PATH>` | `~/.config/crucible/config.json` | JSON config file |
| `--help` / `--version` | | Standard clap help |

### Configuration layering

One `Config` (a superset of both Python prototype CLIs) feeds the headless,
TUI, and export paths identically. Resolution order per field — first match
wins:

1. **CLI flag**
2. **environment variable**
3. **config file** (JSON, all fields optional)
4. **built-in default** (mirrors `llmspeedtest.py`)

Environment variables:

| Variable | Flag equivalent |
| :--- | :--- |
| `CRUCIBLE_URL` | `--url` |
| `CRUCIBLE_MODEL` | `--model` |
| `CRUCIBLE_MODE` | `--mode` |
| `CRUCIBLE_TOKENS` | `--tokens` |
| `CRUCIBLE_ITERATIONS` | `--iterations` |
| `CRUCIBLE_API_KEY` | `--api-key` |
| `CRUCIBLE_TIMEOUT` | `--timeout` |
| `CRUCIBLE_NOCACHE` | `--nocache` |
| `CRUCIBLE_TOKENIZER` | `--tokenizer` |
| `CRUCIBLE_JSON` | `--json` |
| `CRUCIBLE_VERBOSE` | `--verbose` |
| `CRUCIBLE_NO_COLOR` | `--no-color` |
| `CRUCIBLE_TUI` | `--tui` |
| `CRUCIBLE_EXPORT` | `--export` |
| `CRUCIBLE_EXPORT_PATH` | `--export-path` |
| `CRUCIBLE_CONFIG` | `--config` |

Example config file (`~/.config/crucible/config.json`):

```json
{
  "url": "http://192.168.51.163:8080/v1/chat/completions",
  "model": "qwen3",
  "mode": "long",
  "tokens": 8192,
  "timeout": 60
}
```

## The TUI dashboard

Five views (blueprint §6), rendered at 60 Hz from the lock-free metrics
snapshot:

| Key | Action |
| :---: | :--- |
| `1` | **Live Monitor** — telemetry gauges (aggregate t/s, VRAM, GPU clock, J/token), ITL distribution (p50/p90/p99 + histogram), active-stream matrix (PP/TG split, TTFT, gen speed, MTP rate, progress), rolling throughput chart, log/event stream |
| `2` | **Concurrency Matrix** — the sweep curve (concurrency × aggregate t/s × p90 TPOT) with the detected knee (red) and optimal operational envelope / sweet spot (green) highlighted |
| `3` | **Needle (NIAH)** — the context-size × depth grid, color-coded: green = accurate + nominal prefill, yellow = accurate + throttled prefill, red = retrieval failed |
| `4` | **History Diff** — side-by-side comparison of two stored runs with signed delta metrics (TTFT, tokens/s, MTP rate, J/token), gains green / regressions red; `j`/`k` navigate, `a`/`b` select run A / run B |
| `5` | **Config** — the resolved run configuration (target, model, mode, tokens, ladder, tokenizer, feature toggles) |
| `Space` | Pause / resume the active run |
| `+` | Step concurrency up (live sweep) |
| `n` | Queue a new NIAH matrix run |
| `e` | Export the current session (JSON / MD / CSV, per `--export`) to the data dir |
| `q` | Quit (terminal state is restored on all exit paths) |

A dropped frame, a resize, or any UI activity never touches the timing path.

## Metric definitions (blueprint §7)

- **TTFT** = `T_first_token − T_request_dispatched` (quanta cycle clock)
- **PP throughput** = `prompt_tokens / TTFT`
- **TG speed** = `(completion_tokens − 1) / (T_stream_end − T_first_token)`
- **ITL jitter** = std-dev of the inter-token deltas
- **MTP multiplier** = `total_output_tokens / SSE_packets` (1.0 = standard
  auto-regressive; >1 quantifies speculative-draft acceptance)
- **Cache status** = `HIT` if `TTFT ≤ 0.15 × TTFT_cold_baseline`, else `MISS`
- **Joules/Token** = `∫P(t)dt / Total_Generated_Tokens` over the 100 ms
  hardware power trace

## Persistence & data locations

| What | Where |
| :--- | :--- |
| SQLite database | `~/.local/share/crucible/benchmarks.db` (Linux) · `%APPDATA%\crucible\benchmarks.db` (Windows) |
| Exports (default) | `~/.local/share/crucible/exports/crucible-<session-id>.<ext>` |
| Config file | `~/.config/crucible/config.json` |

Schema (three tables): `benchmark_sessions` (one row per run: target, model,
backend, GPU, duration), `stream_metrics` (per stream/iteration: token
counts, TTFT, TPOT, MTP, J/token, cache-hit), `needle_evaluations` (per NIAH
cell: context length, depth, retrieved, latency).

Every completed run (headless or TUI) persists automatically; a storage
failure degrades gracefully and never changes the exit code of a benchmark.

## Export formats

| Format | Use |
| :--- | :--- |
| `json` | Zero-alloc, parseable — CI/CD regression gating (e.g., deploy gates on TTFT/throughput thresholds) |
| `md` | GitHub-Flavored Markdown tables (metadata + stream metrics + needle table) for PRs/issues/READMEs |
| `csv` | Raw per-packet arrival times + inter-token intervals for Python / R / Grafana |

```sh
crucible-llm --url http://host:8000/v1 --model my-model \
  --export csv --export-path ./packets.csv
```

In the TUI, press `e` to export the current session (format follows
`--export`, default `json`).

## Graceful degradation

- **No NVIDIA GPU / driver:** all hardware fields report `N/A` (never a
  panic, never a spurious 0.0); CPU/RAM telemetry still works.
- **No `tokenizer.json`:** prompt token counts fall back to `chars/4` and
  are flagged `estimated` in every report.
- **Server omits `usage`:** token counts fall back to counted frames.
- **Dead endpoint / all runs failed:** exit code `1` (the prototype's rule).

## Development

```sh
cargo test                        # 12+ suites: unit + integration + e2e (all offline)
cargo test --release              # same suite, release profile
cargo clippy --all-targets        # must be clean
cargo fmt --check                 # must be clean
```

The test strategy is fully offline: a small in-process `tokio` mock SSE
server (vLLM-style frames, `[DONE]`, `usage`, early-close, HTTP 500, stalls,
flaky endpoints) stands in for a real inference server. The end-to-end smoke
test (`tests/e2e_test.rs`) runs the full pipeline — single stream + sweep +
NIAH + persistence + all three exports — and also exercises the real binary
as a subprocess (headless `--json` + `--export`, isolated data dir).

### Project layout

```
src/
├── main.rs          CLI dispatch (banner / headless / TUI)
├── lib.rs           crate root
├── config.rs        the single-source-of-truth Config (CLI > env > file > defaults)
├── timing.rs        quanta clock wrapper; T0..Tn milestone stamps
├── sse/             incremental low-alloc SSE parser + chunk model
├── client/          StreamWorker (one SSE stream) + WorkerPool (N streams)
├── metrics/         EngineCore aggregation, HdrHistogram, ArcSwap snapshot
├── prompt/          prompt generator (short/long/nocache) + optional tokenizer
├── engines/
│   ├── speed.rs              Engine A
│   ├── concurrency.rs        Engine B (sweep + knee + envelope)
│   ├── capability/           Engine C (niah / reasoning / structured)
│   └── hardware.rs           Engine D (Joules/Token, fragmentation warning)
├── hw/                100 ms hardware poller (NVML feature-gated + sysinfo)
├── storage/           SQLite (bundled) + JSON/MD/CSV exporters
└── ui/                ratatui dashboard: app state machine, 60 Hz event loop, 5 views
tests/
├── e2e_test.rs        full-pipeline smoke test + binary subprocess tests
├── …                  per-module integration suites (mock SSE server, offline)
```

## License

Internal tooling. See repository history.
