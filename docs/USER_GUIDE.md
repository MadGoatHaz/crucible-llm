# Crucible LLM — User Guide

## First Run vs. Subsequent Runs

### First run

1. Build or download the binary
2. Run `./crucible-llm`
3. You'll first see the **theme picker** (choose a color theme), then the **Setup** screen — the interactive walkthrough (below)

### Subsequent runs

If you saved a config with a **URL + model** on a previous run (via the Config view), `./crucible-llm` opens **straight onto the dashboard** — the Setup walkthrough is skipped. Re-open Setup any time with `c`.

## The Setup Flow (first run)

### Step 1: Server URL
Enter the base URL of your OpenAI-compatible server. Crucible works with **any OpenAI-compatible server** — vLLM, llama.cpp, LM Studio, Unsloth Desktop, SGLang, Ollama, and TGI — anything exposing a `/v1/chat/completions` endpoint.
Examples:
- vLLM: `http://localhost:8000/v1`
- Ollama: `http://localhost:11434/v1`
- llama.cpp: `http://localhost:8080/v1`
- Remote: `http://192.168.1.100:8000/v1`

Press Enter to connect. The app will automatically discover available models.

### Step 2: Model Selection
A list of models appears (fetched from `GET /v1/models`).
- Use ↑↓ to navigate
- Type to filter
- Press Enter to select
- If discovery fails, you can type a model name manually

### Step 3: Configuration
- **Engines**: Toggle which benchmarks to run. Defaults: **A, B, C1, C2, C3, F** (D / energy is opt-in)
- **Mode**: short (quick TTFT test) or long (sustained throughput)
- **Tokens**: Target prompt tokens for `long` mode (default **10000**)
- **Iterations**: How many times to repeat (more = more reliable)
- **Ladder**: Concurrency levels to test (default **1,2,3,4,8,12,16,24,32**)

### Step 4: Launch
Press Enter to start. The TUI switches to the Live Monitor.

## GPU Monitoring (Engine D)

Crucible auto-detects your GPU at startup:
- NVIDIA: uses NVML (full telemetry)
- AMD: uses sysfs/hwmon (util, VRAM, temp, power)
- Intel: uses Level Zero (Arc) or sysfs (integrated)

When a GPU is detected, Engine D is enabled by default and a live telemetry panel appears in the TUI showing power, utilization, temperature, VRAM, and energy efficiency.

No GPU? Engine D is disabled automatically. No N/A panels.

## Default Values

| Setting | Default |
|---------|---------|
| Target tokens | `10000` |
| Concurrency ladder | `1, 2, 3, 4, 8, 12, 16, 24, 32` |
| Engines | A (Speed), B (Concurrency), C1 (NIAH), C2 (Reasoning), C3 (Structured), F (Flat Out) |
| Engine D (Energy) | **auto** — on when a GPU is detected (NVIDIA/AMD/Intel), off when not; explicit engine selection always wins. Must run on the GPU machine. |
| Mode | `short` |
| Iterations | `1` |
| Timeout | `120` s |

## Using the TUI

### Views
| Key | View | Shows |
|-----|------|-------|
| 1 | Live Monitor | Throughput graph, key metrics, event log, engine queue |
| 2 | Concurrency | Sweep results, per-stream speed, practical sweet spot |
| 3 | NIAH | Retrieval matrix by context size |
| 4 | GPU & Power | Multi-GPU telemetry, power graphs, energy cost |
| 5 | History | Past runs: list / detail / compare / delete |
| 6 | Config | Editable settings with per-field explanations |

### Global Controls
| Key | Action |
|-----|--------|
| 1-6 | Switch views (always available) |
| Space | Pause / Resume benchmark |
| c | Re-open the Setup flow |
| e | Export results |
| q | Quit — opens a `[y/N]` confirmation (the **only** quit path) |
| Ctrl+C | **Does not quit** — the terminal owns it for copy selection |

> `Ctrl+C` is intentionally inert so you can select text and copy it with the terminal. To quit, press `q` and confirm with `y`. `Esc` never quits outright.

### Themes

Crucible ships three complete color themes, and the whole TUI re-skins when you switch:

- **Cyberpunk** (default) — neon cyan / electric purple / digital glow
- **Vampire** — crimson / gold / dark purple, gothic
- **Monochrome Pastel** — soft blue / lavender / clean, minimal

**First run:** if you've never picked a theme, a full-screen picker appears before Setup. Move with `↑`/`↓` (the whole screen live-previews the hovered theme) and confirm with `Enter` — it's saved, so later runs skip the picker.

**Change later:** open **Config (View 6)**, focus the **Theme** field, and cycle with `←`/`→` (live preview) or jump with `1`/`2`/`3`; `Esc` saves. You can also set it via the `theme` field in `~/.config/crucible/config.json` or the `CRUCIBLE_THEME` environment variable.

### Reading the Live Monitor (View 1)
- **Throughput graph**: Real-time tokens/sec. Green=fast, red=slow. Vertical lines mark engine transitions.
- **Key Metrics**: Current stats for the active engine
- **Overall Metrics**: Cumulative stats across all engines (max/avg/p5)
- **Event Log**: What's happening right now

### Reading the Concurrency View (View 2)
- **Table**: One row per concurrency level. Shows aggregate AND per-stream speed.
- **Recommendation**: "Practical Sweet Spot" = where each user still gets ≥30 t/s (29.4 t/s effective with the 2% margin)
- **Graph**: Visual curve of throughput vs users

### Reading the NIAH View (View 3)
- **Matrix**: Rows = context size, shows pass rate
- **Colors**: Green ≥80%, Yellow 50-79%, Red <50%
- **Verdict**: "Reliable up to ~Xk tokens"

### Reading the History View (View 5)
The History view reads your stored runs (from `~/.local/share/crucible/benchmarks.db`). It has four modes:

- **List** (default) — one row per stored run: date, model, endpoint, duration. Navigate with `j`/`↓` (down) and `k`/`↑` (up).
- **Detail** (`Enter`) — the selected run's per-engine summary: mean TTFT, gen t/s, prompt t/s, MTP rate, J/token, and the NIAH pass count.
- **Compare** (`C`) — puts the cursor's run on the left (A), then `Enter` picks a second run (B). Shows a side-by-side table of **signed % deltas**:
  - **Green ▲** = improvement, **Red ▼** = regression, **—** = no change
  - TTFT and J/token improve when **lower**; gen t/s, prompt t/s, and MTP improve when **higher**
- **Delete** (`D`) — a `[y/N]` prompt before removing the selected run
- `Esc` — back to the List from Detail / Compare / Delete

> The list is empty ("No benchmark runs saved yet") until your first run completes.

### Using the Config View (View 6)
The Config view has an **edit gate**: you land on a read-only screen showing the current settings. It protects you from accidentally typing into a field while just viewing.

- `Enter` — enter **edit mode** (the fields become live)
- While editing:
  - `Tab` / `Shift-Tab` / `↑` / `↓` — move between fields
  - Type characters to edit a field (numbers for numeric fields)
  - `Space` / `Enter` — toggle a boolean / cycle the mode
  - `←` / `→` / `-` / `+` — step a number
  - `Backspace` — delete from the focused field
- `Esc` — **save and stay** on the Config view (returns to the gate)
- `F2` — save to `~/.config/crucible/config.json`
- `F5` — run the selected engines (jumps to the Live view)
- `R` — reset all fields to the built-in defaults
- Each focused field shows a dimmed `ℹ` **explanation** below it

#### Field explanations (shown in the UI)
| Field | What it does |
|-------|--------------|
| Target URL | Base URL of your OpenAI-compatible server; must be reachable from this machine |
| Model | The model to benchmark; usually auto-detected, must match exactly (case-sensitive) |
| Mode | `short` = ~50-token prompt (tests TTFT); `long` = padded to your token target (tests sustained throughput) |
| Target tokens | Target prompt tokens for `long` mode (default 10000) — how big the input document is |
| Iterations | Repeats — 1 = quick, 3-5 = reliable, 10+ = publication-grade |
| Timeout | Max seconds to wait; the stream is killed if silent this long (default 120s) |
| API key | Bearer token for authenticated servers; local servers don't need it; not persisted |
| Cache bypass | ON = cold-start (real first-request perf); OFF = warm (steady-state) |
| Tokenizer | Path to a HuggingFace `tokenizer.json` for exact counts; without it, `chars/4` estimate |
| Concurrency ladder | Engine B levels, comma-separated (default `1,2,3,4,8,12,16,24,32`) |
| Hardware telemetry | Engine D energy poller — auto on when a GPU is detected; shows the detected GPU (e.g. `[✓] Energy — RTX 4090`) |
| Theme | TUI color theme — Cyberpunk / Vampire / Monochrome Pastel; `←`/`→` cycles, `1`/`2`/`3` selects |
| Engine A–F | Toggle each benchmark on/off (D / energy auto-selects with GPU detection; your explicit choice always wins) |
| $/kWh | Electricity rate for cost estimation in the GPU & Power view (default $0.15) |

## GPU & Power Monitor (Tab 4)

Shows comprehensive hardware telemetry during and after benchmark runs:

- **System Power**: Total (GPU + CPU), idle, compute, and peak power draw; energy (kWh); estimated cost; and run duration — which **freezes at its final value when the benchmark completes** (shows `--` when idle)
- **Power Graph**: Real-time power over time (1 Hz sampling), auto-scaled, with mean and peak markers
- **Per-GPU Table**: Full names (no truncation), individual power, utilization, avg utilization, temp, avg temp, max temp, VRAM, clocks for each GPU
- **Auto-scaled Graphs**: Utilization and temperature over time render against the data's real range (a narrow 85–97% band shows its variation, not a flat line), each with grid lines, an area fill, a dashed mean line, and a peak marker
- **Efficiency**: Joules/token, tokens/watt, $/1M tokens (blended)
- **Energy**: Total kWh consumed during the run (GPU + CPU)
- **Idle Baseline**: Auto 5s before the test; press `[i]` for a manual 10s measurement
- **CPU Power**: RAPL / Super I/O / estimated (included in total system draw)

Set your electricity rate ($/kWh) in Config (tab 6) for accurate cost estimates.

### Cost Analysis

The GPU & Power tab shows your local inference cost in the industry-standard format: **$ per 1M tokens**, split into input (prefill) and output (decode), plus a token-weighted blended rate.

- **$/1M Input (prefill)**: Energy during the prefill phase ÷ prompt tokens
- **$/1M Output (decode)**: Energy during the decode phase ÷ completion tokens
- **$/1M Blended**: Token-weighted mean of input and output (always between the two)
- **Your rate**: Set your electricity cost ($/kWh) in Config [6]. Default: $0.16.
- **Model**: Throughput-based energy (total phase tokens × measured phase power ÷ measured phase throughput), with a power×duration fallback for single-request runs.

### Duration & Idle Baseline

- **Duration**: Starts when the first engine begins, freezes at its final value when the run completes. Shows `--` when idle (no active run).
- **Auto idle baseline**: A 5-second no-load measurement is taken before each test starts.
- **Manual idle baseline**: Press `[i]` in the GPU tab for a fresh 10-second no-load measurement (useful after a run to re-establish the floor).
- **Avg power**: Calculated over the active test window only — not from app launch.

### Power Accuracy

GPU power readings are plausibility-clamped to prevent phantom values:
- NVIDIA: capped at 1.5× factory TDP (from `power_management_limit_default`)
- All vendors: global ceiling of 1000W
- Anomalous readings are logged and capped automatically

Average power is calculated over the active test window only (not from app launch). Press `[i]` in the GPU tab to take a fresh idle baseline measurement.

### CPU Power

Crucible also monitors CPU power and includes it in total system energy:
- **Intel**: RAPL (`/sys/class/powercap/intel-rapl/energy_uj`)
- **AMD**: Super I/O chip (motherboard voltage × current sensors)
- **Fallback**: Estimated 40W (labeled as such)

The system power line shows: `System: 2,920 W (GPU 2,847 + CPU 73)`

Your $/1M token cost reflects TRUE system draw, not just the GPU.

## Headless / CLI Mode

For scripting, CI/CD, or when you don't want a TUI:

```bash
# Basic speed test
./crucible-llm --headless --url http://localhost:8000/v1 --model my-model

# JSON output (for parsing)
./crucible-llm --headless --url http://localhost:8000/v1 --model my-model --json

# Specific engines only
./crucible-llm --headless --url http://localhost:8000/v1 --model my-model --engine speed --engine niah

# Export results
./crucible-llm --headless --url http://localhost:8000/v1 --model my-model --export json --export-path results.json

# With API key
./crucible-llm --headless --url https://api.example.com/v1 --model gpt-4 --api-key sk-...

# Custom timeout
./crucible-llm --headless --url http://localhost:8000/v1 --model my-model --timeout 60
```

### Exit Codes
| Code | Meaning |
|------|---------|
| 0 | Success (all selected engines completed) |
| 1 | Failure (one or more engines failed) |
| 2 | Configuration error |

### CI/CD Example (GitHub Actions)
```yaml
- name: Benchmark LLM
  run: |
    ./crucible-llm --headless \
      --url ${{ secrets.LLM_URL }} \
      --model ${{ secrets.LLM_MODEL }} \
      --json \
      --export json \
      --export-path ./benchmark-results.json
    echo "Benchmark complete"
```

## Exporting Results

### From TUI
Press `e` during or after a run. Choose format: JSON, Markdown, or CSV.

### From CLI
```bash
--export json|md|csv --export-path /path/to/file
```

### Default Export Location
`~/.local/share/crucible/exports/crucible-<session-id>.<ext>`

## Understanding Your Results

### Is my server fast enough?
- **Chat use**: 30+ t/s generation, TTFT < 500ms = comfortable
- **Coding agents**: 50+ t/s, TTFT < 300ms = productive
- **RAG pipelines**: Prompt throughput > 1000 t/s = efficient context loading

### How many users can I serve?
Look at the Concurrency view's "Practical Sweet Spot." That's your answer.
- Sweet spot of 8 = 8 simultaneous users at comfortable speed (each ≥ 30 t/s)
- Each additional user beyond that makes everyone slower

### What's the headline number?
That's **Flat Out (Engine F)** — it uses your server's optimal user count (from the Concurrency sweet spot) for a **60-second full-load test**. The aggregate t/s it reports is the single number to quote when comparing servers or configurations. If you didn't run Concurrency first, it falls back to a small default load (3 streams).

### Can I trust this model?
- **Reasoning ≥ 90%**: Yes, for most tasks
- **NIAH reliable to 16k+**: Yes, for RAG with moderate context
- **Structured 3/3**: Yes, for API integration
- Any of these failing? The model has limitations for that use case.

### Are the token counts accurate?
Yes. Crucible uses the **server-reported `usage.completion_tokens`** as the authoritative count, with a re-tokenization fallback (your exact tokenizer, or a `chars/4` estimate) when a stream is cut off before a usage frame arrives. It never counts raw SSE frames — which matters because batched servers like vLLM pack ~2.4 tokens into one frame. The result is accurate across vLLM, llama.cpp, LM Studio, SGLang, Ollama, and TGI.

## Troubleshooting

### Connection refused
- Is your server running? Check: `curl http://your-host:8000/v1/models`
- Is the URL correct? Should end with `/v1`

### Timeout
- Increase with `--timeout 300` (seconds)
- Check if your server is overloaded

### No models discovered
- Server might not implement `/v1/models` endpoint
- Type the model name manually in the setup screen

### Energy shows N/A
- You're not on the GPU machine (all GPU telemetry — NVML, sysfs, Level Zero — is local), OR
- No supported GPU driver found (NVIDIA / AMD / Intel), OR
- There is no GPU at all — in that case the GPU panel is simply hidden (by design, not an error) and Engine D stays off

### Setup appears every time
- You don't have a saved config with a URL + model yet. Save one from the Config view (View 6 → `Enter` → `F2`), and the next run opens straight to the dashboard.

### Ctrl+C doesn't quit
- By design. `Ctrl+C` is reserved for the terminal's copy selection. Press `q` and confirm with `y` to quit.

### Numbers stop updating after a run
- The metrics pipeline **freezes on completion** so your final numbers are stable. Start a new run (`r` / `F5` in the Config view) to re-arm the live telemetry.

## System Requirements

- **Minimum**: Any terminal (Linux, macOS, WSL2, Windows via WSL)
- **For GPU monitoring (Engine D)**: Must run on the GPU machine — NVIDIA (NVML, on by default), AMD (amdgpu sysfs), or Intel (Level Zero / i915/xe) drivers. No GPU = Engine D auto-off, everything else works normally
- **Binary size**: ~14 MB (static, no dependencies)
- **Memory**: < 100 MB RAM for the tool itself
