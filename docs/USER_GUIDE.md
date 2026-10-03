# Crucible LLM — User Guide

## First Run vs. Subsequent Runs

### First run

1. Build or download the binary
2. Run `./crucible-llm`
3. You'll see the **Setup** screen — the interactive walkthrough (below)

### Subsequent runs

If you saved a config with a **URL + model** on a previous run (via the Config view), `./crucible-llm` opens **straight onto the dashboard** — the Setup walkthrough is skipped. Re-open Setup any time with `c`.

## The Setup Flow (first run)

### Step 1: Server URL
Enter the base URL of your OpenAI-compatible server.
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
- **Engines**: Toggle which benchmarks to run. Defaults: **A, B, C1, C2, C3** (D / energy is opt-in)
- **Mode**: short (quick TTFT test) or long (sustained throughput)
- **Tokens**: Target prompt tokens for `long` mode (default **10000**)
- **Iterations**: How many times to repeat (more = more reliable)
- **Ladder**: Concurrency levels to test (default **1,2,3,4,8,12,16,24,32**)

### Step 4: Launch
Press Enter to start. The TUI switches to the Live Monitor.

## Default Values

| Setting | Default |
|---------|---------|
| Target tokens | `10000` |
| Concurrency ladder | `1, 2, 3, 4, 8, 12, 16, 24, 32` |
| Engines | A (Speed), B (Concurrency), C1 (NIAH), C2 (Reasoning), C3 (Structured) |
| Engine D (Energy) | **off** (opt-in — run it on the GPU machine) |
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
| 4 | History | Past runs: list / detail / compare / delete |
| 5 | Config | Editable settings with per-field explanations |

### Global Controls
| Key | Action |
|-----|--------|
| 1-5 | Switch views (always available) |
| Space | Pause / Resume benchmark |
| c | Re-open the Setup flow |
| e | Export results |
| q | Quit — opens a `[y/N]` confirmation (the **only** quit path) |
| Ctrl+C | **Does not quit** — the terminal owns it for copy selection |

> `Ctrl+C` is intentionally inert so you can select text and copy it with the terminal. To quit, press `q` and confirm with `y`. `Esc` never quits outright.

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

### Reading the History View (View 4)
The History view reads your stored runs (from `~/.local/share/crucible/benchmarks.db`). It has four modes:

- **List** (default) — one row per stored run: date, model, endpoint, duration. Navigate with `j`/`↓` (down) and `k`/`↑` (up).
- **Detail** (`Enter`) — the selected run's per-engine summary: mean TTFT, gen t/s, prompt t/s, MTP rate, J/token, and the NIAH pass count.
- **Compare** (`C`) — puts the cursor's run on the left (A), then `Enter` picks a second run (B). Shows a side-by-side table of **signed % deltas**:
  - **Green ▲** = improvement, **Red ▼** = regression, **—** = no change
  - TTFT and J/token improve when **lower**; gen t/s, prompt t/s, and MTP improve when **higher**
- **Delete** (`D`) — a `[y/N]` prompt before removing the selected run
- `Esc` — back to the List from Detail / Compare / Delete

> The list is empty ("No benchmark runs saved yet") until your first run completes.

### Using the Config View (View 5)
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
| Mode | `short` = ~100-token prompt (tests TTFT); `long` = your token target (tests sustained throughput) |
| Target tokens | `max_tokens` per request — 256 = quick, 10000 = standard, 8192+ = stress |
| Iterations | Repeats — 1 = quick, 3-5 = reliable, 10+ = publication-grade |
| Timeout | Max seconds to wait; the stream is killed if silent this long (default 120s) |
| API key | Bearer token for authenticated servers; local servers don't need it; not persisted |
| Cache bypass | ON = cold-start (real first-request perf); OFF = warm (steady-state) |
| Tokenizer | Path to a HuggingFace `tokenizer.json` for exact counts; without it, `chars/4` estimate |
| Concurrency ladder | Engine B levels, comma-separated (default `1,2,3,4,8,12,16,24,32`) |
| Hardware telemetry | Engine D energy poller (opt-in; reports N/A without a GPU driver) |
| Engine A–C3 | Toggle each benchmark on/off |

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

### Can I trust this model?
- **Reasoning ≥ 90%**: Yes, for most tasks
- **NIAH reliable to 16k+**: Yes, for RAG with moderate context
- **Structured 3/3**: Yes, for API integration
- Any of these failing? The model has limitations for that use case.

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
- You're not on the GPU machine, OR
- Built without `--features nvml`, OR
- No supported GPU driver found

### Setup appears every time
- You don't have a saved config with a URL + model yet. Save one from the Config view (View 5 → `Enter` → `F2`), and the next run opens straight to the dashboard.

### Ctrl+C doesn't quit
- By design. `Ctrl+C` is reserved for the terminal's copy selection. Press `q` and confirm with `y` to quit.

### Numbers stop updating after a run
- The metrics pipeline **freezes on completion** so your final numbers are stable. Start a new run (`r` / `F5` in the Config view) to re-arm the live telemetry.

## System Requirements

- **Minimum**: Any terminal (Linux, macOS, WSL2, Windows via WSL)
- **For Energy monitoring**: Must run on the GPU machine with appropriate drivers
- **Binary size**: ~14 MB (static, no dependencies)
- **Memory**: < 100 MB RAM for the tool itself
