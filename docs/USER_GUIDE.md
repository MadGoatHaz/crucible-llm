# Crucible LLM — User Guide

## First Run

1. Build or download the binary
2. Run `./crucible-llm`
3. You'll see the Setup screen

## The Setup Flow

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
- **Engines**: Toggle which benchmarks to run (A-D). Defaults: A, B, C1, C2, C3
- **Mode**: short (quick TTFT test) or long (sustained throughput)
- **Tokens**: Target output length per request
- **Iterations**: How many times to repeat (more = more reliable)
- **Ladder**: Concurrency levels to test (for Engine B)

### Step 4: Launch
Press Enter to start. The TUI switches to the Live Monitor.

## Using the TUI

### Views
| Key | View | Shows |
|-----|------|-------|
| 1 | Live Monitor | Throughput graph, key metrics, event log |
| 2 | Concurrency | Sweep results, per-stream speed, sweet spot |
| 3 | NIAH | Retrieval matrix by context size |
| 4 | History | Compare past runs |
| 5 | Config | View/edit settings (Enter to edit) |

### Controls
| Key | Action |
|-----|--------|
| 1-5 | Switch views (always available) |
| Space | Pause/Resume benchmark |
| q | Quit (with confirmation) |
| e | Export results |

### Reading the Live Monitor
- **Throughput graph**: Real-time tokens/sec. Green=fast, red=slow. Vertical lines mark engine transitions.
- **Key Metrics**: Current stats for the active engine
- **Overall Metrics**: Cumulative stats across all engines (max/avg/p5)
- **Event Log**: What's happening right now

### Reading the Concurrency View
- **Table**: One row per concurrency level. Shows aggregate AND per-stream speed.
- **Recommendation**: "Practical Sweet Spot" = where each user still gets ≥40 t/s
- **Graph**: Visual curve of throughput vs users

### Reading the NIAH View
- **Matrix**: Rows = context size, shows pass rate
- **Colors**: Green ≥80%, Yellow 50-79%, Red <50%
- **Verdict**: "Reliable up to ~Xk tokens"

### Reading Capability Scores
- Horizontal bars with percentage
- Color-coded: Green (good), Yellow (okay), Red (poor), Gray (N/A)
- Overall verdict at bottom

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
- Sweet spot of 8 = 8 simultaneous users at comfortable speed
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

## System Requirements

- **Minimum**: Any terminal (Linux, macOS, WSL2, Windows via WSL)
- **For Energy monitoring**: Must run on the GPU machine with appropriate drivers
- **Binary size**: ~14 MB (static, no dependencies)
- **Memory**: < 100 MB RAM for the tool itself
