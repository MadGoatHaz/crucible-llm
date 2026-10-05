I built **Crucible LLM** — a terminal-based benchmarking suite that stress-tests any OpenAI-compatible inference server and measures what a *real user* actually experiences. One static Rust binary. No Python. No venv. No dependencies.

## Why it exists

Most benchmarks shout "xx.x tokens/sec!" and stop. But when 32 users hit that server at once, each one gets 2 tokens/sec. A number that looks fast in the aggregate can feel painfully slow per user. Crucible measures **that gap** — the difference between a server's peak throughput and what each individual connection actually gets — as well as how competent the LLM is at real tasks.

## What it does

* **Single static binary** — built in Rust, runs from any terminal on any box. No install, no runtime, no system libraries.
* **TUI-first** — a 60 Hz interactive dashboard with real-time throughput graphs, live model discovery, and per-engine metrics.
* **Seven benchmark engines** covering the full arc: speed → concurrency → long-context retrieval → reasoning → structured output → energy → flat-out max throughput.
* **The "practical sweet spot"** — not just where aggregate throughput plateaus, but the maximum number of users where *each one* still gets a comfortable ≥ 30 tokens/sec.
* **Multi-vendor GPU monitoring** — auto-detects NVIDIA, AMD, or Intel. Live power, utilization, VRAM, temperature, and energy-per-token.
* **Any OpenAI-compatible server** — vLLM, Ollama, llama.cpp, LM Studio, SGLang, Unsloth, TGI. Point it at a `/v1` endpoint and go.
* **3 visual themes** — Cyberpunk, Vampire, or Monochrome Pastel. Pick on first run, change anytime.
* **Headless / CI-ready** — pure-JSON output, meaningful exit codes, and export to JSON, Markdown, or CSV for automated regression pipelines.

## See it in action

*Live Monitor screenshot — real-time throughput graph with auto-scaling y-axis, engine-transition markers, and live event log.*

![Live Monitor|551x550](upload://pJiDgYUNFfovKBY2cxj9r28q0Pq.png)

## Try it

Grab the latest pre-compiled binary from the [Releases](https://github.com/MadGoatHaz/crucible-llm/releases) page:

```
chmod +x crucible-llm-linux-x86_64
./crucible-llm-linux-x86_64
```

Or build from source (requires a stable Rust toolchain):

```
git clone https://github.com/MadGoatHaz/crucible-llm
cd crucible-llm
cargo build --release
./target/release/crucible-llm
```

On first run, the TUI asks you to pick a theme, then walks you through entering your server URL and picking a model.

---

## — **Crucible LLM v0.1.3 — GPU & Power Monitor**

**View 6: Dedicated Hardware Telemetry Tab**
A new tab that puts the entire power and energy picture on one screen:
- **Multi-GPU table** — per-card power, utilization, temp, VRAM, clocks, throttle (built for 8-GPU servers)
- **Power-over-time graph** — 1 Hz sampling during your run, rendered live
- **Idle baseline** — measures power before the test; compute power = total − idle
- **Energy & cost** — kWh consumed + estimated $ cost (set your rate in Config)
- **Efficiency** — J/token, tokens/watt, $/1M tokens
- **Peak tracking** — max power and max temp across the entire run

Point it at your 8× A4000 box and watch every card in real-time.

**Download**
https://github.com/MadGoatHaz/crucible-llm/releases/tag/v0.1.3

---

## — **Crucible LLM v0.1.2 — GPU Monitoring for Everyone**

**Multi-Vendor GPU Telemetry**
Crucible now auto-detects your GPU and shows live hardware monitoring — zero config:
- **NVIDIA** — full NVML telemetry (power, util, VRAM, temp, clocks, throttle reasons)
- **AMD** — sysfs/hwmon (util, VRAM, temp, power, clocks)
- **Intel** — Level Zero Sysman for Arc GPUs, sysfs fallback for integrated

No GPU? No panel. No N/A. Just clean.

**NVIDIA On by Default**
In v0.1.1, NVIDIA monitoring was behind a build flag. Now it's just... on. `cargo build` gives you full GPU telemetry.

**Energy Efficiency**
Joules-per-token calculated live during any benchmark. Compare quantization levels, model sizes, or hardware on actual energy cost.

**Download**
https://github.com/MadGoatHaz/crucible-llm/releases/tag/v0.1.2

---

## — **Crucible LLM v0.1.1 — Big Update**

**3 Visual Themes**
Pick your vibe. Cyberpunk (cyan/purple/neon) is the default. Vampire (crimson/gold/gothic) for the dark aesthetic. Monochrome Pastel (soft/clean) for minimalists. First run asks you to pick — live preview as you arrow between them. Change anytime from Config.

**Flat Out Engine**
The number you want to quote. 60 seconds at your server's optimal concurrency (the sweet spot). Your real-world maximum throughput with all users loaded.

**Accurate Token Counting**
Turns out vLLM batches ~2.4 tokens per SSE frame. I was counting frames. That's a 3x undercount. Fixed. I now use the server's own token count (`usage.completion_tokens`) as the authoritative source.

**Works With Everything**
Confirmed: vLLM, llama.cpp, LM Studio, Unsloth Desktop. Any OpenAI-compatible endpoint. If it speaks `/v1/chat/completions`, I can benchmark it.

**30 T/S Sweet Spot**
The number that actually matters. Concurrency sweep identifies your "Practical Sweet Spot" — the most users where each still gets ≥30 tokens/sec. Not the max-throughput knee. The real deployment number.

**Methodology Transparency**
Every JSON export includes a methodology block: exact formulas, timing resolution (nanoseconds), token counting method.

**QoL**
TUI is the default. Settings persist. Ctrl+C copies (doesn't quit). Full run logs. History view with compare. Decode loop guard.

**Download**
https://github.com/MadGoatHaz/crucible-llm/releases/tag/v0.1.1

---

## Changelog

| Version | Date | Highlights |
|---------|------|-----------|
| **0.1.3** | 2026-10-05 | GPU & Power Monitor (View 6): multi-GPU table, power-over-time graphs, energy (kWh), cost estimation, efficiency metrics |
| **0.1.2** | 2026-10-04 | Multi-vendor GPU monitoring (NVIDIA/AMD/Intel), auto-detection, NVIDIA default, full TUI hardware panel, J/token |
| **0.1.1** | 2026-10-03 | 3 themes, Flat Out engine, accurate token counting, multi-backend support, 30 t/s sweet spot, methodology transparency, loop guard, history view |
| **0.1.0** | 2026-10-01 | Initial release. 6 engines, TUI, headless CLI, SQLite history, export, concurrency sweep |

---

## Contribute

👉 **[github.com/MadGoatHaz/crucible-llm](https://github.com/MadGoatHaz/crucible-llm)** — star the repo, open an issue, or fork and contribute.

**Feedback:** If you're running any of these backends, hit me up with your numbers. I want to know if the sweet spot and Flat Out results match what you're seeing. Issues, PRs, and war stories all welcome.
