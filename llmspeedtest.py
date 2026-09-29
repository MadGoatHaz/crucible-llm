#!/usr/bin/env python3
"""
llmspeedtest.py — Zero-dependency LLM endpoint speed tester.

Measures TTFT, prompt processing speed, token generation speed,
and MTP efficiency for any OpenAI-compatible streaming endpoint.

No external packages required. Uses only Python stdlib.

Usage:
    python3 llmspeedtest.py --url http://localhost:8080/v1/chat/completions
    python3 llmspeedtest.py --model mymodel --tokens 4000 --iterations 3
    python3 llmspeedtest.py --json 2>/dev/null | jq '.results[0].tg_speed'
"""

import argparse
import json
import os
import signal
import sys
import time
import uuid
from dataclasses import dataclass, field
from http.client import HTTPConnection, HTTPSConnection
from typing import Any, Dict, List, Optional, Tuple


# ─── Constants ────────────────────────────────────────────────────────────────

MIN_PYTHON = (3, 8)

DEFAULT_URL = "http://192.168.51.163:8080/v1/chat/completions"
DEFAULT_MODEL = "default"
DEFAULT_TIMEOUT = 120
DEFAULT_TOKENS = 2000
DEFAULT_MAX_GEN_TOKENS = 256

# ANSI colors (disabled automatically when not TTY)
class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    BG_CYAN = "\033[46m"


# ─── Data Structures ──────────────────────────────────────────────────────────

@dataclass
class TestResult:
    """Metrics from a single speed-test run."""
    ttft: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    pp_speed: float = 0.0
    tg_speed: float = 0.0
    mtp_efficiency: float = 0.0
    stream_time: float = 0.0
    total_chunks: int = 0
    content_chunks: int = 0
    reasoning_chunks: int = 0
    other_chunks: int = 0
    estimated: bool = False
    model: str = ""
    mode: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ttft_s": round(self.ttft, 4),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "pp_speed_tok_s": round(self.pp_speed, 2),
            "tg_speed_tok_s": round(self.tg_speed, 2),
            "mtp_efficiency": round(self.mtp_efficiency, 4),
            "stream_time_s": round(self.stream_time, 4),
            "total_chunks": self.total_chunks,
            "content_chunks": self.content_chunks,
            "reasoning_chunks": self.reasoning_chunks,
            "estimated": self.estimated,
            "model": self.model,
            "mode": self.mode,
            "error": self.error,
        }


# ─── Dependency Self-Check ────────────────────────────────────────────────────

def check_environment() -> None:
    """Verify Python version and required stdlib modules are available."""
    if sys.version_info < MIN_PYTHON:
        print(
            f"ERROR: Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required. "
            f"You have {sys.version_info.major}.{sys.version_info.minor}.\n"
            f"Install a newer Python: sudo apt install python3 "
            f"(Debian/Ubuntu) or your distro equivalent.",
            file=sys.stderr,
        )
        sys.exit(1)

    # These are always in stdlib, but be defensive
    try:
        import http.client
        import urllib.request
    except ImportError as e:
        print(
            f"ERROR: Missing stdlib module: {e}\n"
            f"This should not happen. Your Python installation may be corrupt.\n"
            f"Try: sudo apt install --reinstall python3",
            file=sys.stderr,
        )
        sys.exit(1)


# ─── Color Helper ─────────────────────────────────────────────────────────────

class Output:
    """TTY-aware output handler."""

    def __init__(self, force_color: Optional[bool] = None) -> None:
        self.use_color = (
            force_color
            if force_color is not None
            else sys.stdout.isatty()
        )
        self.is_tty = sys.stdout.isatty()

    def _c(self, code: str, text: str) -> str:
        if self.use_color:
            return f"{code}{text}{C.RESET}"
        return text

    def info(self, msg: str) -> None:
        print(self._c(C.CYAN, "ℹ "), msg, file=sys.stderr)

    def success(self, msg: str) -> None:
        print(self._c(C.GREEN, "✓ "), msg, file=sys.stderr)

    def warning(self, msg: str) -> None:
        print(self._c(C.YELLOW, "⚠ "), msg, file=sys.stderr)

    def error(self, msg: str) -> None:
        print(self._c(C.RED, "✗ "), msg, file=sys.stderr)

    def dim(self, msg: str) -> None:
        print(self._c(C.DIM, msg), file=sys.stderr)

    def bold(self, msg: str) -> None:
        print(self._c(C.BOLD, msg))

    def progress(self, msg: str) -> None:
        """Live progress (only on TTY, overwrites line)."""
        if self.is_tty:
            sys.stderr.write(f"\r\033[K{msg}")
            sys.stderr.flush()
        # Non-TTY: skip (avoid spam)


# ─── Prompt Generation ────────────────────────────────────────────────────────

def generate_prompt(
    mode: str,
    target_tokens: int,
    nocache: bool = False,
) -> str:
    """Generate a test prompt of the target size.

    Args:
        mode: "short" for ~50 tokens, "long" for target_tokens.
        target_tokens: Target token count for long mode.
        nocache: Prepend random hex to bypass KV cache.

    Returns:
        The prompt string.
    """
    if mode == "short":
        text = (
            "Write a brief explanation of how a B-tree database index works, "
            "including when it's preferred over a hash index."
        )
    else:
        # Build a realistic-looking long prompt by repeating varied content
        base_sentences = [
            "The quick brown fox jumps over the lazy dog.",
            "Python is a high-level interpreted programming language.",
            "Machine learning models require careful data preprocessing.",
            "Distributed systems must handle network partitions gracefully.",
            "The compiler optimizes intermediate representation for speed.",
            "Async I/O allows concurrent non-blocking operations.",
            "Vector databases store embeddings for similarity search.",
            "Kubernetes orchestrates containerized microservices at scale.",
            "The kernel manages memory, processes, and device drivers.",
            "Type systems prevent entire classes of runtime errors.",
        ]
        # Repeat until we hit approximately target_tokens * 4 chars
        # (rough: 1 token ≈ 4 chars for English)
        target_chars = target_tokens * 4
        parts: List[str] = []
        total = 0
        i = 0
        while total < target_chars:
            parts.append(base_sentences[i % len(base_sentences)])
            total += len(parts[-1]) + 1
            i += 1
        text = " ".join(parts)

    if nocache:
        text = f"[{uuid.uuid4().hex}] {text}"

    return text


# ─── SSE Stream Reader ────────────────────────────────────────────────────────

def stream_sse(
    url: str,
    payload: Dict[str, Any],
    api_key: Optional[str],
    timeout: int,
) -> Tuple[List[Dict[str, Any]], float, float, List[str]]:
    """Send a streaming chat-completion request and parse SSE frames.

    Args:
        url: Full endpoint URL.
        payload: JSON body to POST.
        api_key: Optional Bearer token.
        timeout: Connection + read timeout in seconds.

    Returns:
        (chunks, ttft, total_time, errors)
        - chunks: list of parsed JSON objects from data: frames
        - ttft: time to first content token (seconds)
        - total_time: total stream duration (seconds)
        - errors: list of error strings (malformed frames, etc.)
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = parsed.path or "/"
    if parsed.query:
        path += f"?{parsed.query}"

    is_https = parsed.scheme == "https"

    # Build headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Cache-Control": "no-cache",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    body = json.dumps(payload)

    # Open connection
    if is_https:
        conn = HTTPSConnection(host, port, timeout=timeout)
    else:
        conn = HTTPConnection(host, port, timeout=timeout)

    t_start = time.perf_counter()
    t_first_token: Optional[float] = None

    try:
        conn.request("POST", path, body=body, headers=headers)
        resp = conn.getresponse()

        if resp.status != 200:
            error_body = resp.read().decode("utf-8", errors="replace")
            raise ConnectionError(
                f"HTTP {resp.status}: {error_body[:500]}"
            )

        content_type = resp.getheader("Content-Type", "")

        # If server returns plain JSON (not SSE), handle that
        if "application/json" in content_type and "event-stream" not in content_type:
            raw = resp.read().decode("utf-8", errors="replace")
            t_end = time.perf_counter()
            try:
                data = json.loads(raw)
                # Non-streaming response — extract what we can
                chunks = []
                usage = data.get("usage", {})
                if usage:
                    chunks.append({"__usage__": usage})
                choices = data.get("choices", [])
                if choices:
                    msg = choices[0].get("message", {})
                    if msg.get("content"):
                        chunks.append({"choices": [{"delta": {"content": msg["content"]}}]})
                return chunks, t_end - t_start, t_end - t_start, []
            except json.JSONDecodeError:
                raise ConnectionError(f"Invalid JSON response: {raw[:200]}")

        # Parse SSE stream line by line
        chunks: List[Dict[str, Any]] = []
        errors: List[str] = []
        data_buffer: List[str] = []

        while True:
            line_bytes = resp.readline()
            if not line_bytes:
                break

            line = line_bytes.decode("utf-8", errors="replace")

            # Strip BOM if present
            if line.startswith("\ufeff"):
                line = line[1:]

            line = line.rstrip("\n\r")

            # Empty line = frame separator
            if line == "":
                if data_buffer:
                    data_str = "\n".join(data_buffer)
                    data_buffer = []

                    if data_str == "[DONE]":
                        break

                    try:
                        obj = json.loads(data_str)
                        chunks.append(obj)

                        # Check for first content token
                        if t_first_token is None:
                            choices = obj.get("choices", [])
                            if choices:
                                delta = choices[0].get("delta", {})
                                if delta.get("content") or delta.get("reasoning_content"):
                                    t_first_token = time.perf_counter() - t_start
                    except json.JSONDecodeError:
                        errors.append(f"Malformed JSON: {data_str[:100]}")
                continue

            # data: field
            if line.startswith("data:"):
                data_value = line[5:]
                if data_value.startswith(" "):
                    data_value = data_value[1:]
                data_buffer.append(data_value)
                continue

            # event:, id:, retry: — ignore per SSE spec
            # (we only care about data:)

        t_end = time.perf_counter()
        total_time = t_end - t_start
        ttft = t_first_token if t_first_token is not None else total_time

        return chunks, ttft, total_time, errors

    finally:
        conn.close()


# ─── Single Test Run ──────────────────────────────────────────────────────────

def run_single_test(
    url: str,
    model: str,
    prompt: str,
    api_key: Optional[str],
    timeout: int,
    max_tokens: int,
    out: Output,
) -> TestResult:
    """Execute one speed-test run against the endpoint.

    Args:
        url: Endpoint URL.
        model: Model name.
        prompt: The test prompt.
        api_key: Optional API key.
        timeout: Timeout in seconds.
        max_tokens: Max generation tokens.
        out: Output handler.

    Returns:
        TestResult with all metrics populated.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    result = TestResult(model=model, mode=prompt[:20] + "…")

    try:
        chunks, ttft, total_time, errors = stream_sse(
            url, payload, api_key, timeout
        )
    except ConnectionRefusedError:
        result.error = "Connection refused — is the server running?"
        return result
    except ConnectionError as e:
        result.error = str(e)
        return result
    except TimeoutError:
        result.error = f"Timed out after {timeout}s"
        return result
    except OSError as e:
        result.error = f"Network error: {e}"
        return result

    if not chunks:
        result.error = "No data received from server"
        return result

    # Extract usage (from the final chunk that carries it)
    prompt_tokens = 0
    completion_tokens = 0
    estimated = False

    for chunk in chunks:
        usage = chunk.get("usage") or chunk.get("__usage__")
        if usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)

    # Fallback estimation if no usage reported
    if prompt_tokens == 0:
        prompt_tokens = max(1, len(prompt) // 4)
        estimated = True
    if completion_tokens == 0:
        # Estimate from content chunks
        content_chars = 0
        for chunk in chunks:
            choices = chunk.get("choices", [])
            if choices:
                delta = choices[0].get("delta", {})
                content_chars += len(delta.get("content", ""))
        completion_tokens = max(1, content_chars // 4)
        estimated = True

    # Count chunk types
    content_chunks = 0
    reasoning_chunks = 0
    other_chunks = 0
    for chunk in chunks:
        if "__usage__" in chunk:
            other_chunks += 1
            continue
        choices = chunk.get("choices", [])
        if not choices:
            other_chunks += 1
            continue
        delta = choices[0].get("delta", {})
        if delta.get("content"):
            content_chunks += 1
        elif delta.get("reasoning_content"):
            reasoning_chunks += 1
        else:
            other_chunks += 1

    total_chunks = len(chunks)

    # Compute metrics
    generation_time = total_time - ttft
    if generation_time <= 0:
        generation_time = 0.001  # avoid division by zero

    result.ttft = ttft
    result.prompt_tokens = prompt_tokens
    result.completion_tokens = completion_tokens
    result.pp_speed = prompt_tokens / ttft if ttft > 0 else 0
    result.tg_speed = completion_tokens / generation_time
    result.mtp_efficiency = (
        completion_tokens / content_chunks if content_chunks > 0 else 0
    )
    result.stream_time = total_time
    result.total_chunks = total_chunks
    result.content_chunks = content_chunks
    result.reasoning_chunks = reasoning_chunks
    result.other_chunks = other_chunks
    result.estimated = estimated

    if errors:
        result.error = f"{len(errors)} malformed chunk(s) skipped"

    return result


# ─── Output Formatters ────────────────────────────────────────────────────────

def print_result_box(result: TestResult, out: Output, iteration: int = 0, total: int = 1) -> None:
    """Print a formatted results box to stdout."""
    est_tag = " [ESTIMATED]" if result.estimated else ""

    if result.error and result.completion_tokens == 0:
        out.error(f"Test {iteration+1}/{total} FAILED: {result.error}")
        return

    lines = []
    lines.append("╔" + "═" * 50 + "╗")
    title = f" LLM SpeedTest" + (f"  [{iteration+1}/{total}]" if total > 1 else "")
    lines.append(f"║{title:<50}║")
    lines.append("╠" + "═" * 50 + "╣")

    def row(label: str, value: str) -> None:
        lines.append(f"║ {label:<18} {value:<31} ║")

    row("Model:", result.model)
    row("Prompt tokens:", f"{result.prompt_tokens}{est_tag}")
    row("Gen tokens:", str(result.completion_tokens))
    lines.append("╠" + "─" * 50 + "╣")
    row("TTFT:", f"{result.ttft:.4f}s")
    row("PP speed:", f"{result.pp_speed:.1f} tok/s{est_tag}")
    row("TG speed:", f"{result.tg_speed:.1f} tok/s{est_tag}")
    row("MTP eff:", f"{result.mtp_efficiency:.2f} tok/chunk")
    lines.append("╠" + "─" * 50 + "╣")
    row("Stream time:", f"{result.stream_time:.2f}s")
    row("Chunks:", f"{result.total_chunks} ({result.content_chunks} content)")

    if result.reasoning_chunks:
        row("  reasoning:", str(result.reasoning_chunks))

    lines.append("╚" + "═" * 50 + "╝")

    if result.error:
        out.warning(f"Note: {result.error}")

    for line in lines:
        if out.use_color:
            print(f"{C.CYAN}{line}{C.RESET}")
        else:
            print(line)


def print_summary(results: List[TestResult], out: Output) -> None:
    """Print aggregate summary for multi-iteration runs."""
    valid = [r for r in results if not r.error or r.completion_tokens > 0]
    if len(valid) < 2:
        return

    def stat(key: str, fmt: str = "{:.2f}") -> str:
        vals = [getattr(r, key) for r in valid if getattr(r, key, 0) > 0]
        if not vals:
            return "N/A"
        return (
            f"avg={fmt.format(sum(vals)/len(vals))} "
            f"min={fmt.format(min(vals))} "
            f"max={fmt.format(max(vals))}"
        )

    print()
    out.bold("── Summary (averaged) " + "─" * 30)
    print(f"  TTFT:       {stat('ttft', '{:.4f}s')}")
    print(f"  PP speed:   {stat('pp_speed', '{:.1f} tok/s')}")
    print(f"  TG speed:   {stat('tg_speed', '{:.1f} tok/s')}")
    print(f"  MTP eff:    {stat('mtp_efficiency', '{:.2f} tok/chunk')}")
    print(f"  Stream:     {stat('stream_time', '{:.2f}s')}")
    print()


def output_json(results: List[TestResult], args: argparse.Namespace) -> None:
    """Output results as JSON to stdout."""
    output = {
        "url": args.url,
        "model": args.model,
        "mode": args.mode,
        "iterations": len(results),
        "results": [r.to_dict() for r in results],
    }
    valid = [r for r in results if not r.error or r.completion_tokens > 0]
    if len(valid) > 1:
        output["summary"] = {
            "avg_ttft": round(sum(r.ttft for r in valid) / len(valid), 4),
            "avg_pp_speed": round(sum(r.pp_speed for r in valid) / len(valid), 2),
            "avg_tg_speed": round(sum(r.tg_speed for r in valid) / len(valid), 2),
            "avg_mtp": round(
                sum(r.mtp_efficiency for r in valid) / len(valid), 4
            ),
        }
    print(json.dumps(output, indent=2))


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse and return CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Zero-dependency LLM endpoint speed tester",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  %(prog)s --url http://localhost:8080/v1/chat/completions
  %(prog)s --model mymodel --tokens 4000 --iterations 3
  %(prog)s --json 2>/dev/null | jq '.results[0].tg_speed'
  %(prog)s --nocache --mode long --tokens 8000
        """,
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Endpoint URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL!r})",
    )
    parser.add_argument(
        "--mode",
        choices=["short", "long"],
        default="short",
        help="Prompt mode: short (~50 tok) or long (padded) (default: short)",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=DEFAULT_TOKENS,
        help=f"Target prompt tokens for long mode (default: {DEFAULT_TOKENS})",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of runs to average (default: 1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key sent as Bearer token",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Connection/read timeout in seconds (default: {DEFAULT_TIMEOUT})",
    )
    parser.add_argument(
        "--nocache",
        action="store_true",
        help="Prepend random prefix to bypass KV cache",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show per-chunk detail",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Force-disable ANSI colors",
    )
    return parser.parse_args()


def main() -> None:
    """Entry point."""
    check_environment()
    args = parse_args()

    # Handle --no-color and auto-detection
    force_color = None
    if args.no_color:
        force_color = False
    out = Output(force_color=force_color)

    # Handle Ctrl+C gracefully
    def handle_sigint(signum, frame):
        out.warning("\nInterrupted. Reporting partial results...")
        sys.exit(130)

    signal.signal(signal.SIGINT, handle_sigint)

    # Generate prompt
    prompt = generate_prompt(args.mode, args.tokens, args.nocache)
    char_count = len(prompt)
    est_tok = char_count // 4

    if not args.json:
        out.info(
            f"Mode: {args.mode} | Prompt: {char_count} chars "
            f"(~{est_tok} tok) | Target: {args.url}"
        )
        if args.nocache:
            out.dim("  KV-cache bypass enabled")
        if args.iterations > 1:
            out.dim(f"  Running {args.iterations} iterations")

    # Run tests
    results: List[TestResult] = []
    for i in range(args.iterations):
        if not args.json and args.iterations > 1:
            out.progress(f"  [{i+1}/{args.iterations}] Running...")

        result = run_single_test(
            url=args.url,
            model=args.model,
            prompt=prompt,
            api_key=args.api_key,
            timeout=args.timeout,
            max_tokens=DEFAULT_MAX_GEN_TOKENS,
            out=out,
        )
        results.append(result)

        if args.verbose and not args.json:
            out.dim(f"  chunks={result.total_chunks} "
                    f"content={result.content_chunks} "
                    f"reasoning={result.reasoning_chunks}")

    # Clear progress line
    if not args.json and args.iterations > 1 and out.is_tty:
        sys.stderr.write("\r\033[K")
        sys.stderr.flush()

    # Output
    if args.json:
        output_json(results, args)
    else:
        for i, r in enumerate(results):
            print_result_box(r, out, i, args.iterations)
        if args.iterations > 1:
            print_summary(results, out)
    # Exit non-zero if all tests failed
    all_failed = all(r.error and r.completion_tokens == 0 for r in results)
    if all_failed:
        if not args.json:
            out.error("All test runs failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
