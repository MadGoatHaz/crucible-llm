import time
import requests
import json
import sys
import argparse
import random
import string

# ANSI Color Codes
C_BLUE = "\033[94m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_CYAN = "\033[96m"
C_MAGENTA = "\033[95m"
C_BOLD = "\033[1m"
C_RESET = "\033[0m"

def generate_long_prompt(target_tokens=2000, bypass_cache=False):
    """
    Generates a long prompt to test Prompt Processing (PP) speed.
    Uses a repetitive common word to ensure a predictable 1-to-1 token ratio.
    """
    base = "Repeat the following sequence back to me briefly, then explain the importance of context window size: "

    if bypass_cache:
        # Inject a random 8-character string so the LLM cannot use its KV cache
        rand_str = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        base = f"Ignore this sequence [{rand_str}]. " + base

    # The word " apple" (with a leading space) is almost universally treated as exactly 1 token.
    padding = " apple" * (target_tokens - 20) # -20 to account for the base string
    return base + padding

def run_test():
    parser = argparse.ArgumentParser(description="LLM Speedtest Utility for vLLM and llama.cpp")
    parser.add_argument("--url", type=str, default="http://192.168.51.148:8080/v1/chat/completions", help="API URL")
    parser.add_argument("--model", type=str, default="Qwen3.6-35B-A3B-UD-Q4_K_XL", help="Model name")
    parser.add_argument("--mode", choices=["base", "long"], default="base", help="Test mode: base (short) or long (2k+ tokens)")
    parser.add_argument("--nocache", action="store_true", help="Inject random text to bypass KV cache")
    args = parser.parse_args()

    prompt = "Write a complex Rust macro to measure function execution time."
    if args.mode == "long":
        print(f"{C_MAGENTA}Generating ~2000 token prompt...{C_RESET}")
        prompt = generate_long_prompt(2000, bypass_cache=args.nocache)

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "stream_options": {"include_usage": True}
    }

    print(f"{C_BLUE}{C_BOLD}Mode:{C_RESET} {args.mode.upper()}{' (NO CACHE)' if args.nocache else ''}")
    # We calculate the character count for transparency
    print(f"{C_BLUE}{C_BOLD}Input Size:{C_RESET} {len(prompt)} characters\n")

    start_time = time.time()

    try:
        response = requests.post(args.url, json=payload, stream=True, timeout=(60, 60))
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"{C_YELLOW}Connection error: {e}{C_RESET}")
        exit(1)

    first_token = True
    ttft = 0
    reasoning_chunks = 0
    content_chunks = 0
    other_chunks = 0

    exact_completion_tokens = 0
    exact_prompt_tokens = 0
    premature_exit = False

    spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
    spin_idx = 0
    in_thinking_phase = True

    print(f"{C_MAGENTA}--- WAITING FOR SERVER (PP PHASE) ---{C_RESET}")

    try:
        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode('utf-8')
            if decoded_line.startswith("data: "):
                data_str = decoded_line[6:].strip()

                if data_str == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)

                    # Capture usage info as soon as it appears (vLLM/llama.cpp check)
                    if "usage" in data and data["usage"] is not None:
                        exact_completion_tokens = data["usage"].get("completion_tokens", 0)
                        exact_prompt_tokens = data["usage"].get("prompt_tokens", 0)

                    if not data.get("choices"):
                        continue

                    delta = data["choices"][0].get("delta", {})

                    if first_token:
                        ttft = time.time() - start_time
                        sys.stdout.write(f"\r\033[K{C_GREEN}[TTFT / PP Time: {ttft:.3f} seconds]{C_RESET}\n")
                        first_token = False

                    reasoning = delta.get("reasoning", "") or delta.get("reasoning_content", "")
                    content = delta.get("content", "")

                    spin_char = spinner[spin_idx % len(spinner)]
                    spin_idx += 1

                    if reasoning:
                        reasoning_chunks += 1
                        sys.stdout.write(f"\r\033[K{C_YELLOW}{spin_char} Thinking... [{reasoning_chunks} pkts]{C_RESET}")
                        sys.stdout.flush()
                    elif content:
                        if in_thinking_phase and reasoning_chunks > 0:
                            sys.stdout.write(f"\r\033[K{C_YELLOW}✓ Thinking Complete{C_RESET}\n")
                            in_thinking_phase = False

                        content_chunks += 1
                        sys.stdout.write(f"\r\033[K{C_CYAN}{spin_char} Generating... [{content_chunks} pkts]{C_RESET}")
                        sys.stdout.flush()
                    else:
                        other_chunks += 1

                except (json.JSONDecodeError, KeyError):
                    pass
    except requests.exceptions.ChunkedEncodingError:
        sys.stdout.write(f"\n{C_YELLOW}Error: Response ended prematurely (ChunkedEncodingError).{C_RESET}\n")
        premature_exit = True
    except Exception as e:
        sys.stdout.write(f"\n{C_YELLOW}Error during streaming: {e}{C_RESET}\n")
        premature_exit = True

    end_time = time.time()
    generation_time = end_time - start_time - ttft
    total_chunks = reasoning_chunks + content_chunks + other_chunks

    # Metrics calculation
    tps = exact_completion_tokens / generation_time if generation_time > 0 else 0
    pp_tps = exact_prompt_tokens / ttft if ttft > 0 else 0
    mtp_rate = exact_completion_tokens / total_chunks if total_chunks > 0 else 0

    if not premature_exit:
        sys.stdout.write(f"\r\033[K{C_CYAN}✓ Generation Complete{C_RESET}\n")

    print(f"\n{C_BOLD}=== TELEMETRY ({args.model}) ==={C_RESET}")
    if premature_exit:
        print(f"{C_YELLOW}NOTE: Results are partial due to connection error.{C_RESET}")
    print(f"Prompt Tokens:     {C_MAGENTA}{exact_prompt_tokens}{C_RESET}")
    print(f"Generated Tokens:  {C_MAGENTA}{exact_completion_tokens}{C_RESET}")
    print(f"---")

    cache_hint = ""
    if args.mode == "long" and ttft < 0.2:
         cache_hint = f" {C_YELLOW}(Likely KV Cache Hit - Test invalidated){C_RESET}"

    print(f"PP Speed:          {C_GREEN}{pp_tps:.2f} tokens/sec{C_RESET}{cache_hint}")
    print(f"TG Speed:          {C_GREEN}{tps:.2f} tokens/sec{C_RESET} (Token Generation)")
    print(f"MTP Efficiency:    {C_CYAN}{mtp_rate:.2f} tokens/packet{C_RESET}")
    print(f"---")
    print(f"TTFT (Latent PP):  {ttft:.3f} s")
    print(f"Total Stream Time: {generation_time:.2f} s")
    print(f"Total Packets:     {total_chunks}\n")

if __name__ == "__main__":
    run_test()
