#!/usr/bin/env python3
"""Streaming bench for OpenAI-compatible MLX servers.
- Forces /no_think prefix (Qwen3 convention)
- Per-scenario full-size warm-up (discarded)
- Reports median, mean, variance, stddev for prefill_tps, decode_tps, ttft_ms
"""
import json, math, os, sys, time, urllib.request, argparse, statistics

_WORDS = (
    "the quick brown fox jumps over the lazy dog Alice was beginning to get "
    "very tired of sitting by her sister on the bank and of having nothing to "
    "do once or twice she had peeped into the book her sister was reading but "
    "it had no pictures or conversations in it and what is the use of a book "
    "thought Alice without pictures or conversations "
)

def make_prompt(target_tokens: int) -> str:
    target_chars = target_tokens * 4
    s = (_WORDS * (target_chars // len(_WORDS) + 2))[:target_chars]
    return f"/no_think\nSummarize the following text in detail:\n\n{s}"

def clear_cache(base: str) -> None:
    """Best-effort prefix-cache clear. Rapid-mlx supports /v1/cache/clear; others 404 silently."""
    req = urllib.request.Request(f"{base}/v1/cache/clear", data=b"",
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception:
        pass

def stream(base, model, prompt, max_tokens):
    body = {"model": model, "messages": [{"role":"user","content":prompt}],
            "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0,
            "thinking": {"type": "disabled"}}
    req = urllib.request.Request(f"{base}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type":"application/json"}, method="POST")
    t0 = time.perf_counter()
    t1 = None
    pt = ct = think = vis = 0
    with urllib.request.urlopen(req, timeout=900) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"): continue
            ds = line[5:].strip()
            if ds == "[DONE]": break
            try: ch = json.loads(ds)
            except: continue
            ch_choices = ch.get("choices") or []
            if ch_choices:
                d = ch_choices[0].get("delta", {})
                rc = d.get("reasoning_content","") or ""
                cc = d.get("content","") or ""
                if rc:
                    think += 1
                    if t1 is None: t1 = time.perf_counter()
                if cc:
                    vis += 1
                    if t1 is None: t1 = time.perf_counter()
            u = ch.get("usage")
            if u:
                pt = u.get("prompt_tokens", 0)
                ct = u.get("completion_tokens", 0)
    t2 = time.perf_counter()
    if t1 is None: t1 = t2
    out = ct or (think + vis)
    return {
      "prompt_tokens": pt, "completion_tokens": out, "think": think, "vis": vis,
      "ttft_ms": (t1-t0)*1000, "decode_s": t2-t1, "total_s": t2-t0,
      "prefill_tps": (pt/(t1-t0)) if (t1-t0)>0 and pt else 0.0,
      "decode_tps":  (out/(t2-t1)) if (t2-t1)>0 and out else 0.0,
    }

def stat(vals):
    if not vals: return None
    n = len(vals)
    mean = sum(vals)/n
    var = sum((v-mean)**2 for v in vals)/(n-1) if n > 1 else 0.0
    return {
        "n": n,
        "median": statistics.median(vals),
        "mean": mean,
        "var": var,
        "std": math.sqrt(var),
        "min": min(vals),
        "max": max(vals),
    }

def fmt_stat(s, fmt="{:.1f}"):
    if s is None: return "N/A"
    return f"med={fmt.format(s['median'])} avg={fmt.format(s['mean'])} std={fmt.format(s['std'])} min={fmt.format(s['min'])} max={fmt.format(s['max'])}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sizes", default="64,512,2048,4096,8192,16384,32768")
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--cooldown", type=float, default=2.0)
    p.add_argument("--json-out", default=None, help="Path to write per-run results as JSONL")
    a = p.parse_args()
    base = a.url.rstrip("/")

    sizes = [int(s) for s in a.sizes.split(",")]
    print(f"server={base} model={a.model} sizes={sizes} runs={a.runs} max_tokens={a.max_tokens}")

    all_rows = []
    leaked_thinking = False
    failed_sizes = set()

    jf = open(a.json_out, "w") if a.json_out else None

    for sz in sizes:
        prompt = make_prompt(sz)
        clear_cache(base)
        sys.stdout.write(f"[size={sz}] warm-up ... ")
        sys.stdout.flush()
        try:
            w = stream(base, a.model, prompt, a.max_tokens)
            print(f"in={w['prompt_tokens']} out={w['completion_tokens']} ({w['total_s']:.1f}s)")
            if w["think"] > 0:
                leaked_thinking = True
        except Exception as e:
            print(f"FAILED: {e}")
            failed_sizes.add(sz)
            continue

        for i in range(a.runs):
            time.sleep(a.cooldown)
            clear_cache(base)
            sys.stdout.write(f"  run {i+1}/{a.runs} ... ")
            sys.stdout.flush()
            try:
                r = stream(base, a.model, prompt, a.max_tokens)
                row = {"size":sz, **r}
                all_rows.append(row)
                if jf:
                    jf.write(json.dumps(row) + "\n"); jf.flush()
                print(f"TTFT {r['ttft_ms']:.0f}ms  prefill {r['prefill_tps']:.0f}  decode {r['decode_tps']:.1f}  in={r['prompt_tokens']} out={r['completion_tokens']}  total {r['total_s']:.1f}s")
            except Exception as e:
                print(f"ERR {e}")

    if jf: jf.close()

    print("\n=== SUMMARY ===")
    print(f"{'size':>6} | {'decode tps (med/avg/std/min/max)':<46} | {'prefill tps (med/avg/std/min/max)':<46} | {'TTFT ms (med/avg/std/min/max)':<46}")
    print("-" * 160)
    for sz in sizes:
        if sz in failed_sizes:
            print(f"{sz:>6} | FAILED")
            continue
        rs = [r for r in all_rows if r["size"]==sz]
        if not rs: continue
        d = stat([r["decode_tps"] for r in rs])
        p = stat([r["prefill_tps"] for r in rs])
        t = stat([r["ttft_ms"] for r in rs])
        print(f"{sz:>6} | {fmt_stat(d):<46} | {fmt_stat(p, '{:.0f}'):<46} | {fmt_stat(t, '{:.0f}'):<46}")
    if leaked_thinking:
        print("\nNote: reasoning_content was emitted despite /no_think — decode tps mixes thinking + visible tokens.")

if __name__ == "__main__":
    main()
