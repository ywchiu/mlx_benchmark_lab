# dflash-mlx — Detailed Report

[繁體中文版](03-dflash-mlx_zh.md)

## How we ran it

```bash
dflash serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765 \
  --no-prefix-cache \
  --no-prefix-cache-l2 \
  --chat-template-args '{"enable_thinking":false}'
```

Tested against dflash-mlx **v0.1.7** with `bench_inline.py`, n=5 per cell,
`max_tokens=256`, `cooldown=60` between runs (long enough for the M5 Max to
recover thermally between requests, so we measure the framework's peak
performance rather than back-to-back-load throttled performance). Both tiers
of prefix cache disabled to measure cold-per-request compute. Raw data:
[`data/dflash_v6_c60.jsonl`](../data/dflash_v6_c60.jsonl).

The most important thing to know about dflash-mlx is that the
`z-lab/Qwen3.6-35B-A3B-DFlash` model is a *draft* model, not a target. It has
only 8 transformer layers plus an output head, designed to be paired with the
full 35B model to drive speculative decoding. Trying to load it as the main
`--model` will crash on startup with an error about 91 missing parameters
(the layers it doesn't have). The correct invocation always pairs
`--model <full target>` with `--draft <DFlash variant>`.

---

## Full statistics (n=5 per cell)

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 148.7 | 147.8 | 1.8 | 145.0 | 149.5 |
| 512 | 138.9 | 138.4 | 2.1 | 134.7 | 140.1 |
| 2,048 | **154.5** | 154.4 | 0.4 | 153.6 | 154.7 |
| 4,096 | 127.7 | 127.7 | 0.2 | 127.5 | 128.0 |
| 8,192 | 124.6 | 124.6 | 0.2 | 124.4 | 125.0 |
| 16,384 | 119.0 | 118.5 | 0.9 | 117.3 | 119.2 |
| 32,768 | **121.2** | 121.2 | 0.2 | 121.0 | 121.4 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 182 | 178 | 8 | 165 | 186 |
| 512 | 801 | 797 | 27 | 762 | 833 |
| 2,048 | 2,229 | 2,219 | 34 | 2,176 | 2,256 |
| 4,096 | 2,971 | 3,005 | 96 | 2,946 | 3,175 |
| 8,192 | **3,425** | 3,427 | 5 | 3,423 | 3,435 |
| 16,384 | 3,633 | 3,631 | 10 | 3,612 | 3,639 |
| 32,768 | 3,271 | 3,277 | 17 | 3,258 | 3,299 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 434 | 444 | 21 | 424 | 478 |
| 512 | 548 | 551 | 19 | 527 | 576 |
| 2,048 | 753 | 757 | 12 | 744 | 771 |
| 4,096 | 1,123 | 1,111 | 34 | 1,051 | 1,132 |
| 8,192 | 1,938 | 1,937 | 3 | 1,933 | 1,939 |
| 16,384 | 3,647 | 3,650 | 10 | 3,642 | 3,668 |
| 32,768 | **8,094** | 8,079 | 42 | 8,027 | 8,127 |

---

## What the data says

dflash-mlx leads decode tps at every context length we tested. The peak is at
2K (154 tps) where natural-language summarization happens to match the draft
model's prediction patterns well. The line stays above 120 tps from 4K all
the way through 32K, and the standard deviation is consistently under 1 tps
at long context — speculative decoding is delivering both speed and stability
once you give the chip enough thermal headroom between requests.

Prefill peaks at 16K (3,633 tps) and stays above 3,200 tps at 32K. TTFT is
8.1s at 32K, the best of the frameworks tested.

The 512-token cell deserves a note: decode there (138.9 tps) is essentially
identical to the non-speculative baseline at the same context — meaning
speculative decoding provided no measurable speedup at that specific size
on this specific natural-language workload. The 64-, 2K-, and longer-context
cells all benefit substantially. Draft hit rate is content-dependent, so your
code-completion or structured-output workloads might do better than the
natural-language numbers shown here.

---

## When to use dflash-mlx

dflash-mlx is the right call when decode throughput is the priority and you
can manage TTFT separately (warm-up requests, prefix caching for known
prefixes, or simply tolerating ~500 ms more TTFT than omlx at short context).
It leads at every tested context length and is competitive on stability
(stddev < 1 tps at most cells) once thermal pressure is controlled.

If you have known prefix patterns — multi-turn conversations, agent loops
with repeated system prompts, RAG with stable headers — drop the
`--no-prefix-cache` and `--no-prefix-cache-l2` flags. v0.1.7's prefix cache
is enabled by default and is designed exactly for that case. We disabled it
for this benchmark to measure cold per-request compute, matching the
methodology used for the other frameworks.

If stability matters more than peak throughput, omlx remains a strong
alternative — it gives up roughly 10% of peak decode but has the lowest TTFT
variance in this benchmark.

---

## Things to watch out for

There's no `/health` endpoint — use `GET /v1/models` to check readiness
instead. The bench script's call to `/v1/cache/clear` returns 404 (silently
handled) since dflash uses its own prefix-cache controls via CLI flags
instead of an HTTP endpoint.

The companion CLI tools `dflash generate` (one-shot) and `dflash benchmark`
(built-in baseline-vs-DFlash comparison) are both useful for quick testing
without spinning up a server.

Cooldown matters on Apple Silicon. The numbers above are with `cooldown=60`
between runs; under shorter cooldowns the M5 Max throttles at long context
and the 32K decode rate drops measurably (back-to-back, no-cooldown
benchmarks will report lower numbers than this report shows). If you're
sizing for production, measure with the cooldown that matches your traffic
shape.

---

## Credit

Thanks to the community contributor who flagged that earlier numbers from
dflash-mlx v0.1.0 no longer reflected current behavior and prompted this
re-test against v0.1.7 with the contributor's exact configuration.
