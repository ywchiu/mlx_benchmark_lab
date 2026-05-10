# rapid-mlx — Detailed Report

[繁體中文版](01-rapid-mlx_zh.md)

## How we ran it

```bash
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --disable-prefix-cache
```

The `--disable-prefix-cache` flag is essential, not optional. rapid-mlx caches prompt prefixes by default to speed up repeated requests, and during a benchmark where every run uses the same prompt, that cache will completely skip the prefill computation for runs 2 through 5 — leaving you with prefill-tps numbers in the hundreds of thousands, which is physically impossible for a 35B model. We learned this the hard way in the first round of testing. With the cache disabled, every measurement reflects honest cold-prefill performance.

---

## Full statistics (n=5 per cell)

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 124.9 | 124.7 | 0.6 | 123.8 | 125.3 |
| 512 | 119.5 | 120.6 | 2.6 | 118.0 | 124.4 |
| 2,048 | 102.5 | 105.3 | 5.2 | 100.7 | 112.9 |
| 4,096 | 97.6 | 97.7 | 0.5 | 97.2 | 98.4 |
| 8,192 | 90.3 | 90.1 | 1.6 | 87.8 | 91.7 |
| 16,384 | 83.2 | 82.9 | 0.7 | 82.0 | 83.7 |
| 32,768 | 72.3 | 72.3 | 0.2 | 72.1 | 72.7 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 456 | 359 | 150 | 167 | 473 |
| 512 | 1,026 | 1,227 | 373 | 899 | 1,634 |
| 2,048 | 3,221 | 2,819 | 596 | 2,130 | 3,279 |
| 4,096 | 3,070 | 2,943 | 250 | 2,644 | 3,207 |
| 8,192 | 2,696 | 2,705 | 94 | 2,572 | 2,814 |
| 16,384 | 2,323 | 2,285 | 93 | 2,120 | 2,343 |
| 32,768 | 1,987 | 1,982 | 21 | 1,957 | 2,005 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 169 | 259 | 136 | 163 | 462 |
| 512 | 426 | 382 | 107 | 268 | 486 |
| 2,048 | 521 | 619 | 142 | 511 | 787 |
| 4,096 | 1,086 | 1,139 | 99 | 1,039 | 1,260 |
| 8,192 | 2,461 | 2,455 | 86 | 2,359 | 2,580 |
| 16,384 | 5,703 | 5,807 | 250 | 5,656 | 6,251 |
| 32,768 | 13,326 | 13,361 | 140 | 13,207 | 13,531 |

---

## What the data says

rapid-mlx's strength is short-context decoding. At 64 to 512 tokens it sits comfortably in the leading group at 119–125 tokens per second, and importantly, those numbers barely move from run to run — the standard deviation at 32K is just 0.2 tps, the lowest of any framework we tested at long context. This is the kind of stability you want when you're trying to make a latency promise to a client.

The prefill numbers in the 64–512 range are oddly low (456 and 1,026 tps respectively) compared to omlx and mlx-vlm, both of which clear 1,800 tps at 512 tokens. We suspect this is the chunked-prefill path being suboptimal for tiny prompts — rapid-mlx is optimized for production-style traffic where prompts arrive in bursts and can be batched, not for one-tiny-prompt-at-a-time micro-benchmarks. Once the prompt grows past 2K tokens, prefill jumps up to 3,000+ tps and stays competitive.

The weakness is TTFT jitter at small context. At 64 tokens we measured a median of 169 ms but a maximum of 462 ms across the five runs — meaning some requests are nearly three times slower than the typical case for no obvious reason. The standard deviation of 136 ms is close to the median itself, which is a red flag if you care about consistent response latency. By 4K context this jitter has dropped to about 10% of the median, so it only affects short-prompt workloads.

Decode degradation across context lengths is steady and predictable. From 64 to 32K the median falls from 124.9 to 72.3 tps, a drop of 42%. This is worse than omlx (-34%) but better than mlx-vlm. In absolute terms rapid-mlx loses its long-context lead to omlx by 16K tokens — at that point omlx is already 22% faster.

---

## When to use rapid-mlx

It's the right pick when you need short-to-medium-context decode and can tolerate some TTFT jitter. The most flexible feature set among the four frameworks (paged KV cache, multi-token prediction, prefix cache, KV-cache quantization) makes it a good experimentation platform for new optimization techniques. If you're building something like a coding agent that processes 1K–4K context at a time and you want to be able to tune cache memory, prefix-cache size, and other parameters, rapid-mlx gives you the most knobs.

It's the wrong pick for ultra-long-context workloads (omlx wins past 16K) and for SLA-driven applications where TTFT consistency matters more than peak speed.

---

## Things to watch out for

The `/no_think` Qwen3 prompt convention does not work reliably with rapid-mlx. The model still emits reasoning tokens (`reasoning_content` chunks in the stream), and worse, `max_tokens=256` does not cap the thinking phase — we observed outputs as long as 2,155 tokens (1,879 thinking + 276 visible). If you need strict output length control, you'll need to filter `reasoning_content` on the client side or pass `enable_thinking=false` through the chat-template parameters.

The health endpoint at `GET /health` returns `{"ready":true}` once the model is loaded, which is the right thing to poll for before kicking off a benchmark. Don't use `/v1/models` — that's available immediately on startup, before the model is actually loaded.
