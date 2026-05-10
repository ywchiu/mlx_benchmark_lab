# omlx — Detailed Report

[繁體中文版](02-omlx_zh.md)

## How we ran it

```bash
# omlx looks for models in ~/.omlx/models/<model_id>/, not your HuggingFace cache.
# A symlink works fine:
ln -snf \
  ~/.cache/huggingface/hub/models--mlx-community--Qwen3.6-35B-A3B-4bit/snapshots/<HASH>/ \
  ~/.omlx/models/Qwen3.6-35B-A3B-4bit

omlx serve --port 8765 --log-level warning --no-cache
```

A few things about the API are worth flagging up front. omlx uses the directory basename as the model ID, not the HuggingFace repo path — so when your client sends a request, the `model` field should be `Qwen3.6-35B-A3B-4bit`, not `mlx-community/Qwen3.6-35B-A3B-4bit`. The `--no-cache` flag disables both the in-memory prefix cache and the SSD cache; without it, repeated benchmarks will hit cached state and your prefill numbers won't reflect cold performance.

---

## Full statistics (n=5 per cell)

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 123.7 | 123.8 | 0.4 | 123.5 | 124.5 |
| 512 | 119.4 | 120.3 | 1.9 | 118.4 | 122.9 |
| 2,048 | 121.1 | 121.0 | 0.4 | 120.4 | 121.2 |
| 4,096 | 120.4 | 119.5 | 2.3 | 115.5 | 120.9 |
| 8,192 | 118.0 | 117.9 | 0.3 | 117.4 | 118.3 |
| 16,384 | 105.3 | 105.0 | 2.7 | 101.3 | 107.9 |
| 32,768 | 82.1 | 82.6 | 1.5 | 81.0 | 84.7 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 520 | 536 | 39 | 510 | 604 |
| 512 | 1,735 | 1,741 | 23 | 1,720 | 1,774 |
| 2,048 | 3,569 | 3,563 | 33 | 3,530 | 3,608 |
| 4,096 | 3,989 | 3,966 | 116 | 3,843 | 4,097 |
| 8,192 | 3,467 | 3,406 | 259 | 3,122 | 3,707 |
| 16,384 | 2,826 | 2,842 | 84 | 2,742 | 2,955 |
| 32,768 | 2,083 | 2,084 | 68 | 2,004 | 2,186 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 148 | 144 | 10 | 128 | 151 |
| 512 | 252 | 251 | 3 | 246 | 254 |
| 2,048 | 470 | 471 | 4 | 465 | 475 |
| 4,096 | 836 | 841 | 25 | 813 | 867 |
| 8,192 | 1,914 | 1,957 | 150 | 1,790 | 2,125 |
| 16,384 | 4,689 | 4,666 | 138 | 4,483 | 4,833 |
| 32,768 | 12,709 | 12,713 | 412 | 12,113 | 13,209 |

---

## What the data says

omlx is the long-context champion in this benchmark, and the numbers make the case unambiguously. From 4K tokens onward it has the highest decode tps in every cell, and at 32K it's still hitting 82 tokens per second — 14% faster than rapid-mlx and roughly six times faster than dflash-mlx. The decode-degradation curve is the gentlest of any framework: from 64 to 32K it loses 34% of its baseline speed, where rapid-mlx loses 42% and dflash-mlx loses 92%.

What's almost more impressive is the consistency. Look at the TTFT standard deviations: 3 ms at 512 tokens, 4 ms at 2K, 25 ms at 4K. The other frameworks are at 100+ ms variability in this range. If you're building a service with strict latency requirements — say, a p99 promise to a downstream consumer — omlx makes those promises a lot easier to keep. Decode tps standard deviation at most context lengths is under 1 tps, which is exceptional.

omlx also has the best prefill numbers we measured, peaking at 3,989 tokens per second at 4K context. This is meaningfully ahead of mlx-vlm (3,672) and rapid-mlx (3,070) at the same prompt size. For RAG applications where you're stuffing 4K–8K of retrieved context into every request, this directly translates to faster end-to-end response times.

The few weak spots are minor. Decode tps at 64 tokens is 123.7, slightly behind rapid-mlx's 124.9 — but the difference is well within noise. TTFT variance at 16K and 32K does grow into the 100–400 ms range, which is more variability than at shorter contexts but still better than the alternatives.

---

## When to use omlx

It's the default recommendation for any production deployment. Long-context RAG, document summarization, code analysis on big files, multi-tenant inference servers — omlx will give you the most predictable performance with the least tuning.

The multi-model management is a real bonus that other frameworks don't have. omlx supports loading multiple models simultaneously and uses LRU eviction to manage memory; you can swap between Qwen, Llama, and Mistral on the same server without restart. For development environments where you're testing many models or running both a chat and a coding model side by side, this is a feature with real teeth.

It's the wrong pick if you have only short prompts and need every last token per second of decode speed — dflash-mlx will be faster there. It's also worse than rapid-mlx if you want fine-grained control over cache memory and KV quantization, since omlx exposes fewer of those knobs.

---

## Things to watch out for

The setup friction is real. omlx requires models to be in `~/.omlx/models/<id>/` and won't read your HuggingFace cache directly. The fix is a one-line symlink, but it's not obvious until you've hit the "Available models: (none)" error message. The default `~/.omlx/settings.json` also caps `max_context_window` at 32,768 — if you want to test 64K or 128K context, raise this first.

The admin API at `/admin/api/...` requires cookie-based authentication that's only available after logging in through the admin web UI at `/admin`. There's no easy way to script "load this model" without spinning up a separate server with the model specified at launch, which is what we did for our benchmark. If you're doing automated testing, this is a friction point.

The default `--cache` setting (i.e., not passing `--no-cache`) enables both the SSD cache (writes to `~/.omlx/cache/`) and the in-memory prefix cache. For benchmarking, always disable both. For production use the cache is helpful and you should leave it on.
