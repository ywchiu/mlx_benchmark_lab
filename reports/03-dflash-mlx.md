# dflash-mlx — Detailed Report

[繁體中文版](03-dflash-mlx_zh.md)

## How we ran it

```bash
dflash-serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765
```

The most important thing to know about dflash-mlx is that the `z-lab/Qwen3.6-35B-A3B-DFlash` model is a *draft* model, not a target. It has only 8 transformer layers plus an output head, designed to be paired with the full 35B model to drive speculative decoding. Trying to load it as the main `--model` will crash on startup with an error about 91 missing parameters (the layers it doesn't have). The correct invocation always pairs `--model <full target>` with `--draft <DFlash variant>`.

---

## Full statistics (n=5 per cell)

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | **167.3** | 167.3 | 2.9 | 163.1 | 170.9 |
| 512 | 122.9 | 122.1 | 2.2 | 118.3 | 123.5 |
| 2,048 | **160.1** | 158.7 | 2.8 | 153.8 | 160.5 |
| 4,096 | 104.5 | 104.4 | 0.8 | 103.5 | 105.6 |
| 8,192 | 96.3 | 95.9 | 1.6 | 93.6 | 97.9 |
| 16,384 | 84.1 | 84.4 | 2.4 | 81.7 | 88.0 |
| 32,768 | **12.6** ⚠️ | 12.8 | 1.3 | 11.3 | 14.5 |

### Prefill tps

dflash-serve does not return `prompt_tokens` in the streaming `usage` object — every response shows zero. So we can't compute prefill tps directly from the API response. If you need an estimate you can back it out from TTFT and externally-tokenized prompt counts (in this benchmark the actual input was 77/437/1677/3333/6636/13250/26475 tokens for the seven sizes, after Qwen tokenization).

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 334 | 340 | 17 | 328 | 371 |
| 512 | 360 | 364 | 8 | 357 | 378 |
| 2,048 | 587 | 572 | 36 | 507 | 591 |
| 4,096 | 1,072 | 1,063 | 27 | 1,019 | 1,090 |
| 8,192 | 2,205 | 2,217 | 33 | 2,178 | 2,261 |
| 16,384 | 6,023 | 5,957 | 185 | 5,651 | 6,119 |
| 32,768 | **31,205** ⚠️ | 31,796 | 2,663 | 28,469 | 35,830 |

---

## What the data says

dflash-mlx is a story of two extremes. At short context it's the clear winner: 167 tokens per second at 64 tokens is 35% faster than the second-place finisher, and the 160 tps result at 2,048 tokens is similarly dominant. The standard deviation across runs is low (under 3 tps), so the speedup is consistent — when speculative decoding works, it works reliably.

But at 32K context, dflash-mlx breaks. Decode tps drops to 12.6, which is roughly six times slower than the next slowest framework. TTFT climbs to 31 seconds, more than twice everyone else. And the run-to-run TTFT variability balloons to 2.7 seconds — by far the highest variance we measured.

The root cause is structural. Speculative decoding works by having a small draft model generate token candidates that the main model then verifies in parallel. At short context this is a clear win because the draft pass is cheap relative to the main pass. But the draft model has to ingest the full prompt too — when the prompt is 32K tokens, the draft is doing nearly as much work as the main model, eliminating the cost advantage of having a draft in the first place. Worse, draft prediction accuracy tends to fall as context grows: the more context the model is conditioning on, the harder it is for a small draft to predict the next token correctly. Failed predictions trigger expensive verify-then-rollback cycles. By 32K, the verify cost dominates, and you end up paying for both a draft pass and a main pass per accepted token, plus the rollback overhead — which is why decode rate falls below baseline rather than just falling back to baseline.

The 512-token result deserves a note. Decode tps was 122.9 there, essentially identical to the non-speculative baseline — meaning speculative decoding provided no speedup at that context length. The 2K and 64-token cases benefited substantially, the 512 case did not. We believe this is because the draft hit rate happens to be content-dependent, and the natural-language summarization prompt at 512 tokens didn't match the draft model's prediction patterns well. Your code-completion or structured-output workloads might do better.

---

## When to use dflash-mlx

It's the right call when your context length is bounded under 4K and your output is highly predictable — code, JSON, structured templates, repetitive content. In those conditions you're getting a real 35% decode speedup over the alternatives, and the slightly elevated TTFT (an extra 200–300 ms compared to omlx) is rarely the bottleneck for a generation-bound workload.

Never use it past 16K context, and especially not at 32K. The 12 tps decode rate is essentially unusable for anything interactive. Don't use it for TTFT-sensitive workloads in general — even at 64 tokens, dflash's 334 ms TTFT is more than double omlx's 148 ms.

---

## Things to watch out for

There's no `/health` endpoint — use `GET /v1/models` to check readiness instead. The `/v1/cache/clear` endpoint that rapid-mlx exposes also doesn't exist on dflash-serve, but the bench script handles the 404 silently.

The companion CLI tools `dflash` (one-shot generate) and `dflash-benchmark` (built-in baseline-vs-DFlash comparison) are both useful if you want to test the speculative speedup quickly without spinning up a server. `dflash-benchmark` is particularly handy because it directly compares baseline-MLX-decoding against DFlash-decoding on the same prompt, so you can see the speedup without doing the bookkeeping yourself.

The 32K context window appears to be a hard ceiling. We didn't try 64K but expect the failure mode to be even worse — the draft model's context window probably matters as much as the main model's, and this draft was trained on a specific context budget.
