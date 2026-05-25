# dflash-mlx — Detailed Report

[繁體中文版](03-dflash-mlx_zh.md)

> **Update — 2026-05-25 (dflash-mlx v0.1.7 re-run)**
>
> A community contributor pointed out that the long-context numbers in the original
> v0.1.0 sweep (below) no longer reflect current dflash-mlx behavior. The 32K
> collapse was a v0.1.0 issue; v0.1.7 ships an adaptive verify mode, prefix-cache
> controls, and a tunable verify-len cap that together eliminate the cliff. Re-ran
> the same 7-size sweep on the same M5 Max with v0.1.7 — results in
> [§ v0.1.7 results](#v017-results-2026-05-25) below. The original v0.1.0 tables
> are kept intact as a historical record so the version delta is visible.

## v0.1.7 results (2026-05-25)

### How we ran it (v0.1.7)

The CLI changed in v0.1.7 — `dflash-serve` is now a subcommand of `dflash`, and a
large set of runtime flags landed (prefix cache, verify mode, snapshot caps,
diagnostics, etc.). We used the configuration the contributor proposed, which
disables both tiers of the prefix cache so the benchmark measures cold compute
per request (matching how we run the other frameworks):

```bash
dflash serve \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --draft z-lab/Qwen3.6-35B-A3B-DFlash \
  --port 8765 \
  --no-prefix-cache \
  --no-prefix-cache-l2 \
  --chat-template-args '{"enable_thinking":false}'
```

Bench script and methodology are unchanged: `scripts/bench_inline.py`, n=5 per
size, max_tokens=256, 2s cooldown, full-size warm-up discarded. Raw data is in
[`data/dflash_v6.jsonl`](../data/dflash_v6.jsonl).

### Decode tps (v0.1.7, n=5)

| size | median | mean | stddev | min | max | vs v0.1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 149.5 | 149.7 | 0.8 | 148.7 | 150.5 | −11% |
| 512 | 140.0 | 139.9 | 0.6 | 139.1 | 140.5 | +14% |
| 2,048 | 153.6 | 153.9 | 0.5 | 153.4 | 154.5 | −4% |
| 4,096 | 126.8 | 126.3 | 1.4 | 123.7 | 127.1 | +21% |
| 8,192 | 122.1 | 122.1 | 0.4 | 121.5 | 122.6 | +27% |
| 16,384 | 103.8 | 105.2 | 6.4 | 100.0 | 116.3 | +23% |
| 32,768 | **89.7** | 90.5 | 6.5 | 82.7 | 98.8 | **+612%** |

### Prefill tps (v0.1.7, n=5)

v0.1.7 also fixes the streaming `usage` object — `prompt_tokens` is now reported
correctly, so we can compute prefill tps directly from the API:

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 233 | 282 | 100 | 171 | 390 |
| 512 | 819 | 861 | 152 | 721 | 1,040 |
| 2,048 | 2,115 | 2,521 | 617 | 2,023 | 3,202 |
| 4,096 | 2,734 | 2,874 | 251 | 2,652 | 3,150 |
| 8,192 | 3,115 | 3,128 | 165 | 2,915 | 3,377 |
| 16,384 | 3,119 | 3,126 | 95 | 2,995 | 3,217 |
| 32,768 | 2,221 | 2,243 | 172 | 2,036 | 2,491 |

### TTFT (v0.1.7, ms, n=5)

| size | median | mean | stddev | min | max | vs v0.1.0 |
|---:|---:|---:|---:|---:|---:|---:|
| 64 | 339 | 310 | 109 | 202 | 463 | ~same |
| 512 | 536 | 522 | 89 | 422 | 609 | +49% slower |
| 2,048 | 794 | 697 | 157 | 524 | 830 | +35% slower |
| 4,096 | 1,220 | 1,167 | 99 | 1,059 | 1,258 | +14% slower |
| 8,192 | 2,131 | 2,127 | 111 | 1,966 | 2,277 | −3% |
| 16,384 | 4,249 | 4,242 | 129 | 4,119 | 4,425 | **−29%** |
| 32,768 | **11,919** | 11,857 | 898 | 10,629 | 13,005 | **−62%** |

### What changed

The headline is the 32K cell: decode went from 12.6 → 89.7 tps (7×) and TTFT from
31.2s → 11.9s (−62%). The structural collapse described in the v0.1.0 analysis
below is gone — long-context speculative decoding is now usable.

Looking across the table:

- **4K–32K decode all improved**, between +21% and +612%. The biggest wins are at
  the long end, which is where the v0.1.0 cliff lived. Adaptive verify mode
  (which shortens low-acceptance verify blocks instead of paying full verify cost
  on a missed draft) is doing the work here.
- **64-token and 2K decode slightly regressed** (−11% and −4%). Some of this is
  almost certainly the cost of disabling both prefix-cache tiers; some may be
  adaptive-mode overhead at sizes where the v0.1.0 path was already optimal. The
  loss is small in absolute terms and the v0.1.7 numbers are still competitive
  with the other frameworks at short context.
- **Short-context TTFT got worse**: 512–2K TTFT is 35–49% higher than v0.1.0,
  likely from added per-request setup in the new verify path. Long-context TTFT
  improved sharply, so the trade-off cost goes the right direction for the
  workloads it matters on.
- **Prefill tps is now available** — v0.1.7 reports `prompt_tokens` correctly, so
  we have a real prefill number instead of the back-of-envelope estimate the
  v0.1.0 report had to use.

### Note on contributor's reported numbers

The contributor reported 113.1 tps median at 32K; we measured 89.7. The
per-run sequence on our hardware is 98.8 → 94.9 → 89.7 → 86.1 → 82.7 — a clear
monotonic degradation across the 5 runs, suggesting either thermal throttling or
cache pressure accumulating across runs of the same workload. The contributor's
113 is within our run-1 ballpark (~99) plus run-to-run variance, so the two
results aren't really in conflict; they're sampling different points on the same
distribution. The conclusion (the cliff is gone) holds either way.

### Updated recommendation

Replace the "never use past 16K" rule below with: **dflash-mlx v0.1.7 is now
viable at every context length we tested.** The short-context speedup is smaller
than v0.1.0 reported, and the long-context one is dramatically better. If you
were avoiding this framework specifically because of the 32K cliff, you no
longer need to.

If you have known prefix patterns (multi-turn conversations, agent loops with
repeated system prompts), drop the `--no-prefix-cache` and `--no-prefix-cache-l2`
flags — v0.1.7's prefix cache is designed exactly for that case and is enabled by
default. We disabled it here only so the benchmark measures cold per-request
compute, matching the methodology used for the other frameworks.

### Credit

Thanks to the contributor who flagged this and provided the exact reproduction
command. The original v0.1.0 results below stand as a record of where the
framework was at the time, not a current recommendation.

---

## v0.1.0 results (original sweep)

The remainder of this document is the original report against dflash-mlx v0.1.0.
Read it as historical context for the v0.1.7 deltas above, not as a current
recommendation.

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
