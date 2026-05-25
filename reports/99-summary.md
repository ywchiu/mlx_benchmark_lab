# Cross-Framework Summary and Lessons Learned

[繁體中文版](99-summary_zh.md)

> **Note on dflash-mlx in this summary.** This document was originally written
> against dflash-mlx v0.1.0, which had a since-fixed 32K decode collapse. All
> dflash-mlx numbers below are now from the v0.1.7 re-test on 2026-05-25
> ([`reports/03-dflash-mlx.md`](03-dflash-mlx.md) has the full version-delta
> story). The other four frameworks were not re-tested; their numbers are still
> from the 2026-05-09 sweep.

## Who wins what

The headline metrics fan out across the four frameworks pretty cleanly.
dflash-mlx (v0.1.7) wins decode at every context length we tested — by 20–30%
in the short-context band (140–154 tps from 64 to 2K) and by 10–25% in the
long-context band (104–127 tps from 4K to 16K, 90 tps at 32K). omlx wins
stability — its decode-tps standard deviation is the lowest of the four at
nearly every cell, and its TTFT standard deviation sits in the 3–25 ms range
at moderate context, ten times tighter than the others. omlx also wins prefill
in the long-context band (peaking at 3,989 tps at 4K) and is essentially tied
with dflash-mlx at 32K decode once you account for run-to-run variance.
mlx-vlm wins decode-stability at short context (stddev of 0.2 tps at 64 and
512 tokens) and ties for best prefill at 8K. rapid-mlx never wins outright on
any single metric but is competitive everywhere and has the most flexible
feature set.

For multimodal input — image, audio, video — mlx-vlm is the only choice in this set, so the comparison there is moot.

## The fast-versus-stable tradeoff

The most interesting pattern in the data is how dramatically the framework rankings depend on what you're optimizing for. dflash-mlx leads decode tps everywhere but its speculative-decoding architecture introduces real per-request variance — at long context the standard deviation across 5 runs is 6+ tps because the draft hit rate is content- and thermally-dependent. omlx's peak decode is lower (124 tps short-context, 82 at 32K) but its decode is perfectly steady, run to run, which makes SLAs much easier to write against. Choosing between them is a question about your traffic patterns, not a question about which framework is "better."

Two principles fall out of this. First, peak speed and stability tend to trade off — the speedup from speculative decoding requires hitting a draft prediction, and when that hit rate varies by content (or by thermal state of the chip), speed varies too. Second, there is no universal winner. Every framework here has a regime where it wins and a regime where it loses, and picking the right one starts with identifying which regime your workload is in.

## Why rapid-mlx looked stronger in the first round

In our initial benchmarking pass, rapid-mlx's prefill numbers were impressive (over 5,000 tps at 4K and over 100,000 tps at 8K). It turned out those numbers were artifacts of prefix-cache hits — the warm-up run cached the prompt prefix, and runs 2 through 5 reused the cached KV state instead of running prefill at all. The fix was launching with `--disable-prefix-cache`, after which rapid-mlx's prefill dropped to a more realistic 3,070 tps at 4K, behind omlx and mlx-vlm.

This is worth noting because in production, your traffic *does* benefit from prefix caching when prompts share a system prefix or RAG header. So if your real workload has high prefix repetition (a fixed system message, for instance), rapid-mlx's effective prefill in production will be much higher than what this cold-cache benchmark shows. The cold numbers are the lower bound; the cached numbers are the upper bound.

## The original 32K cliff — and how v0.1.7 fixed it

The original 2026-05-09 sweep against dflash-mlx v0.1.0 showed a structural
failure at 32K: decode dropped from 84 tps at 16K to 12.6 tps at 32K, a 6.7×
regression in a single doubling of context. The mechanism was understood at
the time: speculative decoding requires that the draft model's prediction
accuracy be high enough that the verify-then-accept cycle costs less per
accepted token than just running the main model directly. At long context two
things broke that bargain. First, the draft model had to ingest the full 32K
prompt the same way the main model did, eliminating much of the cost
asymmetry. Second, the draft's prediction accuracy degrades as context grows,
because long context conditioning is fundamentally harder for a small model to
handle. Failed predictions cost both a draft pass and a wasted main-model
forward, plus the rollback work to recover. By 32K, the verify cost exceeded
the cost of just decoding directly.

dflash-mlx v0.1.7 ships an **adaptive verify mode** that addresses this. When
the draft hit rate falls below a threshold, the verify block length is
shortened automatically — so a missed draft costs less, and the verify
overhead doesn't blow up at long context. The re-test showed 32K decode
recovering from 12.6 tps to 89.7 tps under our standard cooldown=2
methodology, and 121.2 tps under a cooldown=60 methodology that lets the
chip recover thermally between runs. Either number is competitive with the
other frameworks at 32K; the cliff is gone.

The broader lesson is that "framework X collapses at context length Y" can be
a transient state of the framework, not a fundamental property of the
algorithm. The original v0.1.0 analysis was correct *as a description of what
v0.1.0 was doing*, but it was wrong to extrapolate that to "speculative
decoding can't work at long context." It can; you just need the verify-cost
control loop to be in place.

## TTFT matters for RAG

When context climbs into the 16K and beyond range, TTFT — the time before the first token streams to the user — becomes a real user-experience problem. At 32K, the best-case TTFT in our updated test is 11.9 seconds (dflash-mlx v0.1.7), with omlx essentially tied at 12.7 seconds. For RAG applications, that's a long time to leave the user staring at a blank screen. The fix is partly UX (show a thinking indicator, stream the prompt back, or display partial retrieval results), but partly architectural — if your RAG pipeline pulls 32K of context per query, you should expect roughly 12 seconds of overhead before the model starts responding, and design accordingly.

## Lessons from the methodology

We learned a few things about how to benchmark these frameworks fairly that are worth documenting.

**Prefix cache is a double-edged sword.** Our first pass produced impossible-looking prefill numbers because cached KV reuse was being measured. The fix is to disable prefix caching at server start (with the framework's specific flag), or alternatively to add a unique prefix to each prompt to defeat the cache. Just calling `/v1/cache/clear` between runs is *not* enough — we tried that and saw residual cache effects.

**Variance testing is mandatory at long context.** At short context (under 512 tokens) every framework's standard deviation across five runs was under 3 tps. Single-shot benchmarks are fine in that regime. But at 16K and beyond, real run-to-run variability appeared, and one-shot numbers can be misleading by 10–20%. The cost of doing five-run benchmarks (about 10 extra minutes per framework) is small compared to the cost of making a wrong decision based on noisy data.

**Cooldown between runs is a real methodology variable on Apple Silicon.** The dflash-mlx v0.1.7 re-test exposed this — at 32K, decode tps with `cooldown=2` was 89.7 with std=6.5 (a clear monotonic degradation across the 5 runs from thermal pressure), but with `cooldown=60` it was 121.2 with std=0.2. The 35% gap is the chip throttling under back-to-back load. Neither number is "right" — they measure different things ("steady-state throughput under back-to-back load" vs. "single-request peak performance"). The lab's standard is cooldown=2 for cross-framework fairness, but the cooldown=60 numbers are the right reference if you want to know what the framework can actually do when the chip is cool. We didn't re-test the other frameworks at cooldown=60; if you're comparing them at long context for production sizing, it's worth measuring your own.

**Qwen3's `/no_think` is honored inconsistently.** Of the four frameworks, only mlx-vlm and dflash-mlx fully suppressed the reasoning tokens. rapid-mlx and omlx still emitted `reasoning_content` chunks, and rapid-mlx specifically didn't even respect `max_tokens` for the thinking phase — we saw outputs as long as 2,155 tokens when we asked for 256. If you need strict thinking control, set `enable_thinking=false` through the chat-template parameters rather than relying on the in-prompt convention.

## What we'd test next

The most interesting follow-up is mlx-vlm with `--draft-kind dflash`. mlx-vlm has the strongest prefill in the mid-context range and dflash has the strongest short-context decode; combining them might give the best of both worlds, or might surface an interaction effect where neither speedup actually applies. We didn't have time for it in this round.

Continuous-batching throughput under concurrent load is the second priority. Both rapid-mlx and omlx support it but we tested only single-sequence latency. Under realistic multi-user load, the gap between batched and non-batched frameworks will widen, and it's worth knowing by how much. dflash-mlx specifically does not benefit from batching, which means its short-context advantage might disappear in production.

KV-cache quantization is the third. rapid-mlx and mlx-vlm both support 4-bit and 8-bit KV caches; this should help long-context performance considerably, particularly for rapid-mlx which currently trails omlx at 32K. If quantized KV closes that gap, the framework choice for production starts looking different.

Beyond 32K is the fourth — at 64K and 128K, KV-cache memory pressure (16 GB to 32 GB respectively for this model in fp16) starts to dominate, and the comparison becomes as much about memory management as about compute throughput. That's a different benchmark entirely, but it's a regime that's becoming more important for real applications.
