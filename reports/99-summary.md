# Cross-Framework Summary and Lessons Learned

[繁體中文版](99-summary_zh.md)

## Who wins what

The headline metrics fan out across the four frameworks pretty cleanly. dflash-mlx wins short-context decode (64 to 2K) thanks to its speculative-decoding speedup, often by a 30–35% margin. omlx wins long-context decode (4K through 32K), and by a meaningful gap — at 32K it's running at 82 tokens per second while dflash-mlx has fallen to 12. omlx also wins prefill (peaking at 3,989 tps at 4K) and TTFT consistency (standard deviation in the 3–25 ms range at moderate context, ten times tighter than the others). mlx-vlm wins decode-stability at short context (stddev of 0.2 tps at 64 and 512 tokens) and ties for best prefill at 8K. rapid-mlx never wins outright on any single metric but is competitive everywhere, and has the most flexible feature set.

For multimodal input — image, audio, video — mlx-vlm is the only choice in this set, so the comparison there is moot.

## The fast-versus-stable tradeoff

The most interesting pattern in the data is how dramatically the framework rankings depend on what you're optimizing for. dflash-mlx hits 167 tps at 64 tokens, but pays for it with 200+ ms of extra TTFT compared to omlx, and with the catastrophic 32K crash. omlx peaks at 124 tps but does so in a perfectly steady, predictable way that's far easier to build SLAs around. Choosing between them is a question about your traffic patterns, not a question about which framework is "better."

Two principles fall out of this. First, peak speed and stability tend to trade off — the speedup from speculative decoding requires hitting a draft prediction, and when that hit rate varies by content, speed varies too. Second, there is no universal winner. Every framework here has a regime where it wins and a regime where it loses, and picking the right one starts with identifying which regime your workload is in.

## Why rapid-mlx looked stronger in the first round

In our initial benchmarking pass, rapid-mlx's prefill numbers were impressive (over 5,000 tps at 4K and over 100,000 tps at 8K). It turned out those numbers were artifacts of prefix-cache hits — the warm-up run cached the prompt prefix, and runs 2 through 5 reused the cached KV state instead of running prefill at all. The fix was launching with `--disable-prefix-cache`, after which rapid-mlx's prefill dropped to a more realistic 3,070 tps at 4K, behind omlx and mlx-vlm.

This is worth noting because in production, your traffic *does* benefit from prefix caching when prompts share a system prefix or RAG header. So if your real workload has high prefix repetition (a fixed system message, for instance), rapid-mlx's effective prefill in production will be much higher than what this cold-cache benchmark shows. The cold numbers are the lower bound; the cached numbers are the upper bound.

## Why dflash-mlx falls apart at 32K

Speculative decoding wins by amortizing main-model forward passes across many proposed tokens. At short context this works because the draft model is much faster than the main model — running it once is cheap, and the verification batch through the main model accepts most of its proposals. But two things break at long context. First, the draft model has to ingest the full 32K prompt the same way the main model does, eliminating much of the cost asymmetry. Second, the draft's prediction accuracy degrades as context grows, because long context conditioning is fundamentally harder for a small model to handle. Failed predictions cost both a draft pass and a wasted main-model forward, plus the rollback work to recover.

The numbers are stark: at 64 tokens the per-decoded-token amortized cost was about 6 ms; at 32K it was 79 ms — a 13× slowdown in per-token cost for the same model on the same hardware. That's not a small regression; that's the speculative-decoding strategy ceasing to work.

## TTFT matters for RAG

When context climbs into the 16K and beyond range, TTFT — the time before the first token streams to the user — becomes a real user-experience problem. At 32K, the best-case TTFT in our test was 12.7 seconds (omlx). For RAG applications, that's a long time to leave the user staring at a blank screen. The fix is partly UX (show a thinking indicator, stream the prompt back, or display partial retrieval results), but partly architectural — if your RAG pipeline pulls 32K of context per query, you should expect roughly 13 seconds of overhead before the model starts responding, and design accordingly.

## Lessons from the methodology

We learned a few things about how to benchmark these frameworks fairly that are worth documenting.

**Prefix cache is a double-edged sword.** Our first pass produced impossible-looking prefill numbers because cached KV reuse was being measured. The fix is to disable prefix caching at server start (with the framework's specific flag), or alternatively to add a unique prefix to each prompt to defeat the cache. Just calling `/v1/cache/clear` between runs is *not* enough — we tried that and saw residual cache effects.

**Variance testing is mandatory at long context.** At short context (under 512 tokens) every framework's standard deviation across five runs was under 3 tps. Single-shot benchmarks are fine in that regime. But at 16K and beyond, real run-to-run variability appeared, and one-shot numbers can be misleading by 10–20%. The cost of doing five-run benchmarks (about 10 extra minutes per framework) is small compared to the cost of making a wrong decision based on noisy data.

**Qwen3's `/no_think` is honored inconsistently.** Of the four frameworks, only mlx-vlm and dflash-mlx fully suppressed the reasoning tokens. rapid-mlx and omlx still emitted `reasoning_content` chunks, and rapid-mlx specifically didn't even respect `max_tokens` for the thinking phase — we saw outputs as long as 2,155 tokens when we asked for 256. If you need strict thinking control, set `enable_thinking=false` through the chat-template parameters rather than relying on the in-prompt convention.

## What we'd test next

The most interesting follow-up is mlx-vlm with `--draft-kind dflash`. mlx-vlm has the strongest prefill in the mid-context range and dflash has the strongest short-context decode; combining them might give the best of both worlds, or might surface an interaction effect where neither speedup actually applies. We didn't have time for it in this round.

Continuous-batching throughput under concurrent load is the second priority. Both rapid-mlx and omlx support it but we tested only single-sequence latency. Under realistic multi-user load, the gap between batched and non-batched frameworks will widen, and it's worth knowing by how much. dflash-mlx specifically does not benefit from batching, which means its short-context advantage might disappear in production.

KV-cache quantization is the third. rapid-mlx and mlx-vlm both support 4-bit and 8-bit KV caches; this should help long-context performance considerably, particularly for rapid-mlx which currently trails omlx at 32K. If quantized KV closes that gap, the framework choice for production starts looking different.

Beyond 32K is the fourth — at 64K and 128K, KV-cache memory pressure (16 GB to 32 GB respectively for this model in fp16) starts to dominate, and the comparison becomes as much about memory management as about compute throughput. That's a different benchmark entirely, but it's a regime that's becoming more important for real applications.
