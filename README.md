# MLX Inference Framework Benchmark Lab

[中文版 (Traditional Chinese)](README_zh.md)

## What this is

This repository contains the raw data, scripts, and analysis from a head-to-head speed comparison of four MLX inference frameworks running on Apple Silicon: **rapid-mlx**, **omlx**, **dflash-mlx**, and **mlx-vlm**. All four were tested against the same model (`mlx-community/Qwen3.6-35B-A3B-4bit`, a 4-bit quantized mixture-of-experts model with 35B total parameters and 3B active per token) across seven prompt-context lengths from 64 tokens up to 32,768 tokens, with five repeated runs per cell so we could measure both the typical performance and the variability.

A fifth framework, **mtplx**, was added in a later run against a *different* model (`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, a 27B dense model purpose-built for MTPLX with speculative-decoding optimizations). Because the model differs from the four-framework comparison, mtplx is plotted with a dashed line and marked with `*` in tables — its numbers are interesting on their own merits but not directly comparable to the others. See [§4.1 mtplx (different model)](#41-mtplx-different-model) for the standalone results and methodology.

The short version of the conclusion: if you mostly serve long-context workloads (RAG, document summarization, code analysis on big files), use **omlx** — it has the fastest decode speed from 4K context onward and the most stable timing of any framework tested. If your prompts are short and your output is structured or predictable, **dflash-mlx** is the fastest by a wide margin at 64–2,048 tokens because its speculative decoding hits often on those workloads. But dflash-mlx fails catastrophically at 32K, dropping to 12.6 tokens per second — roughly six times slower than the others — so you absolutely cannot use it for long-context applications. **mlx-vlm** is the only framework here that supports image, video, and audio input, but for pure text it runs about 25–30% slower than the others, so reach for it only when you actually need multimodal capability. **mtplx** running its speed-optimized 27B model decodes at 31–60 tokens per second across the ladder with steady memory growth (15.6 GB at 512 tokens, 22.1 GB at 32K), so its appeal is the predictable memory envelope rather than peak throughput.

---

## 1. Test environment

The hardware was an Apple M5 Max with 64 GB of unified memory. The four primary frameworks all loaded the same target model (`mlx-community/Qwen3.6-35B-A3B-4bit`); only dflash-mlx additionally loaded the companion draft model `z-lab/Qwen3.6-35B-A3B-DFlash` to drive its speculative decoding. All servers exposed an OpenAI-compatible streaming endpoint at `/v1/chat/completions`, and the benchmark client talked to them via that endpoint, so the comparison is genuinely apples-to-apples at the API surface even though the internals differ. Those tests were run on 2026-05-09.

The follow-on **mtplx** run was conducted on 2026-05-16 against `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, using MTPLX's built-in `prefill-ladder` harness rather than the shared HTTP client. Settings: `sustained` mode, MTP with `depth=3`, `disable-thinking` enabled, and `max_tokens=128` per cell. Only one run per context length was collected, and peak memory was sampled at each cell.

---

## 2. Methodology

We tested seven prompt context lengths — 64, 512, 2,048, 4,096, 8,192, 16,384, and 32,768 tokens — by generating filler text of the appropriate length and asking the model to summarize it. For each cell we ran the request five times and took the median, mean, and standard deviation, so we could distinguish "this framework is faster" from "this framework happens to have been faster on this one run." Before the timed runs we ran one full-size warm-up that we discarded, to make sure Metal kernels and weights were already on the GPU before we started measuring.

A subtle but important point: we explicitly disabled prefix caching on every framework that supports it (`--disable-prefix-cache` on rapid-mlx, `--no-cache` on omlx). Our first round of tests had wildly inflated prefill numbers — over 100,000 tokens per second on a 35B model, which is physically impossible — because the warm-up primed the prefix cache and subsequent runs reused the cached KV state instead of actually computing prefill. With caching disabled, every run measures honest cold-prefill performance.

The prompt was prefixed with `/no_think` (Qwen3's convention to suppress reasoning output), and we set `max_tokens=256`. Each framework was launched on its own server on port 8765, benchmarked, then shut down before the next framework started, so there was no memory contention between frameworks. We did not test batching or concurrent requests — these numbers reflect single-sequence latency.

The benchmark client is at [`scripts/bench_inline.py`](scripts/bench_inline.py); the chart generator is at [`scripts/plot_results.py`](scripts/plot_results.py). Raw logs are in [`logs/`](logs/) and raw per-run data is in [`data/`](data/) as JSONL.

---

## 3. Visualizations

### 3.1 Decode speed by context length

![Decode tps](charts/decode_tps.png)

This is the headline chart. It plots median decode tokens per second against prompt context length. The picture tells the story: dflash-mlx (red) has dramatic peaks at 64 and 2,048 tokens where speculative decoding lands well, but its line cliff-dives at 32K. omlx (green) is the boring-but-effective straight line that beats everything from 4K onward. rapid-mlx (blue) starts strong at small context but degrades faster than omlx as context grows. mlx-vlm (orange) is consistently the slowest but also the flattest line. The dashed purple line is mtplx running its 27B speed-optimized model — it sits well below the four 35B-MoE frameworks because the 27B dense weights are heavier per active parameter, but the curve from 4K onward is unusually flat (43→31 tps from 4K to 32K).

### 3.2 Decode degradation curve

![Degradation](charts/degradation.png)

The same data normalized so each framework starts at 100% of its own smallest-context baseline (64 tokens for the original four; 512 tokens for mtplx, which wasn't run at 64). This isolates "how much does decoding slow down as the KV cache grows" from "which framework is fastest in absolute terms." omlx degrades the least (from 100% to 66% of baseline at 32K), mlx-vlm is even flatter relatively but only because its baseline was already lowest. rapid-mlx loses about 42% of its speed by 32K. dflash-mlx falls off a cliff: by 32K it's running at 7.5% of its 64-token speed. mtplx ends up at about 52% of its 512-token baseline at 32K — a steeper drop than omlx but with no cliff, which matches its calmer memory profile.

### 3.3 Decode stability (standard deviation)

![Decode stddev](charts/decode_stddev.png)

This is the variability across the five runs at each context length. Lower bars mean the framework was more predictable, run to run. At short context every framework is stable (under 3 tps stddev), but as context grows you start to see real jitter. The takeaway here is that single-shot benchmarks become unreliable at 16K and beyond — your actual results in production might be ±5 tps from a single measurement, so when you're picking a framework based on long-context performance, run it multiple times yourself. *(mtplx is omitted from this chart and the boxplot below because only a single run was collected per context — there is no variance to plot.)*

### 3.4 Per-context decode distribution

![Decode boxplot](charts/decode_box.png)

Boxplots of decode tps for each (framework, context) cell. The boxes show the interquartile range; the whiskers extend to min and max across the 5 runs. This is useful for spotting cases where the median hides a wide spread (e.g., rapid-mlx at 2,048 tokens has a noticeable spread because thinking-token output length varied across runs).

### 3.5 Prefill speed

![Prefill tps](charts/prefill_tps.png)

Prefill tokens per second measures how fast the model digests the prompt before it starts generating. dflash-mlx isn't shown because its OpenAI server doesn't return `prompt_tokens` in `usage` — we'd have to back it out from TTFT. The three 35B-MoE frameworks we can measure all peak in the 4K–8K range, which is the sweet spot for the attention-and-bandwidth tradeoff on this hardware. mtplx's prefill is dramatically lower in absolute terms (peaks at 879 tps at 1K) because the 27B dense model is more prefill-heavy per token, and its TTFT cost dominates total latency well before decode does.

### 3.6 TTFT (time to first token, log scale)

![TTFT](charts/ttft.png)

Time to first token is what your end user perceives as latency before the model starts streaming. The y-axis is log scale because TTFT spans nearly three orders of magnitude across context lengths. Note dflash-mlx jumping above the rest at 32K — that's the 31-second TTFT, more than twice everyone else. mtplx sits noticeably above the four 35B-MoE frameworks at every context length and reaches 62 seconds at 32K — its weaker prefill is the single biggest tax this configuration pays, so mtplx is much more attractive for short prompts than long ones.

---

## 4. Decode tps median summary table

| Prompt size | rapid-mlx | omlx | dflash-mlx | mlx-vlm | mtplx\* |
|---:|---:|---:|---:|---:|---:|
| 64 | 124.9 | 123.7 | **167.3** | 95.5 | — |
| 512 | 119.5 | 119.4 | **122.9** | 94.8 | 59.8 |
| 1,024 | — | — | — | — | 49.6 |
| 2,048 | 102.5 | 121.1 | **160.1** | 88.5 | 55.7 |
| 4,096 | 97.6 | **120.4** | 104.5 | 91.4 | 43.3 |
| 8,192 | 90.3 | **118.0** | 96.3 | 87.2 | 43.1 |
| 16,384 | 83.2 | **105.3** | 84.1 | 83.1 | 41.4 |
| 32,768 | 72.3 | **82.1** | 12.6 ⚠️ | 67.7 | 31.3 |

\* mtplx ran a *different* model (`Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`, 27B dense) and a different methodology (n=1, 128 output tokens, MTP depth=3, `disable-thinking`), so its column is not directly comparable to the others.

For the full statistics — mean, standard deviation, min, max — see the per-framework deep-dive reports under [`reports/`](reports/).

### 4.1 mtplx (different model)

mtplx is included here as a reference point for what the MTPLX framework looks like on the model it was tuned for. Because both the runtime *and* the model differ from the other four cells, treat this section as a standalone profile rather than a fifth comparison column.

**Configuration.** Model: `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`. Harness: MTPLX's built-in `prefill-ladder` in `sustained` mode, with `--mtp --depth 3 --disable-thinking --max-tokens 128`. Single run per context length.

| Context | Decode tps | Prefill tps | TTFT | Peak memory |
|---:|---:|---:|---:|---:|
| 512 | 59.76 | 800.21 | 0.65 s | 15.58 GB |
| 1,024 | 49.56 | 879.06 | 1.17 s | 16.18 GB |
| 2,048 | 55.69 | 720.81 | 2.84 s | 17.29 GB |
| 4,096 | 43.28 | 693.90 | 5.90 s | 17.73 GB |
| 8,192 | 43.09 | 664.82 | 12.32 s | 18.37 GB |
| 16,384 | 41.40 | 646.68 | 25.35 s | 19.62 GB |
| 32,768 | 31.34 | 530.79 | 61.74 s | 22.12 GB |

The shape of the curve is what stands out. Decode is roughly 50–60 tps in the short-context band (512–2K), slides into a 41–43 tps plateau from 4K through 16K, and only really gives up at 32K where it drops to 31 tps. The decode rate is well below the 35B-MoE frameworks because the 35B-MoE model only activates 3B parameters per token while this 27B dense model activates all 27B — memory-bandwidth-per-token is much higher. What compensates is the memory envelope: peak GPU memory grows from 15.6 GB at 512 tokens to only 22.1 GB at 32K, which is small enough that a 36 GB Mac would have headroom for the full ladder.

The weak spot is TTFT. Prefill tops out below 900 tps and the 32K cell costs over a minute before the first token streams — so this configuration is much better suited to short prompts with long outputs than to RAG-style long-prompt workloads.

---

## 5. Practical recommendations

For an interactive chat application where the user types a short message and waits for a response, dflash-mlx is the right choice if you can tolerate roughly 300 milliseconds of extra time-to-first-token. Its 167 tokens-per-second decode at short context is dramatically faster than the alternatives, and the slightly worse TTFT often goes unnoticed because total response time is still dominated by the generation phase. If TTFT matters more than throughput — say, you want responses to start streaming as quickly as possible — use omlx; it has both the lowest median TTFT and the lowest TTFT variance across the small-context range.

For retrieval-augmented generation, long-document summarization, or any workload that pushes context into the 4K–32K range, omlx is the clear winner. It maintains over 100 tokens per second through 16K context, hits 82 tokens per second even at 32K, and its decode tps barely moves between runs. Frameworks that look great in short-context benchmarks may not survive the move to long context — dflash-mlx is the cautionary tale here, going from class leader to class disaster as context grows.

For code generation specifically, dflash-mlx is worth considering even at moderate context lengths because code is highly predictable and speculative decoding hits more often on structured output than on free-form natural language. We saw 160 tokens per second at 2K context where the natural-language test produced only 122 — your code-completion workload might do better than even that.

If you need to serve images, audio, or video to the model, mlx-vlm is the only choice in this set. The 25–30% slower text decode is a tax you pay for the multimodal stack, but if you need vision capability, no other framework here can deliver it. mlx-vlm does have an interesting feature we did not test: it supports `--draft-kind dflash`, which would in theory combine its strong prefill performance with dflash's decode speedup. If your context lengths are bounded under 8K, this combination might be the best of both worlds — but be aware of dflash's 32K problem.

mtplx on the 27B speed-optimized model is worth considering if your bottleneck is memory rather than tokens-per-second, or if your prompts are short. Peak memory stays under 23 GB even at 32K context, so an M-series Mac with 36 GB unified memory has plenty of headroom; the cost is a decode rate that lands between 30 and 60 tps depending on context. It's a good fit for short-prompt, long-output generation (drafting, code completion, structured output) on memory-constrained machines, and a poor fit for long-context RAG where TTFT dominates total latency.

For production deployments where stability and predictability matter more than peak speed, omlx is again the right call. Its standard deviation in TTFT and decode is the lowest in this benchmark, often by a factor of ten compared to rapid-mlx. If you're wiring up an SLA and need to make promises about p99 latency, omlx will let you make those promises more confidently.

---

## 6. Conclusions

The benchmark exposed a clear pattern: **omlx is the strongest all-around long-context framework**. It wins or ties every metric from 4K context onward, has the lowest variance, and has the gentlest decode-tps degradation curve as context grows. Setup is a little more involved (it expects models in a specific directory rather than reading from your HuggingFace cache directly), but for production use it's the most defensible default.

**dflash-mlx is the most interesting case**. Its speculative-decoding architecture genuinely delivers a 35% decode speedup at small context, but the architectural cost of running a draft model alongside the main model becomes ruinous at long context. The draft network has to process the full prompt too, and the verification phase between draft and main becomes the bottleneck. By 32K, the speculative-decoding overhead exceeds the cost of just decoding directly — and the framework's decode rate falls to 12.6 tokens per second, which is honestly unusable. Treat dflash-mlx as a specialized tool for short-context workloads with predictable output.

**rapid-mlx is the dependable middle option**. It's never the absolute fastest at any size, but it's never far from the leader either. Its main weakness is TTFT jitter at small context, where we measured a standard deviation of 136 milliseconds against a median of 169 milliseconds — meaning some requests are nearly twice as slow as others for no obvious reason. If your application can tolerate that variance, rapid-mlx is a solid choice with the most flexible feature set (paged KV cache, MTP, prefix cache, KV quantization).

**mlx-vlm is the multimodal special case**. For pure text it's the slowest in every cell. But it's the only framework here with vision and audio support, so the comparison is somewhat unfair: you don't pick mlx-vlm for raw text speed, you pick it because you need to feed it images.

**mtplx is the memory-constrained option**, but evaluated against a different model so the comparison needs an asterisk. On its own 27B dense speed-optimized model it never crosses 60 tps and pays an outsized TTFT cost at long context, but it also keeps peak memory under 23 GB through 32K — substantially less than any of the four 35B-MoE frameworks would consume at the same context. If your hardware constraint is "fit in 24 GB at long context," mtplx + this model is one of the few combinations that meets that bar; if your constraint is "maximize throughput," it isn't competitive with omlx.

The other major finding from this round was about benchmarking methodology itself. Single-shot speed numbers at long context are misleading — variance is real, and the difference between two frameworks at 32K can easily be smaller than the difference between two runs of the same framework. The five-run-with-warmup methodology used here costs about 10 extra minutes per framework, but it makes the difference between "this is faster" and "this looks faster on one run."

---

## 7. Repository layout

```
mlx_benchmark_lab/
├── README.md                     # This file (English, primary)
├── README_zh.md                  # Traditional Chinese version
├── data/                         # Raw JSONL (one run per line)
│   ├── rapid_v5.jsonl
│   ├── omlx_v5.jsonl
│   ├── dflash_v5.jsonl
│   ├── vlm_v5.jsonl
│   └── mtplx_v5.jsonl            # mtplx + Qwen3.6-27B-MTPLX-Optimized-Speed (n=1)
├── logs/                         # Full test logs
├── scripts/
│   ├── bench_inline.py           # Streaming benchmark client
│   └── plot_results.py           # Chart generator (--lang en|zh)
├── reports/                      # Per-framework deep dives (EN + ZH)
│   ├── 01-rapid-mlx.md
│   ├── 01-rapid-mlx_zh.md
│   ├── 02-omlx.md
│   ├── 02-omlx_zh.md
│   ├── 03-dflash-mlx.md
│   ├── 03-dflash-mlx_zh.md
│   ├── 04-mlx-vlm.md
│   ├── 04-mlx-vlm_zh.md
│   ├── 99-summary.md
│   └── 99-summary_zh.md
└── charts/
    ├── *.png                     # English-labeled charts
    └── zh/                       # Chinese-labeled charts
        └── *.png
```

---

## 8. Reproducing this benchmark

The full set of steps to reproduce these numbers on your own Mac is below. Each step assumes you have Python 3.11+ and the relevant framework already installed.

```bash
# 1. Install the framework you want to test
pip install rapid-mlx       # or omlx, dflash-mlx, mlx-vlm

# 2. Download the model (HuggingFace cache)
huggingface-cli download mlx-community/Qwen3.6-35B-A3B-4bit
huggingface-cli download z-lab/Qwen3.6-35B-A3B-DFlash   # only for dflash

# 3. Launch the server (rapid-mlx example)
rapid-mlx serve mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765 --disable-prefix-cache &

# 4. Run the benchmark
python3 scripts/bench_inline.py \
  --url http://localhost:8765 \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --sizes 64,512,2048,4096,8192,16384,32768 \
  --runs 5 \
  --max-tokens 256 \
  --json-out data/rapid_v5.jsonl > logs/rapid_v5.log

# 5. Generate charts
python3 scripts/plot_results.py --lang en
python3 scripts/plot_results.py --lang zh    # optional Chinese variant
```

The mtplx row in `data/mtplx_v5.jsonl` was produced with MTPLX's built-in `prefill-ladder` harness (not `bench_inline.py`), against `Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed`:

```bash
mtplx prefill-ladder \
  --model /Users/david/.mtplx/models/Youssofal--Qwen3.6-27B-MTPLX-Optimized-Speed \
  --mode sustained --mtp --depth 3 --disable-thinking \
  --sizes 512,1024,2048,4096,8192,16384,32768 --max-tokens 128
```

The bench script handles streaming, parses Server-Sent Events, separates thinking tokens (`reasoning_content`) from visible content tokens, and computes per-run statistics. It also handles the case where the server doesn't expose `/v1/cache/clear` by silently swallowing the 404.

---

## 9. Limitations and follow-up work

This benchmark only covers the single-sequence case. Both rapid-mlx and omlx support continuous batching, which would change the comparison significantly under concurrent load — under heavy traffic, omlx and rapid-mlx might pull farther ahead because they can amortize prefill across simultaneous requests. dflash-mlx's speculative decoding is fundamentally single-stream and doesn't benefit from batching at all.

We tested only the 4-bit MoE model. Dense models (e.g., Qwen3-32B-Dense) and larger active-parameter MoE models would have different bottlenecks; on a dense 32B model, prefill becomes more compute-bound than memory-bound, and the relative ordering of frameworks could shift. KV-cache quantization is supported by rapid-mlx and mlx-vlm but we did not explore how it changes long-context decode performance — it would likely shrink the omlx advantage at 32K because rapid-mlx specifically benefits from quantized KV.

The mlx-vlm + dflash combination was identified but not measured. It might be the most interesting follow-up: vlm has the strongest prefill at mid-context, dflash has the strongest decode at short context, and combining them would test whether the speedups stack or interfere.

`/no_think` was honored partially — mlx-vlm and dflash-mlx fully respect it, but rapid-mlx and omlx still emit reasoning tokens. The decode tps numbers from those two therefore mix thinking-token throughput with visible-content throughput, and although the rates are similar, they're not identical. A more rigorous follow-up would set `enable_thinking=false` via the chat-template parameter rather than relying on the in-prompt convention.

dflash-mlx's OpenAI-compatible server doesn't return `prompt_tokens` in its `usage` object, which means we couldn't compute prefill tps for it without resorting to externally-tokenized prompt counts. We instead omitted dflash from the prefill chart. A small patch to dflash-serve to populate `usage.prompt_tokens` would make future comparisons cleaner.

Finally, we didn't go beyond 32K context. At 64K and 128K the KV cache becomes a major memory consumer (roughly 16 GB and 32 GB respectively for this model in fp16), and the comparison would start measuring memory pressure as much as compute speed. That's an important regime for some applications and we'd like to revisit it.
