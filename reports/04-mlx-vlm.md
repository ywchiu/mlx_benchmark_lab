# mlx-vlm — Detailed Report

[繁體中文版](04-mlx-vlm_zh.md)

## How we ran it

```bash
python3 -m mlx_vlm server \
  --model mlx-community/Qwen3.6-35B-A3B-4bit \
  --port 8765
```

mlx-vlm is the vision-language-model framework in this set, which means its primary purpose is feeding images, video, or audio to multimodal models alongside text. We tested it with text-only input here, partly to keep the comparison fair (the other three frameworks are text-only) and partly because the Qwen3.6-35B-A3B-4bit model can run text-only requests through mlx-vlm's stack. Note that mlx-vlm also supports `--draft-model` with `--draft-kind dflash`, which would in theory combine vlm's strong prefill with dflash's decode speedup — we did not test this combination, but it's an obvious follow-up.

---

## Full statistics (n=5 per cell)

### Decode tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 95.5 | 95.4 | 0.2 | 95.1 | 95.6 |
| 512 | 94.8 | 94.8 | 0.2 | 94.5 | 95.1 |
| 2,048 | 88.5 | 89.2 | 4.1 | 83.3 | 93.1 |
| 4,096 | 91.4 | 91.5 | 0.3 | 91.1 | 92.0 |
| 8,192 | 87.2 | 85.8 | 2.4 | 82.4 | 87.9 |
| 16,384 | 83.1 | 82.7 | 1.1 | 81.3 | 84.0 |
| 32,768 | 67.7 | 67.3 | 2.7 | 63.1 | 70.6 |

### Prefill tps

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 592 | 605 | 41 | 570 | 677 |
| 512 | 1,826 | 1,817 | 30 | 1,779 | 1,846 |
| 2,048 | 3,370 | 3,287 | 126 | 3,143 | 3,395 |
| 4,096 | 3,672 | 3,679 | 21 | 3,655 | 3,712 |
| 8,192 | 3,818 | 3,762 | 150 | 3,498 | 3,853 |
| 16,384 | 2,850 | 2,855 | 80 | 2,741 | 2,964 |
| 32,768 | 2,075 | 2,064 | 70 | 1,987 | 2,162 |

### TTFT (ms)

| size | median | mean | stddev | min | max |
|---:|---:|---:|---:|---:|---:|
| 64 | 133 | 131 | 8 | 117 | 139 |
| 512 | 240 | 242 | 4 | 238 | 247 |
| 2,048 | 498 | 511 | 20 | 494 | 534 |
| 4,096 | 908 | 907 | 5 | 899 | 913 |
| 8,192 | 1,739 | 1,767 | 74 | 1,723 | 1,898 |
| 16,384 | 4,650 | 4,645 | 131 | 4,471 | 4,834 |
| 32,768 | 12,761 | 12,840 | 432 | 12,247 | 13,326 |

---

## What the data says

mlx-vlm has the lowest decode throughput across every context length we tested, sitting 25–30% behind the next slowest framework at short context and roughly 18% behind omlx at 32K. That's the cost of running through the vision-language stack even for text-only requests — there are tokenizer and processor layers in the inference pipeline that aren't strictly needed when the input is text.

But it has two compensating strengths. First, the variability is exceptionally low. At 64 and 512 tokens the standard deviation across five runs is just 0.2 tps — the lowest of any framework we measured at any context length. If you're optimizing for predictability rather than peak speed, mlx-vlm is unusually consistent. Second, prefill is excellent. mlx-vlm hits the highest prefill rate of any framework at 8K context (3,818 tps), and it stays competitive with omlx through 32K. So while the per-token generation is slow, the per-token prompt ingestion is among the fastest.

The decode degradation curve is also the gentlest in this comparison: from 64 to 32K mlx-vlm loses only 29% of its baseline speed, slightly better than omlx's 34%. This is partly because the 64-token baseline was already low — there's less to lose — but the 32K result of 67.7 tps still puts it ahead of dflash-mlx and within striking distance of rapid-mlx (72.3).

The 2,048-token cell shows higher variance (stddev 4.1 tps, min 83 max 93) than its neighbors — likely an artifact of one or two slower runs at that size. Excluding outliers, mlx-vlm is otherwise tightly bunched.

---

## When to use mlx-vlm

The honest answer is: when you need vision, audio, or video input. None of the other three frameworks support those, so for any multimodal use case mlx-vlm is the only option in this set. The Qwen3.6-35B-A3B-4bit model used here is itself multimodal — its preprocessor configs include vision components — so feeding it images is a one-flag change to the request payload.

For pure text, mlx-vlm is hard to recommend over omlx unless you specifically value low variance over peak throughput. The 25–30% decode tps gap at short context is a real cost, and rapid-mlx, omlx, and dflash-mlx all beat it on absolute speed in their respective comfort zones.

The interesting middle ground is mlx-vlm's `--draft-kind dflash` support. If you have moderate context (under 8K) and predictable output, you might be able to combine mlx-vlm's strong prefill with dflash's decode speedup. This would test whether the speculative-decoding gains are additive with mlx-vlm's prefill efficiency, or whether one cancels the other out. We didn't measure this combination — it's the most interesting follow-up experiment from this round.

---

## Things to watch out for

The default port is 8080, which collides with many other tools. We explicitly set 8765 for our benchmarks. There's no `/health` endpoint, so use `GET /v1/models` for readiness checks.

The `--prefill-step-size` parameter (default 2048) controls how big each prefill batch is. For very long prompts you may want to tune this; smaller values use less memory but more wall-clock time. We used the default for our tests.

mlx-vlm respects the `/no_think` Qwen3 convention more strictly than rapid-mlx and omlx do — output stayed at exactly 256 tokens (max_tokens), with no reasoning_content emitted. So if you need clean control over output length and content, mlx-vlm is the most well-behaved framework in this set.

The CLI also exposes `python3 -m mlx_vlm generate` for one-shot inference and `python3 -m mlx_vlm convert` for model preparation. The streaming server at `python3 -m mlx_vlm server` is what we benchmarked. Note that the `mlx_vlm.generate` deprecation warning suggests using the new `mlx_vlm generate` form instead.
