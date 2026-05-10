#!/usr/bin/env python3
"""從 JSONL 原始資料產生圖表。

輸出至 ../charts/：
- decode_tps.png       Decode tps 隨 context 變化（折線）
- prefill_tps.png      Prefill tps 隨 context 變化（折線）
- ttft.png             TTFT 隨 context 變化（對數刻度）
- decode_box.png       每個 context 的 decode tps 變異盒鬚圖
- degradation.png      decode tps 衰退率比較
"""
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 中文字型 (Heiti TC 為 macOS 內建台灣正體字型)
rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "Arial Unicode MS"]
rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 110

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
CHARTS = ROOT / "charts"
CHARTS.mkdir(exist_ok=True)

FRAMEWORKS = [
    ("rapid-mlx", "rapid_v5.jsonl",  "#1f77b4", "o"),
    ("omlx",      "omlx_v5.jsonl",   "#2ca02c", "s"),
    ("dflash-mlx","dflash_v5.jsonl", "#d62728", "^"),
    ("mlx-vlm",   "vlm_v5.jsonl",    "#ff7f0e", "D"),
]
SIZES = [64, 512, 2048, 4096, 8192, 16384, 32768]


def load(path):
    """讀 JSONL 並依 size 群組。"""
    by_size = {s: [] for s in SIZES}
    for line in open(path):
        r = json.loads(line)
        if r["size"] in by_size:
            by_size[r["size"]].append(r)
    return by_size


def median_for(rows, key):
    vals = [r[key] for r in rows if r.get(key)]
    return statistics.median(vals) if vals else None


def plot_metric(metric_key, title, ylabel, fname, log_y=False):
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, fn, color, marker in FRAMEWORKS:
        data = load(DATA / fn)
        x, y = [], []
        for sz in SIZES:
            v = median_for(data[sz], metric_key)
            if v and v > 0:
                x.append(sz)
                y.append(v)
        if x:
            ax.plot(x, y, marker=marker, color=color, label=name, linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    if log_y:
        ax.set_yscale("log")
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{s:,}" for s in SIZES], rotation=15)
    ax.set_xlabel("Prompt context (tokens)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHARTS / fname)
    plt.close(fig)
    print(f"Wrote {CHARTS / fname}")


def plot_decode_box():
    """每個 size × framework 的 decode tps 盒鬚圖。"""
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
    axes = axes.flatten()
    sizes_to_plot = SIZES
    all_data = {name: load(DATA / fn) for name, fn, *_ in FRAMEWORKS}

    for i, sz in enumerate(sizes_to_plot):
        ax = axes[i]
        bp_data, labels, colors = [], [], []
        for name, _, color, _ in FRAMEWORKS:
            vals = [r["decode_tps"] for r in all_data[name][sz] if r.get("decode_tps", 0) > 0]
            if vals:
                bp_data.append(vals)
                labels.append(name.replace("-mlx", "").replace("mlx-", ""))
                colors.append(color)
        bp = ax.boxplot(bp_data, tick_labels=labels, patch_artist=True, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.6)
        ax.set_title(f"context {sz:,}", fontsize=10)
        ax.set_ylabel("decode tps")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)

    # 第 8 格留空
    axes[7].axis("off")
    fig.suptitle("Decode tps 變異分佈（n=5/each）", fontsize=14)
    fig.tight_layout()
    fig.savefig(CHARTS / "decode_box.png")
    plt.close(fig)
    print(f"Wrote {CHARTS / 'decode_box.png'}")


def plot_degradation():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, fn, color, marker in FRAMEWORKS:
        data = load(DATA / fn)
        baseline = median_for(data[SIZES[0]], "decode_tps")
        if not baseline:
            continue
        x, y = [], []
        for sz in SIZES:
            v = median_for(data[sz], "decode_tps")
            if v:
                x.append(sz)
                y.append(v / baseline * 100)
        ax.plot(x, y, marker=marker, color=color, label=name, linewidth=2, markersize=7)
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.set_xticklabels([f"{s:,}" for s in SIZES], rotation=15)
    ax.axhline(100, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Prompt context (tokens)")
    ax.set_ylabel("Decode tps（相對於 64-token 基準的 %）")
    ax.set_title("Decode 速度衰退曲線（基準 = 各框架在 64 tokens 的速度）")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(CHARTS / "degradation.png")
    plt.close(fig)
    print(f"Wrote {CHARTS / 'degradation.png'}")


def plot_decode_stddev():
    fig, ax = plt.subplots(figsize=(9, 5.4))
    width = 0.2
    for i, (name, fn, color, _) in enumerate(FRAMEWORKS):
        data = load(DATA / fn)
        stds = []
        for sz in SIZES:
            vals = [r["decode_tps"] for r in data[sz] if r.get("decode_tps", 0) > 0]
            stds.append(statistics.stdev(vals) if len(vals) > 1 else 0)
        xs = [j + i * width for j in range(len(SIZES))]
        ax.bar(xs, stds, width=width, color=color, label=name, alpha=0.8)
    ax.set_xticks([j + 1.5 * width for j in range(len(SIZES))])
    ax.set_xticklabels([f"{s:,}" for s in SIZES], rotation=15)
    ax.set_xlabel("Prompt context (tokens)")
    ax.set_ylabel("Decode tps stddev")
    ax.set_title("Decode 速度穩定性（stddev 越低越穩定）")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHARTS / "decode_stddev.png")
    plt.close(fig)
    print(f"Wrote {CHARTS / 'decode_stddev.png'}")


if __name__ == "__main__":
    plot_metric("decode_tps", "Decode 速度隨 context 長度變化", "Decode tps（中位數）", "decode_tps.png")
    plot_metric("prefill_tps", "Prefill 速度隨 context 長度變化", "Prefill tps（中位數）", "prefill_tps.png")
    plot_metric("ttft_ms", "TTFT 隨 context 長度變化（對數刻度）", "TTFT (ms, log scale)", "ttft.png", log_y=True)
    plot_decode_box()
    plot_degradation()
    plot_decode_stddev()
    print("Done.")
