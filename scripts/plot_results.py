#!/usr/bin/env python3
"""Generate charts from raw JSONL data.

Outputs to ../charts/:
- decode_tps.png       Decode tps vs context length
- prefill_tps.png      Prefill tps vs context length
- ttft.png             TTFT vs context length (log-y)
- decode_box.png       Per-context decode tps boxplot
- decode_stddev.png    Decode tps variability bar chart
- degradation.png      Decode tps degradation curve

Pass --lang zh to render with Traditional Chinese labels (uses Heiti TC font).
Charts go to charts/ (English) or charts/zh/ (Chinese).
"""
import argparse
import json
import statistics
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 110

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

FRAMEWORKS = [
    ("rapid-mlx", "rapid_v5.jsonl",  "#1f77b4", "o"),
    ("omlx",      "omlx_v5.jsonl",   "#2ca02c", "s"),
    ("dflash-mlx","dflash_v5.jsonl", "#d62728", "^"),
    ("mlx-vlm",   "vlm_v5.jsonl",    "#ff7f0e", "D"),
]
SIZES = [64, 512, 2048, 4096, 8192, 16384, 32768]

LABELS = {
    "en": {
        "x_context": "Prompt context (tokens)",
        "decode_title":  "Decode speed vs context length",
        "decode_y":      "Decode tps (median)",
        "prefill_title": "Prefill speed vs context length",
        "prefill_y":     "Prefill tps (median)",
        "ttft_title":    "TTFT vs context length (log scale)",
        "ttft_y":        "TTFT (ms, log scale)",
        "box_title":     "Decode tps distribution (n=5 per cell)",
        "box_y":         "decode tps",
        "deg_title":     "Decode degradation (baseline = each framework's tps at 64 tokens)",
        "deg_y":         "Decode tps (% of 64-token baseline)",
        "std_title":     "Decode stability (lower stddev = more stable)",
        "std_y":         "Decode tps stddev",
    },
    "zh": {
        "x_context": "Prompt context (tokens)",
        "decode_title":  "Decode 速度隨 context 長度變化",
        "decode_y":      "Decode tps（中位數）",
        "prefill_title": "Prefill 速度隨 context 長度變化",
        "prefill_y":     "Prefill tps（中位數）",
        "ttft_title":    "TTFT 隨 context 長度變化（對數刻度）",
        "ttft_y":        "TTFT (ms, log scale)",
        "box_title":     "Decode tps 變異分佈（n=5/each）",
        "box_y":         "decode tps",
        "deg_title":     "Decode 速度衰退曲線（基準 = 各框架在 64 tokens 的速度）",
        "deg_y":         "Decode tps（相對於 64-token 基準的 %）",
        "std_title":     "Decode 速度穩定性（stddev 越低越穩定）",
        "std_y":         "Decode tps stddev",
    },
}


def load(path):
    by_size = {s: [] for s in SIZES}
    for line in open(path):
        r = json.loads(line)
        if r["size"] in by_size:
            by_size[r["size"]].append(r)
    return by_size


def median_for(rows, key):
    vals = [r[key] for r in rows if r.get(key)]
    return statistics.median(vals) if vals else None


def plot_metric(charts_dir, L, metric_key, title_key, ylabel_key, fname, log_y=False):
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
    ax.set_xlabel(L["x_context"])
    ax.set_ylabel(L[ylabel_key])
    ax.set_title(L[title_key])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts_dir / fname)
    plt.close(fig)
    print(f"Wrote {charts_dir / fname}")


def plot_decode_box(charts_dir, L):
    fig, axes = plt.subplots(2, 4, figsize=(15, 7), sharey=False)
    axes = axes.flatten()
    all_data = {name: load(DATA / fn) for name, fn, *_ in FRAMEWORKS}

    for i, sz in enumerate(SIZES):
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
        ax.set_ylabel(L["box_y"])
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelsize=8)

    axes[7].axis("off")
    fig.suptitle(L["box_title"], fontsize=14)
    fig.tight_layout()
    fig.savefig(charts_dir / "decode_box.png")
    plt.close(fig)
    print(f"Wrote {charts_dir / 'decode_box.png'}")


def plot_degradation(charts_dir, L):
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
    ax.set_xlabel(L["x_context"])
    ax.set_ylabel(L["deg_y"])
    ax.set_title(L["deg_title"])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(charts_dir / "degradation.png")
    plt.close(fig)
    print(f"Wrote {charts_dir / 'degradation.png'}")


def plot_decode_stddev(charts_dir, L):
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
    ax.set_xlabel(L["x_context"])
    ax.set_ylabel(L["std_y"])
    ax.set_title(L["std_title"])
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    fig.tight_layout()
    fig.savefig(charts_dir / "decode_stddev.png")
    plt.close(fig)
    print(f"Wrote {charts_dir / 'decode_stddev.png'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", choices=["en", "zh"], default="en")
    args = ap.parse_args()

    L = LABELS[args.lang]
    if args.lang == "zh":
        rcParams["font.sans-serif"] = ["Heiti TC", "PingFang HK", "Arial Unicode MS"]
        charts_dir = ROOT / "charts" / "zh"
    else:
        rcParams["font.sans-serif"] = rcParams["font.sans-serif"]
        charts_dir = ROOT / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    plot_metric(charts_dir, L, "decode_tps",  "decode_title",  "decode_y",  "decode_tps.png")
    plot_metric(charts_dir, L, "prefill_tps", "prefill_title", "prefill_y", "prefill_tps.png")
    plot_metric(charts_dir, L, "ttft_ms",     "ttft_title",    "ttft_y",    "ttft.png", log_y=True)
    plot_decode_box(charts_dir, L)
    plot_degradation(charts_dir, L)
    plot_decode_stddev(charts_dir, L)
    print("Done.")


if __name__ == "__main__":
    main()
