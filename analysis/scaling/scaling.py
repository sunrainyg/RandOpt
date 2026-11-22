#!/usr/bin/env python
import argparse
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

THRESHOLDS = [0.90, 0.95, 0.98, 1.00, 1.03, 1.05]


def load_run(run_dir):
    """-> (model size in B, pooled base accuracy, per-seed accuracies)"""
    d = Path(run_dir)
    bases = [json.loads(f.read_text()) for f in sorted(d.glob("base*.json"))]
    seeds = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    if not bases or not seeds:
        raise SystemExit(f"{d}: need base*.json and seed*.json (found {len(bases)} / {len(seeds)})")
    size = float(re.search(r"(\d+(?:\.\d+)?)B", Path(seeds[0]["model"]).name).group(1))
    acc = lambda r: r["tasks"]["gsm8k"]["acc"]
    return size, float(np.mean([acc(b) for b in bases])), np.array([acc(r) for r in seeds])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one run directory per model")
    ap.add_argument("--out", default="figure3.png")
    args = ap.parse_args()

    runs = sorted(load_run(r) for r in args.runs)
    sizes = [s for s, _, _ in runs]
    density = np.array([[100 * np.mean(acc >= m * base - 1e-9) for m in THRESHOLDS] for _, base, acc in runs])
    print("GSM8K delta(m) in %\nmodel  base   " + " ".join(f"{m:>6.0%}" for m in THRESHOLDS))
    for (s, base, _), row in zip(runs, density):
        print(f"{s:>4g}B  {base:.3f} " + " ".join(f"{d:6.1f}" for d in row))

    fig, ax = plt.subplots(figsize=(5.5, 4.3))
    for m, col, colour in zip(THRESHOLDS, density.T, plt.cm.viridis(np.linspace(0.1, 0.9, len(THRESHOLDS)))):
        ax.plot(sizes, col, "-o", ms=4, color=colour, label=f">= {m:.0%} of base")
    ax.set(xscale="log", ylim=(-3, 103), xticks=sizes, xticklabels=[f"{s:g}B" for s in sizes],
           xlabel="model scale", ylabel="solution density  δ(m)  [%]")
    ax.set_title("GSM8K", fontweight="bold")
    ax.minorticks_off()
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print("figure:", args.out)


if __name__ == "__main__":
    main()
