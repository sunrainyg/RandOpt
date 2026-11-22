#!/usr/bin/env python
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, hsv_to_rgb, rgb_to_hsv

COLS = [("nll", "Pretrain LL"), ("gsm8k", "GSM8K"), ("countdown", "Countdown"), ("olympiad", "OlyBench")]
RGB = ("gsm8k", "olympiad", "countdown")
GRID = 120
EXTENT = 2.6
BANDWIDTH = 0.30
MIN_BASE_CORRECT = 5
BLACK, GOLD = "#1a1a1a", "#FFD700"


def load_run(run_dir):
    """-> (model label, projected coordinates, {task: pooled base}, {task: per-seed values}, problems per task)"""
    d = Path(run_dir)
    bases = [json.loads(f.read_text())["tasks"] for f in sorted(d.glob("base*.json"))]
    seeds = [json.loads(f.read_text()) for f in sorted(d.glob("seed*.json"))]
    if not bases or not seeds:
        raise SystemExit(f"{d}: need base*.json and seed*.json (found {len(bases)} / {len(seeds)})")
    key = lambda t: "nll" if t == "nll" else "acc"
    base = {t: float(np.mean([b[t][key(t)] for b in bases])) for t, _ in COLS}
    vals = {t: np.array([r["tasks"][t][key(t)] for r in seeds]) for t, _ in COLS}
    label = Path(seeds[0]["model"]).name.replace("Qwen2.5-", "").replace("-Instruct", "")
    proj = np.array([r["proj"] for r in seeds]) * seeds[0]["sigma"]
    return label, proj, base, vals, bases[0]["gsm8k"]["n"]


def smooth(xy, vals, grid, h):
    """Nadaraya-Watson: Gaussian-kernel weighted mean of vals at every grid point"""
    w = np.exp(-0.5 * ((grid[:, None, :] - xy[None, :, :]) ** 2).sum(-1) / h ** 2) + 1e-12
    return ((w * vals[None, :]).sum(1) / w.sum(1)).reshape(GRID, GRID)


def rgb_composite(lands, brightness):
    """per-channel [2, 98] percentile stretch -> brightness -> 16 colour levels -> saturation boost"""
    img = np.stack([np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
                    for z in lands for lo, hi in [np.percentile(z, [2, 98])]], -1)
    img = np.floor(img * brightness * 255 / 16) * 17 / 255
    lum = (img * [0.299, 0.587, 0.114]).sum(-1, keepdims=True)
    hsv = rgb_to_hsv(np.clip(lum + 1.5 * (img - lum), 0, 1))
    hsv[..., 1] *= 0.85
    return hsv_to_rgb(hsv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one run directory per row (small -> large model)")
    ap.add_argument("--out", default="figure2.png")
    args = ap.parse_args()

    rows = []
    for run in args.runs:
        label, proj, base, vals, n_problems = load_run(run)
        radius = float(np.sqrt((proj ** 2).sum(1)).mean())
        lim, h = EXTENT * radius, BANDWIDTH * radius
        g = np.linspace(-lim, lim, GRID)
        grid = np.stack([a.ravel() for a in np.meshgrid(g, g)], 1)
        xy = np.vstack([proj, [[0.0, 0.0]]])
        panels, density = {}, []
        for t, _ in COLS:
            b, v = base[t], vals[t]
            small_base = t != "nll" and b * n_problems < MIN_BASE_CORRECT
            if t == "nll":
                change = (b - v) / b * 100
            else:
                change = (v - b) * 100 if small_base else (v - b) / b * 100
                density.append(f"{t} base={b:.3f} P(acc>=base)=" + ("n/a" if small_base
                                                                     else f"{np.mean(v >= b - 1e-9):.1%}"))
            best = int(np.argmax(change))
            panels[t] = dict(z=smooth(xy, np.append(change, 0.0), grid, h), small_base=small_base,
                             best=proj[best] if change[best] > 0 else (0.0, 0.0))  # nothing beats the base
        lands = [smooth(xy, np.append(vals[t], base[t]), grid, h) for t in RGB]
        rows.append(dict(label=label, lim=lim, radius=radius, g=g, panels=panels, lands=lands))
        print(f"{label:5s} n={len(proj):5d} | " + " | ".join(density))

    # RGB brightness: where the model's mean accuracy sits between the minimum and maximum over all models
    lo = np.mean([min(r["lands"][c].min() for r in rows) for c in range(3)])
    hi = np.mean([max(r["lands"][c].max() for r in rows) for c in range(3)])

    plt.rcParams["font.family"] = "serif"
    cmap = plt.get_cmap("RdYlBu_r")
    n = len(rows)
    fig = plt.figure(figsize=(18.5, 3.1 * n + 0.3))
    gs = fig.add_gridspec(n, 5, left=0.04, right=0.83, top=0.95, bottom=0.03, hspace=0.04, wspace=0.02)
    for i, r in enumerate(rows):
        lim, g = r["lim"], r["g"]
        for j, (t, title) in enumerate(COLS):
            ax, p = fig.add_subplot(gs[i, j]), r["panels"][t]
            z = p["z"]
            if t == "nll":
                q5, q95 = np.percentile(z, [5, 95])
                vmin, vmax = q5 - 0.5 * (q95 - q5), q95 + 4.0 * (q95 - q5)
            else:
                vmax = max(np.abs(np.percentile(z, [3, 97])).max(), 5.0 if p["small_base"] else 0.1)
                vmin = -vmax
            ax.pcolormesh(g, g, z, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto", rasterized=True)
            ax.add_patch(plt.Circle((0, 0), r["radius"], fill=False, ls="--", lw=1.1, color="#333333", alpha=0.8))
            ax.scatter(0, 0, marker="*", s=150, c=BLACK, edgecolors="white", linewidths=0.8, zorder=10)
            ax.scatter(*p["best"], marker="*", s=130, c=GOLD, edgecolors=BLACK, linewidths=1.0, zorder=11)
            ax.set(xlim=(-lim, lim), ylim=(-lim, lim), aspect="equal", xticks=[], yticks=[])
            if j == 0:
                ax.set_ylabel(r["label"], fontsize=20, fontweight="bold", labelpad=10)
            if i == 0:
                ax.set_title(title, fontsize=20, fontweight="bold", pad=8)
        ax = fig.add_subplot(gs[i, 4])
        frac = np.clip((np.mean([z.mean() for z in r["lands"]]) - lo) / max(hi - lo, 1e-6), 0, 1)
        ax.imshow(rgb_composite(r["lands"], 0.35 + 0.65 * frac ** 1.5), origin="lower",
                  extent=[-lim, lim, -lim, lim], interpolation="nearest")
        ax.scatter(0, 0, marker="*", s=120, c="black", edgecolors="white", linewidths=0.8, zorder=10)
        ax.set(xticks=[], yticks=[])
        if i == 0:
            ax.set_title("RGB", fontsize=20, fontweight="bold", pad=8)

    # ---- legend and colorbar ----
    lg = fig.add_axes([0.845, 0.03, 0.15, 0.92])
    lg.set(xlim=(0, 1), ylim=(0, 1))
    lg.axis("off")
    lg.add_patch(plt.Rectangle((0.001, 0.655), 0.998, 0.345, fc="none", ec="black", lw=0.8))
    for y, colour, text in ((0.94, BLACK, "Pretrained Model"), (0.875, GOLD, "Best Perturbation")):
        lg.scatter(0.08, y, marker="*", s=260, c=colour, edgecolors=BLACK, linewidths=0.8)
        lg.text(0.17, y, text, fontsize=14, fontweight="bold", va="center", color="#222222")
    lg.text(0.09, 0.815, "RGB Composite:", fontsize=14, fontweight="bold", va="center", color="#222222")
    for y, colour, text in ((0.772, "#cc2222", "R = GSM8K"), (0.734, "#22aa22", "G = OlyBench"),
                            (0.696, "#2255cc", "B = Countdown")):
        lg.text(0.09, y, text, fontsize=13, va="center", color=colour)
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(-1, 1)),
                        cax=fig.add_axes([0.89, 0.06, 0.0375, 0.53]))
    cbar.set_label("Accuracy Change", fontsize=14, labelpad=6)
    cbar.set_ticks([-1, 0, 1], labels=["Worst", "Base", "Best"], fontsize=13)

    fig.savefig(args.out, dpi=150, facecolor="white")
    print("figure:", args.out)


if __name__ == "__main__":
    main()
