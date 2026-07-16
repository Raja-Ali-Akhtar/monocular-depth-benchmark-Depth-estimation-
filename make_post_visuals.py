"""
Generate LinkedIn-ready post visuals (square, dark theme) for the optimization win:
  output/post_hero_speedup.png   - before->after FPS bar chart (the hook)
  output/post_what_worked.png    - what worked / what didn't card
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE = os.path.dirname(__file__)
OUT = os.path.join(HERE, "output")

BG = "#0e1117"
FG = "#f5f6fa"
MUTED = "#9aa4b2"
ACCENT = "#ffb454"
GREEN = "#4caf50"
RED = "#ef5350"


def hero_speedup():
    labels = ["PyTorch fp32\n@518 (original)", "+ resolution\n294px",
              "+ TensorRT FP16\n@294px", "max speed\n210px"]
    fps = [22.9, 63.8, 104.9, 171.1]
    colors = ["#5b6472", "#e0803a", ACCENT, "#ffd28a"]

    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    bars = ax.bar(range(len(fps)), fps, color=colors, width=0.68, zorder=3)

    for i, (b, v) in enumerate(zip(bars, fps)):
        ax.text(b.get_x() + b.get_width() / 2, v + 3, f"{v:.0f}", ha="center",
                va="bottom", fontsize=26, fontweight="bold", color=FG)
        mult = v / fps[0]
        if i > 0:
            ax.text(b.get_x() + b.get_width() / 2, v / 2, f"{mult:.1f}x",
                    ha="center", va="center", fontsize=22, fontweight="bold", color=BG)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=14, color=FG, fontweight="bold")
    ax.set_ylim(0, 195)
    ax.set_ylabel("Inference speed (FPS)", fontsize=16, color=MUTED)
    ax.tick_params(axis="y", colors=MUTED)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(axis="y", alpha=0.12, zorder=0)

    fig.suptitle("Depth Anything V2  —  4.6x faster inference", x=0.5, y=0.975,
                 fontsize=22, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.925, "same model, same GPU (GTX 1660 Ti, 6 GB)  •  visually identical depth",
             ha="center", fontsize=15, color=MUTED, style="italic")
    fig.tight_layout(rect=[0, 0.01, 1, 0.90])
    p = os.path.join(OUT, "post_hero_speedup.png")
    fig.savefig(p, dpi=120, facecolor=BG); print("saved", p)


def what_worked():
    rows = [
        (True,  "Input resolution 518->294", "2.8x free, visually identical"),
        (True,  "TensorRT FP16 engine",       "+1.6x, numerically stable (corr 1.000)"),
        (False, "ONNX Runtime CUDA EP",       "no gain (same as PyTorch eager)"),
        (False, "Eager PyTorch FP16",         "4x SLOWER + NaNs (no tensor cores)"),
        (False, "torch.compile",              "needs Triton (no Windows support)"),
        (False, "INT8 (ORT QDQ -> TensorRT)", "TensorRT can't consume QDQ for this ViT"),
    ]
    fig, ax = plt.subplots(figsize=(9, 9))
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.axis("off")

    fig.suptitle("What worked, what didn't", x=0.5, y=0.95,
                 fontsize=28, fontweight="bold", color=ACCENT)
    fig.text(0.5, 0.895, "optimizing depth inference on a tensor-core-less GPU",
             ha="center", fontsize=15, color=MUTED, style="italic")

    y = 0.80
    for ok, name, note in rows:
        mark = "OK" if ok else "X"
        mcol = GREEN if ok else RED
        ax.text(0.03, y, mark, transform=ax.transAxes, fontsize=22, fontweight="bold",
                color=mcol, ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", fc=BG, ec=mcol, lw=2))
        ax.text(0.16, y + 0.018, name, transform=ax.transAxes, fontsize=18,
                fontweight="bold", color=FG, ha="left", va="center")
        ax.text(0.16, y - 0.028, note, transform=ax.transAxes, fontsize=13.5,
                color=MUTED, ha="left", va="center")
        y -= 0.132

    fig.text(0.5, 0.045, "Winner: resolution + TensorRT FP16  =  4.6x faster, same quality",
             ha="center", fontsize=15.5, fontweight="bold", color=ACCENT)
    p = os.path.join(OUT, "post_what_worked.png")
    fig.savefig(p, dpi=120, facecolor=BG); print("saved", p)


if __name__ == "__main__":
    hero_speedup()
    what_worked()
