"""
Stitch saved results into a single image.

Layout (transposed):
  - Left column  = text labels: top cell "Original image", then one model name per row.
  - Each other column = one input image: original on top, each model's prediction below.

Reads originals from input/ and depth maps from output/<model_key>/<image>.png
(produced by compare.py). No models are re-run.
"""
import os
import csv
import glob
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
INPUT_DIR = os.path.join(HERE, "input")
OUTPUT_DIR = os.path.join(HERE, "output")

# (model_key, display label) — order = top-to-bottom rows under the originals.
MODELS = [
    ("depth_anything_v2_small", "Depth Anything V2-S"),
    ("dpt_large",               "DPT-Large (MiDaS)"),
    ("dpt_hybrid",              "DPT-Hybrid (MiDaS)"),
    ("zoedepth",                "ZoeDepth (metric)"),
    ("glpn_nyu",                "GLPN-NYU"),
]


def load_timings():
    """Return {display_label: avg_infer_seconds} from output/timings.csv."""
    path = os.path.join(OUTPUT_DIR, "timings.csv")
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                out[row["model"]] = float(row["avg_infer_s"])
            except (ValueError, KeyError):
                pass
    return out


def image_names():
    paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return [os.path.splitext(os.path.basename(p))[0] for p in paths]


def main():
    names = image_names()
    if not names:
        raise SystemExit("No input images found.")

    n_rows = 1 + len(MODELS)      # originals + one per model
    n_cols = 1 + len(names)       # label column + one per image

    # Nicer display names for the top row.
    pretty = {
        "indoor_cats": "Indoor",
        "landscape": "Outdoor crowd",
        "person": "Person (skier)",
        "street": "Indoor kitchen",
    }

    BG = "#0e1117"      # dark, post-friendly background
    FG = "#f5f6fa"      # near-white text
    ACCENT = "#ffb454"  # warm accent for the title

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.0 + 4.0 * len(names), 0.6 + 3.4 * n_rows),
        gridspec_kw={"width_ratios": [1.15] + [1.0] * len(names)},
    )
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)

    # blank every axis first
    for r in range(n_rows):
        for c in range(n_cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            axes[r, c].set_facecolor(BG)
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)

    # ---- left label column (model name + speed) ----
    timings = load_timings()
    axes[0, 0].text(0.5, 0.5, "Original\nimage", ha="center", va="center",
                    fontsize=28, fontweight="bold", color=FG, linespacing=1.3)
    for r, (_, lbl) in enumerate(MODELS, start=1):
        axes[r, 0].text(0.5, 0.60, lbl, ha="center", va="center",
                        fontsize=27, fontweight="bold", color=FG, linespacing=1.3)
        secs = timings.get(lbl)
        if secs:
            speed = f"{secs * 1000:.0f} ms  •  {1.0 / secs:.1f} FPS"
            axes[r, 0].text(0.5, 0.36, speed, ha="center", va="center",
                            fontsize=19, color=ACCENT, fontweight="bold")

    # ---- top row: original images ----
    for c, name in enumerate(names, start=1):
        p = next((q for q in glob.glob(os.path.join(INPUT_DIR, name + ".*"))), None)
        if p:
            axes[0, c].imshow(Image.open(p).convert("RGB"))
        axes[0, c].set_title(pretty.get(name, name), fontsize=24,
                             fontweight="bold", color=FG, pad=14)

    # ---- prediction rows ----
    for r, (key, _) in enumerate(MODELS, start=1):
        for c, name in enumerate(names, start=1):
            mp = os.path.join(OUTPUT_DIR, key, name + ".png")
            if os.path.exists(mp):
                axes[r, c].imshow(Image.open(mp))
            else:
                axes[r, c].text(0.5, 0.5, "N/A", ha="center", va="center", color=FG)

    fig.suptitle(
        "Monocular Depth Estimation  —  5-Model Comparison",
        fontsize=40, fontweight="bold", color=ACCENT, y=0.995,
    )
    fig.text(0.5, 0.952,
             "Relative-depth models: bright = near, dark = far   •   ZoeDepth is metric: bright = far"
             "   •   speed measured on NVIDIA GTX 1660 Ti (6 GB)",
             ha="center", fontsize=18, color=FG, style="italic")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(OUTPUT_DIR, "stitched_comparison.png")
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor=BG)
    print(f"Saved {out}  ({n_rows} rows x {n_cols} cols)")


if __name__ == "__main__":
    main()
