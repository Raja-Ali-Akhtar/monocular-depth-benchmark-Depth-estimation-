"""
Compare 5 monocular depth-estimation models on the same images.

Models:
  1. Depth Anything V2 (Small)  - relative depth, SOTA generalization
  2. DPT-Large (MiDaS 3.0)       - relative depth, transformer backbone
  3. DPT-Hybrid (MiDaS)          - relative depth, lighter hybrid backbone
  4. ZoeDepth (NYU+KITTI)        - METRIC depth (meters)
  5. GLPN (NYU)                  - relative/metric-ish indoor depth

For each model we record load time, per-image inference time, and save:
  - per-model colorized depth maps in  output/<model>/<image>.png
  - a combined side-by-side grid       output/comparison_grid.png
  - a timing table                     output/timings.csv
"""
import os
import csv
import time
import glob
import gc

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import pipeline

HERE = os.path.dirname(__file__)
INPUT_DIR = os.path.join(HERE, "input")
OUTPUT_DIR = os.path.join(HERE, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = 0 if torch.cuda.is_available() else -1
print(f"Device: {'cuda:0 (' + torch.cuda.get_device_name(0) + ')' if DEVICE == 0 else 'cpu'}")

# (key, label, hf_model_id, is_metric)
MODELS = [
    ("depth_anything_v2_small", "Depth Anything V2-S", "depth-anything/Depth-Anything-V2-Small-hf", False),
    ("dpt_large",               "DPT-Large (MiDaS)",   "Intel/dpt-large",                          False),
    ("dpt_hybrid",              "DPT-Hybrid (MiDaS)",  "Intel/dpt-hybrid-midas",                   False),
    ("zoedepth",                "ZoeDepth (metric)",   "Intel/zoedepth-nyu-kitti",                 True),
    ("glpn_nyu",                "GLPN-NYU",            "vinvino02/glpn-nyu",                       False),
]

CMAP = "inferno"  # near = bright, far = dark (after normalization)


def load_images():
    paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    if not paths:
        raise SystemExit(f"No images found in {INPUT_DIR}. Run download_samples.py first.")
    imgs = [(os.path.splitext(os.path.basename(p))[0], Image.open(p).convert("RGB")) for p in paths]
    print(f"Loaded {len(imgs)} images: {[n for n, _ in imgs]}")
    return imgs


def to_depth_array(result):
    """Extract a float32 HxW depth array from a pipeline result."""
    pd = result.get("predicted_depth")
    if pd is not None:
        arr = pd.squeeze().detach().cpu().numpy().astype(np.float32)
    else:
        arr = np.asarray(result["depth"]).astype(np.float32)
    return arr


def normalize_for_view(arr):
    """Min-max normalize to [0,1] for colorization (robust to outliers)."""
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    if hi <= lo:
        hi = arr.max() if arr.max() > lo else lo + 1e-6
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def main():
    images = load_images()
    n_img = len(images)
    n_mod = len(MODELS)

    # depth_store[model_key][img_name] = normalized array; meta for metric range
    depth_store = {k: {} for k, _, _, _ in MODELS}
    raw_ranges = {k: {} for k, _, _, _ in MODELS}
    timings = []  # rows: model, load_s, avg_infer_s, per-image...

    for key, label, model_id, is_metric in MODELS:
        print(f"\n=== {label}  ({model_id}) ===")
        t0 = time.perf_counter()
        try:
            pipe = pipeline("depth-estimation", model=model_id, device=DEVICE)
        except Exception as e:
            print(f"  !! failed to load: {e}")
            timings.append({"model": label, "load_s": "ERR", "avg_infer_s": "ERR"})
            continue
        load_s = time.perf_counter() - t0
        print(f"  loaded in {load_s:.1f}s")

        # warmup (first call includes CUDA kernel compilation)
        try:
            _ = pipe(images[0][1])
        except Exception as e:
            print(f"  !! inference failed: {e}")
            del pipe; gc.collect(); torch.cuda.empty_cache()
            timings.append({"model": label, "load_s": f"{load_s:.1f}", "avg_infer_s": "ERR"})
            continue

        infer_times = []
        for name, img in images:
            t1 = time.perf_counter()
            res = pipe(img)
            if DEVICE == 0:
                torch.cuda.synchronize()
            dt = time.perf_counter() - t1
            infer_times.append(dt)

            arr = to_depth_array(res)
            raw_ranges[key][name] = (float(arr.min()), float(arr.max()))
            depth_store[key][name] = normalize_for_view(arr)

            # save individual colorized map
            mdir = os.path.join(OUTPUT_DIR, key)
            os.makedirs(mdir, exist_ok=True)
            plt.imsave(os.path.join(mdir, f"{name}.png"),
                       depth_store[key][name], cmap=CMAP)
            print(f"  {name:14s} {dt*1000:7.1f} ms   raw[min,max]=[{arr.min():.3f},{arr.max():.3f}]"
                  + ("  (meters)" if is_metric else ""))

        avg = sum(infer_times) / len(infer_times)
        row = {"model": label, "load_s": f"{load_s:.1f}", "avg_infer_s": f"{avg:.3f}"}
        for (name, _), t in zip(images, infer_times):
            row[name] = f"{t:.3f}"
        timings.append(row)
        print(f"  avg inference: {avg*1000:.1f} ms/image")

        del pipe
        gc.collect()
        if DEVICE == 0:
            torch.cuda.empty_cache()

    # ---- build comparison grid: rows = images, cols = [orig + each model] ----
    print("\nBuilding comparison grid...")
    cols = 1 + n_mod
    fig, axes = plt.subplots(n_img, cols, figsize=(3.2 * cols, 3.0 * n_img))
    if n_img == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["Input"] + [lbl for _, lbl, _, _ in MODELS]

    for r, (name, img) in enumerate(images):
        axes[r, 0].imshow(img)
        axes[r, 0].set_ylabel(name, fontsize=11)
        for c, (key, lbl, _, is_metric) in enumerate(MODELS, start=1):
            ax = axes[r, c]
            if name in depth_store[key]:
                ax.imshow(depth_store[key][name], cmap=CMAP)
                if is_metric and name in raw_ranges[key]:
                    lo, hi = raw_ranges[key][name]
                    ax.set_xlabel(f"{lo:.1f}-{hi:.1f} m", fontsize=8)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
        for c in range(cols):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=11)

    fig.suptitle("Monocular Depth Estimation — Model Comparison (inferno: bright=near, dark=far)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    grid_path = os.path.join(OUTPUT_DIR, "comparison_grid.png")
    fig.savefig(grid_path, dpi=110)
    print(f"  saved {grid_path}")

    # ---- timings csv ----
    csv_path = os.path.join(OUTPUT_DIR, "timings.csv")
    fields = ["model", "load_s", "avg_infer_s"] + [n for n, _ in images]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in timings:
            w.writerow(row)
    print(f"  saved {csv_path}")

    # ---- console summary ----
    print("\n================ TIMING SUMMARY ================")
    print(f"{'Model':22s} {'Load(s)':>8s} {'Avg infer(s)':>13s}")
    for row in timings:
        print(f"{row['model']:22s} {str(row['load_s']):>8s} {str(row['avg_infer_s']):>13s}")
    print("\nDone. See output/comparison_grid.png")


if __name__ == "__main__":
    main()
