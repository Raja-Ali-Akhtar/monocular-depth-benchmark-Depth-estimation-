"""
Visual quality check for the resolution optimization of Depth Anything V2-Small.
Renders baseline (518) vs 294 vs 210 depth maps so we can confirm low-res
inference keeps the depth structure. Saves output/optimization_quality.png
"""
import os, glob
import numpy as np
from PIL import Image
import torch, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
DEV = "cuda"
SIZES = [518, 294, 210]
TITLES = {518: "518 (baseline)\n23.6 FPS", 294: "294\n84 FPS (3.6x)", 210: "210\n168 FPS (7.2x)"}

proc = AutoImageProcessor.from_pretrained(MODEL)
model = AutoModelForDepthEstimation.from_pretrained(MODEL).to(DEV).eval()

imgs = [Image.open(p).convert("RGB") for p in sorted(glob.glob(os.path.join(HERE, "input", "*")))[:3]]


@torch.inference_mode()
def depth(img, size):
    px = proc(img, return_tensors="pt", size={"height": size, "width": size},
              keep_aspect_ratio=False, ensure_multiple_of=14)["pixel_values"].to(DEV)
    d = model(pixel_values=px).predicted_depth.float()
    d = F.interpolate(d.unsqueeze(1), size=img.size[::-1], mode="bilinear", align_corners=False)
    a = d.squeeze().cpu().numpy()
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)

ncol = 1 + len(SIZES)
fig, axes = plt.subplots(len(imgs), ncol, figsize=(3.2 * ncol, 2.8 * len(imgs)))
axes = np.atleast_2d(axes)
for r, img in enumerate(imgs):
    axes[r, 0].imshow(img)
    if r == 0: axes[r, 0].set_title("Input", fontsize=12, fontweight="bold")
    for c, s in enumerate(SIZES, 1):
        axes[r, c].imshow(depth(img, s), cmap="inferno")
        if r == 0: axes[r, c].set_title(TITLES[s], fontsize=11, fontweight="bold")
    for c in range(ncol):
        axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
fig.suptitle("Depth Anything V2-S — resolution vs speed (GTX 1660 Ti)", fontsize=14, fontweight="bold")
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = os.path.join(HERE, "output", "optimization_quality.png")
fig.savefig(out, dpi=120)
print("saved", out)
