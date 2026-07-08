"""
Benchmark optimization variants for Depth Anything V2-Small.

For each variant we measure inference speed (FPS) and a quality delta vs. the
FP32 / full-resolution baseline, so no optimization silently trades away
accuracy. Uses the model + processor directly (not the pipeline) for full
control over dtype, memory format, resolution and torch.compile.

Variants:
  - fp32                 : baseline (518x518, float32)
  - fp16                 : half precision
  - fp16 + channels_last : half precision + channels_last memory format
  - fp16 + compile       : torch.compile(reduce-overhead)  [may fail on Windows]
  - res-<N> (fp16)       : input resolution sweep at half precision

Quality vs baseline is reported on the predicted (inverse) depth, normalized
per-image to [0,1] then compared:
  - AbsRel : mean |pred - base| / (base + eps)   (lower = closer to baseline)
  - Corr   : Pearson correlation of the depth maps (higher = better, 1.0 = identical)
"""
import os
import glob
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForDepthEstimation, AutoImageProcessor

HERE = os.path.dirname(__file__)
INPUT_DIR = os.path.join(HERE, "input")
MODEL_PATH = os.path.join(HERE, "models", "depth_anything_v2_small")
if not os.path.isdir(MODEL_PATH):
    MODEL_PATH = "depth-anything/Depth-Anything-V2-Small-hf"  # fall back to hub

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 5
ITERS = 20          # timed iterations per image
PATCH = 14          # DINOv2 patch size -> resolutions must be multiples of 14

assert DEVICE == "cuda", "This benchmark targets GPU inference."
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Model:  {MODEL_PATH}\n")


def load_images():
    paths = sorted(
        p for p in glob.glob(os.path.join(INPUT_DIR, "*"))
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
    )
    return [Image.open(p).convert("RGB") for p in paths]


IMAGES = load_images()
print(f"{len(IMAGES)} images loaded\n")

processor = AutoImageProcessor.from_pretrained(MODEL_PATH)


def preprocess(images, size):
    """Return a single batched float tensor at the given square size."""
    inputs = processor(
        images, return_tensors="pt",
        size={"height": size, "width": size},
        keep_aspect_ratio=False,      # force exact square so the batch shares one shape
        ensure_multiple_of=PATCH,
    )
    return inputs["pixel_values"]


def build_model(dtype, channels_last=False, compile_model=False):
    m = AutoModelForDepthEstimation.from_pretrained(MODEL_PATH, torch_dtype=dtype)
    m = m.to(DEVICE).eval()
    if channels_last:
        m = m.to(memory_format=torch.channels_last)
    if compile_model:
        m = torch.compile(m, mode="reduce-overhead")
    return m


@torch.inference_mode()
def run_variant(name, dtype=torch.float32, size=518,
                channels_last=False, compile_model=False):
    """Returns (avg_ms, fps, depth_maps) or None on failure."""
    try:
        model = build_model(dtype, channels_last, compile_model)
    except Exception as e:
        print(f"  [{name}] BUILD FAILED: {type(e).__name__}: {e}")
        return None

    px = preprocess(IMAGES, size).to(DEVICE, dtype=dtype)
    if channels_last:
        px = px.to(memory_format=torch.channels_last)

    try:
        # warmup (compile/cudnn autotune happens here)
        for _ in range(WARMUP):
            _ = model(pixel_values=px)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(ITERS):
            out = model(pixel_values=px)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  [{name}] RUN FAILED: {type(e).__name__}: {e}")
        del model; torch.cuda.empty_cache()
        return None

    n = ITERS * len(IMAGES)
    avg_ms = elapsed / n * 1000.0
    fps = n / elapsed

    # capture depth maps (upsampled to a common size for quality comparison)
    depth = out.predicted_depth.float()                       # (B, h, w)
    depth = F.interpolate(depth.unsqueeze(1), size=(384, 384),
                          mode="bilinear", align_corners=False).squeeze(1)
    maps = depth.cpu().numpy()

    print(f"  [{name:22s}] {avg_ms:6.1f} ms/img   {fps:6.1f} FPS")
    del model; torch.cuda.empty_cache()
    return avg_ms, fps, maps


def norm01(a):
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    if hi <= lo:
        hi = a.max() if a.max() > lo else lo + 1e-6
    return np.clip((a - lo) / (hi - lo), 0, 1)


def quality_delta(base_maps, var_maps):
    """Mean AbsRel and Pearson corr across images vs baseline."""
    absrels, corrs = [], []
    for b, v in zip(base_maps, var_maps):
        bn, vn = norm01(b), norm01(v)
        absrels.append(np.mean(np.abs(vn - bn) / (bn + 1e-3)))
        corrs.append(np.corrcoef(bn.ravel(), vn.ravel())[0, 1])
    return float(np.mean(absrels)), float(np.mean(corrs))


def main():
    rows = []  # (name, avg_ms, fps, speedup, absrel, corr)

    print("Running variants...\n")
    base = run_variant("fp32 (baseline)", dtype=torch.float32, size=518)
    base_ms, base_fps, base_maps = base
    rows.append(("fp32 (baseline)", base_ms, base_fps, 1.0, 0.0, 1.0))

    variants = [
        # FP32 resolution sweep — the real lever on a tensor-core-less Turing GPU.
        ("fp32 res-392",        dict(dtype=torch.float32, size=392)),
        ("fp32 res-294",        dict(dtype=torch.float32, size=294)),
        ("fp32 res-252",        dict(dtype=torch.float32, size=252)),
        ("fp32 res-210",        dict(dtype=torch.float32, size=210)),
        # Documented dead-ends on this hardware (see notes):
        ("fp16 (slower here)",  dict(dtype=torch.float16, size=518)),
        ("fp16 + compile",      dict(dtype=torch.float16, size=518, compile_model=True)),
    ]

    for name, kw in variants:
        r = run_variant(name, **kw)
        if r is None:
            rows.append((name, None, None, None, None, None))
            continue
        avg_ms, fps, maps = r
        absrel, corr = quality_delta(base_maps, maps)
        rows.append((name, avg_ms, fps, base_ms / avg_ms, absrel, corr))

    # ---- summary table ----
    print("\n" + "=" * 78)
    print(f"{'Variant':22s} {'ms/img':>8s} {'FPS':>7s} {'Speedup':>8s} "
          f"{'AbsRel':>8s} {'Corr':>7s}")
    print("-" * 78)
    for name, ms, fps, spd, absrel, corr in rows:
        if ms is None:
            print(f"{name:22s} {'FAILED':>8s}")
            continue
        print(f"{name:22s} {ms:8.1f} {fps:7.1f} {spd:7.2f}x "
              f"{absrel:8.3f} {corr:7.3f}")
    print("=" * 78)
    print("AbsRel: lower = closer to fp32 baseline | Corr: 1.000 = identical depth")


if __name__ == "__main__":
    main()
