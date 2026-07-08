"""
Map the TensorRT-FP16 speed/quality Pareto front across input resolutions.

For each resolution we build/cached a TRT FP16 engine, measure per-image FPS,
and measure quality as correlation vs the PyTorch fp32 @518 baseline (all maps
upsampled to 518 for a fair comparison). Produces output/trt_pareto.png.
"""
import os, glob, time, site
import numpy as np
from PIL import Image


def _reg():
    if not hasattr(os, "add_dll_directory"):
        return []
    added = []
    for root in site.getsitepackages() + [site.getusersitepackages()]:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if os.path.isdir(p) and name in ("tensorrt_libs", "tensorrt"):
                os.add_dll_directory(p); added.append(p)
    return added


_a = _reg()
if _a:
    os.environ["PATH"] = os.pathsep.join(_a + [os.environ.get("PATH", "")])

import onnxruntime as ort
import torch, torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
ONNX_DIR = os.path.join(HERE, "onnx")
TRT_CACHE = os.path.join(ONNX_DIR, "trt_cache")
RES = [518, 392, 294, 252, 210]
BASE = 518
WARMUP, ITERS = 8, 30

proc = AutoImageProcessor.from_pretrained(MODEL)
images = [Image.open(p).convert("RGB")
          for p in sorted(glob.glob(os.path.join(HERE, "input", "*")))]


def pre(img, size):
    return proc(img, return_tensors="np", size={"height": size, "width": size},
                keep_aspect_ratio=False, ensure_multiple_of=14)["pixel_values"].astype(np.float32)


def up(dmap):
    """Upsample a HxW depth map to BASExBASE for fair comparison."""
    t = torch.from_numpy(np.asarray(dmap)).float()[None, None]
    return F.interpolate(t, size=(BASE, BASE), mode="bilinear", align_corners=False)[0, 0].numpy()


def baseline():
    m = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    outs = []
    with torch.inference_mode():
        for img in images:
            px = torch.from_numpy(pre(img, BASE)).to("cuda")
            outs.append(up(m(pixel_values=px).predicted_depth.float().cpu().numpy()[0]))
    del m; torch.cuda.empty_cache()
    return outs


def corr(a, b):
    return float(np.mean([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(a, b)]))


def trt_opts():
    return [("TensorrtExecutionProvider", {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": TRT_CACHE,
        "trt_timing_cache_enable": True,
    })]


def bench(size, ref):
    path = os.path.join(ONNX_DIR, f"depth_anything_v2_small_{size}.onnx")
    if not os.path.exists(path):
        print(f"  missing {path} -- run export_onnx.py"); return None
    prov = trt_opts()
    sess = ort.InferenceSession(path, providers=[p[0] for p in prov],
                                provider_options=[p[1] for p in prov])
    if sess.get_providers()[0] != "TensorrtExecutionProvider":
        print(f"  [{size}] TRT did not load"); return None
    feeds = [{"pixel_values": pre(img, size)} for img in images]
    for _ in range(WARMUP):
        _ = sess.run(None, feeds[0])
    outs = None
    t0 = time.perf_counter()
    for _ in range(ITERS):
        outs = [sess.run(None, f)[0][0] for f in feeds]
    dt = time.perf_counter() - t0
    n = ITERS * len(images)
    ms, fps = dt / n * 1000, n / dt
    c = corr(ref, [up(o) for o in outs])
    print(f"  [TRT-FP16 @ {size:3d}] {ms:6.1f} ms  {fps:6.1f} FPS  corr_vs_518={c:.3f}")
    return size, ms, fps, c


def main():
    print("PyTorch fp32 @518 baseline...")
    ref = baseline()
    rows = []
    for s in RES:
        r = bench(s, ref)
        if r:
            rows.append(r)

    print("\n" + "=" * 58)
    print(f"{'Res':>5s} {'ms/img':>8s} {'FPS':>8s} {'corr vs 518':>12s}")
    print("-" * 58)
    for s, ms, fps, c in rows:
        print(f"{s:>5d} {ms:8.1f} {fps:8.1f} {c:12.3f}")
    print("=" * 58)

    # Pareto plot: FPS (x) vs quality corr (y)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    fps = [r[2] for r in rows]; cc = [r[3] for r in rows]; res = [r[0] for r in rows]
    ax.plot(fps, cc, "-o", color="#ff7043", lw=2, ms=9)
    for s, x, y in zip(res, fps, cc):
        ax.annotate(f"{s}px", (x, y), textcoords="offset points", xytext=(8, 6),
                    fontsize=11, fontweight="bold")
    ax.set_xlabel("Speed (FPS, per image, TensorRT-FP16)", fontsize=12)
    ax.set_ylabel("Quality (correlation vs 518px baseline)", fontsize=12)
    ax.set_title("Depth Anything V2-S — TensorRT-FP16 speed/quality Pareto (GTX 1660 Ti)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = os.path.join(HERE, "output", "trt_pareto.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
