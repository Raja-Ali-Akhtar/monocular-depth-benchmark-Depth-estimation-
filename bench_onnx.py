"""
Benchmark the ONNX model across execution providers (TensorRT FP16, CUDA, CPU)
and verify output correctness vs PyTorch. Uses fixed-size ONNX per resolution.

Registers the pip-installed TensorRT / CUDA lib dirs on the Windows DLL search
path so ORT's TensorRT execution provider can load (fixes "error 126").
"""
import os, glob, time, site
import numpy as np
from PIL import Image


def register_nvidia_dll_dirs():
    """Add pip-wheel nvidia lib dirs (tensorrt_libs, cudnn, cublas, ...) to the
    Windows DLL search path before onnxruntime is imported."""
    if not hasattr(os, "add_dll_directory"):
        return []
    added = []
    roots = site.getsitepackages() + [site.getusersitepackages()]
    wanted = ("tensorrt_libs", "tensorrt", "nvidia")
    for root in roots:
        if not os.path.isdir(root):
            continue
        for name in os.listdir(root):
            p = os.path.join(root, name)
            if not os.path.isdir(p):
                continue
            if name in ("tensorrt_libs", "tensorrt"):
                os.add_dll_directory(p); added.append(p)
            elif name == "nvidia":  # nvidia/*/bin or lib
                for sub in glob.glob(os.path.join(p, "*", "bin")) + glob.glob(os.path.join(p, "*", "lib")):
                    os.add_dll_directory(sub); added.append(sub)
    return added


_added = register_nvidia_dll_dirs()
# Also prepend to PATH so ORT's native LoadLibrary for the TRT provider finds nvinfer.
if _added:
    os.environ["PATH"] = os.pathsep.join(_added + [os.environ.get("PATH", "")])
print(f"Registered {len(_added)} nvidia DLL dir(s)")

import onnxruntime as ort
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
ONNX_DIR = os.path.join(HERE, "onnx")
TRT_CACHE = os.path.join(ONNX_DIR, "trt_cache")
os.makedirs(TRT_CACHE, exist_ok=True)

SIZES = [518, 294]
WARMUP, ITERS = 8, 30

proc = AutoImageProcessor.from_pretrained(MODEL)
images = [Image.open(p).convert("RGB")
          for p in sorted(glob.glob(os.path.join(HERE, "input", "*")))]


def onnx_path(size):
    return os.path.join(ONNX_DIR, f"depth_anything_v2_small_{size}.onnx")


def batch(size):
    return proc(images, return_tensors="np", size={"height": size, "width": size},
                keep_aspect_ratio=False, ensure_multiple_of=14)["pixel_values"].astype(np.float32)


def torch_reference(size):
    m = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    px = torch.from_numpy(batch(size)).to("cuda")
    with torch.inference_mode():
        d = m(pixel_values=px).predicted_depth.float().cpu().numpy()
    del m; torch.cuda.empty_cache()
    return d


def torch_bench(size, ref):
    """PyTorch eager fp32, batch=1 per image — apples-to-apples with ORT."""
    m = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    px = torch.from_numpy(batch(size)).to("cuda")
    per = [px[i:i+1] for i in range(px.shape[0])]
    with torch.inference_mode():
        for _ in range(WARMUP):
            _ = m(pixel_values=per[0])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(ITERS):
            for p in per:
                out = m(pixel_values=p).predicted_depth
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    del m; torch.cuda.empty_cache()
    n = ITERS * len(images)
    ms, fps = elapsed / n * 1000, n / elapsed
    print(f"  [{'PyTorch-fp32':13s} @ {size}] {ms:6.1f} ms/img  {fps:6.1f} FPS  corr=1.000")
    return "PyTorch-fp32", size, ms, fps, 1.0


def corr(a, b):
    return float(np.mean([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(a, b)]))


def ep_for(size):
    return {
        "TensorRT-FP16": [("TensorrtExecutionProvider", {
            "trt_fp16_enable": True,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE,
            "trt_timing_cache_enable": True,
        })],
        "CUDA": [("CUDAExecutionProvider", {})],
    }


def bench_ep(name, providers, size, ref):
    try:
        sess = ort.InferenceSession(
            onnx_path(size),
            providers=[p[0] for p in providers],
            provider_options=[p[1] for p in providers],
        )
        actual = sess.get_providers()[0]
        if actual != providers[0][0]:
            print(f"  [{name:13s} @ {size}] provider did not load (got {actual}) -- skipping")
            return None
    except Exception as e:
        print(f"  [{name:13s} @ {size}] session FAILED: {type(e).__name__}: {str(e)[:140]}")
        return None

    # fixed-size ONNX has batch=1, so run one image at a time (realistic latency)
    px = batch(size)                       # (N,3,size,size)
    feeds = [{"pixel_values": px[i:i+1]} for i in range(px.shape[0])]
    try:
        for _ in range(WARMUP):            # TRT builds/caches the engine here (slow first time)
            _ = sess.run(None, feeds[0])[0]
        outs = None
        t0 = time.perf_counter()
        for _ in range(ITERS):
            outs = [sess.run(None, f)[0][0] for f in feeds]
        elapsed = time.perf_counter() - t0
    except Exception as e:
        print(f"  [{name:13s} @ {size}] run FAILED: {type(e).__name__}: {str(e)[:140]}")
        return None

    n = ITERS * len(images)
    ms, fps = elapsed / n * 1000, n / elapsed
    c = corr(ref, np.stack(outs))
    print(f"  [{name:13s} @ {size}] {ms:6.1f} ms/img  {fps:6.1f} FPS  corr={c:.3f}")
    return name, size, ms, fps, c


def main():
    print(f"ORT {ort.__version__}  providers: {ort.get_available_providers()}\n")
    rows = []
    for size in SIZES:
        if not os.path.exists(onnx_path(size)):
            print(f"missing {onnx_path(size)} -- run export_onnx.py first"); continue
        print(f"--- size {size} ---")
        ref = torch_reference(size)
        rows.append(torch_bench(size, ref))          # PyTorch eager baseline
        for name, prov in ep_for(size).items():
            r = bench_ep(name, prov, size, ref)
            if r:
                rows.append(r)
        print()

    print("=" * 72)
    print(f"{'Provider':16s} {'Size':>5s} {'ms/img':>8s} {'FPS':>8s} {'corr':>7s}")
    print("-" * 72)
    for name, size, ms, fps, c in rows:
        print(f"{name:16s} {size:>5d} {ms:8.1f} {fps:8.1f} {c:7.3f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
