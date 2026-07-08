"""
INT8 quantization (static QDQ) of Depth Anything V2-S at 294px, then benchmark
INT8 vs FP16 vs PyTorch on TensorRT, with a quality check vs the PyTorch baseline.

INT8 can degrade depth, so we verify correlation, not just speed.
"""
import os, glob, time, site
import numpy as np
from PIL import Image


def register_nvidia_dll_dirs():
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


_added = register_nvidia_dll_dirs()
if _added:
    os.environ["PATH"] = os.pathsep.join(_added + [os.environ.get("PATH", "")])

import onnxruntime as ort
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
ONNX_DIR = os.path.join(HERE, "onnx")
SIZE = 294
FP32_ONNX = os.path.join(ONNX_DIR, f"depth_anything_v2_small_{SIZE}.onnx")
INT8_ONNX = os.path.join(ONNX_DIR, f"depth_anything_v2_small_{SIZE}_int8.onnx")
TRT_CACHE = os.path.join(ONNX_DIR, "trt_cache")
WARMUP, ITERS = 8, 30

proc = AutoImageProcessor.from_pretrained(MODEL)
images = [Image.open(p).convert("RGB")
          for p in sorted(glob.glob(os.path.join(HERE, "input", "*")))]


def pre(img):
    return proc(img, return_tensors="np", size={"height": SIZE, "width": SIZE},
                keep_aspect_ratio=False, ensure_multiple_of=14)["pixel_values"].astype(np.float32)


# --- calibration data: input images + horizontal flips for more samples ---
class Reader(CalibrationDataReader):
    def __init__(self):
        arrs = []
        for img in images:
            arrs.append(pre(img))
            arrs.append(pre(img.transpose(Image.FLIP_LEFT_RIGHT)))
        self.it = iter([{"pixel_values": a} for a in arrs])

    def get_next(self):
        return next(self.it, None)


def torch_ref():
    m = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()
    outs = []
    with torch.inference_mode():
        for img in images:
            px = torch.from_numpy(pre(img)).to("cuda")
            outs.append(m(pixel_values=px).predicted_depth.float().cpu().numpy()[0])
    del m; torch.cuda.empty_cache()
    return outs


def corr(a, b):
    return float(np.mean([np.corrcoef(x.ravel(), y.ravel())[0, 1] for x, y in zip(a, b)]))


def bench(name, onnx_file, providers):
    try:
        sess = ort.InferenceSession(onnx_file, providers=[p[0] for p in providers],
                                    provider_options=[p[1] for p in providers])
        if sess.get_providers()[0] != providers[0][0]:
            print(f"  [{name}] provider {providers[0][0]} did not load (got {sess.get_providers()[0]})")
            return None
    except Exception as e:
        print(f"  [{name}] session FAILED: {type(e).__name__}: {str(e)[:140]}")
        return None
    feeds = [{"pixel_values": pre(img)} for img in images]
    try:
        for _ in range(WARMUP):
            _ = sess.run(None, feeds[0])
        outs = None
        t0 = time.perf_counter()
        for _ in range(ITERS):
            outs = [sess.run(None, f)[0][0] for f in feeds]
        dt = time.perf_counter() - t0
    except Exception as e:
        print(f"  [{name}] run FAILED: {type(e).__name__}: {str(e)[:140]}")
        return None
    n = ITERS * len(images)
    ms, fps = dt / n * 1000, n / dt
    return name, ms, fps, outs


def trt(int8):
    opts = {
        "trt_fp16_enable": True,
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": TRT_CACHE,
        "trt_timing_cache_enable": True,
    }
    if int8:
        opts["trt_int8_enable"] = True
    return [("TensorrtExecutionProvider", opts)]


def main():
    print(f"Quantizing {SIZE}px model to INT8 (static QDQ)...")
    if not os.path.exists(INT8_ONNX):
        quantize_static(
            FP32_ONNX, INT8_ONNX,
            calibration_data_reader=Reader(),
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QInt8,
        )
        print(f"  wrote {INT8_ONNX} ({os.path.getsize(INT8_ONNX)/1e6:.0f} MB)")
    else:
        print("  (already exists, skipping)")

    ref = torch_ref()
    print(f"\nBenchmarking at {SIZE}px (corr vs PyTorch fp32):")
    results = []
    r = bench("TensorRT-FP16", FP32_ONNX, trt(int8=False))
    if r: results.append((r[0], r[1], r[2], corr(ref, r[3])))
    r = bench("TensorRT-INT8", INT8_ONNX, trt(int8=True))
    if r: results.append((r[0], r[1], r[2], corr(ref, r[3])))

    print("\n" + "=" * 60)
    print(f"{'Variant':16s} {'ms/img':>8s} {'FPS':>8s} {'corr':>8s}")
    print("-" * 60)
    for name, ms, fps, c in results:
        print(f"{name:16s} {ms:8.1f} {fps:8.1f} {c:8.3f}")
    print("=" * 60)
    print("corr vs PyTorch fp32 @294: 1.000 = identical, lower = INT8 degraded quality")


if __name__ == "__main__":
    main()
