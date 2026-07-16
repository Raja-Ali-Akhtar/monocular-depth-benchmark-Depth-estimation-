"""
Run Depth Anything V2-S on a video and compare speed + accuracy of two configs:
  - PyTorch fp32 @518  (baseline)
  - TensorRT-FP16 @294 (optimized)

Writes output/depth_comparison.mp4 = [ Original | Depth@518 | Depth@294-TRT ]
with a live FPS overlay on each depth panel (inference-only timing).
"""
import os, glob, site, time
import numpy as np
import cv2


def _reg_trt():
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


_a = _reg_trt()
if _a:
    os.environ["PATH"] = os.pathsep.join(_a + [os.environ.get("PATH", "")])

import torch
import onnxruntime as ort
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
ONNX_294 = os.path.join(HERE, "onnx", "depth_anything_v2_small_294.onnx")
TRT_CACHE = os.path.join(HERE, "onnx", "trt_cache")
IN_VIDEO = os.path.join(HERE, "candidates", "nat_3121327.mp4")
OUT_VIDEO = os.path.join(HERE, "output", "depth_comparison.mp4")

START_S, END_S = 1.0, 13.0           # mountain-ridgelines segment
LETTERBOX = 0.0                      # full-frame 16:9, no cinematic bars
PANEL_H, PANEL_W = 360, 640          # display size per panel (16:9)
proc = AutoImageProcessor.from_pretrained(MODEL)


def deletterbox(frame):
    h = frame.shape[0]
    c = int(h * LETTERBOX)
    return frame[c:h - c] if c > 0 else frame


def pre(frame_rgb, size):
    return proc(frame_rgb, return_tensors="np", size={"height": size, "width": size},
                keep_aspect_ratio=False, ensure_multiple_of=14)["pixel_values"].astype(np.float32)


def colorize(depth):
    """HxW float depth -> BGR inferno image at PANEL size (near=bright)."""
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    col = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return cv2.resize(col, (PANEL_W, PANEL_H))


def label(img, title, fps):
    cv2.rectangle(img, (0, 0), (PANEL_W, 34), (0, 0, 0), -1)
    cv2.putText(img, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    if fps is not None:
        txt = f"{fps:5.1f} FPS"
        cv2.putText(img, txt, (PANEL_W - 150, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    (100, 200, 255), 2, cv2.LINE_AA)
    return img


class TorchCfg:
    def __init__(self, size):
        self.size = size
        self.m = AutoModelForDepthEstimation.from_pretrained(MODEL).to("cuda").eval()

    def depth(self, frame_rgb):
        px = torch.from_numpy(pre(frame_rgb, self.size)).to("cuda")
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            d = self.m(pixel_values=px).predicted_depth
        torch.cuda.synchronize(); dt = time.perf_counter() - t0
        return d.float().cpu().numpy()[0], dt


class TRTCfg:
    def __init__(self, size, onnx_path):
        self.size = size
        prov = [("TensorrtExecutionProvider", {
            "trt_fp16_enable": True, "trt_engine_cache_enable": True,
            "trt_engine_cache_path": TRT_CACHE, "trt_timing_cache_enable": True})]
        self.sess = ort.InferenceSession(onnx_path, providers=[p[0] for p in prov],
                                         provider_options=[p[1] for p in prov])
        self.ok = self.sess.get_providers()[0] == "TensorrtExecutionProvider"

    def depth(self, frame_rgb):
        feed = {"pixel_values": pre(frame_rgb, self.size)}
        t0 = time.perf_counter()
        out = self.sess.run(None, feed)[0]
        dt = time.perf_counter() - t0
        return out[0], dt


def smooth(prev, cur, a=0.9):
    return cur if prev is None else a * prev + (1 - a) * cur


def main():
    cap = cv2.VideoCapture(IN_VIDEO)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    start_f, end_f = int(START_S * src_fps), int(END_S * src_fps)
    n = end_f - start_f
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    print(f"input: {IN_VIDEO}  segment {START_S}-{END_S}s = {n} frames @ {src_fps:.0f} fps")

    base = TorchCfg(518)
    fast = TRTCfg(294, ONNX_294)
    print(f"TensorRT loaded: {fast.ok}")
    if not fast.ok:
        print("WARNING: TRT provider not active; fast panel will still run but not on TRT")

    writer = cv2.VideoWriter(OUT_VIDEO, cv2.VideoWriter_fourcc(*"mp4v"),
                             src_fps, (PANEL_W * 3, PANEL_H))

    sm_base = sm_fast = None
    sum_base = sum_fast = 0.0
    cnt = 0
    while cnt < n:
        ok, frame = cap.read()
        if not ok:
            break
        frame = deletterbox(frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        d_base, t_base = base.depth(rgb)
        d_fast, t_fast = fast.depth(rgb)

        # skip first 2 frames from averages (warmup/engine load)
        if cnt >= 2:
            sum_base += t_base; sum_fast += t_fast
        sm_base = smooth(sm_base, 1.0 / t_base)
        sm_fast = smooth(sm_fast, 1.0 / t_fast)

        orig = cv2.resize(frame, (PANEL_W, PANEL_H))
        orig = label(orig, "Input", None)
        pb = label(colorize(d_base), "PyTorch @518", sm_base)
        pf = label(colorize(d_fast), "TensorRT-FP16 @294", sm_fast)

        writer.write(np.hstack([orig, pb, pf]))
        cnt += 1
        if cnt % 30 == 0:
            print(f"  {cnt}/{n} frames  (base {sm_base:.0f} / fast {sm_fast:.0f} FPS)")

    cap.release(); writer.release()
    m = max(cnt - 2, 1)
    print("\n=== average inference-only speed ===")
    print(f"PyTorch fp32 @518 : {1000*sum_base/m:6.1f} ms/frame  {m/sum_base:6.1f} FPS")
    print(f"TensorRT-FP16 @294: {1000*sum_fast/m:6.1f} ms/frame  {m/sum_fast:6.1f} FPS")
    print(f"speedup: {sum_base/sum_fast:.2f}x")
    print(f"saved {OUT_VIDEO}")


if __name__ == "__main__":
    main()
