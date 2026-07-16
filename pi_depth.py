#!/usr/bin/env python3
"""
Depth Anything V2-S on Raspberry Pi 5 (ONNX Runtime, ARM CPU).

Deps: onnxruntime, numpy, opencv-python-headless   (no torch, no transformers)
Preprocessing replicates the desktop DPTImageProcessor exactly:
  square resize to 294 (bicubic) -> /255 -> ImageNet mean/std -> CHW -> NCHW

Usage:
  python3 pi_depth.py --image test.jpg
  python3 pi_depth.py --image test.jpg --model depth_anything_v2_small_294_int8.onnx
  python3 pi_depth.py --image test.jpg --runs 20 --threads 4
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 294                                    # must match the fixed-size ONNX export
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(bgr):
    """BGR uint8 HxWx3 -> NCHW float32 (1,3,294,294)."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])   # (1,3,H,W)


def colorize(depth, out_w, out_h):
    """HxW float depth (inverse depth: big = near) -> BGR inferno at original size."""
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    d8 = (d * 255).astype(np.uint8)
    col = cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO)
    return cv2.resize(col, (out_w, out_h), interpolation=cv2.INTER_LINEAR)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--model", default="depth_anything_v2_small_294.onnx")
    ap.add_argument("--runs", type=int, default=10, help="timed iterations")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--threads", type=int, default=4, help="Pi 5 has 4 cores")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read image: {args.image}")
    h, w = bgr.shape[:2]

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    x = preprocess(bgr)

    for _ in range(args.warmup):
        sess.run(None, {inp: x})

    ts = []
    for _ in range(args.runs):
        t0 = time.perf_counter()
        out = sess.run(None, {inp: x})[0]
        ts.append(time.perf_counter() - t0)

    depth = np.squeeze(out)
    ts = np.array(ts)
    size_mb = os.path.getsize(args.model) / 1e6

    print(f"model      : {os.path.basename(args.model)}  ({size_mb:.0f} MB)")
    print(f"providers  : {sess.get_providers()}")
    print(f"threads    : {args.threads}   input: {SIZE}x{SIZE}   image: {w}x{h}")
    print(f"latency    : mean {ts.mean()*1000:7.1f} ms   min {ts.min()*1000:7.1f} ms   "
          f"max {ts.max()*1000:7.1f} ms")
    print(f"throughput : {1.0/ts.mean():.2f} FPS")
    print(f"depth      : shape {depth.shape}  range [{depth.min():.2f}, {depth.max():.2f}]")

    out_path = args.out or (os.path.splitext(os.path.basename(args.image))[0] + "_depth.png")
    cv2.imwrite(out_path, colorize(depth, w, h))
    print(f"saved      : {out_path}")


if __name__ == "__main__":
    main()
