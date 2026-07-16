#!/usr/bin/env python3
"""
Resolution/precision sweep on the Pi: how fast can Depth Anything V2-S actually go?

Benchmarks every ONNX in the given dir (input size auto-detected from the graph),
reports latency/FPS, and measures depth quality vs a reference model so we see the
quality cliff alongside the speed gain.

Usage on Pi:
  python3 pi_sweep.py --dir . --image landscape.jpg --runs 8
"""
import argparse
import glob
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
REF_KEY = "294"          # reference model for quality comparison
CMP = 256                # common size for comparing depth maps


def pre(bgr, size):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def norm(d):
    d = cv2.resize(d.astype(np.float32), (CMP, CMP))
    lo, hi = np.percentile(d, 2), np.percentile(d, 98)
    return np.clip((d - lo) / (hi - lo + 1e-6), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".")
    ap.add_argument("--image", required=True)
    ap.add_argument("--runs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")

    models = sorted(glob.glob(os.path.join(args.dir, "*.onnx")))
    if not models:
        raise SystemExit("no .onnx found")

    rows, ref = [], None
    for mp in models:
        so = ort.SessionOptions()
        so.intra_op_num_threads = args.threads
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        try:
            sess = ort.InferenceSession(mp, sess_options=so, providers=["CPUExecutionProvider"])
        except Exception as e:
            print(f"{os.path.basename(mp)}: LOAD FAILED {str(e)[:60]}")
            continue

        i = sess.get_inputs()[0]
        shp = i.shape
        size = int(shp[2]) if isinstance(shp[2], int) else 0
        if not size:
            print(f"{os.path.basename(mp)}: dynamic input, skipping")
            continue

        x = pre(bgr, size)
        for _ in range(args.warmup):
            sess.run(None, {i.name: x})
        ts = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            out = sess.run(None, {i.name: x})[0]
            ts.append(time.perf_counter() - t0)
        ts = np.array(ts)
        depth = np.squeeze(out)

        name = os.path.basename(mp).replace(".onnx", "")
        is_ref = (REF_KEY in name) and ("int8" not in name)
        if is_ref:
            ref = norm(depth)

        rows.append({
            "name": name, "size": size,
            "int8": "int8" in name,
            "ms": ts.mean() * 1000, "fps": 1.0 / ts.mean(),
            "mb": os.path.getsize(mp) / 1e6,
            "depth": norm(depth),
        })
        print(f"  {name:26s} {size:4d}px  {ts.mean()*1000:7.1f} ms  {1/ts.mean():6.2f} FPS")

    print("\n" + "=" * 74)
    print(f"{'model':26s} {'px':>4s} {'MB':>5s} {'ms':>8s} {'FPS':>7s} {'corr@294':>9s}")
    print("-" * 74)
    for r in sorted(rows, key=lambda r: -r["ms"]):
        c = ""
        if ref is not None:
            c = f"{np.corrcoef(ref.ravel(), r['depth'].ravel())[0,1]:.3f}"
        print(f"{r['name']:26s} {r['size']:4d} {r['mb']:5.0f} {r['ms']:8.1f} {r['fps']:7.2f} {c:>9s}")
    print("=" * 74)
    print("target for real-time: 33.3 ms / 30 FPS")


if __name__ == "__main__":
    main()
