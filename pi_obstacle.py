#!/usr/bin/env python3
"""
Obstacle proximity alert on Raspberry Pi 5 — Depth Anything V2-S (ONNX, ARM CPU).

Pipeline: frame -> depth (INT8 ONNX, 294px) -> proximity score in a path-ahead ROI
          -> ALERT when something is much closer than the rest of the scene.

NOTE ON UNITS: Depth Anything outputs *relative inverse depth*, not metres. So the
alert is a relative proximity score, not a metric distance. `--calib K` optionally
maps inverse depth to approximate metres (metres ~= K / inv_depth) if you calibrate
K once against a known distance.

Usage:
  python3 pi_obstacle.py --source 0                      # USB webcam
  python3 pi_obstacle.py --source clip.mp4 --save out.mp4
  python3 pi_obstacle.py --source test.jpg
"""
import argparse
import os
import time

import cv2
import numpy as np
import onnxruntime as ort

SIZE = 294
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

# path-ahead ROI as fractions of the frame (x0, y0, x1, y1): centre, lower half
ROI = (0.30, 0.45, 0.70, 0.95)


def preprocess(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def proximity(depth, thresh):
    """depth: HxW inverse depth (big = near). Returns (score, alert, roi_mask_frac).

    score = how near the closest stuff in the ROI is, normalised against the whole
    frame's depth range -> 0 (far as anything in view) .. 1 (nearest thing in view).
    """
    h, w = depth.shape
    x0, y0, x1, y1 = ROI
    roi = depth[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]

    lo, hi = np.percentile(depth, 5), np.percentile(depth, 95)
    rng = max(hi - lo, 1e-6)

    near = np.percentile(roi, 90)                 # robust "closest" in the ROI
    score = float(np.clip((near - lo) / rng, 0, 1))

    # fraction of ROI that is "near" (above threshold) -> how big the obstacle is
    roi_near = float(((roi - lo) / rng > thresh).mean())
    return score, score > thresh, roi_near


def colorize(depth, w, h):
    lo, hi = np.percentile(depth, 2), np.percentile(depth, 98)
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1)
    col = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(col, (w, h), interpolation=cv2.INTER_LINEAR)


def annotate(frame, depth_vis, score, alert, roi_near, fps, calib):
    h, w = frame.shape[:2]
    x0, y0, x1, y1 = ROI
    p0, p1 = (int(x0 * w), int(y0 * h)), (int(x1 * w), int(y1 * h))

    col = (0, 0, 255) if alert else (0, 200, 0)
    for img in (frame, depth_vis):
        cv2.rectangle(img, p0, p1, col, 2)

    txt = f"proximity {score:.2f}"
    if calib:
        txt += f"  (~{calib / max(score, 1e-3):.1f} m)"
    cv2.putText(frame, txt, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"{fps:.1f} FPS", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 220, 255), 2)

    if alert:
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        msg = "! OBSTACLE CLOSE !"
        (tw, _), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 3)
        cv2.rectangle(frame, (w // 2 - tw // 2 - 12, h - 58), (w // 2 + tw // 2 + 12, h - 18), (0, 0, 255), -1)
        cv2.putText(frame, msg, (w // 2 - tw // 2, h - 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 3)
    return np.hstack([frame, depth_vis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="camera index, video file, or image")
    ap.add_argument("--model", default="depth_anything_v2_small_294_int8.onnx")
    ap.add_argument("--thresh", type=float, default=0.55, help="alert threshold on proximity score")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--calib", type=float, default=0.0, help="metres ~= calib / score (optional)")
    ap.add_argument("--save", default=None, help="write annotated output (mp4 or png)")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--show", action="store_true", help="display window (needs a desktop)")
    args = ap.parse_args()

    so = ort.SessionOptions()
    so.intra_op_num_threads = args.threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(args.model, sess_options=so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    print(f"model {os.path.basename(args.model)} | threads {args.threads} | thresh {args.thresh}")

    src = args.source
    is_image = isinstance(src, str) and src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))

    if is_image:
        frames = [cv2.imread(src)]
        if frames[0] is None:
            raise SystemExit(f"cannot read {src}")
        cap = None
    else:
        cap = cv2.VideoCapture(int(src) if src.isdigit() else src)
        if not cap.isOpened():
            raise SystemExit(f"cannot open source {src}")
        frames = None

    writer = None
    n, alerts, t_sum = 0, 0, 0.0
    fps_s = 0.0

    while True:
        if frames is not None:
            if n >= len(frames):
                break
            frame = frames[n]
        else:
            ok, frame = cap.read()
            if not ok:
                break
        if args.max_frames and n >= args.max_frames:
            break

        t0 = time.perf_counter()
        depth = np.squeeze(sess.run(None, {inp: preprocess(frame)})[0])
        dt = time.perf_counter() - t0
        t_sum += dt
        fps_s = 1.0 / dt if n == 0 else 0.8 * fps_s + 0.2 * (1.0 / dt)

        score, alert, roi_near = proximity(depth, args.thresh)
        alerts += int(alert)

        h, w = frame.shape[:2]
        out = annotate(frame.copy(), colorize(depth, w, h), score, alert, roi_near, fps_s, args.calib)

        if args.save:
            if args.save.lower().endswith((".png", ".jpg")):
                cv2.imwrite(args.save, out)
            else:
                if writer is None:
                    src_fps = cap.get(cv2.CAP_PROP_FPS) if cap else 10
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(src_fps, 1), (out.shape[1], out.shape[0]))
                writer.write(out)
        if args.show:
            cv2.imshow("depth obstacle alert", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        n += 1
        if n % 10 == 0:
            print(f"  frame {n}  score {score:.2f}  alert {alert}  {1/dt:.2f} FPS")

    if cap:
        cap.release()
    if writer:
        writer.release()

    if n:
        print(f"\nframes {n} | mean {1000*t_sum/n:.0f} ms | {n/t_sum:.2f} FPS | alerts {alerts}/{n}")
    if args.save:
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
