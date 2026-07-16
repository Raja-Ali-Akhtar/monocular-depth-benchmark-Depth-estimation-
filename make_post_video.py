"""
Reformat the wide 3-panel depth comparison into a LinkedIn-ready vertical video
(1080x1350) with a persistent '4.6x faster' overlay, plus a GIF export.

Reads output/depth_comparison.mp4 (1920x360 = three 640x360 panels) and re-stacks
the panels vertically. No model inference — pure compositing, so it's fast.
"""
import os
import numpy as np
import cv2
import imageio

HERE = os.path.dirname(__file__)
SRC = os.path.join(HERE, "output", "depth_comparison.mp4")
OUT_MP4 = os.path.join(HERE, "output", "post_video_vertical.mp4")
OUT_GIF = os.path.join(HERE, "output", "post_video.gif")

W = 1080
H_HEAD, H_TILE, H_FOOT = 130, 380, 80
H = H_HEAD + 3 * H_TILE + H_FOOT          # 1350
PANEL_W, PANEL_N = 640, 3                 # source: 3 x 640-wide panels

BG = (23, 17, 14)          # dark (BGR of #0e1117-ish)
FG = (245, 246, 245)
MUTED = (170, 164, 154)
ACCENT = (84, 180, 255)    # BGR of #ffb454

TILES = [
    ("Input",                    None,       None),
    ("PyTorch  @518",            "23 FPS",   MUTED),
    ("TensorRT-FP16  @294",      "106 FPS",  ACCENT),
]


def crop_to(tile, target_w, target_h):
    h, w = tile.shape[:2]
    a = target_w / target_h
    new_h = int(w / a)
    if new_h <= h:
        y0 = (h - new_h) // 2
        tile = tile[y0:y0 + new_h, :]
    else:
        new_w = int(h * a)
        x0 = (w - new_w) // 2
        tile = tile[:, x0:x0 + new_w]
    return cv2.resize(tile, (target_w, target_h))


def draw_text(img, text, org, scale, color, thick=2, shadow=True):
    if shadow:
        cv2.putText(img, text, (org[0] + 2, org[1] + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def compose(frame):
    canvas = np.full((H, W, 3), BG, np.uint8)
    panels = [frame[:, i * PANEL_W:(i + 1) * PANEL_W] for i in range(PANEL_N)]

    # header
    draw_text(canvas, "Depth Anything V2  -  4.6x faster inference",
              (40, 62), 1.15, ACCENT, 3)
    draw_text(canvas, "same model, same GPU (GTX 1660 Ti)  -  visually identical depth",
              (42, 104), 0.62, MUTED, 1, shadow=False)

    y = H_HEAD
    for (label, fps, fcol), panel in zip(TILES, panels):
        tile = crop_to(panel, W, H_TILE)
        canvas[y:y + H_TILE, :] = tile
        # label pill
        draw_text(canvas, label, (26, y + 44), 0.9, FG, 2)
        if fps:
            (tw, _), _ = cv2.getTextSize(fps, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
            draw_text(canvas, fps, (W - tw - 30, y + 50), 1.1, fcol, 3)
        y += H_TILE

    # footer
    draw_text(canvas, "TensorRT FP16 @294px  =  4.6x the speed, same depth",
              (40, H - 30), 0.72, ACCENT, 2)
    return canvas


def main():
    cap = cv2.VideoCapture(SRC)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(OUT_MP4, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    gif_frames = []
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out = compose(frame)
        writer.write(out)
        # GIF: subsample to ~12 fps, downscale, cap length
        if i % 2 == 0 and len(gif_frames) < 120:
            small = cv2.resize(out, (W // 2, H // 2))
            gif_frames.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        i += 1

    cap.release(); writer.release()
    print(f"saved {OUT_MP4}  ({W}x{H}, {i} frames @ {fps:.0f} fps)")

    imageio.mimsave(OUT_GIF, gif_frames, fps=12, loop=0)
    print(f"saved {OUT_GIF}  ({len(gif_frames)} frames, {os.path.getsize(OUT_GIF)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
