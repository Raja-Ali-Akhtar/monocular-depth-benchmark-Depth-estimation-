"""
The Pi quality cliff: speed vs depth-validity for Depth Anything V2-S on a Pi 5.
Shows you can hit 30 FPS -- but only where the model has already broken.
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)

# measured on Pi 5 (ONNX Runtime, 4 threads, INT8) + fp32 corr measured on desktop
# (px, fps_pi, corr_fp32)
DATA = [
    (294, 3.49, 0.985),
    (252, 5.38, 0.981),
    (210, 8.34, 0.987),
    (168, 13.56, 0.724),
    (126, 23.15, 0.406),
    (112, 29.44, -0.061),
]
VALID = 0.90          # usable-depth threshold

BG, FG, MUTED = "#0e1117", "#f5f6fa", "#9aa4b2"
GOOD, BAD, ACC = "#4caf50", "#ef5350", "#ffb454"

fps = [d[1] for d in DATA]
corr = [d[2] for d in DATA]
px = [d[0] for d in DATA]

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor(BG); ax.set_facecolor(BG)

ax.plot(fps, corr, "-", color=MUTED, lw=2, zorder=2)
# stagger label offsets so the clustered 294/252/210 points stay readable
OFF = {294: (-6, -30), 252: (2, 22), 210: (10, 20), 168: (0, 22), 126: (0, 22), 112: (0, 24)}
for f, c, p in zip(fps, corr, px):
    col = GOOD if c >= VALID else BAD
    ax.scatter([f], [c], s=280, color=col, zorder=3, edgecolor=BG, linewidth=2)
    ax.annotate(f"{p}px", (f, c), textcoords="offset points", xytext=OFF[p],
                ha="center", fontsize=13, fontweight="bold", color=FG)

# 30 FPS real-time line
ax.axvline(30, color=ACC, ls="--", lw=2.5, zorder=1)
ax.text(30.4, 0.55, "30 FPS\nreal-time", color=ACC, fontsize=13, fontweight="bold", va="center")

# usable-depth band
ax.axhspan(VALID, 1.05, color=GOOD, alpha=0.10, zorder=0)
ax.axhspan(-0.2, VALID, color=BAD, alpha=0.08, zorder=0)
ax.text(19.5, 0.975, "USABLE DEPTH", color=GOOD, fontsize=13, fontweight="bold")
ax.text(2.0, 0.22, "MODEL HAS BROKEN\n(depth is noise)", color=BAD, fontsize=13, fontweight="bold")

ax.set_xlabel("Speed on Raspberry Pi 5 (FPS, INT8, 4 threads)", fontsize=13, color=FG)
ax.set_ylabel("Depth validity  (correlation vs 294px)", fontsize=13, color=FG)
ax.set_title("Depth Anything V2-S on a Raspberry Pi 5:\nyou can have 30 FPS, or you can have depth",
             fontsize=16, fontweight="bold", color=ACC, pad=16)
ax.tick_params(colors=MUTED)
ax.set_xlim(0, 34); ax.set_ylim(-0.2, 1.08)
ax.grid(alpha=0.12)
for s in ax.spines.values():
    s.set_color(MUTED)

ax.annotate("usable ceiling\n210px - 8.3 FPS", (8.34, 0.987), textcoords="offset points",
            xytext=(38, -46), fontsize=12, fontweight="bold", color=GOOD,
            arrowprops=dict(arrowstyle="->", color=GOOD, lw=2))

fig.tight_layout()
out = os.path.join(HERE, "output", "pi_quality_cliff.png")
fig.savefig(out, dpi=130, facecolor=BG)
print("saved", out)
