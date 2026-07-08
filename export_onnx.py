"""
Export Depth Anything V2-Small to ONNX.

Produces:
  onnx/depth_anything_v2_small.onnx        (dynamic batch/H/W - for ORT CUDA/CPU)
  onnx/depth_anything_v2_small_518.onnx     (fixed 518x518 - best for TensorRT)
  onnx/depth_anything_v2_small_294.onnx     (fixed 294x294 - best for TensorRT)

Fixed-size exports avoid dynamic-shape complications in TensorRT and the
baked-upsample-size issue that affects the dynamic model at non-518 inputs.
"""
import os
import torch
from transformers import AutoModelForDepthEstimation

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
OUT_DIR = os.path.join(HERE, "onnx")
os.makedirs(OUT_DIR, exist_ok=True)

FIXED_SIZES = [518, 392, 294, 252, 210]


class Wrapper(torch.nn.Module):
    """Return just the depth tensor so ONNX has a clean single output."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model(pixel_values=pixel_values).predicted_depth


def export(wrapped, path, size, dynamic):
    dummy = torch.randn(1, 3, size, size)
    kw = {}
    if dynamic:
        kw["dynamic_axes"] = {
            "pixel_values": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "out_h", 2: "out_w"},
        }
    torch.onnx.export(
        wrapped, (dummy,), path,
        input_names=["pixel_values"], output_names=["depth"],
        opset_version=17, do_constant_folding=True, **kw,
    )
    import onnx
    onnx.checker.check_model(path)
    print(f"  {os.path.basename(path):40s} {os.path.getsize(path)/1e6:5.0f} MB  ok")


def main():
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    wrapped = Wrapper(model).eval()

    print("Exporting ONNX models...")
    export(wrapped, os.path.join(OUT_DIR, "depth_anything_v2_small.onnx"), 518, dynamic=True)
    for s in FIXED_SIZES:
        export(wrapped, os.path.join(OUT_DIR, f"depth_anything_v2_small_{s}.onnx"), s, dynamic=False)
    print("Done.")


if __name__ == "__main__":
    main()
