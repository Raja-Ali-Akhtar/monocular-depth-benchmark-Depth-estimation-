"""
Export + INT8-quantize Depth Anything V2-S at multiple low resolutions for the Pi.
Produces onnx/pi/da2s_<size>.onnx and onnx/pi/da2s_<size>_int8.onnx for each size.
Sizes must be multiples of 14 (DINOv2 patch size).
"""
import os
import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForDepthEstimation
from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantType, QuantFormat

HERE = os.path.dirname(__file__)
MODEL = os.path.join(HERE, "models", "depth_anything_v2_small")
OUT = os.path.join(HERE, "onnx", "pi")
os.makedirs(OUT, exist_ok=True)

SIZES = [252, 210, 168, 126, 112]      # 18,15,12,9,8 patches
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)

import glob
import cv2
IMGS = sorted(glob.glob(os.path.join(HERE, "input", "*.jpg")))


def pre(path, size):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_CUBIC)
    x = rgb.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


class Reader(CalibrationDataReader):
    def __init__(self, size):
        arrs = []
        for p in IMGS:
            a = pre(p, size)
            arrs.append(a)
            arrs.append(a[:, :, :, ::-1].copy())      # hflip for a few more samples
        self.it = iter([{"pixel_values": a} for a in arrs])

    def get_next(self):
        return next(self.it, None)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, pixel_values):
        return self.m(pixel_values=pixel_values).predicted_depth


def main():
    model = AutoModelForDepthEstimation.from_pretrained(MODEL).eval()
    w = Wrap(model).eval()

    for s in SIZES:
        fp32 = os.path.join(OUT, f"da2s_{s}.onnx")
        int8 = os.path.join(OUT, f"da2s_{s}_int8.onnx")
        if not os.path.exists(fp32):
            torch.onnx.export(w, (torch.randn(1, 3, s, s),), fp32,
                              input_names=["pixel_values"], output_names=["depth"],
                              opset_version=17, do_constant_folding=True)
            print(f"  exported {os.path.basename(fp32)} ({os.path.getsize(fp32)/1e6:.0f} MB)")
        if not os.path.exists(int8):
            quantize_static(fp32, int8, calibration_data_reader=Reader(s),
                            quant_format=QuantFormat.QDQ, per_channel=False,
                            activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
            print(f"  quantized {os.path.basename(int8)} ({os.path.getsize(int8)/1e6:.0f} MB)")
    print("done")


if __name__ == "__main__":
    main()
