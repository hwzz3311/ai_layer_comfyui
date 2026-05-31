"""RGB ↔ LAB color space conversion.

LAB is more perceptually uniform than RGB; downstream quantization and
region merging use ΔE in LAB space."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_image, to_float, to_uint8, torch_image_to_np


class VR_LABConvert:
    CATEGORY = "VectorReady/color"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "direction": (["rgb_to_lab", "lab_to_rgb"], {"default": "rgb_to_lab"}),
            }
        }

    def convert(self, image, direction):
        arr = torch_image_to_np(image)  # [B,H,W,3] float 0-1
        code = cv2.COLOR_RGB2LAB if direction == "rgb_to_lab" else cv2.COLOR_LAB2RGB
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            frame_u8 = to_uint8(arr[i])
            converted = cv2.cvtColor(frame_u8, code)
            out[i] = to_float(converted)
        return (np_to_torch_image(out),)
