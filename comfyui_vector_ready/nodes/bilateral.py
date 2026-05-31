"""Edge-preserving smoothing to remove diffusion grain while keeping boundaries crisp."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_image, to_float, to_uint8, torch_image_to_np


class VR_Bilateral:
    CATEGORY = "VectorReady/filter"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "diameter": ("INT", {"default": 9, "min": 3, "max": 25, "step": 2}),
                "sigma_color": ("FLOAT", {"default": 75.0, "min": 1.0, "max": 200.0}),
                "sigma_space": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 200.0}),
            }
        }

    def apply(self, image, diameter, sigma_color, sigma_space):
        arr = torch_image_to_np(image)
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            frame = to_uint8(arr[i])
            filtered = cv2.bilateralFilter(frame, int(diameter), float(sigma_color), float(sigma_space))
            out[i] = to_float(filtered)
        return (np_to_torch_image(out),)
