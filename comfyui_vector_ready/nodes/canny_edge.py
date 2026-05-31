"""Canny edge detection, output as MASK (0-1).

ComfyUI core has a Canny node, but we ship a self-contained variant so the
plugin's preset chains don't depend on external packages."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_mask, to_uint8, torch_image_to_np


class VR_CannyEdge:
    CATEGORY = "VectorReady/edge"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "detect"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "low_threshold": ("INT", {"default": 60, "min": 0, "max": 255}),
                "high_threshold": ("INT", {"default": 160, "min": 0, "max": 255}),
                "dilate": ("INT", {"default": 1, "min": 0, "max": 5}),
            }
        }

    def detect(self, image, low_threshold, high_threshold, dilate):
        arr = torch_image_to_np(image)
        out = np.zeros(arr.shape[:3], dtype=np.float32)
        for i in range(arr.shape[0]):
            gray = cv2.cvtColor(to_uint8(arr[i]), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, int(low_threshold), int(high_threshold))
            if dilate > 0:
                kernel = np.ones((int(dilate) * 2 + 1,) * 2, np.uint8)
                edges = cv2.dilate(edges, kernel)
            out[i] = edges.astype(np.float32) / 255.0
        return (np_to_torch_mask(out),)
