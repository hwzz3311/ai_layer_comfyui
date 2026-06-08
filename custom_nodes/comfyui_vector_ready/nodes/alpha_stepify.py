"""Quantize alpha to 2 or 3 discrete levels — eliminates diffusion alpha halo
and makes the output vectorizer-friendly."""

from __future__ import annotations

import numpy as np

from ._utils import np_to_torch_mask, torch_mask_to_np


class VR_AlphaStepify:
    CATEGORY = "VectorReady/alpha"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "stepify"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha": ("MASK",),
                "steps": ([2, 3], {"default": 3}),
                "low_threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "high_threshold": ("FLOAT", {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def stepify(self, alpha, steps, low_threshold, high_threshold):
        arr = torch_mask_to_np(alpha)
        out = np.zeros_like(arr)
        if int(steps) == 2:
            mid = (float(low_threshold) + float(high_threshold)) / 2.0
            out[arr >= mid] = 1.0
        else:
            out[arr >= float(high_threshold)] = 1.0
            mid_mask = (arr >= float(low_threshold)) & (arr < float(high_threshold))
            out[mid_mask] = 0.5
        return (np_to_torch_mask(out),)
