"""Spatial cleanup for alpha channels: median + morph open + min-area + morph close.

VAE-decoded alpha from Qwen-Image-Layered contains scattered semi-transparent
pixels (~10% of the image is mid-range alpha). Stepify alone is per-pixel and
leaves visible speckles. The pipeline runs (in strict order — do NOT reorder):

  1. median blur     — kills isolated salt-and-pepper noise
  2. morph OPEN      — erodes then dilates: disconnects sub-kernel speckles
                        from the main blob, shrinking them into standalone
                        components
  3. min-area filter — drops connected components smaller than `min_area`
                        pixels (binarized at >127). Must run AFTER open
                        (so speckles are disconnected) and BEFORE close
                        (else close re-bridges speckles into the blob as
                        spurs, defeating the filter — this was the v0.7.1
                        bug). Set min_area=0 to disable.
  4. morph CLOSE     — fills tiny holes inside the surviving blob(s)."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_mask, to_float, to_uint8, torch_mask_to_np


def _drop_small_components(frame_u8: np.ndarray, min_area: int) -> np.ndarray:
    """Zero connected components in `frame_u8` whose pixel area < min_area.

    Components are found on the binarized mask (>127) with 8-connectivity, but
    the original grayscale values are preserved for components that survive."""
    binary = (frame_u8 > 127).astype(np.uint8)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n_labels <= 1:
        return frame_u8
    keep_mask = np.zeros(n_labels, dtype=bool)
    keep_mask[0] = True  # background label
    keep_mask[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    keep_px = keep_mask[labels]
    return np.where(keep_px, frame_u8, np.uint8(0))


class VR_AlphaCleanup:
    CATEGORY = "VectorReady/alpha"
    RETURN_TYPES = ("MASK",)
    FUNCTION = "clean"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha": ("MASK",),
                "median_ksize": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2}),
                "morph_ksize": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2}),
                "min_area": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 50}),
            }
        }

    def clean(self, alpha, median_ksize, morph_ksize, min_area=0):
        arr = torch_mask_to_np(alpha)
        out = np.empty_like(arr)
        k = max(1, int(morph_ksize))
        kernel = np.ones((k, k), np.uint8)
        ma = max(0, int(min_area))
        for i in range(arr.shape[0]):
            frame = to_uint8(arr[i])
            mk = max(1, int(median_ksize))
            if mk % 2 == 0:
                mk += 1
            if mk >= 3:
                frame = cv2.medianBlur(frame, mk)
            if k >= 2:
                frame = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
            if ma > 0:
                frame = _drop_small_components(frame, ma)
            if k >= 2:
                frame = cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel)
            out[i] = to_float(frame)
        return (np_to_torch_mask(out),)
