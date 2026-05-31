"""Restore crisp source details only where edge structure agrees.

This is a conservative A-path detail recovery pass: it copies high-frequency
source RGB back into the processed image only near edges that also exist in the
processed layer. Extra source edges that do not match the layer are exposed as a
diagnostic mismatch mask instead of being blindly restored.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import (
    np_to_torch_image,
    np_to_torch_mask,
    to_uint8,
    torch_image_to_np,
    torch_mask_to_np,
)
from .debug_probe import _stats, vr_log


def _kernel(radius: int) -> np.ndarray:
    return np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)


class VR_EdgeConsistencyRestore:
    CATEGORY = "VectorReady/edge"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK")
    RETURN_NAMES = ("restored_image", "restore_mask", "mismatch_edges")
    FUNCTION = "restore"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "processed_image": ("IMAGE",),
                "alpha": ("MASK",),
                "source_low": ("INT", {"default": 45, "min": 0, "max": 255}),
                "source_high": ("INT", {"default": 140, "min": 0, "max": 255}),
                "processed_low": ("INT", {"default": 35, "min": 0, "max": 255}),
                "processed_high": ("INT", {"default": 120, "min": 0, "max": 255}),
                "match_dilate": ("INT", {"default": 3, "min": 0, "max": 12}),
                "restore_dilate": ("INT", {"default": 1, "min": 0, "max": 8}),
                "restore_amount": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.05}),
                "alpha_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    def restore(
        self,
        source_image,
        processed_image,
        alpha,
        source_low,
        source_high,
        processed_low,
        processed_high,
        match_dilate,
        restore_dilate,
        restore_amount,
        alpha_threshold,
    ):
        source = torch_image_to_np(source_image)
        processed = torch_image_to_np(processed_image)
        masks = torch_mask_to_np(alpha)
        batch = processed.shape[0]

        out = np.empty_like(processed)
        restore_masks = np.zeros(processed.shape[:3], dtype=np.float32)
        mismatch_masks = np.zeros(processed.shape[:3], dtype=np.float32)

        for i in range(batch):
            src = source[i if source.shape[0] > i else 0]
            proc = processed[i]
            fg = masks[i if masks.shape[0] > i else 0] >= float(alpha_threshold)

            src_u8 = to_uint8(src * fg[..., None])
            proc_u8 = to_uint8(proc * fg[..., None])
            src_gray = cv2.cvtColor(src_u8, cv2.COLOR_RGB2GRAY)
            proc_gray = cv2.cvtColor(proc_u8, cv2.COLOR_RGB2GRAY)

            src_edges = cv2.Canny(src_gray, int(source_low), int(source_high))
            proc_edges = cv2.Canny(proc_gray, int(processed_low), int(processed_high))
            src_edges = ((src_edges > 0) & fg).astype(np.uint8)
            proc_edges = ((proc_edges > 0) & fg).astype(np.uint8)

            if int(match_dilate) > 0:
                support = cv2.dilate(proc_edges, _kernel(int(match_dilate))) > 0
            else:
                support = proc_edges > 0

            restore_mask = (src_edges > 0) & support & fg
            mismatch_mask = (src_edges > 0) & (~support) & fg

            if int(restore_dilate) > 0:
                restore_mask = cv2.dilate(
                    restore_mask.astype(np.uint8), _kernel(int(restore_dilate))
                ) > 0

            # Feather by one pixel so restored strokes do not form jagged stamps.
            restore_f = cv2.GaussianBlur(
                restore_mask.astype(np.float32), (3, 3), 0
            )[..., None]
            restore_f *= float(restore_amount)

            out[i] = proc * (1.0 - restore_f) + src * restore_f
            restore_masks[i] = restore_mask.astype(np.float32)
            mismatch_masks[i] = mismatch_mask.astype(np.float32)

        restored = np_to_torch_image(out)
        restore_mask_t = np_to_torch_mask(restore_masks)
        mismatch_t = np_to_torch_mask(mismatch_masks)
        vr_log("VR_EdgeConsistencyRestore restored", _stats(restored))
        vr_log("VR_EdgeConsistencyRestore restore_mask", _stats(restore_mask_t))
        vr_log("VR_EdgeConsistencyRestore mismatch_edges", _stats(mismatch_t))
        return (restored, restore_mask_t, mismatch_t)
