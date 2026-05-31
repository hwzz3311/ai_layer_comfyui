"""Subtract an inner (cutout) mask from an outer (subject) mask.

Designed for subjects whose silhouette has topological holes — e.g. card
holders with a photo-slot window, picture frames, donut shapes. The outer
mask is produced by the positive LA+SAM3 chain; the inner mask comes from
a symmetric LA+SAM3 chain driven by a "cutout query" describing the hole.

Subtraction is soft (operates on float alpha, not binary) and supports a
small dilation of the inner mask before subtracting, to avoid leaving a
hairline rim of the subject color around the cutout boundary.

When `inner` is all zeros (e.g. cutout query was empty and the negative
chain short-circuited), this node is a no-op passthrough of `outer`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_mask, torch_mask_to_np
from .debug_probe import _stats, vr_log


def _dilate(mask_np: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_np
    k = radius * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    out = np.empty_like(mask_np)
    for i in range(mask_np.shape[0]):
        binary = (mask_np[i] > 0.0).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel)
        # Preserve soft edges by taking max(original, dilated_hard) so the
        # subtracted area is at least the dilated binary, but keeps the soft
        # alpha values where the original mask already had them.
        out[i] = np.maximum(mask_np[i], dilated.astype(np.float32))
    return out


def _expand_batch(arr: np.ndarray, batch: int) -> np.ndarray:
    if arr.shape[0] == batch:
        return arr
    return np.repeat(arr[:1], batch, axis=0)


class VR_MaskSubtract:
    """final = clamp(outer - dilate(inner, inner_dilate_px), 0, 1)."""

    CATEGORY = "VectorReady/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "subtract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outer": ("MASK",),
                "inner": ("MASK",),
                # Slight expansion of the inner mask before subtraction so the
                # subject's color doesn't bleed a 1-2px rim into the cutout.
                # Set to 0 for exact subtraction.
                "inner_dilate_px": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                # Below this threshold the inner mask is treated as empty —
                # short-circuits to pure passthrough so an empty negative
                # chain (cutout query unset) doesn't pay any compute.
                "min_inner_area_ratio": (
                    "FLOAT",
                    {"default": 0.0005, "min": 0.0, "max": 1.0, "step": 0.0001},
                ),
            }
        }

    def subtract(self, outer, inner, inner_dilate_px, min_inner_area_ratio):
        outer_np = torch_mask_to_np(outer)
        inner_np = torch_mask_to_np(inner)
        batch = max(outer_np.shape[0], inner_np.shape[0])
        outer_np = _expand_batch(outer_np, batch)
        inner_np = _expand_batch(inner_np, batch)

        total = float(outer_np.shape[1] * outer_np.shape[2])
        inner_area_ratio = float((inner_np > 0.5).sum()) / max(total * batch, 1.0)

        if inner_area_ratio < float(min_inner_area_ratio):
            vr_log(
                "VR_MaskSubtract",
                f"inner_area_ratio={inner_area_ratio:.6f} < "
                f"min={float(min_inner_area_ratio):.6f} → passthrough (no subtraction)",
            )
            return (np_to_torch_mask(outer_np),)

        dilated_inner = _dilate(inner_np, int(inner_dilate_px))
        result = np.clip(outer_np - dilated_inner, 0.0, 1.0)

        result_t = np_to_torch_mask(result)
        vr_log(
            "VR_MaskSubtract",
            f"outer_mean={outer_np.mean():.4f} inner_mean={inner_np.mean():.4f} "
            f"inner_dilate_px={int(inner_dilate_px)} "
            f"inner_area_ratio={inner_area_ratio:.6f} {_stats(result_t)}",
        )
        return (result_t,)
