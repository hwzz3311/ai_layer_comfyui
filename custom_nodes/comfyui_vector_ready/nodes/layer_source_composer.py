"""Layer source composer for ownership-first RGBA reconstruction.

This node does not decide what the layer is. It routes pixel sources after the
layer prior has already been estimated:

- visible/trusted region -> original RGB
- explicit completion region -> Qwen RGB
- unsupported candidate region -> transparent / low-confidence diagnostic

The goal is to keep SAM or other broad masks from turning unsupported areas
into real pixels while still allowing Qwen Layered to own occluded completion
regions when the workflow provides such a mask.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import (
    np_to_torch_image,
    np_to_torch_mask,
    split_rgba,
    torch_image_to_np,
    torch_mask_to_np,
)
from .debug_probe import _stats, vr_log


def _expand_batch(arr: np.ndarray, batch: int) -> np.ndarray:
    if arr.shape[0] == batch:
        return arr
    return np.repeat(arr[:1], batch, axis=0)


def _soften(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    return cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)


class VR_LayerSourceComposer:
    CATEGORY = "VectorReady/layer"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "rgb",
        "alpha",
        "original_region",
        "qwen_region",
        "transparent_region",
        "low_confidence",
    )
    FUNCTION = "compose"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_image": ("IMAGE",),
                "original_image": ("IMAGE",),
                "alpha": ("MASK",),
                "support_mask": ("MASK",),
                "candidate_mask": ("MASK",),
                "blend_radius": ("INT", {"default": 2, "min": 0, "max": 24}),
            },
            "optional": {
                "completion_mask": ("MASK",),
            },
        }

    def compose(
        self,
        qwen_image,
        original_image,
        alpha,
        support_mask,
        candidate_mask,
        blend_radius,
        completion_mask=None,
    ):
        qwen_rgb, _ = split_rgba(qwen_image)
        original_rgb = torch_image_to_np(original_image)
        alpha_np = torch_mask_to_np(alpha)
        support_np = torch_mask_to_np(support_mask)
        candidate_np = torch_mask_to_np(candidate_mask)
        if completion_mask is None:
            completion_np = np.zeros_like(candidate_np, dtype=np.float32)
        else:
            completion_np = torch_mask_to_np(completion_mask)

        batch = max(
            qwen_rgb.shape[0],
            original_rgb.shape[0],
            alpha_np.shape[0],
            support_np.shape[0],
            candidate_np.shape[0],
            completion_np.shape[0],
        )
        qwen_rgb = _expand_batch(qwen_rgb, batch)
        original_rgb = _expand_batch(original_rgb, batch)
        alpha_np = _expand_batch(alpha_np, batch)
        support_np = _expand_batch(support_np, batch)
        candidate_np = _expand_batch(candidate_np, batch)
        completion_np = _expand_batch(completion_np, batch)

        out_rgb = np.zeros_like(qwen_rgb, dtype=np.float32)
        out_alpha = np.zeros_like(alpha_np, dtype=np.float32)
        original_region = np.zeros_like(alpha_np, dtype=np.float32)
        qwen_region = np.zeros_like(alpha_np, dtype=np.float32)
        transparent_region = np.zeros_like(alpha_np, dtype=np.float32)
        low_confidence = np.zeros_like(alpha_np, dtype=np.float32)

        for i in range(batch):
            candidate = np.clip(candidate_np[i], 0.0, 1.0)
            support = np.clip(support_np[i], 0.0, 1.0) * candidate
            completion = np.clip(completion_np[i], 0.0, 1.0) * candidate

            # Completion is explicit: when present it is the only region where
            # Qwen RGB is allowed to be a primary color source.
            qwen_w = _soften(completion, int(blend_radius)) * candidate
            orig_w = _soften(support * (1.0 - completion), int(blend_radius)) * candidate
            total_w = np.clip(orig_w + qwen_w, 0.0, 1.0)

            routed_alpha = np.clip(alpha_np[i], 0.0, 1.0) * total_w
            rgb = original_rgb[i] * orig_w[..., None] + qwen_rgb[i] * qwen_w[..., None]
            denom = np.maximum(total_w[..., None], 1e-6)
            rgb = np.where(total_w[..., None] > 0.0, rgb / denom, 0.0)
            rgb *= (routed_alpha[..., None] >= 0.05).astype(np.float32)

            out_rgb[i] = np.clip(rgb, 0.0, 1.0)
            out_alpha[i] = np.clip(routed_alpha, 0.0, 1.0)
            original_region[i] = np.clip(orig_w, 0.0, 1.0)
            qwen_region[i] = np.clip(qwen_w, 0.0, 1.0)
            transparent_region[i] = np.clip(candidate * (1.0 - total_w), 0.0, 1.0)
            low_confidence[i] = np.clip(candidate * (1.0 - support) * (1.0 - completion), 0.0, 1.0)

        rgb_t = np_to_torch_image(out_rgb)
        alpha_t = np_to_torch_mask(out_alpha)
        original_t = np_to_torch_mask(original_region)
        qwen_t = np_to_torch_mask(qwen_region)
        transparent_t = np_to_torch_mask(transparent_region)
        low_t = np_to_torch_mask(low_confidence)
        vr_log("VR_LayerSourceComposer rgb", _stats(rgb_t))
        vr_log("VR_LayerSourceComposer alpha", _stats(alpha_t))
        vr_log("VR_LayerSourceComposer original_region", _stats(original_t))
        vr_log("VR_LayerSourceComposer qwen_region", _stats(qwen_t))
        vr_log("VR_LayerSourceComposer transparent_region", _stats(transparent_t))
        vr_log("VR_LayerSourceComposer low_confidence", _stats(low_t))
        return (rgb_t, alpha_t, original_t, qwen_t, transparent_t, low_t)
