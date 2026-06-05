"""Build target-aware trimaps from Qwen layer prior and a broad candidate mask.

The trimap is a constraint artifact, not a final matte:

- sure foreground: conservative core supported by Qwen native alpha
- sure background: outside an expanded candidate region
- unknown: candidate/boundary area where a matting model should decide alpha

This prepares the workflow for trimap-guided models such as ViTMatte while
remaining useful for constraining generic foreground matting outputs.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_image, np_to_torch_mask, split_rgba, torch_mask_to_np
from .debug_probe import _stats, vr_log


def _kernel(radius: int) -> np.ndarray:
    return np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)


def _expand_batch(arr: np.ndarray, batch: int) -> np.ndarray:
    if arr.shape[0] == batch:
        return arr
    return np.repeat(arr[:1], batch, axis=0)


class VR_TargetTrimapBuilder:
    CATEGORY = "VectorReady/matting"
    RETURN_TYPES = ("MASK", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("sure_foreground", "sure_background", "unknown_region", "trimap")
    FUNCTION = "build"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_image": ("IMAGE",),
                "candidate_mask": ("MASK",),
                "native_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "foreground_erode": ("INT", {"default": 2, "min": 0, "max": 32}),
                "candidate_dilate": ("INT", {"default": 8, "min": 0, "max": 80}),
                "unknown_dilate": ("INT", {"default": 8, "min": 0, "max": 80}),
            }
        }

    def build(
        self,
        qwen_image,
        candidate_mask,
        native_threshold,
        foreground_erode,
        candidate_dilate,
        unknown_dilate,
    ):
        _, native_alpha = split_rgba(qwen_image)
        candidate = torch_mask_to_np(candidate_mask)
        if native_alpha is None:
            native_alpha = candidate

        batch = max(native_alpha.shape[0], candidate.shape[0])
        native_alpha = _expand_batch(native_alpha, batch)
        candidate = _expand_batch(candidate, batch)
        h, w = candidate.shape[1:3]

        sure_fg = np.zeros((batch, h, w), dtype=np.float32)
        sure_bg = np.zeros((batch, h, w), dtype=np.float32)
        unknown = np.zeros((batch, h, w), dtype=np.float32)
        trimap = np.zeros((batch, h, w, 3), dtype=np.float32)

        for i in range(batch):
            cand = candidate[i] > 0.05
            native = (native_alpha[i] > float(native_threshold)) & cand

            if int(foreground_erode) > 0:
                fg = cv2.erode(native.astype(np.uint8), _kernel(int(foreground_erode))) > 0
            else:
                fg = native

            if int(candidate_dilate) > 0:
                cand_outer = cv2.dilate(cand.astype(np.uint8), _kernel(int(candidate_dilate))) > 0
            else:
                cand_outer = cand

            if int(unknown_dilate) > 0:
                fg_outer = cv2.dilate(fg.astype(np.uint8), _kernel(int(unknown_dilate))) > 0
            else:
                fg_outer = fg

            bg = ~cand_outer
            unk = (cand_outer & (~fg)) | (cand & (~fg_outer))
            unk = unk & (~bg)

            sure_fg[i] = fg.astype(np.float32)
            sure_bg[i] = bg.astype(np.float32)
            unknown[i] = unk.astype(np.float32)
            t = np.zeros((h, w), dtype=np.float32)
            t[unk] = 0.5
            t[fg] = 1.0
            trimap[i] = np.repeat(t[..., None], 3, axis=-1)

        sure_fg_t = np_to_torch_mask(sure_fg)
        sure_bg_t = np_to_torch_mask(sure_bg)
        unknown_t = np_to_torch_mask(unknown)
        trimap_t = np_to_torch_image(trimap)
        vr_log("VR_TargetTrimapBuilder sure_foreground", _stats(sure_fg_t))
        vr_log("VR_TargetTrimapBuilder sure_background", _stats(sure_bg_t))
        vr_log("VR_TargetTrimapBuilder unknown_region", _stats(unknown_t))
        vr_log("VR_TargetTrimapBuilder trimap", _stats(trimap_t))
        return (sure_fg_t, sure_bg_t, unknown_t, trimap_t)
