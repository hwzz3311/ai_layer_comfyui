"""Append a brush reference latent only when the SAM mask is usable."""

from __future__ import annotations

from copy import copy

import torch

from ._utils import torch_mask_to_np
from .debug_probe import vr_log


class VR_ReferenceLatentIfMaskUsable:
    """Conditionally mirror ComfyUI's ReferenceLatent behavior.

    Qwen Layered V2 can use a second reference latent as the red/green brush.
    When SAM fails and returns an empty or nonsensical mask, passing that brush
    reference makes the target constraint worse. This node keeps the text +
    original-image conditioning intact and appends the brush latent only when
    the mask area is within a conservative usable range.
    """

    CATEGORY = "VectorReady/conditioning"
    RETURN_TYPES = ("CONDITIONING", "BOOLEAN", "IMAGE")
    RETURN_NAMES = ("conditioning", "mask_usable", "status_image")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent": ("LATENT",),
                "mask": ("MASK",),
                "threshold": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01},
                ),
                "min_area_ratio": (
                    "FLOAT",
                    {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0005},
                ),
                "max_area_ratio": (
                    "FLOAT",
                    {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
                "min_area_px": (
                    "INT",
                    {"default": 64, "min": 0, "max": 1000000, "step": 1},
                ),
            }
        }

    def _append_reference_latent(self, conditioning, latent):
        out = []
        ref = latent["samples"]
        for cond, pooled in conditioning:
            pooled = copy(pooled)
            existing = list(pooled.get("reference_latents", []))
            pooled["reference_latents"] = existing + [ref]
            out.append([cond, pooled])
        return out

    def apply(
        self,
        conditioning,
        latent,
        mask: torch.Tensor,
        threshold: float,
        min_area_ratio: float,
        max_area_ratio: float,
        min_area_px: int,
    ):
        arr = torch_mask_to_np(mask)
        fg = arr > float(threshold)
        area_px = int(fg.sum())
        total_px = int(fg.size)
        area_ratio = float(area_px / total_px) if total_px else 0.0
        usable = (
            area_px >= int(min_area_px)
            and area_ratio >= float(min_area_ratio)
            and area_ratio <= float(max_area_ratio)
        )

        verdict = "append_brush_reference" if usable else "skip_brush_reference"
        vr_log(
            "VR_ReferenceLatentIfMaskUsable",
            (
                f"{verdict} area_px={area_px} total_px={total_px} "
                f"area_ratio={area_ratio:.6f} threshold={float(threshold):.3f} "
                f"min_area_ratio={float(min_area_ratio):.6f} "
                f"max_area_ratio={float(max_area_ratio):.6f} "
                f"min_area_px={int(min_area_px)}"
            ),
        )

        status = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        if usable:
            status[..., 1] = 1.0
        else:
            status[..., 0] = 1.0

        if usable:
            return (self._append_reference_latent(conditioning, latent), True, status)
        return (conditioning, False, status)
