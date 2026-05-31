"""Solid-color blank image that matches a reference image's size.

Replaces hardcoded `EmptyImage` widgets that bake in 1024x1024 — when input
is 1024x768 the resulting brush map has wrong aspect ratio, corrupting
ReferenceLatent conditioning downstream.

Output shape mirrors reference: same batch, H, W (3-channel RGB, no alpha)."""

from __future__ import annotations

import torch

from .debug_probe import vr_log


class VR_EmptyImageLike:
    CATEGORY = "VectorReady/util"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "make"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "reference": ("IMAGE",),
                "r": ("INT", {"default": 0, "min": 0, "max": 255}),
                "g": ("INT", {"default": 0, "min": 0, "max": 255}),
                "b": ("INT", {"default": 0, "min": 0, "max": 255}),
            }
        }

    def make(self, reference, r, g, b):
        b_, h, w, _ = reference.shape
        out = torch.zeros((b_, h, w, 3), dtype=torch.float32)
        out[..., 0] = r / 255.0
        out[..., 1] = g / 255.0
        out[..., 2] = b / 255.0
        vr_log(
            "VR_EmptyImageLike",
            f"ref_shape={tuple(reference.shape)} → out_shape={tuple(out.shape)} "
            f"color=(R={r}, G={g}, B={b})",
        )
        return (out,)
