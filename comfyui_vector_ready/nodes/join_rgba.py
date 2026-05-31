"""Correct RGBA join using OPACITY convention (alpha=1 means opaque).

ComfyUI core `JoinImageWithAlpha` uses SELECTION convention internally —
it writes `1.0 - alpha` into the alpha channel, treating MASK as 'which
pixels to make transparent'. That's the opposite of what VR pipelines
produce (where alpha=1 means foreground/opaque).

This node uses alpha as opacity, then applies a final transparent-region clamp
so decoder speckles cannot survive in fully transparent pixels."""

from __future__ import annotations

import torch

from .debug_probe import vr_log


_FINAL_TRANSPARENT_ALPHA = 0.05


class VR_JoinRGBA:
    CATEGORY = "VectorReady/util"
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("rgba",)
    FUNCTION = "join"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
            }
        }

    def join(self, image, alpha):
        # image: (B, H, W, 3 or 4), alpha: (B, H, W) — both float32 0..1
        if image.shape[-1] >= 4:
            rgb = image[..., :3]
        else:
            rgb = image[..., :3]
        a = alpha
        if a.dim() == 3 and a.shape[0] != rgb.shape[0]:
            a = a.expand(rgb.shape[0], -1, -1)
        if a.shape[1:] != rgb.shape[1:3]:
            raise ValueError(
                f"VR_JoinRGBA size mismatch: image HxW={tuple(rgb.shape[1:3])} "
                f"alpha HxW={tuple(a.shape[1:])}"
            )
        a = a.clamp(0.0, 1.0)
        transparent = a < _FINAL_TRANSPARENT_ALPHA
        cleared = int(transparent.sum().item())
        total = int(transparent.numel())
        a = torch.where(transparent, torch.zeros_like(a), a)
        rgb = torch.where(transparent.unsqueeze(-1), torch.zeros_like(rgb), rgb)
        rgba = torch.cat([rgb, a.unsqueeze(-1)], dim=-1)
        vr_log(
            "VR_JoinRGBA",
            f"image={tuple(image.shape)} alpha={tuple(alpha.shape)} → rgba={tuple(rgba.shape)} "
            f"alpha_mean={float(a.mean()):.4f} transparent_clamped={cleared}/{total} "
            f"({cleared / max(total, 1):.1%}) (this is opacity: 1=opaque)",
        )
        return (rgba,)
