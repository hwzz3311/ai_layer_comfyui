"""Model-ready matting bridge for layer reconstruction.

This node defines the stable contract for A-path matting:

original image + Qwen layer prior + broad candidate mask -> matte signals.

v0.9.0 keeps the existing OpenCV implementation behind the bridge so the
workflow remains runnable. A learned backend (BiRefNet/RMBG/ViTMatte/etc.) can
replace the internals later without changing the preset pipelines.
"""

from __future__ import annotations

import torch

from .debug_probe import _stats, vr_log
from .layer_matting_refine import VR_LayerMattingRefine


BACKEND_CHOICES = ["opencv_fallback", "external_matte"]


class VR_LayerMattingModelBridge:
    CATEGORY = "VectorReady/matting"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = (
        "matte_rgb",
        "matte_alpha",
        "visible_alpha",
        "unknown_region",
        "matte_confidence",
        "detail_mask",
        "line_mask",
    )
    FUNCTION = "matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "original_image": ("IMAGE",),
                "qwen_image": ("IMAGE",),
                "candidate_mask": ("MASK",),
                "backend": (BACKEND_CHOICES, {"default": "opencv_fallback"}),
            },
            "optional": {
                "trimap": ("IMAGE",),
                "external_matte_alpha": ("MASK",),
                "external_confidence": ("MASK",),
            }
        }

    def matte(
        self,
        original_image,
        qwen_image,
        candidate_mask,
        backend,
        trimap=None,
        external_matte_alpha=None,
        external_confidence=None,
    ):
        if trimap is not None:
            vr_log("VR_LayerMattingModelBridge trimap", _stats(trimap))

        if backend == "external_matte" and external_matte_alpha is not None:
            vr_log("VR_LayerMattingModelBridge backend", "external_matte")
            matte_rgb = original_image
            matte_alpha = torch.minimum(external_matte_alpha, candidate_mask)
            visible_alpha = matte_alpha
            if external_confidence is None:
                matte_confidence = matte_alpha
            else:
                matte_confidence = torch.minimum(external_confidence, candidate_mask)
            unknown_region = torch.clamp(candidate_mask - matte_confidence, 0.0, 1.0)
            detail_mask = torch.zeros_like(candidate_mask)
            line_mask = torch.zeros_like(candidate_mask)
        else:
            if backend == "external_matte":
                vr_log(
                    "VR_LayerMattingModelBridge backend",
                    "external_matte requested without external alpha; using opencv_fallback",
                )
            else:
                vr_log("VR_LayerMattingModelBridge backend", backend)

            (
                matte_rgb,
                matte_alpha,
                matte_confidence,
                hole_mask,
                detail_mask,
                line_mask,
            ) = VR_LayerMattingRefine().refine(
                qwen_image,
                candidate_mask,
                original_image,
                0.05,
                3,
                22,
                3,
                8,
                1.0,
            )

            visible_alpha = torch.minimum(matte_alpha, matte_confidence)
            unknown_region = hole_mask

        vr_log("VR_LayerMattingModelBridge matte_rgb", _stats(matte_rgb))
        vr_log("VR_LayerMattingModelBridge matte_alpha", _stats(matte_alpha))
        vr_log("VR_LayerMattingModelBridge visible_alpha", _stats(visible_alpha))
        vr_log("VR_LayerMattingModelBridge unknown_region", _stats(unknown_region))
        vr_log("VR_LayerMattingModelBridge matte_confidence", _stats(matte_confidence))
        vr_log("VR_LayerMattingModelBridge detail_mask", _stats(detail_mask))
        vr_log("VR_LayerMattingModelBridge line_mask", _stats(line_mask))

        return (
            matte_rgb,
            matte_alpha,
            visible_alpha,
            unknown_region,
            matte_confidence,
            detail_mask,
            line_mask,
        )
