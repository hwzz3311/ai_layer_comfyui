"""Composite preset nodes wiring the atomic VectorReady ops into A/B archetypes.

- VR_PipelineLight  → A path (foreground extraction; color already trustworthy
                       from original RGB×alpha; only edge + alpha cleanup needed)
- VR_PipelineStrong → B path (background reconstruction containing Qwen-generated
                       RGB in inpainted regions; needs color quantization +
                       region merge + edge sharpening + alpha cleanup)
"""

from __future__ import annotations

import torch

from ..nodes._utils import split_rgba
from ..nodes.alpha_cleanup import VR_AlphaCleanup
from ..nodes.alpha_edge_refine import VR_AlphaEdgeRefine
from ..nodes.alpha_stepify import VR_AlphaStepify
from ..nodes.bilateral import VR_Bilateral
from ..nodes.canny_edge import VR_CannyEdge
from ..nodes.debug_probe import _stats, _stats_rgb_channels, vr_log
from ..nodes.edge_aware_merge import VR_EdgeAwareMerge
from ..nodes.kmeans_quantize import VR_KMeansQuantize
from ..nodes.lab_convert import VR_LABConvert
from ..nodes.layer_matting_model_bridge import VR_LayerMattingModelBridge
from ..nodes.layer_source_composer import VR_LayerSourceComposer
from ..nodes.roi_unsharp import VR_ROIUnsharpMask
from ..nodes.target_trimap_builder import VR_TargetTrimapBuilder

# Pixels with alpha below this are pure transparent-region decoder noise; their
# RGB gets zeroed. Pixels above keep their original color so edges aren't darkened.
_HARD_TRANSPARENT_ALPHA = 0.05


ALPHA_SOURCE_CHOICES = ["auto", "native", "mask_socket"]
MATTING_BACKEND_CHOICES = ["opencv_fallback", "external_matte"]


def _resolve_alpha(image: torch.Tensor, alpha_input: torch.Tensor,
                   source: str = "auto") -> torch.Tensor:
    """Pick the alpha source per the `source` arg.

    - "auto"        : prefer native RGBA alpha if present, else fall back to
                       the MASK socket. Default (2026-05-27 invariant).
    - "native"      : force native RGBA alpha; raise if image is RGB only.
    - "mask_socket" : ignore native alpha entirely, use the wired MASK socket.
                       Needed when the upstream model's alpha doesn't represent
                       the object silhouette — e.g., Qwen-Image-Layered's
                       native alpha marks "where white was painted" rather
                       than "where the layer's content is", which kills line-
                       art detail (eyes / whiskers / outlines) when used as a
                       mask. In that case wire an external silhouette
                       (SAM3 / RMBG / ViTMatte) into the MASK socket and set
                       this to "mask_socket"."""
    if source == "mask_socket":
        return alpha_input
    _, native_alpha = split_rgba(image)
    if native_alpha is not None:
        return torch.from_numpy(native_alpha)
    if source == "native":
        raise ValueError("alpha_source='native' but image has no alpha channel")
    return alpha_input


def _clean_transparent_rgb(image: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """Zero RGB only where alpha is essentially zero — preserves edge color.

    Replaces the old full premultiply (rgb *= alpha), which darkened every
    semi-transparent edge pixel. Decoder garbage lives in alpha≈0 regions, so
    a hard threshold is enough to remove it without touching real edges."""
    rgb_np, _ = split_rgba(image)
    rgb = torch.from_numpy(rgb_np)
    a = alpha
    if a.dim() == 3 and rgb.shape[0] != a.shape[0]:
        a = a.expand(rgb.shape[0], -1, -1)
    keep = (a >= _HARD_TRANSPARENT_ALPHA).to(rgb.dtype).unsqueeze(-1)
    return rgb * keep


class VR_PipelineLight:
    """A-path archetype: minimal cleanup, trust the source RGB."""

    CATEGORY = "VectorReady/preset"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "palette_k": ("INT", {"default": 0, "min": 0, "max": 32}),
                "alpha_steps": ([2, 3], {"default": 2}),
                "alpha_min_area": ("INT", {"default": 400, "min": 0, "max": 100000, "step": 50}),
                "alpha_source": (ALPHA_SOURCE_CHOICES, {"default": "auto"}),
                "matting_backend": (MATTING_BACKEND_CHOICES, {"default": "opencv_fallback"}),
            },
            "optional": {
                "source_image": ("IMAGE",),
                "external_matte_alpha": ("MASK",),
                "external_confidence": ("MASK",),
            }
        }

    def run(self, image, alpha, palette_k, alpha_steps, alpha_min_area=400,
            alpha_source="auto", matting_backend="opencv_fallback",
            source_image=None, external_matte_alpha=None, external_confidence=None):
        vr_log("Light INPUT image", _stats(image))
        vr_log("Light INPUT alpha (MASK socket)", _stats(alpha))
        if source_image is not None:
            vr_log("Light INPUT source_image", _stats(source_image))
        if external_matte_alpha is not None:
            vr_log("Light INPUT external_matte_alpha", _stats(external_matte_alpha))
        if external_confidence is not None:
            vr_log("Light INPUT external_confidence", _stats(external_confidence))
        alpha = _resolve_alpha(image, alpha, alpha_source)
        vr_log(f"Light resolved alpha (source={alpha_source})", _stats(alpha))

        (alpha,) = VR_AlphaCleanup().clean(alpha, 3, 5, int(alpha_min_area))
        vr_log("Light [0.3] alpha_cleanup", _stats(alpha))
        candidate_alpha = alpha

        detail_source = source_image if source_image is not None else image
        detail_label = "original/source_image" if source_image is not None else "qwen_rgb"
        vr_log("Light [0.5] detail_source", detail_label)
        (sure_fg, sure_bg, trimap_unknown, trimap) = VR_TargetTrimapBuilder().build(
            image, alpha, 0.05, 2, 8, 8
        )
        vr_log("Light [0.45] sure_foreground", _stats(sure_fg))
        vr_log("Light [0.45] sure_background", _stats(sure_bg))
        vr_log("Light [0.45] trimap_unknown", _stats(trimap_unknown))
        vr_log("Light [0.45] trimap", _stats(trimap))
        (rgb, alpha, visible_alpha, unknown_region, confidence, detail_mask, line_mask) = (
            VR_LayerMattingModelBridge().matte(
                detail_source,
                image,
                alpha,
                matting_backend,
                trimap,
                external_matte_alpha,
                external_confidence,
            )
        )
        vr_log("Light [0.5] matte_rgb", _stats(rgb))
        vr_log("Light [0.5] matte_alpha", _stats(alpha))
        vr_log("Light [0.5] visible_alpha", _stats(visible_alpha))
        vr_log("Light [0.5] unknown_region", _stats(unknown_region))
        vr_log("Light [0.5] matte_confidence", _stats(confidence))
        vr_log("Light [0.5] detail_mask", _stats(detail_mask))
        vr_log("Light [0.5] line_mask", _stats(line_mask))
        vr_log("Light [0.5] rgb channels", _stats_rgb_channels(rgb, alpha))

        (rgb, alpha, original_region, qwen_region, transparent_region, low_confidence) = (
            VR_LayerSourceComposer().compose(
                image, detail_source, alpha, confidence, candidate_alpha, 2, None
            )
        )
        vr_log("Light [0.7] source_compose_rgb", _stats(rgb))
        vr_log("Light [0.7] source_compose_alpha", _stats(alpha))
        vr_log("Light [0.7] original_region", _stats(original_region))
        vr_log("Light [0.7] qwen_region", _stats(qwen_region))
        vr_log("Light [0.7] transparent_region", _stats(transparent_region))
        vr_log("Light [0.7] low_confidence", _stats(low_confidence))
        vr_log("Light [0.7] composed channels", _stats_rgb_channels(rgb, alpha))

        if int(palette_k) >= 2:
            (smoothed,) = VR_Bilateral().apply(rgb, 7, 40.0, 15.0)
            vr_log("Light [1] bilateral_smooth", _stats(smoothed))

            # Optional palette quantization: useful when the caller wants
            # flat-color/vector-friendly regions. A-path defaults to fidelity
            # mode (palette_k=0) because foreground RGB is usually trustworthy.
            (lab,) = VR_LABConvert().convert(smoothed, "rgb_to_lab")
            (quant_lab, k_used) = VR_KMeansQuantize().quantize(lab, int(palette_k), False, int(palette_k))
            (quantized,) = VR_LABConvert().convert(quant_lab, "lab_to_rgb")
            vr_log("Light [1.5] palette_quantize", f"K_used={k_used} {_stats(quantized)}")
        else:
            smoothed = rgb
            quantized = rgb
            vr_log("Light [1] bilateral_smooth", "skipped (fidelity mode: palette_k < 2)")
            vr_log("Light [1.5] palette_quantize", "skipped (fidelity mode: palette_k < 2)")

        (alpha, alpha_edge_roi) = VR_AlphaEdgeRefine().refine(alpha, detail_source, 2, 0.55, 8.0)
        vr_log("Light [1.8] alpha_edge_refined", _stats(alpha))
        vr_log("Light [1.8] alpha_edge_roi", _stats(alpha_edge_roi))

        (rgb_edges,) = VR_CannyEdge().detect(quantized, 60, 160, 1)
        edges = torch.maximum(rgb_edges, alpha_edge_roi)
        edges = torch.minimum(edges, original_region)
        vr_log("Light [2] canny_edges + alpha_edge_roi", _stats(edges))

        (sharpened,) = VR_ROIUnsharpMask().sharpen(quantized, edges, 0.9, 3, 2)
        vr_log("Light [3] roi_sharpened", _stats(sharpened))
        vr_log("Light [3] roi_sharpened channels", _stats_rgb_channels(sharpened, alpha))

        (alpha_clean,) = VR_AlphaStepify().stepify(alpha, alpha_steps, 0.4, 0.6)
        vr_log("Light [4] alpha_stepified", _stats(alpha_clean))
        vr_log("Light OUTPUT image", _stats(sharpened))
        vr_log("Light OUTPUT alpha", _stats(alpha_clean))
        return (sharpened, alpha_clean)


class VR_PipelineStrong:
    """B-path archetype: full vectorization-ready treatment of Qwen output."""

    CATEGORY = "VectorReady/preset"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "alpha")
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "alpha": ("MASK",),
                "max_k": ("INT", {"default": 12, "min": 2, "max": 32}),
                "delta_e": ("FLOAT", {"default": 6.0, "min": 1.0, "max": 25.0}),
                "alpha_steps": ([2, 3], {"default": 2}),
                "alpha_min_area": ("INT", {"default": 400, "min": 0, "max": 100000, "step": 50}),
                "alpha_source": (ALPHA_SOURCE_CHOICES, {"default": "auto"}),
            }
        }

    def run(self, image, alpha, max_k, delta_e, alpha_steps, alpha_min_area=400, alpha_source="auto"):
        vr_log("Strong INPUT image", _stats(image))
        vr_log("Strong INPUT alpha (MASK socket)", _stats(alpha))
        alpha = _resolve_alpha(image, alpha, alpha_source)
        vr_log(f"Strong resolved alpha (source={alpha_source})", _stats(alpha))

        (alpha,) = VR_AlphaCleanup().clean(alpha, 3, 5, int(alpha_min_area))
        vr_log("Strong [0.3] alpha_cleanup", _stats(alpha))

        rgb = _clean_transparent_rgb(image, alpha)
        vr_log("Strong [0.5] clean_transparent_rgb", _stats(rgb))
        vr_log("Strong [0.5] rgb channels", _stats_rgb_channels(rgb, alpha))

        (smoothed,) = VR_Bilateral().apply(rgb, 11, 90.0, 30.0)
        vr_log("Strong [1] bilateral_smooth", _stats(smoothed))

        (edges,) = VR_CannyEdge().detect(smoothed, 70, 180, 1)
        vr_log("Strong [2] canny_edges (on smoothed)", _stats(edges))

        (lab,) = VR_LABConvert().convert(smoothed, "rgb_to_lab")
        (quant_lab, k_used) = VR_KMeansQuantize().quantize(lab, max_k, True, 8)
        vr_log("Strong [3] kmeans_quantized (LAB)", f"K_used={k_used} {_stats(quant_lab)}")

        (merged_lab,) = VR_EdgeAwareMerge().merge(quant_lab, edges, delta_e, 0.15)
        vr_log("Strong [4] region_merged (LAB)", _stats(merged_lab))

        (rgb,) = VR_LABConvert().convert(merged_lab, "lab_to_rgb")
        vr_log("Strong [5] back_to_rgb", _stats(rgb))
        vr_log("Strong [5] back_to_rgb channels", _stats_rgb_channels(rgb, alpha))

        (sharp,) = VR_ROIUnsharpMask().sharpen(rgb, edges, 1.0, 3, 2)
        vr_log("Strong [6] roi_sharpened", _stats(sharp))
        vr_log("Strong [6] roi_sharpened channels", _stats_rgb_channels(sharp, alpha))

        (alpha_clean,) = VR_AlphaStepify().stepify(alpha, alpha_steps, 0.4, 0.6)
        vr_log("Strong [7] alpha_stepified", _stats(alpha_clean))
        vr_log("Strong OUTPUT image", _stats(sharp))
        vr_log("Strong OUTPUT alpha", _stats(alpha_clean))

        return (sharp, alpha_clean)
