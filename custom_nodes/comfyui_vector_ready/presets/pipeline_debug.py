"""Debug variants — same chain as production presets, with intermediate outputs."""

from __future__ import annotations

import torch

from ..nodes._utils import split_rgba
from ..nodes.alpha_cleanup import VR_AlphaCleanup
from ..nodes.alpha_edge_refine import VR_AlphaEdgeRefine
from ..nodes.alpha_stepify import VR_AlphaStepify
from ..nodes.bilateral import VR_Bilateral
from ..nodes.canny_edge import VR_CannyEdge
from ..nodes.debug_probe import _stats, vr_log
from ..nodes.edge_aware_merge import VR_EdgeAwareMerge
from ..nodes.kmeans_quantize import VR_KMeansQuantize
from ..nodes.lab_convert import VR_LABConvert
from ..nodes.layer_matting_model_bridge import VR_LayerMattingModelBridge
from ..nodes.layer_source_composer import VR_LayerSourceComposer
from ..nodes.roi_unsharp import VR_ROIUnsharpMask
from ..nodes.target_trimap_builder import VR_TargetTrimapBuilder
from .pipeline import (
    ALPHA_SOURCE_CHOICES,
    _HARD_TRANSPARENT_ALPHA,
    MATTING_BACKEND_CHOICES,
    _clean_transparent_rgb,
    _edge_color_inpaint,
    _align_to_alpha_hw,
    _resolve_alpha,
)


def _mask_to_image(m: torch.Tensor) -> torch.Tensor:
    return m.unsqueeze(-1).expand(*m.shape, 3)


class VR_PipelineStrongDebug:
    """Mirrors VR_PipelineStrong; emits every stage for PreviewImage inspection."""

    CATEGORY = "VectorReady/debug"
    RETURN_TYPES = ("IMAGE",) * 10 + ("MASK",)
    RETURN_NAMES = (
        "input_rgb",
        "native_alpha_viz",
        "alpha_cleaned_viz",
        "bilateral_smooth",
        "canny_edges_viz",
        "kmeans_quantized",
        "region_merged",
        "roi_sharpened",
        "final_defringed",
        "edge_inpainted",
        "alpha_stepified",
    )
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
        vr_log("StrongDebug INPUT image", _stats(image))
        vr_log("StrongDebug INPUT alpha (MASK socket)", _stats(alpha))

        alpha = _resolve_alpha(image, alpha, alpha_source)
        vr_log(f"StrongDebug resolved alpha (source={alpha_source})", _stats(alpha))
        image, alpha, _, _, _ = _align_to_alpha_hw(image, alpha, label="StrongDebug")
        native_alpha_viz = _mask_to_image(alpha)

        (alpha,) = VR_AlphaCleanup().clean(alpha, 3, 5, int(alpha_min_area))
        vr_log("[0.3] alpha_cleanup", _stats(alpha))
        alpha_cleaned_viz = _mask_to_image(alpha)

        rgb = _clean_transparent_rgb(image, alpha)
        vr_log("[0.5] clean_transparent_rgb", _stats(rgb))

        (smoothed,) = VR_Bilateral().apply(rgb, 11, 90.0, 30.0)
        vr_log("[1] bilateral_smooth", _stats(smoothed))

        (edges,) = VR_CannyEdge().detect(smoothed, 70, 180, 1)
        vr_log("[2] canny_edges", _stats(edges))
        canny_viz = _mask_to_image(edges)

        (lab,) = VR_LABConvert().convert(smoothed, "rgb_to_lab")
        (quant_lab, k_used) = VR_KMeansQuantize().quantize(lab, max_k, True, 8)
        (quant_rgb,) = VR_LABConvert().convert(quant_lab, "lab_to_rgb")
        vr_log("[3] kmeans_quantized", f"K_used={k_used} {_stats(quant_rgb)}")

        (merged_lab,) = VR_EdgeAwareMerge().merge(quant_lab, edges, delta_e, 0.15)
        (merged_rgb,) = VR_LABConvert().convert(merged_lab, "lab_to_rgb")
        vr_log("[4] region_merged", _stats(merged_rgb))

        (sharp,) = VR_ROIUnsharpMask().sharpen(merged_rgb, edges, 1.0, 3, 2)
        vr_log("[5] roi_sharpened", _stats(sharp))

        (alpha_clean,) = VR_AlphaStepify().stepify(alpha, alpha_steps, 0.4, 0.6)
        vr_log("[6] alpha_stepified", _stats(alpha_clean))

        # Final defringe with the post-stepify alpha — pixels whose soft alpha
        # got collapsed to 0 still carried processed RGB here. Zero them so
        # the saved PNG matches a "clean for vector tracing" contract.
        final_defringed = _clean_transparent_rgb(sharp, alpha_clean)
        vr_log("[7] final_defringed", _stats(final_defringed))

        # Edge color inpaint: rewrite the alpha-boundary halo with nearest
        # interior color. See VR_PipelineStrong for full rationale.
        edge_inpainted = _edge_color_inpaint(final_defringed, alpha_clean, ring_px=2, radius=3)
        vr_log("[8] edge_inpainted", _stats(edge_inpainted))

        return (rgb, native_alpha_viz, alpha_cleaned_viz, smoothed,
                canny_viz, quant_rgb, merged_rgb, sharp, final_defringed,
                edge_inpainted, alpha_clean)


class VR_PipelineLightDebug:
    """Mirrors VR_PipelineLight; emits every stage for PreviewImage inspection."""

    CATEGORY = "VectorReady/debug"
    RETURN_TYPES = ("IMAGE",) * 25 + ("MASK",)
    RETURN_NAMES = (
        "input_rgb",
        "native_alpha_viz",
        "alpha_cleaned_viz",
        "sure_foreground_viz",
        "sure_background_viz",
        "trimap_unknown_viz",
        "trimap_viz",
        "matting_rgb",
        "visible_alpha_viz",
        "unknown_region_viz",
        "matte_confidence_viz",
        "source_composed",
        "original_region_viz",
        "qwen_region_viz",
        "transparent_region_viz",
        "low_confidence_viz",
        "hole_viz",
        "detail_viz",
        "line_viz",
        "bilateral_smooth",
        "palette_quantized",
        "canny_edges_viz",
        "roi_sharpened",
        "final_defringed",
        "edge_inpainted",
        "alpha_stepified",
    )
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
        vr_log("LightDebug INPUT image", _stats(image))
        vr_log("LightDebug INPUT alpha (MASK socket)", _stats(alpha))
        if source_image is not None:
            vr_log("LightDebug INPUT source_image", _stats(source_image))
        if external_matte_alpha is not None:
            vr_log("LightDebug INPUT external_matte_alpha", _stats(external_matte_alpha))
        if external_confidence is not None:
            vr_log("LightDebug INPUT external_confidence", _stats(external_confidence))

        alpha = _resolve_alpha(image, alpha, alpha_source)
        vr_log(f"LightDebug resolved alpha (source={alpha_source})", _stats(alpha))
        image, alpha, source_image, external_matte_alpha, external_confidence = _align_to_alpha_hw(
            image,
            alpha,
            source_image=source_image,
            external_matte_alpha=external_matte_alpha,
            external_confidence=external_confidence,
            label="LightDebug",
        )
        native_alpha_viz = _mask_to_image(alpha)

        (alpha,) = VR_AlphaCleanup().clean(alpha, 3, 5, int(alpha_min_area))
        vr_log("[0.3] alpha_cleanup", _stats(alpha))
        candidate_alpha = alpha
        alpha_cleaned_viz = _mask_to_image(alpha)

        detail_source = source_image if source_image is not None else image
        detail_label = "original/source_image" if source_image is not None else "qwen_rgb"
        vr_log("[0.5] detail_source", detail_label)
        (sure_fg, sure_bg, trimap_unknown, trimap) = VR_TargetTrimapBuilder().build(
            image, alpha, 0.05, 2, 8, 8
        )
        sure_fg_viz = _mask_to_image(sure_fg)
        sure_bg_viz = _mask_to_image(sure_bg)
        trimap_unknown_viz = _mask_to_image(trimap_unknown)
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
        vr_log("[0.5] matte_rgb", _stats(rgb))
        vr_log("[0.5] matte_alpha", _stats(alpha))
        matting_rgb = rgb
        visible_alpha_viz = _mask_to_image(visible_alpha)
        unknown_region_viz = _mask_to_image(unknown_region)
        matte_confidence_viz = _mask_to_image(confidence)
        hole_viz = _mask_to_image(unknown_region)
        detail_viz = _mask_to_image(detail_mask)
        line_viz = _mask_to_image(line_mask)

        (rgb, alpha, original_region, qwen_region, transparent_region, low_confidence) = (
            VR_LayerSourceComposer().compose(
                image, detail_source, alpha, confidence, candidate_alpha, 2, None
            )
        )
        vr_log("[0.7] source_compose_rgb", _stats(rgb))
        vr_log("[0.7] source_compose_alpha", _stats(alpha))
        original_viz = _mask_to_image(original_region)
        qwen_viz = _mask_to_image(qwen_region)
        transparent_viz = _mask_to_image(transparent_region)
        low_confidence_viz = _mask_to_image(low_confidence)

        if int(palette_k) >= 2:
            (smoothed,) = VR_Bilateral().apply(rgb, 7, 40.0, 15.0)
            vr_log("[1] bilateral_smooth", _stats(smoothed))

            (lab,) = VR_LABConvert().convert(smoothed, "rgb_to_lab")
            (quant_lab, k_used) = VR_KMeansQuantize().quantize(lab, int(palette_k), False, int(palette_k))
            (quantized,) = VR_LABConvert().convert(quant_lab, "lab_to_rgb")
            vr_log("[1.5] palette_quantize", f"K_used={k_used} {_stats(quantized)}")
        else:
            smoothed = rgb
            quantized = rgb
            vr_log("[1] bilateral_smooth", "skipped (fidelity mode: palette_k < 2)")
            vr_log("[1.5] palette_quantize", "skipped (fidelity mode: palette_k < 2)")

        (alpha, alpha_edge_roi) = VR_AlphaEdgeRefine().refine(alpha, detail_source, 2, 0.55, 8.0)
        vr_log("[1.8] alpha_edge_refined", _stats(alpha))
        vr_log("[1.8] alpha_edge_roi", _stats(alpha_edge_roi))

        (rgb_edges,) = VR_CannyEdge().detect(quantized, 60, 160, 1)
        edges = torch.maximum(rgb_edges, alpha_edge_roi)
        edges = torch.minimum(edges, original_region)
        vr_log("[2] canny_edges + alpha_edge_roi", _stats(edges))
        canny_viz = _mask_to_image(edges)

        (sharpened,) = VR_ROIUnsharpMask().sharpen(quantized, edges, 0.9, 3, 2)
        vr_log("[3] roi_sharpened", _stats(sharpened))

        (alpha_clean,) = VR_AlphaStepify().stepify(alpha, alpha_steps, 0.4, 0.6)
        vr_log("[4] alpha_stepified", _stats(alpha_clean))

        # Final defringe — see VR_PipelineLight for rationale. The earlier
        # composer / matting steps preserve edge color, but stepify hard-cuts
        # alpha; the RGB at "just-collapsed" pixels needs to be zeroed too.
        final_defringed = _clean_transparent_rgb(sharpened, alpha_clean)
        vr_log("[5] final_defringed", _stats(final_defringed))

        # Edge color inpaint — rewrite the post-stepify alpha-boundary halo
        # with nearest interior color via cv2.inpaint(TELEA).
        edge_inpainted = _edge_color_inpaint(final_defringed, alpha_clean, ring_px=2, radius=3)
        vr_log("[6] edge_inpainted", _stats(edge_inpainted))

        return (
            matting_rgb,
            native_alpha_viz,
            alpha_cleaned_viz,
            sure_fg_viz,
            sure_bg_viz,
            trimap_unknown_viz,
            trimap,
            matting_rgb,
            visible_alpha_viz,
            unknown_region_viz,
            matte_confidence_viz,
            rgb,
            original_viz,
            qwen_viz,
            transparent_viz,
            low_confidence_viz,
            hole_viz,
            detail_viz,
            line_viz,
            smoothed,
            quantized,
            canny_viz,
            sharpened,
            final_defringed,
            edge_inpainted,
            alpha_clean,
        )


class VR_PipelineLayeredDebug:
    """Mirrors VR_PipelineLayered; emits every stage for PreviewImage inspection."""

    CATEGORY = "VectorReady/debug"
    RETURN_TYPES = ("IMAGE",) * 6 + ("MASK",)
    RETURN_NAMES = (
        "input_rgb",
        "native_alpha_viz",
        "alpha_cleaned_viz",
        "edge_roi_viz",
        "roi_sharpened",
        "edge_inpainted",
        "alpha_out",
    )
    FUNCTION = "run"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "median_ksize": ("INT", {"default": 1, "min": 1, "max": 11, "step": 2}),
                "morph_ksize": ("INT", {"default": 1, "min": 1, "max": 11, "step": 2}),
                "alpha_min_area": ("INT", {"default": 16, "min": 0, "max": 100000, "step": 8}),
                "sharpen_strength": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
        }

    def run(self, image, median_ksize=1, morph_ksize=1, alpha_min_area=16,
            sharpen_strength=0.9):
        vr_log("LayeredDBG INPUT image", _stats(image))
        alpha = _resolve_alpha(image, None, "native")
        native_viz = _mask_to_image(alpha)

        (alpha,) = VR_AlphaCleanup().clean(
            alpha, int(median_ksize), int(morph_ksize), int(alpha_min_area)
        )
        alpha_cleaned_viz = _mask_to_image(alpha)

        rgb_np, _ = split_rgba(image)
        rgb = torch.from_numpy(rgb_np)
        input_rgb = rgb

        (rgb_edges,) = VR_CannyEdge().detect(rgb, 60, 160, 1)
        visible = (alpha >= _HARD_TRANSPARENT_ALPHA).to(rgb_edges.dtype)
        if visible.shape[0] != rgb_edges.shape[0] and visible.shape[0] == 1:
            visible = visible.expand(rgb_edges.shape[0], -1, -1)
        edges = torch.minimum(rgb_edges, visible)
        edge_roi_viz = _mask_to_image(edges)

        (sharpened,) = VR_ROIUnsharpMask().sharpen(rgb, edges, float(sharpen_strength), 3, 2)
        roi_sharpened = sharpened

        sharpened = _clean_transparent_rgb(sharpened, alpha)
        sharpened = _edge_color_inpaint(sharpened, alpha, ring_px=2, radius=3)

        return (
            input_rgb,
            native_viz,
            alpha_cleaned_viz,
            edge_roi_viz,
            roi_sharpened,
            sharpened,
            alpha,
        )
