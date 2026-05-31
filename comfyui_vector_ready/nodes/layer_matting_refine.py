"""Layer-aware matting refinement for A-path foreground extraction.

This node turns three noisy hints into a PSD-like foreground layer:

- Qwen RGBA: content existence and coarse color for the requested layer
- SAM mask: broad silhouette
- Original image: high-frequency detail source

It produces a refined RGB/alpha pair plus diagnostic confidence, hole, detail,
and line masks. The implementation is intentionally model-free for v1 so the
behavior is inspectable; a learned matting backend can later replace the
guided-alpha step behind the same node contract.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import (
    np_to_torch_image,
    np_to_torch_mask,
    split_rgba,
    to_uint8,
    torch_image_to_np,
    torch_mask_to_np,
)
from .debug_probe import _stats, vr_log


def _kernel(radius: int) -> np.ndarray:
    return np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)


def _guided_filter_gray(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Single-channel guided filter, He et al. style, implemented with boxFilter."""
    guide = guide.astype(np.float32)
    src = src.astype(np.float32)
    ksize = (radius * 2 + 1, radius * 2 + 1)
    mean_i = cv2.boxFilter(guide, -1, ksize, normalize=True)
    mean_p = cv2.boxFilter(src, -1, ksize, normalize=True)
    corr_i = cv2.boxFilter(guide * guide, -1, ksize, normalize=True)
    corr_ip = cv2.boxFilter(guide * src, -1, ksize, normalize=True)
    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p
    a = cov_ip / (var_i + float(eps))
    b = mean_p - a * mean_i
    mean_a = cv2.boxFilter(a, -1, ksize, normalize=True)
    mean_b = cv2.boxFilter(b, -1, ksize, normalize=True)
    return np.clip(mean_a * guide + mean_b, 0.0, 1.0)


def _keep_components_touching(mask: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Keep only connected components that touch trusted layer content."""
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask
    keep_ids = np.unique(labels[anchor & (labels > 0)])
    if keep_ids.size == 0:
        return np.zeros_like(mask, dtype=bool)
    return np.isin(labels, keep_ids)


def _adaptive_agreement_mask(qwen_rgb: np.ndarray, guide_rgb: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return pixels where Qwen content agrees with the original visible RGB.

    A-path extraction is not allowed to trust Qwen-only hallucinated fill. The
    cutoff is estimated per image from the Qwen/original color disagreement
    distribution inside the candidate layer, so this is a data-adaptive gate
    rather than a scene-specific constant.
    """
    dist = np.mean(np.abs(qwen_rgb - guide_rgb), axis=-1).astype(np.float32)
    vals = dist[candidate]
    if vals.size < 16:
        return candidate
    vals_u8 = np.clip(vals * 255.0, 0, 255).astype(np.uint8)
    otsu, _ = cv2.threshold(vals_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cutoff = max(float(otsu) / 255.0, float(np.percentile(vals, 35)))
    return candidate & (dist <= cutoff)


class VR_LayerMattingRefine:
    CATEGORY = "VectorReady/matting"
    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("rgb", "alpha", "confidence", "hole_mask", "detail_mask", "line_mask")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qwen_image": ("IMAGE",),
                "sam_mask": ("MASK",),
                "original_image": ("IMAGE",),
                "native_threshold": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sure_fg_dilate": ("INT", {"default": 3, "min": 0, "max": 24}),
                "anchor_dilate": ("INT", {"default": 22, "min": 0, "max": 80}),
                "detail_dilate": ("INT", {"default": 3, "min": 0, "max": 16}),
                "matte_radius": ("INT", {"default": 8, "min": 1, "max": 40}),
                "detail_amount": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            }
        }

    def refine(
        self,
        qwen_image,
        sam_mask,
        original_image,
        native_threshold,
        sure_fg_dilate,
        anchor_dilate,
        detail_dilate,
        matte_radius,
        detail_amount,
    ):
        qwen_rgb_np, native_alpha_np = split_rgba(qwen_image)
        original = torch_image_to_np(original_image)
        sam = torch_mask_to_np(sam_mask)
        if native_alpha_np is None:
            native_alpha_np = sam

        batch = max(qwen_rgb_np.shape[0], original.shape[0], sam.shape[0], native_alpha_np.shape[0])
        h, w = qwen_rgb_np.shape[1:3]
        out_rgb = np.zeros((batch, h, w, 3), dtype=np.float32)
        out_alpha = np.zeros((batch, h, w), dtype=np.float32)
        confidence = np.zeros((batch, h, w), dtype=np.float32)
        holes = np.zeros((batch, h, w), dtype=np.float32)
        details = np.zeros((batch, h, w), dtype=np.float32)
        lines = np.zeros((batch, h, w), dtype=np.float32)

        for i in range(batch):
            qwen_rgb = qwen_rgb_np[i if qwen_rgb_np.shape[0] > i else 0]
            native_alpha = native_alpha_np[i if native_alpha_np.shape[0] > i else 0]
            guide_rgb = original[i if original.shape[0] > i else 0]
            sam_fg = sam[i if sam.shape[0] > i else 0] > 0.05

            native_candidate = (native_alpha > float(native_threshold)) & sam_fg
            native_core = _adaptive_agreement_mask(qwen_rgb, guide_rgb, native_candidate)
            if int(sure_fg_dilate) > 0:
                sure_fg = cv2.dilate(native_core.astype(np.uint8), _kernel(int(sure_fg_dilate))) > 0
            else:
                sure_fg = native_core
            if int(anchor_dilate) > 0:
                anchor = cv2.dilate(native_core.astype(np.uint8), _kernel(int(anchor_dilate))) > 0
                attach_anchor = cv2.dilate(
                    native_core.astype(np.uint8),
                    _kernel(max(1, int(anchor_dilate) // 3)),
                ) > 0
            else:
                anchor = native_core
                attach_anchor = native_core

            guide_u8 = to_uint8(guide_rgb)
            qwen_u8 = to_uint8(qwen_rgb * sam_fg[..., None])
            guide_gray = cv2.cvtColor(guide_u8, cv2.COLOR_RGB2GRAY)
            qwen_gray = cv2.cvtColor(qwen_u8, cv2.COLOR_RGB2GRAY)

            original_edges = cv2.Canny(guide_gray, 35, 110) > 0
            qwen_edges = cv2.Canny(qwen_gray, 30, 110) > 0
            qwen_edge_support = cv2.dilate(qwen_edges.astype(np.uint8), _kernel(4)) > 0
            detail = original_edges & qwen_edge_support & anchor & sam_fg
            detail = _keep_components_touching(detail, attach_anchor)
            if int(detail_dilate) > 0:
                detail = cv2.dilate(detail.astype(np.uint8), _kernel(int(detail_dilate))) > 0
                detail = detail & anchor & sam_fg

            # Thin strokes (whiskers, mouths, outlines, small icon glyphs) are
            # often topologically cleaner in Qwen than in the noisy original.
            # Use Qwen only as an alpha/structure prior: RGB is still routed
            # from the original later by VR_LayerSourceComposer.
            line = qwen_edges & anchor & sam_fg
            line = _keep_components_touching(line, attach_anchor)
            if int(detail_dilate) > 0:
                line_radius = max(1, int(detail_dilate) // 2)
                line = cv2.dilate(line.astype(np.uint8), _kernel(line_radius)) > 0
                line = line & anchor & sam_fg

            support = sam_fg & (sure_fg | detail | line)
            hole = sam_fg & (~support)

            trimap_alpha = np.zeros((h, w), dtype=np.float32)
            trimap_alpha[sure_fg] = 1.0
            trimap_alpha[detail] = 1.0
            trimap_alpha[line] = 1.0
            trimap_alpha[hole] = 0.0
            guide_luma = guide_gray.astype(np.float32) / 255.0
            refined_alpha = _guided_filter_gray(
                guide_luma, trimap_alpha, int(matte_radius), 1e-3
            )
            refined_alpha = refined_alpha * support.astype(np.float32)

            visible_detail = detail | line
            detail_f = cv2.GaussianBlur(visible_detail.astype(np.float32), (3, 3), 0)[..., None]
            detail_f *= float(detail_amount)
            rgb = qwen_rgb * (1.0 - detail_f) + guide_rgb * detail_f
            rgb *= refined_alpha[..., None]

            out_rgb[i] = np.clip(rgb, 0.0, 1.0)
            out_alpha[i] = np.clip(refined_alpha, 0.0, 1.0)
            confidence[i] = support.astype(np.float32)
            holes[i] = hole.astype(np.float32)
            details[i] = detail.astype(np.float32)
            lines[i] = line.astype(np.float32)

        rgb_t = np_to_torch_image(out_rgb)
        alpha_t = np_to_torch_mask(out_alpha)
        conf_t = np_to_torch_mask(confidence)
        hole_t = np_to_torch_mask(holes)
        detail_t = np_to_torch_mask(details)
        line_t = np_to_torch_mask(lines)
        vr_log("VR_LayerMattingRefine rgb", _stats(rgb_t))
        vr_log("VR_LayerMattingRefine alpha", _stats(alpha_t))
        vr_log("VR_LayerMattingRefine confidence", _stats(conf_t))
        vr_log("VR_LayerMattingRefine hole_mask", _stats(hole_t))
        vr_log("VR_LayerMattingRefine detail_mask", _stats(detail_t))
        vr_log("VR_LayerMattingRefine line_mask", _stats(line_t))
        return (rgb_t, alpha_t, conf_t, hole_t, detail_t, line_t)
