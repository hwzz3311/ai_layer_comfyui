"""Resolve target masks from SAM and LocateAnything fallbacks."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_image, np_to_torch_mask, torch_image_to_np, torch_mask_to_np, to_uint8
from .debug_probe import _stats, vr_log


def _expand_batch(arr: np.ndarray, batch: int) -> np.ndarray:
    if arr.shape[0] == batch:
        return arr
    return np.repeat(arr[:1], batch, axis=0)


def _largest_component(mask: np.ndarray) -> tuple[np.ndarray, int, int]:
    binary = (mask > 0.5).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=np.float32), 0, 0
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest = int(areas.argmax()) + 1
    return (labels == largest).astype(np.float32), int(areas.max()), int(n - 1)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a > 0.5
    bb = b > 0.5
    union = np.logical_or(aa, bb).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(aa, bb).sum() / union)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask.astype(np.float32)
    k = radius * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    return cv2.dilate((mask > 0.5).astype(np.uint8), kernel).astype(np.float32)


def _preview(frame: np.ndarray, resolved: np.ndarray, sam: np.ndarray, fallback: np.ndarray, used_fallback: bool) -> np.ndarray:
    img = to_uint8(frame).copy()
    overlay = img.copy()
    overlay[fallback > 0.5] = (80, 80, 255)
    overlay[sam > 0.5] = (80, 255, 80)
    overlay[resolved > 0.5] = (255, 80, 80) if used_fallback else (80, 255, 80)
    return (cv2.addWeighted(overlay, 0.45, img, 0.55, 0).astype(np.float32) / 255.0)


class VR_TargetMaskResolver:
    CATEGORY = "VectorReady/mask"
    RETURN_TYPES = ("MASK", "IMAGE", "BOOLEAN", "BOOLEAN")
    RETURN_NAMES = ("resolved_mask", "quality_preview", "sam_usable", "fallback_used")
    FUNCTION = "resolve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sam_mask": ("MASK",),
                "fallback_mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 1.0, "step": 0.01}),
                "min_area_ratio": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0005}),
                "max_area_ratio": ("FLOAT", {"default": 0.90, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_iou_with_fallback": ("FLOAT", {"default": 0.02, "min": 0.0, "max": 1.0, "step": 0.01}),
                "fallback_dilate_px": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "keep_largest_component": ("BOOLEAN", {"default": True}),
            }
        }

    def resolve(
        self,
        image,
        sam_mask,
        fallback_mask,
        threshold,
        min_area_ratio,
        max_area_ratio,
        min_iou_with_fallback,
        fallback_dilate_px,
        keep_largest_component,
    ):
        frames = torch_image_to_np(image)
        sam = torch_mask_to_np(sam_mask)
        fallback = torch_mask_to_np(fallback_mask)
        batch = max(frames.shape[0], sam.shape[0], fallback.shape[0])
        frames = _expand_batch(frames, batch)
        sam = _expand_batch(sam, batch)
        fallback = _expand_batch(fallback, batch)

        resolved = np.zeros_like(sam, dtype=np.float32)
        previews = np.zeros_like(frames, dtype=np.float32)
        usable_flags = []
        fallback_flags = []

        for i in range(batch):
            s = (sam[i] >= float(threshold)).astype(np.float32)
            f = (fallback[i] >= float(threshold)).astype(np.float32)
            f = _dilate(f, int(fallback_dilate_px))
            total = s.size
            area = int(s.sum())
            area_ratio = float(area / total) if total else 0.0
            iou = _iou(s, f)
            fallback_area = int(f.sum())

            sam_usable = (
                area > 0
                and area_ratio >= float(min_area_ratio)
                and area_ratio <= float(max_area_ratio)
                and (fallback_area == 0 or iou >= float(min_iou_with_fallback))
            )

            if sam_usable:
                out = np.minimum(s, f) if fallback_area > 0 else s
                if out.sum() == 0:
                    out = s
                used_fallback = False
            else:
                out = f
                used_fallback = True

            if bool(keep_largest_component) and out.sum() > 0:
                out, largest_area, component_count = _largest_component(out)
            else:
                largest_area, component_count = int(out.sum()), 0

            resolved[i] = out
            usable_flags.append(bool(sam_usable))
            fallback_flags.append(bool(used_fallback))
            previews[i] = _preview(frames[i], out, s, f, bool(used_fallback))
            vr_log(
                "VR_TargetMaskResolver frame",
                (
                    f"i={i} sam_area={area} sam_ratio={area_ratio:.6f} "
                    f"fallback_area={fallback_area} iou={iou:.4f} "
                    f"sam_usable={sam_usable} fallback_used={used_fallback} "
                    f"largest_area={largest_area} components={component_count}"
                ),
            )

        resolved_t = np_to_torch_mask(resolved)
        preview_t = np_to_torch_image(previews)
        vr_log("VR_TargetMaskResolver resolved", _stats(resolved_t))
        return (
            resolved_t,
            preview_t,
            bool(all(usable_flags) if usable_flags else False),
            bool(any(fallback_flags)),
        )
