"""Diagnostic node — emits a structured JSON report describing how
"vectorization-ready" a saved RGBA layer is.

Intended placement: between VR_JoinRGBA and SaveImage. The image is passed
through unchanged so the report sits inline without rewiring. The agent
that orchestrates the workflow can read either the returned STRING or the
`vr_debug.log` line and decide whether to accept the layer or re-run with
adjusted parameters (e.g. higher palette_k, looser min_inner_area_ratio,
tighter SAM3 threshold).

Stats reported:
- canvas size, content bbox + ratio
- transparent_ratio (fraction of canvas with alpha == 0)
- unique_colors (distinct RGB triples in alpha > 0 region)
- alpha_levels (distinct alpha values — should be ≤ stepify steps + 1)
- connected_components: count + top-5 areas + largest_ratio
- flags: stringly-typed warnings agents can pattern-match
- verdict: "clean" if flags are empty else "needs_review"

No pixels touched. Adds ~10-30 ms per layer at 1024² resolution.
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from ._utils import split_rgba
from .debug_probe import vr_log


def _pack_rgb_u8(rgb_u8: np.ndarray) -> np.ndarray:
    """Pack [N,3] uint8 RGB to [N] uint32 for fast unique counting."""
    return (
        rgb_u8[:, 0].astype(np.uint32) << 16
        | rgb_u8[:, 1].astype(np.uint32) << 8
        | rgb_u8[:, 2].astype(np.uint32)
    )


class VR_VectorReadyReport:
    CATEGORY = "VectorReady/diagnostic"
    # Passthrough image so the node can be inserted inline (image in → image
    # out) without rewiring SaveImage. The report STRING is the new payload.
    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("report_json", "image")
    FUNCTION = "report"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Free-form label so multi-layer workflows can tell layers
                # apart in the log. Usually "A_foreground" / "B_background".
                "layer_label": ("STRING", {"default": "layer"}),
                # Above this color count the layer is flagged "too_many_colors"
                # — vectorizers handle <= ~32 distinct colors gracefully; more
                # than that and they posterize internally, defeating our
                # k-means quantization.
                "max_colors_clean": ("INT", {"default": 32, "min": 2, "max": 256, "step": 1}),
                # Below this content_ratio the layer is flagged "low_content"
                # — usually means SAM3 mask failed and we extracted near-empty.
                "min_content_ratio": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.001},
                ),
                # Above this small_island ratio (sum of non-largest CCs /
                # largest CC area) the layer is flagged "small_islands" —
                # usually speckle that survived alpha_cleanup.
                "small_island_warn_ratio": (
                    "FLOAT",
                    {"default": 0.05, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def report(
        self,
        image,
        layer_label,
        max_colors_clean,
        min_content_ratio,
        small_island_warn_ratio,
    ):
        rgb_np, alpha_np = split_rgba(image)

        # Report on the first frame only — VectorReady workflows are
        # single-image. If batched, the rest are skipped (would just bloat
        # the log). Cast away the batch axis here.
        rgb = rgb_np[0]
        if alpha_np is None:
            alpha = np.ones(rgb.shape[:2], dtype=np.float32)
        else:
            alpha = alpha_np[0]

        h, w = rgb.shape[:2]
        total = int(h * w)

        rgb_u8 = np.clip(rgb, 0.0, 1.0)
        rgb_u8 = (rgb_u8 * 255.0 + 0.5).astype(np.uint8)
        alpha_u8 = (np.clip(alpha, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        opaque = alpha_u8 > 0

        opaque_count = int(opaque.sum())
        if opaque_count > 0:
            packed = _pack_rgb_u8(rgb_u8[opaque])
            unique_colors = int(np.unique(packed).size)
        else:
            unique_colors = 0

        alpha_levels = int(np.unique(alpha_u8).size)

        binary = opaque.astype(np.uint8)
        n_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        # Label 0 is the background (alpha=0 region); skip it.
        areas = sorted(
            (int(stats[i, cv2.CC_STAT_AREA]) for i in range(1, n_labels)),
            reverse=True,
        )

        if opaque_count > 0:
            ys, xs = np.where(opaque)
            x1, x2 = int(xs.min()), int(xs.max())
            y1, y2 = int(ys.min()), int(ys.max())
            content_bbox = [x1, y1, x2 - x1 + 1, y2 - y1 + 1]
            content_area = content_bbox[2] * content_bbox[3]
        else:
            content_bbox = [0, 0, 0, 0]
            content_area = 0

        content_ratio = content_area / float(total) if total else 0.0
        transparent_ratio = float(np.count_nonzero(alpha_u8 == 0)) / float(total or 1)
        largest_ratio = (areas[0] / float(total)) if areas else 0.0
        small_island_ratio = (
            (sum(areas[1:]) / float(areas[0])) if (len(areas) > 1 and areas[0] > 0) else 0.0
        )

        flags: list[str] = []
        if unique_colors > int(max_colors_clean):
            flags.append(f"too_many_colors:{unique_colors}>{int(max_colors_clean)}")
        if alpha_levels > 4:
            # alpha_stepify with steps=3 produces 3 levels (+1 if any value
            # rounds differently); >4 means stepify didn't run or got bypassed.
            flags.append(f"alpha_not_stepified:{alpha_levels}_levels")
        if content_ratio < float(min_content_ratio):
            flags.append(f"low_content:{content_ratio:.6f}<{float(min_content_ratio)}")
        if small_island_ratio > float(small_island_warn_ratio):
            flags.append(f"small_islands:{small_island_ratio:.3f}")
        if opaque_count == 0:
            flags.append("empty_layer")

        report: dict[str, Any] = {
            "layer_label": str(layer_label),
            "canvas": {"w": w, "h": h},
            "content_bbox": {
                "x": content_bbox[0],
                "y": content_bbox[1],
                "w": content_bbox[2],
                "h": content_bbox[3],
            },
            "content_ratio": round(content_ratio, 6),
            "transparent_ratio": round(transparent_ratio, 6),
            "unique_colors": unique_colors,
            "alpha_levels": alpha_levels,
            "connected_components": {
                "count": max(0, n_labels - 1),
                "top_areas": areas[:5],
                "largest_ratio": round(largest_ratio, 6),
                "small_island_ratio": round(small_island_ratio, 6),
            },
            "flags": flags,
            "verdict": "clean" if not flags else "needs_review",
        }

        report_json = json.dumps(report, ensure_ascii=False, indent=2)
        vr_log(
            "VR_VectorReadyReport",
            f"label={layer_label} verdict={report['verdict']} "
            f"unique_colors={unique_colors} alpha_levels={alpha_levels} "
            f"content_ratio={content_ratio:.4f} cc={n_labels - 1} flags={flags}",
        )
        return (report_json, image)
