"""Alpha edge refinement and edge ROI extraction.

Tightens soft model mattes near the silhouette boundary and emits an edge mask
that combines alpha-boundary structure with guide-image edges.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_mask, torch_image_to_np, torch_mask_to_np


class VR_AlphaEdgeRefine:
    CATEGORY = "VectorReady/alpha"
    RETURN_TYPES = ("MASK", "MASK")
    RETURN_NAMES = ("alpha", "edge_roi")
    FUNCTION = "refine"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "alpha": ("MASK",),
                "guide_image": ("IMAGE",),
                "edge_radius": ("INT", {"default": 2, "min": 1, "max": 8}),
                "snap_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.05}),
                "contrast": ("FLOAT", {"default": 8.0, "min": 1.0, "max": 24.0, "step": 0.5}),
            }
        }

    def refine(self, alpha, guide_image, edge_radius, snap_strength, contrast):
        alpha_np = torch_mask_to_np(alpha)
        guide_np = torch_image_to_np(guide_image)
        batch = max(alpha_np.shape[0], guide_np.shape[0])
        out_alpha = np.empty((batch, *alpha_np.shape[1:]), dtype=np.float32)
        out_edges = np.empty_like(out_alpha)

        radius = max(1, int(edge_radius))
        kernel = np.ones((radius * 2 + 1, radius * 2 + 1), np.uint8)
        strength = np.clip(float(snap_strength), 0.0, 1.0)
        slope = max(1.0, float(contrast))

        for i in range(batch):
            a = np.clip(alpha_np[i if alpha_np.shape[0] > i else 0], 0.0, 1.0)
            guide = np.clip(guide_np[i if guide_np.shape[0] > i else 0], 0.0, 1.0)

            soft_band = ((a > 0.02) & (a < 0.98)).astype(np.uint8)
            binary = (a >= 0.5).astype(np.uint8)
            alpha_boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)
            edge_band = cv2.dilate(np.maximum(soft_band, alpha_boundary), kernel).astype(np.float32)

            gray = cv2.cvtColor((guide * 255.0 + 0.5).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            guide_edges = (cv2.Canny(gray, 40, 120) > 0).astype(np.float32)
            guide_edges = cv2.dilate(guide_edges, kernel).astype(np.float32) * edge_band
            edge_roi = np.clip(np.maximum(edge_band, guide_edges), 0.0, 1.0)

            blurred = cv2.GaussianBlur(a.astype(np.float32), (3, 3), 0)
            sharpened = np.clip(a + 0.75 * (a - blurred), 0.0, 1.0)
            snapped = 1.0 / (1.0 + np.exp(-(sharpened - 0.5) * slope))
            refined = a * (1.0 - strength * edge_band) + snapped * (strength * edge_band)

            out_alpha[i] = np.clip(refined, 0.0, 1.0)
            out_edges[i] = edge_roi

        return (np_to_torch_mask(out_alpha), np_to_torch_mask(out_edges))
