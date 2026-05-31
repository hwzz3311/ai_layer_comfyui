"""K-means color quantization in LAB space.

K is chosen adaptively from the smoothed luminance/chroma histogram when
auto_k is enabled; otherwise the user-supplied k is used directly."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_image, torch_image_to_np


def _estimate_k(image_lab: np.ndarray, max_k: int) -> int:
    """Coarse peak count on the L channel histogram, clipped to [2, max_k]."""
    L = (image_lab[..., 0] * 255.0).astype(np.uint8)
    hist = cv2.calcHist([L], [0], None, [32], [0, 256]).flatten()
    hist = cv2.GaussianBlur(hist.reshape(-1, 1), (5, 1), 1.0).flatten()
    # local maxima
    peaks = 0
    for i in range(1, len(hist) - 1):
        if hist[i] > hist[i - 1] and hist[i] > hist[i + 1] and hist[i] > hist.max() * 0.05:
            peaks += 1
    return int(max(2, min(max_k, peaks if peaks > 0 else max_k // 2)))


def _kmeans_lab(image_lab: np.ndarray, k: int) -> np.ndarray:
    h, w, _ = image_lab.shape
    samples = image_lab.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(samples, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    return centers[labels.flatten()].reshape(h, w, 3)


class VR_KMeansQuantize:
    CATEGORY = "VectorReady/color"
    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("image", "k_used")
    FUNCTION = "quantize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_lab": ("IMAGE",),
                "max_k": ("INT", {"default": 12, "min": 2, "max": 64}),
                "auto_k": ("BOOLEAN", {"default": True}),
                "fixed_k": ("INT", {"default": 8, "min": 2, "max": 64}),
            }
        }

    def quantize(self, image_lab, max_k, auto_k, fixed_k):
        arr = torch_image_to_np(image_lab)
        out = np.empty_like(arr)
        k_used = 0
        for i in range(arr.shape[0]):
            k = _estimate_k(arr[i], max_k) if auto_k else int(fixed_k)
            out[i] = _kmeans_lab(arr[i], k)
            k_used = k
        return (np_to_torch_image(out), int(k_used))
