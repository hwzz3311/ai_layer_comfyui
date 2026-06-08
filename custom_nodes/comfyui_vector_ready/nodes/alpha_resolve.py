"""Tiered alpha fallback: resolved -> rmbg raw -> qwen native.

Module top-level stays torch-free so resolve_alpha_core is unit-testable in
envs without torch/cv2. torch-dependent helpers (_utils, debug_probe) are
imported lazily inside resolve().
"""
from __future__ import annotations

import numpy as np


def _ratio(m: np.ndarray) -> float:
    return float((m >= 0.5).sum()) / m.size if m.size else 0.0


def resolve_alpha_core(resolved, rmbg, native, min_area_ratio):
    """Pick first non-empty alpha by tier. Pure numpy, unit-testable."""
    if _ratio(resolved) >= min_area_ratio:
        return resolved.astype(np.float32), "resolved"
    if _ratio(rmbg) >= min_area_ratio:
        return rmbg.astype(np.float32), "rmbg"
    return native.astype(np.float32), "native"


class VR_AlphaResolve:
    CATEGORY = "VectorReady/mask"
    RETURN_TYPES = ("MASK", "STRING")
    RETURN_NAMES = ("alpha", "source_used")
    FUNCTION = "resolve"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "resolved_alpha": ("MASK",),
                "rmbg_alpha": ("MASK",),
                "native_alpha": ("MASK",),
                "min_area_ratio": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1.0, "step": 0.0005}),
            }
        }

    def resolve(self, resolved_alpha, rmbg_alpha, native_alpha, min_area_ratio):
        from ._utils import np_to_torch_mask, torch_mask_to_np  # lazy: torch only at runtime
        from .debug_probe import _stats, vr_log

        resolved = torch_mask_to_np(resolved_alpha)
        rmbg = torch_mask_to_np(rmbg_alpha)
        native = torch_mask_to_np(native_alpha)
        batch = max(resolved.shape[0], rmbg.shape[0], native.shape[0])
        out = np.zeros((batch,) + resolved.shape[1:], dtype=np.float32)
        sources = []
        for i in range(batch):
            r = resolved[i if resolved.shape[0] > i else 0]
            g = rmbg[i if rmbg.shape[0] > i else 0]
            n = native[i if native.shape[0] > i else 0]
            picked, src = resolve_alpha_core(r, g, n, float(min_area_ratio))
            out[i] = picked
            sources.append(src)
        alpha_t = np_to_torch_mask(out)
        source_used = sources[0] if sources else "native"
        vr_log("VR_AlphaResolve", f"source_used={source_used} {_stats(alpha_t)}")
        return (alpha_t, source_used)
