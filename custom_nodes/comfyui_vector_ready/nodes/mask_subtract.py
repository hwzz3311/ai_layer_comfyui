"""Subtract an inner (cutout) mask from an outer (subject) mask.

Designed for subjects whose silhouette has topological holes — e.g. card
holders with a photo-slot window, picture frames, donut shapes. The outer
mask is produced by the positive LA+SAM3 chain; the inner mask comes from
a symmetric LA+SAM3 chain driven by a "cutout query" describing the hole.

Subtraction is soft (operates on float alpha, not binary) and supports a
small dilation of the inner mask before subtracting, to avoid leaving a
hairline rim of the subject color around the cutout boundary.

When `inner` is all zeros (e.g. cutout query was empty and the negative
chain short-circuited), this node is a no-op passthrough of `outer`.
"""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import np_to_torch_mask, torch_mask_to_np
from .debug_probe import _stats, vr_log


def _dilate(mask_np: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return mask_np
    k = radius * 2 + 1
    kernel = np.ones((k, k), np.uint8)
    out = np.empty_like(mask_np)
    for i in range(mask_np.shape[0]):
        binary = (mask_np[i] > 0.0).astype(np.uint8)
        dilated = cv2.dilate(binary, kernel)
        # Preserve soft edges by taking max(original, dilated_hard) so the
        # subtracted area is at least the dilated binary, but keeps the soft
        # alpha values where the original mask already had them.
        out[i] = np.maximum(mask_np[i], dilated.astype(np.float32))
    return out


def _fill_holes(mask_np: np.ndarray) -> np.ndarray:
    """Fill interior holes of each mask in the batch, preserving soft edges.

    A "hole" is any 0-region fully enclosed by the mask — e.g. the white
    line-art interior of a cat face that SAM3 failed to ground because it
    shares the background color. We flood-fill the *exterior* from the image
    border, so whatever the flood can't reach is an enclosed hole, and raise
    those pixels to 1.0. Existing soft-alpha values are kept via max(), so
    only hole interiors change.

    This is intentionally gated by the caller (only run when the negative
    cutout chain is non-empty): a subject's hole is treated as legitimate
    transparency unless the agent explicitly declared a cutout for it, so an
    empty negative chain must leave genuine holes (donuts, ring letters,
    eyeglass frames) untouched.
    """
    out = np.empty_like(mask_np)
    for i in range(mask_np.shape[0]):
        binary = (mask_np[i] > 0.5).astype(np.uint8)
        # Pad a 1px background border so the flood seed is guaranteed to land
        # on exterior background even when the subject touches an image edge
        # (without the pad, a corner-touching subject would invert the fill
        # and swallow the whole frame).
        padded = cv2.copyMakeBorder(binary, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        ff_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(padded, ff_mask, (0, 0), 1)
        flood = padded[1:-1, 1:-1]
        # Anything the exterior flood couldn't reach is an enclosed hole.
        holes = (flood == 0)
        out[i] = np.maximum(mask_np[i], holes.astype(np.float32))
    return out


def _expand_batch(arr: np.ndarray, batch: int) -> np.ndarray:
    if arr.shape[0] == batch:
        return arr
    return np.repeat(arr[:1], batch, axis=0)


class VR_MaskSubtract:
    """final = clamp(outer - dilate(inner, inner_dilate_px), 0, 1)."""

    CATEGORY = "VectorReady/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "subtract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "outer": ("MASK",),
                "inner": ("MASK",),
                # Slight expansion of the inner mask before subtraction so the
                # subject's color doesn't bleed a 1-2px rim into the cutout.
                # Set to 0 for exact subtraction.
                "inner_dilate_px": ("INT", {"default": 2, "min": 0, "max": 32, "step": 1}),
                # Below this threshold the inner mask is treated as empty —
                # short-circuits to pure passthrough so an empty negative
                # chain (cutout query unset) doesn't pay any compute.
                "min_inner_area_ratio": (
                    "FLOAT",
                    {"default": 0.0005, "min": 0.0, "max": 1.0, "step": 0.0001},
                ),
                # When the negative cutout chain is ACTIVE (inner non-empty),
                # fill the outer mask's interior holes before subtracting so a
                # subject whose silhouette has white/line-art interiors that
                # SAM3 failed to ground (e.g. cat-card faces) becomes a solid
                # piece, and only the explicitly-queried cutout is re-opened.
                # Gated on inner being non-empty: an empty cutout chain leaves
                # genuine holes (donuts, ring letters, frames) untouched.
                "fill_outer_holes": ("BOOLEAN", {"default": True}),
                # Over-subtraction guard. If the subtraction would erase more
                # of the outer subject than this leaves behind (result area /
                # outer area < threshold), the inner mask almost certainly
                # over-grounded (e.g. SAM3 #2 segmented the whole subject as
                # the "window"). Revert to outer and warn instead of emitting a
                # near-empty layer.
                "min_retained_ratio": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01},
                ),
            }
        }

    def subtract(
        self,
        outer,
        inner,
        inner_dilate_px,
        min_inner_area_ratio,
        fill_outer_holes,
        min_retained_ratio,
    ):
        outer_np = torch_mask_to_np(outer)
        inner_np = torch_mask_to_np(inner)
        batch = max(outer_np.shape[0], inner_np.shape[0])
        outer_np = _expand_batch(outer_np, batch)
        inner_np = _expand_batch(inner_np, batch)

        total = float(outer_np.shape[1] * outer_np.shape[2])
        inner_area_ratio = float((inner_np > 0.5).sum()) / max(total * batch, 1.0)
        inner_max = float(inner_np.max())
        inner_mean = float(inner_np.mean())
        outer_area_ratio = float((outer_np > 0.5).sum()) / max(total * batch, 1.0)

        if inner_area_ratio < float(min_inner_area_ratio):
            # Distinguish "caller intentionally left cutout query empty" from
            # "cutout query was set but upstream chain (SAM3 / MaskFix /
            # MaskUnion) silently dropped everything" — the latter is a
            # WORKFLOW bug and is the more dangerous case because the saved
            # PNG will include the region that should have been cut out.
            if inner_max == 0.0 and outer_area_ratio > 0.01:
                vr_log(
                    "VR_MaskSubtract",
                    f"WARN [WORKFLOW?] inner is literally all-zero "
                    f"(max={inner_max:.4f} mean={inner_mean:.6f}) while outer has "
                    f"content (area_ratio={outer_area_ratio:.4f}). If the cutout "
                    f"query was non-empty this means the negative chain "
                    f"(LA#2 → SAM3#2 → MaskFix+ → VR_MaskUnion) is broken — "
                    f"check that VR_MaskUnion actually executed and that its "
                    f"output is wired to VR_MaskSubtract.inner.",
                )
            else:
                vr_log(
                    "VR_MaskSubtract",
                    f"inner_area_ratio={inner_area_ratio:.6f} < "
                    f"min={float(min_inner_area_ratio):.6f} "
                    f"(inner_max={inner_max:.4f} inner_mean={inner_mean:.6f}) "
                    f"→ passthrough (no subtraction)",
                )
            return (np_to_torch_mask(outer_np),)

        # Negative chain is active → fill the outer subject's interior holes
        # so SAM3 under-segmentation (white line-art faces, etc.) doesn't leak
        # through as spurious transparency. Only the explicitly-queried cutout
        # below is allowed to re-open the silhouette.
        filled = bool(fill_outer_holes)
        outer_solid = _fill_holes(outer_np) if filled else outer_np

        dilated_inner = _dilate(inner_np, int(inner_dilate_px))
        result = np.clip(outer_solid - dilated_inner, 0.0, 1.0)

        # Over-subtraction guard: if subtracting the inner cutout erases almost
        # the entire subject, the inner mask over-grounded — revert to the
        # (hole-filled) outer rather than ship a near-empty layer.
        solid_area = float((outer_solid > 0.5).sum())
        result_area = float((result > 0.5).sum())
        retained = result_area / solid_area if solid_area > 0 else 1.0
        if solid_area > 0 and retained < float(min_retained_ratio):
            vr_log(
                "VR_MaskSubtract",
                f"WARN [over-subtract] retained={retained:.4f} < "
                f"min={float(min_retained_ratio):.4f} "
                f"(outer_solid_area={solid_area:.0f} result_area={result_area:.0f} "
                f"inner_mean={inner_np.mean():.4f}) — inner cutout over-grounded; "
                f"reverting to outer (filled={filled}, no subtraction).",
            )
            outer_solid_t = np_to_torch_mask(outer_solid)
            return (outer_solid_t,)

        result_t = np_to_torch_mask(result)
        vr_log(
            "VR_MaskSubtract",
            f"outer_mean={outer_np.mean():.4f} inner_mean={inner_np.mean():.4f} "
            f"fill_outer_holes={filled} retained={retained:.4f} "
            f"inner_dilate_px={int(inner_dilate_px)} "
            f"inner_area_ratio={inner_area_ratio:.6f} {_stats(result_t)}",
        )
        return (result_t,)


class VR_MaskUnion:
    """Pixel-wise max of two masks — used by the v8.2 negative chain to
    combine LA #2's coarse N-box union mask with SAM3 #2's refined single-
    instance mask. This lets multi-hole subjects (picture frames, grids of
    photo slots) be fully subtracted even when SAM3's dual-prompt only
    refines the primary bbox: any extra holes LA found are still subtracted
    via their (looser) rectangle masks.

    Empty inputs are no-ops — if `mask_b` is all zeros (e.g. SAM3 returned
    nothing) the output is `mask_a`, and vice versa, so passthrough
    semantics for empty negative chains are preserved.
    """

    CATEGORY = "VectorReady/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = "union"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "mask_b": ("MASK",),
            }
        }

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Same rationale as VR_GatedPassthrough.IS_CHANGED: this node's log
        # line is the single source of truth for "did the negative-cutout
        # chain run?" — caching it across runs hides the answer.
        import time
        return time.time()

    def union(self, mask_a, mask_b):
        a_np = torch_mask_to_np(mask_a)
        b_np = torch_mask_to_np(mask_b)
        batch = max(a_np.shape[0], b_np.shape[0])
        a_np = _expand_batch(a_np, batch)
        b_np = _expand_batch(b_np, batch)

        # If shapes disagree (e.g. SAM3 returned an empty placeholder of a
        # different resolution), fall back to whichever side carries content.
        if a_np.shape[1:] != b_np.shape[1:]:
            a_has = float(a_np.max()) > 0.0
            b_has = float(b_np.max()) > 0.0
            if a_has and not b_has:
                vr_log("VR_MaskUnion", f"shape mismatch, returning A {a_np.shape}")
                return (np_to_torch_mask(a_np),)
            if b_has and not a_has:
                vr_log("VR_MaskUnion", f"shape mismatch, returning B {b_np.shape}")
                return (np_to_torch_mask(b_np),)
            # Both have content but disagree on shape — log and prefer A.
            vr_log(
                "VR_MaskUnion",
                f"shape mismatch with content on both sides "
                f"A={a_np.shape} B={b_np.shape}; preferring A",
            )
            return (np_to_torch_mask(a_np),)

        result = np.maximum(a_np, b_np)
        result_t = np_to_torch_mask(result)
        vr_log(
            "VR_MaskUnion",
            f"a_mean={a_np.mean():.4f} b_mean={b_np.mean():.4f} {_stats(result_t)}",
        )
        return (result_t,)
