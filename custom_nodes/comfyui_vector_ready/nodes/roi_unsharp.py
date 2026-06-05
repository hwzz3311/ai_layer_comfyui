"""Unsharp mask restricted to an ROI (typically the edge map).

Sharpens designer-intended edges without amplifying diffusion noise in
interior color blocks."""

from __future__ import annotations

import cv2
import numpy as np

from ._utils import (
    np_to_torch_image,
    to_float,
    to_uint8,
    torch_image_to_np,
    torch_mask_to_np,
)


class VR_ROIUnsharpMask:
    CATEGORY = "VectorReady/filter"
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "sharpen"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "roi_mask": ("MASK",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 3.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 3, "min": 1, "max": 15, "step": 2}),
                "dilate_roi": ("INT", {"default": 2, "min": 0, "max": 10}),
            }
        }

    def sharpen(self, image, roi_mask, strength, blur_radius, dilate_roi):
        arr = torch_image_to_np(image)
        masks = torch_mask_to_np(roi_mask)
        out = np.empty_like(arr)
        for i in range(arr.shape[0]):
            frame = to_uint8(arr[i])
            mask = masks[i] if masks.shape[0] > i else masks[0]
            if dilate_roi > 0:
                kernel = np.ones((int(dilate_roi) * 2 + 1,) * 2, np.uint8)
                mask = cv2.dilate(mask, kernel)
            blurred = cv2.GaussianBlur(frame, (int(blur_radius), int(blur_radius)), 0)
            sharp = cv2.addWeighted(frame, 1.0 + float(strength), blurred, -float(strength), 0)
            blend = (frame.astype(np.float32) * (1.0 - mask[..., None])
                     + sharp.astype(np.float32) * mask[..., None])
            out[i] = to_float(np.clip(blend, 0, 255).astype(np.uint8))
        return (np_to_torch_image(out),)
