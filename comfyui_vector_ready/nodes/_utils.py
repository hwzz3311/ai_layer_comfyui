"""Tensor format helpers for ComfyUI ↔ OpenCV/numpy bridging."""

from __future__ import annotations

import numpy as np
import torch


def torch_image_to_np(image: torch.Tensor) -> np.ndarray:
    """ComfyUI IMAGE [B,H,W,C] float32 0-1 → np.ndarray [B,H,W,3] float32 RGB.

    Strips any alpha channel — cv2 operations downstream require 3-channel input.
    Use `split_rgba` if you need the alpha back."""
    arr = image.detach().cpu().numpy().astype(np.float32)
    if arr.shape[-1] >= 4:
        arr = arr[..., :3]
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr


def split_rgba(image: torch.Tensor):
    """Return (rgb_np[B,H,W,3], alpha_np[B,H,W] or None) from a ComfyUI IMAGE tensor."""
    arr = image.detach().cpu().numpy().astype(np.float32)
    if arr.shape[-1] >= 4:
        return arr[..., :3], arr[..., 3]
    return arr[..., :3] if arr.shape[-1] >= 3 else np.repeat(arr, 3, axis=-1), None


def np_to_torch_image(arr: np.ndarray) -> torch.Tensor:
    """np [B,H,W,C] float → torch IMAGE."""
    if arr.ndim == 3:
        arr = arr[None, ...]
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))


def torch_mask_to_np(mask: torch.Tensor) -> np.ndarray:
    """ComfyUI MASK [B,H,W] float32 0-1 → np [B,H,W] float32."""
    return mask.detach().cpu().numpy().astype(np.float32)


def np_to_torch_mask(arr: np.ndarray) -> torch.Tensor:
    if arr.ndim == 2:
        arr = arr[None, ...]
    return torch.from_numpy(np.clip(arr, 0.0, 1.0).astype(np.float32))


def to_uint8(img_float: np.ndarray) -> np.ndarray:
    return (np.clip(img_float, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)


def to_float(img_uint8: np.ndarray) -> np.ndarray:
    return img_uint8.astype(np.float32) / 255.0
