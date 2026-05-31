"""Diagnostic probe — logs tensor statistics to the ComfyUI console without
modifying the value. Wire it onto any IMAGE or MASK link to see what's
flowing through. Output is exactly the input.

Logging strategy (defense in depth — uses all four channels at once):
  1. Python `logging` module          → ComfyUI's standard log handler / Docker stdout
  2. `print()` with flush=True        → fallback for any stdout-capturing setup
  3. File tee to plugin directory     → guaranteed writable, user can `ls` to find
  4. Optional override via VR_DEBUG_LOG environment variable

The plugin-dir log path uses `__file__` so it follows the install location."""

from __future__ import annotations

import datetime as _dt
import logging
import os
import sys
from pathlib import Path

import torch

logger = logging.getLogger("comfyui_vector_ready")
logger.setLevel(logging.INFO)

# Default: log file lives next to this .py file. With a standard install that
# means /<ComfyUI>/custom_nodes/comfyui_vector_ready/vr_debug.log
_PLUGIN_DIR = Path(__file__).resolve().parent.parent
_DEFAULT_LOG_PATH = _PLUGIN_DIR / "vr_debug.log"
LOG_PATH = Path(os.environ.get("VR_DEBUG_LOG", str(_DEFAULT_LOG_PATH)))


def _write_file(line: str):
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError as e:
        # last-resort: print the file write error so user knows why no file exists
        print(f"[VR_DEBUG_FILE_WRITE_FAILED] {LOG_PATH}: {e}", file=sys.stderr, flush=True)


def vr_log(label: str, message: str):
    """Single entry point — fans out to logging, stdout, and disk."""
    line = f"[VR_DEBUG {_dt.datetime.now().strftime('%H:%M:%S.%f')[:-3]}] {label} :: {message}"
    logger.info(line)
    print(line, flush=True)
    _write_file(line)


# ──── Import-time heartbeat ────────────────────────────────────────────────
# Writing this line at import means: if the file appears, the plugin is loaded.
# If the file doesn't appear after restart, ComfyUI didn't load the plugin at all.
vr_log(
    "PLUGIN_LOADED",
    f"comfyui_vector_ready imported; log file = {LOG_PATH}",
)


def _stats(t: torch.Tensor) -> str:
    arr = t.detach().float().cpu()
    n = arr.numel()
    mn, mx = float(arr.min()), float(arr.max())
    mean = float(arr.mean())
    near0 = float((arr < 0.05).sum()) / n
    near1 = float((arr > 0.95).sum()) / n
    mid = 1.0 - near0 - near1
    return (
        f"shape={tuple(arr.shape)} dtype={t.dtype} "
        f"min={mn:.4f} max={mx:.4f} mean={mean:.4f} "
        f"pct<0.05={near0:.1%} pct>0.95={near1:.1%} mid={mid:.1%}"
    )


def _stats_rgb_channels(t: torch.Tensor, alpha: torch.Tensor | None = None) -> str:
    """Per-pixel grayscale detection + per-channel means.

    'Per-pixel grayscale' = fraction of pixels where max(R,G,B)-min(R,G,B) < 0.02.
    This catches the actual failure mode (every pixel is gray) without false
    positives from naturally balanced palettes (red flower + green leaf can
    have equal channel MEANS but each pixel is vividly colored).

    Also reports per-channel std-dev: if std collapses, the image lost color
    variance even if means look reasonable."""
    if t.shape[-1] < 3:
        return f"<not RGB: shape={tuple(t.shape)}>"
    arr = t.detach().float().cpu()
    if alpha is not None:
        a = alpha.detach().float().cpu()
        if a.dim() == 3 and a.shape[0] != arr.shape[0]:
            a = a.expand(arr.shape[0], -1, -1)
        mask = a > 0.1
        if mask.sum() == 0:
            return "<no foreground pixels (alpha>0.1)>"
        pixels = arr[mask]  # (N, C)
        scope = f"fg (n={int(mask.sum())})"
    else:
        pixels = arr.reshape(-1, arr.shape[-1])
        scope = "all"
    r, g, b = pixels[:, 0], pixels[:, 1], pixels[:, 2]
    r_m, g_m, b_m = float(r.mean()), float(g.mean()), float(b.mean())
    r_s, g_s, b_s = float(r.std()), float(g.std()), float(b.std())
    # per-pixel grayscale: how many pixels are individually gray?
    px_max = pixels[:, :3].max(dim=-1).values
    px_min = pixels[:, :3].min(dim=-1).values
    px_spread = px_max - px_min
    gray_frac = float((px_spread < 0.02).sum()) / pixels.shape[0]
    mean_spread = float(px_spread.mean())
    max_spread = float(px_spread.max())
    if gray_frac > 0.9:
        verdict = f"GRAYSCALE per-pixel ({gray_frac:.1%} gray pixels)"
    elif gray_frac > 0.5:
        verdict = f"mostly desaturated ({gray_frac:.1%} gray pixels)"
    else:
        verdict = f"color OK ({gray_frac:.1%} gray pixels)"
    return (
        f"[{scope}] R={r_m:.3f}±{r_s:.3f} G={g_m:.3f}±{g_s:.3f} B={b_m:.3f}±{b_s:.3f} "
        f"per-px spread mean={mean_spread:.3f} max={max_spread:.3f} → {verdict}"
    )


class _Base:
    CATEGORY = "VectorReady/debug"
    FUNCTION = "probe"

    def _log(self, label: str, t: torch.Tensor):
        vr_log(label or "<unlabeled>", _stats(t))


class VR_DebugProbeImage(_Base):
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "label": ("STRING", {"default": "probe"}),
            }
        }

    def probe(self, image, label):
        self._log(label, image)
        if image.shape[-1] >= 3:
            alpha = image[..., 3] if image.shape[-1] >= 4 else None
            vr_log(f"{label} RGB channels", _stats_rgb_channels(image, alpha))
        if image.shape[-1] >= 4:
            vr_log(f"{label} ALPHA channel", _stats(image[..., 3]))
        return (image,)


class VR_DebugProbeMask(_Base):
    RETURN_TYPES = ("MASK",)

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "label": ("STRING", {"default": "probe"}),
            }
        }

    def probe(self, mask, label):
        self._log(label, mask)
        return (mask,)


class VR_SplitRGBA:
    """Expose the 4 channels of an IMAGE separately so each can be PreviewImage'd."""

    CATEGORY = "VectorReady/debug"
    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("rgb", "alpha")
    FUNCTION = "split"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",)}}

    def split(self, image):
        if image.shape[-1] >= 4:
            rgb = image[..., :3].contiguous()
            alpha = image[..., 3].contiguous()
        else:
            rgb = image[..., :3].contiguous()
            alpha = torch.ones(image.shape[:3], dtype=image.dtype, device=image.device)
        vr_log("SplitRGBA rgb", _stats(rgb))
        vr_log("SplitRGBA alpha", _stats(alpha))
        return (rgb, alpha)
