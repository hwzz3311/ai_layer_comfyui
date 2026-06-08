"""HuggingFace matting/segmentation alpha node.

This is the first real-model entry point for A-path target-aware matting. It
loads a local or HuggingFace model that behaves like an image-segmentation
model (for example BRIA RMBG or BiRefNet), returns a soft alpha mask, and clips
the result by the candidate target mask so the model cannot select unrelated
foregrounds outside the requested layer.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
import torch
import torch.nn.functional as F

from ._utils import np_to_torch_mask, torch_image_to_np, torch_mask_to_np
from .debug_probe import _stats, vr_log


DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]
DEFAULT_RMBG_MODEL_PATH = "/root/ComfyUI/models/RMBG-2.0"


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


@lru_cache(maxsize=2)
def _load_model(model_id: str, device_name: str):
    try:
        from transformers import AutoModelForImageSegmentation
    except Exception as exc:  # pragma: no cover - depends on ComfyUI env
        raise RuntimeError(
            "VR_HFMattingAlpha requires transformers. Install it in the ComfyUI "
            "Python environment or provide an existing matting node output to "
            "VR_PipelineLight.external_matte_alpha."
        ) from exc

    device = _pick_device(device_name)
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    return model, device


def _preprocess(frame: np.ndarray, input_size: int, device: torch.device) -> torch.Tensor:
    frame_t = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
    frame_t = F.interpolate(
        frame_t,
        size=(int(input_size), int(input_size)),
        mode="bilinear",
        align_corners=False,
    )
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    frame_t = (frame_t - mean) / std
    return frame_t.to(device)


def _extract_prediction(output) -> torch.Tensor:
    if hasattr(output, "logits"):
        pred = output.logits
    elif isinstance(output, dict):
        pred = output.get("logits") or output.get("out") or next(iter(output.values()))
    elif isinstance(output, (list, tuple)):
        pred = output[-1]
    else:
        pred = output
    if isinstance(pred, (list, tuple)):
        pred = pred[-1]
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if pred.shape[1] > 1:
        pred = pred[:, :1]
    return pred


class VR_HFMattingAlpha:
    CATEGORY = "VectorReady/matting"
    RETURN_TYPES = ("MASK", "MASK", "MASK")
    RETURN_NAMES = ("matte_alpha", "confidence", "raw_matte")
    FUNCTION = "matte"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "candidate_mask": ("MASK",),
                "model_id": ("STRING", {"default": DEFAULT_RMBG_MODEL_PATH}),
                "input_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 64}),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
            }
        }

    def matte(self, image, candidate_mask, model_id, input_size, device):
        frames = torch_image_to_np(image)
        candidates = torch_mask_to_np(candidate_mask)
        model, resolved_device = _load_model(str(model_id), str(device))

        batch = max(frames.shape[0], candidates.shape[0])
        h, w = frames.shape[1:3]
        out = np.zeros((batch, h, w), dtype=np.float32)
        raw = np.zeros((batch, h, w), dtype=np.float32)

        for i in range(batch):
            frame = frames[i if frames.shape[0] > i else 0]
            cand = candidates[i if candidates.shape[0] > i else 0]
            inp = _preprocess(frame, int(input_size), resolved_device)
            with torch.no_grad():
                pred_raw = _extract_prediction(model(inp))
                pred = torch.sigmoid(pred_raw)
                pred = F.interpolate(pred, size=(h, w), mode="bilinear", align_corners=False)
            alpha = pred[0, 0].detach().float().cpu().numpy()
            # raw_matte: unclipped full-image foreground matte (for tiered fallback)
            raw[i] = np.clip(alpha, 0.0, 1.0)
            # matte_alpha: clipped by candidate so the model can't select unrelated objects
            out[i] = np.clip(alpha * np.clip(cand, 0.0, 1.0), 0.0, 1.0)

        matte_t = np_to_torch_mask(out)
        raw_t = np_to_torch_mask(raw)
        # For current BRIA/BiRefNet-style models there is no separate confidence.
        conf_t = matte_t
        vr_log(
            "VR_HFMattingAlpha",
            f"model_id={model_id} device={resolved_device} input_size={input_size}",
        )
        vr_log("VR_HFMattingAlpha matte_alpha", _stats(matte_t))
        vr_log("VR_HFMattingAlpha raw_matte", _stats(raw_t))
        return (matte_t, conf_t, raw_t)
