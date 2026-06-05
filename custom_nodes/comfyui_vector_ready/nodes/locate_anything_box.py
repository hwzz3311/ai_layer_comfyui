"""LocateAnything grounding node.

Converts a text target description into a coarse rectangular MASK. The box is
used as a hard spatial fallback for Qwen Layered brush conditioning when SAM
fails to produce a usable target mask.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
import re

import cv2
import numpy as np
import torch
from PIL import Image

from ._utils import np_to_torch_image, np_to_torch_mask, torch_image_to_np, to_uint8
from .debug_probe import _stats, vr_log


DEVICE_CHOICES = ["auto", "cuda", "mps", "cpu"]
GENERATION_MODES = ["hybrid", "fast", "slow"]
PROMPT_MODES = ["single", "multi", "raw"]
ATTN_IMPLS = ["keep", "sdpa", "eager", "flash_attention_2"]
DEFAULT_MODEL_ID = "/root/ComfyUI/models/LocateAnything-3B"


def _pick_device(device: str) -> torch.device:
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device)


def _pick_dtype(device: torch.device):
    if device.type == "cuda":
        return torch.bfloat16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def _patch_flash_attn_cu_seqlens():
    """Wrap flash_attn_varlen_func to normalize cu_seqlens shape/dtype.

    LocateAnything-3B was authored against an older flash-attn that accepted
    cu_seqlens with relaxed shape/dtype. flash-attn >=2.7 enforces strict
    1-D int32 contiguous tensors of shape (batch_size+1,). We coerce inputs
    in-flight so the model code (which we can't easily edit) keeps working.

    Must run BEFORE the LocateAnything custom modeling module is imported,
    because that module does `from flash_attn import flash_attn_varlen_func`
    at import-time and caches the binding locally — patching the flash_attn
    package later won't reach those bindings.
    """
    try:
        import flash_attn  # noqa: F401
        from flash_attn import flash_attn_interface as _fai
    except Exception:
        return
    if getattr(_fai, "_vr_cu_seqlens_patched", False):
        return

    _orig = _fai.flash_attn_varlen_func

    def _coerce(t):
        if t is None or not hasattr(t, "reshape"):
            return t
        return t.reshape(-1).contiguous().to(torch.int32)

    def _wrapped(*args, **kwargs):
        for k in ("cu_seqlens_q", "cu_seqlens_k"):
            if k in kwargs:
                kwargs[k] = _coerce(kwargs[k])
        return _orig(*args, **kwargs)

    _fai.flash_attn_varlen_func = _wrapped
    # Top-level re-export — some code does `from flash_attn import ...`
    import flash_attn as _fa_pkg
    if hasattr(_fa_pkg, "flash_attn_varlen_func"):
        _fa_pkg.flash_attn_varlen_func = _wrapped

    # Also patch the deeper torch custom_op path. FA >=2.7 routes through
    # `_wrapped_flash_attn_varlen_forward` which is registered as a custom op;
    # the model goes Func.apply -> custom_op -> _flash_attn_varlen_forward.
    # We need to intercept at the lowest pure-Python layer so coercion happens
    # right before the C++ kernel call.
    try:
        _orig_low = _fai._flash_attn_varlen_forward
        def _wrapped_low(q, k, v, cu_seqlens_q, cu_seqlens_k, *args, **kwargs):
            cu_seqlens_q = _coerce(cu_seqlens_q)
            cu_seqlens_k = _coerce(cu_seqlens_k)
            return _orig_low(q, k, v, cu_seqlens_q, cu_seqlens_k, *args, **kwargs)
        _fai._flash_attn_varlen_forward = _wrapped_low
    except Exception:
        pass

    _fai._vr_cu_seqlens_patched = True
    print("[VR_LocateAnythingBox] flash_attn cu_seqlens patch installed")


def _patch_loaded_model_module(model):
    """After the model is loaded, also overwrite the cached binding inside
    the custom modeling module — covers the case where the module was already
    imported in this process before our pre-import patch ran.
    """
    try:
        import sys
        from flash_attn.flash_attn_interface import flash_attn_varlen_func as patched
    except Exception:
        return
    mod_name = type(model).__module__  # e.g. transformers_modules.LocateAnything_hyphen_3B.modeling_locateanything
    root = mod_name.rsplit(".", 1)[0] if "." in mod_name else mod_name
    for name, mod in list(sys.modules.items()):
        if not name.startswith(root):
            continue
        if mod is None:
            continue
        if hasattr(mod, "flash_attn_varlen_func"):
            try:
                setattr(mod, "flash_attn_varlen_func", patched)
            except Exception:
                pass


def _magi_available() -> bool:
    try:
        import magi_attention  # noqa: F401
        return True
    except Exception:
        return False


@lru_cache(maxsize=2)
def _load_worker(model_id: str, device_name: str, attn_impl: str):
    # Patch BEFORE importing transformers / triggering custom modeling import.
    _patch_flash_attn_cu_seqlens()

    try:
        from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on ComfyUI env
        raise RuntimeError(
            "VR_LocateAnythingBox requires transformers with LocateAnything "
            "custom code support. Install the dependencies in the ComfyUI "
            "Python environment."
        ) from exc

    model_path = os.path.expanduser(os.path.expandvars(str(model_id)))
    device = _pick_device(device_name)
    dtype = _pick_dtype(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Resolve "keep" against the actual environment. LocateAnything-3B's
    # config.json sets _attn_implementation to "magi" — a blockwise non-causal
    # attention from the magi_attention package. When magi_attention isn't
    # installed, the model's own code internally swaps "magi" → "flash_attention_2",
    # but the bundled modeling_qwen2.py forward only branches on a few impls
    # and raises NotImplementedError on "flash_attention_2". Result: load
    # succeeds, generate() crashes.
    #
    # When magi is unavailable we therefore force-override "keep" to "sdpa"
    # before loading, so AutoModel never enters that broken fallback path.
    # The historical warning that "sdpa produces garbage boxes" was about
    # forcing sdpa on top of a WORKING magi install; with magi absent, sdpa
    # is strictly better than crashing and empirically yields usable boxes.
    if attn_impl == "keep" and not _magi_available():
        print(
            "[VR_LocateAnythingBox] magi_attention not installed; "
            "auto-resolving attn_implementation 'keep' → 'sdpa' to avoid the "
            "model's broken flash_attention_2 fallback path"
        )
        attn_impl = "sdpa"

    if attn_impl != "keep":
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

        def _force_attn(cfg):
            if cfg is None:
                return
            try:
                cfg._attn_implementation = attn_impl
                cfg.attn_implementation = attn_impl
                cfg._attn_implementation_internal = attn_impl
            except Exception:
                pass
            for sub_name in (
                "text_config", "vision_config", "language_model",
                "language_config", "vision_tower", "audio_config",
            ):
                sub = getattr(cfg, sub_name, None)
                if sub is not None and sub is not cfg:
                    _force_attn(sub)

        _force_attn(config)
        model = AutoModel.from_pretrained(
            model_path,
            config=config,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation=attn_impl,
        ).to(device).eval()
    else:
        # Respect the model's native attention impl ("magi"). Requires a
        # compatible flash-attn version (pin to 2.5.x for LocateAnything-3B).
        model = AutoModel.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).to(device).eval()
    # Belt-and-suspenders: re-bind the patched flash_attn_varlen_func into
    # the custom modeling module after load, in case it was already imported
    # before our pre-import patch ran (e.g., a previous workflow run).
    _patch_loaded_model_module(model)

    # Only mutate runtime attrs when the user opted into an override —
    # otherwise we'd clobber the model's native "magi" mode.
    if attn_impl != "keep":
        for module in model.modules():
            if hasattr(module, "_attn_implementation"):
                module._attn_implementation = attn_impl
        if hasattr(model, "config"):
            model.config._attn_implementation = attn_impl
    return tokenizer, processor, model, device, dtype


def _prompt(query: str, mode: str) -> str:
    if mode == "raw":
        return query
    if mode == "multi":
        return f"Locate all the instances that match the following description: {query}."
    return f"Locate a single instance that matches the following description: {query}."


def _parse_boxes(answer: str, width: int, height: int) -> list[dict[str, float]]:
    boxes: list[dict[str, float]] = []
    patterns = [
        r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>",
        r"<box>\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*</box>",
    ]
    for pattern in patterns:
        for m in re.finditer(pattern, answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            x1, x2 = sorted((x1, x2))
            y1, y2 = sorted((y1, y2))
            boxes.append({
                "x1": x1 / 1000.0 * width,
                "y1": y1 / 1000.0 * height,
                "x2": x2 / 1000.0 * width,
                "y2": y2 / 1000.0 * height,
            })
    return boxes


def _box_to_mask(box: dict[str, float], height: int, width: int, padding_px: int) -> np.ndarray:
    x1 = int(np.floor(box["x1"])) - int(padding_px)
    y1 = int(np.floor(box["y1"])) - int(padding_px)
    x2 = int(np.ceil(box["x2"])) + int(padding_px)
    y2 = int(np.ceil(box["y2"])) + int(padding_px)
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    mask = np.zeros((height, width), dtype=np.float32)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1.0
    return mask


def _draw_preview(frame: np.ndarray, mask: np.ndarray, boxes: list[dict[str, float]]) -> np.ndarray:
    preview = to_uint8(frame).copy()
    overlay = preview.copy()
    overlay[mask > 0.5] = (255, 64, 64)
    preview = cv2.addWeighted(overlay, 0.35, preview, 0.65, 0)
    for box in boxes:
        x1, y1, x2, y2 = [int(round(box[k])) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(preview, (x1, y1), (x2, y2), (255, 255, 0), 3)
    return preview.astype(np.float32) / 255.0


class VR_LocateAnythingBox:
    CATEGORY = "VectorReady/grounding"
    RETURN_TYPES = ("MASK", "IMAGE", "STRING", "BOOLEAN", "BBOX")
    RETURN_NAMES = ("box_mask", "preview_image", "bbox_json", "box_usable", "bboxes")
    FUNCTION = "locate"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "query": ("STRING", {"default": "main object"}),
                "model_id": ("STRING", {"default": DEFAULT_MODEL_ID}),
                "device": (DEVICE_CHOICES, {"default": "auto"}),
                "generation_mode": (GENERATION_MODES, {"default": "hybrid"}),
                "prompt_mode": (PROMPT_MODES, {"default": "single"}),
                "padding_px": ("INT", {"default": 8, "min": 0, "max": 256, "step": 1}),
                "max_new_tokens": ("INT", {"default": 2048, "min": 64, "max": 8192, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                # "keep" = respect the model's native config (LocateAnything
                # uses a custom "magi" impl — see config.json). Only switch
                # to sdpa/eager if you're debugging or know what you're doing;
                # overriding will run but may produce wrong outputs.
                "attn_implementation": (ATTN_IMPLS, {"default": "keep"}),
                # Free-text tag to distinguish multiple LA instances in logs
                # (e.g. "positive" vs "negative-cutout" chains).
                "label": ("STRING", {"default": "LA"}),
            },
        }

    def locate(
        self,
        image,
        query,
        model_id,
        device,
        generation_mode,
        prompt_mode,
        padding_px,
        max_new_tokens,
        temperature,
        attn_implementation="keep",
        label="LA",
    ):
        frames = torch_image_to_np(image)
        log_tag = f"VR_LocateAnythingBox[{label}]"
        # Log the request inputs BEFORE inference so a crash still leaves a
        # trace of what the caller asked for. Truncate very long queries so
        # the log line stays readable.
        q_disp = str(query)
        if len(q_disp) > 200:
            q_disp = q_disp[:200] + f"...(+{len(q_disp) - 200} chars)"
        vr_log(
            log_tag,
            f"REQUEST query={q_disp!r} prompt_mode={prompt_mode} "
            f"gen_mode={generation_mode} padding_px={padding_px} "
            f"max_new_tokens={max_new_tokens} temperature={temperature} "
            f"frames={frames.shape}",
        )

        # Empty-query short-circuit: when the caller (or the shared
        # PrimitiveNode that drives both LA and SAM3) passes an empty string,
        # skip the 3B-model inference entirely. Returns an empty mask + empty
        # bbox list so downstream subtract / dual-prompt SAM3 nodes naturally
        # behave as no-ops. This is the negative-chain optimization path —
        # workflows that don't need cutout extraction pay zero compute.
        if not str(query).strip():
            empty_mask = np.zeros(frames.shape[:3], dtype=np.float32)
            empty_mask_t = np_to_torch_mask(empty_mask)
            empty_preview_t = np_to_torch_image(frames)
            vr_log(
                log_tag,
                f"empty query → short-circuit (no inference) shape={tuple(empty_mask.shape)}",
            )
            return (empty_mask_t, empty_preview_t, "[]", False, [])

        tokenizer, processor, model, resolved_device, dtype = _load_worker(
            str(model_id), str(device), str(attn_implementation)
        )

        masks = np.zeros(frames.shape[:3], dtype=np.float32)
        previews = np.zeros_like(frames, dtype=np.float32)
        all_results = []
        all_bboxes = []
        any_usable = False

        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            pil = Image.fromarray(to_uint8(frame)).convert("RGB")
            question = _prompt(str(query), str(prompt_mode))
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil},
                    {"type": "text", "text": question},
                ],
            }]
            text = processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = processor.process_vision_info(messages)
            inputs = processor(text=[text], images=images, videos=videos, return_tensors="pt").to(resolved_device)
            pixel_values = inputs["pixel_values"].to(dtype)
            with torch.no_grad():
                response = model.generate(
                    pixel_values=pixel_values,
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    image_grid_hws=inputs.get("image_grid_hws", None),
                    tokenizer=tokenizer,
                    max_new_tokens=int(max_new_tokens),
                    use_cache=True,
                    generation_mode=str(generation_mode),
                    temperature=float(temperature),
                    do_sample=float(temperature) > 0.0,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    verbose=False,
                )
            answer = response[0] if isinstance(response, tuple) else response
            if not isinstance(answer, str):
                answer = str(answer)
            # Log the raw model output (truncated) so caller-side issues
            # (vague query → vague answer) can be distinguished from
            # parsing / downstream issues.
            ans_disp = answer if len(answer) <= 500 else answer[:500] + f"...(+{len(answer) - 500} chars)"
            vr_log(
                log_tag,
                f"frame={i} answer={ans_disp!r}",
            )
            boxes = _parse_boxes(answer, w, h)
            # Caller-side smell: model returned multiple boxes but the caller
            # asked for "single" / "raw" mode, so all but the first will be
            # silently dropped. Surface this clearly so log readers can spot
            # the misconfiguration immediately.
            if len(boxes) > 1 and str(prompt_mode) != "multi":
                vr_log(
                    log_tag,
                    f"WARN [CALLER] model returned {len(boxes)} boxes but "
                    f"prompt_mode={prompt_mode!r} — keeping only the first, "
                    f"{len(boxes) - 1} box(es) dropped. If the query describes "
                    f"multiple instances, set prompt_mode='multi'.",
                )
            if not boxes:
                previews[i] = frame
                all_results.append({"answer": answer, "boxes": []})
                continue

            # multi mode: union every detected box into the frame's mask and
            # emit all bboxes downstream. Used by the v8.2 negative chain so
            # subjects with N internal holes (frames, donuts, gridded windows)
            # get every cutout segmented, not just the first one LA emits.
            # single / raw modes preserve the legacy "take first box" behavior.
            selected = boxes if str(prompt_mode) == "multi" else boxes[:1]

            frame_mask = np.zeros((h, w), dtype=np.float32)
            for box in selected:
                frame_mask = np.maximum(
                    frame_mask, _box_to_mask(box, h, w, int(padding_px))
                )
                bbox_xyxy = [
                    float(max(0.0, min(w, box["x1"]))),
                    float(max(0.0, min(h, box["y1"]))),
                    float(max(0.0, min(w, box["x2"]))),
                    float(max(0.0, min(h, box["y2"]))),
                ]
                all_bboxes.append(bbox_xyxy)

            masks[i] = frame_mask
            previews[i] = _draw_preview(frame, frame_mask, selected)
            any_usable = any_usable or bool(frame_mask.max() > 0.5)
            all_results.append({"answer": answer, "boxes": boxes})

        mask_t = np_to_torch_mask(masks)
        preview_t = np_to_torch_image(previews)
        vr_log(
            log_tag,
            f"RESULT query={q_disp!r} model={model_id} device={resolved_device} "
            f"mode={generation_mode} boxes_found={len(all_bboxes)} "
            f"usable={any_usable} {_stats(mask_t)}",
        )
        return (
            mask_t,
            preview_t,
            json.dumps(all_results, ensure_ascii=False),
            bool(any_usable),
            all_bboxes,
        )
