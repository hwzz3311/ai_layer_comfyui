# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project purpose

This repo is the **AI Layer Reconstruction** project — *not* matting / SAM cutout. The goal is to reverse-engineer a finished design image back into a PSD-like stack of RGBA layers (with z-order and occluded-region reconstruction). Callers are VL agents, not humans; the agent provides which layer to extract and whether it's occluded, and the workflow itself does the heavy lifting (the agent is treated as an unreliable hint source).

Two execution archetypes flow through the same ComfyUI workflow, selected by a `foreground_mode` boolean:

- **A path (Light)** — foreground extraction. RGB from the original image is trustworthy; only edge + alpha cleanup are needed.
- **B path (Strong)** — background reconstruction. RGB is Qwen-generated in inpainted regions; needs full color quantization + region merge + edge sharpening + alpha cleanup.

The runtime selector is `VR_GatedPassthrough` on each KSampler's `latent_image` input — the unselected branch receives `ExecutionBlocker`, pruning its downstream chain.

## Repository layout

- `comfyui_vector_ready/` — the ComfyUI custom-nodes package. Installed by symlinking/copying into `ComfyUI/custom_nodes/`.
  - `nodes/` — atomic RGBA post-processing ops (LAB convert, k-means, bilateral, edge-aware merge, ROI unsharp, canny, alpha stepify, gated passthrough, join RGBA, debug probes).
  - `presets/pipeline.py` — `VR_PipelineLight` (A) and `VR_PipelineStrong` (B) composite nodes that wire the atomic ops in the canonical order.
  - `presets/pipeline_debug.py` — DEBUG variants that expose every intermediate stage as an extra output (used by the debug workflow JSON).
  - `nodes/_utils.py` — the **only** correct way to bridge ComfyUI tensors ↔ numpy/cv2. Use these helpers; don't roll your own conversions.
  - `nodes/debug_probe.py` — `vr_log()` and the `VR_DebugProbe*` passthrough nodes. Logs go to stdout AND `vr_debug.log` next to the plugin (override with `VR_DEBUG_LOG` env var).
- `scripts/build_v8_json.py` — generates `qwen_layered_v8_ab_vector_ready.json` from the v7 base by injecting `VR_GatedPassthrough` switches and VectorReady tails on both paths.
- `scripts/patch_v8_to_debug.py` — derives `qwen_layered_v8_debug.json` by swapping production pipelines for their `*Debug` variants and wiring `PreviewImage` to every stage.
- `qwen_layered_v*.json` — ComfyUI workflow files. Treat `v8_ab_vector_ready` as the production workflow and `v8_debug` as the diagnostic one; regenerate them via the scripts rather than hand-editing.
- `docs/plans/` — design docs for in-progress reconstruction work.
- `ai_layer_reconstruction_state.md`, `Qwen-Image-Layered*_README.md` — long-form project context (in Chinese) covering design philosophy, A/B path rationale, and Qwen model behaviors.

## Tensor conventions (matches ComfyUI core)

- `IMAGE`: `torch.Tensor [B, H, W, C]` float32 0–1, RGB or RGBA.
- `MASK`: `torch.Tensor [B, H, W]` float32 0–1.
- Many Qwen-Layered outputs arrive as 4-channel RGBA. The pipeline nodes call `_resolve_alpha(image, alpha, alpha_source)` with `alpha_source` chosen per node-widget. Defaults to `"auto"` (prefer native RGBA alpha, fall back to MASK socket if image is RGB only — the 2026-05-27 invariant). Override to `"mask_socket"` when the upstream model's native alpha doesn't represent the layer silhouette — e.g., Qwen-Image-Layered's native alpha marks "where white was painted" and **drops line-art detail** (eyes/whiskers/outlines) if used as a mask; the production A path now uses `"mask_socket"` with SAM3 as the external silhouette source (fixed 2026-05-28, v0.8.0).
- Qwen-Image-Layered's RGB in `alpha=0` regions contains decoder noise. Both pipelines call `_premultiply()` before Canny / unsharp to avoid amplifying it into speckles. Don't remove this step.

## Common tasks

Run from the repo root.

```bash
# Rebuild the production workflow JSON from v7
python scripts/build_v8_json.py

# Re-derive the debug workflow from the production one
python scripts/patch_v8_to_debug.py

# Install plugin into a ComfyUI checkout (symlink during development)
ln -s "$PWD/comfyui_vector_ready" /path/to/ComfyUI/custom_nodes/
```

There is no test suite (`comfyui_vector_ready/tests/` is empty) and no lint config. Validate changes by loading the updated workflow JSON in ComfyUI and inspecting `comfyui_vector_ready/vr_debug.log` for per-stage tensor stats — the debug pipeline emits `_stats` / `_stats_rgb_channels` at every intermediate.

## Adding or editing a VR node

1. Implement in `comfyui_vector_ready/nodes/<name>.py` following the ComfyUI node protocol (`INPUT_TYPES`, `RETURN_TYPES`, `CATEGORY = "VectorReady/*"`, `FUNCTION`).
2. Register the class in both `NODE_CLASS_MAPPINGS` and `NODE_DISPLAY_NAME_MAPPINGS` in `comfyui_vector_ready/__init__.py`.
3. If it belongs in the canonical A or B pipeline, also wire it into `presets/pipeline.py` AND mirror the change in `presets/pipeline_debug.py` so debug parity is maintained.
4. Use `vr_log("StageName", _stats(tensor))` at boundaries — every existing pipeline stage logs in/out, and the debug workflow depends on these lines.
