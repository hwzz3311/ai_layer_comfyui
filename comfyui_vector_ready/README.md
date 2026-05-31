# comfyui_vector_ready

Vectorization-friendly RGBA post-processing nodes for ComfyUI. Designed for
the AI Layer Reconstruction workflow (Qwen-Image-Layered + SAM3 + custom
brush conditioning), but usable in any pipeline that needs to turn
diffusion-grade RGBA into vectorizer-friendly RGBA.

## Install

Drop this directory under `ComfyUI/custom_nodes/`:

```
ComfyUI/custom_nodes/comfyui_vector_ready/
```

Install dependencies into ComfyUI's Python env:

```
pip install opencv-python scipy
```

(`numpy` and `torch` are already required by ComfyUI itself.)

Restart ComfyUI. Nodes appear under the **VectorReady** category.

## Nodes

| Node | Category | Purpose |
|---|---|---|
| `VR · LAB Convert` | color | RGB ↔ LAB |
| `VR · K-means Quantize (LAB)` | color | Color palette reduction, adaptive K from histogram peaks |
| `VR · Edge-aware Region Merge` | color | Merge similar adjacent regions unless an edge separates them |
| `VR · Bilateral Filter` | filter | Edge-preserving denoise |
| `VR · ROI Unsharp Mask` | filter | Sharpen only on the edge ROI |
| `VR · Canny Edge` | edge | Edge map as MASK |
| `VR · Alpha Stepify` | alpha | Quantize alpha to 2 or 3 levels |
| `VR · Pipeline (Light, A-path)` | preset | Composite for foreground-extraction paths |
| `VR · Pipeline (Strong, B-path)` | preset | Composite for background-reconstruction paths |

## Tensor conventions

Matches ComfyUI core:

- `IMAGE`: `torch.Tensor` `[B, H, W, 3]` float32 0-1 RGB
- `MASK`: `torch.Tensor` `[B, H, W]` float32 0-1
