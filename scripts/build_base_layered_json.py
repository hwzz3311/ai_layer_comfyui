"""Build workflows/layered/base_layered.json — base multi-layer decomposition.

This mirrors ComfyUI's **official** "Image to Layers (Qwen-Image-Layered)"
template (image_qwen_image_layered_comfyui.json, flattened from its subgraph),
then appends our VR_PipelineLayered post-processing tail. It is the official
Qwen-Image-Layered behavior: feed one image, get N RGBA layers in one pass.

Complement to v8 (Control single-target extraction): the agent calls THIS for
the step-one full split; v8 is the targeted / fallback extractor.

CRITICAL parameters copied verbatim from the official template (earlier
hand-derived-from-v8 values were wrong and produced garbage):
  - UNETLoader = qwen_image_layered_fp8mixed.safetensors, weight_dtype "default"
    (NOT fp8_e4m3fn; NO control LoRA).
  - KSampler steps=50, cfg=4.0 (quality preset; lower steps for preview speed).
    The old v8 control values (7 / 0.8) are for control+LoRA and ruin base output.
  - EmptyQwenImageLayeredLatentImage widgets = [W, H, layers, 1]. The LAYER
    COUNT is widget index 2; index 3 (batch_size) stays 1. The old build put
    layers in index 3 while index 2 batch=4, multiplying outputs 4×.
  - Conditioning: TWO CLIPTextEncode (positive = agent prompt, negative = empty)
    each wrapped in its own ReferenceLatent fed by the SAME VAEEncode latent.
    Both positive and negative carry the image reference; they differ only in
    text. (Official uses this, NOT ConditioningZeroOut.)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT = PROJECT / "workflows/layered/base_layered.json"

BASE_UNET = "qwen_image_layered_fp8mixed.safetensors"
UNET_DTYPE = "default"
CLIP_NAME = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE_NAME = "qwen_image_layered_vae.safetensors"
# Model authors' TRUE settings (the official modelscope Space app.py), not the
# ComfyUI template's speed-preset (20 / 2.5 / 1024). The Space — which produces
# the best output — uses steps=50, true_cfg_scale=4.0, resolution=640.
#   - 640 is the model's RECOMMENDED bucket for this version; 1024 is the
#     "high-res" bucket with worse decomposition coherence. At 1024 the layered
#     decomposition collapses entirely and EVERY decoded layer comes out fully
#     transparent (alpha=0) — confirmed regression 2026-06-17. We scale the
#     input longest side to MAX_DIM so generation runs on the 640 bucket.
#     DO NOT raise this to 1024.
#   - steps=50 / cfg=4.0 roughly double generation time vs the 20/2.5 preset.
# To trade quality for speed, lower KSAMPLER_STEPS (e.g. 30) — it is the
# biggest single quality lever.
MAX_DIM = 640
KSAMPLER_STEPS = 50
KSAMPLER_CFG = 4.0
# fp8mixed (downloaded) "maintains good quality" per ComfyUI docs and is the
# main reason the workflow runs on modest VRAM. For the final marginal quality
# bump, download qwen_image_layered_bf16.safetensors and set BASE_UNET to it
# (needs high VRAM). Quantization is NOT the dominant quality factor — params
# are (see docs/plans/2026-06-07-base-layered-workflow-design.md).
DEFAULT_LAYERS = 4        # agent overrides node `layers` widget (index 2)
DEFAULT_PROMPT = ""       # agent fills positive whole-image description
# Post-processing disabled by default — net-negative on base output (recolors
# thin line-art). See the Output section below. Flip to True to re-attach the
# VR_PipelineLayered tail.
ENABLE_POSTPROCESS = False
# VR_PipelineLayered widgets [median_ksize, morph_ksize, alpha_min_area,
# sharpen_strength]. Line-art-safe: median/morph OFF (=1), tiny min_area that
# only drops isolated speckle dots — aggressive cleanup ate fine line-art
# (eyes/whiskers/outlines) from the base model's native alpha.
LAYERED_MEDIAN_KSIZE = 1
LAYERED_MORPH_KSIZE = 1
LAYERED_ALPHA_MIN_AREA = 16
LAYERED_SHARPEN_STRENGTH = 0.9


class Graph:
    def __init__(self):
        self.nodes = []
        self.links = []
        self._nid = 0
        self._lid = 0
        self._by_id = {}

    def node(self, ntype, *, title, pos, inputs=None, outputs=None, widgets=None):
        self._nid += 1
        n = {
            "id": self._nid, "type": ntype, "pos": pos, "size": [300, 110],
            "flags": {}, "order": 0, "mode": 0, "title": title,
            "inputs": inputs or [], "outputs": outputs or [],
            "properties": {"Node name for S&R": ntype}, "widgets_values": widgets or [],
        }
        self.nodes.append(n)
        self._by_id[self._nid] = n
        return self._nid

    def link(self, s, ss, d, ds, t):
        self._lid += 1
        self.links.append([self._lid, s, ss, d, ds, t])
        out = self._by_id[s]["outputs"][ss]
        out.setdefault("links", [])
        out["links"].append(self._lid)
        self._by_id[d]["inputs"][ds]["link"] = self._lid
        return self._lid

    def dump(self, path):
        g = {
            "nodes": self.nodes, "links": self.links, "groups": [],
            "config": {}, "extra": {}, "version": 0.4,
            "last_node_id": self._nid, "last_link_id": self._lid,
        }
        path.write_text(json.dumps(g, ensure_ascii=False, indent=2))


def _in(name, t):
    return {"name": name, "type": t, "link": None}


def _out(name, t):
    return {"name": name, "type": t, "links": []}


def main():
    g = Graph()

    # ── Loaders ──
    unet = g.node("UNETLoader", title="Load Diffusion Model (base layered)",
                  pos=[-1200, 0], outputs=[_out("MODEL", "MODEL")],
                  widgets=[BASE_UNET, UNET_DTYPE])
    clip = g.node("CLIPLoader", title="Load CLIP", pos=[-1200, 160],
                  outputs=[_out("CLIP", "CLIP")],
                  widgets=[CLIP_NAME, "qwen_image", "default"])
    vae = g.node("VAELoader", title="Load VAE", pos=[-1200, 320],
                 outputs=[_out("VAE", "VAE")], widgets=[VAE_NAME])

    # ── Input image → scale → GetImageSize + VAEEncode ──
    load = g.node("LoadImage", title="Load Image", pos=[-1200, 480],
                  outputs=[_out("IMAGE", "IMAGE"), _out("MASK", "MASK")],
                  widgets=["input.png", "image"])
    scale = g.node("ImageScaleToMaxDimension", title="Scale ≤ max dim",
                   pos=[-880, 480], inputs=[_in("image", "IMAGE")],
                   outputs=[_out("IMAGE", "IMAGE")], widgets=["lanczos", MAX_DIM])
    g.link(load, 0, scale, 0, "IMAGE")

    imgsize = g.node("GetImageSize", title="Get Image Size", pos=[-560, 360],
                     inputs=[_in("image", "IMAGE")],
                     outputs=[_out("width", "INT"), _out("height", "INT"),
                              _out("batch_size", "INT")])
    g.link(scale, 0, imgsize, 0, "IMAGE")

    encode = g.node("VAEEncode", title="VAE Encode (input image)", pos=[-560, 540],
                    inputs=[_in("pixels", "IMAGE"), _in("vae", "VAE")],
                    outputs=[_out("LATENT", "LATENT")])
    g.link(scale, 0, encode, 0, "IMAGE")
    g.link(vae, 0, encode, 1, "VAE")

    # ── Conditioning: positive (agent prompt) + negative (empty), each wrapped
    #    in a ReferenceLatent fed by the same encoded image. ──
    pos_text = g.node("CLIPTextEncode", title="Positive (whole-image prompt)",
                      pos=[-880, 0], inputs=[_in("clip", "CLIP")],
                      outputs=[_out("CONDITIONING", "CONDITIONING")],
                      widgets=[DEFAULT_PROMPT])
    # Official Space uses neg_prompt=" " (single space), not empty string.
    neg_text = g.node("CLIPTextEncode", title="Negative (\" \")", pos=[-880, 200],
                      inputs=[_in("clip", "CLIP")],
                      outputs=[_out("CONDITIONING", "CONDITIONING")], widgets=[" "])
    g.link(clip, 0, pos_text, 0, "CLIP")
    g.link(clip, 0, neg_text, 0, "CLIP")

    pos_ref = g.node("ReferenceLatent", title="Ref Latent (positive)", pos=[-560, 0],
                     inputs=[_in("conditioning", "CONDITIONING"), _in("latent", "LATENT")],
                     outputs=[_out("CONDITIONING", "CONDITIONING")])
    neg_ref = g.node("ReferenceLatent", title="Ref Latent (negative)", pos=[-560, 200],
                     inputs=[_in("conditioning", "CONDITIONING"), _in("latent", "LATENT")],
                     outputs=[_out("CONDITIONING", "CONDITIONING")])
    g.link(pos_text, 0, pos_ref, 0, "CONDITIONING")
    g.link(encode, 0, pos_ref, 1, "LATENT")
    g.link(neg_text, 0, neg_ref, 0, "CONDITIONING")
    g.link(encode, 0, neg_ref, 1, "LATENT")

    # ── Model sampling + empty layered latent ──
    modelsampling = g.node("ModelSamplingAuraFlow", title="ModelSamplingAuraFlow",
                           pos=[-880, 360], inputs=[_in("model", "MODEL")],
                           outputs=[_out("MODEL", "MODEL")], widgets=[1])
    g.link(unet, 0, modelsampling, 0, "MODEL")

    # widgets = [width_placeholder, height_placeholder, layers, batch_size].
    # LAYER COUNT is index 2; batch_size index 3 stays 1. width/height are
    # input-driven from GetImageSize so the placeholders are overridden.
    empty = g.node("EmptyQwenImageLayeredLatentImage", title="Empty Layered Latent",
                   pos=[-240, 360],
                   inputs=[_in("width", "INT"), _in("height", "INT")],
                   outputs=[_out("LATENT", "LATENT")],
                   widgets=[MAX_DIM, MAX_DIM, DEFAULT_LAYERS, 1])
    g.link(imgsize, 0, empty, 0, "INT")
    g.link(imgsize, 1, empty, 1, "INT")

    # ── Sample → cut layered latent to batch → decode ──
    ksampler = g.node("KSampler", title="KSampler (steps=50, cfg=4.0)", pos=[-240, 0],
                      inputs=[_in("model", "MODEL"), _in("positive", "CONDITIONING"),
                              _in("negative", "CONDITIONING"), _in("latent_image", "LATENT")],
                      outputs=[_out("LATENT", "LATENT")],
                      widgets=[0, "randomize", KSAMPLER_STEPS, KSAMPLER_CFG,
                               "euler", "simple", 1])
    g.link(modelsampling, 0, ksampler, 0, "MODEL")
    g.link(pos_ref, 0, ksampler, 1, "CONDITIONING")   # positive
    g.link(neg_ref, 0, ksampler, 2, "CONDITIONING")   # negative
    g.link(empty, 0, ksampler, 3, "LATENT")

    cut = g.node("LatentCutToBatch", title="Latent Cut To Batch", pos=[80, 0],
                 inputs=[_in("samples", "LATENT")],
                 outputs=[_out("LATENT", "LATENT")], widgets=["t", 1])
    g.link(ksampler, 0, cut, 0, "LATENT")

    decode = g.node("VAEDecode", title="VAE Decode (N×RGBA)", pos=[400, 0],
                    inputs=[_in("samples", "LATENT"), _in("vae", "VAE")],
                    outputs=[_out("IMAGE", "IMAGE")])
    g.link(cut, 0, decode, 0, "LATENT")
    g.link(vae, 0, decode, 1, "VAE")

    # ── Output ──────────────────────────────────────────────────────────
    # Post-processing is DISABLED by default. Empirically (2026-06-07) the
    # VR_PipelineLayered tail was net-negative on base output: edge_color_inpaint
    # rewrites thin line-art color (a 1-2px line has no "interior" to inpaint
    # from, so the whole line gets recolored) and ROI unsharp overshoots edge
    # color. The official modelscope Space — the quality ceiling — does ZERO
    # post-processing; the base model's native RGBA is already clean at the
    # correct sampling settings. So we save the raw VAEDecode RGBA directly,
    # exactly like the official ComfyUI template.
    #
    # Set ENABLE_POSTPROCESS=True to re-attach the VR_PipelineLayered tail (and
    # an extra raw "layer_raw_" save for A/B comparison) if a future use case
    # (e.g. downstream vectorization) needs the cleanup despite the color cost.
    if not ENABLE_POSTPROCESS:
        preview = g.node("PreviewImage", title="Preview Layers (raw RGBA)",
                         pos=[720, -160], inputs=[_in("images", "IMAGE")])
        g.link(decode, 0, preview, 0, "IMAGE")

        save = g.node("SaveImage", title="Save (layer_00..NN, raw model output)",
                      pos=[720, 0], inputs=[_in("images", "IMAGE")], widgets=["layer_"])
        g.link(decode, 0, save, 0, "IMAGE")
    else:
        save_raw = g.node("SaveImage", title="Save RAW (before post-proc)",
                          pos=[720, -260], inputs=[_in("images", "IMAGE")],
                          widgets=["layer_raw_"])
        g.link(decode, 0, save_raw, 0, "IMAGE")

        layered = g.node("VR_PipelineLayered",
                         title="✨ VectorReady · Layered (base multi-layer)", pos=[720, 0],
                         inputs=[_in("image", "IMAGE")],
                         outputs=[_out("image", "IMAGE"), _out("alpha", "MASK")],
                         widgets=[LAYERED_MEDIAN_KSIZE, LAYERED_MORPH_KSIZE,
                                  LAYERED_ALPHA_MIN_AREA, LAYERED_SHARPEN_STRENGTH])
        g.link(decode, 0, layered, 0, "IMAGE")

        join = g.node("VR_JoinRGBA", title="🧷 RGBA Join (layers)", pos=[1040, 0],
                      inputs=[_in("image", "IMAGE"), _in("alpha", "MASK")],
                      outputs=[_out("rgba", "IMAGE")])
        g.link(layered, 0, join, 0, "IMAGE")
        g.link(layered, 1, join, 1, "MASK")

        save = g.node("SaveImage", title="Save (layer_00..NN, processed)", pos=[1360, 0],
                      inputs=[_in("images", "IMAGE")], widgets=["layer_"])
        g.link(join, 0, save, 0, "IMAGE")

    g.node("MarkdownNote", title="📘 base_layered 说明", pos=[720, 260], widgets=[
        "## base 全量分层 (Qwen-Image-Layered 官方多图层)\n\n"
        "照搬 ComfyUI 官方 'Image to Layers' 模板 + 模型真实采样参数 (Space app.py).\n\n"
        "**Agent 旋钮**:\n"
        "- 节点 `Empty Layered Latent` widget[2] = `layers` 层数 (默认 4)\n"
        "- 节点 `Positive` = 整图全局描述 (英文; 风格指南见分层 agent)\n\n"
        "**关键参数 (勿改)**: UNET=base/default dtype, KSampler steps=50 cfg=4.0, "
        "scale 640, 正/负各一个 ReferenceLatent (负=\" \"), layers 在 widget[2] / batch=1.\n\n"
        "**后处理: 默认关闭** (与官方一致). 实测对细线条变色, 净负向. "
        "如需可在 build 脚本设 ENABLE_POSTPROCESS=True.\n\n"
        "**输出**: layer_00..layer_NN 原始 RGBA (batch 序 = z-order, 00=最底层)"
    ])

    g.dump(OUT)
    print(f"wrote {OUT} — {len(g.nodes)} nodes, {len(g.links)} links")


if __name__ == "__main__":
    sys.exit(main())
