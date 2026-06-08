"""Build workflows/inpaint/ip_consistent.json from the autodetect-only base.

Injects, on top of the converted autodetect graph:
  - an entry VR_RequestBanner that routes this workflow's logs to a dedicated
    file (vr_ip_consistent.log), inserted in the main image path so it always
    executes;
  - an alpha-mask branch for transparent IP layers (protection mask = the
    layer's own alpha; condition image = IP composited over a white plate;
    a duplicated sampler tail);
  - two VR_GatedPassthrough latent gates (one per branch) so the unselected
    branch is pruned via ExecutionBlocker — flipping the two `enable` widgets
    selects the path and prunes the heavy SAM3 chain when alpha is used.

The lean production graph carries NO preview/probe nodes (the base's inherited
previews are stripped); patch_ip_to_debug.py re-adds per-stage previews+probes.

Duplicated tail nodes are CLONED from the canonical base nodes (preserving their
widgets_values, incl. KSampler's control_after_generate slot) rather than hand
-authored. Only genuinely-new node types are built from scratch.

Run: python scripts/build_ip_consistent.py
"""
from __future__ import annotations

from pathlib import Path

import _uigraph as u

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "workflows/inpaint/ip_consistent_base.json"
OUT = ROOT / "workflows/inpaint/ip_consistent.json"

# Canonical base node ids (preserved from the API source through conversion).
LOAD = 1            # LoadImage            → IMAGE@0, MASK@1
SCALE_MAX = 5       # ImageScaleToMaxDimension (lanczos, 1024)
WORK = 301          # FluxKontextImageScale  (work image)
PROTECT_AUTO = 232  # VR_MaskSubtract        (autodetect protection mask, out "mask")
VAE = 304
CLIP = 303
MODEL = 307         # CFGNorm (model feeding KSampler)
PROMPT = 308        # PrimitiveStringMultiline (positive design prompt)
ENC_POS = 309       # TextEncodeQwenImageEditPlus (positive)
ENC_NEG = 310       # TextEncodeQwenImageEditPlus (negative, prompt widget "")
VAENC = 311         # VAEEncode
GROW = 318          # GrowMask (expand 6)
SETMASK = 314       # SetLatentNoiseMask
KSAMPLER = 312
VDECODE = 315
COMPOSITE = 316     # ImageCompositeMasked (paste original pixels back)
SAVE = 317          # SaveImage
INHERITED_PREVIEWS = (319, 320, 321)  # PreviewImage + 2× MaskPreview+ → strip


def _new_invert(g, src_id, src_out, *, pos, title):
    nid = u.add_node(g, ntype="InvertMask", title=title, pos=pos,
                     inputs=[{"name": "mask", "type": "MASK", "link": None}],
                     outputs=[{"name": "MASK", "type": "MASK", "links": []}])
    u.add_link(g, src_id, src_out, nid, "mask", "MASK")
    return nid


def main():
    g = u.load(BASE)

    # 1) strip inherited preview nodes → lean production
    for nid in INHERITED_PREVIEWS:
        try:
            u.remove_node(g, nid)
        except StopIteration:
            pass

    # 2) entry banner in the main image path (always executes) → dedicated log
    banner = u.add_node(
        g, ntype="VR_RequestBanner", title="🚩 入口 / 独立日志", pos=[-500, -200],
        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
        outputs=[{"name": "image", "type": "IMAGE", "links": []},
                 {"name": "request_id", "type": "STRING", "links": []}],
        widgets=["ip_consistent", "", "vr_ip_consistent.log"])  # tag, request_id, log_file
    u.replace_input_link(g, SCALE_MAX, "image", banner, "image", "IMAGE")
    u.add_link(g, LOAD, "IMAGE", banner, "image", "IMAGE")

    # 3) autodetect gate on base KSampler.latent_image (enable widget, default ON)
    gate_auto = u.add_node(
        g, ntype="VR_GatedPassthrough", title="🔀 门·autodetect (enable=ON 走检测)",
        pos=[2400, 200],
        inputs=[{"name": "value", "type": "LATENT", "link": None}],
        outputs=[{"name": "value", "type": "LATENT", "links": []}],
        widgets=[True, False, "autodetect"])  # enable, invert, label
    u.replace_input_link(g, KSAMPLER, "latent_image", gate_auto, "value", "LATENT")
    u.add_link(g, SETMASK, "LATENT", gate_auto, "value", "LATENT")

    # ───── 4) alpha branch ───────────────────────────────────────────────────
    Y = 1400
    # 4a) protection mask = layer alpha (LoadImage.MASK = 1-alpha → invert = alpha)
    inv_protect = _new_invert(g, LOAD, "MASK", pos=[-200, Y],
                              title="alpha→保护蒙版(IP=白)")
    # 4b) resample the protection mask to work-image size by mirroring the image
    #     scaling chain (MaskToImage → ImageScaleToMaxDimension → FluxKontext → ImageToMask)
    m2i = u.add_node(g, ntype="MaskToImage", title="alpha→IMAGE", pos=[100, Y],
                     inputs=[{"name": "mask", "type": "MASK", "link": None}],
                     outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}])
    u.add_link(g, inv_protect, "MASK", m2i, "mask", "MASK")
    scale1 = u.clone_node(g, SCALE_MAX, pos=[400, Y], title="缩放(镜像图像链)")
    u.add_link(g, m2i, "IMAGE", scale1, "image", "IMAGE")
    scale2 = u.clone_node(g, WORK, pos=[700, Y], title="Qwen对齐(镜像)")
    u.add_link(g, scale1, "IMAGE", scale2, "image", "IMAGE")
    i2m = u.add_node(g, ntype="ImageToMask", title="IMAGE→保护蒙版(work尺寸)",
                     pos=[1000, Y],
                     inputs=[{"name": "image", "type": "IMAGE", "link": None}],
                     outputs=[{"name": "MASK", "type": "MASK", "links": []}],
                     widgets=["red"])
    u.add_link(g, scale2, "IMAGE", i2m, "image", "IMAGE")
    protect_work = (i2m, "MASK")
    # 4c) editable region = invert(protection) then grow (mirror base 318)
    editable = _new_invert(g, i2m, "MASK", pos=[1300, Y], title="可编辑区(透明=白)")
    grow = u.clone_node(g, GROW, pos=[1600, Y], title="可编辑区膨胀")
    u.add_link(g, editable, "MASK", grow, "mask", "MASK")

    # 4d) white plate sized to work image, then IP-over-white condition image
    getsize = u.add_node(
        g, ntype="GetImageSize+", title="取work尺寸", pos=[400, Y + 250],
        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
        outputs=[{"name": "width", "type": "INT", "links": []},
                 {"name": "height", "type": "INT", "links": []},
                 {"name": "count", "type": "INT", "links": []}])
    u.add_link(g, WORK, "IMAGE", getsize, "image", "IMAGE")
    white = u.add_node(
        g, ntype="EmptyImage", title="白底(work尺寸)", pos=[700, Y + 250],
        inputs=[{"name": "width", "type": "INT", "link": None},
                {"name": "height", "type": "INT", "link": None}],
        outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
        widgets=[1, 0xFFFFFF])  # batch_size, color=white
    u.add_link(g, getsize, "width", white, "width", "INT")
    u.add_link(g, getsize, "height", white, "height", "INT")
    cond = u.add_node(
        g, ntype="ImageCompositeMasked", title="IP叠白底(条件图)", pos=[1000, Y + 250],
        inputs=[{"name": "destination", "type": "IMAGE", "link": None},
                {"name": "source", "type": "IMAGE", "link": None},
                {"name": "mask", "type": "MASK", "link": None}],
        outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
        widgets=[0, 0, False])  # x, y, resize_source
    u.add_link(g, white, "IMAGE", cond, "destination", "IMAGE")
    u.add_link(g, WORK, "IMAGE", cond, "source", "IMAGE")
    u.add_link(g, i2m, "MASK", cond, "mask", "MASK")

    # 4e) duplicated sampler tail (cloned canonical nodes)
    venc = u.clone_node(g, VAENC, pos=[1300, Y + 250], title="alpha 原图编码")
    u.add_link(g, cond, "IMAGE", venc, "pixels", "IMAGE")
    u.add_link(g, VAE, "VAE", venc, "vae", "VAE")
    pos_enc = u.clone_node(g, ENC_POS, pos=[1300, Y + 500], title="alpha 正向编码")
    u.add_link(g, PROMPT, "STRING", pos_enc, "prompt", "STRING")
    u.add_link(g, CLIP, "CLIP", pos_enc, "clip", "CLIP")
    u.add_link(g, VAE, "VAE", pos_enc, "vae", "VAE")
    u.add_link(g, cond, "IMAGE", pos_enc, "image1", "IMAGE")
    neg_enc = u.clone_node(g, ENC_NEG, pos=[1300, Y + 750], title="alpha 负向编码(空)")
    u.add_link(g, CLIP, "CLIP", neg_enc, "clip", "CLIP")
    u.add_link(g, VAE, "VAE", neg_enc, "vae", "VAE")
    u.add_link(g, cond, "IMAGE", neg_enc, "image1", "IMAGE")
    setmask = u.clone_node(g, SETMASK, pos=[1900, Y + 250], title="alpha 仅可编辑区加噪")
    u.add_link(g, venc, "LATENT", setmask, "samples", "LATENT")
    u.add_link(g, grow, "MASK", setmask, "mask", "MASK")
    gate_alpha = u.add_node(
        g, ntype="VR_GatedPassthrough", title="🔀 门·alpha (enable=ON 走alpha)",
        pos=[2200, Y + 250],
        inputs=[{"name": "value", "type": "LATENT", "link": None}],
        outputs=[{"name": "value", "type": "LATENT", "links": []}],
        widgets=[False, False, "alpha"])  # enable, invert, label (default OFF)
    u.add_link(g, setmask, "LATENT", gate_alpha, "value", "LATENT")
    ks = u.clone_node(g, KSAMPLER, pos=[2500, Y + 250], title="alpha KSampler")
    u.add_link(g, MODEL, "MODEL", ks, "model", "MODEL")
    u.add_link(g, pos_enc, "CONDITIONING", ks, "positive", "CONDITIONING")
    u.add_link(g, neg_enc, "CONDITIONING", ks, "negative", "CONDITIONING")
    u.add_link(g, gate_alpha, "value", ks, "latent_image", "LATENT")
    vdec = u.clone_node(g, VDECODE, pos=[2800, Y + 250], title="alpha VAE解码")
    u.add_link(g, ks, "LATENT", vdec, "samples", "LATENT")
    u.add_link(g, VAE, "VAE", vdec, "vae", "VAE")
    comp = u.clone_node(g, COMPOSITE, pos=[3100, Y + 250], title="alpha 原像素盖回")
    u.add_link(g, vdec, "IMAGE", comp, "destination", "IMAGE")
    u.add_link(g, WORK, "IMAGE", comp, "source", "IMAGE")
    u.add_link(g, i2m, "MASK", comp, "mask", "MASK")
    save = u.clone_node(g, SAVE, pos=[3400, Y + 250], title="📤 保存 (alpha)")
    save_node = u.find_by_id(g, save)
    save_node["widgets_values"] = ["ip_consistent_alpha"]
    u.add_link(g, comp, "IMAGE", save, "images", "IMAGE")

    u.assert_graph_valid(g)
    u.dump(g, OUT)
    print(f"wrote {OUT} ({len(g['nodes'])} nodes, {len(g['links'])} links)")


if __name__ == "__main__":
    main()
