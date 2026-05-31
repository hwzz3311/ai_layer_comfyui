"""Build qwen_layered_v8_ab_vector_ready.json from v7.

Changes vs v7:
- A/B真正分流: insert VR_GatedPassthrough on each KSampler's latent_image input,
  driven by the foreground_mode boolean (node 215). The unselected branch's
  KSampler receives ExecutionBlocker and the entire downstream chain is pruned.
- VectorReady tails: VR_PipelineLight on A path, VR_PipelineStrong on B path,
  between VAEDecode and SaveImage. Alpha source is the existing SAM3 mask
  (node 20) for A and the InvertMask (node 205) for B — placeholder until
  v8.1 brings ViTMatte/RMBG.
- Final SaveImage receives RGBA via VR_JoinRGBA (opacity convention + final
  transparent-region RGB/alpha clamp).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
V7 = PROJECT / "qwen_layered_v7_ab_dual_path.json"
V8 = PROJECT / "qwen_layered_v8_ab_vector_ready.json"
DEFAULT_RMBG_MODEL_PATH = "/root/ComfyUI/models/RMBG-2.0"
DEFAULT_LOCATE_MODEL_ID = "/root/ComfyUI/models/LocateAnything-3B"
DEFAULT_LOCATE_QUERY = "main target object"

# Width (in pixels) of the "unknown" black band between red positive and green
# negative brush regions sent to V2. v7 inherited 48, which leaves giant holes
# in the brush conditioning. ~18 keeps enough uncertainty zone for V2's inpaint
# behavior without swallowing the cat outline.
GROW_MASK_PX = 18

# If SAM3 produces an unusable target mask, skip the red/green brush reference
# instead of feeding Qwen V2 a misleading empty brush. The Qwen sampler still
# receives text + original-image ReferenceLatent.
BRUSH_MASK_THRESHOLD = 0.5
BRUSH_MIN_AREA_RATIO = 0.002
BRUSH_MAX_AREA_RATIO = 0.90
BRUSH_MIN_AREA_PX = 64

RESOLVER_MASK_THRESHOLD = 0.5
RESOLVER_MIN_AREA_RATIO = 0.002
RESOLVER_MAX_AREA_RATIO = 0.90
RESOLVER_MIN_IOU_WITH_LOCATE = 0.02
RESOLVER_FALLBACK_DILATE_PX = 0

# Nodes whose hardcoded EmptyImage size (1024×1024) breaks non-square inputs by
# producing brush bases with wrong aspect ratio. Replace each with a size-matched
# VR_EmptyImageLike pulling its reference from the scaled input (node 5).
BRUSH_BASE_NODES = {
    201: (255, 0, 0),  # red positive base
    202: (0, 0, 0),    # black uncertainty base
    207: (0, 255, 0),  # green negative base
}
SCALED_INPUT_NODE = 5


def next_link_id(g):
    return max((l[0] for l in g["links"]), default=0) + 1


def next_node_id(g):
    return max((n["id"] for n in g["nodes"]), default=0) + 1


def find_node(g, nid):
    return next(n for n in g["nodes"] if n["id"] == nid)


def remove_link(g, link_id):
    g["links"] = [l for l in g["links"] if l[0] != link_id]
    for n in g["nodes"]:
        for sock in n.get("inputs", []):
            if sock.get("link") == link_id:
                sock["link"] = None
        for sock in n.get("outputs", []):
            if sock.get("links") and link_id in sock["links"]:
                sock["links"].remove(link_id)


def add_link(g, src_node, src_slot, dst_node, dst_slot, link_type):
    lid = next_link_id(g)
    g["links"].append([lid, src_node, src_slot, dst_node, dst_slot, link_type])
    # mirror into node sockets
    src = find_node(g, src_node)
    dst = find_node(g, dst_node)
    while len(src.get("outputs", [])) <= src_slot:
        return lid  # tolerate, caller responsible
    out = src["outputs"][src_slot]
    out.setdefault("links", [])
    if out["links"] is None:
        out["links"] = []
    out["links"].append(lid)
    inp = dst["inputs"][dst_slot]
    inp["link"] = lid
    return lid


def add_node(g, *, ntype, title, pos, inputs=None, outputs=None, widgets=None, props=None):
    nid = next_node_id(g)
    node = {
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": [320, 100],
        "flags": {},
        "order": 0,
        "mode": 0,
        "title": title,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": props or {"Node name for S&R": ntype},
        "widgets_values": widgets or [],
    }
    g["nodes"].append(node)
    return nid


def rewire_input(g, dst_node, dst_input_name, new_src_node, new_src_slot, link_type):
    dst = find_node(g, dst_node)
    dst_slot = next(i for i, sock in enumerate(dst.get("inputs", [])) if sock["name"] == dst_input_name)
    old_link = dst["inputs"][dst_slot].get("link")
    if old_link is not None:
        remove_link(g, old_link)
    return add_link(g, new_src_node, new_src_slot, dst_node, dst_slot, link_type)


def ensure_input(g, node_id, name, input_type):
    node = find_node(g, node_id)
    for idx, sock in enumerate(node.get("inputs", [])):
        if sock.get("name") == name:
            return idx
    node.setdefault("inputs", []).append({"name": name, "type": input_type, "link": None})
    return len(node["inputs"]) - 1


def main():
    g = json.loads(V7.read_text())

    # ─────────────── Stage 1: A/B switch ───────────────
    # Identify existing wiring: 55 (latent) → 60 (A KSampler).input[3], 55 → 210 (B).input[3]
    # 215 PrimitiveNode BOOLEAN (foreground_mode) — currently orphan.
    link_a_latent = None  # link from 55 to 60
    link_b_latent = None  # link from 55 to 210
    for l in g["links"]:
        if l[1] == 55 and l[3] == 60 and l[4] == 3:
            link_a_latent = l[0]
        if l[1] == 55 and l[3] == 210 and l[4] == 3:
            link_b_latent = l[0]
    assert link_a_latent and link_b_latent, "expected 55→60 and 55→210 latent links"
    remove_link(g, link_a_latent)
    remove_link(g, link_b_latent)

    # Create two VR_GatedPassthrough nodes.
    gate_a_id = add_node(
        g,
        ntype="VR_GatedPassthrough",
        title="🔀 Gate A (foreground_mode=true → pass)",
        pos=[2200, 100],
        inputs=[
            {"name": "value", "type": "LATENT", "link": None},
            {"name": "enable", "type": "BOOLEAN", "link": None, "widget": {"name": "enable"}},
        ],
        outputs=[{"name": "value", "type": "LATENT", "links": []}],
        widgets=[True, False],  # enable, invert
    )
    gate_b_id = add_node(
        g,
        ntype="VR_GatedPassthrough",
        title="🔀 Gate B (foreground_mode=false → pass)",
        pos=[2200, 320],
        inputs=[
            {"name": "value", "type": "LATENT", "link": None},
            {"name": "enable", "type": "BOOLEAN", "link": None, "widget": {"name": "enable"}},
        ],
        outputs=[{"name": "value", "type": "LATENT", "links": []}],
        widgets=[True, True],  # enable, invert=true
    )

    # Rewire: 55.LATENT → gate_a.value → 60.latent_image
    add_link(g, 55, 0, gate_a_id, 0, "LATENT")
    add_link(g, gate_a_id, 0, 60, 3, "LATENT")
    # 55.LATENT → gate_b.value → 210.latent_image
    add_link(g, 55, 0, gate_b_id, 0, "LATENT")
    add_link(g, gate_b_id, 0, 210, 3, "LATENT")

    # 215.BOOLEAN → gate_a.enable, gate_b.enable
    n215 = find_node(g, 215)
    if not n215.get("outputs"):
        n215["outputs"] = [{"name": "BOOLEAN", "type": "BOOLEAN", "links": []}]
    # convert widget enable inputs to actual sockets (already added above)
    add_link(g, 215, 0, gate_a_id, 1, "BOOLEAN")
    add_link(g, 215, 0, gate_b_id, 1, "BOOLEAN")

    # ─────────────── Stage 2: LocateAnything spatial fallback ───────────────
    # LocateAnything gives a robust coarse box for the target. SAM3 remains the
    # preferred precise mask: LocateAnything feeds SAM3's BBOX input first.
    # If SAM/MaskFix is still empty or badly misaligned, VR_TargetMaskResolver
    # falls back to the LocateAnything rectangle so Qwen V2 still receives a
    # spatial brush instead of blindly relying on text.
    locate_id = add_node(
        g,
        ntype="VR_LocateAnythingBox",
        title="📍 LocateAnything · Target Box",
        pos=[840, 430],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
        ],
        outputs=[
            {"name": "box_mask", "type": "MASK", "links": []},
            {"name": "preview_image", "type": "IMAGE", "links": []},
            {"name": "bbox_json", "type": "STRING", "links": []},
            {"name": "box_usable", "type": "BOOLEAN", "links": []},
            {"name": "bboxes", "type": "BBOX", "links": []},
        ],
        # query, model_id, device, generation_mode, prompt_mode, padding_px,
        # max_new_tokens, temperature.
        widgets=[
            DEFAULT_LOCATE_QUERY,
            DEFAULT_LOCATE_MODEL_ID,
            "auto",
            "hybrid",
            "single",
            8,
            2048,
            0.7,
        ],
    )
    add_link(g, SCALED_INPUT_NODE, 0, locate_id, 0, "IMAGE")

    # Easy-SAM3 only applies geometric bbox prompts when its text prompt is
    # empty. The semantic target is now expressed through LocateAnything.query;
    # SAM3 receives LocateAnything's box as the precise segmentation prompt.
    sam3_node = find_node(g, 11)
    if sam3_node.get("widgets_values"):
        sam3_node["widgets_values"][0] = ""
    sam3_bbox_slot = ensure_input(g, 11, "bboxes", "BBOX")
    add_link(g, locate_id, 4, 11, sam3_bbox_slot, "BBOX")

    locate_preview_id = add_node(
        g,
        ntype="PreviewImage",
        title="🔍 [诊断10] LocateAnything 矩形框",
        pos=[1180, 430],
        inputs=[{"name": "images", "type": "IMAGE", "link": None}],
    )
    add_link(g, locate_id, 1, locate_preview_id, 0, "IMAGE")

    resolver_id = add_node(
        g,
        ntype="VR_TargetMaskResolver",
        title="🎯 Target Mask Resolver (SAM优先/矩形兜底)",
        pos=[1180, 700],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "sam_mask", "type": "MASK", "link": None},
            {"name": "fallback_mask", "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "resolved_mask", "type": "MASK", "links": []},
            {"name": "quality_preview", "type": "IMAGE", "links": []},
            {"name": "sam_usable", "type": "BOOLEAN", "links": []},
            {"name": "fallback_used", "type": "BOOLEAN", "links": []},
        ],
        # threshold, min_area_ratio, max_area_ratio, min_iou_with_fallback,
        # fallback_dilate_px, keep_largest_component.
        widgets=[
            RESOLVER_MASK_THRESHOLD,
            RESOLVER_MIN_AREA_RATIO,
            RESOLVER_MAX_AREA_RATIO,
            RESOLVER_MIN_IOU_WITH_LOCATE,
            RESOLVER_FALLBACK_DILATE_PX,
            True,
        ],
    )
    add_link(g, SCALED_INPUT_NODE, 0, resolver_id, 0, "IMAGE")
    add_link(g, 20, 0, resolver_id, 1, "MASK")
    add_link(g, locate_id, 0, resolver_id, 2, "MASK")

    resolver_preview_id = add_node(
        g,
        ntype="PreviewImage",
        title="🔍 [诊断11] Resolver 最终目标 mask",
        pos=[1540, 700],
        inputs=[{"name": "images", "type": "IMAGE", "link": None}],
    )
    add_link(g, resolver_id, 1, resolver_preview_id, 0, "IMAGE")

    # Existing brush construction should consume the resolved mask. Keep the
    # old MaskFix preview wired to node 20 for diagnostics.
    rewire_input(g, 203, "mask", resolver_id, 0, "MASK")
    rewire_input(g, 204, "mask", resolver_id, 0, "MASK")

    # ─────────────── Stage 4: VectorReady tails ───────────────
    # A path: 62 VAEDecode.IMAGE + resolver target MASK + 5 scaled original IMAGE
    #         → VR_PipelineLight → VR_JoinRGBA → 63 SaveImage
    # B path: 212 VAEDecode.IMAGE + 205 InvertMask.MASK → VR_PipelineStrong → VR_JoinRGBA → 214 SaveImage

    # Disconnect existing 62→63 and 212→214
    link_a_save = next((l[0] for l in g["links"] if l[1] == 62 and l[3] == 63), None)
    link_b_save = next((l[0] for l in g["links"] if l[1] == 212 and l[3] == 214), None)
    assert link_a_save and link_b_save
    remove_link(g, link_a_save)
    remove_link(g, link_b_save)

    # A-path real matting alpha baseline. The node is intentionally separate
    # from VR_PipelineLight so model failures are isolated and the bridge can be
    # fed by another matting model later.
    hf_matte_id = add_node(
        g,
        ntype="VR_HFMattingAlpha",
        title="🧠 HF Matting Alpha (A path)",
        pos=[3100, -120],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "candidate_mask", "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "matte_alpha", "type": "MASK", "links": []},
            {"name": "confidence", "type": "MASK", "links": []},
        ],
        # model_id, input_size, device.
        widgets=[DEFAULT_RMBG_MODEL_PATH, 1024, "auto"],
    )
    add_link(g, SCALED_INPUT_NODE, 0, hf_matte_id, 0, "IMAGE")
    add_link(g, resolver_id, 0, hf_matte_id, 1, "MASK")

    # A pipeline node
    vr_light_id = add_node(
        g,
        ntype="VR_PipelineLight",
        title="✨ VectorReady · Light (A path)",
        pos=[3400, 100],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "alpha", "type": "MASK", "link": None},
            {"name": "source_image", "type": "IMAGE", "link": None},
            {"name": "external_matte_alpha", "type": "MASK", "link": None},
            {"name": "external_confidence", "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "image", "type": "IMAGE", "links": []},
            {"name": "alpha", "type": "MASK", "links": []},
        ],
        # palette_k, alpha_steps, alpha_min_area, alpha_source, matting_backend.
        # palette_k=0 puts A path in fidelity mode: no bilateral smoothing and
        # no color quantization; foreground RGB is trusted.
        # source_image is the scaled original image (node 5). It is used only
        # as a gated detail source, never as a whole-mask overwrite.
        # alpha_source="mask_socket" forces using the SAM3 mask wired into the
        # MASK input — Qwen-Image-Layered's native alpha marks "where white was
        # painted", which kills line-art detail (eyes/whiskers) if used as a mask.
        widgets=[0, 3, 1500, "mask_socket", "external_matte"],
    )
    add_link(g, 62, 0, vr_light_id, 0, "IMAGE")
    add_link(g, resolver_id, 0, vr_light_id, 1, "MASK")
    add_link(g, SCALED_INPUT_NODE, 0, vr_light_id, 2, "IMAGE")
    add_link(g, hf_matte_id, 0, vr_light_id, 3, "MASK")
    add_link(g, hf_matte_id, 1, vr_light_id, 4, "MASK")

    # B pipeline node
    vr_strong_id = add_node(
        g,
        ntype="VR_PipelineStrong",
        title="✨ VectorReady · Strong (B path)",
        pos=[3400, 360],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "alpha", "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "image", "type": "IMAGE", "links": []},
            {"name": "alpha", "type": "MASK", "links": []},
        ],
        # max_k, delta_e, alpha_steps, alpha_min_area, alpha_source.
        # B path keeps "auto" — background reconstruction Qwen outputs typically
        # have correctly-shaped alpha (whole bg region). Flip to "mask_socket"
        # if the InvertMask source proves more reliable.
        widgets=[12, 6.0, 3, 1500, "auto"],
    )
    add_link(g, 212, 0, vr_strong_id, 0, "IMAGE")
    add_link(g, 205, 0, vr_strong_id, 1, "MASK")

    # VR_JoinRGBA (opacity convention — alpha=1 means opaque, no inversion).
    # ComfyUI core's JoinImageWithAlpha writes (1 - alpha) into the alpha
    # channel, which would invert the SAM3 mask and render cat regions as
    # holes. VR_JoinRGBA also clamps final transparent regions to RGBA=0 so
    # decoder speckles cannot survive in invisible pixels.
    join_a_id = add_node(
        g,
        ntype="VR_JoinRGBA",
        title="🧷 RGBA Join (A)",
        pos=[3800, 100],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "alpha", "type": "MASK", "link": None},
        ],
        outputs=[{"name": "rgba", "type": "IMAGE", "links": []}],
    )
    join_b_id = add_node(
        g,
        ntype="VR_JoinRGBA",
        title="🧷 RGBA Join (B)",
        pos=[3800, 360],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "alpha", "type": "MASK", "link": None},
        ],
        outputs=[{"name": "rgba", "type": "IMAGE", "links": []}],
    )
    add_link(g, vr_light_id, 0, join_a_id, 0, "IMAGE")
    add_link(g, vr_light_id, 1, join_a_id, 1, "MASK")
    add_link(g, vr_strong_id, 0, join_b_id, 0, "IMAGE")
    add_link(g, vr_strong_id, 1, join_b_id, 1, "MASK")

    # Wire back to SaveImage
    add_link(g, join_a_id, 0, 63, 0, "IMAGE")
    add_link(g, join_b_id, 0, 214, 0, "IMAGE")

    # Update SaveImage filename prefix so A/B outputs are distinguishable
    s63 = find_node(g, 63)
    if s63.get("widgets_values"):
        s63["widgets_values"][0] = "v8_A_foreground_RGBA"
    s214 = find_node(g, 214)
    if s214.get("widgets_values"):
        s214["widgets_values"][0] = "v8_B_background_RGBA"

    # Bookkeeping: add a markdown note
    add_node(
        g,
        ntype="MarkdownNote",
        title="📘 v8.0 说明",
        pos=[2200, 600],
        widgets=[
            "## v8.0 · A/B 真正分流 + VectorReady\n\n"
            "**A/B 切换**: 节点 215 (foreground_mode) → 两个 VR_GatedPassthrough\n"
            "- Gate A: enable=215, invert=false → 只在 true 时放行\n"
            "- Gate B: enable=215, invert=true  → 只在 false 时放行\n"
            "未选中的分支会因为 ExecutionBlocker 被自动跳过整条管线\n"
            "(KSampler + VAEDecode + VectorReady + SaveImage 全部不执行)\n\n"
            "**VectorReady**:\n"
            "- A 路径: VR_PipelineLight (alpha 源 = TargetMaskResolver)\n"
            "- B 路径: VR_PipelineStrong (alpha 源 = InvertMask 节点 205)\n"
            "alpha 当前是 mask 复用占位,v8.1 接入 ViTMatte/RMBG 替换\n\n"
            "**定位兜底**:\n"
            "- LocateAnything 生成目标矩形 mask\n"
            "- SAM/MaskFix 可用时优先使用 SAM\n"
            "- SAM 不可用时回退到矩形 mask,保证 Qwen V2 仍收到 brush\n\n"
            "**输出**: RGBA PNG via VR_JoinRGBA\n"
            "- A: v8_A_foreground_RGBA_*.png\n"
            "- B: v8_B_background_RGBA_*.png"
        ],
    )

    # ─────────────── Stage 4: tune brush GrowMask (node 204) ───────────────
    grow_node = find_node(g, 204)
    assert grow_node["type"] == "GrowMask", grow_node["type"]
    grow_node["widgets_values"][0] = GROW_MASK_PX

    # ─────────────── Stage 5: size-match brush base images ───────────────
    # Replace hardcoded 1024×1024 EmptyImage nodes with VR_EmptyImageLike that
    # mirrors the scaled-input image's H×W, so brush maps match aspect ratio.
    for nid, (r, gc, b) in BRUSH_BASE_NODES.items():
        node = find_node(g, nid)
        assert node["type"] == "EmptyImage", f"node {nid} is {node['type']}, not EmptyImage"
        node["type"] = "VR_EmptyImageLike"
        node["properties"]["Node name for S&R"] = "VR_EmptyImageLike"
        node["inputs"] = [{"name": "reference", "type": "IMAGE", "link": None}]
        node["widgets_values"] = [int(r), int(gc), int(b)]
        link_id = next_link_id(g)
        # ImageScaleToMaxDimension output → this node's reference input
        g["links"].append([link_id, SCALED_INPUT_NODE, 0, nid, 0, "IMAGE"])
        node["inputs"][0]["link"] = link_id
        # also append this link to the source node's output[0].links
        src = find_node(g, SCALED_INPUT_NODE)
        src["outputs"][0].setdefault("links", []).append(link_id)

    # ─────────────── Stage 6: gate brush ReferenceLatent by SAM mask usability ───────────────
    # Replace ReferenceLatent #2 (brush) with a VectorReady equivalent that only
    # appends the brush latent when MaskFix+'s SAM mask has enough foreground.
    # Unusable masks keep conditioning as: text + original-image reference.
    brush_ref = find_node(g, 53)
    assert brush_ref["type"] == "ReferenceLatent", brush_ref["type"]
    cond_link = brush_ref["inputs"][0]["link"]
    latent_link = brush_ref["inputs"][1]["link"]
    brush_ref["type"] = "VR_ReferenceLatentIfMaskUsable"
    brush_ref["title"] = "[条件] ★ RefLatent #2: 画笔 (mask 可用时追加)"
    brush_ref["properties"]["Node name for S&R"] = "VR_ReferenceLatentIfMaskUsable"
    brush_ref["inputs"] = [
        {"name": "conditioning", "type": "CONDITIONING", "link": cond_link},
        {"name": "latent", "type": "LATENT", "link": latent_link},
        {"name": "mask", "type": "MASK", "link": None},
    ]
    brush_ref["outputs"] = [
        {"name": "conditioning", "type": "CONDITIONING", "links": brush_ref["outputs"][0]["links"]},
        {"name": "mask_usable", "type": "BOOLEAN", "links": []},
        {"name": "status_image", "type": "IMAGE", "links": []},
    ]
    brush_ref["widgets_values"] = [
        BRUSH_MASK_THRESHOLD,
        BRUSH_MIN_AREA_RATIO,
        BRUSH_MAX_AREA_RATIO,
        BRUSH_MIN_AREA_PX,
    ]
    add_link(g, resolver_id, 0, 53, 2, "MASK")

    brush_status_preview_id = add_node(
        g,
        ntype="PreviewImage",
        title="🔍 [诊断9] Brush 是否送入 Qwen (绿=使用, 红=跳过)",
        pos=[2550, 900],
        inputs=[{"name": "images", "type": "IMAGE", "link": None}],
    )
    add_link(g, 53, 2, brush_status_preview_id, 0, "IMAGE")

    g["last_node_id"] = max((n["id"] for n in g["nodes"]), default=0)
    g["last_link_id"] = max((l[0] for l in g["links"]), default=0)
    V8.write_text(json.dumps(g, ensure_ascii=False, indent=2))
    print(f"wrote {V8} — {len(g['nodes'])} nodes, {len(g['links'])} links")


if __name__ == "__main__":
    sys.exit(main())
