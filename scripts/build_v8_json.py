"""Build qwen_layered_v8_ab_vector_ready.json from v7.

Changes vs v7:
- A/B真正分流: insert VR_GatedPassthrough on each KSampler's latent_image input,
  driven by the foreground_mode boolean (node 215). The unselected branch's
  KSampler receives ExecutionBlocker and the entire downstream chain is pruned.
- Positive silhouette chain: PrimitiveNode "Target Query" → LA → SAM3
  (dual-prompt: text + bbox) → MaskFix → VR_TargetMaskResolver (with LA
  rectangle fallback).
- Negative cutout chain (v8.2): PrimitiveNode "Cutout Query" → LA #2
  → SAM3 #2 (dual-prompt, shares model loader) → MaskFix → VR_MaskSubtract.
  Default cutout query = "" short-circuits the LA inference and turns the
  subtract into a passthrough, so v8.0 workflows behave identically.
- VectorReady tails: VR_PipelineLight on A path, VR_PipelineStrong on B path,
  between VAEDecode and SaveImage. Alpha source for A is the post-subtract
  final mask; for B the InvertMask (node 205).
- Final SaveImage receives RGBA via VR_JoinRGBA (opacity convention + final
  transparent-region RGB/alpha clamp).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
V7 = PROJECT / "workflows/layered/v7_ab_dual_path.json"
V8 = PROJECT / "workflows/layered/v8_ab_vector_ready.json"
DEFAULT_RMBG_MODEL_PATH = "/root/ComfyUI/models/RMBG-2.0"
DEFAULT_LOCATE_MODEL_ID = "/root/ComfyUI/models/LocateAnything-3B"
DEFAULT_LOCATE_QUERY = "main target object"
# Cutout query is empty by default: the negative chain short-circuits at the
# LA #2 empty-query check (zero inference cost) and VR_MaskSubtract becomes a
# passthrough. Agents fill this in (e.g. "rectangular photo window inside the
# card holder") for subjects with internal cutouts. See VR_MaskSubtract +
# docs/qwen_layered_v8_api_agent_guide.md for the calling contract.
DEFAULT_CUTOUT_QUERY = ""
CUTOUT_INNER_DILATE_PX = 2
CUTOUT_MIN_INNER_AREA_RATIO = 0.0005
# Fill the positive subject's interior holes before subtracting the cutout, so
# SAM3 under-segmentation (white line-art interiors) doesn't leak as spurious
# transparency. Gated inside the node on the cutout chain being non-empty, so
# subjects without a cutout query keep their genuine holes (donuts, frames).
CUTOUT_FILL_OUTER_HOLES = True
# Revert to the (hole-filled) outer mask when the cutout subtraction would
# erase more of the subject than this fraction leaves behind — guards against
# SAM3 #2 over-grounding the whole subject as the "window".
CUTOUT_MIN_RETAINED_RATIO = 0.2

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

# ── KSampler tuning (2026-06-07) ────────────────────────────────────────
# v7 inherited A-path steps=7 / cfg=0.8, which is BELOW even the Qwen-Image-
# Layered-Control-V2 brush-mode recommendation (≥10 steps, cfg≈1.0; "raise
# steps when the target is occluded"). The official ComfyUI control template
# uses 20 / 2.5 (no LoRA/brush). v8 runs control_bf16 + V2 brush LoRA, so cfg≈1
# is correct, but 7 steps under-samples — the same under-stepping that ruined
# the base workflow before we matched the model's real settings. Bump A to
# 16 / 1.0: generous step budget for 1024-px output + occluded layers, brush-
# appropriate low cfg. B-path (background reconstruction) keeps its 16 / 1.0.
A_KSAMPLER_NODE = 60
B_KSAMPLER_NODE = 210
A_KSAMPLER_STEPS = 16
A_KSAMPLER_CFG = 1.0
B_KSAMPLER_STEPS = 16
B_KSAMPLER_CFG = 1.0

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
        widgets=[True, False, "foreground_mode_A"],  # enable, invert, label
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
        widgets=[True, True, "foreground_mode_B"],  # enable, invert=true, label
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

    # ─────────────── Stage 2: LocateAnything + SAM3 dual-prompt ───────────────
    # LocateAnything gives a robust coarse box; SAM3 gives a precise silhouette.
    # We want SAM3 to consume BOTH text (semantic) AND bbox (spatial) — they are
    # complementary, not interchangeable. Empirically (vr_debug.log run on the
    # cat-card frame), SAM3 with empty text + bbox returned area=0; SAM3 with
    # text alone produced unstable silhouettes that the LA bbox then could only
    # AND-mask, never repair. Dual-prompt fixes both modes.
    #
    # To keep LA.query and SAM3.text from drifting apart, both are converted to
    # input sockets driven by a single PrimitiveNode (STRING). Edit the query in
    # one place in the UI and both nodes update.
    target_query_id = add_node(
        g,
        ntype="PrimitiveNode",
        title="🎯 Target Query (shared: LA + SAM3)",
        pos=[480, 380],
        outputs=[{"name": "STRING", "type": "STRING", "links": [], "widget": {"name": "value"}}],
        widgets=[DEFAULT_LOCATE_QUERY],
    )

    locate_id = add_node(
        g,
        ntype="VR_LocateAnythingBox",
        title="📍 LocateAnything · Target Box",
        pos=[840, 430],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            # `query` widget converted to input so the shared PrimitiveNode
            # drives it. The widgets_values entry below is a placeholder kept
            # for positional alignment with the remaining widgets.
            {"name": "query", "type": "STRING", "link": None, "widget": {"name": "query"}},
        ],
        outputs=[
            {"name": "box_mask", "type": "MASK", "links": []},
            {"name": "preview_image", "type": "IMAGE", "links": []},
            {"name": "bbox_json", "type": "STRING", "links": []},
            {"name": "box_usable", "type": "BOOLEAN", "links": []},
            {"name": "bboxes", "type": "BBOX", "links": []},
        ],
        # Positional widgets: query (converted to input, placeholder),
        # model_id, device, generation_mode, prompt_mode, padding_px,
        # max_new_tokens, temperature.
        widgets=[
            DEFAULT_LOCATE_QUERY,  # placeholder for converted widget
            DEFAULT_LOCATE_MODEL_ID,
            "auto",
            "hybrid",
            "single",
            8,
            2048,
            0.7,
            "keep",        # attn_implementation
            "positive",    # label (distinguishes the two LA instances in vr_debug.log)
        ],
    )
    add_link(g, SCALED_INPUT_NODE, 0, locate_id, 0, "IMAGE")
    add_link(g, target_query_id, 0, locate_id, 1, "STRING")

    # SAM3 receives BOTH the shared text query AND the LA bbox. easy-sam3's
    # `text` widget (widgets_values[0]) is converted to an input socket so it
    # tracks the shared PrimitiveNode; the bbox input is added separately.
    sam3_node = find_node(g, 11)
    # Convert widgets_values[0] (text) to an input. Keep the value as a
    # placeholder so positional widgets after it (threshold, multimask, etc.)
    # remain aligned.
    sam3_node.setdefault("inputs", []).append(
        {"name": "text", "type": "STRING", "link": None, "widget": {"name": "text"}}
    )
    sam3_text_slot = len(sam3_node["inputs"]) - 1
    if sam3_node.get("widgets_values"):
        # Seed placeholder with the shared default; UI value comes from the
        # PrimitiveNode at runtime.
        sam3_node["widgets_values"][0] = DEFAULT_LOCATE_QUERY
    add_link(g, target_query_id, 0, 11, sam3_text_slot, "STRING")
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

    # ─────────── Stage 2b: Negative chain (cutout / hole extraction) ───────────
    # Subjects with topological holes — card-holder photo slots, picture
    # frames, donut shapes — cannot be expressed by any single LA/SAM3 mask
    # (both produce solid silhouettes). Instead of relying on Qwen-Layered V2
    # to "infer" the hole as transparent (unstable), we mirror the proven
    # positive LA+SAM3 chain as a negative chain whose output is subtracted
    # from the positive silhouette.
    #
    # Differences vs the positive chain:
    #  - No Resolver / no LA-rectangle fallback. If cutout SAM3 finds nothing,
    #    the inner mask stays empty → subtract is a no-op. We never want to
    #    over-subtract a full rectangle on a SAM3 miss.
    #  - LA #2 short-circuits when its query is empty (see locate_anything_box
    #    early-return), so the default workflow with cutout_query="" pays
    #    zero compute and behaves identically to v8.0.
    #  - SAM3 #2 reuses the existing easy sam3ModelLoader (node 10) and the
    #    scaled-input image (node 5) — only one model load.
    cutout_query_id = add_node(
        g,
        ntype="PrimitiveNode",
        title="🕳 Cutout Query (shared: LA#2 + SAM3#2; empty = no subtraction)",
        pos=[480, 1080],
        outputs=[{"name": "STRING", "type": "STRING", "links": [], "widget": {"name": "value"}}],
        widgets=[DEFAULT_CUTOUT_QUERY],
    )

    locate_neg_id = add_node(
        g,
        ntype="VR_LocateAnythingBox",
        title="📍 LocateAnything · Cutout Box (negative)",
        pos=[840, 1130],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
            {"name": "query", "type": "STRING", "link": None, "widget": {"name": "query"}},
        ],
        outputs=[
            {"name": "box_mask", "type": "MASK", "links": []},
            {"name": "preview_image", "type": "IMAGE", "links": []},
            {"name": "bbox_json", "type": "STRING", "links": []},
            {"name": "box_usable", "type": "BOOLEAN", "links": []},
            {"name": "bboxes", "type": "BBOX", "links": []},
        ],
        widgets=[
            DEFAULT_CUTOUT_QUERY,  # placeholder for converted query widget
            DEFAULT_LOCATE_MODEL_ID,
            "auto",
            "hybrid",
            # multi: subjects can have several internal cutouts (e.g. a frame
            # with multiple window slots). Single-instance prompt would force
            # LA to pick one; multi lets SAM3 receive all bboxes.
            "multi",
            8,
            2048,
            0.7,
            "keep",              # attn_implementation
            "negative-cutout",   # label
        ],
    )
    add_link(g, SCALED_INPUT_NODE, 0, locate_neg_id, 0, "IMAGE")
    add_link(g, cutout_query_id, 0, locate_neg_id, 1, "STRING")

    locate_neg_preview_id = add_node(
        g,
        ntype="PreviewImage",
        title="🔍 [诊断12] Cutout LocateAnything 矩形框",
        pos=[1180, 1130],
        inputs=[{"name": "images", "type": "IMAGE", "link": None}],
    )
    add_link(g, locate_neg_id, 1, locate_neg_preview_id, 0, "IMAGE")

    # SAM3 #2 — symmetric dual-prompt (text + bbox) for the cutout.
    sam3_neg_id = add_node(
        g,
        ntype="easy sam3ImageSegmentation",
        title="[SAM3] 分割 (Cutout / negative)",
        pos=[1540, 1130],
        inputs=[
            {"name": "sam3_model", "type": "EASY_SAM3_MODEL", "link": None},
            {"name": "images", "type": "IMAGE", "link": None},
            {"name": "text", "type": "STRING", "link": None, "widget": {"name": "text"}},
            {"name": "bboxes", "type": "BBOX", "link": None},
        ],
        outputs=[
            {"name": "masks", "type": "MASK", "links": []},
            {"name": "images", "type": "IMAGE", "links": []},
            {"name": "obj_masks", "type": "MASK", "links": []},
            {"name": "boxes", "type": "BBOX", "links": []},
            {"name": "scores", "type": "FLOAT", "links": []},
        ],
        # Same positional widgets as SAM3 #1: text (converted), threshold,
        # multimask, mask_processing, max_objects.
        widgets=[DEFAULT_CUTOUT_QUERY, 0.4, False, "none", -1],
    )
    # Share the existing sam3 model loader (node 10) — no extra load.
    add_link(g, 10, 0, sam3_neg_id, 0, "EASY_SAM3_MODEL")
    add_link(g, SCALED_INPUT_NODE, 0, sam3_neg_id, 1, "IMAGE")
    add_link(g, cutout_query_id, 0, sam3_neg_id, 2, "STRING")
    add_link(g, locate_neg_id, 4, sam3_neg_id, 3, "BBOX")

    maskfix_neg_id = add_node(
        g,
        ntype="MaskFix+",
        title="[Mask] MaskFix+ 修复 (Cutout / negative)",
        pos=[1900, 1130],
        inputs=[{"name": "mask", "type": "MASK", "link": None}],
        outputs=[{"name": "MASK", "type": "MASK", "links": []}],
        # Same defaults as positive-side MaskFix+ node 20.
        widgets=[3, 1, 5, 4, 4],
    )
    add_link(g, sam3_neg_id, 0, maskfix_neg_id, 0, "MASK")

    sam3_neg_preview_id = add_node(
        g,
        ntype="PreviewImage",
        title="🔍 [诊断13] Cutout SAM3 + MaskFix mask",
        pos=[2240, 1130],
        inputs=[{"name": "images", "type": "IMAGE", "link": None}],
    )
    # Preview the MaskFix output directly — MASK is not an IMAGE so we go
    # through SAM3's preview-image socket which already overlays the mask.
    add_link(g, sam3_neg_id, 1, sam3_neg_preview_id, 0, "IMAGE")

    # Union: LA #2 emits a union mask of all detected cutout boxes (multi
    # mode, since v0.12.0), while SAM3 #2's dual-prompt path only refines
    # around the primary bbox. ORing the two means N-hole subjects (picture
    # frames, photo grids) still get every hole subtracted: SAM3-refined
    # where it landed, LA-rectangle elsewhere. Both inputs being empty
    # (cutout_query="") leaves the result empty → MaskSubtract passthrough.
    cutout_union_id = add_node(
        g,
        ntype="VR_MaskUnion",
        title="∪ Cutout Union (LA boxes ∪ SAM3 refined)",
        pos=[2240, 970],
        inputs=[
            {"name": "mask_a", "type": "MASK", "link": None},
            {"name": "mask_b", "type": "MASK", "link": None},
        ],
        outputs=[{"name": "mask", "type": "MASK", "links": []}],
    )
    # mask_a = SAM3-refined (tight); mask_b = LA box-union (coarse, all N).
    add_link(g, maskfix_neg_id, 0, cutout_union_id, 0, "MASK")
    add_link(g, locate_neg_id, 0, cutout_union_id, 1, "MASK")

    cutout_union_preview_id = add_node(
        g,
        ntype="MaskPreview+",
        title="🔍 [诊断13.5] Cutout 合并 mask (LA boxes ∪ SAM3)",
        pos=[2560, 970],
        inputs=[{"name": "mask", "type": "MASK", "link": None}],
    )
    add_link(g, cutout_union_id, 0, cutout_union_preview_id, 0, "MASK")

    # Subtract: final = clamp(resolver_positive - dilate(cutout_union, px), 0, 1).
    # When cutout_query is empty the union mask is zeros and this node
    # passes the outer mask straight through.
    mask_subtract_id = add_node(
        g,
        ntype="VR_MaskSubtract",
        title="🎯− Mask Subtract (positive − cutout union)",
        pos=[2240, 700],
        inputs=[
            {"name": "outer", "type": "MASK", "link": None},
            {"name": "inner", "type": "MASK", "link": None},
        ],
        outputs=[{"name": "mask", "type": "MASK", "links": []}],
        widgets=[
            CUTOUT_INNER_DILATE_PX,
            CUTOUT_MIN_INNER_AREA_RATIO,
            CUTOUT_FILL_OUTER_HOLES,
            CUTOUT_MIN_RETAINED_RATIO,
        ],
    )
    add_link(g, resolver_id, 0, mask_subtract_id, 0, "MASK")
    add_link(g, cutout_union_id, 0, mask_subtract_id, 1, "MASK")

    # Final post-subtract alpha preview. Lets the operator confirm at a glance
    # that the cutout window was actually removed from the positive silhouette
    # (or that the chain passed through unchanged when cutout_query is empty).
    final_mask_preview_id = add_node(
        g,
        ntype="MaskPreview+",
        title="🔍 [诊断14] MaskSubtract 最终 alpha (positive − cutout)",
        pos=[2560, 700],
        inputs=[{"name": "mask", "type": "MASK", "link": None}],
    )
    add_link(g, mask_subtract_id, 0, final_mask_preview_id, 0, "MASK")

    # All downstream consumers (brush ref, trimap, HFMatting, PipelineLight)
    # read from final_mask_id instead of resolver_id directly, so the cutout
    # subtraction is applied uniformly without touching their wiring sites.
    final_mask_id = mask_subtract_id

    # Existing brush construction should consume the final (post-subtract) mask.
    # Keep the old MaskFix preview wired to node 20 for diagnostics.
    rewire_input(g, 203, "mask", final_mask_id, 0, "MASK")
    rewire_input(g, 204, "mask", final_mask_id, 0, "MASK")

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
            {"name": "raw_matte", "type": "MASK", "links": []},
        ],
        # model_id, input_size, device.
        widgets=[DEFAULT_RMBG_MODEL_PATH, 1024, "auto"],
    )
    add_link(g, SCALED_INPUT_NODE, 0, hf_matte_id, 0, "IMAGE")
    add_link(g, final_mask_id, 0, hf_matte_id, 1, "MASK")

    # ── Tiered alpha fallback (防线一) ──────────────────────────────
    # When SAM3+LocateAnything both fail, the resolver(final_mask_id) is empty
    # and would zero out the whole A path. VR_AlphaResolve degrades gracefully:
    #   resolved (SAM3/LA) → rmbg raw_matte (unclipped) → Qwen native alpha.
    # Native alpha is split out of node 62 (A-path VAEDecode RGBA).
    native_alpha_id = add_node(
        g,
        ntype="VR_SplitRGBA",
        title="[兜底] Qwen 原生 alpha (A path)",
        pos=[3100, 80],
        inputs=[
            {"name": "image", "type": "IMAGE", "link": None},
        ],
        outputs=[
            {"name": "rgb", "type": "IMAGE", "links": []},
            {"name": "alpha", "type": "MASK", "links": []},
        ],
    )
    add_link(g, 62, 0, native_alpha_id, 0, "IMAGE")

    alpha_resolve_id = add_node(
        g,
        ntype="VR_AlphaResolve",
        title="🪜 Alpha 分级兜底 (A path)",
        pos=[3260, 100],
        inputs=[
            {"name": "resolved_alpha", "type": "MASK", "link": None},
            {"name": "rmbg_alpha", "type": "MASK", "link": None},
            {"name": "native_alpha", "type": "MASK", "link": None},
        ],
        outputs=[
            {"name": "alpha", "type": "MASK", "links": []},
            {"name": "source_used", "type": "STRING", "links": []},
        ],
        # min_area_ratio
        widgets=[0.002],
    )
    add_link(g, final_mask_id, 0, alpha_resolve_id, 0, "MASK")  # resolved (SAM3/LA)
    add_link(g, hf_matte_id, 2, alpha_resolve_id, 1, "MASK")  # rmbg raw_matte (unclipped)
    add_link(g, native_alpha_id, 1, alpha_resolve_id, 2, "MASK")  # Qwen native alpha

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
    add_link(g, alpha_resolve_id, 0, vr_light_id, 1, "MASK")  # tiered fallback (防线一)
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

    # VR_VectorReadyReport — inline diagnostic between JoinRGBA and SaveImage.
    # Image passes through unchanged; the STRING report_json is logged + wired
    # to a ShowText-style display so the agent can read quality stats per layer.
    # Defaults tuned for the v0.14.0 pipeline: ≤32 colors, ≥0.001 content
    # ratio, ≤5% small-island fraction. Flags fire when assumptions break.
    report_a_id = add_node(
        g,
        ntype="VR_VectorReadyReport",
        title="📋 VectorReady Report (A)",
        pos=[4080, 100],
        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
        outputs=[
            {"name": "report_json", "type": "STRING", "links": []},
            {"name": "image", "type": "IMAGE", "links": []},
        ],
        widgets=["A_foreground", 32, 0.001, 0.05],
    )
    report_b_id = add_node(
        g,
        ntype="VR_VectorReadyReport",
        title="📋 VectorReady Report (B)",
        pos=[4080, 360],
        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
        outputs=[
            {"name": "report_json", "type": "STRING", "links": []},
            {"name": "image", "type": "IMAGE", "links": []},
        ],
        widgets=["B_background", 32, 0.001, 0.05],
    )
    add_link(g, join_a_id, 0, report_a_id, 0, "IMAGE")
    add_link(g, join_b_id, 0, report_b_id, 0, "IMAGE")

    # Wire back to SaveImage via the report's passthrough image output.
    add_link(g, report_a_id, 1, 63, 0, "IMAGE")
    add_link(g, report_b_id, 1, 214, 0, "IMAGE")

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
            "## v8.2 · A/B 分流 + VectorReady + 镂空通道\n\n"
            "**A/B 切换**: 节点 215 (foreground_mode) → 两个 VR_GatedPassthrough\n\n"
            "**正向 silhouette 通道** (Target Query → LA → SAM3 双提示 → MaskFix → Resolver):\n"
            "- 改 Target Query 一处，LA + SAM3.text 同步更新\n"
            "- SAM3 不可用时 Resolver 回退到 LA 矩形\n\n"
            "**镂空通道 (v8.2 / v0.12.0 多孔加强)** "
            "(Cutout Query → LA#2 multi → SAM3#2 双提示 → MaskFix → MaskUnion → MaskSubtract):\n"
            "- Cutout Query 留空 = 整链零计算 (LA 短路, MaskUnion 双空, MaskSubtract 透传)\n"
            "- LA#2 prompt_mode='multi' → 输出 N 个 box 的 union mask + 全部 bbox\n"
            "- SAM3#2 双提示只精修主 bbox; MaskUnion = (SAM3 精修 ∪ LA 全部矩形) 补齐其余 N-1 个洞\n"
            "- 多孔主体推荐复数 query: 'rectangular photo windows', 'all photo slots'\n"
            "- 减法在 Resolver 输出后单点应用, 下游消费者无需感知\n"
            "- 不复用 Resolver 主体级矩形兜底: cutout SAM3+LA 都找不到 = 不减, 决不过减\n\n"
            "**VectorReady**:\n"
            "- A 路径: VR_PipelineLight (alpha = MaskSubtract 输出)\n"
            "- B 路径: VR_PipelineStrong (alpha = InvertMask 节点 205)\n"
            "- 两路径出口: final_defringe → edge_color_inpaint (v0.13-0.14) 杀边缘色幻影\n"
            "- 两路径终: VR_VectorReadyReport (v0.15.0) 输出 JSON 质量诊断到 vr_debug.log\n\n"
            "**输出**: RGBA PNG via VR_JoinRGBA\n"
            "- A: v8_A_foreground_RGBA_*.png (附 report JSON)\n"
            "- B: v8_B_background_RGBA_*.png (附 report JSON)"
        ],
    )

    # ─────────────── Stage 3.5: tune KSampler sampling params ───────────────
    # KSampler widgets: [seed, seed_mode, steps, cfg, sampler, scheduler, denoise].
    # Fix A-path under-stepping (7→16) and align cfg to V2 brush mode (0.8→1.0).
    for nid, steps, cfg in (
        (A_KSAMPLER_NODE, A_KSAMPLER_STEPS, A_KSAMPLER_CFG),
        (B_KSAMPLER_NODE, B_KSAMPLER_STEPS, B_KSAMPLER_CFG),
    ):
        ks = find_node(g, nid)
        assert ks["type"] == "KSampler", f"node {nid} is {ks['type']}, not KSampler"
        ks["widgets_values"][2] = int(steps)
        ks["widgets_values"][3] = float(cfg)

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
    add_link(g, final_mask_id, 0, 53, 2, "MASK")

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
