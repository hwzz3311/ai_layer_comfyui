"""Convert a ComfyUI API/prompt-format workflow (flat {id: {class_type, inputs}})
into a UI-graph workflow (nodes[]/links[]) that loads in the ComfyUI canvas and
that the _uigraph injection helpers can operate on.

Why this exists: the autodetect IP-keep workflow only exists in API format
(backend/workflows/inpaint/ip_consistent_generate.json). API format cannot be
dragged into the canvas, and build_ip_consistent.py needs UI-graph. Rather than
hand-rebuild ~33 nodes in the canvas, we convert deterministically:

- An API input value that is a 2-element list [src_id, src_slot] is a LINK;
  any scalar value is a WIDGET. (Verified clean for every node in this graph.)
- Link slot INDICES come straight from the API refs, so wiring is exact.
- Output slot NAMES are assigned from OUTPUT_SPECS so the name-resolving build
  script works pre-load; on load ComfyUI reconciles names against live node
  defs anyway, so cosmetic gaps on external nodes are harmless.

Run: python scripts/api_to_uigraph.py
  → workflows/inpaint/ip_consistent_base.json
"""
from __future__ import annotations

from pathlib import Path

import _uigraph as u

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows/inpaint/ip_consistent_autodetect.api.json"
DST = ROOT / "workflows/inpaint/ip_consistent_base.json"

# (name, type) per output slot, in slot order. VR names come from each node's
# RETURN_NAMES; core nodes from ComfyUI conventions; external (sam3/MaskFix+)
# only need the slots actually referenced — names are cosmetic (reconciled on load).
OUTPUT_SPECS: dict[str, list[tuple[str, str]]] = {
    "LoadImage": [("IMAGE", "IMAGE"), ("MASK", "MASK")],
    "ImageScaleToMaxDimension": [("IMAGE", "IMAGE")],
    "easy sam3ModelLoader": [("sam3_model", "SAM3_MODEL")],
    "easy sam3ImageSegmentation": [("mask", "MASK"), ("image", "IMAGE")],
    "MaskFix+": [("MASK", "MASK")],
    "VR_LocateAnythingBox": [("box_mask", "MASK"), ("preview_image", "IMAGE"),
                             ("bbox_json", "STRING"), ("box_usable", "BOOLEAN"),
                             ("bboxes", "BBOX")],
    "VR_TargetMaskResolver": [("resolved_mask", "MASK"), ("quality_preview", "IMAGE"),
                              ("sam_usable", "BOOLEAN"), ("fallback_used", "BOOLEAN")],
    "VR_MaskUnion": [("mask", "MASK")],
    "VR_MaskSubtract": [("mask", "MASK")],
    "FluxKontextImageScale": [("IMAGE", "IMAGE")],
    "UNETLoader": [("MODEL", "MODEL")],
    "CLIPLoader": [("CLIP", "CLIP")],
    "VAELoader": [("VAE", "VAE")],
    "LoraLoaderModelOnly": [("MODEL", "MODEL")],
    "ModelSamplingAuraFlow": [("MODEL", "MODEL")],
    "CFGNorm": [("MODEL", "MODEL")],
    "PrimitiveStringMultiline": [("STRING", "STRING")],
    "TextEncodeQwenImageEditPlus": [("CONDITIONING", "CONDITIONING")],
    "VAEEncode": [("LATENT", "LATENT")],
    "InvertMask": [("MASK", "MASK")],
    "GrowMask": [("MASK", "MASK")],
    "SetLatentNoiseMask": [("LATENT", "LATENT")],
    "KSampler": [("LATENT", "LATENT")],
    "VAEDecode": [("IMAGE", "IMAGE")],
    "ImageCompositeMasked": [("IMAGE", "IMAGE")],
    "MaskPreview+": [("MASK", "MASK")],
    "SaveImage": [],
    "PreviewImage": [],
}


def is_link(v) -> bool:
    return (isinstance(v, list) and len(v) == 2
            and isinstance(v[0], (str, int)) and isinstance(v[1], int))


def _outputs_for(class_type: str, max_slot: int) -> list[dict]:
    spec = OUTPUT_SPECS.get(class_type)
    if spec is None:
        spec = [(f"out{i}", "*") for i in range(max_slot + 1)]
    elif len(spec) <= max_slot:  # referenced beyond known spec → pad
        spec = spec + [(f"out{i}", "*") for i in range(len(spec), max_slot + 1)]
    return [{"name": n, "type": t, "links": [], "slot_index": i}
            for i, (n, t) in enumerate(spec)]


def _depth(nid: str, api: dict, memo: dict) -> int:
    if nid in memo:
        return memo[nid]
    memo[nid] = 0  # guard against cycles
    srcs = [v[0] for v in api[nid]["inputs"].values() if is_link(v)]
    memo[nid] = 0 if not srcs else 1 + max(_depth(str(s), api, memo) for s in srcs)
    return memo[nid]


def convert(api: dict) -> dict:
    # max output slot referenced per source node (to size outputs[])
    max_slot: dict[str, int] = {}
    for node in api.values():
        for v in node["inputs"].values():
            if is_link(v):
                sid = str(v[0])
                max_slot[sid] = max(max_slot.get(sid, 0), int(v[1]))

    memo: dict = {}
    depth_count: dict[int, int] = {}
    g = {"last_node_id": 0, "last_link_id": 0, "nodes": [], "links": [],
         "groups": [], "config": {}, "extra": {}, "version": 0.4}

    # pass 1: node shells (inputs = link-inputs in order; widgets = scalars in order)
    for nid in sorted(api, key=lambda k: int(k)):
        node = api[nid]
        ct = node["class_type"]
        inputs, widgets = [], []
        for name, v in node["inputs"].items():
            if is_link(v):
                src_ct = api[str(v[0])]["class_type"]
                src_outs = OUTPUT_SPECS.get(src_ct) or []
                itype = src_outs[v[1]][1] if v[1] < len(src_outs) else "*"
                inputs.append({"name": name, "type": itype, "link": None})
            else:
                widgets.append(v)
        d = _depth(nid, api, memo)
        col = depth_count.get(d, 0)
        depth_count[d] = col + 1
        g["nodes"].append({
            "id": int(nid), "type": ct, "pos": [d * 360, col * 210],
            "size": [260, 130], "flags": {}, "order": int(nid), "mode": 0,
            "inputs": inputs,
            "outputs": _outputs_for(ct, max_slot.get(nid, 0)),
            "properties": {"Node name for S&R": ct,
                           "_api_title": node.get("_meta", {}).get("title", "")},
            "widgets_values": widgets,
            "title": node.get("_meta", {}).get("title", ct),
        })
    g["last_node_id"] = max(int(k) for k in api)

    # pass 2: links (dst input slot = position in that node's inputs[] array)
    for nid in sorted(api, key=lambda k: int(k)):
        node = api[nid]
        link_inputs = [(name, v) for name, v in node["inputs"].items() if is_link(v)]
        for di, (name, v) in enumerate(link_inputs):
            src_id, src_slot = int(v[0]), int(v[1])
            src_out_name = u.find_by_id(g, src_id)["outputs"][src_slot]["name"]
            u.add_link(g, src_id, src_out_name, int(nid), name,
                       u.find_by_id(g, src_id)["outputs"][src_slot]["type"])
    u.assert_graph_valid(g)
    return g


def main():
    api = u.load(SRC)
    g = convert(api)
    u.dump(g, DST)
    print(f"wrote {DST} ({len(g['nodes'])} nodes, {len(g['links'])} links)")


if __name__ == "__main__":
    main()
