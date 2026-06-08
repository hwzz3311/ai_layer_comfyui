"""Patch base_layered.json → base_layered_debug.json.

Replaces VR_PipelineLayered with VR_PipelineLayeredDebug, tapping every
intermediate stage into a PreviewImage / MaskPreview+ so each shows up in the
ComfyUI preview panel and gets logged to vr_debug.log. Production output
(final RGB + alpha → VR_JoinRGBA → SaveImage) is preserved.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows/layered/base_layered.json"
DST = ROOT / "workflows/layered/base_layered_debug.json"

# Mirrors VR_PipelineLayeredDebug.RETURN_NAMES / RETURN_TYPES.
DEBUG_OUTPUTS = [
    ("input_rgb", "IMAGE"),
    ("native_alpha_viz", "IMAGE"),
    ("alpha_cleaned_viz", "IMAGE"),
    ("edge_roi_viz", "IMAGE"),
    ("roi_sharpened", "IMAGE"),
    ("edge_inpainted", "IMAGE"),
    ("alpha_out", "MASK"),
]
FINAL_RGB_SLOT = 5    # edge_inpainted → JoinRGBA.image
FINAL_ALPHA_SLOT = 6  # alpha_out → JoinRGBA.alpha


def find_node(g, nid):
    return next(n for n in g["nodes"] if n["id"] == nid)


def find_by_type(g, t):
    return next(n for n in g["nodes"] if n["type"] == t)


def remove_link(g, lid):
    g["links"] = [l for l in g["links"] if l[0] != lid]
    for n in g["nodes"]:
        for sock in n.get("inputs", []):
            if sock.get("link") == lid:
                sock["link"] = None
        for sock in n.get("outputs", []):
            if sock.get("links") and lid in sock["links"]:
                sock["links"].remove(lid)


def add_link(g, s, ss, d, ds, t):
    lid = max((l[0] for l in g["links"]), default=0) + 1
    g["links"].append([lid, s, ss, d, ds, t])
    out = find_node(g, s)["outputs"][ss]
    out.setdefault("links", [])
    out["links"].append(lid)
    find_node(g, d)["inputs"][ds]["link"] = lid
    return lid


def add_node(g, *, ntype, title, pos, inputs=None, outputs=None, widgets=None):
    nid = max((n["id"] for n in g["nodes"]), default=0) + 1
    g["nodes"].append({
        "id": nid, "type": ntype, "pos": pos, "size": [240, 80],
        "flags": {}, "order": 0, "mode": 0, "title": title,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype}, "widgets_values": widgets or [],
    })
    return nid


def main():
    g = json.loads(SRC.read_text())

    # When post-processing is disabled (default), base_layered has no
    # VR_PipelineLayered node — the debug variant is meaningless, so just mirror
    # the production graph unchanged.
    if not any(n["type"] == "VR_PipelineLayered" for n in g["nodes"]):
        DST.write_text(json.dumps(g, ensure_ascii=False, indent=2))
        print(f"wrote {DST} — post-processing off, mirrors production "
              f"({len(g['nodes'])} nodes, {len(g['links'])} links)")
        return

    old = find_by_type(g, "VR_PipelineLayered")
    old_id = old["id"]

    # Capture the image input source and the downstream JoinRGBA wiring.
    img_link = old["inputs"][0]["link"]
    img_l = next(l for l in g["links"] if l[0] == img_link)
    img_src = (img_l[1], img_l[2])

    out_links = [l for l in g["links"] if l[1] == old_id]
    join_id = out_links[0][3]  # both outputs go to VR_JoinRGBA
    widgets = list(old.get("widgets_values", []))

    # Tear down old node + its links.
    for l in list(out_links):
        remove_link(g, l[0])
    remove_link(g, img_link)
    g["nodes"] = [n for n in g["nodes"] if n["id"] != old_id]

    debug_id = add_node(
        g,
        ntype="VR_PipelineLayeredDebug",
        title="🔬 VectorReady Layered DEBUG (7 outs)",
        pos=[old["pos"][0] + 100, old["pos"][1]],
        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
        outputs=[{"name": n, "type": t, "links": []} for n, t in DEBUG_OUTPUTS],
        widgets=widgets,
    )
    add_link(g, img_src[0], img_src[1], debug_id, 0, "IMAGE")
    add_link(g, debug_id, FINAL_RGB_SLOT, join_id, 0, "IMAGE")
    add_link(g, debug_id, FINAL_ALPHA_SLOT, join_id, 1, "MASK")

    base_x, base_y = old["pos"][0] + 100, old["pos"][1] + 300
    for i, (name, t) in enumerate(DEBUG_OUTPUTS):
        col, row = i % 4, i // 4
        if t == "MASK":
            pv = add_node(g, ntype="MaskPreview+", title=f"🔍 [{i}] {name}",
                          pos=[base_x + col * 320, base_y + row * 260],
                          inputs=[{"name": "mask", "type": "MASK", "link": None}])
        else:
            pv = add_node(g, ntype="PreviewImage", title=f"🔍 [{i}] {name}",
                          pos=[base_x + col * 320, base_y + row * 260],
                          inputs=[{"name": "images", "type": "IMAGE", "link": None}])
        add_link(g, debug_id, i, pv, 0, t)

    add_node(
        g, ntype="MarkdownNote", title="📘 base_layered DEBUG 说明",
        pos=[base_x, base_y - 220],
        widgets=[
            "## base_layered DEBUG\n\n"
            "VR_PipelineLayeredDebug (7 outs): input_rgb / native_alpha / "
            "alpha_cleaned / edge_roi / roi_sharpened / edge_inpainted / alpha_out\n\n"
            "**日志**: vr_debug.log (每阶段 _stats)\n"
            "**生产输出**: SaveImage(layer_) 仍正常保存"
        ],
    )

    g["last_node_id"] = max((n["id"] for n in g["nodes"]), default=0)
    g["last_link_id"] = max((l[0] for l in g["links"]), default=0)
    DST.write_text(json.dumps(g, ensure_ascii=False, indent=2))
    print(f"wrote {DST} — {len(g['nodes'])} nodes, {len(g['links'])} links")


if __name__ == "__main__":
    main()
