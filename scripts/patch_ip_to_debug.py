"""Derive workflows/inpaint/ip_consistent_debug.json from ip_consistent.json.

For each pipeline stage, hang a VR_DebugProbe (logs tensor stats to
vr_ip_consistent.log) feeding a Preview node (PreviewImage / MaskPreview+) so
the stage is visible on the canvas AND logged. Each stage output gains a parallel
probe→preview tap; the original wiring is untouched.

Pruning behaviour: taps on sampler-tail stages (VAEDecode / paste-back) sit
downstream of the latent gates, so the unselected branch's previews prune with
it — only the live branch's final image renders. Taps on the mask/condition
stages sit upstream of the gates, so in debug runs BOTH branches' masks compute
(intentional: lets you compare the alpha vs autodetect mask sources side by
side). The lean production graph has none of these and prunes fully.

Run: python scripts/patch_ip_to_debug.py
"""
from __future__ import annotations

from pathlib import Path

import _uigraph as u

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows/inpaint/ip_consistent.json"
DST = ROOT / "workflows/inpaint/ip_consistent_debug.json"

# (kind, locator, out_name, label) — locator is ("id", n) or ("title", str).
# kind "image" → VR_DebugProbeImage + PreviewImage; "mask" → Mask variants.
STAGES = [
    ("image", ("id", 301), "IMAGE", "work_image"),
    # autodetect mask chain (upstream of gate; runs in debug to expose masks)
    ("mask", ("id", 11), "mask", "autodetect_sam3_raw"),
    ("mask", ("id", 20), "MASK", "autodetect_maskfix"),
    ("mask", ("id", 222), "resolved_mask", "autodetect_resolver"),
    ("mask", ("id", 230), "mask", "autodetect_cutout_union"),
    ("mask", ("id", 232), "mask", "autodetect_protect"),
    # alpha branch masks / condition (located by injected titles)
    ("mask", ("title", "IMAGE→保护蒙版(work尺寸)"), "MASK", "alpha_protect"),
    ("mask", ("title", "可编辑区膨胀"), "MASK", "alpha_editable_grown"),
    ("image", ("title", "IP叠白底(条件图)"), "IMAGE", "alpha_condition_white"),
    # sampler tails (downstream of gates → prune with their branch)
    ("image", ("id", 315), "IMAGE", "autodetect_decoded"),
    ("image", ("id", 316), "IMAGE", "autodetect_final"),
    ("image", ("title", "alpha VAE解码"), "IMAGE", "alpha_decoded"),
    ("image", ("title", "alpha 原像素盖回"), "IMAGE", "alpha_final"),
]


def _resolve(g, locator):
    kind, val = locator
    return u.find_by_id(g, val) if kind == "id" else u.find_by_title(g, val)


def _tap_mask(g, node_id, out_name, label, pos):
    p = u.add_node(g, ntype="VR_DebugProbeMask", title=f"probe:{label}", pos=pos,
                   inputs=[{"name": "mask", "type": "MASK", "link": None}],
                   outputs=[{"name": "MASK", "type": "MASK", "links": []}],
                   widgets=[label])
    u.add_link(g, node_id, out_name, p, "mask", "MASK")
    prev = u.add_node(g, ntype="MaskPreview+", title=f"👁 {label}",
                      pos=[pos[0] + 240, pos[1]],
                      inputs=[{"name": "mask", "type": "MASK", "link": None}],
                      outputs=[])
    u.add_link(g, p, "MASK", prev, "mask", "MASK")


def _tap_image(g, node_id, out_name, label, pos):
    p = u.add_node(g, ntype="VR_DebugProbeImage", title=f"probe:{label}", pos=pos,
                   inputs=[{"name": "image", "type": "IMAGE", "link": None}],
                   outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
                   widgets=[label])
    u.add_link(g, node_id, out_name, p, "image", "IMAGE")
    prev = u.add_node(g, ntype="PreviewImage", title=f"👁 {label}",
                      pos=[pos[0] + 240, pos[1]],
                      inputs=[{"name": "images", "type": "IMAGE", "link": None}],
                      outputs=[])
    u.add_link(g, p, "IMAGE", prev, "images", "IMAGE")


def main():
    g = u.load(SRC)
    y = -700
    for kind, locator, out_name, label in STAGES:
        node = _resolve(g, locator)
        pos = [3900, y]
        if kind == "mask":
            _tap_mask(g, node["id"], out_name, label, pos)
        else:
            _tap_image(g, node["id"], out_name, label, pos)
        y += 230
    u.assert_graph_valid(g)
    u.dump(g, DST)
    print(f"wrote {DST} ({len(g['nodes'])} nodes, {len(g['links'])} links)")


if __name__ == "__main__":
    main()
