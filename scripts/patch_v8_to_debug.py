"""Patch qwen_layered_v8_ab_vector_ready.json → qwen_layered_v8_debug.json.

Replaces BOTH VR_PipelineLight (A path) AND VR_PipelineStrong (B path) with
their debug variants, wiring every intermediate output to a PreviewImage so
each stage shows up in the ComfyUI preview panel AND every stage gets logged
to vr_debug.log.

Production output (final RGB + alpha → JoinAlpha → SaveImage) is preserved
so each path still saves its final RGBA PNG."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows/layered/v8_ab_vector_ready.json"
DST = ROOT / "workflows/layered/v8_debug.json"


def find_node(g, nid):
    return next(n for n in g["nodes"] if n["id"] == nid)


def find_node_by_type(g, ntype):
    return next(n for n in g["nodes"] if n["type"] == ntype)


def add_link(g, src_node, src_slot, dst_node, dst_slot, link_type):
    lid = max((l[0] for l in g["links"]), default=0) + 1
    g["links"].append([lid, src_node, src_slot, dst_node, dst_slot, link_type])
    src = find_node(g, src_node)
    dst = find_node(g, dst_node)
    out = src["outputs"][src_slot]
    out.setdefault("links", [])
    if out["links"] is None:
        out["links"] = []
    out["links"].append(lid)
    inp = dst["inputs"][dst_slot]
    inp["link"] = lid
    return lid


def remove_link(g, link_id):
    g["links"] = [l for l in g["links"] if l[0] != link_id]
    for n in g["nodes"]:
        for sock in n.get("inputs", []):
            if sock.get("link") == link_id:
                sock["link"] = None
        for sock in n.get("outputs", []):
            if sock.get("links") and link_id in sock["links"]:
                sock["links"].remove(link_id)


def add_node(g, *, ntype, title, pos, inputs=None, outputs=None, widgets=None):
    nid = max((n["id"] for n in g["nodes"]), default=0) + 1
    g["nodes"].append({
        "id": nid,
        "type": ntype,
        "pos": pos,
        "size": [240, 80],
        "flags": {},
        "order": 0,
        "mode": 0,
        "title": title,
        "inputs": inputs or [],
        "outputs": outputs or [],
        "properties": {"Node name for S&R": ntype},
        "widgets_values": widgets or [],
    })
    return nid


def swap_preset(g, *, old_type, new_type, new_title, output_specs,
                widgets, final_rgb_slot, final_alpha_slot,
                preview_origin_pos):
    """Replace the preset with its debug variant + PreviewImage taps.

    output_specs: list of (name, type) pairs for the new node's outputs (in order).
    final_rgb_slot / final_alpha_slot: which slots feed back into JoinAlpha.
    """
    old = find_node_by_type(g, old_type)
    old_id = old["id"]
    img_link_id = old["inputs"][0]["link"]
    alpha_link_id = old["inputs"][1]["link"]
    img_l = next(l for l in g["links"] if l[0] == img_link_id)
    alpha_l = next(l for l in g["links"] if l[0] == alpha_link_id)
    img_src = (img_l[1], img_l[2])
    alpha_src = (alpha_l[1], alpha_l[2])
    source_src = None
    source_link_id = None
    extra_optional_sources = []
    if len(old.get("inputs", [])) > 2 and old["inputs"][2].get("link") is not None:
        source_link_id = old["inputs"][2]["link"]
        source_l = next(l for l in g["links"] if l[0] == source_link_id)
        source_src = (source_l[1], source_l[2])
    for idx in range(3, len(old.get("inputs", []))):
        link_id = old["inputs"][idx].get("link")
        if link_id is not None:
            link = next(l for l in g["links"] if l[0] == link_id)
            extra_optional_sources.append((idx, old["inputs"][idx]["name"], old["inputs"][idx]["type"], link_id, (link[1], link[2])))

    # VR_JoinRGBA is wired old.0 → join.0 and old.1 → join.1
    out_links = [l for l in g["links"] if l[1] == old_id]
    join_node_id = out_links[0][3]

    # Tear down old node
    for l in list(out_links):
        remove_link(g, l[0])
    remove_link(g, img_link_id)
    remove_link(g, alpha_link_id)
    if source_link_id is not None:
        remove_link(g, source_link_id)
    for _, _, _, link_id, _ in extra_optional_sources:
        remove_link(g, link_id)
    g["nodes"] = [n for n in g["nodes"] if n["id"] != old_id]

    # Add debug node
    debug_inputs = [
        {"name": "image", "type": "IMAGE", "link": None},
        {"name": "alpha", "type": "MASK", "link": None},
    ]
    if source_src is not None:
        debug_inputs.append({"name": "source_image", "type": "IMAGE", "link": None})
    if old_type == "VR_PipelineLight":
        existing_names = {sock["name"] for sock in debug_inputs}
        for name, typ in (
            ("external_matte_alpha", "MASK"),
            ("external_confidence", "MASK"),
        ):
            if name not in existing_names:
                debug_inputs.append({"name": name, "type": typ, "link": None})
    for _, name, typ, _, _ in extra_optional_sources:
        if name not in {sock["name"] for sock in debug_inputs}:
            debug_inputs.append({"name": name, "type": typ, "link": None})

    debug_id = add_node(
        g,
        ntype=new_type,
        title=new_title,
        pos=[old["pos"][0] + 100, old["pos"][1]],
        inputs=debug_inputs,
        outputs=[{"name": n, "type": t, "links": []} for n, t in output_specs],
        widgets=widgets,
    )
    add_link(g, img_src[0], img_src[1], debug_id, 0, "IMAGE")
    add_link(g, alpha_src[0], alpha_src[1], debug_id, 1, "MASK")
    if source_src is not None:
        add_link(g, source_src[0], source_src[1], debug_id, 2, "IMAGE")
    name_to_slot = {sock["name"]: i for i, sock in enumerate(debug_inputs)}
    for _, name, typ, _, src in extra_optional_sources:
        add_link(g, src[0], src[1], debug_id, name_to_slot[name], typ)
    add_link(g, debug_id, final_rgb_slot, join_node_id, 0, "IMAGE")
    add_link(g, debug_id, final_alpha_slot, join_node_id, 1, "MASK")

    # PreviewImage / MaskPreview+ for every output
    base_x, base_y = preview_origin_pos
    for i, (name, t) in enumerate(output_specs):
        col = i % 4
        row = i // 4
        if t == "MASK":
            pv = add_node(
                g,
                ntype="MaskPreview+",
                title=f"🔍 [{i}] {name}",
                pos=[base_x + col * 320, base_y + row * 260],
                inputs=[{"name": "mask", "type": "MASK", "link": None}],
            )
        else:
            pv = add_node(
                g,
                ntype="PreviewImage",
                title=f"🔍 [{i}] {name}",
                pos=[base_x + col * 320, base_y + row * 260],
                inputs=[{"name": "images", "type": "IMAGE", "link": None}],
            )
        add_link(g, debug_id, i, pv, 0, t)
    return debug_id


def main():
    g = json.loads(SRC.read_text())

    # ── A path ──
    a_outputs = [
        ("input_rgb", "IMAGE"),
        ("native_alpha_viz", "IMAGE"),
        ("alpha_cleaned_viz", "IMAGE"),
        ("sure_foreground_viz", "IMAGE"),
        ("sure_background_viz", "IMAGE"),
        ("trimap_unknown_viz", "IMAGE"),
        ("trimap_viz", "IMAGE"),
        ("matting_rgb", "IMAGE"),
        ("visible_alpha_viz", "IMAGE"),
        ("unknown_region_viz", "IMAGE"),
        ("matte_confidence_viz", "IMAGE"),
        ("source_composed", "IMAGE"),
        ("original_region_viz", "IMAGE"),
        ("qwen_region_viz", "IMAGE"),
        ("transparent_region_viz", "IMAGE"),
        ("low_confidence_viz", "IMAGE"),
        ("hole_viz", "IMAGE"),
        ("detail_viz", "IMAGE"),
        ("line_viz", "IMAGE"),
        ("bilateral_smooth", "IMAGE"),
        ("palette_quantized", "IMAGE"),
        ("canny_edges_viz", "IMAGE"),
        ("roi_sharpened", "IMAGE"),
        ("final_defringed", "IMAGE"),
        ("edge_inpainted", "IMAGE"),
        ("alpha_stepified", "MASK"),
    ]
    a_id = swap_preset(
        g,
        old_type="VR_PipelineLight",
        new_type="VR_PipelineLightDebug",
        new_title="🔬 VectorReady Light DEBUG (A path, 26 outs)",
        output_specs=a_outputs,
        widgets=[0, 3, 1500, "mask_socket", "external_matte"],
        # final_rgb now points at edge_inpainted — the absolute last RGB step
        # in VR_PipelineLight. Saved PNG matches production output.
        final_rgb_slot=24,
        final_alpha_slot=25,
        preview_origin_pos=[3800, -800],
    )
    print(f"A path: debug node id = {a_id}")

    # ── B path ──
    b_outputs = [
        ("input_rgb", "IMAGE"),
        ("native_alpha_viz", "IMAGE"),
        ("alpha_cleaned_viz", "IMAGE"),
        ("bilateral_smooth", "IMAGE"),
        ("canny_edges_viz", "IMAGE"),
        ("kmeans_quantized", "IMAGE"),
        ("region_merged", "IMAGE"),
        ("roi_sharpened", "IMAGE"),
        ("final_defringed", "IMAGE"),
        ("edge_inpainted", "IMAGE"),
        ("alpha_stepified", "MASK"),
    ]
    b_id = swap_preset(
        g,
        old_type="VR_PipelineStrong",
        new_type="VR_PipelineStrongDebug",
        new_title="🔬 VectorReady Strong DEBUG (B path, 11 outs)",
        output_specs=b_outputs,
        widgets=[12, 6.0, 3, 1500, "auto"],
        # final_rgb -> edge_inpainted (the actual last RGB step).
        final_rgb_slot=9,
        final_alpha_slot=10,
        preview_origin_pos=[3800, 800],
    )
    print(f"B path: debug node id = {b_id}")

    # Banner note
    add_node(
        g,
        ntype="MarkdownNote",
        title="📘 v8 DEBUG 说明",
        pos=[3400, 1500],
        widgets=[
            "## v8 DEBUG (A + B 双路径)\n\n"
            "A 路径: VR_PipelineLightDebug (26 outs, incl. final_defringed + edge_inpainted)\n"
            "B 路径: VR_PipelineStrongDebug (11 outs, incl. final_defringed + edge_inpainted)\n\n"
            "**日志**: `/root/ComfyUI/custom_nodes/comfyui_vector_ready/vr_debug.log`\n"
            "**生产输出**: 两条路径的 SaveImage 仍然正常工作\n"
            "**执行**: ExecutionBlocker 仍生效,foreground_mode 切换只跑一条链路"
        ],
    )

    g["last_node_id"] = max((n["id"] for n in g["nodes"]), default=0)
    g["last_link_id"] = max((l[0] for l in g["links"]), default=0)
    DST.write_text(json.dumps(g, ensure_ascii=False, indent=2))
    print(f"wrote {DST} — {len(g['nodes'])} nodes, {len(g['links'])} links")


if __name__ == "__main__":
    main()
