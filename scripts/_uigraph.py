"""UI-graph (ComfyUI workflow) construction helpers — shared by
build_ip_consistent.py and patch_ip_to_debug.py.

UI-graph links are positional records:
    [link_id, src_node_id, src_out_slot, dst_node_id, dst_in_slot, type]
Slots are resolved BY NAME so injection is independent of the base's exact
slot ordering."""
from __future__ import annotations

import json
from pathlib import Path


def load(path) -> dict:
    return json.loads(Path(path).read_text())


def dump(g: dict, path) -> None:
    Path(path).write_text(json.dumps(g, ensure_ascii=False, indent=2))


def find_by_type(g, ntype):
    return next(n for n in g["nodes"] if n["type"] == ntype)


def find_all_by_type(g, ntype):
    return [n for n in g["nodes"] if n["type"] == ntype]


def find_by_title(g, title):
    return next(n for n in g["nodes"]
               if n.get("title") == title or n.get("properties", {}).get("Node name for S&R") == title)


def find_by_id(g, nid):
    return next(n for n in g["nodes"] if n["id"] == nid)


def out_slot(node, name) -> int:
    for i, o in enumerate(node.get("outputs", [])):
        if o.get("name") == name:
            return i
    raise KeyError(f"node {node['id']} ({node['type']}) has no output {name!r}")


def in_slot(node, name) -> int:
    for i, inp in enumerate(node.get("inputs", [])):
        if inp.get("name") == name:
            return i
    raise KeyError(f"node {node['id']} ({node['type']}) has no input {name!r}")


def _new_node_id(g) -> int:
    nid = int(g.get("last_node_id", 0)) + 1
    g["last_node_id"] = nid
    return nid


def _new_link_id(g) -> int:
    lid = int(g.get("last_link_id", 0)) + 1
    g["last_link_id"] = lid
    return lid


def add_node(g, *, ntype, title, pos, inputs=None, outputs=None,
             widgets=None, props=None) -> int:
    nid = _new_node_id(g)
    g["nodes"].append({
        "id": nid, "type": ntype, "pos": list(pos), "size": [220, 120],
        "flags": {}, "order": len(g["nodes"]), "mode": 0,
        "inputs": inputs or [], "outputs": outputs or [],
        "properties": props or {"Node name for S&R": ntype},
        "widgets_values": widgets if widgets is not None else [],
        "title": title,
    })
    return nid


def clone_node(g, src_id, *, pos, title=None) -> int:
    """Deep-copy an existing node (preserving its canonical widgets_values and
    slot layout), assign a fresh id, clear all link references so the caller
    rewires explicitly. Used to duplicate a sampler tail without re-authoring
    widget arrays by hand (error-prone for nodes like KSampler that carry a
    control_after_generate slot)."""
    import copy
    src = find_by_id(g, src_id)
    nid = _new_node_id(g)
    node = copy.deepcopy(src)
    node["id"] = nid
    node["pos"] = list(pos)
    node["order"] = len(g["nodes"])
    if title is not None:
        node["title"] = title
    for inp in node.get("inputs", []):
        inp["link"] = None
    for out in node.get("outputs", []):
        out["links"] = []
    g["nodes"].append(node)
    return nid


def remove_node(g, nid) -> None:
    """Delete a node and every link touching it (used to strip the base's
    inherited preview nodes from the lean production graph)."""
    g["nodes"] = [n for n in g["nodes"] if n["id"] != nid]
    g["links"] = [l for l in g["links"] if l[1] != nid and l[3] != nid]
    for n in g["nodes"]:
        for out in n.get("outputs", []):
            if out.get("links"):
                out["links"] = [lid for lid in out["links"]
                                if any(l[0] == lid for l in g["links"])]
        for inp in n.get("inputs", []):
            if inp.get("link") is not None and not any(l[0] == inp["link"] for l in g["links"]):
                inp["link"] = None


def add_link(g, src_id, src_out_name, dst_id, dst_in_name, link_type) -> int:
    src, dst = find_by_id(g, src_id), find_by_id(g, dst_id)
    so, di = out_slot(src, src_out_name), in_slot(dst, dst_in_name)
    lid = _new_link_id(g)
    g["links"].append([lid, src_id, so, dst_id, di, link_type])
    src["outputs"][so].setdefault("links", [])
    if src["outputs"][so]["links"] is None:
        src["outputs"][so]["links"] = []
    src["outputs"][so]["links"].append(lid)
    dst["inputs"][di]["link"] = lid
    return lid


def replace_input_link(g, dst_id, dst_in_name, new_src_id, new_src_out_name, link_type) -> int:
    """Repoint an existing input to a new source (used to insert a gate/probe
    in front of a node). Drops the old link record."""
    dst = find_by_id(g, dst_id)
    di = in_slot(dst, dst_in_name)
    old = dst["inputs"][di].get("link")
    if old is not None:
        g["links"] = [l for l in g["links"] if l[0] != old]
    return add_link(g, new_src_id, new_src_out_name, dst_id, dst_in_name, link_type)


def assert_graph_valid(g) -> None:
    ids = {n["id"] for n in g["nodes"]}
    for l in g["links"]:
        lid, sid, so, did, di, _ = l
        assert sid in ids, f"link {lid}: src node {sid} missing"
        assert did in ids, f"link {lid}: dst node {did} missing"
        src, dst = find_by_id(g, sid), find_by_id(g, did)
        assert so < len(src.get("outputs", [])), f"link {lid}: src slot {so} OOB on {src['type']}"
        assert di < len(dst.get("inputs", [])), f"link {lid}: dst slot {di} OOB on {dst['type']}"
