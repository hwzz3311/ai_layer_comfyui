"""api_to_uigraph.convert must preserve every API connection as a UI link with
the exact (src_id, src_slot, dst_id) and the same dst-input ordering, keep all
nodes, and preserve widget values — so the converted base is wiring-faithful."""
import json
from pathlib import Path

import _uigraph as u
import api_to_uigraph as conv

ROOT = Path(__file__).resolve().parents[2]
API = ROOT / "workflows/inpaint/ip_consistent_autodetect.api.json"


def _api():
    return json.loads(API.read_text())


def test_node_count_preserved():
    api = _api()
    g = conv.convert(api)
    assert len(g["nodes"]) == len(api)


def test_every_api_link_becomes_a_ui_link():
    api = _api()
    g = conv.convert(api)
    # expected set of (src_id, src_slot, dst_id, dst_input_name)
    expected = set()
    for nid, node in api.items():
        for name, v in node["inputs"].items():
            if conv.is_link(v):
                expected.add((int(v[0]), int(v[1]), int(nid), name))
    got = set()
    for lid, sid, so, did, di, _t in g["links"]:
        dst = u.find_by_id(g, did)
        got.add((sid, so, did, dst["inputs"][di]["name"]))
    assert got == expected
    assert len(g["links"]) == len(expected)


def test_widgets_preserved_for_ksampler_with_seed_control():
    api = _api()
    g = conv.convert(api)
    ks = u.find_by_type(g, "KSampler")
    ks_api = next(n for n in api.values() if n["class_type"] == "KSampler")
    # scalar inputs in order, with "control_after_generate" inserted after seed
    expected = []
    for name, v in ks_api["inputs"].items():
        if conv.is_link(v):
            continue
        expected.append(v)
        if name in ("seed", "noise_seed"):
            expected.append("fixed")
    assert ks["widgets_values"] == expected
    # seed value preserved, control slot present right after it
    assert expected[1] == "fixed" and isinstance(expected[0], int)


def test_mask_subtract_output_name_is_lowercase_mask():
    # guards the T4/T5 name-resolution contract: real RETURN_NAME is "mask"
    g = conv.convert(_api())
    assert u.out_slot(u.find_by_type(g, "VR_MaskSubtract"), "mask") == 0


def test_loadimage_has_image_and_mask_outputs():
    g = conv.convert(_api())
    li = u.find_by_type(g, "LoadImage")
    assert u.out_slot(li, "IMAGE") == 0 and u.out_slot(li, "MASK") == 1


def test_graph_valid_and_ui_format():
    g = conv.convert(_api())
    assert {"nodes", "links"} <= g.keys()
    u.assert_graph_valid(g)
