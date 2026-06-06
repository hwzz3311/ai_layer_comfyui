"""build_ip_consistent.py produces a lean production graph with: two latent
gates (autodetect default ON, alpha default OFF), two SaveImage tails, an entry
banner routing to vr_ip_consistent.log, an alpha branch (white plate + alpha
protection mask), no preview/probe nodes, and no dangling links."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import _uigraph as u

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "workflows/inpaint/ip_consistent.json"


@pytest.fixture(scope="module")
def graph():
    base = ROOT / "workflows/inpaint/ip_consistent_base.json"
    if not base.exists():
        pytest.skip("base not generated yet")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_ip_consistent.py")],
                   check=True, cwd=ROOT)
    return json.loads(OUT.read_text())


def test_two_gates_with_opposite_default_enable(graph):
    gates = u.find_all_by_type(graph, "VR_GatedPassthrough")
    assert len(gates) == 2
    by_label = {gt["widgets_values"][2]: gt for gt in gates}
    assert set(by_label) == {"autodetect", "alpha"}
    assert by_label["autodetect"]["widgets_values"][0] is True   # default ON
    assert by_label["alpha"]["widgets_values"][0] is False        # default OFF


def test_two_save_images(graph):
    saves = u.find_all_by_type(graph, "SaveImage")
    assert len(saves) == 2
    prefixes = {s["widgets_values"][0] for s in saves}
    assert "ip_consistent_alpha" in prefixes


def test_banner_routes_independent_log(graph):
    b = u.find_by_type(graph, "VR_RequestBanner")
    assert "vr_ip_consistent.log" in b["widgets_values"]
    # banner sits in the main path: its IMAGE output is consumed downstream
    assert b["outputs"][0]["links"]


def test_no_preview_or_probe_in_production(graph):
    types = {n["type"] for n in graph["nodes"]}
    assert not (types & {"PreviewImage", "MaskPreview+",
                         "VR_DebugProbeImage", "VR_DebugProbeMask"})


def test_alpha_branch_present(graph):
    types = [n["type"] for n in graph["nodes"]]
    assert "EmptyImage" in types and "GetImageSize+" in types
    assert types.count("InvertMask") >= 2   # base editable + alpha protection/editable
    assert types.count("KSampler") == 2      # duplicated sampler tail
    assert types.count("VAEEncode") == 2


def test_alpha_ksampler_has_seed_control_slot(graph):
    # cloned from base → canonical widgets incl control_after_generate preserved
    for ks in u.find_all_by_type(graph, "KSampler"):
        assert ks["widgets_values"][1] == "fixed"


def test_graph_valid(graph):
    u.assert_graph_valid(graph)
