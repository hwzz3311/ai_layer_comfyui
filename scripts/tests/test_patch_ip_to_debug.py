"""patch_ip_to_debug.py taps every listed stage with a probe→preview pair,
preserves the two gates, leaves the production file untouched, and stays valid."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

import _uigraph as u

ROOT = Path(__file__).resolve().parents[2]
PROD = ROOT / "workflows/inpaint/ip_consistent.json"
OUT = ROOT / "workflows/inpaint/ip_consistent_debug.json"


@pytest.fixture(scope="module")
def graph():
    if not PROD.exists():
        pytest.skip("production not built yet")
    subprocess.run([sys.executable, str(ROOT / "scripts/patch_ip_to_debug.py")],
                   check=True, cwd=ROOT)
    return json.loads(OUT.read_text())


def test_probes_for_every_stage(graph):
    probes = (u.find_all_by_type(graph, "VR_DebugProbeMask")
              + u.find_all_by_type(graph, "VR_DebugProbeImage"))
    assert len(probes) == 13  # one per STAGES entry


def test_every_probe_labeled_and_feeds_a_preview(graph):
    for p in graph["nodes"]:
        if p["type"].startswith("VR_DebugProbe"):
            assert p["widgets_values"] and p["widgets_values"][0]
            assert p["outputs"][0]["links"]  # output consumed by a preview


def test_previews_present(graph):
    types = [n["type"] for n in graph["nodes"]]
    assert types.count("PreviewImage") == 6   # work + 2 conditions/decodes ×2 branches + finals
    assert types.count("MaskPreview+") == 7   # 5 autodetect-chain + 2 alpha masks


def test_gates_preserved(graph):
    assert len(u.find_all_by_type(graph, "VR_GatedPassthrough")) == 2


def test_graph_valid(graph):
    u.assert_graph_valid(graph)


def test_production_file_untouched(graph):
    prod = json.loads(PROD.read_text())
    assert not [n for n in prod["nodes"] if n["type"].startswith("VR_DebugProbe")]
