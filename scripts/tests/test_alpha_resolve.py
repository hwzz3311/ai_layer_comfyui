# Load the module file directly to avoid triggering the package __init__
# (which imports torch-heavy nodes unavailable in this test env).
import importlib.util
from pathlib import Path

import numpy as np

_MOD = (
    Path(__file__).resolve().parents[2]
    / "custom_nodes/comfyui_vector_ready/nodes/alpha_resolve.py"
)
_spec = importlib.util.spec_from_file_location("alpha_resolve", _MOD)
alpha_resolve = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(alpha_resolve)
resolve_alpha_core = alpha_resolve.resolve_alpha_core


def _mask(ratio, shape=(10, 10)):
    m = np.zeros(shape, dtype=np.float32)
    n = int(ratio * m.size)
    m.flat[:n] = 1.0
    return m


def test_prefers_resolved_when_nonempty():
    out, src = resolve_alpha_core(_mask(0.2), _mask(0.5), _mask(0.9), 0.002)
    assert src == "resolved" and out.sum() > 0


def test_falls_back_to_rmbg_when_resolved_empty():
    out, src = resolve_alpha_core(_mask(0.0), _mask(0.5), _mask(0.9), 0.002)
    assert src == "rmbg"


def test_falls_back_to_native_when_both_empty():
    out, src = resolve_alpha_core(_mask(0.0), _mask(0.0), _mask(0.9), 0.002)
    assert src == "native"
