import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # comfyui_workflows/
sys.path.insert(0, str(ROOT / "custom_nodes"))
sys.path.insert(0, str(ROOT / "scripts"))

# The venv intentionally has no torch/scipy/cv2 (ComfyUI runs remotely).
# Importing comfyui_vector_ready normally triggers its package __init__, which
# eagerly imports many heavy-dependency sibling modules. To exercise only the
# torch-free debug_probe under test, pre-load it by file path under its full
# dotted name (with empty namespace-package stubs for the parents) so that the
# test's importlib.import_module finds it already in sys.modules and never runs
# the heavy package __init__.
_PKG = "comfyui_vector_ready"
_NODES_PKG = f"{_PKG}.nodes"
_DP = f"{_NODES_PKG}.debug_probe"
if _DP not in sys.modules:
    pkg_root = ROOT / "custom_nodes" / "comfyui_vector_ready"
    for name, path in ((_PKG, pkg_root), (_NODES_PKG, pkg_root / "nodes")):
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [str(path)]  # mark as package
            sys.modules[name] = stub
    spec = importlib.util.spec_from_file_location(
        _DP, pkg_root / "nodes" / "debug_probe.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_DP] = mod
    spec.loader.exec_module(mod)
