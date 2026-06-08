# ip_consistent 双分支可视化工作流 实现计划

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 `comfyui_workflows` 仓库交付一条可在 ComfyUI 画布验证的 IP 保持重绘工作流——单文件双分支（alpha / autodetect）、脚本派生 debug 版（每阶段预览 + 探针）、独立日志文件。

**Architecture:** 照搬分层项目 v8 范式：手工 UI-graph 基底 + 注入脚本（`build_ip_consistent.py`）生成生产版，再由 `patch_ip_to_debug.py` 派生 debug 版。分支用 `VR_GatedPassthrough`（一个布尔 + invert 驱动两个互斥门，未选路 `ExecutionBlocker` 剪枝）。日志靠给 `VR_RequestBanner` 加 `log_file` 参数把活动日志重绑到 `vr_ip_consistent.log`。

**Tech Stack:** Python 3（纯标准库 json/pathlib 构图）、pytest（结构校验测试）、ComfyUI UI-graph JSON、`comfyui_vector_ready` 自定义节点包。

**Spec:** `comfyui_workflows/docs/inpaint/ip_consistent_design.md`

---

## 文件结构

| 文件 | 职责 |
|---|---|
| `custom_nodes/comfyui_vector_ready/nodes/debug_probe.py` | **改**：加 `set_log_path()` + 让 `LOG_PATH` 可重绑；`VR_RequestBanner` 加 `log_file` 参数 |
| `scripts/_uigraph.py` | **新**：UI-graph 构图助手（按名解析 slot、加节点/连线、图完整性校验）——build 与 patch 共用，避免 v8 脚本式复制 |
| `workflows/inpaint/ip_consistent_base.json` | **新（前置产出）**：autodetect-only 工作流的 UI-graph 基底（画布导出，等价 v7 base） |
| `scripts/build_ip_consistent.py` | **新**：读基底，注入 alpha 分支 + 两个门 + 入口 banner → 生产版 |
| `workflows/inpaint/ip_consistent.json` | **新（脚本生成，勿手改）**：生产版 |
| `scripts/patch_ip_to_debug.py` | **新**：从生产版派生 debug 版（挂 Preview/Probe，置于门下游） |
| `workflows/inpaint/ip_consistent_debug.json` | **新（脚本派生，勿手改）**：debug 版 |
| `scripts/tests/conftest.py` | **新**：把 `custom_nodes` 与 `scripts` 加入 sys.path |
| `scripts/tests/test_log_routing.py` | **新**：`set_log_path` / banner 日志路由单测 |
| `scripts/tests/test_build_ip_consistent.py` | **新**：生产版结构校验 |
| `scripts/tests/test_patch_ip_to_debug.py` | **新**：debug 版结构校验 |

**注**：AGENTS.md 记录本仓库原本无测试套件、无 lint 配置。本计划引入 `scripts/tests/`（pytest）作为构图脚本的自动化校验层；最终验收仍是在 ComfyUI 画布加载（spec 第 8 节）。

**测试环境（执行期确认）**：本机 `.venv`（`LayerForge/.venv`）是后端环境，**无 torch**（ComfyUI 远程运行）。因此 `debug_probe.py` 的 `import torch` 必须改成**防御式导入**（`try/except ImportError` → `torch = None`），与 `gated_passthrough.py` 对 `ExecutionBlocker` 的处理同款（该处注释即 "allows unit tests outside ComfyUI"）。日志相关函数（`set_log_path`/`vr_log`/banner 日志）本就不依赖 torch；torch 仅在 `_stats*`/`VR_SplitRGBA` 被实际张量调用时用到（ComfyUI 运行时）。测试用 `python -m pytest`（pytest 已装入 `.venv`），banner 测试用带 `.shape` 的桩对象、不造真张量。

---

## Chunk 1: 节点改动 —— 独立日志路由

### Task 1: `VR_RequestBanner` 支持 `log_file`，`LOG_PATH` 可重绑

**Files:**
- Modify: `custom_nodes/comfyui_vector_ready/nodes/debug_probe.py`（`LOG_PATH` 区块 + `VR_RequestBanner`）
- Create: `scripts/tests/conftest.py`
- Create: `scripts/tests/test_log_routing.py`

- [ ] **Step 1: 写 conftest 让包可导入**

`scripts/tests/conftest.py`:

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # comfyui_workflows/
# 让 `import comfyui_vector_ready.*` 与 `import _uigraph` / build 脚本可用
sys.path.insert(0, str(ROOT / "custom_nodes"))
sys.path.insert(0, str(ROOT / "scripts"))
```

- [ ] **Step 2: 写失败测试**

`scripts/tests/test_log_routing.py`:

```python
"""VR_RequestBanner.log_file 必须把后续 vr_log 重绑到独立文件，
不传则维持默认 vr_debug.log（不破坏 layered 行为）。"""
import importlib

import pytest

dp = importlib.import_module("comfyui_vector_ready.nodes.debug_probe")


@pytest.fixture
def isolate_plugin_dir(tmp_path, monkeypatch):
    # 把日志根目录指到 tmp，避免污染真实 vr_debug.log
    monkeypatch.setattr(dp, "_PLUGIN_DIR", tmp_path, raising=True)
    monkeypatch.setattr(dp, "_DEFAULT_LOG_PATH", tmp_path / "vr_debug.log", raising=True)
    dp.set_log_path("")  # 复位到默认
    yield tmp_path
    dp.set_log_path("")  # 收尾复位


def test_named_log_file_routes_next_to_plugin(isolate_plugin_dir):
    dp.set_log_path("vr_ip_consistent.log")
    dp.vr_log("T", "hello-ipc")
    target = isolate_plugin_dir / "vr_ip_consistent.log"
    assert target.exists()
    assert "hello-ipc" in target.read_text()
    assert not (isolate_plugin_dir / "vr_debug.log").exists()


def test_empty_resets_to_default(isolate_plugin_dir):
    dp.set_log_path("vr_ip_consistent.log")
    dp.set_log_path("")
    dp.vr_log("T", "back-to-default")
    assert "back-to-default" in (isolate_plugin_dir / "vr_debug.log").read_text()


class _ImgStub:
    """torch-free stand-in: banner() only reads image.shape for the log line."""
    shape = (1, 4, 4, 3)


def test_banner_sets_log_file_before_first_line(isolate_plugin_dir):
    banner = dp.VR_RequestBanner()
    banner.banner(_ImgStub(), tag="ip_consistent", log_file="vr_ip_consistent.log")
    text = (isolate_plugin_dir / "vr_ip_consistent.log").read_text()
    assert "REQUEST START" in text and "ip_consistent" in text
```

- [ ] **Step 3: 跑测试确认失败**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_log_routing.py -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'set_log_path'` / `banner() got unexpected keyword 'log_file'`

- [ ] **Step 3b: 让 `import torch` 防御化（使模块在无 torch 环境可导入）**

把 `debug_probe.py` 顶部的 `import torch` 改为（紧跟现有 import 块）：

```python
try:
    import torch
except ImportError:  # pragma: no cover - allows unit tests outside ComfyUI
    torch = None  # type: ignore
```

这是 enabler：`set_log_path`/`vr_log`/banner 日志不依赖 torch，仅 `_stats*`/`VR_SplitRGBA` 在 ComfyUI 运行时用张量调用。与 `gated_passthrough.py` 同款防御式导入。

- [ ] **Step 4: 改 `debug_probe.py` 的 LOG_PATH 区块**

把现有：

```python
LOG_PATH = Path(os.environ.get("VR_DEBUG_LOG", str(_DEFAULT_LOG_PATH)))
```

改为（在其后追加 setter）：

```python
LOG_PATH = Path(os.environ.get("VR_DEBUG_LOG", str(_DEFAULT_LOG_PATH)))


def set_log_path(name_or_path: str) -> None:
    """Rebind the active log file for the current workflow run.

    Called by VR_RequestBanner at workflow entry. A bare filename routes next
    to the plugin dir (e.g. "vr_ip_consistent.log"); an absolute path is used
    as-is; empty string resets to the VR_DEBUG_LOG env / default path. ComfyUI
    runs one prompt at a time per process, so this module-level rebind is
    request-safe — same contract as set_request_id()."""
    global LOG_PATH
    if not name_or_path:
        LOG_PATH = Path(os.environ.get("VR_DEBUG_LOG", str(_DEFAULT_LOG_PATH)))
        return
    p = Path(name_or_path)
    LOG_PATH = p if p.is_absolute() else (_PLUGIN_DIR / p)
```

`_write_file` 已在每次调用时读模块全局 `LOG_PATH`，重绑即时生效，无需改动。

- [ ] **Step 5: 给 `VR_RequestBanner` 加 `log_file` 参数**

`INPUT_TYPES` 的 `optional` 改为：

```python
            "optional": {
                "request_id": ("STRING", {"default": ""}),
                "log_file": ("STRING", {"default": ""}),
            },
```

`banner` 方法签名与开头改为（**在第一条 vr_log 之前**设好日志）：

```python
    def banner(self, image, tag, request_id="", log_file=""):
        if log_file:
            set_log_path(log_file)
        import random as _r
        rid = str(request_id).strip()
        ...  # 其余不变
```

- [ ] **Step 6: 跑测试确认通过**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_log_routing.py -v`
Expected: PASS（3 passed）

- [ ] **Step 7: 提交**

```bash
cd comfyui_workflows
git add custom_nodes/comfyui_vector_ready/nodes/debug_probe.py scripts/tests/conftest.py scripts/tests/test_log_routing.py
git commit -m "feat(debug): VR_RequestBanner.log_file routes per-workflow logs

Adds set_log_path() so ip_consistent writes to vr_ip_consistent.log instead
of sharing layered's vr_debug.log. Empty log_file keeps default behavior."
```

---

## Chunk 2: 基底 + 构图助手 + 生产版

### Task 2: 取得 autodetect-only UI-graph 基底

**Files:**
- Create: `workflows/inpaint/ip_consistent_base.json`

**背景**：生产版要 UI-graph 格式（画布可开）。现有 autodetect 工作流只有 API 格式（`backend/workflows/inpaint/ip_consistent_generate.json`），外部节点（SAM3 / MaskFix+ / Qwen 系列）的 slot 名/顺序需以真实 ComfyUI 定义为准，**不可靠地从 API 格式硬合成**。因此基底从画布导出。

- [ ] **Step 1: 在 ComfyUI 画布得到 autodetect 工作流**

把 `backend/workflows/inpaint/ip_consistent_generate.json` 在目标 ComfyUI 里跑通（用户已有此环境），确认 33 节点全部解析、无红框。

- [ ] **Step 2: 导出 UI-graph 基底**

ComfyUI 菜单 → Save（**非** "Save (API Format)"）→ 保存为 `workflows/inpaint/ip_consistent_base.json`。确认文件含 `nodes`/`links`/`groups` 顶层键（UI-graph 标志）。

- [ ] **Step 3: 校验基底完整性**

Run:
```bash
cd comfyui_workflows && python -c "
import json
g=json.load(open('workflows/inpaint/ip_consistent_base.json'))
assert {'nodes','links'} <= g.keys(), 'not UI-graph format'
types={n['type'] for n in g['nodes']}
need={'LoadImage','FluxKontextImageScale','VR_MaskSubtract','InvertMask','GrowMask','SetLatentNoiseMask','KSampler','VAEDecode','ImageCompositeMasked','SaveImage','TextEncodeQwenImageEditPlus','VAEEncode'}
missing=need-types
assert not missing, f'base missing nodes: {missing}'
print('base OK,', len(g['nodes']), 'nodes')
"
```
Expected: `base OK, 33 nodes`（数目可略有出入，关键是 `missing` 为空）

- [ ] **Step 4: 提交基底**

```bash
cd comfyui_workflows
git add workflows/inpaint/ip_consistent_base.json
git commit -m "chore(inpaint): add autodetect-only UI-graph base for ip_consistent"
```

> **若画布导出不可得**（无现成画布）：回退方案是用 `scripts/_uigraph.py` 的 `add_node` 按 spec 第 4.1 节拓扑重建基底，但外部节点 slot 名须对照目标 ComfyUI 的节点定义逐一核验。优先走画布导出。

### Task 3: 构图助手 `_uigraph.py`

**Files:**
- Create: `scripts/_uigraph.py`
- Create: `scripts/tests/test_uigraph.py`

- [ ] **Step 1: 写失败测试**

`scripts/tests/test_uigraph.py`:

```python
import _uigraph as u


def _mini_graph():
    return {
        "last_node_id": 2, "last_link_id": 0, "nodes": [
            {"id": 1, "type": "LoadImage", "pos": [0, 0],
             "inputs": [], "outputs": [
                 {"name": "IMAGE", "type": "IMAGE", "links": []},
                 {"name": "MASK", "type": "MASK", "links": []}]},
            {"id": 2, "type": "InvertMask", "pos": [0, 0],
             "inputs": [{"name": "mask", "type": "MASK", "link": None}],
             "outputs": [{"name": "MASK", "type": "MASK", "links": []}]},
        ], "links": [],
    }


def test_out_slot_and_in_slot_by_name():
    g = _mini_graph()
    assert u.out_slot(u.find_by_type(g, "LoadImage"), "MASK") == 1
    assert u.in_slot(u.find_by_type(g, "InvertMask"), "mask") == 0


def test_add_link_wires_both_ends_and_validates():
    g = _mini_graph()
    u.add_link(g, 1, "MASK", 2, "mask", "MASK")
    link = g["links"][0]
    assert link[1] == 1 and link[3] == 2  # src id, dst id
    assert g["nodes"][1]["inputs"][0]["link"] == link[0]
    assert link[0] in u.find_by_type(g, "LoadImage")["outputs"][1]["links"]
    u.assert_graph_valid(g)  # no dangling endpoints


def test_assert_graph_valid_catches_dangling():
    g = _mini_graph()
    g["links"].append([99, 1, 5, 2, 0, "MASK"])  # src slot 5 doesn't exist
    try:
        u.assert_graph_valid(g)
    except AssertionError:
        return
    raise AssertionError("expected dangling link to be caught")


def test_add_node_allocates_id_and_increments():
    g = _mini_graph()
    nid = u.add_node(g, ntype="EmptyImage", title="white", pos=[10, 10],
                     outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
                     widgets=[512, 512, 1, 1.0])
    assert nid == 3 and g["last_node_id"] == 3
    assert u.find_by_type(g, "EmptyImage")["widgets_values"] == [512, 512, 1, 1.0]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_uigraph.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named '_uigraph'`

- [ ] **Step 3: 实现 `scripts/_uigraph.py`**

```python
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
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_uigraph.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 提交**

```bash
cd comfyui_workflows
git add scripts/_uigraph.py scripts/tests/test_uigraph.py
git commit -m "feat(scripts): _uigraph helpers for UI-graph injection (name-resolved slots)"
```

### Task 4: `build_ip_consistent.py` —— 注入 alpha 分支 + 两个门 + banner

**Files:**
- Create: `scripts/build_ip_consistent.py`
- Create: `scripts/tests/test_build_ip_consistent.py`
- Output: `workflows/inpaint/ip_consistent.json`

**注入清单**（基底 = autodetect-only；下列为新增，alpha 分支与门）：

1. 入口 `VR_RequestBanner`（`image` ← LoadImage.IMAGE 经过的 work 图入口；`tag="ip_consistent"`、`log_file="vr_ip_consistent.log"`）。其输出 IMAGE 透传给原工作图链首（把基底里 `ImageScaleToMaxDimension.image` 或 `FluxKontextImageScale` 的输入源改接 banner 输出，用 `replace_input_link`）。
2. **alpha 保护蒙版**：`InvertMask`（mask ← LoadImage.MASK）→ 得 IP=白 的 alpha。
3. **alpha 蒙版对齐尺寸**：把上一步与 `LoadImage.MASK` 重采样到工作图(301)分辨率（见 Step 0 的节点选择校验）。
4. **白底**：`EmptyImage`（白，宽高对齐工作图）。
5. **alpha 条件图**：`ImageCompositeMasked`（destination=白底, source=工作图 RGB, mask=alpha对齐后, resize_source=false）→ IP 叠白底拍平图。
6. **alpha 分支采样尾巴**（复制基底的尾巴）：`VAEEncode(alpha条件图)` → `SetLatentNoiseMask`（mask=alpha 可编辑区=对齐后的 LoadImage.MASK 经 GrowMask）→ `KSampler` → `VAEDecode` → `ImageCompositeMasked`(原像素盖回，mask=alpha保护蒙版) → `SaveImage`。条件编码：两个 `TextEncodeQwenImageEditPlus` 的 `image1` ← alpha 条件图。
7. **两个门**（`VR_GatedPassthrough`，`value` 类型 = LATENT）：
   - autodetect 门：插在**基底** KSampler 的 `latent_image` 前（`replace_input_link`），`enable` ← `use_alpha_mask`，`invert=True`，`label="autodetect"`。
   - alpha 门：插在 alpha 分支 KSampler 的 `latent_image` 前，`enable` ← `use_alpha_mask`，`invert=False`，`label="alpha"`。
8. **选择器**：`PrimitiveNode`(BOOLEAN, title `use_alpha_mask`)。注意 ComfyUI 里布尔常用 `PrimitiveNode` + 子节点 widget；若目标环境用的是 `easy boolean` 之类，按基底惯例选（Step 0 校验）。两个门的 `enable` 输入都连它。

> `invert` 让一个布尔驱动两门：`enable=use_alpha, invert=True` 的 autodetect 门在 `use_alpha=True` 时 BLOCK。

- [ ] **Step 0: 校验注入依赖的节点名/slot（写进脚本顶部常量）**

Run（确认基底里这些 slot 名，并确定蒙版重采样节点）：
```bash
cd comfyui_workflows && python -c "
import json,_uigraph as u,sys; sys.path.insert(0,'scripts')
g=json.load(open('workflows/inpaint/ip_consistent_base.json'))
li=u.find_by_type(g,'LoadImage'); print('LoadImage outs:',[o['name'] for o in li['outputs']])
ks=u.find_by_type(g,'KSampler'); print('KSampler ins:',[i['name'] for i in ks['inputs']])
print('available types for mask-resize:', sorted(t for t in {n['type'] for n in g['nodes']} if 'ale' in t or ' size' in t or 'esize' in t))
"
```
Expected: LoadImage 含 `IMAGE`,`MASK`；KSampler 含 `latent_image`。据输出确定蒙版重采样节点（优先复用基底已用的缩放节点类型，如 essentials `ImageResize+`/`MaskPreview`；ComfyUI 核心 `SetLatentNoiseMask` 会自动适配 latent 尺寸，但盖回用的 `ImageCompositeMasked` 需要蒙版与图同尺寸 → 必须重采样）。把选定节点类型写入 `build_ip_consistent.py` 顶部 `MASK_RESIZE_TYPE` 常量。

- [ ] **Step 1: 写失败测试（结构校验）**

`scripts/tests/test_build_ip_consistent.py`:

```python
"""build_ip_consistent.py 产出的生产版必须：双门、双 SaveImage、
banner 设独立日志、无悬空连线、生产版无任何 Preview/Probe。"""
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
        pytest.skip("base not exported yet (Task 2)")
    subprocess.run([sys.executable, str(ROOT / "scripts/build_ip_consistent.py")],
                   check=True, cwd=ROOT)
    return json.loads(OUT.read_text())


def test_two_gates_one_boolean(graph):
    gates = u.find_all_by_type(graph, "VR_GatedPassthrough")
    assert len(gates) == 2
    labels = {g["widgets_values"][-1] if g["widgets_values"] else "" for g in gates}
    # 一个 invert=True (autodetect)、一个 invert=False (alpha)
    inverts = sorted(bool(g["widgets_values"][1]) for g in gates)
    assert inverts == [False, True]


def test_two_save_images(graph):
    assert len(u.find_all_by_type(graph, "SaveImage")) == 2


def test_banner_sets_independent_log(graph):
    b = u.find_by_type(graph, "VR_RequestBanner")
    assert "vr_ip_consistent.log" in b["widgets_values"]


def test_production_has_no_preview_or_probe(graph):
    types = {n["type"] for n in graph["nodes"]}
    assert not (types & {"PreviewImage", "MaskPreview+",
                         "VR_DebugProbeImage", "VR_DebugProbeMask"})


def test_alpha_branch_present(graph):
    # alpha 分支特征：白底 EmptyImage + 由 LoadImage.MASK 取反的保护蒙版
    assert u.find_all_by_type(graph, "EmptyImage")
    assert len(u.find_all_by_type(graph, "InvertMask")) >= 2  # 基底1 + alpha分支1


def test_graph_valid(graph):
    u.assert_graph_valid(graph)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_build_ip_consistent.py -v`
Expected: FAIL（脚本不存在 → CalledProcessError；或基底未导出 → skip。基底已存在时必须 fail-not-skip）

- [ ] **Step 3: 实现 `scripts/build_ip_consistent.py`**

按"注入清单"用 `_uigraph` 助手实现。骨架（关键调用，slot 名以 Step 0 校验为准）：

```python
"""Build workflows/inpaint/ip_consistent.json from the autodetect-only base by
injecting: entry banner (independent log), alpha mask branch (alpha → white-plate
composite → duplicated sampler tail), and two VR_GatedPassthrough latent gates
driven by one `use_alpha_mask` boolean. Production output has NO previews."""
from __future__ import annotations

from pathlib import Path

import _uigraph as u

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "workflows/inpaint/ip_consistent_base.json"
OUT = ROOT / "workflows/inpaint/ip_consistent.json"

MASK_RESIZE_TYPE = "..."  # 由 Task 4 Step 0 校验确定


def main():
    g = u.load(BASE)

    load = u.find_by_type(g, "LoadImage")
    work = u.find_by_type(g, "FluxKontextImageScale")   # 工作图(301)
    base_ks = u.find_by_type(g, "KSampler")
    protect_sub = u.find_by_type(g, "VR_MaskSubtract")  # 232 保护蒙版(autodetect)

    # 1) 入口 banner —— 透传工作图、设独立日志
    banner = u.add_node(g, ntype="VR_RequestBanner", title="🚩 入口/日志",
                        pos=[-400, -400],
                        inputs=[{"name": "image", "type": "IMAGE", "link": None}],
                        outputs=[{"name": "image", "type": "IMAGE", "links": []},
                                 {"name": "request_id", "type": "STRING", "links": []}],
                        widgets=["ip_consistent", "", "vr_ip_consistent.log"])
    # banner 接在 work 图之后、喂给两个 TextEncode 的 image1 与 VAEEncode 的来源不变；
    # 仅把日志挂上：image ← work.IMAGE（透传，不改下游拓扑）
    u.add_link(g, work["id"], "IMAGE", banner, "image", "IMAGE")

    # 2) 选择器
    sel = u.add_node(g, ntype="PrimitiveNode", title="use_alpha_mask",
                     pos=[-400, 0],
                     outputs=[{"name": "BOOLEAN", "type": "BOOLEAN", "links": []}],
                     widgets=[False])

    # 3) alpha 保护蒙版 + 尺寸对齐
    inv = u.add_node(g, ntype="InvertMask", title="alpha→保护蒙版(IP=白)",
                     pos=[-100, 200],
                     inputs=[{"name": "mask", "type": "MASK", "link": None}],
                     outputs=[{"name": "MASK", "type": "MASK", "links": []}])
    u.add_link(g, load["id"], "MASK", inv, "mask", "MASK")
    # 重采样 inv 与 LoadImage.MASK 到 work 尺寸（MASK_RESIZE_TYPE）—— 省略细节，
    # 按 Step 0 选定节点的 INPUT_TYPES 连线；得 protect_alpha / editable_alpha。

    # 4) 白底 + alpha 条件图（IP 叠白底拍平）
    # 5) alpha 采样尾巴（VAEEncode→SetLatentNoiseMask→KSampler→VAEDecode→
    #    ImageCompositeMasked 盖回→SaveImage），两个 TextEncodeQwenImageEditPlus
    #    的 image1 ← alpha 条件图。
    # （以上用 add_node + add_link 按 spec 4.2 接线）

    # 6) 两个门：插在各自 KSampler 的 latent_image 前
    gate_auto = u.add_node(g, ntype="VR_GatedPassthrough", title="门·autodetect",
                           pos=[base_ks["pos"][0]-260, base_ks["pos"][1]],
                           inputs=[{"name": "value", "type": "LATENT", "link": None},
                                   {"name": "enable", "type": "BOOLEAN", "link": None},
                                   {"name": "invert", "type": "BOOLEAN", "link": None}],
                           outputs=[{"name": "value", "type": "LATENT", "links": []}],
                           widgets=[True, True, "autodetect"])  # enable,invert,label
    # 把基底 KSampler.latent_image 的来源改接 gate_auto，再把原来源接进 gate_auto.value
    src_lid = base_ks["inputs"][u.in_slot(base_ks, "latent_image")]["link"]
    src = next(l for l in g["links"] if l[0] == src_lid)
    u.replace_input_link(g, base_ks["id"], "latent_image", gate_auto, "value", "LATENT")
    u.add_link(g, src[1], u.find_by_id(g, src[1])["outputs"][src[2]]["name"],
               gate_auto, "value", "LATENT")
    u.add_link(g, sel, "BOOLEAN", gate_auto, "enable", "BOOLEAN")
    # alpha 门同理插在 alpha 分支 KSampler 前，widgets=[True, False, "alpha"]

    u.assert_graph_valid(g)
    u.dump(g, OUT)
    print(f"wrote {OUT} ({len(g['nodes'])} nodes)")


if __name__ == "__main__":
    main()
```

> 实现时严格按 spec 4.2 接线；每加一段 `assert_graph_valid` 早暴露悬空。`widgets_values` 顺序须对照各节点 ComfyUI 定义（`VR_GatedPassthrough` = `[enable, invert, label]`，见 `gated_passthrough.py` INPUT_TYPES 顺序）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_build_ip_consistent.py -v`
Expected: PASS（6 passed）

- [ ] **Step 5: 提交**

```bash
cd comfyui_workflows
git add scripts/build_ip_consistent.py scripts/tests/test_build_ip_consistent.py workflows/inpaint/ip_consistent.json
git commit -m "feat(inpaint): build_ip_consistent injects alpha branch + dual gates + banner"
```

---

## Chunk 3: debug 派生 + 验证

### Task 5: `patch_ip_to_debug.py` —— 每阶段 Preview/Probe（置门下游）

**Files:**
- Create: `scripts/patch_ip_to_debug.py`
- Create: `scripts/tests/test_patch_ip_to_debug.py`
- Output: `workflows/inpaint/ip_consistent_debug.json`

**派生规则**（对照 spec 5.3 阶段表）：对每个目标阶段输出，并联一个展示节点（IMAGE→`PreviewImage`，MASK→`MaskPreview+`）+ 一个探针（`VR_DebugProbeImage`/`VR_DebugProbeMask`，label=阶段名，写 `vr_ip_consistent.log`）。**分支内阶段的展示/探针接在该分支门的下游链上**（不接门上游），使未选分支随门一起剪枝。共享阶段（工作图、最终合成）始终展示。

- [ ] **Step 1: 写失败测试**

`scripts/tests/test_patch_ip_to_debug.py`:

```python
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
        pytest.skip("production not built yet (Task 4)")
    subprocess.run([sys.executable, str(ROOT / "scripts/patch_ip_to_debug.py")],
                   check=True, cwd=ROOT)
    return json.loads(OUT.read_text())


def test_has_previews_and_probes(graph):
    types = [n["type"] for n in graph["nodes"]]
    assert types.count("VR_DebugProbeMask") + types.count("VR_DebugProbeImage") >= 8
    assert "PreviewImage" in types and "MaskPreview+" in types


def test_each_probe_labeled(graph):
    for n in graph["nodes"]:
        if n["type"].startswith("VR_DebugProbe"):
            assert n["widgets_values"] and n["widgets_values"][0]  # non-empty label


def test_gates_preserved_and_valid(graph):
    assert len(u.find_all_by_type(graph, "VR_GatedPassthrough")) == 2
    u.assert_graph_valid(graph)


def test_production_untouched(graph):
    # 派生不得改动 PROD 文件本身（debug 是独立产物）
    prod = json.loads(PROD.read_text())
    assert not [n for n in prod["nodes"] if n["type"].startswith("VR_DebugProbe")]
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_patch_ip_to_debug.py -v`
Expected: FAIL（脚本不存在）

- [ ] **Step 3: 实现 `scripts/patch_ip_to_debug.py`**

参考 `scripts/patch_v8_to_debug.py` 的"对每个阶段挂 PreviewImage"做法，改用 `_uigraph` 助手；阶段清单按 spec 5.3。要点：探针为**透传**（probe 输出再喂展示节点；或探针并联、展示并联，二选一——并联更简单，不改主链）。每个探针/展示作为新输出节点并联到目标 slot（`add_link` 到目标节点输出，目标 slot 可有多条 link）。

```python
"""Derive workflows/inpaint/ip_consistent_debug.json from ip_consistent.json:
hang a PreviewImage/MaskPreview+ AND a VR_DebugProbe on each stage listed in
the design doc (section 5.3). Branch-internal stages attach downstream of their
gate so the unselected branch's previews prune with it."""
from __future__ import annotations

from pathlib import Path

import _uigraph as u

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "workflows/inpaint/ip_consistent.json"
DST = ROOT / "workflows/inpaint/ip_consistent_debug.json"


def _probe_mask(g, node_id, out_name, label, pos):
    p = u.add_node(g, ntype="VR_DebugProbeMask", title=f"probe:{label}", pos=pos,
                   inputs=[{"name": "mask", "type": "MASK", "link": None},
                           {"name": "label", "type": "STRING", "widget": {"name": "label"}}],
                   outputs=[{"name": "MASK", "type": "MASK", "links": []}],
                   widgets=[label])
    u.add_link(g, node_id, out_name, p, "mask", "MASK")
    prev = u.add_node(g, ntype="MaskPreview+", title=f"👁{label}", pos=[pos[0]+220, pos[1]],
                      inputs=[{"name": "mask", "type": "MASK", "link": None}], outputs=[])
    u.add_link(g, p, "MASK", prev, "mask", "MASK")
    return p


def _probe_image(g, node_id, out_name, label, pos):
    p = u.add_node(g, ntype="VR_DebugProbeImage", title=f"probe:{label}", pos=pos,
                   inputs=[{"name": "image", "type": "IMAGE", "link": None},
                           {"name": "label", "type": "STRING", "widget": {"name": "label"}}],
                   outputs=[{"name": "IMAGE", "type": "IMAGE", "links": []}],
                   widgets=[label])
    u.add_link(g, node_id, out_name, p, "image", "IMAGE")
    prev = u.add_node(g, ntype="PreviewImage", title=f"👁{label}", pos=[pos[0]+220, pos[1]],
                      inputs=[{"name": "images", "type": "IMAGE", "link": None}], outputs=[])
    u.add_link(g, p, "IMAGE", prev, "images", "IMAGE")
    return p


def main():
    g = u.load(SRC)
    y = -600
    # 共享阶段
    _probe_image(g, u.find_by_type(g, "FluxKontextImageScale")["id"], "IMAGE", "work_image", [1600, y]); y += 200
    # autodetect 分支阶段（门下游）：SAM3/MaskFix/Resolver/cutout/232 …
    _probe_mask(g, u.find_by_type(g, "VR_MaskSubtract")["id"], "MASK", "protect_232", [1600, y]); y += 200
    # alpha 分支阶段：派生时按 title 定位 alpha 节点（InvertMask "alpha→保护蒙版"、白底拍平 composite）
    # 可编辑区(GrowMask)、最终合成(ImageCompositeMasked 盖回) …
    # 逐一 _probe_* 并 += y
    u.assert_graph_valid(g)
    u.dump(g, DST)
    print(f"wrote {DST} ({len(g['nodes'])} nodes)")


if __name__ == "__main__":
    main()
```

> 完整阶段清单按 spec 5.3 表逐行补齐。`MaskPreview+`/`PreviewImage` 的精确 input 名以目标 ComfyUI 定义为准（`MaskPreview+`=`mask`，`PreviewImage`=`images`）。

- [ ] **Step 4: 跑测试确认通过**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/test_patch_ip_to_debug.py -v`
Expected: PASS（4 passed）

- [ ] **Step 5: 全套测试 + 提交**

Run: `cd comfyui_workflows && python -m pytest scripts/tests/ -v`
Expected: 全部 PASS

```bash
cd comfyui_workflows
git add scripts/patch_ip_to_debug.py scripts/tests/test_patch_ip_to_debug.py workflows/inpaint/ip_consistent_debug.json
git commit -m "feat(inpaint): patch_ip_to_debug derives per-stage preview+probe debug graph"
```

### Task 6: ComfyUI 画布验证（最终验收，spec 第 8 节）

**Files:** 无（人工/Agent 在 ComfyUI 操作）

- [ ] **Step 1: 重新生成两版**

Run:
```bash
cd comfyui_workflows
python scripts/build_ip_consistent.py && python scripts/patch_ip_to_debug.py
```
Expected: 两行 `wrote ... (N nodes)`，无异常。

- [ ] **Step 2: 加载 debug 版，确认无红框**

把 `workflows/inpaint/ip_consistent_debug.json` 拖进 ComfyUI。Expected: 所有节点解析、无未知节点红框。

- [ ] **Step 3: autodetect 路验证**（`use_alpha_mask=false`）

喂不透明设计图运行。Expected：232 保护蒙版几何对齐主体；最终图主体零变动；`custom_nodes/comfyui_vector_ready/vr_ip_consistent.log` 出现 `门·autodetect → PASS`、`门·alpha → BLOCK` 与各蒙版统计。

- [ ] **Step 4: alpha 路验证**（`use_alpha_mask=true`）

喂真透明 IP PNG 运行。Expected：保护蒙版极性正确（IP=白）；白底拍平条件图正常；**日志无 SAM3 行**（链被剪枝）；最终图 IP 零变动；`门·alpha → PASS`、`门·autodetect → BLOCK`。

- [ ] **Step 5: 日志隔离验证**

Run: `ls custom_nodes/comfyui_vector_ready/*.log`
Expected: `vr_ip_consistent.log` 存在；其内容不与 layered 的 `vr_debug.log` 互混（grep 各自 tag 确认）。

- [ ] **Step 6: 收尾提交（若画布中微调了基底）**

仅当 Step 2-5 暴露基底问题需修正时，更新基底并重跑脚本后提交。否则本任务无产物。

---

## 备注
- **DRY**：build 与 patch 共用 `_uigraph.py`（不复制 v8 脚本里的 find/add 助手）。
- **YAGNI**：本轮不做后端 API 格式、Agent 路由、matting 集成、参数调优（spec 第 9 节非目标）。
- **每步小提交**：每个 Task 末尾一次 commit；测试先红后绿。
- **外部 slot 名风险**：所有外部节点（SAM3/MaskFix+/Qwen/Empty 系列）的 input/output/widget 名以**目标 ComfyUI 的节点定义**为准，按名解析（`_uigraph.out_slot/in_slot`）已最大化容错；Task 4 Step 0 是集中校验点。
```
