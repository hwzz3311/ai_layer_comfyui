# ip_consistent 工作流 — 进度状态（2026-06-06）

本文件记录 IP 保持重绘工作流这一轮的**已完成 / 未完成 / 风险 / 续作指引**，供下次接续。
配套文档：设计 [`ip_consistent_design.md`](./ip_consistent_design.md)、计划 [`../plans/2026-06-06-ip-consistent-workflow.md`](../plans/2026-06-06-ip-consistent-workflow.md)。

## 0. 一句话状态

双分支（alpha / autodetect）IP 保持重绘工作流的 **UI-graph 生产版 + debug 版已生成、28 项结构测试全绿、关键连线已 trace 验证**。
**唯一未做的关键一步：在真实 ComfyUI 画布加载验证（T6）**——本地无 torch、ComfyUI 远程，外部节点 slot 兼容性只能由画布加载确认。

## 1. 代码位置与分支

- 仓库：`comfyui_workflows`（独立 git 仓库，嵌套在 LayerForge 下）
- 分支：**`feature/ip-consistent-workflow`**（未合并 main）
- 测试：`cd comfyui_workflows && source ../.venv/bin/activate && python -m pytest scripts/tests/ -q` → **28 passed**

### 提交历史（本分支，自 main 起）
```
b731969 feat(inpaint): patch_ip_to_debug taps every stage with probe+preview (T5)
14a8f9d feat(inpaint): build_ip_consistent injects alpha branch + dual gates + banner (T4)
1e2c78f feat(scripts): clone_node/remove_node helpers + converter seed control_after_generate
35517b9 feat(inpaint): API→UI-graph converter + autodetect base for ip_consistent
3e3cbd5 feat(scripts): _uigraph helpers for UI-graph injection (name-resolved slots)
e72c6af feat(debug): VR_RequestBanner.log_file routes per-workflow logs
43b495c docs(inpaint): ip_consistent dual-branch workflow spec + implementation plan
```

## 2. 已完成（✅）

| 任务 | 产物 | 说明 |
|---|---|---|
| **T1 节点改动** | `custom_nodes/comfyui_vector_ready/nodes/debug_probe.py` | `VR_RequestBanner` 加 `log_file` 参数 + `set_log_path()`；`import torch` 防御化（无 torch 也能导入测试）。独立日志 → `vr_ip_consistent.log` |
| **T3 构图助手** | `scripts/_uigraph.py` | 按名解析 slot 的 UI-graph 助手：`add_node/add_link/replace_input_link/clone_node/remove_node/assert_graph_valid` |
| **T2（改为转换器）** | `scripts/api_to_uigraph.py` → `workflows/inpaint/ip_consistent_base.json` | 把 API 格式 autodetect 工作流**忠实转**成 UI-graph 基底（每条连接逐一核验）。源 `ip_consistent_autodetect.api.json` 已 vendored |
| **T4 生产版** | `scripts/build_ip_consistent.py` → `workflows/inpaint/ip_consistent.json`（51 节点） | 注入 alpha 分支 + 双门 + 入口 banner；剥离继承的预览节点。clone 基底节点复用 canonical widgets |
| **T5 debug 版** | `scripts/patch_ip_to_debug.py` → `workflows/inpaint/ip_consistent_debug.json`（77 节点） | 13 个阶段各挂 probe→preview（6 图 + 7 蒙版）；生产版保持无探针 |

### 关键设计落点（已实现）
- **分支**：`use_alpha_mask` 语义由两个 `VR_GatedPassthrough` 的 `enable` widget 控制（alpha 门默认 OFF、autodetect 门默认 ON）。未选分支 KSampler 收 `ExecutionBlocker` → 整条上游剪枝（alpha 模式不跑 SAM3）。
  - ⚠️ 注意：当前是**两个独立 enable 开关**（为 JSON 健壮性，避开 PrimitiveNode 的脆弱接线）。切路要**同时翻转两个门**。"单布尔驱动两门"留给下一轮 Agent 自动写入。
- **alpha 路**：保护蒙版 = `InvertMask(LoadImage.MASK)`；蒙版重采样到工作图尺寸 = **镜像图像缩放链**（`MaskToImage→ImageScaleToMaxDimension→FluxKontextImageScale→ImageToMask`，不依赖第三方 resize 节点）；条件图 = IP 叠白底（`GetImageSize+ → EmptyImage(白) → ImageCompositeMasked`）。
- **autodetect 路**：保持原 SAM3/LocateAnything 链不变（基底节点 11/20/222/230/232）。
- **独立日志**：banner 插在主图路径（`LoadImage→banner→ImageScaleToMaxDimension`）保证必执行，写 `vr_ip_consistent.log`。
- **生产/debug 分离**：脚本生成，**勿手改 JSON**；改了重跑 `build_ip_consistent.py && patch_ip_to_debug.py`。

### 已修正的隐患
- 转换器为 `seed` widget 补 `control_after_generate` 槽，否则 KSampler 的 steps/cfg 会错位（已加测试守护）。
- KSampler widgets 现为 `[seed,'fixed',4,1,'euler','simple',1]`，正确。

## 3. 未完成（⛔ / 下一轮）

### T6 — ComfyUI 画布验证（**唯一卡口，需用户操作**）
本地无法做。步骤见 `ip_consistent_design.md` 第 8 节，要点：
1. **先把改过的 `comfyui_vector_ready` 包部署到 ComfyUI 并重启**（让 `VR_RequestBanner.log_file` 生效）。
2. 导入 `workflows/inpaint/ip_consistent_debug.json`，确认**无红框**。
3. autodetect 路（门·autodetect=ON / 门·alpha=OFF）喂不透明图；alpha 路（反过来）喂真透明 PNG。
4. 看 `vr_ip_consistent.log` 是否生成、门 PASS/BLOCK、蒙版极性（IP=白）、IP 零变动、日志不与 layered 的 `vr_debug.log` 混。

### 明确的非目标（本轮不做，spec 第 9 节）
- 后端 API 格式版本 + `workflow_manager.py` 参数注入。
- Agent 路由判定（`should_use_alpha_mask`、`alpha_origin`、上游去背景编排）。
- 白底+IP 的 matting：明确交给**上游 Agent 先调 `背景去除.json`** 把白底转真透明，再走 alpha 路（不在本工作流内做颜色阈值，避免 IP 内部白色误镂空）。
- 参数调优（步数/CFG/GrowMask 膨胀量等）——跑通后据日志再调。

## 4. 风险 / 待确认（画布加载才能验证）

graph **逻辑已验证**（连线 trace 全对）；剩余不确定性**纯属外部节点定义兼容性**，任一红框把节点名贴回来，改一处 slot 名重跑脚本即可：

| 节点 | 假设 | 若不符的现象 |
|---|---|---|
| `GetImageSize+`（essentials） | 输出 `width/height/count` | 输出名/数量不同 → 红框 |
| `EmptyImage`（白底） | `width/height` 可转输入、`color`=0xFFFFFF | 该版本不支持转输入 → 红框 |
| `VR_GatedPassthrough` 两门 | `enable` 为 widget，翻转切路 | — |
| `easy sam3*` / `MaskFix+` 等 | slot 名沿用 API/转换器赋值，ComfyUI 加载时按 index 重对齐 | 个别 slot 名偏差通常仍可加载 |

> debug 版里 autodetect 蒙版链与 alpha 蒙版**会同时计算**（为对比两种蒙版来源，刻意为之）；只有被选中分支的最终采样图会出。生产版完全剪枝。

## 5. 运行所需模型（工作流引用，需下载到 ComfyUI）

| 文件 | 目录 | 来源 |
|---|---|---|
| `qwen_image_edit_2511_fp8mixed.safetensors` | `models/diffusion_models/` | `Comfy-Org/Qwen-Image-Edit_ComfyUI` → `split_files/diffusion_models/` |
| `qwen_2.5_vl_7b_fp8_scaled.safetensors` | `models/text_encoders/` | 同上 → `split_files/text_encoders/` |
| `qwen_image_vae.safetensors` | `models/vae/` | 同上 → `split_files/vae/` |
| `Qwen-Image-Edit-2511-Lightning-4steps-V1.0-fp32.safetensors` | `models/loras/` | `lightx2v/Qwen-Image-Edit-2511-Lightning` |
| `sam3-fp16.safetensors`（autodetect 路） | SAM3 模型目录 | （原 autodetect 工作流已依赖） |

国内镜像：`export HF_ENDPOINT=https://hf-mirror.com` 后照常 `hf download`。KSampler 已是 steps=4/cfg=1，匹配 4 步 Lightning。

## 6. 如何续作（给下一次的我/接手人）

```bash
cd comfyui_workflows && source ../.venv/bin/activate
python -m pytest scripts/tests/ -q          # 应 28 passed
python scripts/api_to_uigraph.py            # 基底（基本不用重跑）
python scripts/build_ip_consistent.py       # → ip_consistent.json
python scripts/patch_ip_to_debug.py         # → ip_consistent_debug.json
```
- 改工作流：**只改脚本**（`build_ip_consistent.py` / `patch_ip_to_debug.py`），不手改 JSON，改完重跑。
- 收到 ComfyUI 红框反馈：定位对应 `add_node(...)` 的 slot 名/widgets，修正重跑 + 加测试。
- T6 通过后再开下一轮：后端 API 格式版 + Agent 路由（`should_use_alpha_mask` / `alpha_origin` / 上游去背景）。
