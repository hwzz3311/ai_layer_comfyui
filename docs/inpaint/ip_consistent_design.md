# ip_consistent — IP 保持重绘工作流（双分支 · 可视化 · 独立日志）

> 设计日期：2026-06-06
> 范围：本轮**只交付 `comfyui_workflows` 内可在 ComfyUI 画布验证 + 分阶段可视化 + 独立日志**的工作流。后端 API 格式同步与 Agent 路由判定是**下一轮**单独的工作，不在本设计内。

## 1. 目标与背景

在"IP 保持设计重绘"任务里，画面主体（IP/角色）必须**字节级零变动**，只重绘背景与周边设计元素。蒙版（要保护的区域）有两种本质不同的来源：

- **alpha 路**：IP 本身是透明图层，alpha 通道即精确蒙版（像素级、确定性、零检测误差）。
- **autodetect 路**：输入是不透明的完整设计图，主体嵌在画面里，需 SAM3 + LocateAnything 检测出主体（有模型误差，作兜底）。

现状：`backend/workflows/inpaint/ip_consistent_generate.json` 已实现 autodetect 路，但它是 **API/prompt 格式**（无法在 ComfyUI 画布打开、无分阶段预览），且没有给开发者看的分阶段可视化、也没有给 Claude 看的结构化日志。

本设计把这条工作流升级为 `comfyui_workflows` 仓库的房屋范式（对照分层项目的 `v8_ab_vector_ready` / `v8_debug`）：单文件双分支 + 脚本派生 debug 版 + 独立日志文件。

## 2. 关键架构约束

### 2.1 两种 JSON 格式（不可混）

| 格式 | 文件 | 用途 |
|---|---|---|
| **API/prompt**（扁平 dict，按 id 存 `class_type`） | `backend/workflows/inpaint/*.json` | 后端 `comfyui.py` 提交给 `/prompt`；ComfyUI 画布**打不开** |
| **UI-graph**（`nodes`/`links`/`groups`/positions） | `comfyui_workflows/workflows/**/*.json` | 拖进 ComfyUI 画布验证、挂 Preview、被 debug 脚本处理 |

本轮所有产物为 **UI-graph 格式**。源头是现有 API 格式 autodetect 图（33 节点）的拓扑，程序化重建为 UI-graph 并注入 alpha 分支。

### 2.2 ComfyUI 是静态 DAG —— 分支靠剪枝，不靠图内 switch

`VR_GatedPassthrough`（`custom_nodes/comfyui_vector_ready/nodes/gated_passthrough.py`）是房屋标准选路件：

- 通配类型（`ANY_TYPE`），`enable=True` 透传输入，`enable=False` 发 `ExecutionBlocker(None)`，ComfyUI 执行引擎**剪枝所有收到它的下游节点**（含 KSampler/VAEDecode/SaveImage/Preview）。
- 带 `invert` 布尔 → **一个选择器布尔可驱动两个互斥门**。
- 自带 `vr_log` 的 `PASS/BLOCK` 行 + `IS_CHANGED` 强制每次重跑（决策必被记录）。

分层项目 `v8_ab_vector_ready` 即用 2×`VR_GatedPassthrough` + 2×`KSampler` + 2×`SaveImage` 实现 A/B 双路，未选路整条上游剪枝。本设计照搬。

## 3. 分支拓扑

### 3.1 选择器

入口一个 `PrimitiveNode` 布尔 **`use_alpha_mask`**（Agent 下一轮负责按输入是否透明设定；本轮验证时手动切）：

```
use_alpha_mask (BOOLEAN)
   ├─→ VR_GatedPassthrough(enable=use_alpha_mask)            → alpha 分支 KSampler.latent_image
   └─→ VR_GatedPassthrough(enable=use_alpha_mask, invert=T)  → autodetect 分支 KSampler.latent_image
```

- `use_alpha_mask=true` → autodetect 分支 KSampler 收 `ExecutionBlocker` → **整条 SAM3/LocateAnything 链不加载**（省显存）。
- `use_alpha_mask=false` → alpha 分支剪枝，走 SAM3 兜底。

### 3.2 收敛方式：复制采样尾巴（照搬 v8）

仓库无"合并两个被门控分支"的节点；房屋做法是**每分支各复制一份采样尾巴，在各自 KSampler 的 `latent_image` 处门控**。两分支下游各一份：

```
… → SetLatentNoiseMask → [VR_GatedPassthrough] → KSampler → VAEDecode → ImageCompositeMasked(原像素盖回) → SaveImage
```

运行时只有一条活，另一条从 KSampler 起整条剪枝（连同它专属的蒙版来源）。

> **取舍记录**：尾巴重复（KSampler/VAEDecode/Composite 各两份）是图层面的冗余，但运行时剪枝保证只跑一条，且这是仓库已验证的范式，优于引入脆弱的图内 switch 或惰性求值。

## 4. 两个蒙版来源

枢纽量是**保护蒙版**（IP=白，可编辑区=黑）。下游 `InvertMask → GrowMask → SetLatentNoiseMask`（仅可编辑区加噪）与 `ImageCompositeMasked`（原 IP 像素盖回）全部 key 在它。

### 4.1 autodetect 路（保持现有链不变）

```
工作图(301)
 ├ LocateAnything(220, positive) ─┐
 ├ SAM3(11, bbox=220) → MaskFix+(20) → VR_TargetMaskResolver(222, SAM优先/矩形兜底) ─┐
 ├ LocateAnything(225, negative-cutout, multi) ─┐                                      │
 └ SAM3(227, bbox=225) → MaskFix+(228) → VR_MaskUnion(230) ─┘                          │
                                          VR_MaskSubtract(232) = 222 − 230  ← 保护蒙版 ─┘
```

prompt 由 Agent 生成（下一轮）：喂给 SAM3/LA 的 `query`（主体描述）+ `308` 设计 prompt。本轮验证用占位文案。

### 4.2 alpha 路（新增 · 零误差）

输入：真透明 IP PNG（`alpha_origin=native`）。ComfyUI `LoadImage` 输出 `IMAGE`(RGB) + `MASK`，其中 **`MASK = 1 − alpha`**（透明区→白(1)，不透明 IP→黑(0)）。

- **保护蒙版**（IP=白）= `InvertMask(LoadImage.MASK)` = alpha 本身。
- **可编辑区**（透明区=白）= `LoadImage.MASK` 直接用（与 autodetect 路 `InvertMask(232)→GrowMask` 对齐：alpha 路同样接 `GrowMask` 轻微膨胀消接缝）。
- **条件图 / 被编码图**（喂 `VAEEncode` 与两个 `TextEncodeQwenImageEditPlus` 的 `image1`）：
  `ImageCompositeMasked(destination=白底, source=IP_rgb, mask=alpha)` →
  **IP 叠在纯白底上的拍平图**。白底由一个 `EmptyImage`(白) 或等价节点生成，尺寸对齐工作图。

  > 理由：透明区 alpha=0 没有有效 RGB，直接编码会把解码器噪声/黑边带进 latent。叠白底后，模型在"白底 + IP"上重绘周边，融合自然。

### 4.3 白底+IP 的处理（不在本工作流内）

"纯白背景 + IP"（不透明、无真 alpha）**不在本工作流内做颜色阈值抠图**（会把 IP 内部白色误镂空）。由 **Agent 上游先调现有 `segment/背景去除.json`（matting/RMBG）工作流**把它变成真透明 PNG，再走本工作流的 alpha-native 分支。matting 模型语义理解主体 vs 背景，不会乱打洞，优于描边/阈值。

> 本设计因此只保留**两条**分支（alpha-native / autodetect）；`alpha_origin=matte` 是上游编排步骤，非本图分支。

## 5. 可视化 + 独立日志层

### 5.1 独立日志文件（解决日志混淆）

`vr_log` 的 `LOG_PATH` 是 `debug_probe.py` 模块级全局，同进程内所有工作流默认都写 `vr_debug.log`。**仅改工作流文件名无法分离日志**。

**方案**：给 `VR_RequestBanner` 增加可选 `log_file` 参数 + 在 `debug_probe.py` 加 `set_log_path(name)` 模块级 setter（与现有 `set_request_id` / `_CURRENT_REQUEST_ID` 同构；ComfyUI 一次只跑一个 prompt，模块级全局安全）。工作流入口的 banner 设 `tag="ip_consistent"`、`log_file="vr_ip_consistent.log"`，此后该工作流所有探针 + 两个门的日志全进 `vr_ip_consistent.log`，与 layered 的 `vr_debug.log` 彻底不混。

> 这是对共享包 `comfyui_vector_ready` 的小改动，需同时更新该包；不破坏现有 layered 行为（`log_file` 不传则维持默认 `vr_debug.log`）。

### 5.2 生产版 vs debug 版（脚本派生，照搬 v8）

- **生产版** `ip_consistent.json`：**无任何 Preview/Probe**，干净轻量。仅保留入口 `VR_RequestBanner`（设独立日志 + 门的 PASS/BLOCK 日志已足够定位走了哪条路）。
- **debug 版** `ip_consistent_debug.json`：由 `patch_ip_to_debug.py` 从生产版**脚本派生**，给每个关键阶段挂 `PreviewImage`/`MaskPreview+`（给开发者看）+ `VR_DebugProbeImage`/`VR_DebugProbeMask`（给 Claude 看，写张量统计到独立日志）。**预览/探针挂在各自门的下游**，使未选分支的预览也随之剪枝，画布只显示活分支的阶段。

### 5.3 展示 + 日志阶段清单

| 阶段 | 展示（开发者） | 日志（Claude，写 `vr_ip_consistent.log`） |
|---|---|---|
| 输入工作图(301) | Preview | banner: tag / req-id / shape |
| autodetect: SAM3原始(11) / MaskFix(20) / Resolver(222) / cutout union(230) / 最终(232) | MaskPreview ×5 | 各蒙版覆盖率 %（`_stats`） |
| alpha: LoadImage.MASK / 派生保护alpha / 白底拍平条件图 | Preview ×3 | alpha 占比 %、RGB 通道统计 |
| 选中分支的保护蒙版 + 可编辑区(GrowMask后) | MaskPreview ×2 | 覆盖率 |
| 条件图 image1 | Preview | shape |
| VAEDecode 原始（盖回前） | Preview | — |
| 最终合成(316) | Preview | — |
| 两个门 | — | PASS / BLOCK（`VR_GatedPassthrough` 自带） |

## 6. 文件与工具结构（照搬 v8 范式）

```
comfyui_workflows/
├── custom_nodes/comfyui_vector_ready/nodes/debug_probe.py   ← 小改：set_log_path() + banner.log_file
├── scripts/
│   ├── build_ip_consistent.py     ← 程序化构建生产版 UI-graph（仿 build_v8_json.py）
│   └── patch_ip_to_debug.py       ← 从生产版派生 debug 版（仿 patch_v8_to_debug.py）
├── workflows/inpaint/
│   ├── ip_consistent.json         ← 生产版（脚本生成，勿手改）
│   └── ip_consistent_debug.json   ← debug 版（脚本派生，勿手改）
└── docs/inpaint/
    └── ip_consistent_design.md    ← 本文件
```

`build_ip_consistent.py` 程序化布局节点（栅格坐标助手），从现有 autodetect 拓扑构建 UI-graph 并注入：alpha 分支、白底条件图、两个门、入口 banner。**两个 JSON 经脚本生成/派生，不手搓、不手改**（与 v8 一致）。

## 7. 涉及的自定义节点（验证环境需已安装）

- 本仓库 `comfyui_vector_ready`：`VR_GatedPassthrough`、`VR_LocateAnythingBox`、`VR_TargetMaskResolver`、`VR_MaskUnion`、`VR_MaskSubtract`、`VR_RequestBanner`、`VR_DebugProbe*`（matting `hf_matting_alpha` 本工作流**不**使用）。
- 外部：`easy sam3ModelLoader` / `easy sam3ImageSegmentation`（ComfyUI-Easy-Use）、`MaskFix+` / `MaskPreview+`（ComfyUI-essentials）、Qwen-Image-Edit-2511 相关（UNETLoader/CLIPLoader/VAELoader/TextEncodeQwenImageEditPlus/ModelSamplingAuraFlow/CFGNorm）。

均为现有 autodetect 工作流已依赖项（用户已在跑），无新增外部依赖。

## 8. 验证方法（本轮交付的验收）

1. `python scripts/build_ip_consistent.py` 生成生产版；`python scripts/patch_ip_to_debug.py` 派生 debug 版。
2. 把 `ip_consistent_debug.json` 拖进 ComfyUI 画布，确认无红框（节点全部解析）。
3. **autodetect 路**（`use_alpha_mask=false`）：喂一张不透明设计图，确认 232 保护蒙版几何对齐主体、最终图主体零变动；`vr_ip_consistent.log` 出现门 `BLOCK`(alpha)/`PASS`(autodetect) 与各蒙版统计。
4. **alpha 路**（`use_alpha_mask=true`）：喂一张真透明 IP PNG，确认：保护蒙版极性正确（IP=白）、白底拍平条件图正常、SAM3 链未执行（日志无 SAM3 行）、最终图 IP 零变动。
5. 检查 `vr_ip_consistent.log` 与 layered 的 `vr_debug.log` **互不混入**。

## 9. 明确的非目标（本轮不做）

- 后端 API/prompt 格式版本与 `workflow_manager.py` 参数注入。
- Agent 路由判定（`should_use_alpha_mask`、`alpha_origin`、上游去背景编排）。
- matting/去背景节点集成（明确放在上游 Agent 步骤，复用现有 `背景去除.json`）。
- 工作流参数调优（步数/CFG/GrowMask 膨胀量等）——验证跑通后据日志再调。
```
