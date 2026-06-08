# Base 全量分层工作流设计（Qwen-Image-Layered 官方多图层）

**日期**: 2026-06-07
**状态**: 设计已批准，待实施
**关联**: 与现有 `workflows/layered/v8_ab_vector_ready.json`（Control 单目标提取）并存互补

---

## 背景与动机

现有 v8 工作流实际使用的是 **Control 模型**（`qwen_image_layered_control_bf16` + `control_v2` LoRA），
属于"prompt + 画笔引导的单目标图层提取"模式，需要 SAM3 / LocateAnything / 画笔 mask 一整套轮廓引导机器。

实测表明，对于一张完整设计稿做**第一步全量拆分**时，官方 **base 多图层模型**
（`Qwen/Qwen-Image-Layered`，一次吐 N 张 RGBA 图层）效果显著优于把 v8 当拆分器用。

两者是**互补**关系，非替代：

- **base 全量分层**（本设计）= 第一步、一键全量拆分，模型自主决定如何分层。
- **v8 control 提取** = 当 base 效果不佳，或用户点名要某个具体部件时的定向精修。

Agent 默认先调 base 做第一步拆分；不满意或需具体部件时再调 v8。

## 关键事实（已核实）

- ComfyUI 已原生支持 Qwen-Image-Layered：`EmptyQwenImageLayeredLatentImage` +
  `UNETLoader` + `LatentCutToBatch` + `VAEDecode` 链路。
- 远程 ComfyUI 已下载 base 权重：
  `/root/ComfyUI/models/diffusion_models/qwen_image_layered_fp8mixed.safetensors`
  （区别于 control 的 `qwen_image_layered_control_bf16.safetensors`）。
- base 模型用每层**自带的 native RGBA alpha** 作为图层轮廓——这与 v8 Control 路径不同。
  仓库 CLAUDE.md 中 "native alpha drops line-art detail" 的结论**仅针对 Control 单层路径**
  （那里 native alpha 标记的是"哪里画了白"）；base 多层模型的 native alpha 本就是每层设计输出，应直接采用。
- base 支持可变层数（references 样例有 3 / 4 / 8）；默认 4。

## 设计决策（已与用户对齐）

| # | 决策 | 结论 |
|---|---|---|
| 1 | 两个独立工作流 vs 合并 | **两个独立工作流**。Agent 路由 = 工具选择，比 boolean mode 参数对 LLM 更友好；避免单图加载两套重模型；base 路径无需背负 v8 复杂度 |
| 2 | 后处理范围 | **新建轻量 composite `VR_PipelineLayered`**，只做"边缘锐化 + alpha 清理"，**不做 k-means/双边**（base RGB 可信，量化会压脏原色） |
| 3 | 图层数 `layers` | **agent 显式指定**（不写死），默认 4 |
| 4 | 全局 prompt | **agent 填写**（不硬编码默认）。配套：把 prompt 编写风格指南写进**分层 agent 对接工具处**（`backend/services/agent/prompts/layer_prompt.py` 或前端工具描述），不在 ComfyUI json 内 |
| 5 | 输出契约 / z-order | 保持模型**原生 batch 顺序**，`SaveImage` 文件名带 z-index（`layer_00`…`layer_NN`，`00` = 最底层），backend 按文件名排序映射画布 z-order |
| 6 | alpha 硬阈值 | **不做 `VR_AlphaStepify`**。base 产物落地是画布 PNG，要保留抗锯齿软边；硬阈值会让边缘发齿 |

## 架构

### 组件 1：ComfyUI 工作流 `workflows/layered/base_layered.json`

从 v8 派生，剥离 Control 专属机器，保留 base 解码链。

**剥掉**：
- SAM3 model loader + 2× `easy sam3ImageSegmentation`
- `VR_LocateAnythingBox` ×2、`MaskFix+` ×2
- 画笔合成系列、`VR_MaskUnion`、`VR_MaskSubtract`、`VR_TargetMaskResolver`
- Control LoRA（`LoraLoaderModelOnly` 节点 31）
- 画笔分支的第二个 `VAEEncode` / `ReferenceLatent`、`VR_ReferenceLatentIfMaskUsable`
- A/B 双 KSampler 的 `VR_GatedPassthrough` 切换、`foreground_mode` Primitive
- `VR_PipelineLight` / `VR_PipelineStrong` 尾部

**⚠️ 修正（2026-06-07，实测后）**：初版从 v8 派生时沿用了 Control 的采样参数，
产出图层数翻倍且质量极差。已改为**严格照搬 ComfyUI 官方 "Image to Layers" 模板**
（`workflows/layered/image_qwen_image_layered_comfyui.json` 的子图，扁平化）。
关键参数（勿改）：

- `UNETLoader` → **base 权重 `qwen_image_layered_fp8mixed.safetensors`，dtype `default`**
  （**不是** `fp8_e4m3fn`；不接 LoRA）
- `CLIPLoader`（qwen_2.5_vl）/ `VAELoader`（qwen_image_layered_vae）
- `LoadImage → ImageScaleToMaxDimension(≤1024) → {GetImageSize, VAEEncode}`
- **条件结构（双路）**：两个 `CLIPTextEncode`（正=agent prompt，负=空字符串），
  各自包一个 `ReferenceLatent`，**两者都喂同一个 VAEEncode latent**。
  正负条件都带图像 reference，只是文本不同。**不用 `ConditioningZeroOut`**。
- `EmptyQwenImageLayeredLatentImage` widgets = `[W, H, layers, 1]`：
  **层数在 widget index 2**，index 3（batch_size）恒为 1。
  （初版把层数放 index 3、index 2=batch=4，导致输出 ×4 翻倍——已修正）
  width/height 由 `GetImageSize` 输入驱动。
- **`KSampler` steps=20, cfg=2.5**, euler, simple, denoise=1
  （初版误用 v8 control 的 7 / 0.8 → 糊；官方原始档 50 / 4.0）
- `LatentCutToBatch('t', 1) → VAEDecode` → N 张 RGBA batch
- → `VR_PipelineLayered`（组件 2）→ `VR_JoinRGBA` → `SaveImage(filename_prefix="layer_")`

**构建方式**（遵循仓库脚本约定，不手改 json）：
- `scripts/build_base_layered_json.py`：按官方模板结构从零构建 → 写 `workflows/layered/base_layered.json`
- `scripts/patch_base_to_debug.py`：派生 `workflows/layered/base_layered_debug.json`，
  把 `VR_PipelineLayered` 换成 `VR_PipelineLayeredDebug`，每个中间阶段接 `PreviewImage`

### 组件 2：`VR_PipelineLayered`（新轻量后处理 composite）

输入 batch 的 N 张 RGBA，逐层清理，不做 k-means / 双边，保留 base 原色。

| 顺序 | 步骤 | 复用 | 作用 |
|---|---|---|---|
| 1 | native alpha resolve | `_resolve_alpha(…, "native")` | 用每层自带 RGBA alpha 作轮廓 |
| 2 | alpha 清理 | `VR_AlphaCleanup` | 形态学 close/open + 去小连通域，清理半透明噪点 / 碎屑 / 孔洞 |
| 3 | defringe | `_clean_transparent_rgb` | alpha≈0 区 decoder 噪声 RGB 清零，去透明区脏色 |
| 4 | 边缘彩边修复 | `_edge_color_inpaint(ring_px=2)` | 最近内部色重写 1–2px alpha 边界环，消背景渗色 |
| 5 | 边缘锐化 | `VR_ROIUnsharpMask` | 仅在 alpha 边界 ROI 做 unsharp，提边缘锐度，不动内部平涂 |

- **不含** `VR_AlphaStepify`（保软边）。
- 输入：`image (IMAGE [B,H,W,4])`；可选参数 `alpha_min_area`、`unsharp_amount/radius` 等沿用对应原子节点默认。
- 输出：`image (IMAGE [B,H,W,4])` 清理后的 N 张 RGBA。
- 按仓库规范注册进 `custom_nodes/comfyui_vector_ready/__init__.py` 的
  `NODE_CLASS_MAPPINGS` + `NODE_DISPLAY_NAME_MAPPINGS`，`CATEGORY = "VectorReady/presets"`。
- 在 `presets/pipeline_debug.py` 镜像 `VR_PipelineLayeredDebug`（每阶段额外输出）。
- 每个阶段边界用 `vr_log("StageName", _stats(tensor))`。

### 组件 3：分层 Agent 工具对接 + prompt 风格指南（LayerForge backend）

- 在 `backend/services/workflow_registry.py` 让 `base_layered` 工作流被自动注册，
  暴露参数：`image`、`prompt`（agent 填）、`layers`（agent 填，默认 4）。
- 在 `backend/services/agent/prompts/layer_prompt.py`（或前端工具描述）补充
  **base prompt 编写风格指南**，告诉分层 agent：
  - 全局 prompt 描述的是**整张图的构图内容**（非 per-layer），英文为主。
  - 何时选 base（第一步全量拆分）vs v8（定向单部件 / base 不佳兜底）。
  - `layers` 取值建议（简单稿 3–4，复杂稿 6–8）。
  - 具体措辞在实施时定稿。

## 后处理决策：默认关闭（2026-06-07，实测后推翻初版）

初版假设"即便 base 输出也需要锐化+alpha 清理"。**实测推翻**：后处理对 base 输出净负向，
尤其 `_edge_color_inpaint` 把细线条**整条变色**（1-2px 线腐蚀后无 interior，整条被当边界环
用周围色重写），ROI 锐化也在边缘过冲偏色。

决定性证据：官方 modelscope Space（质量上限）**零后处理**，直接存 raw RGBA；base 模型在
正确采样参数下 native RGBA 本就干净。故 `base_layered.json` 默认 `VAEDecode → SaveImage`，
与官方一致。`VR_PipelineLayered` 节点保留在插件中，build 脚本 `ENABLE_POSTPROCESS=True`
可重新挂回（仅当未来矢量化等场景确需清理、且能接受变色代价）。

→ 组件 2（`VR_PipelineLayered`）保留代码但**不在默认链路**；组件 1 输出 = 模型原始 RGBA。

## 质量根因分析（2026-06-07，对比官方 Space）

官方 modelscope Space (`app.py`) 用 diffusers `QwenImageLayeredPipeline`，是质量上限。
对比发现：**质量差距主因是参数，不是 fp8 量化**。

证据：ComfyUI 官方模板用同一个 `fp8mixed` 模型即可稳定出好图（官方文档：fp8
"maintains good generation quality"）。ComfyUI 模板的 `steps=20 / cfg=2.5 / 1024`
是**为可用性做的快速档**；模型作者真实设置是 `steps=50 / cfg=4.0 / 640px`。

已将工作流对齐模型真实设置（= 对齐 Space，非随意调参）：

| 参数 | ComfyUI 快速档(原) | 模型真实值(现) | 影响 |
|---|---|---|---|
| `num_inference_steps` | 20 | **50** | 最大 |
| `true_cfg_scale` / KSampler cfg | 2.5 | **4.0** | 中-大 |
| `resolution` (scale max dim) | 1024 | **640**（本版本推荐 bucket） | 大（1024 是 off-bucket） |
| `negative_prompt` | "" | **" "** | 微 |
| `cfg_normalize` | — | diffusers 专属，标准 ComfyUI 无平替 | 次要、暂缺 |
| 精度 | fp8mixed | （可选 bf16 再榨 5-10%，需高显存） | 次要 |

速度/质量权衡：steps=50 是最大质量杠杆也是最大耗时项；如需提速优先降 steps（如 30）。
高清输出可把 `MAX_DIM` 升回 1024（牺牲分层一致性）。

## 数据流

```
原始设计稿
  → LoadImage → scale≤1024 → VAEEncode → ReferenceLatent ┐
agent prompt → CLIPTextEncode ──────────────────────────┤(positive)
                                  ConditioningZeroOut ───┘(negative)
EmptyQwenImageLayeredLatentImage(layers=N) → KSampler → LatentCutToBatch
  → VAEDecode → [N×RGBA] → VR_PipelineLayered → SaveImage(layer_00..layer_NN)
  → backend 按文件名 z-index 排序 → 前端画布多图层
```

## 错误处理 / 边界

- `layers` 越界：build 脚本与节点对 N 做下/上限钳制（如 1–8），超界回退默认 4。
- 某层 native alpha 几乎全空（模型没分出东西）：`VR_AlphaCleanup` 去小连通域后该层近乎全透明，
  `vector_ready_report` 可选标记，便于 agent 决定丢弃；本期不强制过滤（YAGNI，先观察）。
- base 不擅长拆"互相遮挡"实体（references 已注明）——这正是回退 v8 的场景，文档化即可。

## 测试 / 验证

仓库无单测。验证方式：
1. `python scripts/build_base_layered_json.py` 生成 json，ComfyUI 加载无报错。
2. 用第一张设计稿（红底火焰人物卡）跑 base，目检 N 层 RGBA：
   - 边缘锐利、无背景彩边
   - 透明区干净、无 decoder 噪点
   - z-order 自底向上合理
3. `patch_base_to_debug.py` 出 debug 版，查 `vr_debug.log` 每阶段 `_stats`。
4. 与 v8 同图输出对比，确认 base 全量拆分质量更优。

## 范围外（YAGNI）

- 图层语义标注 / 自动命名（事后 VL 标注）。
- base 输出递归再拆分。
- 空图层自动过滤（先观察再决定）。

## 实施顺序

1. `VR_PipelineLayered` 节点 + 注册 + debug 镜像。
2. `scripts/build_base_layered_json.py` + 生成 `base_layered.json`。
3. `scripts/patch_base_to_debug.py` + 生成 `base_layered_debug.json`。
4. ComfyUI 加载验证 + 实图跑通目检。
5. LayerForge backend：workflow_registry 注册 + layer_prompt 风格指南。
6. 端到端：agent 调 base → 画布多图层。
