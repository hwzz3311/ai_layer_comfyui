# Qwen Layered v8 — API Agent Guide

面向 multimodal / orchestrating agent 的 ComfyUI workflow 使用指南。

**契约文件**：`qwen_layered_v8_api.json`（位于仓库根目录）

本指南所有节点 ID、字段名都和该文件**逐一对齐**。改 workflow 之前先确认这两者同步——agent 是按节点 ID 寻址的，ID 漂移 = 接口断裂。

---

## 0. TL;DR

```text
1. 上传源图 → 改 1.image
2. 写主体描述 → 改 219.value（自动喂给 LA #1）+ 11.text（SAM3 #1）
3. 写镂空描述 → 改 224.value（喂给 LA #2）；无镂空填 ""
4. 选路径   → 改 215.value，true=A 前景，false=B 重建
5. 写 caption → 改 40.text
6. 改输出名 → 63.filename_prefix（A）/ 214.filename_prefix（B）
7. POST /prompt，读 63 或 214 的 SaveImage 输出
```

---

## 1. 工作流脉络

源图先缩到 ≤1024（节点 5），然后分三条链：

```
              ┌─→ 正向链：LA#1(220) ─→ SAM3#1(11) ─→ MaskFix+(20) ─┐
缩放图(5) ───┼─→ 镂空链：LA#2(225) ─→ SAM3#2(227) ─→ MaskFix+(228) ─┤
              └─→ 原图 latent(50) → ReferenceLatent(52)             │
                                                                    ▼
              Resolver(222, SAM 优先/LA 矩形兜底) ── MaskSubtract(230, 正 − 镂空)
                                                                    │
              红/绿 brush(201–208) ←──────────────────────────────────┤
                                                                    ▼
              RefLatentIfMaskUsable(53) ──→ KSampler A(60) / B(210)
                                              │              │
                                              ▼              ▼
                                         VAEDecode(62)   VAEDecode(212)
                                              │              │
                                         HFMatting(232)  PipelineStrong(234)
                                              │              │
                                         PipelineLight(233)   │
                                              │              │
                                         JoinRGBA(235)    JoinRGBA(236)
                                              │              │
                                         SaveImage(63)   SaveImage(214)
```

A/B 互斥靠两个 `VR_GatedPassthrough` 在 KSampler 的 `latent_image` 上选通——非选中的支路收到 `ExecutionBlocker`，下游链不执行。

---

## 2. API POST 格式约定

ComfyUI 提交执行的 payload 是**扁平 dict**：

```json
{
  "prompt": {
    "<node_id>": {
      "class_type": "<NodeClass>",
      "inputs": {
        "<field_name>": <value_or_["<src_node_id>", <out_slot>]>
      }
    },
    ...
  },
  "client_id": "<uuid>"
}
```

每个 widget 在 API 格式里都按 **input 名字**展开。比如 `KSampler` 的 widget `[seed, control_after_generate, steps, cfg, sampler_name, scheduler, denoise]` 对应 `inputs.{seed, steps, cfg, sampler_name, scheduler, denoise}`（`control_after_generate` 是前端字段，POST 时不需要）。

> ⚠️ 仓库里的 `qwen_layered_v8_api.json` 是**UI 工作流格式**（含 `nodes/links/groups`）。要拿到真正的 API 提交 payload，需要在 ComfyUI 前端 → 设置开启 Dev mode → Save (API Format)，或调用 ComfyUI 的 `/prompt` validator。本指南把 UI 格式作为「节点 ID 与字段定义来源」，把代码示例写成 API 格式。

---

## 3. Agent 可改字段全集

下表是 agent 每次调用通常**只需改这些字段**。其余字段属于结构契约，不要碰。

| 节点 | class_type | 字段 | 类型 | 必填 | 说明 |
|---:|---|---|---|---:|---|
| `1` | `LoadImage` | `image` | str | ✅ | ComfyUI input 目录里的源图文件名 |
| `215` | `PrimitiveNode` | `value` | bool | ✅ | A/B 模式开关，`true`=A 前景，`false`=B 重建 |
| `219` | `PrimitiveNode` | `value` | str | ✅ | **Target Query**，自动喂给 LA #1 (220) |
| `11` | `easy sam3ImageSegmentation` | `text` | str | ✅ | SAM3 #1 文本提示，**应与 219 同义**（不自动同步） |
| `224` | `PrimitiveNode` | `value` | str | ⭕ | **Cutout Query**，自动喂给 LA #2 (225)；无镂空填 `""` |
| `227` | `easy sam3ImageSegmentation` | `text` | str | ⭕ | SAM3 #2 文本提示，**建议留空 `""`**，纯靠 LA bbox |
| `40` | `CLIPTextEncode` | `text` | str | ✅ | Qwen 整图 caption + 当前任务描述（英文自然句） |
| `60` | `KSampler` | `seed` / `steps` / `cfg` | int/int/float | ⭕ | A 路径采样参数，默认 `seed=随机, steps=7, cfg=0.8` |
| `210` | `KSampler` | `seed` / `steps` / `cfg` | int/int/float | ⭕ | B 路径采样参数，默认 `seed=随机, steps=16, cfg=1.0` |
| `63` | `SaveImage` | `filename_prefix` | str | ⭕ | A 输出文件前缀，默认 `v8_A_foreground_RGBA` |
| `214` | `SaveImage` | `filename_prefix` | str | ⭕ | B 输出文件前缀，默认 `v8_B_background_RGBA` |

**3.1 关于 219 / 224 不能完全替代 SAM3 text**

在这份 API 快照里，`219` 与 `224` 两个 `PrimitiveNode` **只连接了对应的 `VR_LocateAnythingBox.query`**（即 220、225），**没有连到** `easy sam3ImageSegmentation.text`（11、227）。所以：

- 改 Target Query 时建议**同步改 `11.text`**，保持 LA bbox + SAM3 text 双 prompt 一致。
- Cutout 链 SAM3 (`227.text`) 当前默认 `""`，含义是**只信任 LA bbox**，文本不影响分割——多数情况下保持 `""` 即可，agent 一般不需要动它。

### 3.2 A/B 模式开关的真实拓扑

`215 PrimitiveNode` 的 boolean 输出**同时**连到两个 gate 的 `enable` 输入：

- `217 VR_GatedPassthrough` widget `[enable=true, invert=false]` → 接 KSampler **A** 的 `latent_image`
- `218 VR_GatedPassthrough` widget `[enable=true, invert=true]` → 接 KSampler **B** 的 `latent_image`

因为 218 是 `invert=true`，A 和 B 永远互斥。**Agent 只改 `215.value` 一个布尔**，不要去碰 217 / 218 的 widget。

---

## 4. Prompt 写法

### 4.1 `219.value` / `11.text` — Target Query（主体定位）

写法：**英文、名词短语、短且具体**，像目标检测的类别。

| 推荐 | 不推荐 |
|---|---|
| `card holder frame body` | `把这个可爱的猫咪相框抠出来` |
| `three cats` / `left cat` / `right cat` | `the thing in the middle` |
| `pink heart sticker` | `foreground` |
| `black outline of the frame` | `extract everything except background` |

规则：

- 单次只针对一个语义目标；目标多就拆多次跑。
- 多 instance 用集合短语（`three cats`、`all star stickers`）让 LA 一次性框出 N 个，SAM3 文本同步覆盖。
- 线稿、小细节 SAM3 不稳，优先用颜色/暗线 mask 类节点替代。

### 4.2 `224.value` — Cutout Query（主体内部镂空）

什么时候要填：

- **要填**：卡套照片窗口、相框中心镂空、圆环洞、镂空贴纸、戒指。
- **不要填**（留 `""`）：实心猫咪 / Logo / 文字。

写法：和 Target Query 同风格，**句子里强调"inside / hole / window"** 帮助 LA 定位。

| 场景 | Cutout Query |
|---|---|
| 卡套中央照片窗 | `rectangular photo window inside the card holder` |
| 多窗相册 | `all rectangular photo slots` |
| 戒指 | `circular hole in the center of the ring` |
| 镂空贴纸 | `hollow inner shape of the sticker` |

**零开销机制**：当 `224.value=""` 时，`VR_LocateAnythingBox.locate` 在 `locate_anything_box.py:337` 早返回，不加载 3B 模型；下游 SAM3 #2、MaskFix #2、MaskSubtract 全部短路成透传。所以"实心主体"不需要为 cutout 链付任何成本。

### 4.3 `40.text` — Qwen Layered caption

用途：给 Qwen Layered 整图语义 + 当前任务意图。写成 caption 风格、不要命令式。

A 路径模板：

```text
A {scene description}. Extract the visible {TARGET} as a clean separate layer with transparent background.
```

B 路径模板：

```text
A {scene description}. Reconstruct the clean underlying {BASE_OBJECT} after removing the {OCCLUDERS}.
```

示例：

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Extract the visible three cats as a clean separate layer with transparent background.
```

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Reconstruct the clean underlying photo frame body after removing the cats and sticker decorations.
```

---

## 5. 输出与诊断节点

### 5.1 输出

| 路径 | 节点 | `filename_prefix` 默认 | 内容 |
|---|---|---|---|
| A | `63 SaveImage` | `v8_A_foreground_RGBA` | 抽出的可见前景层 RGBA |
| B | `214 SaveImage` | `v8_B_background_RGBA` | 重建的干净底层 RGBA |

未选中的路径被 ExecutionBlocker 阻断，对应 SaveImage 不会写文件。

### 5.2 诊断 Preview（按 ID 排序）

| 节点 | 标题 | 用途 |
|---:|---|---|
| `101` | 诊断1 缩放后输入图 | 确认 LoadImage 正确 |
| `102` | 诊断2 SAM3 原始 mask | 看 SAM3 #1 是否命中 |
| `103` | 诊断3 MaskFix+ 后 mask | 看清理后是否还有目标 |
| `104` | 诊断4 SAM3 分割可视化 | mask 叠加到原图 |
| `105` | 诊断6 红色正向画笔图 | brush 红色区域是否在目标上 |
| `106` | 诊断 A 前景输出预览 | KSampler A 原始解码图 |
| `206` | 诊断7 绿色负向 mask | brush 绿色（膨胀取反）区域 |
| `209` | 诊断8 最终红绿 Brush 图 | V2 LoRA 拿到的画笔图 |
| `213` | 诊断 B 重建输出 | KSampler B 原始解码图 |
| `221` | 诊断10 LA 矩形框（正向） | LA #1 bbox 预览 |
| `223` | 诊断11 Resolver 最终目标 mask | 喂给 brush / RMBG / pipeline 的 mask |
| `226` | 诊断12 LA 矩形框（Cutout） | LA #2 bbox 预览 |
| `229` | 诊断13 Cutout SAM3+MaskFix mask | 镂空链最终 mask |
| `231` | 诊断14 MaskSubtract 最终 alpha | 正 − 镂空后的最终 alpha |
| `238` | 诊断9 Brush 是否送入 Qwen | 绿=送入，红=跳过 |

Agent 调度时**不需要读 preview**，但调试或异常重跑前可以请求 ComfyUI 的 `/history` 接口拿对应节点的中间产物。

---

## 6. 常用 API Payload 范例

下面三个示例假设你已经把 UI workflow 转为 API 格式（包含所有节点的 `class_type` 与不变的 `inputs`），这里只列出 **patch（agent 需覆盖的字段）**。实际提交时把 patch 合并进完整 payload。

### 6.1 A 路径 · 抽三只猫（实心主体）

```json
{
  "1":   {"inputs": {"image": "design.png"}},
  "215": {"inputs": {"value": true}},
  "219": {"inputs": {"value": "three cats"}},
  "11":  {"inputs": {"text":  "three cats"}},
  "224": {"inputs": {"value": ""}},
  "227": {"inputs": {"text":  ""}},
  "40":  {"inputs": {"text":  "A cute cartoon cat photo frame with pastel pink and blue colors. Extract the visible three cats as a clean separate layer with transparent background."}},
  "63":  {"inputs": {"filename_prefix": "layer_A_three_cats"}}
}
```

### 6.2 A 路径 · 抽卡套框体（含中央镂空）

```json
{
  "1":   {"inputs": {"image": "design.png"}},
  "215": {"inputs": {"value": true}},
  "219": {"inputs": {"value": "card holder frame body"}},
  "11":  {"inputs": {"text":  "card holder frame body"}},
  "224": {"inputs": {"value": "rectangular photo window inside the card holder"}},
  "227": {"inputs": {"text":  ""}},
  "40":  {"inputs": {"text":  "A pastel card holder with cat-ear top decorations. Extract the card holder frame body as a clean RGBA layer with the inner photo slot transparent."}},
  "63":  {"inputs": {"filename_prefix": "layer_A_card_frame"}}
}
```

### 6.3 B 路径 · 重建干净框体

```json
{
  "1":   {"inputs": {"image": "design.png"}},
  "215": {"inputs": {"value": false}},
  "219": {"inputs": {"value": "the cat decorations and sticker decorations on the frame"}},
  "11":  {"inputs": {"text":  "cats and stickers"}},
  "224": {"inputs": {"value": ""}},
  "227": {"inputs": {"text":  ""}},
  "40":  {"inputs": {"text":  "A cute cartoon cat photo frame with pastel pink and blue colors. Reconstruct the clean underlying photo frame body after removing the cats and sticker decorations."}},
  "214": {"inputs": {"filename_prefix": "layer_B_clean_frame_body"}}
}
```

> B 路径里 `219 / 11` 描述的是「**要移除的遮挡物**」（猫和贴纸），不是「保留的主体」。Qwen Layered 用 mask 区域作为"重绘提示"，所以 mask 必须落在遮挡物上才能正确补全底层。

---

## 7. 节点契约（按 pipeline 顺序）

### 7.1 输入与分割

| 节点 | class_type | 关键 inputs | 输出 |
|---:|---|---|---|
| `1`   | `LoadImage` | `image` | `IMAGE` |
| `5`   | `ImageScaleToMaxDimension` | `max_dim=1024` | scaled `IMAGE` |
| `10`  | `easy sam3ModelLoader` | `model_name="sam3-fp16.safetensors"` | `SAM3_MODEL` |
| `220` | `VR_LocateAnythingBox` (Target) | `query`←219, `model_id`, `prompt_mode="single"` | mask, **bbox**, preview |
| `11`  | `easy sam3ImageSegmentation` (Target) | `text`(widget), `bboxes`←220 | mask, vis |
| `20`  | `MaskFix+` | SAM3 mask | cleaned target mask |
| `225` | `VR_LocateAnythingBox` (Cutout) | `query`←224, `prompt_mode="multi"` | mask, bbox |
| `227` | `easy sam3ImageSegmentation` (Cutout) | `text=""`, `bboxes`←225 | mask |
| `228` | `MaskFix+` | SAM3 #2 mask | cleaned cutout mask |
| `222` | `VR_TargetMaskResolver` | `sam_mask`←20, `fallback_mask`←220.mask, `threshold=0.5, min_area_ratio=0.002, max_area_ratio=0.90, min_iou_with_fallback=0.02, fallback_dilate_px=0, keep_largest_component=true` | resolved target mask |
| `230` | `VR_MaskSubtract` | `outer`←222, `inner`←228, `inner_dilate_px=2, min_inner_area_ratio=0.0005` | **最终 alpha** |

### 7.2 Brush 与条件

| 节点 | class_type | 作用 |
|---:|---|---|
| `40`  | `CLIPTextEncode` | Qwen 文本条件（agent 写） |
| `54`  | `ConditioningZeroOut` | 负向 ZeroOut |
| `52`  | `ReferenceLatent` | 追加原图 latent 作为 ref |
| `53`  | `VR_ReferenceLatentIfMaskUsable` | `threshold=0.5, min_area_ratio=0.002, max_area_ratio=0.9, max_dim=64`；mask 不可用时**跳过**追加 brush latent |
| `201`–`208` | `VR_EmptyImageLike / ImageCompositeMasked / GrowMask / InvertMask` | 拼红/绿 brush 图，`GrowMask` 膨胀 18 px |

### 7.3 A 路径

| 节点 | class_type | 关键 inputs | 备注 |
|---:|---|---|---|
| `217` | `VR_GatedPassthrough` | `enable`←215, `invert=false` | A gate |
| `60`  | `KSampler` | `seed, steps=7, cfg=0.8, sampler="euler", scheduler="simple", denoise=1.0` | 前景 extraction |
| `62`  | `VAEDecode` | — | A 解码 |
| `232` | `VR_HFMattingAlpha` | `model_id="/root/ComfyUI/models/RMBG-2.0", max_dim=1024, fp_mode="auto"` | RMBG alpha refinement |
| `233` | `VR_PipelineLight` | `bilateral_passes=0, alpha_stepify_steps=3, min_component_px=1500, alpha_source="mask_socket", matting_source="external_matte"` | A 后处理（Light） |
| `235` | `VR_JoinRGBA` | RGB ← 233, alpha ← 232 | A RGBA 合成 |
| `63`  | `SaveImage` | `filename_prefix` | A 输出 |

### 7.4 B 路径

| 节点 | class_type | 关键 inputs | 备注 |
|---:|---|---|---|
| `218` | `VR_GatedPassthrough` | `enable`←215, `invert=true` | B gate（A/B 互斥的关键） |
| `210` | `KSampler` | `seed, steps=16, cfg=1.0, sampler="euler", scheduler="simple", denoise=1.0` | 背景/主体 reconstruction |
| `212` | `VAEDecode` | — | B 解码 |
| `234` | `VR_PipelineStrong` | `kmeans_colors=12, bilateral_passes=6, alpha_stepify_steps=3, min_component_px=1500, alpha_source="auto"` | B 后处理（Strong） |
| `236` | `VR_JoinRGBA` | RGB ← 234, alpha ← Resolver mask 链 | B RGBA 合成 |
| `214` | `SaveImage` | `filename_prefix` | B 输出 |

---

## 8. 模型路径

| 模型 | 节点 | 默认路径 |
|---|---|---|
| LocateAnything 3B | `220` / `225` `model_id` | `/root/ComfyUI/models/LocateAnything-3B` |
| RMBG-2.0 | `232` `model_id` | `/root/ComfyUI/models/RMBG-2.0` |
| SAM3 fp16 | `10` `model_name` | `sam3-fp16.safetensors`（ComfyUI models 目录） |
| Qwen Image Layered UNet | `30` `unet_name` | `qwen_image_layered_control_bf16.safetensors` |
| Qwen Image Layered V2 LoRA | `31` `lora_name` | `qwen_image_layered_control_v2.safetensors` |
| Qwen 2.5 VL CLIP | `33` `clip_name` | `qwen_2.5_vl_7b_fp8_scaled.safetensors` |
| Qwen Image Layered VAE | `34` `vae_name` | `qwen_image_layered_vae.safetensors` |

LocateAnything 下载：

```bash
modelscope download --model nv-community/LocateAnything-3B \
    --local_dir /root/ComfyUI/models/LocateAnything-3B
```

如果路径不同，改 `220.inputs.model_id` 和 `225.inputs.model_id`（两处都要改）。

---

## 9. 默认参数 & 调优指南

| 场景 | 改哪 | 默认 | 建议 |
|---|---|---:|---|
| SAM3 漏检小目标（星星 / 贴纸） | `11.threshold` | `0.4` | 降到 `0.25–0.35` |
| LA 框得太松 / 太紧 | `220.padding_px` | `8` | 紧贴主体用 `0–4`，留 ramp 用 `12–24` |
| Resolver 把 SAM3 判定为不可用 | `222.min_area_ratio` | `0.002` | 极小目标降到 `0.0005` |
| Resolver 把超大主体当不可用 | `222.max_area_ratio` | `0.90` | 满版主体放宽到 `0.99` |
| MaskSubtract 留下 1-2 px 残边 | `230.inner_dilate_px` | `2` | 放到 `4-6` |
| 误删整个主体（cutout 误命中） | `230.min_inner_area_ratio` | `0.0005` | 提高到 `0.005` 让 cutout 必须够大才生效 |
| A 路径细节糊 | `60.steps` / `60.cfg` | `7 / 0.8` | 提到 `10 / 1.0` |
| B 路径补全不干净 | `210.steps` / `210.cfg` | `16 / 1.0` | 提到 `24 / 1.2` |
| B 路径色块碎 | `234.kmeans_colors` | `12` | 单色主体降到 `6–8`，多色降到 `16` |

---

## 10. Agent Checklist

每次提交 prompt 前自检：

```text
[ ] 1.image 已上传到 ComfyUI input 目录
[ ] 215.value 已按目标设好（A=true / B=false）
[ ] 219.value 与 11.text 同义（不要只改一个）
[ ] 224.value 与镂空意图一致；无镂空填 ""
[ ] 40.text 是英文 caption + 当前任务句
[ ] B 路径时 219/11 描述的是"遮挡物"而不是"主体"
[ ] 选中路径对应的 SaveImage filename_prefix 能区分层名
[ ] LA / RMBG / SAM3 模型路径都存在
```

返回结果后检查：

```text
[ ] /history 里对应 prompt_id 状态 success
[ ] 63 或 214 SaveImage 文件已生成
[ ] 若可视化异常，对照第 5.2 节诊断 preview 节点定位是哪一段断
```

---

## 11. 已知限制（此 API 快照）

- **不带 `VR_VectorReadyReport`**：本快照里没有质量诊断节点。如果上游编排需要 verdict + flags，请在 `235.output → 63.input` 之间插入 `VR_VectorReadyReport`，或换用 `workflows/layered/v8_ab_vector_ready.json`（带 239/240）。
- **不带 `VR_MaskUnion`**：本快照的镂空链是 `LA → SAM3 → MaskFix+ → MaskSubtract` 单路。多镂空时只能依赖 LA `prompt_mode="multi"` 一次输出多框，SAM3 文本不参与。若需要 "SAM3 refined ∪ LA boxes" 的多镂空兜底，升级到带 `VR_MaskUnion` 的工作流版本。
- **PrimitiveNode 不覆盖 SAM3 text**：219/224 只接 LA。Agent 必须**同时**写 `11.text`（Target Query 时），否则 SAM3 还会按上一次的 prompt 跑。
- **A/B gate 不要直接改 widget**：217/218 的 `enable` 已被 215 的 link 覆盖，改它们没用；只动 215。
