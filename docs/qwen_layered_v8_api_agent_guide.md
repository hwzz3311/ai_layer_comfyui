# Qwen Layered v8 API Workflow Agent Guide

面向多模态 agent 的 ComfyUI API workflow 使用指南。对应文件:

```text
qwen_layered_v8_ab_vector_ready.json
```

该 API workflow 用于把一张设计稿反推为 RGBA 图层。Agent 的职责是提供目标描述、A/B 路径选择和少量采样参数；workflow 负责 SAM3 分割、Qwen Layered 生成、RMBG alpha refinement 和 VectorReady 后处理。

## 最小输入字段

每次调用通常只需要改这些节点:

| 节点 | 字段 | 必填 | 说明 |
|---|---|---:|---|
| `1 LoadImage` | `inputs.image` | 是 | ComfyUI input 目录里的源图文件名 |
| `219 PrimitiveNode (Target Query)` | `inputs.value` | 是 | 主体定位描述，**自动同步**到 `220 LA` 和 `11 SAM3.text`（编辑一处即可） |
| `224 PrimitiveNode (Cutout Query)` | `inputs.value` | 否 | 主体内部镂空描述（卡套卡槽 / 相框窗口 / 圈环洞）；留空 `""` = 不做减法，整条 negative 链零开销 |
| `40 CLIPTextEncode` | `inputs.text` | 是 | Qwen Layered 整图/任务描述，建议英文自然句 |
| `217 VR_GatedPassthrough` | `inputs.enable` | 是 | A 路径 gate |
| `218 VR_GatedPassthrough` | `inputs.enable` | 是 | B 路径 gate |
| `60 KSampler` | `inputs.seed` | 可选 | A 路径 seed |
| `210 KSampler` | `inputs.seed` | 可选 | B 路径 seed |
| `63 SaveImage` | `inputs.filename_prefix` | 可选 | A 输出文件名前缀 |
| `214 SaveImage` | `inputs.filename_prefix` | 可选 | B 输出文件名前缀 |
| `239 VR_VectorReadyReport (A)` | `inputs.layer_label` | 可选 | A 层诊断标签，默认 `A_foreground` |
| `240 VR_VectorReadyReport (B)` | `inputs.layer_label` | 可选 | B 层诊断标签，默认 `B_background` |

不要把中文写进模型 prompt。中文可以用于 agent 内部备注，但实际写入 query / `40.text` 时建议使用英文。

### Target Query 与 Cutout Query 的语义边界

- **Target Query** = "what to extract"。例如 `"card holder frame"`、`"the cat decorations on top"`。
- **Cutout Query** = "what hole(s) the subject contains"。例如卡套传 `"rectangular photo window inside the card holder"`；实心主体（卡通插画、Logo 等）传 `""`。
- v8.2 起，主体 silhouette 减去 cutout silhouette 后再喂给下游 brush / RMBG / VectorReady。这把"镂空"从 Qwen V2 不稳定的生成责任，转为 LA+SAM3 的显式分割责任。
- Cutout Query 是**对称镜像**的 LA+SAM3 通道，参数和正向链一致（dual-prompt: text+bbox 共享 PrimitiveNode）。但 cutout 链**不带 Resolver 矩形兜底**：SAM3 找不到就不减，不会过减成整个 bbox。

## A/B 路径选择

API JSON 里没有 UI workflow 的 `foreground_mode` 节点。路径由两个 gate 字段控制:

```json
{
  "217": {"inputs": {"enable": true}},
  "218": {"inputs": {"enable": true}}
}
```

当前含义:

| 目标路径 | `217.enable` | `218.enable` | 结果 |
|---|---:|---:|---|
| A 前景抽取 | `true` | `true` | A pass，B 被 `invert=true` 阻断 |
| B 背景/主体重建 | `false` | `false` | A 阻断，B pass |

不要设置成 `217=true, 218=false` 或 `217=false, 218=true`，否则 A/B 可能同时执行或同时阻断。

## 输出节点

| 路径 | 保存节点 | 输出前缀默认值 | 输出内容 |
|---|---|---|---|
| A | `63 SaveImage` | `v8_A_foreground_RGBA` | 可见目标前景 RGBA |
| B | `214 SaveImage` | `v8_B_background_RGBA` | 重建后的底层/主体 RGBA |

调试输出:

| 节点 | 内容 |
|---|---|
| `101 PreviewImage` | 缩放后的输入图 |
| `102 MaskPreview+` | SAM3 原始 mask；当前由 LocateAnything bbox 引导 |
| `103 MaskPreview+` | MaskFix 后 mask |
| `104 PreviewImage` | SAM3 分割可视化 |
| `220 PreviewImage` | LocateAnything 矩形框预览 |
| `222 PreviewImage` | TargetMaskResolver 最终目标 mask；SAM 可用时优先 SAM，不可用时回退矩形 |
| `105 PreviewImage` | 红色正向 brush |
| `209 PreviewImage` | 红绿 brush 预览；仅当 MaskFix 后 mask 可用时送入 Qwen Layered |
| `225 PreviewImage` | Brush 是否实际送入 Qwen；绿色=使用，红色=跳过 |
| `106 PreviewImage` | A 路径 Qwen 原始输出 |
| `213 PreviewImage` | B 路径 Qwen 原始输出 |

## VR_VectorReadyReport — 层质量诊断 (v0.15.0+)

`239`（A）和 `240`（B）是新增的 `VR_VectorReadyReport` 直通节点，挂在 `VR_JoinRGBA → SaveImage` 之间。它会对最终 RGBA 做一次轻量统计并通过 `vr_log` 输出一行 JSON 摘要（同时通过 `report_json` 输出口暴露完整 JSON），用于**让 orchestrating agent 自动判断这一层是否需要重跑**。

每次调用都会产出一个 JSON：

```json
{
  "layer_label": "A_foreground",
  "canvas": {"w": 1024, "h": 1024},
  "content_bbox": {"x": 120, "y": 80, "w": 760, "h": 880},
  "content_ratio": 0.6354,
  "transparent_ratio": 0.3142,
  "unique_colors": 24,
  "alpha_levels": 3,
  "connected_components": {
    "count": 3,
    "top_areas": [482190, 1284, 312],
    "largest_ratio": 0.4600,
    "small_island_ratio": 0.0033
  },
  "flags": [],
  "verdict": "clean"
}
```

字段含义和 agent 决策规则：

| 字段 | 含义 | Agent 如何用 |
|---|---|---|
| `verdict` | `clean` / `needs_review` | 直接读取，`needs_review` 触发兜底分支 |
| `flags[]` | 失败标签列表 | 见下表，每个 flag 都建议特定补救动作 |
| `content_ratio` | 内容 bbox 占整幅画面比例 | `< 0.001` 通常说明 SAM3 没找到目标 |
| `unique_colors` | alpha>0 区域内不同 RGB 数 | `> max_colors_clean (32)` 说明 k-means 量化失败 |
| `alpha_levels` | 不同 alpha 值数量 | `> 4` 说明 stepify 没跑或被跳过 |
| `connected_components.small_island_ratio` | 非最大 CC 总面积 / 最大 CC 面积 | `> 0.05` 说明残留斑点未清理 |
| `content_bbox` | 内容真实 bbox | 用来给下游 SVG 服务做 canvas 裁剪 |

常见 flag 与对应建议：

| flag 前缀 | 触发条件 | 建议补救 |
|---|---|---|
| `empty_layer` | 没有 alpha>0 像素 | Target Query 没命中 — 换更具体的英文名词短语，必要时降低 SAM3 threshold |
| `low_content:...` | content_ratio 低于阈值 | 同上；或检查 LA bbox 预览（节点 220）是否落在错误位置 |
| `too_many_colors:N>32` | 颜色数超阈值 | 降低 `VR_PipelineLight/Strong` 内 k-means 的 `n_colors`，或检查 bilateral 是否被跳过 |
| `alpha_not_stepified:N` | alpha 级数 > 4 | 检查 `VR_AlphaStepify` 是否在 pipeline 链路中，多半是上游 RGB-only 没拼 alpha |
| `small_islands:r` | 残留小连通块比例高 | 提高 alpha cleanup 强度，或对原图先做一次去噪 |

调参字段（widget 输入）：

| 字段 | 默认值 | 调整建议 |
|---|---:|---|
| `layer_label` | `layer` | 多层流水线建议传 `A_<name>` / `B_<name>` 以便日志聚合 |
| `max_colors_clean` | `32` | 矢量化服务能处理 ~32 色；若下游能吃更多色，可放宽 |
| `min_content_ratio` | `0.001` | 微小贴纸建议下调到 `0.0002`；大物体保持默认 |
| `small_island_warn_ratio` | `0.05` | 多 instance（多猫、多星星）目标层建议放宽到 `0.2` |

> 节点是直通的（`image` 进 → `image` 出），插入不会改变 `SaveImage` 写出的 PNG。`report_json` 输出口当前在 UI workflow 里**没有连线**——如果要把 JSON 取回 agent，建议在 API 调用时加一个 `SaveText`/`PreviewAny` 节点接到 `239.outputs[0]` 和 `240.outputs[0]`，或直接读取 ComfyUI server 端的 `vr_debug.log`（每次执行一行 `VR_VectorReadyReport label=... verdict=... flags=[...]`）。

## Local LocateAnything Model

`VR_LocateAnythingBox.model_id` 默认指向本地目录:

```text
/root/ComfyUI/models/LocateAnything-3B
```

推荐从 ModelScope 下载到该目录:

```bash
modelscope download --model nv-community/LocateAnything-3B --local_dir /root/ComfyUI/models/LocateAnything-3B
```

也可以用 Python SDK:

```python
from modelscope.hub.snapshot_download import snapshot_download

snapshot_download(
    "nv-community/LocateAnything-3B",
    local_dir="/root/ComfyUI/models/LocateAnything-3B",
)
```

如果放到其他目录,把 `219 VR_LocateAnythingBox` 的 `model_id` 改成该本地目录即可。该节点会把环境变量和 `~` 展开后交给 Transformers `from_pretrained(..., trust_remote_code=True)` 加载。

## Prompt 写法

### `219.inputs.value`: Target Query (主体定位)

v8.2 起，主体语义集中写在 `219 PrimitiveNode (Target Query)` 一个地方，会**自动**分发到 `220 VR_LocateAnythingBox.query` 和 `11 SAM3.text` 两个节点（widget-as-input 转换）。Agent 只改这一个字段即可同步更新 LA + SAM3 双提示。

> ⚠️ 旧版本（v8.1 及之前）让 agent 直接改 `220 LA.query` 或 `11 SAM3.prompt` 已废弃；改 PrimitiveNode 一处更稳。

### `224.inputs.value`: Cutout Query (镂空定位)

对应 negative 链的 PrimitiveNode，同样自动分发到 `225 LA #2.query` 和 `227 SAM3 #2.text`。

- 主体内有镂空（卡套卡槽 / 相框窗口 / 圆环 / 戒指 / 镂空贴纸）：填一句简短英文描述，如 `"rectangular photo window inside the card holder"`、`"circular hole in the center of the ring"`。
- 实心主体：留空 `""`。LA #2 的 query 早返回机制会跳过整条 negative 推理，`230 VR_MaskSubtract` 转为透传，行为与无 negative 链的 v8.1 完全一致。

### 历史回退（仅诊断）

如果临时关闭 LocateAnything 或手工回退到纯文本 SAM3,再使用下面的写法。写法要短、具体、像检测类别。

推荐:

```text
cat
three cats
left cat
right cat
star sticker
heart sticker
photo frame
pink blue photo frame body
black outline
```

不推荐:

```text
把这个可爱的猫咪相框主体抠出来
extract the beautiful cute decorative object with all details
the thing in the middle
foreground
```

规则:

- 用英文。
- 用名词短语，不写长句。
- 单次尽量只指一个语义目标。
- 若目标很多，拆多次跑，例如 `left cat`、`middle cat`、`right cat`。
- 线稿、小圆点这类目标 SAM3 可能不稳，优先用颜色/暗线 mask 或后续专用节点。

### `40.inputs.text`: Qwen Layered 条件 prompt

用途是给 Qwen Layered 一个整图语义和当前任务意图。写法应像训练 caption: 简洁、视觉描述明确、不要命令式过强。

A 路径推荐模板:

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Extract the visible [TARGET] as a clean separate layer with transparent background.
```

B 路径推荐模板:

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Reconstruct the clean underlying [BASE_OBJECT] after removing the selected foreground occluders.
```

示例:

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Extract the visible three cats as a clean separate layer with transparent background.
```

```text
A cute cartoon cat photo frame with pastel pink and blue colors. Reconstruct the clean underlying photo frame body after removing the cats and sticker decorations.
```

不推荐:

```text
cutout
remove
make it good
抠出主体
只要框架不要猫其他都别要
```

规则:

- 用英文自然句。
- 先描述整图，再说明当前目标。
- A 路径使用 `Extract the visible ... as a clean separate layer`。
- B 路径使用 `Reconstruct the clean underlying ... after removing ...`。
- 不要要求多图层一次性输出；当前 workflow 一次只产一个目标层。

## 路径任务映射

| 目标层 | `11.prompt` | `40.text` 任务短语 | 路径 |
|---|---|---|---|
| 完整可见相框 | `cute cat photo frame` | `Extract the visible cute cat photo frame...` | A |
| 三只猫装饰 | `three cats` | `Extract the visible three cats...` | A |
| 单只猫 | `left cat` / `middle cat` / `right cat` | `Extract the visible left cat...` | A |
| 星星贴纸 | `star sticker` | `Extract the visible star stickers...` | A |
| 爱心贴纸 | `heart sticker` | `Extract the visible heart stickers...` | A |
| 黑色线稿 | `black outline` | `Extract the visible black outline details...` | A，但 SAM3 可能不稳 |
| 可见框体 | `photo frame body` | `Extract the visible photo frame body...` | A |
| 干净框体补全 | 遮挡物 prompt，例如 `three cats` 或 `sticker decorations` | `Reconstruct the clean underlying photo frame body after removing...` | B |
| 中间窗口扣洞 | 走 `224.value` (Cutout Query) | 同主体 prompt | 任意路径，由 `230 MaskSubtract` 在 alpha 上扣除 |

## API 修改示例

A 路径抽三只猫（实心主体，无镂空）:

```json
{
  "1": {
    "inputs": {
      "image": "input.png"
    }
  },
  "219": {
    "inputs": {
      "value": "the three cat decorations on top"
    }
  },
  "224": {
    "inputs": {
      "value": ""
    }
  },
  "40": {
    "inputs": {
      "text": "A cute cartoon cat photo frame with pastel pink and blue colors. Extract the visible three cats as a clean separate layer with transparent background."
    }
  },
  "217": {
    "inputs": {
      "enable": true
    }
  },
  "218": {
    "inputs": {
      "enable": true
    }
  },
  "63": {
    "inputs": {
      "filename_prefix": "layer_A_three_cats"
    }
  }
}
```

A 路径抽卡套框体（有镂空 — 中间是照片窗口）:

```json
{
  "1": {
    "inputs": {
      "image": "input.png"
    }
  },
  "219": {
    "inputs": {
      "value": "card holder frame body"
    }
  },
  "224": {
    "inputs": {
      "value": "rectangular photo window inside the card holder"
    }
  },
  "40": {
    "inputs": {
      "text": "A pastel card holder with cat-ear top decorations. Extract the card holder frame body as a clean RGBA layer, with the inner photo slot transparent."
    }
  },
  "217": {
    "inputs": {
      "enable": true
    }
  },
  "218": {
    "inputs": {
      "enable": true
    }
  },
  "63": {
    "inputs": {
      "filename_prefix": "layer_A_card_frame"
    }
  }
}
```

B 路径重建干净框体:

```json
{
  "1": {
    "inputs": {
      "image": "input.png"
    }
  },
  "219": {
    "inputs": {
      "value": "the cat decorations and small sticker decorations on the frame"
    }
  },
  "224": {
    "inputs": {
      "value": ""
    }
  },
  "40": {
    "inputs": {
      "text": "A cute cartoon cat photo frame with pastel pink and blue colors. Reconstruct the clean underlying photo frame body after removing the cats and sticker decorations."
    }
  },
  "217": {
    "inputs": {
      "enable": false
    }
  },
  "218": {
    "inputs": {
      "enable": false
    }
  },
  "214": {
    "inputs": {
      "filename_prefix": "layer_B_clean_frame_body"
    }
  }
}
```

## Node Contract

### Input and segmentation

| 节点 | class | 输入 | 输出 |
|---|---|---|---|
| `1` | `LoadImage` | image filename | source `IMAGE` |
| `5` | `ImageScaleToMaxDimension` | source image | scaled `IMAGE` |
| `10` | `easy sam3ModelLoader` | `sam3-fp16.safetensors` | `sam3_model` |
| `11` | `easy sam3ImageSegmentation` | scaled image + LocateAnything bbox, prompt 留空 | box-guided mask, preview image |
| `20` | `MaskFix+` | SAM3 mask | cleaned SAM target mask |
| `219` | `VR_LocateAnythingBox` | scaled image + query | coarse rectangle mask, bbox, box preview |
| `221` | `VR_TargetMaskResolver` | image + SAM mask + rectangle mask | final target mask for brush construction |

### Brush and conditioning

| 节点 | class | 作用 |
|---|---|---|
| `203` | `ImageCompositeMasked` | 黑底 + 红色目标区，目标 mask 来自 resolver |
| `204` | `GrowMask` | 扩张 resolver 目标 mask |
| `205` | `InvertMask` | 生成绿色负向区域；也是 B 路径 alpha |
| `208` | `ImageCompositeMasked` | 红色目标 + 绿色负向 brush |
| `40` | `CLIPTextEncode` | Qwen 文本条件 |
| `52` | `ReferenceLatent` | 追加原图 latent |
| `53` | `VR_ReferenceLatentIfMaskUsable` | resolver 后 mask 可用时追加 brush latent；SAM 空时可由 LocateAnything 矩形兜底 |

### A path

| 节点 | class | 作用 |
|---|---|---|
| `217` | `VR_GatedPassthrough` | A gate |
| `60` | `KSampler` | Qwen foreground extraction |
| `62` | `VAEDecode` | A RGB/RGBA decode |
| `219` | `VR_HFMattingAlpha` | RMBG-2.0 alpha refinement |
| `220` | `VR_PipelineLight` | A 路径 VectorReady 后处理 |
| `222` | `VR_JoinRGBA` | 合成 RGBA |
| `239` | `VR_VectorReadyReport` | A 层质量诊断（直通） |
| `63` | `SaveImage` | 保存 A 输出 |

### B path

| 节点 | class | 作用 |
|---|---|---|
| `218` | `VR_GatedPassthrough` | B gate，`invert=true` |
| `210` | `KSampler` | Qwen background/body reconstruction |
| `212` | `VAEDecode` | B RGB/RGBA decode |
| `221` | `VR_PipelineStrong` | B 路径 VectorReady 后处理 |
| `223` | `VR_JoinRGBA` | 合成 RGBA |
| `240` | `VR_VectorReadyReport` | B 层质量诊断（直通） |
| `214` | `SaveImage` | 保存 B 输出 |

## 默认参数建议

| 场景 | SAM3 threshold | A steps/cfg | B steps/cfg | 备注 |
|---|---:|---:|---:|---|
| 明确物体，如猫、相框 | `0.4` | `7 / 0.8` | - | 默认即可 |
| 小贴纸、星星 | `0.25-0.4` | `7 / 0.8` | - | mask 漏检时降低 threshold |
| 细线、黑色描边 | 不稳定 | `7 / 0.8` | - | 优先考虑颜色阈值 mask |
| 框体补全 | `0.35-0.5` | - | `16 / 1.0` | prompt 应描述要移除的遮挡物 |

## Agent Checklist

生成 API payload 前，agent 应检查:

```text
1. image 是否已上传到 ComfyUI input 目录
2. 11.prompt 是否为英文短名词短语
3. 40.text 是否为英文视觉 caption + 当前任务
4. A/B gate 是否成对设置
5. 输出 filename_prefix 是否能区分层名
6. 若目标是底层补全，必须走 B
7. 若目标是可见装饰/前景，必须走 A
8. 拿到结果后读 `239 / 240 VR_VectorReadyReport.report_json`，`verdict != "clean"` 时按 flag 表决定是否重跑或换 prompt
```

## 版本变化速查 (v0.12 → v0.15)

Agent 升级到新工作流时关注以下行为差异：

- **v0.12.0**：`VR_MaskUnion(SAM3_refined, LA_box_union)` 接入 negative-cutout 链；当主体含 N 个内部镂空（多窗相册、多卡槽），Cutout Query 用复数集合短语（`"all photo windows"`、`"rectangular photo slots"`）SAM3 能覆盖；剩余 instance 由 LA #2 的 box union 补齐。
- **v0.13.0**：`VR_PipelineLight` / `VR_PipelineStrong` 末尾新增 `clean_transparent_rgb`，A 路径不再有 `alpha=0` 区的 Qwen 噪点；B 路径不再有 stepify 之后 1–2 px 颜色 halo。
- **v0.14.0**：在 v0.13 defringe 之后再加 `_edge_color_inpaint(ring_px=2, radius=3)`，把 alpha 边缘 1–2 px 抗锯齿环 inpaint 成内部色，输出 PNG 不再有背景色描边。
- **v0.15.0**：新增 `VR_VectorReadyReport`（节点 239 / 240），用于让 orchestrating agent 直接拿到层质量裁判。
