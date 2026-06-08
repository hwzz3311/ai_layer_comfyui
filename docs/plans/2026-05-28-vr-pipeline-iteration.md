# VR Pipeline 迭代 — 2026-05-28

A 路径(前景提取)输出从"灰黑一坨" → 斑点消除 → 边缘色统一 → 蒙版尺寸/膨胀正确。
跨 v0.5.0 / v0.6.0 / v0.7.0 三轮修复。

## v0.5.0 — 斑点 + 边缘暗化

**问题**:VAE alpha 11.5% 中间值像素 → stepify 后留 2.2% 半透明散点(肉眼可见麻点);premultiply `rgb*=α` 把半透明边缘像素也按 α 衰减 → 边缘暗化。

**修法**:
- 新增 `VR_AlphaCleanup`(median + open + close),stepify 之前先做形态学清理
- `_premultiply` → `_clean_transparent_rgb`:只在 α<0.05 时清零,保留边缘真实颜色
- pipeline 重排:bilateral → canny → unsharp(canny 看到平滑 RGB,边缘更干净)
- `alpha_steps` 默认 3→2,stepify 阈值 0.4/0.6,彻底二值化
- 文件:`nodes/alpha_cleanup.py` 新增;`presets/pipeline*.py` 改

**结果**:`alpha_stepified mid` 从 2.2% → 0.6%。

## v0.6.0 — 边缘画笔色不一致 + Brush 黑带过厚

**问题**:VAE 把黑色画笔反走样成深浅不一的灰,stepify 不透明后保留为多种色 → 矢量化致命;workflow 节点 204 `GrowMask=48` 太大,诊断 7/8 的红绿之间黑色无人区 ~96px,出现空洞。

**修法**:
- Light pipeline 加 `palette_k` 参数(默认 8),插入 LAB→KMeans→LAB⁻¹ 把"接近黑"画笔归到统一调色板色
- `scripts/build_v8_json.py` 加 `GROW_MASK_PX=12` 常量,Stage 4 强制节点 204 widget = 12
- 重新生成 v8 + v8_debug

**结果**:边缘色应统一,但 `K_used=2`(见 v0.7.0)。

## v0.7.0 — KMeans 退化 + 蒙版尺寸 + 黑带微调

**问题三连**:
1. KMeans `auto_k` 在 95% 黑背景的 L 直方图上只看到 2 个峰 → K=2 → 前景所有颜色被平均成一团灰("灰色斑块")
2. 节点 201/202/207(红/黑/绿底)硬编码 `EmptyImage` 1024×1024,标题写"size-matched"但实现没做 → 768×1024 输入产生 1024×1024 蒙版,后续 ReferenceLatent 错位
3. `GROW_MASK_PX=12` 黑带还嫌窄

**修法**:
- Light KMeans 改 `auto_k=False, fixed_k=palette_k`:1 个 cluster 给透明黑,剩 N-1 个给真正前景颜色
- `build_v8_json.py` 加 `BRUSH_BASE_NODES` 配置 + Stage 5:把节点 201/202/207 换成 `VR_EmptyImageLike`,reference 接 node 5(scaled input),自动跟随输入尺寸
- `GROW_MASK_PX` 12 → 18
- 重新生成 v8 + v8_debug

**结果**:`K_used=6~8`,蒙版 = 输入尺寸,黑带 ≈ 36px。

## v0.7.1 — palette_k=3 退化 + 透明区斑点

**症状**(05-28 实测日志):
- `K_used=3`(预期 8),roi_sharpened `per-px spread mean=0.036 → mostly desaturated 50.4%`
- 输出 RGBA 透明区一片黑且布满 3-8px 棕色斑点
- output alpha pct>0.95=5.0% ≈ 39k 像素,两个真前景 blob 仅 ~30k → 多出 ~9k 是噪点

**根因**:
1. `build_v8_json.py` 的 `widgets=[3]` 只给了一个值。Light 的 widget 顺序是 `palette_k, alpha_steps`,所以 `3` 被吃成 **palette_k=3**(K-means 1 cluster 给黑背景,只剩 2 个给前景 → 颜色塌成灰)
2. `VR_AlphaCleanup(median=3, morph=3)` 的 3×3 形态学开运算只能去掉单像素噪声,扛不住 3-8px 的 alpha 噪点团

**修法**:
- `widgets=[3]` → `widgets=[8, 2, 400]`(Light: palette_k=8, alpha_steps=2, alpha_min_area=400)
- Strong 同步加 `alpha_min_area`: `[12, 6.0, 3, 400]`
- `VR_AlphaCleanup` 新增 `min_area` 参数 + `_drop_small_components`:在 morph open/close 之后,基于 `cv2.connectedComponentsWithStats` 把 area < min_area 的连通域整体清零(保留通过的连通域原始灰度值)。`min_area=0` 时禁用(向后兼容)
- 两个 pipeline 把 morph_ksize 从 3 → 5,叠加连通域过滤,新增 `alpha_min_area` 输入(默认 400)
- pipeline_debug.py 同步签名与默认值

**验证**(synthetic test:2 个 100×100 blob + 150 个 2-5px 噪点):
- 过滤前 opaque = 25566;过滤后 = 19992(噪点全清)
- 两个 blob 中心 mean alpha = 1.000(完全保留)

## v0.7.2 — min_area 没生效的两个原因

v0.7.1 实测:`alpha_cleanup pct>0.95` 只从 4.2% → 4.0%(预期 ≥ 0.5% 降幅),OUTPUT alpha 4.8%,斑点几乎没变化。

**原因 1**:`VR_AlphaCleanup` 顺序写错。原顺序 `median → open → close → min_area`,`morph_close(k=5)` 把所有 ≤5px 距离的散点重新桥接到主前景 blob 上变成"刺",`min_area` 看到的是一个超大连通域整体保留 → 散点幸存。
- 修法:reorder 为 `median → open → min_area → close`,先开运算分离散点,再过滤,最后闭运算只填 blob 内孔洞。
- 实测影响:占总噪点的 ~7%,不是主因。

**原因 2(主因)**:`min_area=400` 太低。实际散点 area 在 400-900 px 量级(图里目测 20×30 团块),过滤阈值压根管不住。
- 修法:workflow widget 默认 400 → 1500。两个真前景 blob 加起来 ~30k px,1500 只是 1/20,不会咬真前景。

**节点新参数**:`alpha_min_area` 暴露到 Light/Strong/Debug 4 个 pipeline 节点,用户可在 UI 直接调。

## v0.8.0 — **真正的主问题**:alpha 来源选错了

前面 v0.7.0~0.7.2 一直在调 alpha 噪点过滤,直到用户贴出 Qwen 原图(完整两只猫,带细节但脏)+ VR 输出图(只剩 2 个白色身体块,所有线条/眼/胡须全没了)+ 说"SAM3 识别非常好",才发现真正的 bug。

**根因**:
- Qwen-Image-Layered 的 native RGBA alpha 表示 **"哪里画了白色"**(身体块),不是"猫的轮廓"
- 猫脸上的黑线条(眼、嘴、胡须)在 RGB 是深色,在 native alpha ≈ 0
- `_clean_transparent_rgb` 把 alpha<0.05 处 RGB 清零 → 黑线条被当背景清掉 → 输出只剩白身块
- 数据印证:native alpha pct>0.95=4.2%(只是白身),SAM3 mask pct>0.95=9.6%(两只猫总轮廓,正确)

`_resolve_alpha` 2026-05-27 设的"优先 native"不变量对当前 Qwen 输出**是错的**。

**修法**:
- `_resolve_alpha` 加 `source` 参数:`"auto"`(原行为) / `"native"`(强制) / `"mask_socket"`(强制用外部 mask)
- Light/Strong 节点暴露 `alpha_source` widget(默认 `"auto"`,保持向后兼容)
- A 路径 workflow widget 显式设为 `"mask_socket"` → 用 SAM3 mask 当真实猫轮廓
- B 路径暂保持 `"auto"`(背景重建场景 Qwen 整张 bg 的 alpha 通常正确)
- CLAUDE.md 的相关不变量同步更新

**调用方式**(在 ComfyUI 节点 UI 上):VR_PipelineLight 节点新增下拉框 `alpha_source`,默认 auto,本工作流加载时是 mask_socket。

**验证**:syn 测试(rgba native alpha 只覆盖 1.2%,SAM mask 覆盖 24.4%):pipeline 用 mask_socket → output alpha 跟着 SAM 走(24.4%)✓。

## v0.8.1 — JoinImageWithAlpha 把 alpha 反相了

v0.8.0 把 alpha 源切到 SAM3 后,日志 OK(pct>0.95=10.1% 对应两只猫面积),但保存的 RGBA 是"猫位置反而是空洞"——alpha 被反相了。

**根因**:build 脚本最终的 RGBA 合成节点用的是 ComfyUI **核心** `JoinImageWithAlpha`,而该节点把 MASK 当 selection 用,写入 `1 - alpha`(`custom_nodes/comfyui_vector_ready/nodes/join_rgba.py` 顶部的注释 2026-05-27 就已经记录了这个坑)。VR 早就准备好了 `VR_JoinRGBA` 用 opacity 规范,但 build 脚本没用上。

**修法**:`build_v8_json.py` 把两条路径的 `JoinImageWithAlpha` → `VR_JoinRGBA`,output port 名 `IMAGE` → `rgba` 对齐 VR_JoinRGBA 的 `RETURN_NAMES`。

**验证**:生成的 JSON 中 `VR_JoinRGBA` ×2,`JoinImageWithAlpha` ×0。

## v0.8.2 — Final alpha clamp + transparent RGB clear

用户观察到透明区仍有细小色斑。前段 `_clean_transparent_rgb` 已经在 alpha cleanup 后清一次,但最终输出应该再做一次保险,避免中间处理或 RGBA 合成前残留不可见脏 RGB。

**修法**:`VR_JoinRGBA` 输出前统一做最终 clamp:
- `alpha = clamp(alpha, 0..1)`
- `alpha < 0.05` 的区域强制 `alpha=0`
- 同一区域强制 `RGB=0`

**验证**:`VR_JoinRGBA` 日志新增 `transparent_clamped=...`,synthetic test 中 alpha=0/0.04 的像素输出为 RGBA=0。

## v0.8.3 — Edge-consistency gated line restore

猫头像清晰度不足不是单纯锐化问题,而是 `bilateral + palette_quantize` 会把眼睛、嘴、胡须、轮廓这类高频线稿磨软。直接按 alpha 区域回贴原始 RGB 风险太高,会把遮挡物/其他图层信息带回来。

**修法**:新增 `VR_EdgeConsistencyRestore`:
- `source_image`:alpha 清理后的 Qwen 原始 RGB
- `processed_image`:Light path 的 `roi_sharpened`
- `alpha`:当前图层轮廓
- 对 source / processed 分别做 Canny
- 只回贴 `source_edges ∩ dilate(processed_edges) ∩ alpha` 附近的 source RGB
- `source_edges - dilate(processed_edges)` 输出为 `mismatch_edges`,暂不自动扣 alpha,先作为遮挡/洞诊断信号

**接线**:A 路径 `roi_sharpened → edge_consistency_restore → alpha_stepified/output`。Debug A 路径新增 `restore_mask_viz`,`mismatch_edges_viz`,`edge_restored`。

**验证**:synthetic test restore mask 正常生成;`VR_PipelineLight` smoke test 通过;workflow 重新生成。

## v0.8.4 — A path fidelity mode by default

v0.8.3 实测后,用户观察到猫头像清晰度仍低,边缘笔触颜色不均。原因是 A 路径仍默认执行 `bilateral_smooth + palette_quantize(K=8)`,这对 B 路径重建有价值,但对 A 路径"可信前景 RGB"会过度色块化,磨掉线稿抗锯齿和细节。

**修法**:
- Light 默认 `palette_k=0`
- 当 `palette_k < 2` 时跳过 `bilateral_smooth` 和 `palette_quantize`
- 仍保留 `alpha_cleanup`,`clean_transparent_rgb`,`canny_edges`,`roi_sharpened`,`edge_consistency_restore`,`alpha_stepify`
- workflow A 节点 widgets 改为 `[0, 2, 1500, "mask_socket"]`
- B 路径不变,继续强处理

**验证**:日志出现 `Light [1] bilateral_smooth :: skipped (fidelity mode: palette_k < 2)` 和 `Light [1.5] palette_quantize :: skipped ...`;`VR_PipelineLight` smoke test 通过。

## v0.8.5 — Use original image as gated detail source

v0.8.4 保住了 Qwen RGB,但用户测试后确认 Qwen 原始输出本身已经比原图糊,所以 A 路径的细节源不能只依赖 Qwen。

**修法**:
- `VR_PipelineLight` / `VR_PipelineLightDebug` 新增可选 `source_image`
- workflow 将缩放后的原图节点 5 接入 A 路径 `source_image`
- `VR_EdgeConsistencyRestore` 的细节候选源从 Qwen RGB 改为 `source_image`
- 仍然只在边缘一致区域回贴,不做 alpha 内整块覆盖,避免把其他图层/遮挡物直接带进来
- 未接 `source_image` 时回退到旧行为:使用 Qwen RGB

**验证**:
- 日志出现 `Light INPUT source_image`
- 日志出现 `Light [3.5] detail_source :: original/source_image`
- 生产 workflow A 节点 inputs 为 `image, alpha, source_image`;B 路径不变

## v0.8.6 — Stronger original restore + unsupported SAM area pruning

v0.8.5 清晰度有提升,但仍低于原图,且两猫中间脏块仍存在。原因是 restore 覆盖偏窄,同时最终 alpha 仍基本等于 SAM3 大轮廓;SAM3 认为中间区域属于前景,但 Qwen native alpha 没有实际绘制内容。

**修法**:
- 原图细节回贴放宽:`source_low/high 45/140 → 35/110`,`processed_low/high 35/120 → 30/110`,`match_dilate 3 → 4`,`restore_dilate 1 → 2`,`restore_amount 0.85 → 1.0`
- A 路径新增 content support pruning:
  - `native_content_support`:来自 Qwen native alpha,保留实际绘制的填色区域
  - `edge_content_support`:来自原图一致边缘,保留线稿/胡须/眼睛等细节
  - `gated_edge_support`:原图边缘必须靠近 Qwen native content anchor,不能单独保留远离 Qwen 内容的噪声块
  - 最终 `alpha = SAM3_alpha ∩ content_support`
- 同步将 unsupported 区域 RGB 清零,避免中间脏块进入最终 RGBA

**验证**:synthetic test 中 SAM 多出的 unsupported blob alpha mean 从 1.0 → 0.0;真实图日志重点看 `content_pruned_alpha` 是否低于原 `alpha_stepified` 且猫主体不被咬掉。

## v0.8.7 — Structured Layer Matting Refine

用户明确指出继续调阈值/半径已经变成打补丁,需要把 A 路径升级为一个更完整的处理流程。v0.8.7 新增 `VR_LayerMattingRefine`,把 Qwen RGBA、SAM mask、原图 RGB 三类信号统一进一个结构化阶段。

**节点契约**:`VR_LayerMattingRefine(qwen_image, sam_mask, original_image) → rgb, alpha, confidence, hole_mask, detail_mask`

**流程**:
- Qwen native alpha → content existence / sure foreground
- SAM mask → broad silhouette,不再直接等于最终 alpha
- 原图 RGB → guided matting 的 guide + 高频 detail source
- 原图边缘必须与 Qwen 边缘/anchor 一致才进入 `detail_mask`
- `hole_mask = SAM 内但没有 Qwen/content/detail 支撑的区域`,用于标记 occluded/missing 而不是在 A 路径硬填
- guided filter 对 trimap alpha 做可解释的 matte refinement

**A 路径变化**:
- `VR_PipelineLight` 在 alpha cleanup 后先调用 `VR_LayerMattingRefine`
- 删除旧的 edge restore + content prune 小规则链
- 后续只保留轻量 canny/ROI sharpen/alpha stepify
- Debug A 路径新增 `matting_rgb`,`confidence_viz`,`hole_viz`,`detail_viz`

**验证**:
- synthetic test:额外 unsupported SAM blob 的最终 alpha mean 为 0.0
- `VR_PipelineLightDebug` 输出 12 个端口
- workflow 已重新生成

## 关键不变量(后续修改务必保留)

- `_resolve_alpha`:优先用 RGBA 内嵌 alpha,不用 MASK socket(2026-05-27 bug 来源)
- `_clean_transparent_rgb`:**只**在 α<0.05 清零,不要回退到全量 premultiply
- Light/Strong pipeline 必须镜像到 `pipeline_debug.py`,debug 节点端口数 = 生产节点逻辑阶段数
- `build_v8_json.py` 的 Stage 4/5 修改 workflow JSON,**不要手改 JSON**,改完跑 build 脚本即可重生
- KMeans 在前景占比小的场景必须固定 K,不要用 `auto_k`(adaptive 估计器被透明区主导)

## v0.8.8: source composer,把 A 路径切到"来源分区融合"

用户明确纠正:Qwen Layers V2 本身已经具备分层 + 遮挡补全能力,主线不应变成普通重绘修复;应围绕 Qwen 的图层先验和补全能力做稳定路由。

本轮新增 `VR_LayerSourceComposer`:

- `support_mask` 区域:使用原图 RGB,用于可见内容的高保真回贴
- `completion_mask` 区域:使用 Qwen RGB,预留给后续黄色画笔 / 遮挡补全
- `candidate_mask` 内但没有 support/completion 的区域:输出为 transparent/low_confidence,不再因为 SAM 大轮廓而保留
- 输出诊断图:`original_region`,`qwen_region`,`transparent_region`,`low_confidence`

当前 A 路径接法:

- candidate = alpha cleanup 后的外部轮廓(SAM)
- support = `VR_LayerMattingRefine` 的 confidence
- completion = 空,所以 A 路径不会使用 Qwen RGB 做补全
- 结果:可见可信区回贴原图,unsupported SAM 区域透明化

下一步 B 路径会把黄色画笔/补全区域接到 `completion_mask`,形成"非遮挡区原图优先,遮挡区 Qwen 补全"的正式主线。

## v0.8.9: 收紧 A 路径细节支撑,避免孤立纹理块变成可信原图

v0.8.8 实测后,猫脸/线条已经接近原图,但中间脏块仍存在。日志显示 `qwen_region=0`,所以问题不在补全/融合,而是 `VR_LayerMattingRefine` 的 `confidence/detail_mask` 把中间区域误判成了原图可信区。

调整:

- `detail_mask` 新增 connected component 约束:原图细节边缘必须连接到 Qwen native alpha 的近邻 anchor,孤立在 SAM 候选区内的纹理/脏块不再进入 support
- detail 膨胀后再次限制在 anchor ∩ SAM 内,避免膨胀越界
- A 路径 ROI sharpen 的 edge map 额外与 `original_region` 相交,只锐化可信原图区域,不再锐化透明/低置信污染边缘

预期日志变化:

- `detail_mask` 和 `confidence_mask` 均应略降
- `transparent_region/low_confidence` 在中间脏块区域应上升
- `canny_edges` 面积应下降,边缘外白边/污染不会再被强化

## v0.8.10: A 路径 native alpha 降级为弱先验,增加原图一致性验证

v0.8.9 实测后,中间脏块仍存在;日志显示 `qwen_region=0`,但 `confidence_mask` 仍把该区域纳入原图可信区。原因是 A 路径把 Qwen native alpha 当作 `sure_fg`,而 Qwen 在两猫之间自生成的桥状内容也带 native alpha,于是被当成真实前景。

调整:

- `native_alpha > threshold` 不再直接成为 `native_core`
- 新增 per-image 自适应一致性门控:`Qwen RGB` 必须与 `source_image/original RGB` 在候选区域内颜色分布一致,才可进入 `native_core`
- 阈值由当前图像内的 Qwen/原图颜色差分布通过 Otsu/分位数估计,不是为猫图写固定颜色规则

通用原则:

- A 路径是可见图层抽取,所以 Qwen-only 内容只能做弱先验
- 与原图不一致的 Qwen 生成区域不能成为 sure foreground
- 后续 B 路径才会通过黄色画笔把 Qwen 补全结果显式接入 `completion_mask`

## v0.8.11: A 路径新增 Qwen 线稿结构通道

v0.8.10 解决了大面积中间脏块,但胡须/嘴/轮廓等细线被削弱。原因是主体区域和细线区域被同一套 support/stepify 规则处理;细线在 Qwen 输出中拓扑更完整,但 Qwen RGB 又不能作为最终颜色来源。

调整:

- `VR_LayerMattingRefine` 新增 `line_mask` 输出
- `line_mask` 从 Qwen edge 拓扑提取,只作为 alpha/结构先验
- 线稿组件必须连接到主体 anchor,孤立组件不会恢复,避免中间残留复活
- `support = sure_fg | detail | line`
- RGB 仍通过 composer 从原图回贴,Qwen RGB 不进入 A 路径最终颜色
- Debug A 路径新增 `line_viz`,共 18 个输出
- A 路径默认 alpha 改为 3 档,为细线边缘保留半透明台阶

预期:

- `line_mask` 应覆盖胡须、嘴、小轮廓线
- 胡须断裂减少
- 中间孤立残留不应明显扩大

## v0.9.0: Matting Model Bridge 接口落地

用户确认当前 OpenCV 规则链已接近上限,继续调 `detail_mask / line_mask / anchor / alpha_stepify` 会变成补丁式修改。主线切换为"模型化 matting",先实现稳定接口,再接真实模型。

新增 `VR_LayerMattingModelBridge`:

- 输入:`original_image`,`qwen_image`,`candidate_mask`,`backend`
- backend 选项:`opencv_fallback`,`external_matte`
- 当前 `external_matte` 尚未接外部模型 tensor,会显式 fallback 到 `opencv_fallback`
- 输出:
  - `matte_rgb`
  - `matte_alpha`
  - `visible_alpha`
  - `unknown_region`
  - `matte_confidence`
  - `detail_mask`
  - `line_mask`

A 路径现在不再直接调用 `VR_LayerMattingRefine`,而是通过 Bridge 调用 fallback。这样 `v0.9.1` 接 BiRefNet/RMBG/ViTMatte 时只需要替换 Bridge 内部或把外部 matte 接入 Bridge,不用重构 production/debug workflow。

Debug A 路径新增:

- `visible_alpha_viz`
- `unknown_region_viz`
- `matte_confidence_viz`

Debug A 路径共 20 个输出。

## v0.9.1 规划:接入真实 Matting 模型

目标:让 A 路径最终 alpha 由通用 matting 模型生成,OpenCV fallback 只保留为诊断/兜底。

优先接入顺序:

1. **BiRefNet / RMBG**:优先尝试,因为它们通常只需要图像输入,ComfyUI 生态较成熟,适合先验证白边/细线改善。
2. **ViTMatte**:如果需要 trimap 约束,再接入。它更适合"已有粗 mask + unknown 区域"的精修场景。
3. **MODNet**:作为轻量备选,但对非人像/设计组件的泛化不一定最好。

v0.9.1 的桥接方式:

- 在工作流中外接 matting 节点,输出 `external_matte_alpha`
- Bridge 增加可选 `external_matte_alpha` 输入
- backend=`external_matte` 时:
  - `matte_alpha = external_matte_alpha ∩ candidate_mask`
  - `visible_alpha = matte_alpha`
  - `unknown_region = candidate_mask - matte_alpha`
  - `matte_confidence` 初版可用 matte_alpha 或模型 confidence;若模型无 confidence,先用 alpha 近似
- source composer 继续用原图 RGB,不使用 matting 模型的 RGB

验收标准:

- 胡须/细线不依赖 Qwen edge 规则也能保留
- 白色背景 halo 明显减少
- 中间脏块不随 matting 放大
- `unknown_region` 能指出 matting 不确定区域
- A 路径不再依赖手写 edge/anchor 作为主 alpha 来源

v0.9.1 之后再做 `VR_ForegroundDecontaminate`,专门解决 alpha 正确但 RGB 边缘混白的问题。

## v0.9.1: Target Trimap + External Matte 接入点

先落地真实模型接入所需的接口,不再让外部 matting 节点侵入后处理链路。

新增 `VR_TargetTrimapBuilder`:

- 输入:`qwen_image`,`candidate_mask`
- 输出:`sure_foreground`,`sure_background`,`unknown_region`,`trimap`
- `sure_foreground`:Qwen native alpha 的保守内核
- `sure_background`:候选区域外扩之后的外部背景
- `unknown_region`:候选边界/不确定带,交给 matting 模型决定
- `trimap`:0/0.5/1 三值可视化图,供 ViTMatte 类模型使用

扩展 `VR_LayerMattingModelBridge`:

- 可选输入:`trimap`,`external_matte_alpha`,`external_confidence`
- backend=`external_matte` 且提供 alpha 时:
  - `matte_alpha = external_matte_alpha ∩ candidate_mask`
  - `visible_alpha = matte_alpha`
  - `matte_confidence = external_confidence ∩ candidate_mask` 或 alpha 近似
  - `unknown_region = candidate_mask - matte_confidence`
- 未提供外部 alpha 时仍 fallback 到 `opencv_fallback`

Debug A 路径新增:

- `sure_foreground_viz`
- `sure_background_viz`
- `trimap_unknown_viz`
- `trimap_viz`

Debug A 路径共 24 个输出。下一步真实模型只需要在 workflow 中插入外部 matting 节点,把其 alpha 接入 Bridge 的 `external_matte_alpha`,并把 backend 切到 `external_matte`。

## v0.9.2: Pipeline 暴露外部 matting 输入

v0.9.1 已经让 Bridge 支持 `external_matte_alpha`,但 `VR_PipelineLight` 还没有把输入暴露到 workflow。v0.9.2 补齐接线面。

调整:

- `VR_PipelineLight` / `VR_PipelineLightDebug` required widget 新增 `matting_backend`
- optional inputs 新增:
  - `external_matte_alpha`
  - `external_confidence`
- production/debug workflow 生成脚本新增对应输入槽
- 默认 widgets:`[0, 3, 1500, "mask_socket", "opencv_fallback"]`

接真实模型时:

1. 在 `VR_PipelineLight` 前插入 matting 模型节点
2. 把模型输出 alpha 接到 `external_matte_alpha`
3. 如有 confidence,接到 `external_confidence`
4. 把 `matting_backend` 从 `opencv_fallback` 改成 `external_matte`

后续 source composer / JoinRGBA 不变,确保模型只决定 alpha,不接管 RGB。

## v0.9.3: 插件内接入 HF Matting 模型节点

新增 `VR_HFMattingAlpha`,完成第一版真实模型接入:

- 默认模型:`briaai/RMBG-2.0`
- 当前默认加载目录:`/root/ComfyUI/models/RMBG-2.0`
- 推荐下载源:`https://modelscope.cn/models/briaai/RMBG-2.0`
- 输入:`image`,`candidate_mask`
- 输出:`matte_alpha`,`confidence`
- 模型输出会与 `candidate_mask` 相交,防止全图 matting 误选非目标组件
- 当前 confidence 暂用 alpha 近似
- 依赖 `transformers`;若 ComfyUI 环境未安装或模型权重不可用,节点会明确报错

workflow 更新:

- A 路径新增 `VR_HFMattingAlpha`
- 输入为 scaled original image + SAM candidate mask
- 输出接入 `VR_PipelineLight.external_matte_alpha / external_confidence`
- `VR_PipelineLight` 默认 backend 切到 `external_matte`

这意味着 A 路径现在已经从 OpenCV fallback 切到真实 matting 模型入口。后续效果主要取决于所选模型与权重可用性;若 RMBG 对目标控制不足,下一步替换为 trimap-guided ViTMatte 节点,Bridge 后链路不需要改。

## v0.9.4: A 路径 alpha 边缘增强

新增 `VR_AlphaEdgeRefine`,在 `VR_PipelineLight` 内部执行:

- 基于 matte alpha 生成 silhouette 边界带
- 在边界带内结合原图 Canny 边缘生成 `alpha_edge_roi`
- 对边界带内 alpha 做轻量 contrast/snap,减少 RMBG soft edge 带来的糊边
- RGB 锐化 ROI 从单纯 RGB Canny 改为 `RGB Canny ∪ alpha_edge_roi`

该改动不改变 workflow JSON 的节点结构;production/debug 仍通过 `VR_PipelineLight` / `VR_PipelineLightDebug` 使用增强后的内部链路。

## 待观察 / 未解决

- Light path `palette_k` 默认 8 是否合适,需视目标素材调整(画笔类 6 通常够,摄影类需要 12+)
- `GROW_MASK_PX=18` 给 V2 inpaint 的不确定带是否足够
- Strong path 的 KMeans 仍是 `auto_k=True`(B 路径背景占比高,直方图不被透明主导,暂时没问题)
