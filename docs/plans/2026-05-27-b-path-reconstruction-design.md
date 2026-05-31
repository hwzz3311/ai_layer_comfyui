# B 路径(遮挡重建)整合方案设计文档

> 起草时间:2026-05-27
> 状态:设计已对齐,待分阶段落地
> 前置上下文:见 `ai_layer_reconstruction_state.md`
> 适用版本:V8 及后续

---

## 一、设计目标与核心约束

### 设计目标

为 AI Layer Reconstruction 工作流的 **B 路径(背景层抽取,含遮挡区域重建)** 设计一套**高抽卡成功率 + 强确定性后处理**的整合方案,使其能够稳定输出可直接喂给矢量化系统的 RGBA 图层。

### 核心约束(已与用户对齐)

1. **agent 输入极简**
   - agent 只可靠传入 `mode = "foreground" | "background"`
   - **不传** 目标位置坐标(agent 不擅长)
   - **不传** 遮挡判断(部件小时 agent 也判断不准)
   - 重活全部由工作流承担,agent 只做最轻量的判断

2. **场景范围**
   - 海报 / 电商 banner / UI 卡片 / 包装设计 / 潮玩周边
   - 规则几何为主,无真实摄影,无自然风景

3. **遮挡模式分布**
   - A 类(单层大面积遮挡)+ B 类(小面积散点遮挡)= 主要场景
   - C 类(多层堆叠)= 偶发场景,v1 不深做

4. **Qwen Layered 模型的正确定位**
   - Qwen V2 **已经具备分层 + 补全能力**,问题不是能力缺失,而是**抽卡稳定性**
   - 不绕开 Qwen,而是通过画笔条件提高抽卡成功率,再用确定性后处理修补 diffusion 的固有缺陷
   - 不把 RGB 完全推给 LaMa(理解偏差,已纠正)

5. **拒绝抽卡**
   - 任何依赖运气的方案不接受
   - 多 seed 投票被否决

---

## 二、B 路径整体架构

```
原图 (来自 agent,mode=background)
  ↓
[Stage 1] 画笔生成(B 路径命门)
  ├─ U²-Net 前景显著性检测
  ├─ SAM3 prompted 精修
  ├─ 形态学 close/open 校正
  └─ 红/绿/黄三色画笔合成
  ↓
[Stage 2] Qwen Layered V2 抽卡
  └─ 双 ReferenceLatent(原图 + 画笔图)→ RGBA 含遮挡补全
  ↓
[Stage 3] 确定性后处理 — 可信源混合
  ├─ 黄色画笔区外:用 (原图 RGB × Qwen alpha) — 可信
  ├─ 黄色画笔区内:用 (Qwen RGB × Qwen alpha) — 必须信任 diffusion
  └─ 黄区边界羽化融合,消除接缝
  ↓
[Stage 4] VectorReady Pipeline(A/B 共享)
  └─ 矢量化友好性优化:见第四章
  ↓
RGBA 输出 → 矢量化系统
```

---

## 三、Stage 1 画笔生成(B 路径命门)

### 关键简化:黄色画笔 ≈ 绿色画笔覆盖区

规则几何场景下,背景的"应有形状"= 整张画布。因此:

- **背景缺失区 ≈ 前景覆盖区**
- 黄色画笔位置 ≈ 绿色画笔位置,语义不同:
  - 绿 = "排除这里,不算目标"
  - 黄 = "这里曾是背景,被压了,帮我补回来"
- V2 画笔通过 RGB 通道实现,黄 = 红 + 绿 叠加,语义自然吻合

**这个简化把"三色画笔从哪来"归约为"前景 mask 怎么准确得到"** 单一问题。

### 流水线

```
原图
  ↓
[1.1] U²-Net 前景显著性检测
    → foreground_probability_map (软 mask)
  ↓
[1.2] SAM3 prompted 精修
    ├─ 高显著性峰值点 → SAM3 → 前景精确 segments
    └─ (可选) 低显著性区采点 → SAM3 → 背景交叉验证
  ↓
[1.3] 形态学校正
    ├─ Close 操作:补孤立小洞
    ├─ Open 操作:去毛刺
    └─ 平滑边缘
  ↓
[1.4] 画笔合成
    ├─ 红 R = (前景 mask) 取反 ∩ 画布
    ├─ 绿 G = 前景 mask
    └─ 黄 Y = 前景 mask  (通过 R+G 同位叠加自然产生)
  ↓
三色画笔图
```

### 模型选型

| 模块 | 选型 | 理由 |
|---|---|---|
| Saliency | **U²-Net** | 海报/banner/包装场景训练充分,ComfyUI 生态成熟(`comfyui-rembg` 等) |
| Prompted segmentation | **SAM3**(V7 已用) | 已验证,prompt-driven 稳定 |
| 形态学操作 | **ComfyUI-Impact-Pack MaskFix 系列** | 现成节点 |

### v1 不做的部分

- ❌ Qwen 双 pass 自引导兜底 — 工程复杂度过高,先单 saliency + 质量门控
- ❌ 几何先验(矩形/对称轴检测) — v1 只做形态学,几何先验 v2 再加
- ❌ Saliency 置信度门控 — 失败率超 20% 再补

---

## 四、Stage 4 VectorReady Pipeline(A/B 共享模块)

### 设计动机

矢量化模型对输入有强偏好:**硬边缘、纯色块、有限调色板、干净 alpha**。Diffusion 输出天然违背这些 — 色块内有 ±3-5 RGB 噪声、alpha 边缘有 halo、大面积渐变有 banding。

VectorReady 是 A/B 路径**共享的后处理**模块,把 Qwen 输出整理成矢量化友好格式。

### 流水线(顺序敏感)

```
RGBA 输入
  ↓
[4.1] 边缘检测 (全局先验)
    └─ Canny + Sobel 融合 → edge_map
       (后续所有 step 都引用 edge_map,避免重复计算和不一致)
  ↓
[4.2] 保边去噪
    └─ Bilateral filter,σ_color 较大、σ_space 较小
       目的:去 diffusion grain,保真边缘
  ↓
[4.3] 颜色量化与区域聚合 (LAB 色彩空间)
    ├─ K-means 聚类(K 用直方图峰值自适应)
    ├─ ΔE 阈值合并相邻相似色块
    └─ 关键:跨 edge_map 边缘的像素不合并
  ↓
[4.4] 渐变区域重建 — v1 跳过,v2 再做
  ↓
[4.5] 边缘锐化 (仅 edge_map 标记区)
    └─ ROI-limited unsharp mask
  ↓
[4.6] Alpha 清理
    ├─ Bilateral 风格的 alpha 平滑
    ├─ 形态学 close + open
    └─ 边缘阶梯化:压成 2-3 个台阶(纯透 / 半透 / 纯不透)
  ↓
干净 RGBA → 矢量化输入
```

### 强度档位

模块支持档位配置,A/B 路径接入时按需选择:

| 档位 | 适用 | 重点 |
|---|---|---|
| `light` | A 路径 (原图×alpha,色彩已稳) | edge + alpha 优化 |
| `strong` | B 路径 (含 Qwen RGB 补全区) | 颜色一致性 + 去噪 + alpha |

### 关键决策记录

- **K-means K**:直方图峰值自适应,不固定
- **色彩空间**:LAB(更符合人眼,矢量化效果质变)
- **渐变重建**:v1 跳过(纯色场景占多)
- **边缘检测放最前**:作为全局先验,避免每个 step 自己猜边缘位置

---

## 五、洋葱剥皮(多层场景) — 评估结论

### 思路

`A - B = C`,从 C 继续抽下一层,递归处理多层堆叠。

### 评估结果:**v1 不做,作为 P2 扩展**

### 关键风险

1. **像素减法不可行**,必须语义化:`next = current × (1 - extracted.alpha)`
2. **alpha 边缘渐变** → 减完后 C 边缘带"鬼影残留"
3. **被扣区域是空洞** → 下一轮 Qwen 看到空洞会被严重干扰
4. **误差累积非线性** → 3 层后基本失控

### 让洋葱剥皮可稳定的前置条件(全部成立才能启用)

1. 每层抽取必须走 A 路径完整优化(matting + 原图×alpha,绝不用 Qwen RGB)
2. 扣除必须语义化 + 对空洞做轻量 inpaint(LaMa / PatchMatch 填成"自然延续")
3. 深度硬限制 ≤ 2 层,超出主动放弃
4. 每层质量门控:alpha halo / 颜色直方图偏移 / 空洞占比,不达标中止

### 决定

- A+B 模式覆盖大多数场景 → 先把单层 B 路径做扎实
- 多层场景作为独立扩展路径,**不进主线**
- 排到 P2,v1 不实现

---

## 六、分阶段落地节奏

每个版本都是**可用工作流**,逐步加能力,避免一次性大改。

```
v7.0 (现状,A/B 未真正分流)
   ↓
v7.1: 修 ImpactSwitch — A/B 真正分流
   工作量:1-2 小时,纯连线改造
   ↓
v8.0: 画笔生成 (Stage 1) 接入 U²-Net + SAM3 prompted
   工作量:半天,全用现成节点
   ↓
v8.1: A 路径接入 Matting (ViTMatte / RMBG)
   工作量:半天,现成节点
   ↓
v9.0: VectorReady 共享模块 (Stage 4) — 最小版
   优先级:颜色量化 + alpha 阶梯化
   工作量:1-2 天,需要自定义 Python 节点
   ↓
v9.1: VectorReady 完整版
   补:LAB 处理、边缘锐化、Bilateral 去噪
   工作量:1-2 天
   ↓
v10: Qwen 双 pass 兜底(仅当 v9 失败率 > 20% 时启用)
   ↓
P2: 洋葱剥皮多层支持(独立扩展路径)
```

### 自定义节点清单(v9 起需要)

| 节点 | 用途 | 实现方式 |
|---|---|---|
| `LAB Color Convert` | RGB ↔ LAB 转换 | cv2.cvtColor |
| `KMeans Color Quantize` | 颜色聚类量化 | sklearn / cv2.kmeans |
| `Edge-aware Color Merge` | 跨边缘不合并的色块聚合 | OpenCV + edge_map |
| `Alpha Stepify` | alpha 阶梯化(2-3 台阶) | numpy 阈值 |
| `ROI Unsharp Mask` | 边缘 ROI 限定锐化 | cv2 + mask |

---

## 七、设计原则总结(供后续接续参考)

1. **拒绝抽卡** — 任何依赖运气的方案不接受;通过画笔条件 + 确定性后处理双重消解不确定性
2. **agent 只做最轻判断** — 重活全部在工作流,不把复杂判断推给 agent
3. **不绕开 Qwen,而是驯服它** — Qwen 已具备分层 + 补全能力,问题是稳定性,不是能力
4. **可信源混合** — 哪里能用原图 RGB 就用原图,只在无可奈何处用 diffusion RGB
5. **共享后处理** — A/B 路径共用 VectorReady,避免重复造轮子
6. **分阶段落地** — 每版本都是可用工作流,不一次性大改
7. **优先级排序** — A+B 主流程优先,C 多层场景延后,兜底机制最后做

---

## 八、待跟进事项

- [ ] v7.1 修复:ImpactSwitch 接 foreground_mode → A/B 两个 KSampler 的输入端
- [ ] v8.0 设计:U²-Net 节点选型确认(comfyui-rembg vs comfyui-u2net),Saliency 输出软 mask 如何转 SAM3 prompt point
- [ ] v8.1 设计:Matting 模型对比(ViTMatte vs RMBG vs MODNet)
- [ ] v9.0 设计:自定义节点开发清单确认,Python 实现方式(独立插件 vs 内联脚本节点)
- [ ] Saliency 置信度门控阈值(失败率监控后再定)
- [ ] VectorReady 强度档位的具体参数(σ、ΔE、K、阶梯化阈值)

---

## 九、附:关键术语对照

| 术语 | 含义 |
|---|---|
| A 路径 | 前景提取,目标 layer 无遮挡或基本无遮挡 |
| B 路径 | 背景重建,目标 layer 被遮挡需要补全 |
| C 类场景 | 多层堆叠的遮挡(非路径,是场景类型) |
| 红画笔 | 标记目标 layer 区域 |
| 绿画笔 | 标记明确排除区域 |
| 黄画笔 | 标记遮挡触发区,R+G 叠加自然产生 |
| ownership | layer 间的 z-order 关系("谁压在谁上面") |
| VectorReady | A/B 共享的矢量化友好性后处理流水线 |
| 抽卡 | diffusion 输出不稳定、依赖随机性的现象 |
