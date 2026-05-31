# AI Layer Reconstruction 项目状态文档

> 文档生成时间:2026-05-26
> 阶段:V7 工作流修复 + B 路径优化方案探讨
> 性质:项目知识沉淀,用于后续接续推进

---

## 一、项目本质与终极目标

### 一句话定义

**这不是抠图,这是 AI PSD 逆向工程。**

### 完整工作流目标

```
Design Image (最终设计稿)
   ↓
Layer Parsing (图层解析)
   ↓
Layer Ownership (归属判定)
   ↓
Layer Reconstruction (图层重建,含被遮挡部分恢复)
   ↓
RGBA Layer (单层输出)
   ↓
Vectorization (外部矢量化系统)
```

### 与传统方案的本质区别

| 不是 | 是 |
|---|---|
| SAM 抠图 | Layer Reconstruction |
| 普通 matting | Ownership + Alpha + 遮挡恢复 |
| diffusion inpaint | Design-aware completion |
| Segmentation | "这个 layer 原本长什么样" |

### 终极目标

从最终设计稿**逆向恢复出原始 PSD 分层结构**,包括:

- 每一层的 RGBA
- 层间 z-order(谁压在谁上面)
- 被遮挡区域的合理重建
- 连续结构(对称、描边、渐变)的恢复
- 隐藏的设计意图(ownership)

**行业现状:几乎没人在做这件事。** 这是项目的核心价值,也是核心风险。

---

## 二、调用方与系统设计哲学

### 调用方:VL 多模态 Agent

工作流不是给人用的,是给 **VL Agent** 用的。每次调用,agent:

- **知道**:本次要抽取的是哪一个图层(前景还是背景)
- **知道**:这一层有没有被遮挡
- **能看图**:可以提供基础的视觉判断辅助
- **不可靠**:agent 的判断只能作为辅助信号,可信度不高,**重点在工作流本身**

### 设计原则

1. **工作流不猜测,agent 传参** — A/B 路径分流由 agent 决定,不靠工作流自动判断
2. **agent 的提示只是辅助,核心能力必须在工作流里** — 不能把重活推给 agent
3. **单次调用只处理一个图层** — 不追求一次性输出多层
4. **拒绝抽卡** — 任何依赖运气的方案都不接受

---

## 三、当前技术栈与核心认知

### 主模型

- **Qwen-Image-Layered** (V1 base)
- **Qwen-Image-Layered-Control-V2** (V2 LoRA,提供画笔能力)

### 辅助模型

- **SAM3** (自动分割,生成初始 mask)

### 关键技术结论(已验证)

#### 1. Qwen V2 的 context_image 本质

不是 ControlNet,不是 Inpaint Mask,而是 **第二个 ReferenceLatent**:

```python
conditioning_set_values(
  cond,
  {'reference_latents': [latent['samples']]},
  append=True
)
```

最终生成由 `原图 latent + 画笔 latent` 共同决定。

#### 2. 红绿黄画笔机制

| 颜色 | 含义 |
|---|---|
| 红色 | 目标 layer 区域 |
| 绿色 | 明确排除区域 |
| 黄色 (红+绿叠加) | 遮挡恢复触发器 |

#### 3. RGB 不应由 diffusion 输出

diffusion 会带来:重绘、色偏、模糊、边缘漂移、纹理 hallucination。

**正确分工:**

```
Qwen 负责:ownership + alpha + layer prior
原图 RGB:最终颜色来源
最终 RGBA = 原图 RGB × Predicted Alpha
```

#### 4. Qwen V2 官方使用建议(来自 README)

- 推荐推理步数 **10 步**(画笔模式)
- 遮挡情况可适当提高步数,但收益递减
- 启用画笔控制时 `cfg_scale=1` 即可
- **官方明确警告**:
  - 模型擅长拆分海报图层,**不擅长摄影图像(尤其有光影)**
  - 模型难以拆分**互相遮挡的多个实体**

> ⚠️ 这两条警告决定了:遮挡区域的重建是模型的本质弱点,**不是调参能解决的**

---

## 四、版本演进总结

| 版本 | 关键突破 | 状态 |
|---|---|---|
| v3.5 | 首次接入 context_image latent,实现双 ReferenceLatent | ✅ |
| v4 | 引入绿色负向、黄色 overlap、negative ring | ⚠️ MaskSubtract 依赖问题,已放弃 |
| v5 | 转向 Alpha-First,不再依赖 diffusion 重绘 | ✅ 思路确立 |
| v6 | 明确 reconstruction ≠ extraction,引入 ownership-first | ✅ 思路确立 |
| v7 | A/B 双路径(前景提取 + 背景重建) | ⚠️ **路径未真正分流(见下)** |

---

## 五、V7 当前实际状态(扫描结果)

### 工作流文件

`qwen_layered_v7_ab_dual_path.json`(45 节点,52 链接)

### 整体架构

```
Input Image (1)
   ↓
ImageScaleToMaxDimension ≤1024 (5)
   ↓
SAM3 分割 (11) → MaskFix+ (20)
   ↓
画笔生成
   ├─ 红色正向 (201/202/203)
   ├─ 绿色负向 (204/205/207/208)
   └─ 黄色 overlap(隐含在红+绿合成中)
   ↓
双 ReferenceLatent
   ├─ #1 原图 (52)
   └─ #2 画笔 (53)
   ↓
A 路径 KSampler (60) [steps=7,  cfg=0.8]  → 61 → 62 VAEDecode → 63 SaveImage
B 路径 KSampler (210)[steps=16, cfg=1.0]  → 211 → 212 VAEDecode → 214 SaveImage
```

### 严重问题(必须修复)

#### 1. ❌ A/B 路径分流未生效

- **节点 215** `PrimitiveNode(BOOLEAN, true)` 是 `foreground_mode` 开关
- **`outputs.links = []` ,完全悬空,没接任何节点**
- 后果:**A 和 B 两条路径同时执行**,共享完全相同的输入(model/positive/negative/latent_image),只是 steps 和 cfg 不同
- 算力浪费一半,且 agent 会同时收到两个输出文件

#### 2. ⚠️ A/B 共享输入是好事还是坏事?

- 共享 conditioning 没问题(prompt、画笔都一样)
- 共享 latent_image 没问题(空 latent)
- **真正的差异只在 sampler 参数** — 这其实是合理设计,只缺一个开关

---

## 六、修复方案(已与用户对齐)

### A/B 切换实现方式

**用 ImpactSwitch 切换 KSampler 的输入端**(不是输出端)。

利用 ComfyUI 按需执行特性:被切断的那条路径,KSampler 不会执行。

```
node 215 (foreground_mode bool)
   ↓ (true→1, false→2)
ImpactSwitch.select
   ├─ A KSampler (60)  [steps=7,  cfg=0.8]  → 独立后处理链
   └─ B KSampler (210) [steps=16, cfg=1.0]  → 独立后处理链
```

### 为什么 A/B 后处理不能合并

**已确认 A/B 后处理流程不同:**

| 路径 | 重点 | 未来后处理 |
|---|---|---|
| A 前景 | 边缘精度 | ViTMatte / RMBG → 原图 × alpha → RGBA |
| B 背景 | 遮挡区域重建质量 | 结构验证 + 几何补全 + RGBA 合成 |

强行合并到单一 SaveImage,会让两套后处理塞进同一根管子,扩展性极差。

### 收益

- 算力省一半
- A/B 后处理链路完全独立,自由扩展
- agent 根据自己传的 `foreground_mode` 知道去读哪个输出

---

## 七、B 路径优化方案(核心矛盾,正在讨论)

### 核心矛盾

B 路径处理被遮挡背景/主体,需要大量重绘。
**diffusion 重绘 ≈ 抽卡 ≈ 不可控。**

### 已排除的方案

#### ❌ 多 seed 投票

- **理由**:本质还是抽卡,只是把抽卡概率提高
- 时间消耗 3-5 倍,用户体验差
- 如果多次抽卡结果都是垃圾,投票也是垃圾投票
- **用户明确否决**

#### ⚠️ Agent 传结构先验(降级为辅助)

- **理由**:agent 判断可信度不高,不能依赖
- 可以作为**辅助信号**(比如帮助识别"这是 UI 卡片"还是"自然形状"),但不能作为重建依据
- **重点必须在工作流本身**

### 当前可能的优化方向(待深入设计)

#### 方向 A:经典 CV 做"骨架先行"

被遮挡的设计稿元素(UI 卡片、几何形状、渐变底)往往是规则结构:

- **检测可见部分的几何属性**:直线、圆弧、对称轴、矩形角点
- **外推规则结构**:基于已知部分的几何规律,预测遮挡区域的形状
- 把骨架图作为额外画笔通道或 ControlNet 条件喂给 Qwen
- diffusion 只负责"在骨架上画细节",而不是凭空想结构

#### 方向 B:专用 inpaint 模型接手遮挡区域

- Qwen 出 **ownership + alpha**(含遮挡区域的预测 alpha)
- 遮挡区域的 RGB **不用 Qwen 输出**
- 改用 **LaMa / MAT** 这类专门为"基于周围像素补全规则结构"训练的 inpaint 模型
- LaMa 在大面积擦除 + 规则结构补全上,稳定性远超 diffusion

#### 方向 C:多算法/多模型集成(用户倾向方向)

这是用户明确指出的方向 —— **大量 CV、算法、模型整合**,而不是依赖单一 diffusion:

可能的整合点:
- 经典 CV(边缘检测、Hough 变换、对称性检测、color quantization)
- 专用 inpaint 模型(LaMa、MAT、ZITS)
- Matting 模型(ViTMatte、RMBG、MODNet)
- Qwen Layered 作为 ownership 判定器
- 几何先验(矩形、圆形、贝塞尔曲线推断)
- 颜色一致性约束(从可见区域采样,约束遮挡区域)

**核心理念**:把"抽卡概率"通过多重约束推到极限 —— 每一个独立模块都把不确定性消解掉一部分,最后留给 diffusion 的只剩最小的"必须创意补全"的部分。

### 工程复杂度警告

- 方向 A:工程复杂度偏高,但收益明确
- 方向 B:架构改动较大,但从根上绕过抽卡
- 方向 C:综合方案,长期最强,短期最重
- **没有"便宜又稳"的方案,这就是项目的本质难度**

---

## 八、长期技术规划

### 必须建立的能力

#### 1. Ownership Graph(层归属图)

```
Layer A 覆盖 Layer B
Layer B 覆盖 Layer C
```

即 z-order graph。**有了它,才能谈遮挡恢复;没有它,黄色画笔本质上还是在瞎猜。**

#### 2. Structure-aware Reconstruction

恢复:对称性、描边连续、渐变连续、UI 几何规律。
**不是普通 inpaint,而是设计感知重建。**

#### 3. Layer Canonicalization

推断"这个 layer 的完整状态是什么",而不是"它的可见部分是什么"。

#### 4. 引入高质量 Matting

- ViTMatte / MODNet / RMBG
- 用于高质量 alpha 输出
- 让 Qwen 专注于 ownership,matting 专注于边缘

#### 5. RGB 链路彻底脱离 diffusion

```
最终输出 = 原图 RGB × Predicted Alpha
```

而不是:

```
最终输出 = Diffusion RGB(色偏 + 重绘)
```

---

## 九、风险与瓶颈

### 技术风险

#### 1. Diffusion 天然不稳定

尤其在以下场景:
- 大面积重建
- 渐变恢复
- UI 几何精度
- 描边连续性

#### 2. 缺乏真实 World Model

当前模型只有 image prior,没有 **design structure prior**。
模型不"理解"设计稿,只是在像素层面学到了一些统计规律。

#### 3. 没有公开的 Layer Dataset

业内没有公开的 PSD reconstruction dataset。
任何想训练专用模型的尝试都会面临数据问题。

#### 4. Qwen Layered 的本质局限

- 更像 latent decomposition prior
- 不是 full reconstruction engine
- 官方都承认遮挡场景表现差

### 产品风险

#### 1. 用户体验对"抽卡"零容忍

- 不能让用户跑多次取最好的
- 不能让用户看到失败的输出
- 输出必须一次性可用

#### 2. Agent 的可信度上限

- VL agent 提供的判断只能辅助,不能依赖
- 工作流必须在没有 agent 强先验的情况下也能工作

---

## 十、当前最关键的认知

### 真正的方向

**不是:**
- 调 cfg
- 调 steps
- 调 mask 大小
- 多 seed 投票
- 等更强的 diffusion 模型

**而是:**

> 从"抠图"升级到 **AI Layer Reconstruction System**
>
> 通过 **CV + 经典算法 + 专用模型 + Qwen Layered** 的深度整合,
> 把每一个环节的不确定性都消解掉,
> 让 diffusion 只承担它真正擅长的最小工作量。

### 项目位置

> 这是一个没人做过的事情。
> 没有现成方案可抄,没有成熟工具链可用。
> 需要做大量的 CV、算法、模型整合工作。
> 短期看不到便宜的捷径,这是项目的本质难度,也是项目的价值所在。

---

## 十一、下一步行动清单

### 必须立即做

1. **修复 V7 的 A/B 切换**
   - 用 ImpactSwitch 接 foreground_mode → 两个 KSampler 的输入端
   - A/B 后处理链路保持独立
   - 产出 v7.1 工作流 json

### 优先级 P0(修复后立即推进)

2. **A 路径接入 Matting**
   - 接 ViTMatte 或 RMBG
   - 走"原图 × alpha"链路
   - 验证 alpha 质量提升

### 优先级 P1(B 路径核心攻坚)

3. **B 路径整合方案设计**
   - 经典 CV 几何检测模块
   - LaMa / MAT inpaint 接入
   - Qwen 仅出 ownership,RGB 交给 inpaint
   - 多模块串联,系统性消解抽卡

### 优先级 P2(长期能力建设)

4. **Ownership Graph 数据结构与生成机制**
5. **Structure-aware Reconstruction 框架**
6. **Layer Canonicalization 推断逻辑**

---

## 附:工作流节点速查(V7 当前版本)

| 节点 ID | 类型 | 作用 |
|---|---|---|
| 1 | LoadImage | 输入设计稿 |
| 5 | ImageScaleToMaxDimension | 缩放 ≤1024 |
| 10/11 | SAM3 | 自动分割 |
| 20 | MaskFix+ | mask 修复 |
| 201-209 | 画笔生成系列 | 红/绿/黄画笔合成 |
| 30-34 | 模型加载 | UNet/LoRA/CLIP/VAE |
| 40 | CLIPTextEncode | prompt 编码 |
| 50/51 | VAEEncode | 原图+画笔图编码 |
| 52/53 | ReferenceLatent | 双 ref latent 注入 |
| 54 | ConditioningZeroOut | 负向 |
| 55 | EmptyQwenImageLayeredLatentImage | 空 latent |
| **60** | **KSampler [A]** | **前景提取 steps=7 cfg=0.8** |
| **210** | **KSampler [B]** | **背景重建 steps=16 cfg=1.0** |
| 61/211 | LatentCutToBatch | 切片 |
| 62/212 | VAEDecode | 解码 |
| 63/214 | SaveImage | 保存(A/B 独立) |
| **215** | **PrimitiveNode(BOOLEAN)** | **foreground_mode 开关 ❌ 当前悬空** |
