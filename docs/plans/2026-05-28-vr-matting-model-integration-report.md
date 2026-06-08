# VR Matting Model Integration Report

## 目标

把 A 路径从规则式 alpha 生成切到真实 matting 模型:模型只负责 `alpha`,最终 RGB 仍来自原图回贴,Qwen 只做图层先验。

## 已完成

- 新增 `VR_HFMattingAlpha`
  - 默认模型目录:`/root/ComfyUI/models/RMBG-2.0`
  - 推荐从 ModelScope 下载:`https://modelscope.cn/models/briaai/RMBG-2.0`
  - 输入:`image`,`candidate_mask`
  - 输出:`matte_alpha`,`confidence`
  - 输出 alpha 会与候选 mask 相交,避免选中非目标区域
- `VR_PipelineLight` 已切到 `matting_backend="external_matte"`
- production workflow 已接通:
  - `scaled original image + SAM mask -> VR_HFMattingAlpha`
  - `VR_HFMattingAlpha -> VR_PipelineLight.external_matte_alpha / external_confidence`
- debug workflow 同步接通,保留 24 个诊断输出
- 插件版本更新为 `v0.9.3`

## 当前运行要求

ComfyUI Python 环境需要安装 `transformers`,并能从本地目录加载 RMBG-2.0 权重。当前默认目录为:

```text
/root/ComfyUI/models/RMBG-2.0
```

无网络环境下,从 ModelScope 下载 `https://modelscope.cn/models/briaai/RMBG-2.0` 后放入该目录即可。若放到别处,把 `VR_HFMattingAlpha.model_id` 改成对应本地目录。

## 验证

- `python3 -m compileall custom_nodes/comfyui_vector_ready scripts` 通过
- `qwen_layered_v8_ab_vector_ready.json`:53 nodes,72 links
- `qwen_layered_v8_debug.json`:87 nodes,105 links
- production/debug 均包含 `VR_HFMattingAlpha`
- A 路径 backend 均为 `external_matte`

## 下一步观察

重点看 `VR_HFMattingAlpha matte_alpha` 是否保留胡须/细线,以及最终输出是否减少白边。若 RMBG 目标控制不足,下一步替换为 trimap-guided ViTMatte,后续 Bridge/Composer 链路无需改。
