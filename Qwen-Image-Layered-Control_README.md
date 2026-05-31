---
frameworks: PyTorch
license: Apache License 2.0
tags: []
tasks:
  - text-to-image-synthesis
base_model:
  - Qwen/Qwen-Image-Layered
base_model_relation: finetune
---
# Qwen-Image-Layered 

## 模型介绍

本模型基于模型 [Qwen/Qwen-Image-Layered](https://modelscope.cn/models/Qwen/Qwen-Image-Layered) 在数据集 [artplus/PrismLayersPro](https://modelscope.cn/datasets/artplus/PrismLayersPro) 上进行了训练，可以通过文本控制拆分的图层内容。


更多关于训练策略和实现细节，欢迎查看我们的[技术博客](https://modelscope.cn/learn/4938)。

## 使用技巧

* 模型结构从多图输出改为了单图输出，仅输出与文本描述相关的图层
* 模型只用英文文本训练过，但仍从基础模型继承了中文理解能力
* 模型训练的原生分辨率是1024x1024，支持以其他分辨率进行推理
* 模型难以拆分“互相遮挡”的多个实体，例如样例中的卡通骷髅头和帽子
* 模型擅长拆分海报图层，不擅长拆分摄影图像，尤其是存在光影的照片
* 模型支持负向提示词，可以通过负向提示词描述不希望出现在结果的内容

## 效果展示

**部分图片为纯白色文本，魔搭社区用户请点击页面右上角的“☀︎”切换到暗色模式**

### 样例1

<div style="display: flex; justify-content: space-between;">

<div style="width: 30%;">

|输入图|
|-|
|![](./assets/image_1_input.png)|

</div>

<div style="width: 66%;">

|提示词|输出图|提示词|输出图|
|-|-|-|-|
|A solid, uniform color with no distinguishable features or objects|![](./assets/image_1_0_0.png)|Text 'TRICK'|![](./assets/image_1_4_0.png)|
|Cloud|![](./assets/image_1_1_0.png)|Text 'TRICK OR TREAT'|![](./assets/image_1_3_0.png)|
|A cartoon skeleton character wearing a purple hat and holding a gift box|![](./assets/image_1_2_0.png)|Text 'TRICK OR'|![](./assets/image_1_7_0.png)|
|A purple hat and a head|![](./assets/image_1_5_0.png)|A gift box|![](./assets/image_1_6_0.png)|

</div>

</div>

### 样例2

<div style="display: flex; justify-content: space-between;">

<div style="width: 30%;">

|输入图|
|-|
|![](./assets/image_2_input.png)|

</div>

<div style="width: 66%;">

|提示词|输出图|提示词|输出图|
|-|-|-|-|
|蓝天，白云，一片花园，花园里有五颜六色的花|![](./assets/image_2_0_0.png)|五彩的精致花环|![](./assets/image_2_2_0.png)|
|少女、花环、小猫|![](./assets/image_2_1_0.png)|少女、小猫|![](./assets/image_2_3_0.png)|

</div>

</div>

### 样例3

<div style="display: flex; justify-content: space-between;">

<div style="width: 30%;">

|输入图|
|-|
|![](./assets/image_3_input.png)|

</div>

<div style="width: 66%;">

|提示词|输出图|提示词|输出图|
|-|-|-|-|
|一片湛蓝的天空和波涛汹涌的大海|![](./assets/image_3_0_0.png)|文字“向往的生活”|![](./assets/image_3_2_0.png)|
|一只海鸥|![](./assets/image_3_1_0.png)|文字“生活”|![](./assets/image_3_3_0.png)|

</div>

</div>

## 推理代码

安装 DiffSynth-Studio：

```
git clone https://github.com/modelscope/DiffSynth-Studio.git  
cd DiffSynth-Studio
pip install -e .
```

模型推理：

```python
from diffsynth.pipelines.qwen_image import QwenImagePipeline, ModelConfig
from PIL import Image
import torch, requests

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    processor_config=ModelConfig(model_id="Qwen/Qwen-Image-Edit", origin_file_pattern="processor/"),
)
prompt = "A cartoon skeleton character wearing a purple hat and holding a gift box"
input_image = requests.get("https://modelscope.oss-cn-beijing.aliyuncs.com/resource/images/trick_or_treat.png", stream=True).raw
input_image = Image.open(input_image).convert("RGBA").resize((1024, 1024))
input_image.save("image_input.png")
images = pipe(
    prompt,
    seed=0,
    num_inference_steps=30, cfg_scale=4,
    height=1024, width=1024,
    layer_input_image=input_image,
    layer_num=0,
)
images[0].save("image.png")
```
