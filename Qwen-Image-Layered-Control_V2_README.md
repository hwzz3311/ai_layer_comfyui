---
base_model:
- Qwen/Qwen-Image-Layered
base_model_relation: adapter
frameworks:
- ""
license: Apache License 2.0
tags: []
tasks: []
---
# Qwen-Image-Layered 画笔控制的图层拆分模型

## 模型介绍

本模型是 [DiffSynth-Studio/Qwen-Image-Layered-Control](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control) 的升级版本，我们在数据集 [artplus/PrismLayersPro](https://modelscope.cn/datasets/artplus/PrismLayersPro) 上进行了进一步训练，为V1模型增加了**画笔控制**能力，现在可以通过文本和画笔控制想要拆分的图层内容。


## 使用技巧

* 本模型保留了模型 [DiffSynth-Studio/Qwen-Image-Layered-Control](https://modelscope.cn/models/DiffSynth-Studio/Qwen-Image-Layered-Control) 的文本控制能力，可以用 `prompt` 描述想要拆分的内容，用 `negative_prompt` 描述不想拆分的内容
* 红色画笔表示目标图层的区域，绿色画笔表示想要删除的图层的区域，两者叠加后是黄色。通过合理组合两种画笔，拆分出被遮挡的图层
* 模型擅长拆分海报图层，不擅长拆分摄影图像，尤其是存在光影的照片
* 推荐推理步数为 10 步，如果需要分割的图层被遮挡情况，请尝试提高步数
* 启用画笔控制时，可以将 `cfg_scale` 设置为 `1`，大幅提升推理速度

## 这个模型可以用来做什么？

### 样例1

这个模型可以快速把画面中的图层拆分出来，只需要用红色画笔涂几笔。

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/1_in.png)|![](./assets/1_mask.png)|![](./assets/1_out.png)|

我们设计了另一个画笔工具，绿色画笔可以用于标注不希望输出的部分。

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/2_in.png)|![](./assets/2_mask.png)|![](./assets/2_out.png)|

### 样例2

画笔涂抹的区域越接近实际的轮廓，图层拆分越准确。

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/3_in.png)|![](./assets/3_mask.png)|![](./assets/3_out.png)|

两种画笔颜色叠加，红 + 绿 = 黄，可以移除遮挡物，得到中间被遮挡的图层。

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/4_in.png)|![](./assets/4_mask.png)|![](./assets/4_out.png)|

### 样例3

这个模型不仅可以“把图层拆分出来”

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/5_in.png)|![](./assets/5_mask.png)|![](./assets/5_out.png)|

也可以“把图层拆分出来”。

|输入图|画笔|输出图|
|-|-|-|
|![](./assets/6_in.png)|![](./assets/6_mask.png)|![](./assets/6_out.png)|

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
from modelscope import dataset_snapshot_download
from PIL import Image
import torch

pipe = QwenImagePipeline.from_pretrained(
    torch_dtype=torch.bfloat16,
    device="cuda",
    model_configs=[
        ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control", origin_file_pattern="transformer/diffusion_pytorch_model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="text_encoder/model*.safetensors"),
        ModelConfig(model_id="Qwen/Qwen-Image-Layered", origin_file_pattern="vae/diffusion_pytorch_model.safetensors"),
    ],
    tokenizer_config=ModelConfig(model_id="Qwen/Qwen-Image", origin_file_pattern="tokenizer/"),
)
pipe.load_lora(pipe.dit, ModelConfig(model_id="DiffSynth-Studio/Qwen-Image-Layered-Control-V2", origin_file_pattern="model.safetensors"))

dataset_snapshot_download(
    dataset_id="DiffSynth-Studio/example_image_dataset",
    local_dir="./data/example_image_dataset",
    allow_file_pattern="layer_v2/*.png"
)

prompt = "Text 'APRIL'"
input_image = Image.open("data/example_image_dataset/layer_v2/image_1.png").convert("RGBA").resize((1024, 1024))
image = pipe(
    prompt, seed=0,
    height=1024, width=1024,
    layer_input_image=input_image, layer_num=0,
    num_inference_steps=10, cfg_scale=4,
)
image[0].save("image_prompt.png")

mask_image = Image.open("data/example_image_dataset/layer_v2/mask_2.png").convert("RGBA").resize((1024, 1024))
input_image = Image.open("data/example_image_dataset/layer_v2/image_2.png").convert("RGBA").resize((1024, 1024))
image = pipe(
    prompt, seed=0,
    height=1024, width=1024,
    layer_input_image=input_image, layer_num=0,
    context_image=mask_image,
    num_inference_steps=10, cfg_scale=1.0,
)
image[0].save("image_mask.png")
```
