# 08. Qwen-Image-2512 图片类型扩展训练计划

> 状态：规划中，尚未开始训练。  
> 目标：让 Qwen-Image-2512 能更稳定地生成某些特定类型的图片，例如工业产品、机械结构、特定风格、特定主体或特定场景。  
> 推荐技术路线：**LoRA 微调 `QwenImageTransformer2DModel`**，而不是直接 full fine-tuning。

---

## 1. 目标和边界

### 1.1 这个计划解决什么问题

本计划的目标是：

```text
准备目标类型图片
→ 编写高质量 caption
→ 基于官方 Qwen-Image DreamBooth LoRA 脚本训练
→ 得到可加载的 LoRA adapter
→ 与 base model 对比验证
→ 满意后再考虑 merge 和量化
```

它适合让模型生成：

- 某种风格；
- 某类产品；
- 某种机械结构；
- 某个 IP 角色；
- 某类建筑或场景；
- 某种工业摄影效果。

### 1.2 这个计划不解决什么问题

它**不是图像识别 / 图片理解训练计划**。

不要用这个计划去做：

- 图片分类；
- 看图问答；
- OCR；
- 视觉检测；
- 图像检索。

如果目标是“看懂图片”，应该训练 Qwen-VL / Qwen3-VL / CLIP / 分类模型，而不是 Qwen-Image 的生成 transformer。

---

## 2. 为什么选择 LoRA

Full fine-tuning 会直接更新 Qwen-Image transformer 的全部参数：

```text
优点：改造能力强
缺点：显存极高、容易遗忘基础能力、训练成本高
```

LoRA 只插入低秩适配层：

```text
优点：
- 显存低很多
- 不容易灾难性遗忘
- 可以单独控制强度
- 可以多个 LoRA 叠加
- 失败成本低
- 便于回滚
```

因此第一阶段使用：

```text
Qwen-Image-2512
+ QwenImageTransformer2DModel LoRA
```

冻结：

```text
tokenizer
text_encoder
VAE
scheduler
```

只训练：

```text
transformer 中的 LoRA 层
```

---

## 3. 数据类型划分

先不要一开始就训练“很多类型”。建议每次只验证一个清晰目标。

### 3.1 风格类

例如：

```text
工业摄影
中国水墨风
赛博朋克插画
真实感人像
产品商业摄影
电影感夜景
```

建议数据量：

```text
500 ~ 5000 张高质量图片
```

caption 应该强调：

- 风格；
- 光线；
- 材质；
- 色调；
- 构图；
- 相机感。

### 3.2 主体 / 物体类

例如：

```text
某个产品
某个 IP 角色
某种机械零件
某种服装款式
```

建议数据量：

```text
20 ~ 500 张
```

主体类可以使用触发词：

```text
a photo of TOK product
in the style of TOK style
```

`TOK` 只是一个占位符，训练时可以换成稳定、不容易与其他概念冲突的触发词。

### 3.3 多类型混合

如果希望模型同时掌握多个类型，有两种方案：

```text
方案 A：一个概念训练一个 LoRA
       稳定、可控、易于诊断

方案 B：一个带 caption 的大数据集训练一个 LoRA
       适合数据量大、caption 质量高的情况
```

第一阶段推荐方案 A。

---

## 4. 数据集质量要求

数据质量通常比数量更重要。

### 4.1 图片要求

每张图片应尽量满足：

- 分辨率足够高；
- 主体清晰；
- 没有严重模糊、遮挡、裁切错误；
- 没有不需要模型学习的水印、边框、噪声；
- 光线和构图有变化；
- 不要全部来自同一背景。

### 4.2 建议目录结构

单概念训练可以先用：

```text
dataset/
├── 0001.jpg
├── 0002.jpg
├── 0003.jpg
├── ...
```

多概念 / 多 caption 数据集建议用：

```text
qwen-image-dataset/
├── images/
│   ├── 0001.jpg
│   ├── 0002.jpg
│   └── ...
└── captions.csv
```

`captions.csv` 示例：

```csv
image,caption
images/0001.jpg,a detailed industrial photo of a silver metal gear on a blue workbench
images/0002.jpg,a realistic photo of a mechanical pump inside a factory
images/0003.jpg,a cinematic photo of a modern glass building at dusk
```

### 4.3 Caption 写法

caption 应描述你希望模型学会的内容：

```text
主体
场景
材质
光线
视角
构图
风格
```

好的 caption：

```text
a realistic industrial photo of a stainless steel pump,
placed on a concrete factory floor,
cool white workshop lighting,
sharp focus,
high detail
```

不好的 caption：

```text
a pump
```

不要把不想要的元素写进 caption，例如：

```text
watermark
blurry
low quality
distorted
```

除非你明确希望模型学习这些瑕疵。

---

## 5. 官方训练脚本

Hugging Face diffusers 最新源码提供：

```text
examples/dreambooth/train_dreambooth_lora_qwen_image.py
examples/dreambooth/README_qwen.md
```

注意：

- 当前项目运行环境中的 diffusers 是 `0.40.0`；
- 官方训练脚本要求更新的 diffusers；
- 建议在 GPU 机器上单独创建训练环境，从源码安装最新版 diffusers；
- 不要直接复用当前 CPU 推理 `.venv` 做训练。

---

## 6. 训练环境准备

在 GPU 训练机器上：

```bash
git clone https://github.com/huggingface/diffusers.git
cd diffusers

uv venv --python 3.12 .venv
source .venv/bin/activate

uv pip install -e .
uv pip install -r examples/dreambooth/requirements_sana.txt
uv pip install bitsandbytes accelerate datasets tensorboard

accelerate config default
```

如果使用 Hugging Face 数据集，需要登录：

```bash
hf auth login
```

本地 Qwen-Image-2512 模型路径可以继续使用：

```bash
MODEL=/path/to/models/Qwen-Image-2512
```

---

## 7. 第一轮实验：单概念 LoRA

先选择一个小而清晰的目标，例如：

```text
机械零件摄影
```

准备 50~200 张图，然后运行：

```bash
cd diffusers/examples/dreambooth

MODEL=/path/to/models/Qwen-Image-2512
INSTANCE=/path/to/dataset
OUTPUT=/path/to/qwen-image-mechanical-lora

accelerate launch train_dreambooth_lora_qwen_image.py \
  --pretrained_model_name_or_path="$MODEL" \
  --instance_data_dir="$INSTANCE" \
  --instance_prompt="a detailed photo of TOK mechanical part" \
  --output_dir="$OUTPUT" \
  --mixed_precision=bf16 \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --learning_rate=2e-4 \
  --lr_scheduler=constant \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --rank=8 \
  --lora_alpha=16 \
  --lora_dropout=0.05 \
  --lora_layers="to_q,to_k,to_v,to_out.0" \
  --cache_latents \
  --offload \
  --gradient_checkpointing \
  --use_8bit_adam \
  --validation_prompt="a detailed photo of TOK mechanical part on a workbench" \
  --validation_epochs=25 \
  --seed=42
```

### 7.1 第一轮参数建议

```text
resolution = 512
steps = 300 ~ 800
rank = 8
lora_alpha = 16
learning_rate = 2e-4
train_batch_size = 1
gradient_accumulation_steps = 4
```

第一轮目标不是得到最终效果，而是验证：

- 数据是否正确；
- caption 是否有效；
- 脚本是否跑通；
- 显存是否够；
- LoRA 是否有方向正确的变化。

---

## 8. 多类型 / 多 caption 训练

当单概念实验跑通后，可以使用 Hugging Face Dataset，让每张图都有自己的 caption。

### 8.1 创建数据集

示例代码：

```python
from datasets import Dataset, Image
from pathlib import Path

image_dir = Path("images")

rows = []
for path in sorted(image_dir.glob("*.jpg")):
    caption = ...  # 读取或人工编写 caption
    rows.append({
        "image": str(path),
        "caption": caption,
    })

ds = Dataset.from_dict({
    "image": [r["image"] for r in rows],
    "caption": [r["caption"] for r in rows],
}).cast_column("image", Image())

ds.push_to_hub(
    "yourname/qwen-image-type-dataset",
    private=True,
)
```

### 8.2 使用 caption 数据集训练

```bash
--dataset_name="yourname/qwen-image-type-dataset" \
--image_column="image" \
--caption_column="caption" \
--instance_prompt="a high quality image"
```

说明：

- `instance_prompt` 仍然需要传；
- 当指定 `caption_column` 后，脚本会使用每张图自己的 caption；
- `instance_prompt` 只作为兜底或元信息。

---

## 9. LoRA 参数选择

### 9.1 rank

| rank | 适合场景 |
|---:|---|
| 4 | 非常轻量的风格 |
| 8 | 小主体、简单风格，推荐起点 |
| 16 | 常用平衡值 |
| 32 | 较强风格或多概念 |
| 64+ | 大规模风格，但显存和过拟合风险增加 |

推荐从：

```text
rank = 8
lora_alpha = 16
```

开始。

### 9.2 learning rate

常用范围：

```text
1e-4 ~ 5e-4
```

推荐先：

```text
2e-4
```

如果变化太慢：

```text
提高到 3e-4 或 5e-4
```

如果过拟合：

```text
降低到 1e-4，减少步数，增加数据多样性
```

### 9.3 target modules

官方脚本默认：

```text
to_q,to_k,to_v,to_out.0
```

这是比较稳的起点。

如果效果不够，可以尝试加入图文联合注意力相关模块：

```text
to_q,to_k,to_v,to_out.0,
add_q_proj,add_k_proj,add_v_proj,to_add_out
```

再进一步可以尝试：

```text
img_mlp.net.0.proj
txt_mlp.net.0.proj
```

注意：

- target modules 越多，显存占用越高；
- 过拟合风险也越高；
- 第一轮不要贪多。

---

## 10. 验证方案

训练时和训练后都要用固定 seed 对比。

### 10.1 固定提示词

准备一组验证 prompt，例如：

```text
a detailed photo of TOK mechanical part on a workbench
a close-up photo of TOK mechanical part in an industrial workshop
a clean product photo of TOK mechanical part on a white background
```

每个 prompt 分别用：

```text
base model
LoRA strength 0.6
LoRA strength 0.8
LoRA strength 1.0
```

生成对比图。

### 10.2 判断是否训练成功

成功迹象：

- 生成图像明显朝训练数据风格靠近；
- 结构更符合目标类型；
- 换场景后仍能保持主体特征；
- base model 的通用能力没有明显下降。

失败或过拟合迹象：

- 几乎复刻训练图；
- 稍微改 prompt 就不稳定；
- 手部、文字、结构质量下降；
- 色调污染所有 prompt；
- base model 原本擅长的内容变差。

### 10.3 建议记录表

| 检查项 | 记录 |
|---|---|
| 数据集版本 |  |
| 图片数量 |  |
| caption 来源 |  |
| rank / alpha |  |
| target modules |  |
| learning rate |  |
| steps |  |
| resolution |  |
| LoRA strength |  |
| 验证 prompt |  |
| 成功 / 失败现象 |  |

---

## 11. 训练后的使用方式

在 GPU 环境中加载 LoRA：

```python
import torch
from diffusers import QwenImagePipeline

MODEL = "/path/to/models/Qwen-Image-2512"
LORA = "/path/to/qwen-image-mechanical-lora"

pipe = QwenImagePipeline.from_pretrained(
    MODEL,
    dtype=torch.bfloat16,
    local_files_only=True,
)
pipe.to("cuda")

pipe.load_lora_weights(LORA)

image = pipe(
    prompt="a detailed photo of TOK mechanical part on a workbench",
    negative_prompt="blurry, low quality, distorted structure",
    true_cfg_scale=4.0,
    height=1024,
    width=1024,
    num_inference_steps=25,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]

image.save("result.png")
```

注意：

当前项目中的 `qwen-image-2512.py` 是 CPU 磁盘流式推理脚本，还没有接入 LoRA 加载逻辑。

后续可以增加：

```text
--lora-path
--lora-strength
```

让流式脚本也能测试 LoRA。

---

## 12. 硬件要求

当前本机环境：

```text
CPU-only
19 GiB RAM
无 CUDA GPU
```

不适合训练 Qwen-Image LoRA。

推荐硬件：

| 显存 | 可行性 |
|---:|---|
| 16GB | 很勉强，不建议 |
| 24GB | 可尝试 4bit QLoRA + offload + 小分辨率 |
| 48GB | LoRA 比较可行 |
| 80GB | 比较舒服 |
| full fine-tune | 通常需要多张高端 GPU |

第一阶段建议租用：

```text
4090 / 5090 / A100 / H100
```

并控制单次实验成本。

---

## 13. 后续 merge 和量化

只有当 LoRA 效果稳定后，才考虑 merge。

流程：

```text
1. 加载 base model
2. 加载 LoRA
3. fuse / merge LoRA
4. save_pretrained 导出完整模型
5. 再做 FP8 / GGUF / uint4 量化
```

不要一开始就训练量化模型。

推荐顺序：

```text
LoRA 验证
→ LoRA 调优
→ merge
→ 量化
→ 推理测试
```

---

## 14. 里程碑

### M0：目标定义

- [ ] 选定第一个图片类型；
- [ ] 明确是风格、主体还是场景；
- [ ] 设计触发词；
- [ ] 准备 10 张样例图作为人工参考。

### M1：小数据集

- [ ] 收集 50~200 张图；
- [ ] 清洗低质量图片；
- [ ] 检查分辨率和构图；
- [ ] 编写统一或独立 caption。

### M2：训练环境

- [ ] 租用 GPU；
- [ ] 安装最新 diffusers；
- [ ] 安装 accelerate / peft / bitsandbytes；
- [ ] 配置 accelerate；
- [ ] 跑通脚本 `--help`。

### M3：第一次 LoRA

- [ ] 使用 rank=8；
- [ ] 训练 300~800 steps；
- [ ] 保存 LoRA；
- [ ] 生成 base vs LoRA 对比图。

### M4：调优

- [ ] 根据结果调整 caption；
- [ ] 调整 steps / learning rate / rank；
- [ ] 尝试扩展 target modules；
- [ ] 增加数据多样性。

### M5：稳定版本

- [ ] 固定数据集版本；
- [ ] 固定训练超参；
- [ ] 建立验证 prompt 集；
- [ ] 保存多个 seed 的结果；
- [ ] 输出训练报告。

### M6：发布形态

- [ ] 决定发布 LoRA 还是 merged model；
- [ ] 如发布完整模型，执行 merge；
- [ ] 评估是否量化；
- [ ] 编写模型说明和使用示例。

---

## 15. 建议的执行顺序

```text
1. 选 1 个具体图片类型
2. 收集 50 ~ 200 张高质量图片
3. 编写详细 caption
4. 用官方 Qwen-Image LoRA 脚本训练
5. 使用 rank=8 做最小实验
6. 固定 seed 对比 base model
7. 判断是否欠拟合或过拟合
8. 调整数据、步数、rank
9. 稳定后再扩展到其他类型
10. 最后才考虑 merge 和量化
```

---

## 16. 核心结论

> 扩展 Qwen-Image 的图片生成类型，应该优先训练 `QwenImageTransformer2DModel` 的 LoRA。先用一个小而干净的数据集验证方向，再逐步提高 rank、步数和 target modules。当前 CPU 机器只适合做数据整理和文档规划，真正训练需要 GPU 环境。
