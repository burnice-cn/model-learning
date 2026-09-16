# 05. Qwen-Image-2512 非流式运行

## 1. 非流式运行的含义

“非流式”指标准加载方式：

```text
启动时：
  从磁盘读取全部主要权重
  → 放入 RAM 或 VRAM
推理时：
  不再读模型文件
  → 每步只做计算
```

这是最直接、最不容易出错的方式，但对 Qwen-Image-2512 来说，本机内存不够。

## 2. 本地模型大小

Qwen-Image-2512 的分片权重实际占用约为 **53.74 GiB**（约 57.7 GB）：

| 组件 | 权重大小 |
|---|---:|
| `text_encoder` | 15.45 GiB |
| `transformer` | 38.06 GiB |
| `vae` | 0.24 GiB |
| 合计 | 53.74 GiB |

这还只是模型文件。非流式推理还需要：

- activations；
- attention 中间结果；
- latent tensors；
- VAE decode 临时内存；
- Python/PyTorch 运行时；
- 操作系统余量。

## 3. 硬件要求

### bf16 权重

如果保持 bfloat16：

```text
权重约 54 GiB + 运行开销
```

建议：

| 场景 | 最低建议 |
|---|---|
| CPU RAM 推理 | 72 GiB 以上，80~128 GiB 更稳 |
| CUDA VRAM 推理 | 64 GiB 以上，80 GiB 更稳 |
| 本机 19 GiB RAM | 无法非流式加载 |

### fp32 权重

如果转换成 float32：

```text
权重约 107 GiB + 运行开销
```

CPU 兼容性更好，但内存和计算量都明显增加。

> 当前 WSL2 环境只有 19 GiB 内存，因此不要在本机直接运行本文的非流式示例。它的意义是先理解标准机制，再理解为什么必须流式。

## 4. 架构概览

`model_index.json`：

```json
{
  "_class_name": "QwenImagePipeline",
  "scheduler": ["diffusers", "FlowMatchEulerDiscreteScheduler"],
  "text_encoder": ["transformers", "Qwen2_5_VLForConditionalGeneration"],
  "tokenizer": ["transformers", "Qwen2Tokenizer"],
  "transformer": ["diffusers", "QwenImageTransformer2DModel"],
  "vae": ["diffusers", "AutoencoderKLQwenImage"]
}
```

### 4.1 Text Encoder

Qwen-Image 使用 `Qwen2_5_VLForConditionalGeneration` 作为文本编码器。它远大于 SD1.5 的 CLIP：

| 项目 | SD1.5 CLIP | Qwen2.5-VL |
|---|---:|---:|
| hidden size | 768 | 3584 |
| 层数 | 12 | 28 |
| 最大位置长度 | 77 | 128000 |
| 权重大小 | 0.23 GiB | 15.45 GiB |

### 4.2 Transformer 去噪网络

配置：

```json
{
  "num_layers": 60,
  "num_attention_heads": 24,
  "attention_head_dim": 128,
  "in_channels": 64,
  "out_channels": 16,
  "joint_attention_dim": 3584,
  "patch_size": 2
}
```

它是整个模型中最大的部分，约 38.06 GiB。相比 SD1.5 的 UNet，它的文本理解、长提示词处理和图文联合注意力能力都强得多。

### 4.3 VAE

Qwen-Image 的 VAE 使用 16 个 latent 通道：

```json
{
  "z_dim": 16
}
```

并且 `vae.enable_tiling()` 可以把解码过程分块，降低高分辨率输出时的峰值内存。

### 4.4 Scheduler

Qwen-Image 使用：

```text
FlowMatchEulerDiscreteScheduler
```

它不是 SD1.5 的 PNDM。概念上，Flow Matching 学习的是从噪声分布到数据分布的“速度场”，推理时沿着这个速度场逐步移动 latent。

## 5. 标准非流式 Python 示例

以下代码适合拥有足够 RAM/VRAM 的机器。不要在当前 19 GiB WSL2 环境运行。

```python
from pathlib import Path

import torch
from diffusers import QwenImagePipeline

MODEL = Path("models/Qwen-Image-2512").resolve()
OUTPUT = Path("qwen-standard.png")

# CUDA + 大显存机器可用 bfloat16。
# CPU 大内存机器也理论上可用 bfloat16，但速度可能很慢。
device = "cuda"          # 或 "cpu"
dtype = torch.bfloat16

pipe = QwenImagePipeline.from_pretrained(
    MODEL,
    dtype=dtype,
    local_files_only=True,
)
pipe.to(device)

# VAE 分块解码，降低高分辨率时的内存峰值。
pipe.vae.enable_tiling()

prompt = (
    "A highly detailed portrait of a corgi sitting on grass, "
    "warm sunlight, sharp focus, natural colors"
)

generator = torch.Generator(device=device).manual_seed(42)

image = pipe(
    prompt=prompt,
    negative_prompt=None,
    true_cfg_scale=1.0,
    height=1024,
    width=1024,
    num_inference_steps=25,
    generator=generator,
).images[0]

image.save(OUTPUT)
print(f"saved {OUTPUT}")
```

### 5.1 启用 True CFG

如果提供负面提示并提高 `true_cfg_scale`：

```python
image = pipe(
    prompt=prompt,
    negative_prompt="blurry, low quality, distorted anatomy",
    true_cfg_scale=4.0,
    ...
).images[0]
```

概念上每一步会进行有条件和无条件两路预测，再按 scale 融合。计算量约为单路的两倍，占用内存也更高。

### 5.2 CPU fp32 版本

某些 CPU 环境对 bf16 支持不佳，可以尝试：

```python
device = "cpu"
dtype = torch.float32
```

但这会把权重大小扩大到约 108 GiB，并进一步降低速度。必须有大内存机器。

## 6. 为什么 diffusers 的 CPU offload 不等于流式？

diffusers 提供：

```python
pipe.enable_model_cpu_offload()
pipe.enable_sequential_cpu_offload()
```

它们可以缓解 VRAM 不足，但基本思路是：

```text
权重仍在 CPU RAM 中
GPU 用到哪部分，就搬哪部分
```

对 Qwen-Image 来说：

- `text_encoder` + `transformer` 约 53.5 GiB；
- 本机 RAM 只有 19 GiB；
- 所以即使用 CPU offload，也放不下。

真正的磁盘流式必须让权重平时留在磁盘，只在执行某块前临时读取。

## 7. 非流式运行的优缺点

### 优点

- 调用方式简单；
- 与官方管线行为最接近；
- 不需要改加载器；
- 推理时没有反复读权重文件的 I/O；
- 更容易排查问题。

### 缺点

- 启动前必须一次性放下整个模型；
- RAM/VRAM 门槛极高；
- 不适合 19 GiB 内存的学习机。

## 8. 验证文件完整性

当前流式脚本启动前会检查：

- `model_index.json`；
- scheduler、text_encoder、tokenizer、transformer、VAE 配置；
- 两个分片 index；
- index 中引用的每个 shard；
- shard 总大小是否小于 `metadata.total_size`。

非流式示例没有这么完整的自定义检查，但可以直接交给 `from_pretrained`。如果缺少文件，它会报出对应组件加载错误。

## 9. 本机不能跑时，如何继续学习？

1. 阅读并理解本文的标准加载代码；
2. 分析 `model_index.json` 和分片 index；
3. 计算 bf16/fp32 内存需求；
4. 进入 [06-Qwen-Image-2512流式运行.md](06-Qwen-Image-2512流式运行.md)，看如何把权重留在磁盘，只加载正在执行的块。

## 10. 学习检查点

1. Qwen-Image 的三个主要大文件组件分别是什么？
2. bf16 和 fp32 非流式内存需求大约是多少？
3. 为什么 CPU offload 仍然救不了 19 GiB 本机？
4. `true_cfg_scale` 为什么会增加计算量？
5. 为什么 `pipe.vae.enable_tiling()` 有助于高分辨率输出？
