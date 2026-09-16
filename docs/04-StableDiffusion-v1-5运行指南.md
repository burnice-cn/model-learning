# 04. Stable Diffusion v1.5 运行指南

## 1. 目标

用本地 `models/stable-diffusion-v1-5` 在 CPU 上生成一张图，并通过这个过程理解图像扩散管线的每个组件。

## 2. 硬件与环境要求

| 项目 | 当前项目状态 |
|---|---|
| CPU | Intel i5-14400，16 线程 |
| GPU | 无可用 NVIDIA GPU |
| 内存 | WSL2 分配 19 GiB |
| 模型大小 | 约 1.99 GiB 权重 |
| Python | 3.12 |
| torch | 2.9.1+cpu |

SD1.5 在 19 GiB 内存下可以直接完整加载。首次加载后，操作系统的文件缓存会让后续运行更快。

## 3. 组件结构

| 组件 | 类 | 作用 |
|---|---|---|
| tokenizer | `CLIPTokenizer` | 把 prompt 变成 token id |
| text_encoder | `CLIPTextModel` | 输出文本语义向量 |
| unet | `UNet2DConditionModel` | 对 latent 去噪 |
| vae | `AutoencoderKL` | latent 与 RGB 图像互转 |
| scheduler | `PNDMScheduler` | 决定时间步和更新公式 |
| safety_checker | `StableDiffusionSafetyChecker` | 安全检查，本项目禁用 |
| feature_extractor | `CLIPImageProcessor` | 配合安全检查器使用，本项目禁用 |

## 4. 权重体积

```text
text_encoder/model.fp16.safetensors              246,144,864 bytes
unet/diffusion_pytorch_model.fp16.safetensors  1,719,125,304 bytes
vae/diffusion_pytorch_model.fp16.safetensors    167,335,342 bytes
合计                                             约 1.99 GiB
```

注意：目录里还有指向 fp16 文件的普通名软链接，例如：

```text
text_encoder/model.safetensors -> model.fp16.safetensors
```

这些软链接让 diffusers 能用默认文件名找到权重，同时磁盘上没有第二份真实文件。

## 5. 运行命令

进入项目并激活虚拟环境：

```bash
cd ~/MyProjects/model-learning
source .venv/bin/activate
```

### 5.1 最小冒烟：1 步

```bash
python sd15_model.py \
  --steps 1 \
  --threads 14 \
  --output .tmp-doc-sd-smoke.png
```

1 步通常只能验证环境和管线，画面可能是糊的。当前环境实测这次命令可完整跑通，整体约 14 秒，其中模型加载约 3 秒，1 步去噪约 9 秒。

### 5.2 常规测试：20 步

```bash
python sd15_model.py \
  --prompt "a cat astronaut floating in space, cinematic lighting" \
  --steps 20 \
  --seed 42 \
  --threads 14 \
  --output sd15.png
```

### 5.3 使用随机种子

```bash
python sd15_model.py \
  --prompt "a cyberpunk city at night, rain, reflections" \
  --steps 25 \
  --seed -1 \
  --output cyberpunk.png
```

`--seed -1` 表示随机选择种子。控制其他参数不变时，相同种子应得到相同初始噪声，结果可复现。

## 6. 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `models/stable-diffusion-v1-5` | 本地模型目录 |
| `--prompt` | a cat astronaut... | 正向提示词 |
| `--steps` | 20 | 去噪步数，别名 `--num-inference-steps` |
| `--seed` | 42 | 随机种子；`-1` 为随机 |
| `--output` | `sd15.png` | 输出路径 |
| `--threads` | 14 | PyTorch CPU 线程数 |

脚本中的固定推理参数：

```python
guidance_scale = 7.5
safety_checker = None
requires_safety_checker = False
dtype = torch.float32
device = cpu
```

## 7. 生成过程详解

### 7.1 文本编码

```text
prompt
→ CLIPTokenizer
→ CLIPTextModel
→ prompt embeddings
```

SD1.5 的 CLIP 文本路径有 77 token 限制。过长 prompt 会被截断，后文不会参与生成。

### 7.2 准备 latent

512×512 图像不会直接以 512×512×3 的 RGB 张量进入 UNet。它先在 VAE 的 latent 空间中处理：

```text
RGB 512×512×3
↔ latent 64×64×4
```

VAE 把空间分辨率缩小 8 倍：

```text
512 / 8 = 64
```

随机初始 latent 的形状约为：

```text
[batch=1, channels=4, height=64, width=64]
```

### 7.3 去噪

UNet 在每个时间步接收：

- 当前 noisy latent；
- 当前 timestep；
- 文本 embeddings。

它输出对噪声的预测，scheduler 再更新 latent。

### 7.4 CFG：Classifier-Free Guidance

SD1.5 默认 `guidance_scale=7.5`。概念上：

```text
噪声预测 = 无条件预测
         + guidance_scale × (有条件预测 - 无条件预测)
```

这会增强 prompt 的引导效果。数值太低图像可能偏离 prompt，太高则可能过饱和或结构变形。

### 7.5 VAE 解码

去噪完成后：

```text
clean latent [1, 4, 64, 64]
→ VAE decoder
→ RGB tensor [1, 3, 512, 512]
→ PIL.Image
```

## 8. 代码里最重要的部分

`sd15_model.py` 的加载代码：

```python
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    variant="fp16",
    dtype=torch.float32,
    safety_checker=None,
    requires_safety_checker=False,
    local_files_only=True,
)
pipe.to("cpu")
```

推理代码：

```python
generator = torch.Generator(device="cpu").manual_seed(seed)
image = pipe(
    prompt=args.prompt,
    num_inference_steps=args.steps,
    guidance_scale=7.5,
    generator=generator,
).images[0]
```

## 9. 预期日志

```text
Loading model: /path/to/models/stable-diffusion-v1-5
Device: cpu, dtype: float32
Prompt: ...
Steps: 20, seed: 42
Output: /path/to/sd15.png
[1/2] pipeline loaded in ...s
[2/2] image generated in ...s (...s/step)
Saved image to: /path/to/sd15.png
```

首次运行可能提示：

```text
Cannot initialize model with low cpu memory usage because `accelerate` was not found.
Defaulting to `low_cpu_mem_usage=False`.
```

SD1.5 仍然可以运行。如果之后想减少加载峰值内存，可以安装 `accelerate`，但对本项目当前 SD1.5 用法不是必需的。

## 10. 常见问题

### 10.1 模型目录不存在

```text
model directory does not exist: ...
```

检查路径和当前工作目录。推荐始终从项目根目录运行。

### 10.2 `--steps` 无效

脚本要求 `--steps > 0`。冒烟用 1，常规用 20~30。

### 10.3 图片很糊

1 步或极少步数只适合验证环境，不代表模型质量。改用 20 步以上。

### 10.4 CPU 太慢

可尝试：

```bash
--threads 16
```

但线程不是越多越好；16 线程机器上 14~16 都可以测试。CPU 推理本身就是慢路径，若要快速生成，需要支持 CUDA 的 GPU。

### 10.5 提示词无效

- 检查是否超过 CLIP 77 token；
- 避免只写抽象词，加入主体、场景、光线、风格；
- 使用相同 seed 对比不同 prompt 更容易观察变化。

## 11. 学习检查点

1. SD1.5 的 512×512 图像为什么对应 64×64 latent？
2. CLIP text encoder 的输出如何影响 UNet？
3. `guidance_scale` 为什么能让图像更贴合 prompt？
4. 为什么要禁用 safety checker？
5. `--seed` 固定了什么随机性？

跑通并理解后，进入 [05-Qwen-Image-2512非流式运行.md](05-Qwen-Image-2512非流式运行.md)。
