# 从 0 到运行图像大模型：学习路线

这个仓库是一套“能落地”的学习环境：先用 **Stable Diffusion v1.5** 建立直觉并验证环境，再理解 **Qwen-Image-2512** 的架构、非流式加载的硬件要求，最后研究如何用“磁盘流式加载”把一个约 54 GiB 的模型放进 19 GiB 内存的机器里跑，并规划后续图片类型扩展的 LoRA 训练。

## 学习顺序

| 顺序 | 文档 | 你会得到什么 |
|---|---|---|
| 1 | [01-大模型入门.md](01-大模型入门.md) | 张量、参数、模型文件类型、Transformer、扩散模型、显存/内存的基础模型 |
| 2 | [02-Tokenizer与文本编码.md](02-Tokenizer与文本编码.md) | Tokenizer、embedding、text encoder 的关系，以及 CLIP/Qwen 的差异 |
| 3 | [03-diffusers管线与模型加载.md](03-diffusers管线与模型加载.md) | `model_index.json`、`from_pretrained`、fp16 variant、组件装配 |
| 4 | [04-StableDiffusion-v1-5运行指南.md](04-StableDiffusion-v1-5运行指南.md) | 用本地模型在 CPU 上真正生成第一张图 |
| 5 | [05-Qwen-Image-2512非流式运行.md](05-Qwen-Image-2512非流式运行.md) | 标准加载方式、模型体积、为什么本机 19 GiB 内存放不下 |
| 6 | [06-Qwen-Image-2512流式运行.md](06-Qwen-Image-2512流式运行.md) | 当前 `qwen-image-2512.py` 的运行方法与流式实现原理 |
| 7 | [07-硬件与性能.md](07-硬件与性能.md) | WSL2/CPU/内存/磁盘约束，以及如何判断方案是否可行 |
| 8 | [08-Qwen-Image训练计划.md](08-Qwen-Image训练计划.md) | 扩展 Qwen-Image 图片生成类型的 LoRA 训练路线和实验计划 |

## 最重要的三个结论

1. **Stable Diffusion v1.5 是教学模型**：本地权重约 1.99 GiB，19 GiB 内存可以直接加载，CPU 也能在几分钟内出图。它适合用来理解 tokenizer、text encoder、UNet、VAE、scheduler。
2. **Qwen-Image-2512 非流式加载很简单，但硬件要求高**：本地权重约 53.74 GiB。标准 `from_pretrained(...).to(device)` 需要一次性把主要组件放进内存/显存，本机 19 GiB 内存无法完成。
3. **流式加载改变的是权重存放方式，不是生成算法**：模型结构仍在磁盘上，只有即将执行的块被读入内存；执行前物化、执行后释放，再用预取掩盖一部分磁盘延迟。理论上质量与标准执行一致，代价是更多 I/O 和调度开销。

## 环境速查

```bash
cd ~/MyProjects/model-learning
source .venv/bin/activate
export HF_HUB_OFFLINE=1

python -V          # Python 3.12
python - <<'PY'
import torch, diffusers, transformers, safetensors
print(torch.__version__)
print(diffusers.__version__)
print(transformers.__version__)
print(safetensors.__version__)
PY
```

当前验证过的版本：

| 软件 | 版本 |
|---|---|
| Python | 3.12 |
| torch | 2.9.1+cpu |
| diffusers | 0.40.0 |
| transformers | 5.17.0 |
| safetensors | 0.8.0 |
| numpy | 2.5.3 |
| pillow | 12.3.0 |
| packaging | 26.3 |

## 快速命令

### 1. Stable Diffusion v1.5

```bash
# 1 步只用于环境冒烟，画面通常是糊的
python sd15_model.py \
  --steps 1 \
  --threads 14 \
  --output .tmp-doc-sd-smoke.png

# 正常测试
python sd15_model.py \
  --prompt "a cat astronaut floating in space, cinematic lighting" \
  --steps 20 \
  --seed 42 \
  --threads 14 \
  --output sd15.png
```

### 2. Qwen-Image-2512 流式运行

```bash
# 先用 2 步验证能不能跑通；宽高必须是 16 的倍数
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 2 \
  --seed 42 \
  --output qwen-smoke.png

# 可选：负面提示词 + true CFG
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 8 \
  --prompt "a corgi sitting on the grass, warm sunlight, highly detailed photo" \
  --negative-prompt "blurry, low quality, distorted anatomy, extra fingers" \
  --true-cfg-scale 4.0 \
  --output corgi-cfg.png
```

> 非流式 Qwen-Image 示例见 [05-Qwen-Image-2512非流式运行.md](05-Qwen-Image-2512非流式运行.md)。它需要远大于本机内存的机器，不要在当前 19 GiB WSL2 环境里直接尝试。

## 本地模型

| 模型 | 目录 | 主要权重 | 能否在本机直接完整加载 |
|---|---|---:|---|
| Stable Diffusion v1.5 | `models/stable-diffusion-v1-5` | 约 1.99 GiB | 可以 |
| Qwen-Image-2512 | `models/Qwen-Image-2512` | 约 53.74 GiB | 不可以；需要流式加载 |

## 术语约定

- **LLM**：严格说是以 Transformer 为基础、逐 token 自回归生成文本的模型。Qwen2.5-VL 的文本编码器有这种 Transformer 结构，但整个 Qwen-Image 项目是图像生成模型，不是纯 LLM。
- **图像大模型**：这里泛指参数量大、结构为深度神经网络、用于图像生成/理解的模型。
- **非流式加载**：启动时把模型组件完整读入内存/显存，之后前向计算不再读权重文件。
- **流式加载**：模型配置常驻内存，权重按执行块临时读入，用完释放。
- **VRAM/RAM**：VRAM 是显卡显存，RAM 是主内存；本机没有可用 NVIDIA GPU，因此本项目主要讨论 RAM。
