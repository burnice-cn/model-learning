# model-learning

这是一个本地学习 Stable Diffusion 与 Qwen-Image 推理，并规划 Qwen-Image LoRA 训练的项目。当前环境为 Windows + WSL2、CPU-only、Python 3.12。

## 学习入口

从 [docs/README.md](docs/README.md) 开始。文档按从 0 到进阶排序：

1. 大模型基础
2. Tokenizer 与文本编码
3. diffusers 管线与模型加载
4. Stable Diffusion v1.5 实跑
5. Qwen-Image-2512 非流式运行
6. Qwen-Image-2512 磁盘流式运行
7. 硬件与性能
8. Qwen-Image 图片类型扩展训练计划

## 快速开始

### 1. 进入项目目录

```bash
cd ~/MyProjects/model-learning
```

### 2. 安装 uv

如果还没有安装 `uv`，在 WSL/Ubuntu 中执行：

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

让当前 shell 重新读取 PATH：

```bash
exec "$SHELL"
```

确认安装成功：

```bash
uv --version
```

> 如果已经安装过 `uv`，跳过这一步即可。

### 3. 创建 Python 3.12 虚拟环境

项目固定使用 Python 3.12。如果 `.venv` 还不存在，执行：

```bash
uv venv --python 3.12 .venv
```

如果系统提示找不到 Python 3.12，`uv` 会提示下载并管理对应版本；也可以先写入项目 Python 版本：

```bash
echo 3.12 > .python-version
uv venv --python 3.12 .venv
```

### 4. 激活虚拟环境

```bash
source .venv/bin/activate
python -V
```

预期输出类似：

```text
Python 3.12.x
```

### 5. 设置 Hugging Face 离线模式

本项目使用本地 `models/` 目录中的模型文件，不希望 Hugging Face 在启动时访问网络、检查远端仓库或尝试下载缺失文件。在当前 shell 中设置一次：

```bash
export HF_HUB_OFFLINE=1
```

这个设置会保留在当前 shell 的环境中，后续启动的 Python 命令都会继承它，因此后面运行 Qwen-Image 时不需要每次重复写 `HF_HUB_OFFLINE=1`。

注意：它只在当前 shell 会话中有效；打开新终端后需要重新执行一次。

作用：

- 强制 Hugging Face Hub 相关代码离线工作；
- 只读取本地文件；
- 网络不通时避免远程请求等待；
- 本地模型缺文件时直接报错，而不是自动下载。

如果之后确实需要访问 Hugging Face 下载模型，可以取消：

```bash
unset HF_HUB_OFFLINE
```

> 脚本内部已经使用了 `local_files_only=True`；`export HF_HUB_OFFLINE=1` 是进程级额外保险。

### 6. 安装依赖

先安装当前项目验证过的 CPU 版 PyTorch：

```bash
uv pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  'https://mirrors.aliyun.com/pytorch-wheels/cpu/torch-2.9.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
```

再安装 torchvision CPU 版：

```bash
uv pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  'https://mirrors.aliyun.com/pytorch-wheels/cpu/torchvision-0.24.1%2Bcpu-cp312-cp312-manylinux_2_28_x86_64.whl'
```

最后安装其他依赖：

```bash
uv pip install \
  --index-url https://mirrors.aliyun.com/pypi/simple/ \
  diffusers transformers safetensors packaging pillow numpy
```

确认关键依赖版本：

```bash
python - <<'PYCHECK'
import torch, diffusers, transformers, safetensors
print("torch:", torch.__version__)
print("diffusers:", diffusers.__version__)
print("transformers:", transformers.__version__)
print("safetensors:", safetensors.__version__)
PYCHECK
```

### 7. 运行 Stable Diffusion v1.5

先查看参数：

```bash
python sd15_model.py --help
```

常规生成：

```bash
python sd15_model.py \
  --prompt "a cat astronaut floating in space, cinematic lighting" \
  --steps 20 \
  --seed 42 \
  --threads 14 \
  --output sd15.png
```

如果只想确认环境是否可用，可以先跑 1 步冒烟测试；1 步画质会很糊：

```bash
python sd15_model.py \
  --steps 1 \
  --threads 14 \
  --output sd15-smoke.png
```

### 8. 运行 Qwen-Image-2512

Qwen-Image-2512 使用磁盘流式加载。先查看参数：

```bash
python qwen-image-2512.py --help
```

低分辨率冒烟测试：

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 2 \
  --seed 42 \
  --output qwen-smoke.png
```

自定义提示词：

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 8 \
  --seed 42 \
  --prompt "a corgi sitting on the grass, warm sunlight, highly detailed photo" \
  --output corgi.png
```

使用负面提示词和 true CFG：

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 8 \
  --seed 42 \
  --prompt "a corgi sitting on the grass, warm sunlight, highly detailed photo" \
  --negative-prompt "blurry, low quality, distorted anatomy, extra fingers" \
  --true-cfg-scale 4.0 \
  --output corgi-cfg.png
```

不传 `--negative-prompt` 时，脚本使用 `true_cfg_scale=1.0`，只做单路预测；传入负面提示词后，默认使用 `true_cfg_scale=4.0`，并把条件/无条件样本合成 batch=2，减少流式权重读取次数。

脚本默认启用两类优化：

- `.cache/qwen-image-prompts/` 缓存精确 prompt embedding；同一 prompt 重复生成时不再重新流式读取 16GB 文本编码器。
- `--cpu-cache-gib 6` 把最多约 6GiB 的 transformer 权重按磁盘原始精度保留在 RAM；命中时仍转换为 fp32，不改变输出。

如需关闭或强制刷新：

```bash
--no-prompt-cache
--refresh-prompt-cache
--cpu-cache-gib 0
```

`--cfg-steps N` 是实验性速度/画质折中：只在前 N 步启用 true CFG，会改变最终图片；省略它则保持全量 CFG。

> 注意：
>
> - 宽高必须是 16 的倍数。
> - `models/Qwen-Image-2512/` 必须已经下载完整；脚本使用本地文件，不会自动下载模型。
> - CPU 流式推理每步可能需要几十秒到数分钟，建议先用 `--steps 2` 验证。

### 9. 远程运行 Qwen-Image-2.1

`qwen-image2.py` 面向约 22GiB GPU、24GiB CPU RAM 的远程环境，只加载本地模型目录，不会下载模型。脚本对全部模型组件使用 block-level 磁盘 group offload，按块把权重加载到 CUDA，避免 GPU 和主机内存同时被完整组件占满。请先使用你自己的下载源获取 `Qwen-Image-2.1`，然后通过 `--model` 指向该目录。首次运行前安装模型主页要求的依赖：

```bash
pip install "torch>=2.4.0" "transformers>=5.17" accelerate pillow
pip uninstall -y diffusers
pip install "diffusers @ git+https://github.com/huggingface/diffusers"
```

注意：不要只执行 `pip install diffusers`。当前 PyPI 的 `diffusers==0.40.0` 还不包含 `QwenImage21Pipeline`；必须安装 GitHub `main` 分支。

文生图：

```bash
python qwen-image2.py \
  --mode text-to-image \
  --prompt "A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night, reflections on wet pavement" \
  --aspect-ratio 16:9 \
  --seed 42 \
  --output qwen-image2-t2i.png
```

图像编辑：

```bash
python qwen-image2.py \
  --mode image-edit \
  --prompt "Change the background to a sunset beach" \
  --aspect-ratio 1:1 \
  --reference-image input.png \
  --output qwen-image2-edit.png
```

说明：

- `--mode` 默认是 `text-to-image`，可选 `image-edit`。
- `--prompt` 默认为空。
- `--aspect-ratio` 默认是 `1:1`，支持 `1:1`、`4:3`、`3:4`、`3:2`、`2:3`、`16:9`、`9:16`。
- `image-edit` 模式必须传入 `--reference-image`；重复传入最多支持 10 张参考图。
- 脚本不再使用 `enable_model_cpu_offload()`，也不会让 transformer 和 VAE 常驻 CUDA；所有组件都会按块 offload 到磁盘。
- offload 目录默认在模型目录旁的 `.qwen-image2-offload/`，也可用环境变量 `QWEN_IMAGE2_OFFLOAD_DIR` 指定。
- 该模式比常驻或整组件 offload 慢，但能同时缓解 22GiB GPU 和 24GiB CPU RAM 的限制。
- `--model` 只接受本地模型目录，默认为 `models/Qwen-Image-2.1`；如模型在其他位置，请显式传入路径。

## 项目结构

```text
docs/                    学习文档
models/stable-diffusion-v1-5/
models/Qwen-Image-2512/
sd15_model.py            SD1.5 CPU 推理入口
qwen-image-2512.py        Qwen-Image CPU 磁盘流式推理入口
qwen-image2.py           Qwen-Image-2.1 远程 GPU 推理入口
.venv/                   本项目虚拟环境
```

## 依赖

当前验证版本：

```text
Python 3.12
torch 2.9.1+cpu
diffusers 0.40.0
transformers 5.17.0
safetensors 0.8.0
numpy 2.5.3
pillow 12.3.0
packaging 26.3
```

## 注意

- 本机没有可用 NVIDIA GPU，推理默认使用 CPU。
- Qwen-Image-2512 权重约 53.74 GiB，不能在本机 19 GiB 内存下完整加载；当前脚本使用磁盘流式加载。
- `models/` 和 `.venv/` 不应提交到版本库。
