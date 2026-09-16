# 03. diffusers 管线与模型加载原理

## 1. 为什么需要 Pipeline？

一个图像生成系统不是单一网络，而是一条流水线：

```text
prompt
→ tokenizer
→ text encoder
→ 初始 latent 噪声
→ scheduler 安排时间步
→ transformer/UNet 去噪
→ VAE 解码
→ PIL Image
```

diffusers 的 `Pipeline` 负责把这些组件装配起来，并定义调用顺序。

## 2. `model_index.json` 是装配清单

打开 `models/stable-diffusion-v1-5/model_index.json`：

```json
{
  "_class_name": "StableDiffusionPipeline",
  "_diffusers_version": "0.6.0",
  "scheduler": ["diffusers", "PNDMScheduler"],
  "text_encoder": ["transformers", "CLIPTextModel"],
  "tokenizer": ["transformers", "CLIPTokenizer"],
  "unet": ["diffusers", "UNet2DConditionModel"],
  "vae": ["diffusers", "AutoencoderKL"],
  "safety_checker": ["stable_diffusion", "StableDiffusionSafetyChecker"],
  "feature_extractor": ["transformers", "CLIPImageProcessor"]
}
```

### 保留字段

| 字段 | 含义 |
|---|---|
| `_class_name` | 应该实例化哪个 Pipeline 类 |
| `_diffusers_version` | 保存时的 diffusers 版本，便于兼容判断 |

### 组件字段

```json
"组件名": ["库名或管线模块名", "类名"]
```

例如：

```json
"unet": ["diffusers", "UNet2DConditionModel"]
```

表示：读取 `unet/config.json` 和权重，创建 `diffusers.UNet2DConditionModel`，再赋给 pipeline 的 `pipe.unet`。

## 3. 本地目录结构

### Stable Diffusion v1.5

```text
models/stable-diffusion-v1-5/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── tokenizer/
│   ├── merges.txt
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   └── vocab.json
├── text_encoder/
│   ├── config.json
│   └── model.fp16.safetensors
├── unet/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
├── vae/
│   ├── config.json
│   └── diffusion_pytorch_model.fp16.safetensors
├── feature_extractor/
│   └── preprocessor_config.json
└── safety_checker/  # 本仓库未下载完整权重，加载时显式禁用
```

### Qwen-Image-2512

```text
models/Qwen-Image-2512/
├── model_index.json
├── scheduler/
│   └── scheduler_config.json
├── tokenizer/
├── text_encoder/
│   ├── config.json
│   ├── model-00001-of-00004.safetensors
│   ├── ...
│   └── model.safetensors.index.json
├── transformer/
│   ├── config.json
│   ├── diffusion_pytorch_model-00001-of-00009.safetensors
│   ├── ...
│   └── diffusion_pytorch_model.safetensors.index.json
└── vae/
    ├── config.json
    └── diffusion_pytorch_model.safetensors
```

Qwen 的两个大组件是分片保存的，需要 index 文件中的 `weight_map` 找到每个参数在哪个 shard。

## 4. `from_pretrained` 的加载流程

概念上执行：

```text
1. 读取 model_index.json
2. 确定 Pipeline 类
3. 遍历组件清单
4. 为每个组件读取 config
5. 根据类名构造模型
6. 根据权重文件名和 variant 找权重
7. 把权重载入参数
8. 创建 scheduler/tokenizer
9. 组装 Pipeline 对象
10. 调用 pipeline(...) 时按流程执行
```

伪代码：

```python
pipeline_cls = StableDiffusionPipeline

pipe = pipeline_cls(
    scheduler=PNDMScheduler.from_pretrained(path, subfolder="scheduler"),
    text_encoder=CLIPTextModel.from_pretrained(path, subfolder="text_encoder"),
    tokenizer=CLIPTokenizer.from_pretrained(path, subfolder="tokenizer"),
    unet=UNet2DConditionModel.from_pretrained(path, subfolder="unet"),
    vae=AutoencoderKL.from_pretrained(path, subfolder="vae"),
    safety_checker=None,
    feature_extractor=None,
)
```

实际代码由 diffusers 的 loading utils 统一处理。

## 5. `variant="fp16"` 是什么？

本仓库 SD1.5 权重文件名带 `.fp16`：

```text
model.fp16.safetensors
diffusion_pytorch_model.fp16.safetensors
```

加载时指定：

```python
StableDiffusionPipeline.from_pretrained(
    model_path,
    variant="fp16",
    dtype=torch.float32,
)
```

这里有两个不同概念：

| 设置 | 含义 |
|---|---|
| `variant="fp16"` | 磁盘上选择哪一套文件 |
| `dtype=torch.float32` | 加载后用哪个 dtype 计算 |

SD1.5 权重以 fp16 存储，脚本再转成 fp32 计算，是为了 CPU 兼容性。文件仍是 fp16 variant，不代表运行 dtype 一定是 fp16。

## 6. `local_files_only=True`

本项目模型已经在本地，脚本使用：

```python
local_files_only=True
```

含义：

- 不访问 Hugging Face Hub；
- 只使用本地目录；
- 如果文件缺失，直接报错，而不是尝试下载。

这是离线运行的推荐方式。

## 7. 分片 safetensors 和 index

Qwen 的 `model.safetensors.index.json` 大致结构：

```json
{
  "metadata": {
    "total_size": 16614700000
  },
  "weight_map": {
    "model.layers.0.attn.qkv.weight": "model-00001-of-00004.safetensors",
    "model.layers.10.attn.qkv.weight": "model-00002-of-00004.safetensors"
  }
}
```

加载器会：

1. 读取 `weight_map`；
2. 打开对应 shard；
3. 找到张量 offset；
4. 读取并赋给模型参数。

### safetensors 为什么适合流式读取？

safetensors 文件开头有元数据：

```text
8 字节 header 长度 + JSON header + 原始数据区
```

JSON header 中记录：

```json
{
  "tensor_name": {
    "dtype": "BF16",
    "shape": [4096, 4096],
    "data_offsets": [0, 33554432]
  }
}
```

因此可以只读某个张量的字节区间，而不必加载整个文件。当前 Qwen 流式脚本正是利用这一点。

## 8. Pipeline 的推理流程

### Stable Diffusion v1.5

```text
1. tokenize(prompt)
2. text_encoder(token_ids) → prompt_embeds
3. prepare latents：随机生成 [1, 4, 64, 64]
4. scheduler.set_timesteps(steps)
5. for t in timesteps:
     latent_input = cat(noisy_latent, empty_latent)
     noise_pred = unet(latent_input, t, encoder_hidden_states=prompt_embeds)
     noise_pred_text, noise_pred_uncond = chunk(2)
     noise_pred = uncond + guidance_scale * (text - uncond)
     noisy_latent = scheduler.step(noise_pred, t, noisy_latent)
6. vae.decode(latent) → image tensor
7. postprocess → PIL.Image
```

### Qwen-Image-2512

```text
1. tokenize(prompt)
2. Qwen text encoder → prompt embeddings
3. prepare flow-matching latents
4. FlowMatchEulerDiscreteScheduler 生成 sigmas/timesteps
5. transformer 反复预测 velocity/noise 方向并更新 latents
6. VAE decode
7. 输出 PIL.Image
```

## 9. Scheduler 和 num_inference_steps

`num_inference_steps` 是推理去噪步数，不是训练时间步。

- 步数太少：快，但结构可能不完整；
- 步数适中：质量/时间平衡；
- 步数很高：不一定明显更好，速度成本却线性增加。

SD1.5 常用 20~50 步；Qwen-Image 官方默认可达 50 步，本项目 CPU 冒烟用 2~8 步。

## 10. 学习检查点

1. `model_index.json` 中的每个字段分别指向哪里？
2. `variant="fp16"` 和 `dtype=torch.float32` 有什么区别？
3. `local_files_only=True` 有什么好处？
4. 分片 safetensors 的 index 文件有什么作用？
5. 为什么 safetensors 能支持按张量读取？

理解加载流程后，请进入 [04-StableDiffusion-v1-5运行指南.md](04-StableDiffusion-v1-5运行指南.md)。
