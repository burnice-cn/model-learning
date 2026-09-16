# 06. Qwen-Image-2512 流式运行

## 1. 流式加载解决什么问题？

当前机器：

```text
RAM: 19 GiB
Qwen-Image-2512 权重: 53.74 GiB
```

非流式加载必然失败。当前 `qwen-image-2512.py` 的方案是：

```text
模型配置和结构常驻内存
权重平时留在磁盘
执行到某个模块块时：
  从 safetensors 读取该块
  → 替换 meta 占位参数
  → forward 计算
  → 释放参数
  → 预取下一块
```

这样峰值内存只需要容纳：

- 当前块；
- 可能的预取块；
- activations；
- VAE 和少量常驻参数；
- Python 运行时。

## 2. 快速开始

### 2.1 激活环境

```bash
cd ~/MyProjects/model-learning
source .venv/bin/activate
export HF_HUB_OFFLINE=1
```

### 2.2 冒烟运行：2 步 / 512×512

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 2 \
  --seed 42 \
  --output qwen-smoke.png
```

> 2 步只验证路径，画质通常很低。CPU 上每一步都可能需要几十秒到数分钟，具体取决于分辨率、磁盘缓存和系统负载。

### 2.3 使用自定义 prompt

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 8 \
  --seed 42 \
  --prompt "a corgi sitting on the grass, warm sunlight, highly detailed photo" \
  --output corgi.png
```

### 2.4 从文件读取 prompt

```bash
python qwen-image-2512.py \
  --width 832 \
  --height 1216 \
  --steps 8 \
  --prompt-file prompt.txt \
  --output vertical.png
```

`--prompt-file -` 表示从 stdin 读取。

### 2.5 使用负面提示词和 true CFG

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

也可以把负面提示词放进 UTF-8 文件：

```bash
python qwen-image-2512.py \
  --width 512 \
  --height 512 \
  --steps 8 \
  --prompt-file prompt.txt \
  --negative-prompt-file negative.txt \
  --output corgi-cfg.png
```

规则：

- 不传负面提示词时，`true_cfg_scale` 为 `1.0`，走单路预测；
- 传入 `--negative-prompt` 或 `--negative-prompt-file` 时，如果不显式传 `--true-cfg-scale`，默认使用 `4.0`；
- 有负面提示词时，`--true-cfg-scale` 必须大于 `1.0`；
- 没有负面提示词时，不能设置大于 `1.0` 的 `true_cfg_scale`；
- `--negative-prompt` 与 `--negative-prompt-file` 只能二选一。

## 3. 参数说明

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--model` | `models/Qwen-Image-2512` | 本地模型目录 |
| `--prompt` | 官方示例 prompt | 正向提示词 |
| `--prompt-file` | 无 | 从 UTF-8 文件或 stdin 读取 |
| `--negative-prompt` | 无 | 负面提示词；提供后启用 batched true CFG |
| `--negative-prompt-file` | 无 | 从 UTF-8 文件或 stdin 读取负面提示词 |
| `--true-cfg-scale` | 条件默认 | 无负面提示词时为 `1.0`；有负面提示词时默认 `4.0` |
| `--width` | 必填 | 宽度，必须能被 16 整除 |
| `--height` | 必填 | 高度，必须能被 16 整除 |
| `--steps` | 25 | 去噪步数，别名 `--num-inference-steps` |
| `--seed` | 42 | 随机种子；`-1` 为随机 |
| `--output` | `output.png` | 输出图片 |
| `--no-vae-tiling` | 关闭开关 | 默认启用 VAE tiling |

当前脚本固定：

```text
device = cpu
dtype = float32
mode = disk-streaming
prefetch = True
```

true CFG 的行为由负面提示词决定：

```text
没有负面提示词:
  negative_prompt = None
  true_cfg_scale = 1.0
  每步 1 路 transformer forward

有负面提示词:
  true_cfg_scale 默认 4.0，可用 --true-cfg-scale 覆盖
  条件/无条件合成 batch=2
  每步 1 次 batched transformer forward
```

默认不做 true CFG 是为了降低首次验证时的计算和内存开销；需要更严格控制画质时再显式传入负面提示词。

## 4. 预期日志

启动时会看到类似：

```text
Building text_encoder with disk streaming...
Building transformer with disk streaming...
Loading model: /path/to/models/Qwen-Image-2512
Device: cpu
Dtype: float32
Mode: disk-streaming, no true CFG
Size: 512x512
Steps: 2, seed: 42
Output: /path/to/qwen-smoke.png
[stream] 首轮同步加载 ...
...
Saved image to: /path/to/qwen-smoke.png
```

如果启用负面提示词，会看到：

```text
Mode: disk-streaming, batched true CFG, scale 4
Using batched true CFG: cond + uncond in one transformer forward per step
```

首轮执行时块顺序还没学完，同步加载会更多；后续步骤会利用预取和已学习的执行顺序。

## 5. 流式实现原理

`qwen-image-2512.py` 内联实现了完整的流式执行器。

### 5.1 SafetensorsIndex：跨分片张量索引

safetensors 文件头结构：

```text
[8 字节 header 长度][JSON header][tensor 数据区]
```

JSON header 记录每个张量的：

```text
name
dtype
shape
data_offsets
```

`SafetensorsIndex` 会：

1. 打开多个分片；
2. 解析每个 header；
3. 建立全局张量索引；
4. 需要某个张量时，只在对应 shard 的字节区间读取它；
5. 为每个线程维护独立句柄，避免预取线程和主线程互相干扰。

### 5.2 meta model：先建结构，不占权重内存

构建模型时使用：

```python
with torch.device("meta"):
    model = model_cls.from_config(config)
```

meta 设备上的张量有正确的 shape 和 dtype，但不分配真实数据。因此模型结构可以完整存在，权重占用接近零。

### 5.3 plan_blocks：把权重划分成执行块

规则：

```text
子树很小 → 常驻，一次性加载
子树太大 → 沿子模块继续下钻
合适大小 → 作为一个流式块
```

脚本中的阈值：

```text
_STREAM_RESIDENT_MAX = 8 MiB
_STREAM_SPLIT_MAX = 400 MiB
```

为什么要按模块分块？因为前向执行有明确结构。把经常一起执行的参数放在同一块，可以减少加载次数。

### 5.4 forward hooks：执行前加载，执行后释放

对每个流式块注册：

```python
register_forward_pre_hook(...)  # 进入模块前：加载块
register_forward_hook(...)      # 离开模块后：释放块
```

流程：

```text
即将进入 block A
  → 从磁盘读 A 的参数
  → owner._parameters[attr] = 真实 Parameter
  → 执行 A.forward
  → owner._parameters[attr] = meta Parameter
  → 释放 A
  → 预取 B
```

参数替换发生在 `_parameters` 字典中，模块本身的 forward 逻辑不变，因此标准算子和标准执行语义保持一致。

### 5.5 预取

当当前块执行完成后，执行器会尝试读取下一个块：

```text
当前块计算 + 下一块磁盘读取 尽量重叠
```

如果磁盘读得够快，计算时等待时间会明显下降。

### 5.6 质量与标准执行的关系

流式执行器：

- 不改算子；
- 不改 forward 公式；
- 不改权重值；
- 只改变权重什么时候在真实设备上存在。

因此理论上，与同一 dtype、同一权重、同一执行路径的标准加载结果一致。实际工程中仍需用同 seed 做端到端对比验证，因为库版本、随机数设备、attention 实现或键名映射差异都可能影响结果。

## 6. Qwen 特有的兼容处理

### 6.1 text encoder 键名重映射

检查点是较旧的 Qwen/Transformers 键名，而当前 `transformers 5.17.0` 的模型结构不同。脚本通过 `RemappedSafetensorsIndex` 做映射：

```text
visual.*       → model.visual.*
model.*        → model.language_model.*
lm_head.*      → lm_head.*
```

读取文件时再映射回原始文件键名。

### 6.2 meta buffer 修复

RoPE 的 `inv_freq` 等非持久 buffer 在 meta 构建时可能是空壳。脚本用配置在 CPU 上重新计算：

```python
fix_meta_buffers(...)
```

### 6.3 QwenEmbedRope 修复

`QwenEmbedRope` 的 `pos_freqs` / `neg_freqs` 不是持久 buffer，而是构造函数里的普通张量属性。meta 模式下也需要显式重算：

```python
fix_qwen_embed_rope(...)
```

### 6.4 不调用 `pipe.to(...)`

流式模型的 text_encoder 和 transformer 中大量参数平时在 meta 设备上。如果调用：

```python
pipe.to("cpu")
```

PyTorch 会尝试搬移这些 meta 参数，可能触发错误。脚本只把小而常驻的 VAE 放到目标设备。

## 7. 负面提示词与 batched true CFG

官方管线做 true CFG 时，每个去噪步通常需要：

```text
1 次有条件 transformer forward
1 次无条件 transformer forward
```

对磁盘流式推理来说，这会带来一个关键问题：同一份 transformer 权重在同一 step 内可能被读取两遍，I/O 成本非常高。

当前脚本使用 `_generate_with_batched_true_cfg()`：

```text
正向 prompt embedding + 负面 prompt embedding
→ 合成 batch=2
→ 每步调用 1 次 transformer
→ 输出后拆成条件/无条件两路预测
→ 按 true_cfg_scale 融合
```

这样每步只需要流过一遍 transformer 权重，比两次独立 forward 更适合磁盘流式模式。

需要注意：

- batch=2 会让每步计算量和激活值内存接近翻倍；
- 权重 I/O 次数减少，但单次 forward 更重；
- CPU 总耗时仍可能明显高于无 CFG；
- 如果不传负面提示词，脚本保持 `true_cfg_scale=1.0`，不做 CFG。

## 8. 性能直觉

### 分辨率影响 token 数

Qwen 的 latent/patch 计算使得注意力规模随面积增长。512×512 尚可做冒烟；1024×1024 的计算和内存开销会显著增加。

### 每步权重 I/O

不做缓存时，每个 transformer step 都要重新读取权重。当前脚本使用预取，但权重不会长期留在 RAM。因此它牺牲速度换取可运行性。

启用 batched true CFG 后，同一 step 内条件/无条件共用一次权重读取；但 batch 变成 2，激活值和计算量会增加。

### 历史实测参考

在同类 WSL2/i5-14400/16~19 GiB 环境中，曾测得 512×512 约 60 秒/step。这个数字只作为量级参考，不代表当前文件系统缓存、后台负载、分辨率和库版本下的稳定性能。

## 9. 常见问题

### 9.1 提示 `width and height must be divisible by 16`

使用 16 的倍数：

```text
512, 832, 1024, 1216
```

### 9.2 模型目录不完整

脚本会输出缺失文件，例如：

```text
The local Qwen-Image model directory is incomplete:
- missing required file: ...
```

不要手工删改分片。缺哪个就补哪个。

### 9.3 磁盘空间不足

Qwen 模型约 54 GiB。确认 WSL ext4 所在的 Windows 物理盘有足够空间。不要把模型放在 `/mnt/c` 或 `/mnt/d` 后通过 9P 挂载随机读取；那会明显拖慢流式 I/O。

### 9.4 内存仍然不足

降低分辨率和步数只能减少 activations，不能减少模型本身。如果当前块、预取块、激活值仍超过 19 GiB，就需要进一步调小块或禁用预取。当前脚本没有暴露这些调参开关。

### 9.5 速度太慢

这是预期结果。当前方案的目标是“在有限 RAM 中可运行”，不是“CPU 快速生图”。要显著提速，需要：

- 大显存 GPU；
- 量化；
- 蒸馏模型；
- 更小的模型；
- 更高带宽的内存和存储。

## 10. 学习检查点

1. safetensors 为什么可以按张量读取？
2. meta model 为什么能先建立结构而不占权重内存？
3. forward hook 在什么时候加载和释放参数？
4. 预取如何隐藏磁盘延迟？
5. 不传负面提示词时，为什么脚本使用 `true_cfg_scale=1.0`？
6. 传入负面提示词后，batched true CFG 如何减少权重 I/O？
7. 为什么不能调用 `pipe.to(...)`？

完成本文后，可查看 [07-硬件与性能.md](07-硬件与性能.md)，了解这台机器为什么适合 SD1.5 实验，但只能以研究速度运行 Qwen-Image。
