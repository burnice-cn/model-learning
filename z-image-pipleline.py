import ctypes
import gc
import os
import readline  # noqa: F401  # 启用 GNU Readline，让 input() 支持 Backspace 和左右方向键。
from pathlib import Path

print("脚本已启动，正在准备运行环境...", flush=True)

# 这必须在导入 torch 之前设置，否则 CUDA 分配器可能不会应用该配置。
# expandable_segments 可以减少显存碎片，降低长推理过程中的 OOM 概率。
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

print("正在导入 torch 和 modelscope...", flush=True)
import torch
from modelscope import ZImagePipeline
print("依赖导入完成。", flush=True)


def read_prompt():
    """读取单行提示词；输入完成后按回车确认。"""
    try:
        prompt = input("请输入提示词（exit 或 quit 退出）: ").strip()
    except EOFError:
        return None

    if prompt.lower() in {"exit", "quit"}:
        return None

    return prompt


def read_size():
    """读取图片尺寸，支持 1024 和 1024x768 两种写法。"""
    while True:
        raw = input("请输入图片尺寸（1024 或 1024x768，默认 1024）: ").strip()
        if not raw:
            return 1024, 1024

        parts = raw.lower().replace("×", "x").split("x")
        try:
            if len(parts) == 1:
                height = width = int(parts[0])
            elif len(parts) == 2:
                height, width = int(parts[0]), int(parts[1])
            else:
                raise ValueError
        except ValueError:
            print("尺寸格式不正确，请输入 1024 或 1024x768。")
            continue

        if height <= 0 or width <= 0:
            print("宽高必须大于 0。")
            continue

        return height, width


def read_steps():
    """读取推理步数，未填写时使用默认值。"""
    default_steps = int(os.getenv("Z_IMAGE_STEPS", "12"))

    while True:
        raw = input(f"请输入推理步数（默认 {default_steps}）: ").strip()
        if not raw:
            return default_steps

        try:
            steps = int(raw)
        except ValueError:
            print("推理步数必须是整数。")
            continue

        if not 1 <= steps <= 50:
            print("推理步数必须在 1 到 50 之间。")
            continue

        return steps


def read_guidance_scale():
    """读取 CFG 强度；默认值可通过 Z_IMAGE_GUIDANCE_SCALE 设置。"""
    default_guidance_scale = float(os.getenv("Z_IMAGE_GUIDANCE_SCALE", "0.0"))

    while True:
        raw = input(f"请输入 guidance_scale（默认 {default_guidance_scale:g}）: ").strip()
        if not raw:
            return default_guidance_scale

        try:
            guidance_scale = float(raw)
        except ValueError:
            print("guidance_scale 必须是数字。")
            continue

        if not 0 <= guidance_scale <= 20:
            print("guidance_scale 必须在 0 到 20 之间。")
            continue

        return guidance_scale


def count_prompt_tokens(pipe, prompt):
    """按 ZImagePipeline 实际使用的 Qwen 聊天模板统计提示词 token 数。"""
    rendered_prompt = pipe.tokenizer.apply_chat_template(
        [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    token_ids = pipe.tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        truncation=False,
        padding=False,
        return_attention_mask=False,
    )["input_ids"]

    return len(token_ids)


def read_filename():
    """读取保存文件名，未填写时默认使用 example.png。"""
    raw = input("请输入保存文件名（默认 example.png）: ").strip()
    filename = raw if raw else "example.png"

    if not filename.lower().endswith(".png"):
        filename += ".png"

    return filename


def release_memory():
    """主动回收 Python 对象、CUDA 缓存和 glibc 堆内存，降低常驻进程的内存压力。"""
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Linux 的 Python 进程在释放张量后，RSS 可能不会立刻下降。
    # malloc_trim 可以把已释放的堆内存归还给操作系统，减少第二轮被 OOM Killer 杀掉的概率。
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


@torch.inference_mode()
def main():
    # 使用 bfloat16 加载模型，显存约为 float32 的四分之一到一半。
    # low_cpu_mem_usage=True 可以避免加载时在主机内存中复制一份完整模型。
    print("正在加载 Z-Image 模型，首次启动可能需要较长时间...", flush=True)
    pipe = ZImagePipeline.from_pretrained(
        "models/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("模型权重加载完成，正在配置显存和磁盘 offload...", flush=True)

    # 默认使用“混合放置”策略：
    # 1. transformer 和 VAE 常驻 CUDA，避免每轮生成反复搬运最大的 DiT 权重；
    # 2. text_encoder 使用磁盘分块 offload，推理时按层从磁盘读入 CUDA；
    # 3. VAE 开启分块解码，降低解码阶段的显存峰值。
    #
    # 之前使用的 enable_model_cpu_offload() 会把完整模型放在主机 RAM 中，
    # 常驻模式下第二轮更容易被系统 OOM Killer 杀掉。
    # 磁盘 offload 会在首次使用文本编码器时把它的权重写入本地磁盘，
    # 之后 RAM 中只保留空壳张量，因此能明显降低常驻 RAM 占用。
    #
    # 注意：offload 目录大约需要容纳文本编码器权重的磁盘空间。
    if not hasattr(pipe, "enable_group_offload"):
        raise RuntimeError(
            "当前 diffusers 版本不支持 enable_group_offload，"
            "请升级 diffusers 到支持 ZImagePipeline 的较新版本。"
        )

    text_encoder_offload_dir = os.getenv(
        "Z_IMAGE_TEXT_ENCODER_OFFLOAD_DIR",
        ".z-image-text-encoder-offload",
    )
    os.makedirs(text_encoder_offload_dir, exist_ok=True)

    pipe.enable_group_offload(
        onload_device=torch.device("cuda"),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        offload_to_disk_path=text_encoder_offload_dir,
        exclude_modules=["transformer", "vae"],
    )

    # 直接开启 VAE 分块解码，降低最后一步图像解码的显存峰值。
    # ZImagePipeline 使用的是 diffusers 的 AutoencoderKL，该类原生支持 enable_tiling()。
    pipe.vae.enable_tiling()

    print("模型加载完成，进入常驻生成模式。", flush=True)

    while True:
        prompt = read_prompt()
        if prompt is None:
            print("收到退出指令，程序结束。")
            break

        if not prompt:
            print("提示词为空，请重新输入。")
            continue

        # ZImagePipeline 默认会把提示词送入 512 token 的文本编码器窗口。
        # 超出后会被静默截断，后半部分约束就不会参与生图，这是长提示词
        # “不遵守约束”的常见原因。这里提前统计并在超限时拒绝生成，
        # 避免用户误以为所有提示词都已生效。
        max_prompt_tokens = int(os.getenv("Z_IMAGE_MAX_PROMPT_TOKENS", "480"))
        prompt_token_count = count_prompt_tokens(pipe, prompt)
        print(f"提示词长度：{prompt_token_count} tokens", flush=True)

        if prompt_token_count > max_prompt_tokens:
            print(
                f"提示词过长：{prompt_token_count} tokens，"
                f"超过限制 {max_prompt_tokens} tokens。请压缩提示词后重试。",
                flush=True,
            )
            print(
                "建议优先保留主体、动作、构图、风格，删除重复形容词和次要背景。",
                flush=True,
            )
            continue

        height, width = read_size()
        num_inference_steps = read_steps()
        guidance_scale = read_guidance_scale()
        filename = read_filename()

        try:
            # 推理步数和 CFG 强度均支持自定义。
            # Z-Image Turbo 是为 guidance_scale=0 的无 CFG 推理优化的蒸馏模型。
            # 提高到 8 可能导致过饱和、伪影、僵硬和结构崩坏，因此默认保持 0。
            image = pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=torch.Generator("cuda").manual_seed(42),
            ).images[0]

            image.save(filename)
            print(f"生成成功，已保存到：{Path(filename).resolve()}")

            # 释放上一轮图片和中间对象，避免常驻循环中 RSS 持续升高。
            del image
            release_memory()
        except torch.cuda.OutOfMemoryError:
            print("显存不足，生成失败。可以尝试输入更小的尺寸，例如 768 或 768x768。")
            release_memory()
        except Exception as error:
            print(f"生成失败：{error}")
            release_memory()

        print("继续等待下一次输入。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n收到 Ctrl+C，程序结束。", flush=True)
    except BaseException as error:
        print(f"程序异常退出：{type(error).__name__}: {error}", flush=True)
        raise
