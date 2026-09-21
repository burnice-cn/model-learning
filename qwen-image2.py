#!/usr/bin/env python3
"""Run Qwen-Image-2.1 text-to-image and image-edit inference.

The script loads a local Qwen-Image-2.1 model directory only. On a 22GiB GPU
with ~24GiB CPU RAM, neither whole-component CPU offload nor keeping the
transformer/VAE resident is viable, so all model components use block-level
disk group offloading. Download the checkpoint yourself with your preferred
source before running it.

Examples
--------
Text to image:

    python qwen-image2.py \
        --prompt 'A neon shop sign that reads "QWEN IMAGE 2.1"' \
        --aspect-ratio 16:9 \
        --seed 42

Image editing:

    python qwen-image2.py \
        --mode image-edit \
        --prompt 'Change the background to a sunset beach' \
        --aspect-ratio 1:1 \
        --reference-image input.png

Use a downloaded local model directory:

    python qwen-image2.py --model /path/to/Qwen-Image-2.1 --prompt 'a cat'

Remote installation requirements (from the model card):

    pip install 'torch>=2.4.0' 'transformers>=5.17' accelerate pillow
    pip uninstall -y diffusers
    pip install 'diffusers @ git+https://github.com/huggingface/diffusers'
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

# Set this before importing PyTorch. It reduces fragmentation-related OOM risk.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1": (2048, 2048),
    "4:3": (2400, 1792),
    "3:4": (1792, 2400),
    "3:2": (2528, 1696),
    "2:3": (1696, 2528),
    "16:9": (2752, 1536),
    "9:16": (1536, 2752),
}

MODE_ALIASES = {
    "text-to-image": "text-to-image",
    "text2image": "text-to-image",
    "t2i": "text-to-image",
    "txt2img": "text-to-image",
    "image-edit": "image-edit",
    "image-editing": "image-edit",
    "edit": "image-edit",
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="Qwen-Image-2.1 inference supporting text-to-image and image editing.",
    )
    parser.add_argument(
        "--mode",
        default="text-to-image",
        choices=tuple(MODE_ALIASES),
        help="Inference mode. Default: text-to-image.",
    )
    parser.add_argument(
        "--prompt",
        default="",
        help="Prompt. Default: no prompt.",
    )
    parser.add_argument(
        "--aspect-ratio",
        default="1:1",
        choices=tuple(ASPECT_RATIOS),
        help=(
            "Output aspect ratio and its native resolution.\n"
            "Default: 1:1 (2048x2048).\n"
            "Choices: 1:1, 4:3, 3:4, 3:2, 2:3, 16:9, 9:16."
        ),
    )
    parser.add_argument(
        "--reference-image",
        "--image",
        dest="reference_images",
        action="append",
        metavar="PATH",
        help=(
            "Reference image for image-edit mode. This argument is required in\n"
            "image-edit mode and is rejected in text-to-image mode. Repeat it to\n"
            "provide up to 10 reference images."
        ),
    )
    parser.add_argument(
        "--model",
        default="models/Qwen-Image-2.1",
        help=(
            "Local Qwen-Image-2.1 model directory.\n"
            "Default: models/Qwen-Image-2.1."
        ),
    )
    parser.add_argument(
        "--output",
        help="Output image path. Default: qwen-image2-<mode>-<timestamp>.png.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=40,
        help="Number of inference steps. Default: 40.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed; use -1 for a random seed. Default: 42.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help=(
            "CUDA accelerator, e.g. cuda or cuda:1. Model blocks are loaded to this device as needed.\n"
            "Default: cuda."
        ),
    )
    parser.add_argument(
        "--dtype",
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype. Default: bfloat16.",
    )
    return parser


def normalize_mode(mode: str) -> str:
    return MODE_ALIASES[mode]


def resolve_output_path(mode: str, requested: str | None) -> Path:
    if requested:
        path = Path(requested).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        path = Path(f"qwen-image2-{mode}-{timestamp}.png")
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resolve_model_path(model: str) -> str:
    """Validate and return a local model path."""
    local_path = Path(model).expanduser()
    if not local_path.exists():
        raise SystemExit(
            f"Model directory does not exist: {local_path}\n"
            "Download Qwen-Image-2.1 with your preferred source, then pass its "
            "local directory with --model."
        )
    if not local_path.is_dir():
        raise SystemExit(f"Model path is not a directory: {local_path}")
    if not (local_path / "model_index.json").is_file():
        raise SystemExit(
            f"Model directory does not contain model_index.json: {local_path}"
        )
    return str(local_path)


def load_reference_images(paths: Sequence[str]) -> list[object]:
    from PIL import Image, ImageOps

    if not 1 <= len(paths) <= 10:
        raise SystemExit("Image-edit mode requires 1 to 10 reference images.")

    images: list[object] = []
    for path_text in paths:
        path = Path(path_text).expanduser()
        if not path.is_file():
            raise SystemExit(f"Reference image does not exist: {path}")
        try:
            with Image.open(path) as image:
                # Follow EXIF orientation and force the file to be fully loaded.
                oriented = ImageOps.exif_transpose(image)
                oriented.load()
                images.append(oriented)
        except Exception as error:
            raise SystemExit(f"Cannot read reference image {path}: {error}") from error
    return images


def build_pipeline(model_path: str, dtype_name: str, device: str):
    import torch

    try:
        from diffusers import QwenImage21Pipeline
    except ImportError as error:
        import diffusers

        raise SystemExit(
            "This diffusers installation does not provide QwenImage21Pipeline.\n"
            f"diffusers version: {getattr(diffusers, '__version__', 'unknown')}\n"
            f"diffusers path: {getattr(diffusers, '__file__', 'unknown')}\n"
            "Qwen-Image-2.1 currently requires the diffusers main branch from GitHub.\n"
            "Install it with:\n"
            "  pip uninstall -y diffusers\n"
            "  pip install 'diffusers @ git+https://github.com/huggingface/diffusers'"
        ) from error

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]

    offload_dir = Path(
        os.environ.get(
            "QWEN_IMAGE2_OFFLOAD_DIR",
            str(Path(model_path).expanduser().parent / ".qwen-image2-offload"),
        )
    ).expanduser()
    offload_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading Qwen-Image-2.1 from: {model_path}")
    print(f"dtype={dtype_name}, device={device}")
    print(f"Model block placement: {device}")
    print(f"Disk group offload directory: {offload_dir}")
    pipe = QwenImage21Pipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )

    # A 22GiB GPU cannot hold transformer+VAE and still load text-encoder blocks,
    # while ~24GiB CPU RAM cannot hold whole inactive components under model CPU
    # offload. Offload every component to disk and load one small block group at
    # a time. This is slower, but is the only strategy here that targets both
    # GPU and host-memory limits.
    pipe.enable_group_offload(
        onload_device=torch.device(device),
        offload_device=torch.device("cpu"),
        offload_type="block_level",
        num_blocks_per_group=1,
        offload_to_disk_path=str(offload_dir),
    )

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_tiling"):
        pipe.vae.enable_tiling()

    return pipe


def check_device(device: str) -> None:
    import torch

    if not device.startswith("cuda"):
        raise SystemExit(
            f"Unsupported device '{device}'. Disk group offloading requires a CUDA device."
        )
    if not torch.cuda.is_available():
        raise SystemExit(
            f"CUDA is not available for device '{device}'. "
            "Install a CUDA build of PyTorch; --device cpu is not supported."
        )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.steps <= 0:
        parser.error("--steps must be a positive integer")
    if args.seed < -1:
        parser.error("--seed must be >= -1")

    mode = normalize_mode(args.mode)
    if mode == "image-edit" and not args.reference_images:
        parser.error("--reference-image is required when --mode is image-edit")
    if mode == "text-to-image" and args.reference_images:
        parser.error("--reference-image is only valid when --mode is image-edit")

    width, height = ASPECT_RATIOS[args.aspect_ratio]
    output_path = resolve_output_path(mode, args.output)
    seed = random.randint(0, 2**32 - 1) if args.seed == -1 else args.seed

    check_device(args.device)
    model_path = resolve_model_path(args.model)

    import torch

    pipe = build_pipeline(
        model_path=model_path,
        dtype_name=args.dtype,
        device=args.device,
    )

    # Keep RNG state on CPU; model blocks are loaded as needed.
    generator = torch.Generator(device="cpu").manual_seed(seed)

    call_kwargs = {
        "prompt": args.prompt,
        "width": width,
        "height": height,
        "num_inference_steps": args.steps,
        "generator": generator,
    }
    if mode == "image-edit":
        reference_images = load_reference_images(args.reference_images)
        call_kwargs["image"] = (
            reference_images[0] if len(reference_images) == 1 else reference_images
        )

    print(
        f"Generating: mode={mode}, aspect_ratio={args.aspect_ratio}, "
        f"size={width}x{height}, steps={args.steps}, seed={seed}"
    )

    try:
        with torch.inference_mode():
            image = pipe(**call_kwargs).images[0]
    except TypeError as error:
        # Older or incompatible diffusers versions may not expose the same call API.
        raise SystemExit(
            f"Pipeline call failed: {error}\n"
            "Qwen-Image-2.1 requires the latest diffusers branch from GitHub."
        ) from error

    if image.mode in ("RGBA", "LA") and output_path.suffix.lower() in {".jpg", ".jpeg"}:
        image = image.convert("RGB")

    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
