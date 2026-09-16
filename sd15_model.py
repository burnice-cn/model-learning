#!/usr/bin/env python3
"""Generate an image with Stable Diffusion v1.5 on CPU."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an image with Stable Diffusion v1.5 on CPU.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/stable-diffusion-v1-5",
        help="local model directory",
    )
    parser.add_argument(
        "--prompt",
        default="a cat astronaut floating in space, cinematic lighting",
        help="positive prompt",
    )
    parser.add_argument(
        "--steps",
        "--num-inference-steps",
        dest="steps",
        type=int,
        default=20,
        help="number of denoising steps",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed; use -1 for a random seed",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="sd15.png",
        help="output image path",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=14,
        help="CPU thread count; override OMP_NUM_THREADS when set",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise SystemExit(f"--steps must be positive, got {args.steps}")
    if args.threads <= 0:
        raise SystemExit(f"--threads must be positive, got {args.threads}")

    threads = int(__import__("os").environ.get("OMP_NUM_THREADS", str(args.threads)))
    torch.set_num_threads(threads)
    torch.set_grad_enabled(False)

    seed = args.seed
    if seed == -1:
        import random
        seed = random.randint(0, 2**32 - 1)

    output_path = Path(args.output).expanduser().resolve()
    model_path = Path(args.model).expanduser().resolve()
    if not model_path.is_dir():
        raise SystemExit(f"model directory does not exist: {model_path}")

    print(f"Loading model: {model_path}")
    print(f"Device: cpu, dtype: float32")
    print(f"Prompt: {args.prompt}")
    print(f"Steps: {args.steps}, seed: {seed}")
    print(f"Output: {output_path}")

    t0 = time.perf_counter()
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        variant="fp16",
        dtype=torch.float32,
        safety_checker=None,
        requires_safety_checker=False,
        local_files_only=True,
    )
    pipe.to("cpu")
    print(f"[1/2] pipeline loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    t1 = time.perf_counter()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    image = pipe(
        prompt=args.prompt,
        num_inference_steps=args.steps,
        guidance_scale=7.5,
        generator=generator,
    ).images[0]
    elapsed = time.perf_counter() - t1
    print(
        f"[2/2] image generated in {elapsed:.1f}s ({elapsed / args.steps:.1f}s/step)",
        flush=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
