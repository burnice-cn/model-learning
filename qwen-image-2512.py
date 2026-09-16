#!/usr/bin/env python3
"""Generate an image with Qwen-Image-2512 using CPU disk-streaming.

Examples
--------
Generate with custom dimensions and prompt:

    python qwen-image-2512.py \
        --width 1216 \
        --height 1824 \
        --steps 25 \
        --seed 123 \
        --output girl.png \
        --prompt "A highly detailed portrait photo, sharp focus, natural lighting"

Generate with a prompt file:

    python qwen-image-2512.py --prompt-file prompt.txt

Run against a local model directory:

    python qwen-image-2512.py \
        --model ./models/Qwen-Image-2512 \
        --width 1056 \
        --height 1584
"""

from __future__ import annotations

import argparse
import json
import math
from collections import OrderedDict
import numpy as np
import os
import random
import struct
import sys
import threading
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from packaging.version import InvalidVersion, Version

# This must be set before CUDA is initialized. It only takes effect when the user
# has not already configured the PyTorch CUDA allocator.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from safetensors import safe_open

try:
    from diffusers import QwenImagePipeline
except ImportError:  # Keep the failure actionable in _validate_runtime_dependencies.
    QwenImagePipeline = None


# ---------------------------------------------------------------------------
# Inlined disk-to-device streaming implementation
#
# This is a self-contained CPU disk-streaming implementation: no separate
# executor or validation helper module is required.
# ---------------------------------------------------------------------------

_STREAM_RESIDENT_MAX = 8 << 20
_STREAM_SPLIT_MAX = 400 << 20

_DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8_E4M3": 1, "F8_E5M2": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}


def read_rss_mb() -> float | None:
    """当前进程 RSS（MB），读 /proc/self/status。"""
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024
    except Exception:
        pass
    return None


class RSSMonitor(threading.Thread):
    """后台采样峰值 RSS 的监视线程。"""

    def __init__(self, interval_s: float = 0.05):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self._stop_event = threading.Event()
        self._peak = 0.0
        self._samples = 0

    def run(self):
        while not self._stop_event.is_set():
            v = read_rss_mb()
            if v is not None:
                self._peak = max(self._peak, v)
                self._samples += 1
            self._stop_event.wait(self.interval_s)

    def stop(self) -> float:
        self._stop_event.set()
        self.join(timeout=2)
        return self._peak

    @property
    def peak_mb(self) -> float:
        return self._peak


# ---------------------------------------------------------------- 张量索引

class SafetensorsIndex:
    """
    跨多个 safetensors 分片的只读张量索引。

    - 头部手工解析（struct + json）：一次读 8 字节长度 + JSON 头，
      即拿到全部张量的 dtype/shape/offset，体现 safetensors
      天生支持"按张量随机访问"的格式优势。
    - 实际取数交给 safetensors 库（内部 mmap，按 offset 区间读取）。
    - 每线程独立文件句柄（threading.local），预取线程与主线程互不干扰。
    """

    def __init__(self, files: list[str]):
        self.files = [str(f) for f in files]
        self._loc: dict[str, int] = {}
        self._meta: dict[str, tuple] = {}
        for i, path in enumerate(self.files):
            with open(path, "rb") as fp:
                (hlen,) = struct.unpack("<Q", fp.read(8))
                header = json.loads(fp.read(hlen))
            for key, meta in header.items():
                if key == "__metadata__":
                    continue
                self._loc[key] = i
                self._meta[key] = (meta["dtype"], tuple(meta["shape"]), meta["data_offsets"])
        self._tls = threading.local()

    # -- 元信息 --------------------------------------------------

    @property
    def names(self) -> list[str]:
        return list(self._loc)

    def numel(self, name: str) -> int:
        return math.prod(self._meta[name][1])

    def size_bytes(self, name: str) -> int:
        dtype, shape, _ = self._meta[name]
        return math.prod(shape) * _DTYPE_BYTES[dtype]

    # -- 读取 ----------------------------------------------------

    def _handle(self, file_idx: int):
        handles = getattr(self._tls, "handles", None)
        if handles is None:
            handles = self._tls.handles = {}
        if file_idx not in handles:
            handles[file_idx] = safe_open(self.files[file_idx], framework="pt")
        return handles[file_idx]

    def read(self, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
        """按张量名从 mmap 读出（只触碰该张量的字节区间），可选上转精度。"""
        t = self._handle(self._loc[name]).get_tensor(name)
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t


# ---------------------------------------------------------------- 块划分

def plan_blocks(model, index, resident_max_bytes: int, split_max_bytes: int):
    """
    把模型参数划分成有序的"执行块"。

    规则（自顶向下）：
      - 子树权重 <= resident_max_bytes  → 常驻（一次性加载，不释放）
      - 子树权重 >  split_max_bytes     → 沿子模块继续下钻切分
      - 其余 / 有直属参数 / 无子模块    → 一个流式块

    返回 (block_roots, resident_roots)
    """
    names = index.names
    named_modules = dict(model.named_modules())
    sizes: dict[str, int] = {}
    for n in names:
        sz = index.size_bytes(n)
        parts = n.split(".")
        for i in range(len(parts)):
            key = ".".join(parts[:i])
            sizes[key] = sizes.get(key, 0) + sz

    children: dict[str, list[str]] = {}
    for mod_name in dict(model.named_modules()):
        if not mod_name:
            continue
        parent = mod_name.rsplit(".", 1)[0] if "." in mod_name else ""
        children.setdefault(parent, []).append(mod_name)

    # 参数直属归属：name 的倒数第二段是所属模块（最后一段是属性名）
    direct_owner = {n: (n.rsplit(".", 1)[0] if "." in n else "") for n in names}

    blocks: list[str] = []
    residents: list[str] = []

    def choose(prefix: str):
        if sizes.get(prefix, 0) <= resident_max_bytes:
            residents.append(prefix)
            return
        # 关键规则：ModuleList/ModuleDict 容器通常只被"遍历"而不被直接调用，
        # 挂在容器上的前向钩子永远不会触发，因此必须下钻到真正被调用的子模块
        mod = named_modules[prefix]
        kids = [c for c in children.get(prefix, []) if sizes.get(c, 0) > 0]
        if isinstance(mod, (torch.nn.ModuleList, torch.nn.ModuleDict)) and kids:
            for c in kids:
                choose(c)
            return
        has_direct = any(direct_owner[n] == prefix for n in names
                         if n == prefix or n.startswith(prefix + "."))
        if has_direct or not kids or sizes[prefix] <= split_max_bytes:
            blocks.append(prefix)
            return
        for c in kids:
            choose(c)

    for root_child in sorted(children.get("", [])):
        choose(root_child)

    # 校验：每个张量必须恰好归属一个块/常驻根，不重不漏
    assigned: dict[str, str] = {}
    for root in blocks + residents:
        for n in names:
            if n == root or n.startswith(root + "."):
                if n in assigned:
                    raise RuntimeError(f"块划分重叠: {n} 同时属于 {assigned[n]} 和 {root}")
                assigned[n] = root
    missing = set(names) - set(assigned)
    if missing:
        raise RuntimeError(f"有 {len(missing)} 个张量未被任何块覆盖，如 {sorted(missing)[:3]}")
    return blocks, residents

class StreamingExecutor:
    """
    给 meta 模型挂上前向钩子，实现"按块加载 → 计算 → 释放 → 预取下一块"。

    钩子机制保证：物化后的参数在 forward 语义下与标准加载完全等价，
    因此前向数值与标准管线逐比特一致（同一份权重、同一组算子）。
    """

    def __init__(self, model, index: SafetensorsIndex, dtype=torch.float32,
                 resident_max_bytes: int = 8 << 20,
                 split_max_bytes: int = 500 << 20,
                 prefetch: bool = True, verbose: bool = True,
                 device: str | torch.device = "cpu",
                 cache_max_bytes: int = 0,
                 cache_reserve_bytes: int = 4 << 30,
                 cache_policy: str = "prefix"):
        self.model = model
        self.index = index
        self.dtype = dtype
        self.device = torch.device(device)
        self.prefetch_enabled = prefetch
        self.verbose = verbose
        self.cache_max_bytes = max(0, int(cache_max_bytes))
        self.cache_reserve_bytes = max(0, int(cache_reserve_bytes))
        self.cache_policy = cache_policy
        if self.cache_policy not in {"prefix", "lru"}:
            raise ValueError(f"unknown cache policy: {cache_policy}")
        self.cache_enabled = self.cache_max_bytes > 0 and self.device.type == "cuda"
        self._gpu_cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self._gpu_cache_bytes = 0
        self._active_tensors: dict[str, dict[str, torch.Tensor]] = {}

        self._named = dict(model.named_modules())
        self._block_params: dict[str, list[str]] = {}
        self._resident_params: dict[str, list[str]] = {}
        self._loaded: set[str] = set()
        self._hooks: list = []

        self._order: list[str] = []   # 执行顺序（首次前向时自动学习）
        self._ptr = -1
        self._staged: dict[str, dict] = {}
        self._prefetch: tuple[str, threading.Thread] | None = None
        self._lock = threading.Lock()

        self.stats = {
            "block_calls": 0, "steps": 0,
            "sync_loads": 0, "prefetch_hits": 0, "cache_hits": 0,
            "bytes_read": 0, "load_seconds": 0.0,
            "block_table": {},
            "peak_rss_mb": 0.0,
            "gpu_cache_bytes": 0,
        }

        blocks, residents = plan_blocks(model, index, resident_max_bytes, split_max_bytes)
        self._blocks = blocks
        self._residents = residents
        # 每个根的参数直属映射：(模块对象, 属性名, 全名, 形状)
        # 物化/释放都通过替换 module._parameters 实现（与 Module.to 的内部机制一致）
        self._owners = {r: self._owners_of(r) for r in blocks + residents}
        for b in blocks:
            self._block_params[b] = sorted(
                n for n in index.names if n == b or n.startswith(b + "."))
        for r in residents:
            self._resident_params[r] = sorted(
                n for n in index.names if n == r or n.startswith(r + "."))

        # 模型参数必须与索引完全对齐（防止配置/文件不匹配）
        model_param_names = {n for n, _ in model.named_parameters()}
        index_names = set(index.names)
        if model_param_names != index_names:
            only_model = sorted(model_param_names - index_names)[:3]
            only_index = sorted(index_names - model_param_names)[:3]
            raise RuntimeError(
                f"模型参数与 safetensors 索引不一致\n"
                f"  仅在模型中: {only_model}\n  仅在索引中: {only_index}")

        self._materialize_residents()
        self._attach_hooks()

    # ---------- 钩子 ----------

    def _make_pre_hook(self, block: str):
        def hook(module, args):
            self._on_enter_block(block)
        return hook

    def _make_post_hook(self, block: str):
        def hook(module, args, output):
            self._on_exit_block(block)
        return hook

    def _attach_hooks(self):
        for b in self._blocks:
            mod = self._named[b]
            self._hooks.append(mod.register_forward_pre_hook(self._make_pre_hook(b)))
            self._hooks.append(mod.register_forward_hook(self._make_post_hook(b)))

    def _owners_of(self, root: str):
        """枚举 root 子树里每个参数的 (直属模块, 属性名, 全名, 形状)。"""
        mod = self._named[root]
        owners = []
        for rel, p in mod.named_parameters(recurse=True):
            parts = rel.split(".")
            owner = mod
            for sub in parts[:-1]:
                owner = getattr(owner, sub)
            owners.append((owner, parts[-1], f"{root}.{rel}" if rel else root, tuple(p.shape)))
        return owners

    def _materialize_root(self, root: str, tensors: dict):
        """物化：用真实张量替换 _parameters 条目（推理原型统一 requires_grad=False）。"""
        for owner, attr, full, shape in self._owners[root]:
            owner._parameters[attr] = torch.nn.Parameter(tensors[full], requires_grad=False)

    def _free_root(self, root: str):
        """释放：把 _parameters 条目换回 meta 占位符（零内存），供下次物化。"""
        for owner, attr, full, shape in self._owners[root]:
            owner._parameters[attr] = torch.nn.Parameter(
                torch.empty(shape, dtype=self.dtype, device="meta"), requires_grad=False)

    def _materialize_residents(self):
        t0 = time.perf_counter()
        for r in self._residents:
            tensors = self._load_tensors(self._resident_params[r], count_stats=False)
            self._materialize_root(r, tensors)
            self._loaded.add(r)
        if self.verbose:
            mb = sum(self.index.size_bytes(n) for r in self._residents
                     for n in self._resident_params[r]) / 1e6
            print(f"[stream] 常驻块 {len(self._residents)} 个（{mb:.1f} MB），"
                  f"{time.perf_counter()-t0:.2f}s 加载完成", flush=True)

    def _root_size_bytes(self, root: str) -> int:
        # size_bytes describes the checkpoint dtype. Account for a requested fp32 load,
        # which doubles the on-device size of a bf16 checkpoint.
        element_size = torch.empty((), dtype=self.dtype).element_size()
        return sum(
            self.index.numel(name) for name in self._block_params[root]
        ) * element_size

    def _maybe_cache_block(self, block: str, tensors: dict[str, torch.Tensor] | None) -> None:
        if tensors is None or not self.cache_enabled or block in self._gpu_cache:
            return

        size = self._root_size_bytes(block)
        if size > self.cache_max_bytes:
            return

        # Keep a hard reserve of unallocated CUDA memory for activations, attention,
        # latents, the current block, and the optionally prefetched next block.
        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        except RuntimeError:
            return
        if free_bytes < self.cache_reserve_bytes:
            return

        if self._gpu_cache_bytes + size > self.cache_max_bytes:
            if self.cache_policy == "prefix":
                # Keep the prefix already selected in the first pass. Evicting it would
                # cause zero hits on the next cyclic pass.
                return
            while self._gpu_cache and self._gpu_cache_bytes + size > self.cache_max_bytes:
                _old_block, old_tensors = self._gpu_cache.popitem(last=False)
                self._gpu_cache_bytes -= self._root_size_bytes(_old_block)
                del old_tensors

        self._gpu_cache[block] = tensors
        self._gpu_cache_bytes += size
        self.stats["gpu_cache_bytes"] = self._gpu_cache_bytes

    def _pop_cache_block(self, block: str) -> dict[str, torch.Tensor] | None:
        if not self.cache_enabled:
            return None
        with self._lock:
            tensors = self._gpu_cache.pop(block, None)
        if tensors is not None:
            self._gpu_cache_bytes -= self._root_size_bytes(block)
            self.stats["gpu_cache_bytes"] = self._gpu_cache_bytes
        return tensors

    # ---------- 块生命周期 ----------

    def _on_enter_block(self, block: str):
        if block in self._loaded:
            return
        # 记录执行顺序（首次前向自动学习，供后续预取使用）
        if block not in self._order:
            self._order.append(block)
        ptr = self._order.index(block)
        if self._ptr >= 0 and self._order[self._ptr] == self._order[-1] and ptr == 0:
            self.stats["steps"] += 1  # 回绕到第一个块 = 新的一轮前向
        self._ptr = ptr

        tensors = self._take(block)
        self._materialize_root(block, tensors)
        self._active_tensors[block] = tensors
        self._loaded.add(block)
        self.stats["block_calls"] += 1
        self._sample_rss()

        if self.verbose and self.stats["block_calls"] % 25 == 0:
            cache_gib = self.stats["gpu_cache_bytes"] / 1024**3
            suffix = f", cache {cache_gib:.2f} GiB"
            if self.device.type == "cuda":
                try:
                    free_bytes, _total = torch.cuda.mem_get_info(self.device)
                    suffix += f", GPU free {free_bytes / 1024**3:.2f} GiB"
                except RuntimeError:
                    pass
            print(
                f"[stream] step {self.stats['steps'] + 1}, "
                f"block {ptr + 1}/{len(self._order)}, "
                f"cache hits {self.stats['cache_hits']}{suffix}",
                flush=True,
            )

    def _on_exit_block(self, block: str):
        tensors = self._active_tensors.pop(block, None)
        self._free_root(block)
        self._loaded.discard(block)
        self._maybe_cache_block(block, tensors)
        self._maybe_start_prefetch_next()

    # ---------- 加载 / 预取 ----------

    def _load_tensors(self, param_names, count_stats=True):
        tensors = {}
        nbytes = 0
        t0 = time.perf_counter()
        for name in param_names:
            t = self.index.read(name)
            if t.dtype != self.dtype or t.device != self.device:
                t = t.to(device=self.device, dtype=self.dtype)
            tensors[name] = t
            nbytes += self.index.size_bytes(name)
        dt = time.perf_counter() - t0
        if count_stats:
            self.stats["bytes_read"] += nbytes
            self.stats["load_seconds"] += dt
        return tensors

    def _load_block(self, block: str):
        t0 = time.perf_counter()
        tensors = self._load_tensors(self._block_params[block])
        dt = time.perf_counter() - t0
        self.stats["block_table"].setdefault(block, {
            "mb": sum(self.index.size_bytes(n) for n in self._block_params[block]) / 1e6,
            "loads": 0, "seconds": 0.0, "hits": 0,
        })
        self.stats["block_table"][block]["loads"] += 1
        self.stats["block_table"][block]["seconds"] += dt
        return tensors

    def _take(self, block: str):
        # 1) 预取命中？（锁内只摘取 future 引用，join 必须在锁外——否则与
        #    worker 的"加锁存结果"互等，构成死锁）
        fut = None
        with self._lock:
            if self._prefetch is not None:
                fut = self._prefetch
                self._prefetch = None
        if fut is not None:
            fut[1].join()               # 锁外等待预取线程完成
            with self._lock:
                staged = self._staged.pop(fut[0], None)
            if fut[0] == block and staged is not None:
                self.stats["prefetch_hits"] += 1
                self.stats["block_table"].setdefault(block, {
                    "mb": sum(self.index.size_bytes(n) for n in self._block_params[block]) / 1e6,
                    "loads": 0, "seconds": 0.0, "hits": 0,
                })
                self.stats["block_table"][block]["loads"] += 1
                self.stats["block_table"][block]["hits"] += 1
                return staged
            # 顺序变化（异常路径）：staged 丢弃

        # 2) GPU LRU cache hit
        cached = self._pop_cache_block(block)
        if cached is not None:
            self.stats["cache_hits"] += 1
            self.stats["block_table"].setdefault(block, {
                "mb": self._root_size_bytes(block) / 1e6,
                "loads": 0, "seconds": 0.0, "hits": 0,
            })
            self.stats["block_table"][block]["hits"] += 1
            return cached

        # 3) 未命中 → 同步加载
        self.stats["sync_loads"] += 1
        tensors = self._load_block(block)
        if self.verbose and self.stats["block_calls"] < len(self._blocks):
            mb = self.stats["block_table"][block]["mb"]
            print(f"[stream] 首轮同步加载 {block}: {mb:.1f} MB, "
                  f"{self.stats['block_table'][block]['seconds']*1000:.0f} ms", flush=True)
        return tensors

    def _maybe_start_prefetch_next(self):
        if not self.prefetch_enabled:
            return
        with self._lock:
            if self._prefetch is not None:
                return  # 已有预取在途
            if not self._order:
                return
            # 回绕预取：最后一个块之后预取第一个块（下一轮前向）
            nxt = self._order[(self._ptr + 1) % len(self._order)]
            if nxt in self._staged or nxt in self._loaded or nxt in self._gpu_cache:
                return

            def worker(name=nxt):
                t0 = time.perf_counter()
                tensors = self._load_tensors(self._block_params[name])
                with self._lock:
                    self._staged[name] = tensors
                    self.stats["block_table"].setdefault(name, {
                        "mb": sum(self.index.size_bytes(n) for n in self._block_params[name]) / 1e6,
                        "loads": 0, "seconds": 0.0, "hits": 0,
                    })
                    self.stats["block_table"][name]["seconds"] += time.perf_counter() - t0

            th = threading.Thread(target=worker, daemon=False)
            self._prefetch = (nxt, th)
            th.start()

    # ---------- 统计 ----------

    def _sample_rss(self):
        v = read_rss_mb()
        if v is not None:
            self.stats["peak_rss_mb"] = max(self.stats["peak_rss_mb"], v)

    def report(self) -> str:
        s = self.stats
        lines = [
            "=" * 62,
            "流式执行器统计",
            "=" * 62,
            f"流式块数: {len(self._blocks)}  常驻块数: {len(self._residents)}",
            f"块前向次数: {s['block_calls']}（约 {s['steps'] + 1} 轮完整前向）",
            f"预取命中: {s['prefetch_hits']}  同步加载(未命中): {s['sync_loads']}",
            f"磁盘读取总量: {s['bytes_read']/1e9:.2f} GB"
            f"（{s['bytes_read']/max(s['steps'] + 1, 1)/1e9:.2f} GB/轮前向）",
            f"累计加载耗时: {s['load_seconds']:.2f}s",
            f"执行器视角峰值 RSS: {s['peak_rss_mb']:.0f} MB",
            "-" * 62,
            f"{'块':<28}{'源(MB)':>8}{'次数':>6}{'平均ms':>9}{'命中':>6}",
        ]
        for b, t in sorted(s["block_table"].items()):
            avg_ms = (t["seconds"] / t["loads"] * 1000) if t["loads"] else 0
            lines.append(f"{b:<28}{t['mb']:>8.1f}{t['loads']:>6}{avg_ms:>9.0f}{t['hits']:>6}")
        lines.append("=" * 62)
        return "\n".join(lines)
# ---------------------------------------------------------------- 构建入口

def build_streaming_model(model_cls, config: dict, index: SafetensorsIndex,
                          dtype=torch.float32, device: str | torch.device = "cpu",
                          **executor_kwargs):
    """
    通用入口：任意 diffusers ModelMixin 架构 + safetensors 分片 → 流式模型。

    返回 (model, executor)。model 是标准架构对象（可直接传给 Pipeline），
    只是参数平时驻留 meta 设备，由 executor 的钩子按块物化。
    """
    try:
        # 路线 A：meta 设备上直接构建（零初始化内存）
        with torch.device("meta"):
            model = model_cls.from_config(config)
    except Exception:
        # 路线 B：常规构建后立刻搬去 meta（短暂占用初始化内存）
        model = model_cls.from_config(config)
        for p in model.parameters():
            p.data = torch.empty(p.shape, dtype=dtype, device="meta")
    ex = StreamingExecutor(model, index, dtype=dtype, device=device, **executor_kwargs)
    return model, ex


# ---------------------------------------------------------------------------
# Qwen-Image-specific streaming setup
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------- 重映射索引

def te_rename(k: str) -> str:
    """旧检查点键名 -> transformers 5.17 模型键名。"""
    if k.startswith("visual."):
        return "model." + k
    if k.startswith("model."):
        return "model.language_model." + k[len("model."):]
    return k  # lm_head.*

class RemappedSafetensorsIndex(SafetensorsIndex):
    """键名重映射索引：对外呈现模型键名，读取时换回文件里的原始键名。"""
    def __init__(self, files, rename):
        super().__init__(files)
        self._file_key: dict[str, str] = {}
        new_loc, new_meta = {}, {}
        for k in list(self._loc.keys()):
            nk = rename(k)
            if nk in self._file_key:
                raise RuntimeError(f"重命名冲突: {k!r} 与 {self._file_key[nk]!r} -> {nk!r}")
            self._file_key[nk] = k
            new_loc[nk] = self._loc[k]
            new_meta[nk] = self._meta[k]
        self._loc, self._meta = new_loc, new_meta

    def read(self, name, dtype=None):
        t = self._handle(self._loc[name]).get_tensor(self._file_key[name])
        if dtype is not None and t.dtype != dtype:
            t = t.to(dtype)
        return t

def fix_meta_buffers(model, theta: float = 1000000.0,
                     device: str | torch.device = "cpu") -> list[str]:
    """meta 构建后，把 rotary inv_freq 等非持久 buffer 换成真实 CPU 张量。"""
    fixed = []
    for name, buf in list(model.named_buffers()):
        if buf.device.type != "meta":
            continue
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        attr = name.rsplit(".", 1)[-1]
        if "inv_freq" in attr:  # inv_freq / original_inv_freq
            half = int(buf.shape[-1]); dim = half * 2
            new = (
                1.0
                / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            ).to(device=device)
        elif attr == "attention_scaling":
            new = torch.ones((), dtype=torch.float32, device=torch.device(device))
        elif name.startswith("model.visual."):
            print(f"  [warn] 跳过视觉塔 meta buffer（纯文路径不调用）: {name}")
            continue
        else:
            raise RuntimeError(f"未处理的 meta buffer: {name} {tuple(buf.shape)}")
        owner.register_buffer(attr, new, persistent=False)
        fixed.append(name)
    return fixed

# ---------------------------------------------------------------- 流式构建

def fix_qwen_embed_rope(model, device: str | torch.device = "cpu"):
    """meta 构建副作用修复：QwenEmbedRope 在 __init__ 里预计算的
    pos_freqs/neg_freqs 是普通张量属性（非 buffer），meta 下为空壳。
    用模块自身参数在 CPU 上重算（约 8MB，一次性）。"""
    pe = model.pos_embed
    with torch.device("cpu"):
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        pe.pos_freqs = torch.cat([
            pe.rope_params(pos_index, pe.axes_dim[0], pe.theta),
            pe.rope_params(pos_index, pe.axes_dim[1], pe.theta),
            pe.rope_params(pos_index, pe.axes_dim[2], pe.theta),
        ], dim=1)
        pe.neg_freqs = torch.cat([
            pe.rope_params(neg_index, pe.axes_dim[0], pe.theta),
            pe.rope_params(neg_index, pe.axes_dim[1], pe.theta),
            pe.rope_params(neg_index, pe.axes_dim[2], pe.theta),
        ], dim=1)
    pe.pos_freqs = pe.pos_freqs.to(device=device)
    pe.neg_freqs = pe.neg_freqs.to(device=device)
    return f"pos_freqs{tuple(pe.pos_freqs.shape)}={pe.pos_freqs.dtype} + neg_freqs"

def build_transformer_streaming(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    prefetch: bool = True,
    cache_max_bytes: int = 0,
    cache_reserve_bytes: int = 4 << 30,
    cache_policy: str = "prefix",
):
    from diffusers import QwenImageTransformer2DModel
    root = Path(model_dir).expanduser().resolve()
    cfg = json.loads((root / "transformer/config.json").read_text())
    files = sorted(str(p) for p in (root / "transformer").glob("*.safetensors"))
    index = SafetensorsIndex(files)
    tr, ex = build_streaming_model(
        QwenImageTransformer2DModel, cfg, index, dtype=dtype, device=device,
        resident_max_bytes=_STREAM_RESIDENT_MAX, split_max_bytes=_STREAM_SPLIT_MAX, prefetch=prefetch,
        cache_max_bytes=cache_max_bytes,
        cache_reserve_bytes=cache_reserve_bytes,
        cache_policy=cache_policy,
    )
    info = fix_qwen_embed_rope(tr, device=device)
    print(f"[transformer] RoPE 频率表重算: {info}")
    return tr, ex

def build_text_encoder_streaming(
    model_dir: str | Path,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    prefetch: bool = True,
):
    from transformers import AutoConfig, Qwen2_5_VLForConditionalGeneration
    root = Path(model_dir).expanduser().resolve()
    tcfg = AutoConfig.from_pretrained(root / "text_encoder")
    theta = getattr(tcfg, "rope_theta", None)
    if theta is None:
        theta = getattr(getattr(tcfg, "text_config", None), "rope_theta", 1000000.0)
    files = sorted(str(p) for p in (root / "text_encoder").glob("*.safetensors"))
    tindex = RemappedSafetensorsIndex(files, te_rename)
    with torch.device("meta"):
        te = Qwen2_5_VLForConditionalGeneration(tcfg)
    fixed = fix_meta_buffers(te, float(theta), device=device)
    ex = StreamingExecutor(te, tindex, dtype=dtype, device=device,
                           resident_max_bytes=_STREAM_RESIDENT_MAX, split_max_bytes=_STREAM_SPLIT_MAX,
                           prefetch=prefetch)
    print(f"[text_encoder] meta buffer 修复: {len(fixed)} 个 {fixed[:3]}")
    return te, ex


DEFAULT_PROMPT = (
    "A Chinese female college student, around 20 years old, with a very short haircut "
    "that conveys a gentle, artistic vibe. Her hair naturally falls to partially cover "
    "her cheeks, projecting a tomboyish yet charming demeanor. She has cool-toned fair "
    "skin and delicate features, with a slightly shy yet subtly confident expression. "
    "She wears an off-shoulder top, revealing one shoulder, with a well-proportioned "
    "figure. The image is framed as a close-up selfie: she dominates the foreground, "
    "while the background clearly shows her dormitory—a neatly made bed with white "
    "linens on the top bunk, a tidy study desk with organized stationery, and wooden "
    "cabinets and drawers. The photo is captured on a smartphone under soft, even "
    "ambient lighting, with natural tones, high clarity, and a bright, lively "
    "atmosphere full of youthful, everyday energy."
)

def _read_text_file(path: str) -> str:
    """Read a UTF-8 text file, or stdin when path is '-'."""
    if path == "-":
        return sys.stdin.read().strip()
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise SystemExit(f"prompt file does not exist: {file_path}")
    return file_path.read_text(encoding="utf-8").strip()


def _resolve_prompt(
    inline: str | None,
    file_path: str | None,
    default: str,
    option_name: str,
) -> str:
    if inline is not None and file_path is not None:
        raise SystemExit(f"please provide either --{option_name} or --{option_name}-file, not both")
    if file_path is not None:
        text = _read_text_file(file_path)
        if not text:
            raise SystemExit(f"{option_name} file is empty: {file_path}")
        return text
    return inline if inline is not None else default


def _resolve_size(args: argparse.Namespace) -> tuple[int, int]:
    """Resolve output size from explicitly passed dimensions."""
    width, height = args.width, args.height

    if width <= 0 or height <= 0:
        raise SystemExit(f"width and height must be positive, got {width}x{height}")
    if width % 16 != 0 or height % 16 != 0:
        raise SystemExit(
            "width and height must both be divisible by 16 "
            f"(VAE 8x + patch 2), got {width}x{height}"
        )

    return width, height


def _resolve_device_and_dtype(device: str, dtype: str) -> tuple[str, torch.dtype]:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    elif device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but no CUDA device is available")

    if dtype == "auto":
        dtype_name = "bfloat16" if device == "cuda" else "float32"
    else:
        dtype_name = dtype

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    return device, dtype_map[dtype_name]


def _validate_runtime_dependencies() -> None:
    """Fail with an actionable message before diffusers returns a dummy-object error."""
    problems: list[str] = []

    required_versions = {
        "diffusers": "0.40.0",
        "transformers": "4.49.0",
    }
    for package, minimum in required_versions.items():
        try:
            installed = version(package)
            if Version(installed) < Version(minimum):
                problems.append(
                    f"{package} {installed} is too old; Qwen-Image needs {package} >= {minimum}"
                )
        except PackageNotFoundError:
            problems.append(f"{package} is not installed; install {package} >= {minimum}")
        except InvalidVersion as exc:
            problems.append(f"could not parse the installed {package} version: {exc}")

    try:
        # Qwen-Image's text encoder is Qwen2.5-VL. Merely importing this class is cheap;
        # it does not load model weights.
        from transformers import Qwen2_5_VLForConditionalGeneration  # noqa: F401
    except ImportError as exc:
        problems.append(
            "transformers cannot provide Qwen2_5_VLForConditionalGeneration; "
            "upgrade transformers (the locally tested version is 5.17.0): "
            f"{exc}"
        )

    if QwenImagePipeline is None:
        problems.append("this diffusers version does not expose QwenImagePipeline")

    # Diffusers falls back to a DummyObject when an optional backend is unavailable.
    # Generic DiffusionPipeline.from_pretrained then exposes a misleading
    # "'super' object has no attribute '__getattr__'" error, so detect that state here.
    elif QwenImagePipeline.__module__.startswith("diffusers.utils.dummy"):
        from diffusers.utils.import_utils import is_torch_available, is_transformers_available

        unavailable = []
        if not is_torch_available():
            unavailable.append(
                "torch is unavailable to diffusers (check the torch installation and USE_TORCH/USE_TF)"
            )
        if not is_transformers_available():
            unavailable.append("transformers is unavailable to diffusers")
        problems.extend(unavailable or ["QwenImagePipeline could not be imported by diffusers"])

    if problems:
        raise SystemExit(
            "Cannot initialize the Qwen-Image pipeline:\n- " + "\n- ".join(problems)
        )


def _estimated_model_bytes(model_path: Path, dtype: torch.dtype) -> int:
    """Estimate checkpoint bytes from the two sharded indexes and the VAE file."""
    total = 0
    for relative_index in (
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    ):
        index_path = model_path / relative_index
        if index_path.is_file():
            try:
                index = json.loads(index_path.read_text(encoding="utf-8"))
                total += int(index.get("metadata", {}).get("total_size", 0))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass

    vae_path = model_path / "vae/diffusion_pytorch_model.safetensors"
    total += vae_path.stat().st_size if vae_path.is_file() else 0

    # The indexes describe a bf16 checkpoint. Adjust for an explicitly requested fp32 load.
    multiplier = 2 if dtype == torch.float32 else 1
    return total * multiplier


def _resolve_offload_strategy(args: argparse.Namespace, device: str, model_path: Path, dtype: torch.dtype) -> str:
    strategy = args.offload
    if strategy == "auto":
        if device != "cuda":
            return "none"
        gpu_bytes = torch.cuda.get_device_properties(torch.device(device)).total_memory
        # Qwen-Image-2512 is about 57.7GB in bf16. A 22GB card cannot hold even the
        # transformer component, and group offloading still keeps all weights in host
        # RAM. Automatic mode therefore uses true disk streaming on small GPUs.
        strategy = "streaming" if gpu_bytes < 64 * 1024**3 else "none"

    if strategy == "none" and device == "cuda":
        gpu_bytes = torch.cuda.get_device_properties(torch.device(device)).total_memory
        model_bytes = _estimated_model_bytes(model_path, dtype)
        if model_bytes and gpu_bytes < model_bytes + 2 * 1024**3:
            raise SystemExit(
                f"refusing to move the full model to a {gpu_bytes / 1024**3:.2f} GiB GPU: "
                f"the model needs about {model_bytes / 1024**3:.2f} GiB for weights alone. "
                "Use --offload group or generate on a larger GPU."
            )
    return strategy


def _build_streaming_pipeline(
    model_path: Path,
    dtype: torch.dtype,
    device: str,
    prefetch: bool,
    enable_vae_tiling: bool,
    cache_max_bytes: int,
    cache_reserve_bytes: int,
    cache_policy: str,
) -> QwenImagePipeline:
    """Build text_encoder/transformer as meta models and stream one disk block at a time."""
    print("Building text_encoder with disk streaming...")
    text_encoder, text_executor = build_text_encoder_streaming(
        model_dir=model_path,
        dtype=dtype,
        device=device,
        prefetch=prefetch,
    )
    print("Building transformer with disk streaming...")
    if cache_max_bytes > 0:
        print(
            "GPU weight cache requested: "
            f"{cache_max_bytes / 1024**3:.2f} GiB, reserve "
            f"{cache_reserve_bytes / 1024**3:.2f} GiB, policy {cache_policy}"
        )
    transformer, transformer_executor = build_transformer_streaming(
        model_dir=model_path,
        dtype=dtype,
        device=device,
        prefetch=prefetch,
        cache_max_bytes=cache_max_bytes,
        cache_reserve_bytes=cache_reserve_bytes,
        cache_policy=cache_policy,
    )

    # Passing the two large components prevents from_pretrained from loading their
    # 57GB of weights into host RAM. It only loads the small VAE and tokenizer here.
    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        text_encoder=text_encoder,
        transformer=transformer,
        dtype=dtype,
        local_files_only=True,
    )
    if enable_vae_tiling:
        pipe.vae.enable_tiling()
    pipe.vae.to(torch.device(device))

    # Keep executors alive with the pipeline. Their forward hooks also retain them,
    # but this makes ownership explicit and simplifies debugging.
    pipe._qwen_streaming_executors = (text_executor, transformer_executor)
    return pipe


def _pad_prompt_embedding(
    embeds: torch.Tensor,
    mask: torch.Tensor | None,
    target_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad one prompt embedding to target_len and return an explicit 0/1 mask."""
    seq_len = embeds.shape[1]
    if mask is None:
        mask = torch.ones((embeds.shape[0], seq_len), dtype=torch.long, device=embeds.device)
    if seq_len > target_len:
        raise ValueError(f"prompt embedding length {seq_len} exceeds target {target_len}")
    if seq_len < target_len:
        pad_len = target_len - seq_len
        embeds = torch.cat(
            [embeds, embeds.new_zeros((embeds.shape[0], pad_len, embeds.shape[2]))],
            dim=1,
        )
        mask = torch.cat(
            [mask, mask.new_zeros((mask.shape[0], pad_len))],
            dim=1,
        )
    return embeds, mask


@torch.no_grad()
def _generate_with_batched_true_cfg(
    pipe: QwenImagePipeline,
    prompt: str,
    negative_prompt: str,
    true_cfg_scale: float,
    height: int,
    width: int,
    num_inference_steps: int,
    generator: torch.Generator,
):
    """Generate with cond/uncond packed into one batch-2 transformer call.

    Diffusers' stock QwenImagePipeline runs cond and uncond as two separate transformer
    forwards. With disk streaming that means every denoising step reads the 40.9GB
    transformer checkpoint twice. Packing both conditions into one batch halves the
    weight-streaming I/O per step and substantially raises GPU utilization.
    """
    from diffusers.pipelines.qwenimage.pipeline_qwenimage import (
        calculate_shift,
        retrieve_timesteps,
    )

    if true_cfg_scale <= 1.0 or not negative_prompt:
        raise SystemExit("batched true CFG requires a negative prompt and true_cfg_scale > 1")

    device = pipe._execution_device
    # Encode positive and negative prompts together. This also streams the 16.6GB
    # text_encoder checkpoint once instead of twice during startup.
    all_embeds, all_mask = pipe.encode_prompt(
        prompt=[prompt, negative_prompt],
        device=device,
    )
    prompt_embeds, negative_embeds = all_embeds.chunk(2, dim=0)
    if all_mask is None:
        prompt_mask = None
        negative_mask = None
    else:
        prompt_mask = all_mask[:1]
        negative_mask = all_mask[1:2]

    max_prompt_len = max(prompt_embeds.shape[1], negative_embeds.shape[1])
    prompt_embeds, prompt_mask = _pad_prompt_embedding(prompt_embeds, prompt_mask, max_prompt_len)
    negative_embeds, negative_mask = _pad_prompt_embedding(
        negative_embeds,
        negative_mask,
        max_prompt_len,
    )
    batched_embeds = torch.cat([prompt_embeds, negative_embeds], dim=0)
    batched_mask = torch.cat([prompt_mask, negative_mask], dim=0)

    num_channels_latents = pipe.transformer.config.in_channels // 4
    latents = pipe.prepare_latents(
        1,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
    )
    img_shapes = [[
        (
            1,
            height // pipe.vae_scale_factor // 2,
            width // pipe.vae_scale_factor // 2,
        )
    ]] * 2

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        pipe.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    num_warmup_steps = max(
        len(timesteps) - num_inference_steps * pipe.scheduler.order,
        0,
    )

    guidance = None
    if pipe.transformer.config.guidance_embeds:
        raise SystemExit(
            "this script does not support guidance-distilled Qwen-Image checkpoints"
        )

    pipe.scheduler.set_begin_index(0)
    pipe._attention_kwargs = {}
    pipe._current_timestep = None
    pipe._interrupt = False

    print("Using batched true CFG: cond + uncond in one transformer forward per step")
    transformer_executor = getattr(pipe, "_qwen_streaming_executors", (None, None))[1]
    with pipe.progress_bar(total=num_inference_steps) as progress_bar:
        for i, t in enumerate(timesteps):
            step_start = time.perf_counter()
            pipe._current_timestep = t
            timestep = t.expand(2).to(latents.dtype)
            batched_latents = torch.cat([latents, latents], dim=0)

            # One cache context is sufficient for the packed batch. The stock pipeline
            # only uses separate contexts because it performs two separate calls.
            with pipe.transformer.cache_context("cond"):
                batched_prediction = pipe.transformer(
                    hidden_states=batched_latents,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    encoder_hidden_states_mask=batched_mask,
                    encoder_hidden_states=batched_embeds,
                    img_shapes=img_shapes,
                    attention_kwargs=pipe._attention_kwargs,
                    return_dict=False,
                )[0]

            noise_pred, negative_noise_pred = batched_prediction.chunk(2, dim=0)
            combined = negative_noise_pred + true_cfg_scale * (
                noise_pred - negative_noise_pred
            )
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True)
            noise_pred = combined * (cond_norm / combined_norm)

            latents_dtype = latents.dtype
            latents = pipe.scheduler.step(
                noise_pred,
                t,
                latents,
                return_dict=False,
            )[0]
            if latents.dtype != latents_dtype:
                latents = latents.to(latents_dtype)

            step_elapsed = time.perf_counter() - step_start
            if transformer_executor is not None:
                cache_stats = transformer_executor.stats
                print(
                    f"[step {i + 1}/{num_inference_steps}] done in {step_elapsed:.1f}s, "
                    f"cache hits {cache_stats['cache_hits']}, "
                    f"resident {cache_stats['gpu_cache_bytes'] / 1024**3:.2f} GiB",
                    flush=True,
                )
            else:
                print(
                    f"[step {i + 1}/{num_inference_steps}] done in {step_elapsed:.1f}s",
                    flush=True,
                )

            if i == len(timesteps) - 1 or (
                (i + 1) > num_warmup_steps and (i + 1) % pipe.scheduler.order == 0
            ):
                progress_bar.update()

    pipe._current_timestep = None
    latents = pipe._unpack_latents(latents, height, width, pipe.vae_scale_factor)
    latents = latents.to(pipe.vae.dtype)
    latents_mean = (
        torch.tensor(pipe.vae.config.latents_mean)
        .view(1, pipe.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(pipe.vae.config.latents_std).view(
        1,
        pipe.vae.config.z_dim,
        1,
        1,
        1,
    ).to(latents.device, latents.dtype)
    latents = latents / latents_std + latents_mean
    image = pipe.vae.decode(latents, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(image, output_type="pil")[0]


def _prepare_pipeline_device(pipe: QwenImagePipeline, args: argparse.Namespace, strategy: str, device: str) -> QwenImagePipeline:
    if strategy == "streaming":
        return pipe
    if strategy == "group":
        if device != "cuda":
            raise SystemExit("--offload group is only supported with a CUDA execution device")

        from diffusers.hooks import apply_group_offloading

        # Keep only a small number of layers on the GPU at a time. This is required for
        # cards smaller than the 40.9GB transformer component, such as a 22GB GPU.
        onload_device = torch.device(device)
        offload_device = torch.device("cpu")
        for component in (pipe.text_encoder, pipe.transformer):
            apply_group_offloading(
                component,
                onload_device=onload_device,
                offload_device=offload_device,
                offload_type="block_level",
                num_blocks_per_group=args.blocks_per_group,
                use_stream=not args.no_offload_stream,
            )

        # The VAE is only about 254MB in bf16. Tiling is useful for high-resolution output.
        if not args.no_vae_tiling:
            pipe.vae.enable_tiling()
        pipe.vae.to(onload_device)
    elif strategy == "sequential":
        if device != "cuda":
            raise SystemExit("--offload sequential is only supported with a CUDA execution device")
        pipe.enable_sequential_cpu_offload()
        if not args.no_vae_tiling:
            pipe.vae.enable_tiling()
    else:
        if not args.no_vae_tiling:
            pipe.vae.enable_tiling()
        return pipe.to(device)

    return pipe


def _validate_model_directory(model_path: str) -> Path:
    """Validate the local diffusers layout before attempting a partial model load."""
    root = Path(model_path).expanduser().resolve()
    problems: list[str] = []

    if not root.is_dir():
        raise SystemExit(f"model directory does not exist: {root}")

    required_files = {
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "tokenizer/tokenizer_config.json",
        "tokenizer/vocab.json",
        "tokenizer/merges.txt",
        "transformer/config.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
        "vae/config.json",
        "vae/diffusion_pytorch_model.safetensors",
    }
    for relative_path in required_files:
        if not (root / relative_path).is_file():
            problems.append(f"missing required file: {root / relative_path}")

    # Diffusers only looks for a single checkpoint file after it fails to find the
    # sharded index. Validating the index and all referenced shards turns that confusing
    # "no file named diffusion_pytorch_model.bin" error into an explicit diagnosis.
    index_files = {
        "text_encoder/model.safetensors.index.json",
        "transformer/diffusion_pytorch_model.safetensors.index.json",
    }
    for relative_index in index_files:
        index_path = root / relative_index
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"cannot read sharded index {index_path}: {exc}")
            continue

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            problems.append(f"invalid or empty weight_map in {index_path}")
            continue

        for shard_name in sorted(set(weight_map.values())):
            shard_path = index_path.parent / shard_name
            if not shard_path.is_file():
                problems.append(f"missing model shard: {shard_path}")

        expected_size = index.get("metadata", {}).get("total_size")
        if isinstance(expected_size, int):
            actual_size = sum(
                (index_path.parent / shard_name).stat().st_size
                for shard_name in set(weight_map.values())
                if (index_path.parent / shard_name).is_file()
            )
            if actual_size < expected_size:
                problems.append(
                    f"model shards in {index_path.parent} are truncated: "
                    f"expected at least {expected_size} bytes, found {actual_size}"
                )

    if problems:
        raise SystemExit(
            "The local Qwen-Image model directory is incomplete:\n- "
            + "\n- ".join(problems)
        )

    return root


def _normalise_output_path(output: str) -> Path:
    path = Path(output).expanduser()
    if not path.suffix:
        path = path.with_suffix(".png")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp"}:
        raise SystemExit(
            f"unsupported image format {path.suffix!r}; use .png, .jpg/.jpeg, or .webp"
        )
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate an image with Qwen-Image-2512 using CPU disk-streaming.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        default="models/Qwen-Image-2512",
        help="local model directory",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="positive prompt; omit to use the official example prompt",
    )
    parser.add_argument(
        "--prompt-file",
        default=None,
        help="read positive prompt from a UTF-8 file, or '-' for stdin",
    )
    parser.add_argument(
        "--negative-prompt",
        default=None,
        help="negative prompt; enables batched true CFG when provided",
    )
    parser.add_argument(
        "--negative-prompt-file",
        default=None,
        help="read negative prompt from a UTF-8 file, or '-' for stdin",
    )
    parser.add_argument(
        "--true-cfg-scale",
        type=float,
        default=None,
        help="true CFG strength; defaults to 4.0 with a negative prompt, otherwise 1.0",
    )
    parser.add_argument(
        "--width",
        type=int,
        required=True,
        help="image width; must be divisible by 16",
    )
    parser.add_argument(
        "--height",
        type=int,
        required=True,
        help="image height; must be divisible by 16",
    )
    parser.add_argument(
        "--steps",
        "--num-inference-steps",
        dest="steps",
        type=int,
        default=25,
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
        default="output.png",
        help="output image path; PNG/JPG/WEBP suffix is respected",
    )
    parser.add_argument(
        "--no-vae-tiling",
        action="store_true",
        help="disable tiled VAE decoding",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    if args.steps <= 0:
        raise SystemExit(f"--steps must be positive, got {args.steps}")
    if args.width <= 0 or args.height <= 0:
        raise SystemExit("--width and --height must be positive")
    if args.width % 16 or args.height % 16:
        raise SystemExit("--width and --height must be divisible by 16")

    prompt = _resolve_prompt(args.prompt, args.prompt_file, DEFAULT_PROMPT, "prompt")
    if not prompt:
        raise SystemExit("prompt must not be empty")

    negative_prompt = _resolve_prompt(
        args.negative_prompt,
        args.negative_prompt_file,
        "",
        "negative-prompt",
    )
    if args.true_cfg_scale is None:
        true_cfg_scale = 4.0 if negative_prompt else 1.0
    else:
        true_cfg_scale = args.true_cfg_scale
        if not math.isfinite(true_cfg_scale) or true_cfg_scale <= 0:
            raise SystemExit("--true-cfg-scale must be a positive finite number")
    if negative_prompt and true_cfg_scale <= 1.0:
        raise SystemExit("--true-cfg-scale must be greater than 1 when a negative prompt is provided")
    if not negative_prompt and true_cfg_scale != 1.0:
        raise SystemExit("--true-cfg-scale greater than 1 requires --negative-prompt or --negative-prompt-file")

    width, height = _resolve_size(args)
    output_path = _normalise_output_path(args.output)

    seed = args.seed
    if seed == -1:
        seed = random.randint(0, 2**32 - 1)

    model_path = _validate_model_directory(args.model)
    _validate_runtime_dependencies()

    print(f"Loading model: {model_path}")
    print("Device: cpu")
    print("Dtype: float32")
    if negative_prompt:
        print(f"Mode: disk-streaming, batched true CFG, scale {true_cfg_scale:g}")
    else:
        print("Mode: disk-streaming, no true CFG")
    print(f"Size: {width}x{height}")
    print(f"Steps: {args.steps}, seed: {seed}")
    print(f"Output: {output_path}")

    pipe = _build_streaming_pipeline(
        model_path=model_path,
        dtype=torch.float32,
        device="cpu",
        prefetch=True,
        enable_vae_tiling=not args.no_vae_tiling,
        cache_max_bytes=0,
        cache_reserve_bytes=0,
        cache_policy="prefix",
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if negative_prompt:
        image = _generate_with_batched_true_cfg(
            pipe=pipe,
            prompt=prompt,
            negative_prompt=negative_prompt,
            true_cfg_scale=true_cfg_scale,
            height=height,
            width=width,
            num_inference_steps=args.steps,
            generator=generator,
        )
    else:
        image = pipe(
            prompt=prompt,
            negative_prompt=None,
            width=width,
            height=height,
            num_inference_steps=args.steps,
            true_cfg_scale=1.0,
            generator=generator,
        ).images[0]
    image.save(output_path)
    print(f"Saved image to: {output_path}")


if __name__ == "__main__":
    main()
