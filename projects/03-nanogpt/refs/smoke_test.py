"""nanoGPT 模型冒烟测试：确认架构能在你的 GPU 上跑起来。

这个脚本**不训练**，只做三件事：
1. 用不同规模的配置实例化 GPT，数参数量
2. 跑一次前向 + 反向，测显存占用和单步耗时
3. 估算完整训练要多久

用法（在 projects/02-nanogpt 目录下）：
    .venv\\Scripts\\python.exe refs\\smoke_test.py

为什么先做这个：
nanoGPT 的 model.py 里 `from nanoGPT...` 之类的 import 需要调整，而且
不同配置的显存需求差很多。先确认「能跑」，再谈「训练」。
"""

import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "nanoGPT"))

from model import GPT, GPTConfig  # noqa: E402


# (名称, 说明, GPTConfig 参数)
CONFIGS = [
    ("tiny", "最小配置，只为验证代码能跑",
     dict(block_size=64, vocab_size=65, n_layer=2, n_head=2, n_embd=64, dropout=0.0, bias=False)),
    ("shakespeare-char", "官方 char-level 配置（词表 65）",
     dict(block_size=256, vocab_size=65, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=True)),
    ("shakespeare-char-zh", "中文版（词表 9214）",
     dict(block_size=256, vocab_size=9214, n_layer=6, n_head=6, n_embd=384, dropout=0.2, bias=True)),
]


def human(n):
    for unit in ("", "K", "M", "B"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000
    return f"{n:.1f}T"


def main():
    if not torch.cuda.is_available():
        print("⚠️  CUDA 不可用，会用 CPU 跑（很慢）")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}, 显存 {props.total_memory/1024**3:.1f} GB")
    print()

    results = []
    for name, desc, cfg_kwargs in CONFIGS:
        print(f"{'='*62}")
        print(f"[{name}] {desc}")
        cfg = GPTConfig(**cfg_kwargs)
        try:
            model = GPT(cfg).to(device)
        except Exception as e:
            print(f"  ❌ 实例化失败: {type(e).__name__}: {e}")
            results.append((name, None, None, str(e)))
            continue

        n_params = model.get_num_params() if hasattr(model, "get_num_params") else sum(p.numel() for p in model.parameters())
        print(f"  参数量: {human(n_params)}")

        # 造一个 batch
        batch_size = 8 if name != "tiny" else 16
        x = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)
        y = torch.randint(0, cfg.vocab_size, (batch_size, cfg.block_size), device=device)

        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        try:
            t0 = time.time()
            _, loss = model(x, y)
            loss.backward()
            if device == "cuda":
                torch.cuda.synchronize()
            dt = time.time() - t0
        except torch.cuda.OutOfMemoryError:
            print(f"  ❌ 显存不足（OOM）：batch_size={batch_size}, block_size={cfg.block_size}")
            results.append((name, n_params, None, "OOM"))
            torch.cuda.empty_cache()
            continue
        except Exception as e:
            print(f"  ❌ 前向/反向失败: {type(e).__name__}: {e}")
            results.append((name, n_params, None, str(e)))
            continue

        if device == "cuda":
            peak = torch.cuda.max_memory_allocated() / 1024**3
            print(f"  单步耗时: {dt*1000:.0f} ms")
            print(f"  峰值显存: {peak:.2f} GB / {props.total_memory/1024**3:.1f} GB")
            print(f"  loss: {loss.item():.4f}  (随机初始化，约等于 ln(vocab)={torch.log(torch.tensor(float(cfg.vocab_size))).item():.4f})")
            results.append((name, n_params, peak, None))
        else:
            print(f"  单步耗时: {dt*1000:.0f} ms")
            results.append((name, n_params, 0.0, None))

        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    print()
    print(f"{'='*62}")
    print("汇总")
    print(f"{'配置':22s} {'参数量':>10s} {'峰值显存':>10s}  状态")
    for name, n_params, peak, err in results:
        if err:
            print(f"{name:22s} {'-':>10s} {'-':>10s}  ❌ {err[:40]}")
        else:
            print(f"{name:22s} {human(n_params):>10s} {peak:>9.2f}G  ✅")


if __name__ == "__main__":
    main()
