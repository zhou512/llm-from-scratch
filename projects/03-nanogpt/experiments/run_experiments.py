"""超参数对比实验：跑多组配置，产出可放进简历的对比表。

为什么要单独写这个脚本：
nanoGPT 的 train.py 是「一次训一个模型」，要对比不同配置得手动改参数、
重复跑、手动记结果。这个脚本把「跑实验 + 记录结果」自动化，
最后产出一张 CSV + Markdown 表。

用法（在 projects/02-nanogpt 目录下）：

    # 跑默认的一组实验（英文语料，约 5-10 分钟）
    .venv\\Scripts\\python.exe experiments\\run_experiments.py

    # 只跑指定实验
    .venv\\Scripts\\python.exe experiments\\run_experiments.py --only baseline,n_layer=3

    # 用中文语料
    .venv\\Scripts\\python.exe experiments\\run_experiments.py --dataset zh_char

    # 快速冒烟（每组只跑 20 步，验证脚本本身）
    .venv\\Scripts\\python.exe experiments\\run_experiments.py --max-iters 20 --only baseline

产出：
    outputs/experiments.csv       原始数据
    outputs/experiments.md        Markdown 表格（直接贴进 README）
"""

import argparse
import csv
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "refs", "nanoGPT"))

from model import GPT, GPTConfig  # noqa: E402

OUT_DIR = os.path.join(PROJ, "outputs")


# ---------------------------------------------------------------- 实验定义

# 每个实验 = (名字, 说明, 覆盖的配置)
# 基线是官方 char-level 配置，其余每组只改一个变量（控制变量法）
BASE = dict(
    block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2,
)

EXPERIMENTS = [
    ("baseline", "官方 char-level 配置", {}),
    ("n_layer=3", "只把层数减半", dict(n_layer=3)),
    ("n_layer=12", "只把层数翻倍", dict(n_layer=12)),
    ("n_embd=192", "只把嵌入维度减半", dict(n_embd=192, n_head=3)),
    ("n_embd=768", "只把嵌入维度翻倍", dict(n_embd=768, n_head=12)),
    ("block_size=128", "只把上下文长度减半", dict(block_size=128)),
    ("no_dropout", "关掉 dropout", dict(dropout=0.0)),
    ("dropout=0.4", "加强 dropout", dict(dropout=0.4)),
]


# ---------------------------------------------------------------- 数据


class BinDataset:
    """按 nanoGPT 的格式读 train.bin / val.bin / meta.pkl。"""

    def __init__(self, dataset, data_root):
        self.dir = os.path.join(data_root, dataset)
        train_path = os.path.join(self.dir, "train.bin")
        val_path = os.path.join(self.dir, "val.bin")
        meta_path = os.path.join(self.dir, "meta.pkl")

        if not os.path.exists(train_path):
            raise SystemExit(
                f"找不到 {train_path}\n"
                f"先准备数据：python data\\prepare.py --out-dir data\\{dataset}"
            )

        self.train = np.memmap(train_path, dtype=np.uint16, mode="r")
        self.val = np.memmap(val_path, dtype=np.uint16, mode="r")

        if os.path.exists(meta_path):
            import pickle
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            self.vocab_size = meta["vocab_size"]
        else:
            self.vocab_size = 50304
            print(f"⚠️  没有 {meta_path}，假定 vocab_size={self.vocab_size}")

    def get_batch(self, split, block_size, batch_size, device):
        data = self.train if split == "train" else self.val
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
        if device == "cuda":
            x = x.pin_memory().to(device, non_blocking=True)
            y = y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y


@torch.no_grad()
def estimate_loss(model, ds, block_size, batch_size, device, eval_iters=30):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = ds.get_batch(split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


# ---------------------------------------------------------------- 单次实验


def run_one(name, desc, overrides, ds, args):
    cfg_kwargs = dict(BASE)
    cfg_kwargs.update(overrides)
    cfg_kwargs["vocab_size"] = ds.vocab_size

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    torch.manual_seed(1337)
    config = GPTConfig(**cfg_kwargs)
    model = GPT(config).to(device)

    n_params = model.get_num_params()

    # 自动调 batch_size 防止 OOM
    batch_size = args.batch_size
    while batch_size >= 1:
        try:
            X, Y = ds.get_batch("train", config.block_size, batch_size, device)
            _, loss = model(X, Y)
            loss.backward()
            model.zero_grad(set_to_none=True)
            if device == "cuda":
                torch.cuda.synchronize()
            break
        except torch.cuda.OutOfMemoryError:
            batch_size //= 2
            torch.cuda.empty_cache()
            if batch_size < 1:
                return dict(name=name, desc=desc, status="OOM", **overrides)

    optimizer = model.configure_optimizers(0.1, args.lr, (0.9, 0.99), device)

    # 计时：先跑几步热身，再正式计时
    t0 = time.time()
    model.train()
    for it in range(args.max_iters):
        X, Y = ds.get_batch("train", config.block_size, batch_size, device)
        _, loss = model(X, Y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    losses = estimate_loss(model, ds, config.block_size, batch_size, device, args.eval_iters)
    mfu = model.estimate_mfu(batch_size * 1, dt / args.max_iters) if device == "cuda" else 0.0
    peak = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else 0.0

    row = dict(
        name=name, desc=desc, status="OK",
        n_params_M=round(n_params / 1e6, 2),
        train_loss=round(losses["train"], 4),
        val_loss=round(losses["val"], 4),
        ms_per_iter=round(dt / args.max_iters * 1000, 1),
        peak_mem_GB=round(peak, 2),
        mfu_pct=round(mfu * 100, 2),
        batch_size=batch_size,
        **{k: v for k, v in cfg_kwargs.items() if k != "vocab_size"},
    )

    del model, optimizer
    if device == "cuda":
        torch.cuda.empty_cache()
    return row


# ---------------------------------------------------------------- 主流程


def main():
    parser = argparse.ArgumentParser(description="nanoGPT 超参对比实验")
    parser.add_argument("--dataset", default="shakespeare_char")
    parser.add_argument("--data-root", default=os.path.join(PROJ, "data"))
    parser.add_argument("--max-iters", type=int, default=300,
                        help="每组训练步数（默认 300，够看出趋势）")
    parser.add_argument("--eval-iters", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--only", default="",
                        help="只跑指定实验，逗号分隔，如 baseline,n_layer=3")
    parser.add_argument("--tag", default="", help="输出文件名后缀")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"设备: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"数据集: {args.dataset}")

    ds = BinDataset(args.dataset, args.data_root)
    print(f"vocab_size: {ds.vocab_size}")
    print(f"训练 token: {len(ds.train):,}  验证 token: {len(ds.val):,}")
    print(f"每组步数: {args.max_iters}")
    print()

    selected = EXPERIMENTS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        selected = [e for e in EXPERIMENTS if e[0] in wanted]
        if not selected:
            raise SystemExit(f"--only 没匹配到实验。可选项: {[e[0] for e in EXPERIMENTS]}")

    rows = []
    for i, (name, desc, overrides) in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {name} —— {desc}")
        t0 = time.time()
        row = run_one(name, desc, overrides, ds, args)
        row["elapsed_s"] = round(time.time() - t0, 1)
        rows.append(row)
        if row.get("status") == "OK":
            print(f"     参数 {row['n_params_M']}M | train {row['train_loss']} | "
                  f"val {row['val_loss']} | {row['ms_per_iter']}ms/iter | "
                  f"{row['peak_mem_GB']}GB | MFU {row['mfu_pct']}%")
        else:
            print(f"     失败: {row.get('status')}")
        print()

    # 写 CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    csv_path = os.path.join(OUT_DIR, f"experiments_{args.dataset}{suffix}.csv")
    fieldnames = sorted({k for r in rows for k in r})
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # 写 Markdown
    md_path = os.path.join(OUT_DIR, f"experiments_{args.dataset}{suffix}.md")
    cols = ["name", "desc", "n_params_M", "train_loss", "val_loss",
            "ms_per_iter", "peak_mem_GB", "mfu_pct"]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# nanoGPT 超参对比实验（{args.dataset}）\n\n")
        f.write(f"- 设备：{torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'}\n")
        f.write(f"- 每组训练 {args.max_iters} 步，batch_size 自动调整（上限 {args.batch_size}）\n")
        f.write(f"- 控制变量法：除标注的变量外，其余配置与 baseline 相同\n\n")
        f.write("| 实验 | 说明 | 参数量(M) | train loss | val loss | ms/iter | 峰值显存(GB) | MFU(%) |\n")
        f.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            if r.get("status") != "OK":
                f.write(f"| {r['name']} | {r['desc']} | - | - | - | - | - | {r['status']} |\n")
                continue
            f.write(f"| {r['name']} | {r['desc']} | {r['n_params_M']} | {r['train_loss']} | "
                    f"{r['val_loss']} | {r['ms_per_iter']} | {r['peak_mem_GB']} | {r['mfu_pct']} |\n")
        f.write("\n## 观察\n\n（待填：你从这张表里看出了什么规律？）\n")

    print(f"CSV:      {csv_path}")
    print(f"Markdown: {md_path}")


if __name__ == "__main__":
    main()
