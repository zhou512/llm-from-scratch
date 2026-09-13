"""实测 mlp_test.py 结构在 CPU 和 GPU 上的训练速度差。

用法（在 projects/02-makemore 目录下）：
    python bench_device.py
"""

import time

import torch
from torch.nn import functional as F

# Karpathy 第三讲的配置
BLOCK = 3
EMB = 10
HIDDEN = 200
VOCAB = 27
BATCH = 64


def build_data():
    names = open("names.txt", "r", encoding="utf-8").read().splitlines()
    chars = sorted(set("".join(names)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    X, Y = [], []
    for n in names:
        seq = [0] + [stoi[c] for c in n] + [0]
        ctx = BLOCK * [0]
        for ix in seq:
            X.append(list(ctx))
            Y.append(ix)
            ctx = ctx[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def bench(dev, steps=2000, batch=BATCH):
    torch.manual_seed(2147483647)
    X, Y = build_data()
    X, Y = X.to(dev), Y.to(dev)

    g = torch.Generator(device="cpu").manual_seed(2147483647)
    C = torch.randn((VOCAB, EMB), generator=g).to(dev)
    W1 = torch.randn((EMB * BLOCK, HIDDEN), generator=g).to(dev)
    B1 = torch.randn(HIDDEN, generator=g).to(dev)
    W2 = torch.randn((HIDDEN, VOCAB), generator=g).to(dev)
    B2 = torch.randn(VOCAB, generator=g).to(dev)
    params = [C, W1, B1, W2, B2]
    for p in params:
        p.requires_grad = True

    gidx = torch.Generator(device="cpu").manual_seed(1)

    # 预热
    for _ in range(20):
        ix = torch.randint(0, X.shape[0], (batch,), generator=gidx).to(dev)
        loss = F.cross_entropy(
            torch.tanh(C[X[ix]].view(-1, EMB * BLOCK) @ W1 + B1) @ W2 + B2, Y[ix]
        )
        loss.backward()
        for p in params:
            p.grad = None

    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        ix = torch.randint(0, X.shape[0], (batch,), generator=gidx).to(dev)
        emb = C[X[ix]]
        act = torch.tanh(emb.view(-1, EMB * BLOCK) @ W1 + B1)
        logits = act @ W2 + B2
        loss = F.cross_entropy(logits, Y[ix])
        loss.backward()
        for p in params:
            p.data -= 0.1 * p.grad
            p.grad = None
    if dev == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0
    return dt, dt / steps * 1000


def main():
    print(f"数据集规模: {len(open('names.txt', encoding='utf-8').read().splitlines())} 个名字")
    print(f"配置: block={BLOCK}, emb={EMB}, hidden={HIDDEN}, batch={BATCH}")
    print()
    print(f"{'设备':10} {'2000 步耗时':>14} {'每步':>10} {'10万步预计':>14}")
    print("-" * 54)

    results = {}
    for dev in ["cpu", "cuda"]:
        if dev == "cuda" and not torch.cuda.is_available():
            print(f"{dev:10} {'CUDA 不可用':>14}")
            continue
        if dev == "cuda":
            torch.cuda.synchronize()
        dt, ms = bench(dev)
        est = ms * 100000 / 1000 / 60
        results[dev] = (dt, ms, est)
        print(f"{dev:10} {dt:>13.2f}s {ms:>9.2f}ms {est:>12.1f}分")

    if "cpu" in results and "cuda" in results:
        speedup = results["cpu"][1] / results["cuda"][1]
        print()
        print(f"★ GPU 比 CPU 快 {speedup:.1f} 倍")
        print(f"  10 万步: {results['cpu'][2]:.1f} 分钟 → {results['cuda'][2]:.1f} 分钟")

    print()
    print("设备名称:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "无 GPU")


if __name__ == "__main__":
    main()
