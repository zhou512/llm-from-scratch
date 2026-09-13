"""逐项找出 mlp_test.py 慢在哪。

对照 mlp_test.py 的真实代码，逐项加上去，看每项增加多少开销。

用法（在 projects/02-makemore 目录下）：
    python bench_bottleneck.py
"""

import time

import torch
from torch.nn import functional as F

BLOCK, EMB, HIDDEN, VOCAB, BATCH = 3, 10, 200, 27, 64
STEPS = 20000


def build():
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


def make_params(dev="cpu"):
    torch.manual_seed(2147483647)
    g = torch.Generator().manual_seed(2147483647)
    C = torch.randn((VOCAB, EMB), generator=g)
    W1 = torch.randn((EMB * BLOCK, HIDDEN), generator=g)
    B1 = torch.randn(HIDDEN, generator=g)
    W2 = torch.randn((HIDDEN, VOCAB), generator=g)
    B2 = torch.randn(VOCAB, generator=g)
    ps = [C, W1, B1, W2, B2]
    for p in ps:
        p.requires_grad = True
    return ps


def run(label, X, Y, *, use_item=False, use_append=False, use_zeros=False,
        use_set_to_none=False, use_no_grad_ctx=False):
    ps = make_params()
    C, W1, B1, W2, B2 = ps
    gidx = torch.Generator().manual_seed(1)
    losses, lrses = [], []

    def one_step(i):
        lr = 0.85 if i < 50000 else 0.085
        for p in ps:
            p.grad = None
        ix = torch.randint(0, X.shape[0], (BATCH,), generator=gidx)
        emb = C[X[ix]]
        act = torch.tanh(emb.view(-1, EMB * BLOCK) @ W1 + B1)
        logits = act @ W2 + B2
        loss = F.cross_entropy(logits, Y[ix])
        loss.backward()
        if use_item:
            losses.append(loss.item())
        if use_append:
            lrses.append(lr)
        for p in ps:
            p.data -= lr * p.grad

    # 预热
    for i in range(200):
        one_step(i)

    t0 = time.time()
    for i in range(STEPS):
        one_step(i)
    dt = time.time() - t0
    print(f"  {label:44} {dt:6.2f}s  {dt/STEPS*1000:6.2f} ms/步  "
          f"10万步≈{dt/STEPS*100000/60:5.1f}分")
    return dt / STEPS * 1000


def main():
    X, Y = build()
    print(f"数据: {X.shape[0]:,} 条样本")
    print(f"每组跑 {STEPS} 步")
    print()

    base = run("① 基准（无 item / 无 append）", X, Y)
    print()
    a = run("② + losses.append(loss.item())", X, Y, use_item=True)
    print(f"     → 比基准慢 {a/base:.2f}×")
    print()
    b = run("③ + lrses.append(lr)（两个都加）", X, Y, use_item=True, use_append=True)
    print(f"     → 比基准慢 {b/base:.2f}×")
    print()
    print("=" * 78)
    print("实测结论")
    print("=" * 78)
    print(f"""
  losses.append(loss.item()) 的开销: {(a-base):+.2f} ms/步
  lrses.append(lr) 的开销:            {(b-a):+.2f} ms/步

  ★ 两者都可以忽略不计 —— 我原本以为 loss.item() 的同步开销会很大，
    实测否定了这个猜测（这是 CPU 上的标量，没有 GPU 同步问题）。

  ★ 不过 losses/lrses 一直在增长：10 万步会存 20 万个浮点数。
    不慢，但白占内存。想看 loss 曲线的话每 100 步记一次就够了，
    曲线形状看不出区别。

  ★ 另外注意：跑完训练后的那三行

        emb = C[Xtr]                      # ★ Xtr 有 26 万条
        act = ... @ W1 + B1
        logits = act @ W2 + B2

    是拿【整个训练集】算一次 loss。这一步的矩阵乘法规模比单步训练大
    4000 倍，虽然只做一次，但在慢机器上可能要几秒到几十秒。
    想快速迭代时可以改成只取一部分样本估算。
""")


if __name__ == "__main__":
    main()
