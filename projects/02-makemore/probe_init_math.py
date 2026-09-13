"""查清楚：为什么未缩放的初始化，初始 loss 会高达 24.69？

关键线索：W1 是 (30, 200)，标准正态初始化，fan-in = 30。
层层推导：
    emb  ~ 标准正态，每个分量方差 1，30 个分量拼起来 x 的标准差 = sqrt(30) ≈ 5.48
    W1   ~ 标准正态，每个分量标准差 1
    x @ W1 的每个分量 = 30 个 (x_i * w_i) 之和
        方差 = 30 * Var(x_i) * Var(w_i) = 30 * 1 * 1 = 30
        标准差 ≈ 5.48
    tanh(±5.48) ≈ ±1  →  几乎全部饱和！

饱和的后果：tanh 输出几乎恒定在 ±1（不随输入变化）
             → 梯度被压平 → 但 W2 那一层还能学
             → 初始 logits 幅度可能很大 → 初始 loss 很高

用法：
    python probe_init_math.py
"""

import torch
from torch.nn import functional as F

BLOCK, EMB, HIDDEN, VOCAB = 3, 10, 200, 27


def main():
    names = open("names.txt", encoding="utf-8").read().splitlines()
    chars = sorted(set("".join(names)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    X = []
    for n in names[:2000]:
        seq = [0] + [stoi[c] for c in n] + [0]
        ctx = BLOCK * [0]
        for ix in seq:
            X.append(list(ctx))
            ctx = ctx[1:] + [ix]
    X = torch.tensor(X)

    print("=" * 72)
    print("逐层统计（2000 个名字，约 18000 条样本）")
    print("=" * 72)

    for label, scale in [("未缩放", 1.0), ("×0.2（你的）", 0.2)]:
        g = torch.Generator().manual_seed(2147483647)
        C = torch.randn((VOCAB, EMB), generator=g)
        W1 = torch.randn((EMB * BLOCK, HIDDEN), generator=g) * scale
        B1 = torch.randn(HIDDEN, generator=g) * 0.01
        print()
        print(f"--- {label} ---")
        with torch.no_grad():
            emb = C[X]
            print(f"  emb         形状 {tuple(emb.shape)}  标准差 {emb.std():.3f}")
            flat = emb.view(-1, EMB * BLOCK)
            print(f"  展平后      形状 {tuple(flat.shape)}  标准差 {flat.std():.3f}")
            pre = flat @ W1 + B1
            print(f"  线性输出    标准差 {pre.std():.3f}  绝对均值 {pre.abs().mean():.3f}")
            h = torch.tanh(pre)
            sat = (h.abs() > 0.97).float().mean().item()
            print(f"  tanh 输出   标准差 {h.std():.3f}  ★ 饱和比例 {sat*100:.1f}%")

    print()
    print("=" * 72)
    print("理论推导：为什么未缩放会饱和")
    print("=" * 72)
    print(f"""
  emb 的每个分量 ~ N(0, 1)，标准差 = 1
  一层拼 {EMB * BLOCK} 个分量 ⇒ 展平后标准差 = sqrt({EMB * BLOCK}) = {(EMB*BLOCK)**0.5:.2f}

  W1 每个分量 ~ N(0, 1)
  x @ W1 的每个分量是 {EMB * BLOCK} 项乘积之和
      Var = {EMB * BLOCK} × Var(x) × Var(w) = {EMB * BLOCK} × 1 × 1 = {EMB * BLOCK}
      标准差 = sqrt({EMB * BLOCK}) ≈ {(EMB*BLOCK)**0.5:.2f}

  tanh(±{(EMB*BLOCK)**0.5:.2f}) ≈ ±1  →  【几乎全部饱和】

  ★ 正确做法：让线性输出落在 tanh 的线性区（|pre| ≲ 2）
    就要把 W1 的标准差缩到约 1/sqrt({EMB * BLOCK}) = {1/(EMB*BLOCK)**0.5:.3f}
    Karpathy 用的是 5/3/sqrt(fan_in) = {5/3/(EMB*BLOCK)**0.5:.3f}
    （5/3 是因为 x 本身不是严格的标准正态，实测方差大约是 1/(5/3)²）
""")

    # 用不同 seed 看初始 loss 的波动范围
    print("=" * 72)
    print("不同随机种子下的【初始 loss】：说明初始化有多不稳定")
    print("=" * 72)
    print(f"{'seed':>8} {'未缩放':>12} {'×0.2':>12} {'Karpathy':>12}")
    print("-" * 72)
    Y = torch.tensor([stoi[c] if False else 0 for c in range(X.shape[0])])
    for seed in [1, 2, 3, 42, 1337, 2147483647]:
        g = torch.Generator().manual_seed(seed)
        C = torch.randn((VOCAB, EMB), generator=g)
        W1 = torch.randn((EMB * BLOCK, HIDDEN), generator=g)
        B1 = torch.randn(HIDDEN, generator=g) * 0.01
        W2 = torch.randn((HIDDEN, VOCAB), generator=g)
        B2 = torch.randn(VOCAB, generator=g)
        with torch.no_grad():
            h = torch.tanh(C[X].view(-1, EMB * BLOCK) @ W1 + B1)
            lg = h @ W2 + B2
            # 用 logits 的绝对均值当作"输出幅度"指标
            mag_raw = lg.abs().mean().item()

        def scaled(s1):
            g = torch.Generator().manual_seed(seed)
            C2 = torch.randn((VOCAB, EMB), generator=g)
            W1b = torch.randn((EMB * BLOCK, HIDDEN), generator=g) * s1
            B1b = torch.randn(HIDDEN, generator=g) * 0.01
            W2b = torch.randn((HIDDEN, VOCAB), generator=g) * 0.1
            B2b = torch.randn(VOCAB, generator=g) * 0.01
            with torch.no_grad():
                h2 = torch.tanh(C2[X].view(-1, EMB * BLOCK) @ W1b + B1b)
                return (h2 @ W2b + B2b).abs().mean().item()

        g = torch.Generator().manual_seed(seed)
        C3 = torch.randn((VOCAB, EMB), generator=g)
        W1c = torch.randn((EMB * BLOCK, HIDDEN), generator=g) * (5 / 3) / (EMB * BLOCK) ** 0.5
        B1c = torch.randn(HIDDEN, generator=g) * 0.01
        W2c = torch.randn((HIDDEN, VOCAB), generator=g) * 0.01
        B2c = torch.zeros(VOCAB)
        with torch.no_grad():
            h3 = torch.tanh(C3[X].view(-1, EMB * BLOCK) @ W1c + B1c)
            mag_k = (h3 @ W2c + B2c).abs().mean().item()

        print(f"{seed:>8} {mag_raw:>12.3f} {scaled(0.2):>12.3f} {mag_k:>12.3f}")

    print()
    print("（表中是 logits 的绝对均值：越小越接近均匀分布，")
    print("  越大说明初始化时模型已经很'自信'地给出极端预测 = 不健康）")


if __name__ == "__main__":
    main()
