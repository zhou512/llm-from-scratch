"""对比两种「特殊标记」设计，解释 mlp_test.py 为什么生成空名字。

- 方案 A：'.' 兼作【填充】【开头】【结尾】三个角色（mlp_test.py 的做法）
- 方案 D：<P> 填充 / <S> 开头 / <E> 结尾，三者分开

用法（在 projects/02-makemore 目录下）：
    python diagnose_token_design.py

依赖：torch、names.txt。有 GPU 会用 GPU（约 2 分钟）。
"""

import random
from collections import Counter

import torch
from torch.nn import functional as F


# ---------------------------------------------------------------- 工具

def load_names(path="names.txt"):
    return open(path, "r", encoding="utf-8").read().splitlines()


def train(Xtr, Ytr, vocab_size, block_size=3, emb=10, hidden=200,
          steps=30000, batch=64, lr0=0.85, decay_at=15000, lr1=0.085):
    """和 mlp_test.py 同结构的 MLP，但初始化按 Karpathy 第三讲的做法缩放。"""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.randint 生成索引要求 generator 在 CPU 上；权重初始化用对应设备
    g_cpu = torch.Generator(device="cpu").manual_seed(2147483647)
    g = torch.Generator(device=dev).manual_seed(2147483647)

    Xtr, Ytr = Xtr.to(dev), Ytr.to(dev)
    C = torch.randn((vocab_size, emb), generator=g, device=dev)
    W1 = torch.randn((emb * block_size, hidden), generator=g, device=dev) * (5 / 3) / (emb * block_size) ** 0.5
    B1 = torch.randn(hidden, generator=g, device=dev) * 0.1
    W2 = torch.randn((hidden, vocab_size), generator=g, device=dev) * 0.1
    B2 = torch.randn(vocab_size, generator=g, device=dev) * 0.01
    params = [C, W1, B1, W2, B2]
    for p in params:
        p.requires_grad = True

    for i in range(steps):
        lr = lr0 if i < decay_at else lr1
        for p in params:
            p.grad = None
        ix = torch.randint(0, Xtr.shape[0], (batch,), generator=g_cpu).to(dev)
        emb_v = C[Xtr[ix]]
        act = torch.tanh(emb_v.view(-1, emb * block_size) @ W1 + B1)
        logits = act @ W2 + B2
        loss = F.cross_entropy(logits, Ytr[ix])
        loss.backward()
        for p in params:
            p.data -= lr * p.grad
    return C, W1, B1, W2, B2


# ---------------------------------------------------------------- 数据构造

def build_scheme(names, pad_tok="<P>", start_tok="<S>", end_tok="<E>"):
    """把名字拼成一个长序列，用不同的特殊标记区分填充/开头/结尾。"""
    letters = sorted(set("".join(names)))
    vocab = [pad_tok, start_tok, end_tok] + letters
    vocab = list(dict.fromkeys(vocab))
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, vocab


def build_dataset_v2(names, stoi, pad_tok, start_tok, end_tok, block_size=3):
    """每个名字独立成一串：<S> name <E>，上下文用 <P> 填充。"""
    X, Y = [], []
    P, S, E = stoi[pad_tok], stoi[start_tok], stoi[end_tok]
    for n in names:
        seq = [S] + [stoi[c] for c in n] + [E]
        context = block_size * [P]        # ★ 用 <P> 填充，而不是 <S>
        for ix in seq:
            X.append(list(context))
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


def build_scheme_A(names):
    """原方案：'.' 兼作填充、开头、结尾。"""
    chars = sorted(set("".join(names)))
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi["."] = 0
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos, list(itos.values())


def build_dataset_A(names, stoi, block_size=3):
    X, Y = [], []
    for n in names:
        seq = [stoi["."]] + [stoi[c] for c in n] + [stoi["."]]
        context = block_size * [stoi["."]]
        for ix in seq:
            X.append(list(context))
            Y.append(ix)
            context = context[1:] + [ix]
    return torch.tensor(X), torch.tensor(Y)


# ---------------------------------------------------------------- 采样

@torch.no_grad()
def generate(C, W1, B1, W2, B2, stoi, itos, pad_tok, start_tok, end_tok,
             n=200, block_size=3, emb=10, seed=11468, max_len=30):
    dev = C.device
    g = torch.Generator(device=dev).manual_seed(seed)
    P, S, E = stoi[pad_tok], stoi[start_tok], stoi[end_tok]
    out = []
    for _ in range(n):
        context = block_size * [P]
        chars = []
        for _step in range(max_len):
            emb_v = C[torch.tensor(context, device=dev)]
            act = torch.tanh(emb_v.view(1, -1) @ W1 + B1)
            probs = F.softmax(act @ W2 + B2, dim=1)
            ix = torch.multinomial(probs, 1, generator=g).item()
            context = context[1:] + [ix]
            if ix == E:
                break
            if ix in (P, S):          # 理论上不该出现，出现了就跳过
                continue
            chars.append(itos[ix])
        out.append("".join(chars))
    return out


@torch.no_grad()
def probe_initial(C, W1, B1, W2, B2, stoi, pad_tok, end_tok, block_size=3, emb=10):
    dev = C.device
    ctx = torch.tensor([stoi[pad_tok]] * block_size, device=dev)
    emb_v = C[ctx]
    act = torch.tanh(emb_v.view(1, -1) @ W1 + B1)
    probs = F.softmax(act @ W2 + B2, dim=1)[0]
    return probs.cpu()


def main():
    names = load_names()
    random.seed(42)
    random.shuffle(names)
    n1, n2 = int(len(names) * 0.8), int(len(names) * 0.9)
    BLOCK = 3

    results = {}

    # ---------------- 方案 A ----------------
    print("=" * 72)
    print("方案 A：'.' 兼作填充/开头/结尾")
    print("=" * 72)
    stoi_a, itos_a, vocab_a = build_scheme_A(names)
    Xa, Ya = build_dataset_A(names, stoi_a, BLOCK)
    S = stoi_a["."]
    tg = Counter(y for x, y in zip(Xa.tolist(), Ya.tolist()) if tuple(x) == (S, S, S))
    print(f"  词表 {len(vocab_a)}  |  context=[....] 共 {sum(tg.values()):,} 条")
    print(f"  其中目标是『.』的: {tg.get(stoi_a['.'], 0):,} "
          f"({tg.get(stoi_a['.'], 0)/sum(tg.values())*100:.1f}%)")
    C, W1, B1, W2, B2 = train(Xa[:n1], Ya[:n1], len(vocab_a), block_size=BLOCK)
    p = probe_initial(C, W1, B1, W2, B2, stoi_a, ".", ".", BLOCK)
    names_a = generate(C, W1, B1, W2, B2, stoi_a, itos_a, ".", ".", ".", n=200, block_size=BLOCK)
    short_a = sum(1 for s in names_a if len(s) <= 2)
    print(f"  训练后 P('.') = {p[stoi_a['.']].item():.4f}")
    print(f"  200 次生成中长度<=2 的: {short_a}")
    print(f"  样例: {names_a[:10]}")

    # ---------------- 方案 D ----------------
    print()
    print("=" * 72)
    print("方案 D：<P> 填充 / <S> 开头 / <E> 结尾，三者分开")
    print("=" * 72)
    stoi_d, itos_d, vocab_d = build_scheme(names)
    Xd, Yd = build_dataset_v2(names, stoi_d, "<P>", "<S>", "<E>", BLOCK)
    P = stoi_d["<P>"]
    tg = Counter(y for x, y in zip(Xd.tolist(), Yd.tolist()) if tuple(x) == (P, P, P))
    print(f"  词表 {len(vocab_d)}  |  context=[<P><P><P>] 共 {sum(tg.values()):,} 条")
    print(f"  其中目标是『<E>』的: {tg.get(stoi_d['<E>'], 0):,} "
          f"({tg.get(stoi_d['<E>'], 0)/sum(tg.values())*100:.1f}%)")
    print(f"  其中目标是『<S>』的: {tg.get(stoi_d['<S>'], 0):,} "
          f"({tg.get(stoi_d['<S>'], 0)/sum(tg.values())*100:.1f}%)")
    Cd, W1d, B1d, W2d, B2d = train(Xd[:n1], Yd[:n1], len(vocab_d), block_size=BLOCK)
    pd = probe_initial(Cd, W1d, B1d, W2d, B2d, stoi_d, "<P>", "<E>", BLOCK)
    names_d = generate(Cd, W1d, B1d, W2d, B2d, stoi_d, itos_d, "<P>", "<S>", "<E>", n=200, block_size=BLOCK)
    short_d = sum(1 for s in names_d if len(s) <= 2)
    print(f"  训练后 P(<E>) = {pd[stoi_d['<E>']].item():.4f}")
    print(f"  训练后 P(<S>) = {pd[stoi_d['<S>']].item():.4f}")
    print(f"  训练后 P(<P>) = {pd[stoi_d['<P>']].item():.4f}")
    top = torch.topk(pd, 4)
    print("  初始上下文下的 top 预测: " +
          ", ".join(f"{itos_d[i]!r}={v:.3f}" for v, i in zip(top.values.tolist(), top.indices.tolist())))
    print(f"  200 次生成中长度<=2 的: {short_d}")
    print(f"  样例: {names_d[:10]}")

    print()
    print("=" * 72)
    print("对比")
    print("=" * 72)
    print(f"  {'方案':6} {'P(结束)':>10} {'短名字/200':>12}")
    print(f"  {'A':6} {p[stoi_a['.']].item():>10.4f} {short_a:>12}")
    print(f"  {'D':6} {pd[stoi_d['<E>']].item():>10.4f} {short_d:>12}")


if __name__ == "__main__":
    main()
