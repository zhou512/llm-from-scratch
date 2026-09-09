"""用自己写的 micrograd 训练一个二分类 MLP —— 项目 01 的最终验收。

跑法（在 projects/01-micrograd 目录下）：
    .venv\\Scripts\\python.exe src\\train.py

    # 想要更好的效果（更慢）：
    .venv\\Scripts\\python.exe src\\train.py --hidden 16 --steps 200

产出：
    outputs/loss_curve.png         训练 loss 曲线
    outputs/decision_boundary.png  决策边界
    outputs/train.log              训练日志

⚠️ 性能提示：micrograd 是纯 Python 标量运算，每一步都要对每个参数做一次
独立的加减乘除。337 个参数训 200 步大约要 5 分钟。这不是 bug，是
「用可读性换性能」的设计取舍 —— 也正因如此，PyTorch 才有存在的意义。
"""

import argparse
import os
import random
import sys

# 让 `python src/train.py` 也能 import 到 src 下的模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import Value          # noqa: E402
from nn import MLP                # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs")


# ---------------------------------------------------------------- 数据


def make_moons(n_samples=100, noise=0.1, seed=42):
    """生成两簇交错的半月形数据（经典的非线性可分数据集）。

    优先用 sklearn；没装就用手写的近似版本。
    """
    rng = random.Random(seed)
    try:
        from sklearn.datasets import make_moons as _sk_moons

        X, y = _sk_moons(n_samples=n_samples, noise=noise, random_state=seed)
        xs = [[Value(float(a)), Value(float(b))] for a, b in X]
        ys = [1.0 if int(t) == 1 else -1.0 for t in y]
        return xs, ys
    except ImportError:
        pass

    # 手写版：两个错开的半圆
    xs, ys = [], []
    half = n_samples // 2
    for i in range(half):
        t = random.uniform(0, 3.14159)
        x = 1.5 * t / 3.14159 - 0.5 + rng.gauss(0, noise)
        y = 0.7 * t / 3.14159 + rng.gauss(0, noise)
        xs.append([Value(x), Value(y)])
        ys.append(-1.0)
    for i in range(half):
        t = random.uniform(0, 3.14159)
        x = 0.5 + 1.5 * t / 3.14159 + rng.gauss(0, noise)
        y = 0.7 - 0.7 * t / 3.14159 + rng.gauss(0, noise)
        xs.append([Value(x), Value(y)])
        ys.append(1.0)
    return xs, ys


# ---------------------------------------------------------------- 训练


def svm_loss(model, xs, ys):
    """最大间隔损失（hinge loss）：sum(max(0, 1 - y * pred))"""
    preds = [model(x) for x in xs]
    losses = [(1 + -y * p).relu() for y, p in zip(ys, preds)]
    # 加一点 L2 正则，防止过拟合
    reg = 1e-4 * sum((p * p for p in model.parameters()), Value(0.0))
    return sum(losses, Value(0.0)) * (1.0 / len(losses)) + reg


def train(model, xs, ys, steps=100, lr=0.1, log_lines=None, log_every=20):
    history = []
    for step in range(steps):
        loss = svm_loss(model, xs, ys)
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        history.append(loss.data)
        if step % log_every == 0 or step == steps - 1:
            acc = accuracy(model, xs, ys)
            line = f"step {step:4d}  loss {loss.data:.4f}  acc {acc:.3f}"
            print(line)
            if log_lines is not None:
                log_lines.append(line)
    return history


def accuracy(model, xs, ys):
    correct = 0
    for x, y in zip(xs, ys):
        pred = model(x).data
        if (pred > 0) == (y > 0):
            correct += 1
    return correct / len(ys)


# ---------------------------------------------------------------- 可视化


def save_plots(history, model, xs, ys, grid_step=0.4):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("未安装 matplotlib，跳过绘图（uv pip install matplotlib）")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    # loss 曲线
    plt.figure(figsize=(6, 4))
    plt.plot(history)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.title("micrograd MLP training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "loss_curve.png"), dpi=120)
    plt.close()

    # 决策边界
    xs_raw = [x[0].data for x in xs]
    ys_raw = [x[1].data for x in xs]
    plt.figure(figsize=(6, 5))
    for label, color, marker in ((1.0, "tab:blue", "o"), (-1.0, "tab:orange", "s")):
        px = [a for a, b in zip(xs_raw, ys) if b == label]
        py = [b for a, b in zip(ys_raw, ys) if b == label]
        plt.scatter(px, py, c=color, marker=marker, s=28, label=f"class {int(label)}")

    # 在规则网格上采样模型的预测
    # 注意：micrograd 是纯 Python 标量运算，网格太密会明显变慢。
    # grid_step=0.4 约 169 个点，很快；想要更平滑的边界可以调到 0.2（约 676 个点）。
    grid_x, grid_y, grid_z = [], [], []
    v = -2.0
    while v <= 3.0 + 1e-9:
        row = []
        u = -2.0
        while u <= 3.0 + 1e-9:
            grid_x.append(u)
            row.append(model([Value(u), Value(v)]).data)
            u += grid_step
        grid_y.append(v)
        grid_z.append(row)
        v += grid_step

    cs = plt.contourf(
        sorted(set(grid_x)), grid_y, grid_z,
        levels=20, cmap="coolwarm", alpha=0.35,
    )
    plt.colorbar(cs, label="model output")
    plt.legend()
    plt.title("decision boundary learned by micrograd MLP")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "decision_boundary.png"), dpi=120)
    plt.close()
    print(f"图已保存到 {OUT_DIR}")


# ---------------------------------------------------------------- 入口


def main():
    parser = argparse.ArgumentParser(description="用 micrograd 训练二分类 MLP")
    parser.add_argument("--hidden", type=int, default=8, help="隐藏层宽度（默认 8）")
    parser.add_argument("--steps", type=int, default=100, help="训练步数（默认 100）")
    parser.add_argument("--lr", type=float, default=0.1, help="学习率（默认 0.1）")
    parser.add_argument("--grid-step", type=float, default=0.4, help="决策边界采样间隔（默认 0.4）")
    parser.add_argument("--no-plot", action="store_true", help="跳过绘图（快很多）")
    args = parser.parse_args()

    random.seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    log_lines = []

    xs, ys = make_moons(n_samples=100, noise=0.1)
    header = f"数据集: {len(xs)} 个样本, 2 维特征, 2 分类"
    print(header)
    log_lines.append(header)

    model = MLP(2, [args.hidden, args.hidden, 1])
    info = (f"模型: 2 -> {args.hidden} -> {args.hidden} -> 1\n"
            f"参数量: {len(model.parameters())}\n"
            f"步数: {args.steps}  学习率: {args.lr}")
    print(info)
    log_lines.append(info)
    print()

    history = train(model, xs, ys, steps=args.steps, lr=args.lr, log_lines=log_lines)

    final_acc = accuracy(model, xs, ys)
    tail = (f"\n最终准确率: {final_acc:.3f}\n"
            f"初始 loss: {history[0]:.4f}\n"
            f"最终 loss: {history[-1]:.4f}")
    print(tail)
    log_lines.append(tail)

    if not args.no_plot:
        save_plots(history, model, xs, ys, grid_step=args.grid_step)

    with open(os.path.join(OUT_DIR, "train.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"\n日志已保存到 {os.path.join(OUT_DIR, 'train.log')}")


if __name__ == "__main__":
    main()
