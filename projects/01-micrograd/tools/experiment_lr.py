"""实验：学习率（learning rate）对训练的影响。

对应思考题：「如果把 --lr 从 0.1 改成 1.0 会怎样？」

用法（在 projects/01-micrograd 目录下）：
    .venv\\Scripts\\python.exe tools\\experiment_lr.py
"""

import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "src"))

from engine import Value       # noqa: E402
from nn import MLP             # noqa: E402
import train as T              # noqa: E402


def run(lr, xs, ys, steps=100, hidden=8, seed=42):
    random.seed(seed)
    model = MLP(2, [hidden, hidden, 1])
    losses = []
    diverged = False
    for step in range(steps):
        preds = [model(x) for x in xs]
        losses_ = [(1 + -y * p).relu() for y, p in zip(ys, preds)]
        loss = sum(losses_, Value(0.0)) * (1.0 / len(losses_))
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)
        if loss.data != loss.data or abs(loss.data) > 1e6:   # NaN 或爆炸
            diverged = True
            break
    acc = T.accuracy(model, xs, ys)
    return losses, acc, diverged


def main():
    xs, ys = T.make_moons(n_samples=100, noise=0.1)
    print(f"数据集: {len(xs)} 个样本")
    print()

    rows = []
    for lr in [0.001, 0.01, 0.1, 0.5, 1.0, 2.0]:
        losses, acc, diverged = run(lr, xs, ys)
        rows.append((lr, losses, acc, diverged))

    print(f"{'学习率':>8}  {'初始 loss':>10}  {'最终 loss':>10}  {'准确率':>8}   状态")
    print("-" * 62)
    for lr, losses, acc, diverged in rows:
        if diverged:
            print(f"{lr:>8}  {losses[0]:>10.4f}  {'发散/NaN':>10}  {'-':>8}   ❌ 训练崩了")
        else:
            trend = "收敛" if losses[-1] < losses[0] else "没学到"
            print(f"{lr:>8}  {losses[0]:>10.4f}  {losses[-1]:>10.4f}  {acc:>8.3f}   {trend}")

    print()
    print("=" * 62)
    print("每个学习率的 loss 曲线（前 30 步）")
    print("=" * 62)
    for lr, losses, acc, diverged in rows:
        pts = " ".join(f"{v:.2f}" for v in losses[:30])
        print(f"lr={lr:<6} {pts}")

    print("""
实测结果与解读
======================================================================
     学习率     最终 loss   准确率
     0.001        0.9288     0.650     ← 几乎没学（步子太小）
      0.01        0.8360     0.670
       0.1        0.2435     0.860     ← train.py 的默认值
       0.5        0.0374     1.000
       1.0        0.0000     1.000     ← 最好？先别急着下结论
       2.0        0.0511     0.990     ← 曲线明显震荡

三个观察
----------------------------------------------------------------------
1) lr 太小（0.001/0.01）：每步挪动极小，100 步到不了谷底，loss 几乎不动。
   看曲线：0.94 一直平着 —— 这叫「欠训练」，不是模型不行。

2) lr 太大（2.0）：loss 曲线在 0.33 -> 0.49 -> 1.74 -> 1.78 -> 0.86 之间
   剧烈震荡。步子跨过了谷底，来回弹。再大就会发散成 NaN。

3) ★ lr=1.0 的 loss 是 0.0000、准确率 100% —— 但这个结论【不可信】！

   为什么不可信：
   - 这个实验在【全部 100 个样本】上训练，也在【同一批样本】上算准确率
   - loss 0.0000 意味着铰链损失 sum(max(0, 1 - y*pred)) 里每一项都是 0，
     也就是所有样本都满足 y*pred >= 1
   - 这是可能的，但更可能是【记住了训练数据】，而不是学到了规律
   - 没有留出验证集，所以无法区分「学会了」和「背下来了」

   正确做法：切一部分数据出来做验证集，只看验证集上的表现。
   nanoGPT 的 train.py 就是这么做的（train.bin / val.bin 分开）。
""")


if __name__ == "__main__":
    main()
