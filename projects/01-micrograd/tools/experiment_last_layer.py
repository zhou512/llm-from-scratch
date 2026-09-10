"""实验：最后一层的 nonlin（是否加 ReLU）会怎样影响训练？

对应思考题：「把最后一层的 nonlin 改成 True，训练效果怎么变？为什么？」
对应思考题：「决策边界是什么形状？为什么不是一条直线？」

用法（在 projects/01-micrograd 目录下）：
    .venv\\Scripts\\python.exe tools\\experiment_last_layer.py
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


def build_mlp(nin, hidden, nout, last_nonlin):
    """搭一个 MLP，但强制指定最后一层是否加激活函数。

    MLP 的 __init__ 里写死了 `nonlin = i != len(nouts)-1`，
    所以这里搭好之后手动把最后一层的 nonlin 改掉。
    """
    m = MLP(nin, hidden)
    if nout == 1:
        m.layers[-1].neurons[0].nonlin = last_nonlin
    else:
        for n in m.layers[-1].neurons:
            n.nonlin = last_nonlin
    return m


def run_case(name, last_nonlin, xs, ys, steps=100, lr=0.1):
    random.seed(42)
    model = build_mlp(2, [8, 8, 1], 1, last_nonlin)

    losses = []
    for step in range(steps):
        # SVM(hinge) loss
        preds = [model(x) for x in xs]
        losses_ = [(1 + -y * p).relu() for y, p in zip(ys, preds)]
        loss = sum(losses_, Value(0.0)) * (1.0 / len(losses_))
        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad
        losses.append(loss.data)

    acc = T.accuracy(model, xs, ys)

    # 看看模型对两个类别的输出范围
    out_pos = [model(x).data for x, y in zip(xs, ys) if y > 0]
    out_neg = [model(x).data for x, y in zip(xs, ys) if y < 0]

    print(f"{'='*66}")
    print(f"{name}（最后一层 nonlin = {last_nonlin}）")
    print(f"{'='*66}")
    print(f"  loss: {losses[0]:.4f} -> {losses[-1]:.4f}   准确率: {acc:.3f}")
    print(f"  正类样本的模型输出范围: [{min(out_pos):8.4f}, {max(out_pos):8.4f}]")
    print(f"  负类样本的模型输出范围: [{min(out_neg):8.4f}, {max(out_neg):8.4f}]")
    print(f"  负类输出里有几个 > 0:    {sum(1 for v in out_neg if v > 0)} / {len(out_neg)}")
    print()
    return losses, acc, model


def main():
    print()
    xs, ys = T.make_moons(n_samples=100, noise=0.1)
    print(f"数据集: {len(xs)} 个样本，正类 {sum(1 for y in ys if y>0)} 个，负类 {sum(1 for y in ys if y<0)} 个")
    print()

    run_case("情况 A：线性输出层（正确做法）", False, xs, ys)
    run_case("情况 B：最后一层也加 ReLU", True, xs, ys)

    print("=" * 66)
    print("结论")
    print("=" * 66)
    print("""
情况 B 的关键问题：ReLU 把所有负输出截断成 0。

SVM(hinge) 损失是：loss = (1 - y * pred).relu()
  - 当 y = -1（负类）、pred 应该为负数才能"分对"
  - 但 ReLU 强制 pred >= 0，于是 -y*pred = pred >= 0
  - 损失恒 >= 1，这个样本永远有损失，永远学不好

换句话说：加了 ReLU，模型就无法表达"我确信这是负类"。
它的输出只能落在 [0, +inf)，没有负半轴可用。
""")


if __name__ == "__main__":
    main()
