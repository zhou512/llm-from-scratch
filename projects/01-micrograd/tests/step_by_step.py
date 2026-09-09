"""分步验证脚本：每实现一步就跑一次，只测这一步。

为什么需要它：
`test_engine.py` 有 23 个测试，一次性跑会一次报一堆错，不知道从哪下手。
这个脚本把实现拆成 6 个步骤，每步只跑该步的检查，通过后再进下一步。

用法（在 projects/01-micrograd 目录下）：
    .venv\\Scripts\\python.exe tests\\step_by_step.py

    # 只跑到第 3 步
    .venv\\Scripts\\python.exe tests\\step_by_step.py --until 3

设计原则：每一步的检查都对应 `notes/04-分步实现指南.md` 里的一节。
"""

import argparse
import math
import os
import sys

# 让 `python tests/step_by_step.py` 也能 import 到 src 下的模块
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check(cond, msg_ok, msg_fail):
    if cond:
        print(f"    ✅ {msg_ok}")
        return True
    print(f"    ❌ {msg_fail}")
    return False


def approx(a, b, tol=1e-6):
    return abs(a - b) < tol


# ---------------------------------------------------------------- 步骤 1


def step1_骨架():
    """Value 能存数据、有 grad、有 _prev/_op/_backward 字段。"""
    from src.engine import Value

    ok = True
    v = Value(3.5)
    ok &= check(approx(v.data, 3.5), "data 正确", f"data 应为 3.5，实际 {v.data}")
    ok &= check(approx(v.grad, 0.0), "grad 初始为 0", f"grad 应为 0，实际 {v.grad}")
    ok &= check(hasattr(v, "_prev"), "有 _prev", "缺少 _prev 属性")
    ok &= check(hasattr(v, "_op"), "有 _op", "缺少 _op 属性")
    ok &= check(callable(getattr(v, "_backward", None)), "有可调用的 _backward", "缺少 _backward")
    ok &= check(len(v._prev) == 0, "_prev 初始为空", f"_prev 应为空，实际 {v._prev}")

    a, b = Value(1.0), Value(2.0)
    c = a + b
    ok &= check(c._op == "+", "c._op 记录为 '+'", f"c._op 应为 '+'，实际 {c._op!r}")
    ok &= check(set(c._prev) == {a, b}, "c._prev 记录了 a 和 b", "c._prev 没记录父节点")
    return ok


# ---------------------------------------------------------------- 步骤 2


def step2_前向():
    """四则运算和幂、取负的前向值正确。"""
    from src.engine import Value

    ok = True
    cases = [
        ("加法", (Value(2.0) + Value(3.0)).data, 5.0),
        ("右加", (1 + Value(2.0)).data, 3.0),
        ("减法", (Value(2.0) - Value(3.0)).data, -1.0),
        ("右减 3-Value(2)", (3 - Value(2.0)).data, 1.0),
        ("乘法", (Value(2.0) * Value(3.0)).data, 6.0),
        ("右乘", (3 * Value(2.0)).data, 6.0),
        ("除法", (Value(6.0) / Value(3.0)).data, 2.0),
        ("右除 6/Value(3)", (6 / Value(3.0)).data, 2.0),
        ("幂", (Value(2.0) ** 3).data, 8.0),
        ("取负", (-Value(2.0)).data, -2.0),
        ("relu 正", Value(2.0).relu().data, 2.0),
        ("relu 负", Value(-2.0).relu().data, 0.0),
    ]
    for name, got, want in cases:
        ok &= check(approx(got, want), f"{name}: {got}", f"{name}: 应为 {want}，实际 {got}")
    return ok


# ---------------------------------------------------------------- 步骤 3


def step3_单节点梯度():
    """最简单的反向传播：c = a+b 和 c = a*b。"""
    from src.engine import Value

    ok = True

    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.backward()
    ok &= check(approx(a.grad, 1.0) and approx(b.grad, 1.0),
                "加法梯度 dc/da=1, dc/db=1", f"加法梯度错: {a.grad}, {b.grad}")

    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    ok &= check(approx(a.grad, 3.0) and approx(b.grad, 2.0),
                "乘法梯度 dc/da=b=3, dc/db=a=2", f"乘法梯度错: {a.grad}, {b.grad}")

    a = Value(3.0)
    b = a * a
    b.backward()
    ok &= check(approx(a.grad, 6.0),
                "同一节点用两次，梯度累加得 2a=6",
                f"梯度应累加为 6，实际 {a.grad}（是不是用了 = 而不是 +=？）")

    a = Value(2.0)
    out = a ** 3
    out.backward()
    ok &= check(approx(a.grad, 12.0),
                "幂函数 d(a^3)/da = 3a² = 12",
                f"幂函数梯度应为 12，实际 {a.grad}")

    return ok


# ---------------------------------------------------------------- 步骤 4


def step4_链式法则():
    """多层依赖 + 菱形依赖，考验拓扑排序和累加。"""
    from src.engine import Value

    ok = True

    # 手推：d = a*b + b**3，a=-4, b=2
    #   dd/da = b = 2
    #   dd/db = a + 3b² = -4 + 12 = 8
    a, b = Value(-4.0), Value(2.0)
    d = a * b + b ** 3
    d.backward()
    ok &= check(approx(d.data, 0.0), "d = (-4)(2) + 8 = 0", f"d 应为 0，实际 {d.data}")
    ok &= check(approx(a.grad, 2.0), "dd/da = 2", f"dd/da 应为 2，实际 {a.grad}")
    ok &= check(approx(b.grad, 8.0), "dd/db = 8", f"dd/db 应为 8，实际 {b.grad}")

    # 菱形：a -> b=a+1, c=a*3, d=b*c
    #   dd/da = c*1 + b*3 = 6 + 9 = 15
    a = Value(2.0)
    b = a + 1.0
    c = a * 3.0
    d = b * c
    d.backward()
    ok &= check(approx(a.grad, 15.0),
                "菱形依赖 dd/da = 15",
                f"菱形依赖梯度应为 15，实际 {a.grad}（梯度有没有累加？）")

    # 原 README 的完整例子
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b ** 3
    c += c + 1
    c += 1 + c + (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c - d
    f = e ** 2
    g = f / 2.0
    g += 10.0 / f
    ok &= check(approx(g.data, 24.7041, 1e-4),
                f"前向 g = 24.7041（实际 {g.data:.4f}）",
                f"前向 g 应为 24.7041，实际 {g.data:.4f}")
    g.backward()
    ok &= check(approx(a.grad, 138.8338, 1e-4),
                f"dg/da = 138.8338（实际 {a.grad:.4f}）",
                f"dg/da 应为 138.8338，实际 {a.grad:.4f}")
    ok &= check(approx(b.grad, 645.5773, 1e-4),
                f"dg/db = 645.5773（实际 {b.grad:.4f}）",
                f"dg/db 应为 645.5773，实际 {b.grad:.4f}")
    return ok


# ---------------------------------------------------------------- 步骤 5


def step5_对比PyTorch():
    """用 PyTorch 的 autograd 当「标准答案」，逐算子核对。

    这是最有说服力的验证方式：如果你手写的梯度和 PyTorch 一致，
    说明链式法则实现正确。
    """
    try:
        import torch
    except ImportError:
        print("    ⏭  跳过（未安装 torch）")
        return True

    from src.engine import Value

    ok = True
    # (表达式, 取值) —— 每个表达式都同时用到 a 和 b，
    # 否则 PyTorch 里没被用到的那个变量的 .grad 是 None，没法比较
    cases = [
        ("a+b", lambda a, b: a + b, (2.0, 3.0)),
        ("a-b", lambda a, b: a - b, (2.0, 3.0)),
        ("a*b", lambda a, b: a * b, (-2.0, 3.0)),
        ("a/b", lambda a, b: a / b, (6.0, 3.0)),
        ("a**2 * b", lambda a, b: a ** 2 * b, (3.0, 2.0)),
        ("a**3 + b**2", lambda a, b: a ** 3 + b ** 2, (-2.0, 3.0)),
        ("(a*b+b**3).relu()", lambda a, b: (a * b + b ** 3).relu(), (-4.0, 2.0)),
        ("a*b + a/b - b**2", lambda a, b: a * b + a / b - b ** 2, (3.0, 2.0)),
    ]
    for name, fn, (av, bv) in cases:
        # 我们的实现
        a, b = Value(av), Value(bv)
        out = fn(a, b)
        out.backward()
        our = (a.grad, b.grad, out.data)

        # PyTorch
        ta = torch.tensor(av, requires_grad=True)
        tb = torch.tensor(bv, requires_grad=True)
        tout = fn(ta, tb)
        tout.backward()
        ref = (ta.grad.item(), tb.grad.item(), tout.item())

        match = all(approx(x, y, 1e-5) for x, y in zip(our, ref))
        ok &= check(match, f"{name}: grad_a={our[0]:.4f}, grad_b={our[1]:.4f} 与 PyTorch 一致",
                    f"{name}: 我们 ({our[0]:.4f}, {our[1]:.4f}) vs PyTorch ({ref[0]:.4f}, {ref[1]:.4f})")
    return ok


# ---------------------------------------------------------------- 步骤 6


def step6_数值梯度():
    """用数值微分（有限差分）验证梯度 —— 不依赖 PyTorch。"""
    from src.engine import Value

    ok = True

    def f(a_val, b_val):
        a, b = Value(a_val), Value(b_val)
        return a * b + a ** 2 * b ** 3 + (a - b).relu()

    av, bv = 1.5, -0.7
    a, b = Value(av), Value(bv)
    out = a * b + a ** 2 * b ** 3 + (a - b).relu()
    out.backward()
    our_a, our_b = a.grad, b.grad

    eps = 1e-6
    num_a = (f(av + eps, bv).data - f(av - eps, bv).data) / (2 * eps)
    num_b = (f(av, bv + eps).data - f(av, bv - eps).data) / (2 * eps)

    ok &= check(approx(our_a, num_a, 1e-4),
                f"∂/∂a: 解析 {our_a:.6f} ≈ 数值 {num_a:.6f}",
                f"∂/∂a: 解析 {our_a:.6f} vs 数值 {num_a:.6f} 不符")
    ok &= check(approx(our_b, num_b, 1e-4),
                f"∂/∂b: 解析 {our_b:.6f} ≈ 数值 {num_b:.6f}",
                f"∂/∂b: 解析 {our_b:.6f} vs 数值 {num_b:.6f} 不符")
    return ok


# ---------------------------------------------------------------- 主流程

STEPS = [
    (1, "Value 骨架（data / grad / _prev / _op / _backward）", step1_骨架),
    (2, "前向计算（+ - * / ** 取负 relu）", step2_前向),
    (3, "单节点反向传播（局部导数 + 累加）", step3_单节点梯度),
    (4, "链式法则（拓扑排序 + 多层/菱形依赖）", step4_链式法则),
    (5, "与 PyTorch autograd 对比", step5_对比PyTorch),
    (6, "数值梯度验证（不依赖 PyTorch）", step6_数值梯度),
]


def main():
    parser = argparse.ArgumentParser(description="micrograd engine.py 分步验证")
    parser.add_argument("--until", type=int, default=6, help="跑到第几步就停（默认 6）")
    args = parser.parse_args()

    passed_all = True
    for num, title, fn in STEPS:
        if num > args.until:
            break
        print(f"\n{'='*66}")
        print(f"步骤 {num} / 6 : {title}")
        print(f"{'='*66}")
        try:
            ok = fn()
        except ImportError as e:
            print(f"    ⛔ 导入失败: {e}")
            print("       → 检查 src/engine.py 是否定义了 Value 类")
            passed_all = False
            break
        except Exception as e:
            print(f"    💥 抛异常: {type(e).__name__}: {e}")
            passed_all = False
            break

        if not ok:
            print(f"\n⛔ 步骤 {num} 未通过，先修好这一步再往下。")
            print("   对照 notes/04-分步实现指南.md 的对应小节，或 notes/03-常见错误对照表.md")
            passed_all = False
            break
        print(f"✅ 步骤 {num} 通过")

    if passed_all:
        print(f"\n{'='*66}")
        print("🎉 全部通过！接下来：")
        print("   1. 跑完整测试：.venv\\Scripts\\python.exe -m pytest tests/ -v")
        print("   2. 训练一个 MLP：.venv\\Scripts\\python.exe src\\train.py")
        print("   3. 提交代码：git add src/engine.py && git commit -m 'feat: ...'")
        print(f"{'='*66}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
