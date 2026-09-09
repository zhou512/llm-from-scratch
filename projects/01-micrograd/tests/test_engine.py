"""engine.py 的验收测试。

规则：这些测试用来验收你自己的 src/engine.py，不许改测试来迁就实现。
如果某个测试失败，说明你的实现有 bug 或理解有偏差 —— 这正是学习发生的地方。
"""

import math

import pytest

from src.engine import Value


# ---------------------------------------------------------------- 基础构造


def test_value_stores_data():
    v = Value(3.5)
    assert v.data == 3.5
    assert v.grad == 0.0


def test_value_has_prev_and_op():
    a = Value(1.0)
    b = Value(2.0)
    c = a + b
    assert c.op == "+"
    assert set(c._prev) == {a, b}


# ---------------------------------------------------------------- 前向计算


def test_addition():
    assert (Value(2.0) + Value(3.0)).data == 5.0


def test_radd():
    """1 + Value(2) —— Python 会先试 int.__add__，失败后调 Value.__radd__。"""
    assert (1 + Value(2.0)).data == 3.0


def test_multiplication():
    assert (Value(2.0) * Value(3.0)).data == 6.0


def test_rmul():
    assert (3 * Value(2.0)).data == 6.0


def test_subtraction():
    assert (Value(2.0) - Value(3.0)).data == -1.0


def test_rsub():
    """3 - Value(2) 必须是 1，不是 -1。__rsub__ 里的顺序最容易写反。"""
    assert (3 - Value(2.0)).data == 1.0


def test_division():
    assert (Value(6.0) / Value(3.0)).data == 2.0


def test_truediv():
    assert (Value(6.0) / 3).data == 2.0


def test_rtruediv():
    """6 / Value(3) 必须是 2，不是 0.5。"""
    assert (6 / Value(3.0)).data == 2.0


def test_power():
    assert (Value(2.0) ** 3).data == 8.0


def test_power_rpow():
    assert (2 ** Value(3.0)).data == 8.0


def test_negation():
    assert (-Value(2.0)).data == -2.0


def test_relu_positive():
    assert (Value(2.0)).relu().data == 2.0


def test_relu_negative():
    assert (Value(-2.0)).relu().data == 0.0


def test_tanh():
    assert Value(0.0).tanh().data == pytest.approx(0.0)
    assert Value(1.0).tanh().data == pytest.approx(math.tanh(1.0))


# ---------------------------------------------------------------- 反向传播


def test_simple_gradient_add():
    """c = a + b  =>  dc/da = 1, dc/db = 1"""
    a, b = Value(2.0), Value(3.0)
    c = a + b
    c.backward()
    assert a.grad == 1.0
    assert b.grad == 1.0


def test_simple_gradient_mul():
    """c = a * b  =>  dc/da = b, dc/db = a"""
    a, b = Value(2.0), Value(3.0)
    c = a * b
    c.backward()
    assert a.grad == 3.0
    assert b.grad == 2.0


def test_gradient_chain():
    """d = a*b + b**3，手推结果：dd/da = b = 2, dd/db = a + 3b^2 = -4+12 = 8"""
    a, b = Value(-4.0), Value(2.0)
    d = a * b + b ** 3
    d.backward()
    assert a.grad == 2.0
    assert b.grad == 8.0


def test_gradient_shared_node_accumulates():
    """同一个节点被用到两次，梯度必须【累加】而不是覆盖。"""
    a = Value(3.0)
    b = a * a          # b = a^2, db/da = 2a = 6
    b.backward()
    assert a.grad == 6.0


def test_gradient_diamond():
    """菱形依赖：a 通过两条路径影响 d，梯度要相加。"""
    a = Value(2.0)
    b = a + 1.0        # db/da = 1
    c = a * 3.0        # dc/da = 3
    d = b * c          # dd/db = c = 9, dd/dc = b = 3
    d.backward()
    # dd/da = dd/db * db/da + dd/dc * dc/da = 9*1 + 3*3 = 18
    assert d.data == 18.0
    assert a.grad == 18.0


def test_gradient_of_karpathy_example():
    """原 README 里的完整例子，两个已知数值答案。"""
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
    assert g.data == pytest.approx(24.7041, abs=1e-4)
    g.backward()
    assert a.grad == pytest.approx(138.8338, abs=1e-4)
    assert b.grad == pytest.approx(645.5773, abs=1e-4)


def test_backward_resets_grad_between_runs():
    """连续两次 backward 不应该把梯度累加到上一次的结果上（根节点要清零）。"""
    a = Value(2.0)
    b = Value(3.0)
    c = a * b
    c.backward()
    first = a.grad
    c.backward()
    assert a.grad == first


def test_gradients_are_floats():
    a = Value(2.0)
    b = Value(3.0)
    (a * b).backward()
    assert isinstance(a.grad, float)
    assert isinstance(b.grad, float)
