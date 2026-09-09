"""nn.py 的验收测试。

规则同 test_engine.py：不许改测试迁就实现。

注意：本文件假设你按下面方式组织代码
    src/engine.py   ->  class Value
    src/nn.py       ->  class Module / Neuron / Layer / MLP
如果你把 nn.py 里的 import 写成 `from .engine import Value`（推荐）或
`from src.engine import Value`，都能跑通。
"""

import pytest

from src.engine import Value
from src.nn import MLP, Layer, Module, Neuron


# ---------------------------------------------------------------- Module 基类


def test_module_base_returns_empty_parameters():
    assert Module().parameters() == []


def test_module_zero_grad_does_nothing_on_base():
    Module().zero_grad()      # 不应抛异常


# ---------------------------------------------------------------- Neuron


def test_neuron_parameter_count():
    n = Neuron(3)
    assert len(n.parameters()) == 4          # 3 个 w + 1 个 b


def test_neuron_bias_starts_at_zero():
    n = Neuron(3)
    assert n.b.data == 0


def test_neuron_weights_are_initialized_in_range():
    n = Neuron(5)
    for w in n.w:
        assert -1.0 <= w.data <= 1.0


def test_neuron_forward_returns_value():
    n = Neuron(2)
    out = n([Value(1.0), Value(1.0)])
    assert isinstance(out, Value)


def test_neuron_relu_never_negative():
    n = Neuron(2)
    n.w = [Value(-1.0), Value(-1.0)]
    n.b = Value(0.0)
    out = n([Value(1.0), Value(1.0)])        # 加权和 = -2
    assert out.data == 0.0                   # ReLU 截断


def test_neuron_linear_can_be_negative():
    n = Neuron(2, nonlin=False)
    n.w = [Value(-1.0), Value(-1.0)]
    n.b = Value(0.0)
    out = n([Value(1.0), Value(1.0)])
    assert out.data == -2.0                  # 不截断


def test_neuron_forward_computes_dot_product_plus_bias():
    """手推：w=[2,3], b=1, x=[4,5] => 2*4 + 3*5 + 1 = 24"""
    n = Neuron(2, nonlin=False)
    n.w = [Value(2.0), Value(3.0)]
    n.b = Value(1.0)
    out = n([Value(4.0), Value(5.0)])
    assert out.data == 24.0


def test_neuron_gradients_flow_to_weights_and_bias():
    """y = w·x + b 对 w 的偏导就是 x，对 b 的偏导是 1。"""
    n = Neuron(2, nonlin=False)
    n.w = [Value(1.0), Value(1.0)]
    n.b = Value(0.0)
    out = n([Value(2.0), Value(3.0)])        # 5
    out.backward()
    assert n.w[0].grad == 2.0
    assert n.w[1].grad == 3.0
    assert n.b.grad == 1.0


def test_neuron_zero_grad_clears_gradients():
    # 用线性神经元，避免 ReLU 在负区间把梯度截断成 0
    n = Neuron(2, nonlin=False)
    n.w = [Value(1.0), Value(1.0)]
    n.b = Value(0.0)
    out = n([Value(1.0), Value(1.0)])        # 加权和 = 2
    out.backward()
    assert any(p.grad != 0 for p in n.parameters())
    n.zero_grad()
    assert all(p.grad == 0 for p in n.parameters())


# ---------------------------------------------------------------- Layer


def test_layer_parameter_count():
    layer = Layer(3, 4)
    assert len(layer.parameters()) == 4 * (3 + 1)


def test_layer_output_length():
    layer = Layer(2, 5)
    out = layer([Value(1.0), Value(1.0)])
    assert isinstance(out, list)
    assert len(out) == 5


def test_layer_single_neuron_returns_scalar():
    """原始实现的特例：只有 1 个神经元时返回单个 Value，不是列表。"""
    layer = Layer(2, 1)
    out = layer([Value(1.0), Value(1.0)])
    assert isinstance(out, Value)


def test_layer_parameters_are_flat():
    layer = Layer(2, 3)
    params = layer.parameters()
    assert len(params) == 3 * 3
    assert all(isinstance(p, Value) for p in params)


# ---------------------------------------------------------------- MLP


def test_mlp_layer_count():
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.layers) == 3


def test_mlp_parameter_count():
    """3->4->4->1 的参数总数：
       第 1 层：4 个神经元 × (3 权重 + 1 偏置) = 16
       第 2 层：4 个神经元 × (4 权重 + 1 偏置) = 20
       第 3 层：1 个神经元 × (4 权重 + 1 偏置) =  5
                                          合计 = 41
    """
    mlp = MLP(3, [4, 4, 1])
    assert len(mlp.parameters()) == 41


def test_mlp_output_is_scalar_for_binary_classification():
    mlp = MLP(3, [4, 4, 1])
    out = mlp([Value(1.0), Value(2.0), Value(3.0)])
    assert isinstance(out, Value)


def test_mlp_output_is_vector_for_multi_output():
    mlp = MLP(2, [4, 3])
    out = mlp([Value(1.0), Value(2.0)])
    assert isinstance(out, list)
    assert len(out) == 3


def test_mlp_parameters_all_value():
    mlp = MLP(2, [3, 1])
    assert all(isinstance(p, Value) for p in mlp.parameters())


def test_mlp_zero_grad():
    mlp = MLP(2, [3, 1])
    out = mlp([Value(1.0), Value(1.0)])
    out.backward()
    mlp.zero_grad()
    assert all(p.grad == 0 for p in mlp.parameters())


def test_mlp_gradients_reach_first_layer():
    """反向传播必须能穿过所有层，第一层的参数也要拿到梯度。"""
    mlp = MLP(3, [4, 4, 1])
    out = mlp([Value(1.0), Value(2.0), Value(3.0)])
    out.backward()
    first_layer_params = mlp.layers[0].parameters()
    assert any(p.grad != 0 for p in first_layer_params)


# ---------------------------------------------------------------- 训练能力


def test_mlp_can_reduce_loss_on_toy_data():
    """最小训练验证：一个极简的二分类任务，loss 必须下降。

    这不是在测"能不能训到 100% 准确"，而是在测
    「前向 + 反向 + 参数更新」这条链路是通的。
    """
    xs = [
        [Value(2.0), Value(3.0), Value(-1.0)],
        [Value(3.0), Value(-1.0), Value(0.5)],
        [Value(0.5), Value(1.0), Value(1.0)],
        [Value(1.0), Value(1.0), Value(-1.0)],
    ]
    ys = [1.0, -1.0, -1.0, 1.0]

    mlp = MLP(3, [4, 4, 1])

    def compute_loss():
        preds = [mlp(x) for x in xs]
        return sum((1 + -y * p).relu() for y, p in zip(ys, preds))

    first_loss = compute_loss().data

    for _ in range(50):
        preds = [mlp(x) for x in xs]
        loss = sum((1 + -y * p).relu() for y, p in zip(ys, preds))
        mlp.zero_grad()
        loss.backward()
        for p in mlp.parameters():
            p.data += -0.05 * p.grad

    last_loss = compute_loss().data
    assert last_loss < first_loss, f"loss 没有下降：{first_loss} -> {last_loss}"
