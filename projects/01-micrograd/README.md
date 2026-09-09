# 01 · micrograd 复现：从零实现自动求导引擎

## 这是什么

复现 Andrej Karpathy 的 [micrograd](https://github.com/karpathy/micrograd)：
一个**标量级自动微分引擎**，用纯 Python 实现反向传播（reverse-mode autodiff）。

原始实现只有 **111 行**（`engine.py` 69 行 + `nn.py` 42 行），零依赖。

## 为什么值得做

`loss.backward()` 这一行在 PyTorch 里你写过几百次，但**梯度到底是怎么算出来的**？
micrograd 用 69 行告诉你答案。做完这个项目：

- 你能手推并手写链式法则
- 你能解释计算图、拓扑排序、`requires_grad`、`.grad` 是怎么来的
- 你再看 nanoGPT 的 `model.py`，就是「读懂」而不是「抄」

## 目录结构

```
01-micrograd/
├── README.md              本文件
├── refs/micrograd/        ★ 原始实现快照，只读参照，不要改
│   ├── engine.py          自动求导引擎（69 行）
│   ├── nn.py              神经网络库（42 行）
│   └── UPSTREAM-README.md 原作者的说明
├── src/                   ★ 我自己的实现（从零写，不许抄）
│   ├── engine.py
│   └── nn.py
├── tests/                 验收测试
│   ├── test_engine.py
│   └── test_nn.py
└── notes/                 学习笔记
    └── 01-读懂引擎.md
```

## 怎么跑

```bash
# 建虚拟环境（用 uv，最快）
uv venv .venv
.venv\Scripts\activate
uv pip install pytest

# 跑测试
python -m pytest tests/ -v
```

## 验收标准

`tests/` 里的测试**全绿**，且是**自己写出来的实现**，不是复制 `refs/`。

## 学习路线

| 阶段 | 内容 | 产出 |
|---|---|---|
| 1 | 读懂 `refs/micrograd/engine.py` | `notes/01-读懂引擎.md` |
| 2 | 自己实现 `src/engine.py` 的 Value 类 | 测试通过 |
| 3 | 读懂并实现 `src/nn.py`（Neuron/Layer/MLP） | 测试通过 |
| 4 | 训练一个二分类 MLP | loss 曲线 + 决策边界图 |
| 5 | 写笔记 + 完善本 README | 可放简历的项目 |

## 结果

（待填：训练曲线、决策边界图、你的发现）
