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
│   ├── engine.py          （待写）
│   ├── nn.py              （待写）
│   └── train.py           训练脚本（已就绪，含 loss 曲线 + 决策边界可视化）
├── tests/                 验收测试
│   ├── test_engine.py     23 个测试
│   └── test_nn.py         23 个测试
└── notes/                 学习笔记
    ├── 01-读懂引擎.md       引擎的原理拆解
    └── 02-搭建神经网络.md   从 Value 搭到 MLP
```

## 怎么跑

```bash
# 建虚拟环境（用 uv，最快）
uv venv .venv
.venv\Scripts\activate
uv pip install pytest matplotlib

# 跑测试
.venv\Scripts\python.exe -m pytest tests/ -v

# 训练（--no-plot 可以快一倍）
.venv\Scripts\python.exe src\train.py
.venv\Scripts\python.exe src\train.py --hidden 16 --steps 200
```

## 验收标准

`tests/` 里的 **46 个测试全绿**，且 `src/` 是**自己写出来的实现**，不是复制 `refs/`。

> 测试本身已经用原始实现验证过一遍：把 `refs/micrograd/` 的内容放进 `src/`
> 时，46 个测试全部通过。所以测试失败一定是你的实现有问题，不是测试写错了。

## 性能提示

micrograd 是**纯 Python 标量运算**——每个参数、每次加减乘除都是独立的 Python 对象。
`2 -> 16 -> 16 -> 1` 的模型有 337 个参数，训练 200 步约需 **5 分钟**。
默认参数调小（8 宽、100 步）后约 **40 秒**。

**这不是 bug，是设计取舍**：用可读性换性能。也正因如此，PyTorch 才有存在的意义——
它把同样的数学用张量 + C++/CUDA 重写，快了 5 个数量级。

## 学习路线

| 阶段 | 内容 | 产出 | 状态 |
|---|---|---|---|
| 1 | 读懂 `refs/micrograd/engine.py` | `notes/01-读懂引擎.md` | ✅ 讲义已就绪 |
| 2 | 自己实现 `src/engine.py` 的 Value 类 | `test_engine.py` 全绿 | ⏳ 待你动手 |
| 3 | 读懂并实现 `src/nn.py` | `test_nn.py` 全绿 | ⏳ 待你动手 |
| 4 | 训练一个二分类 MLP | loss 曲线 + 决策边界图 | ⏳ 脚本已就绪 |
| 5 | 写笔记 + 完善本 README | 可放简历的项目 | ⏳ |

## 结果

（待填：你自己的训练曲线、决策边界图、你的发现）

