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
    ├── 02-搭建神经网络.md   从 Value 搭到 MLP
    └── 03-常见错误对照表.md  按报错查原因的排错表
```

> **为什么 `refs/` 是代码快照而不是 git submodule？**
> 上游仓库完整历史（25 次提交）在 `D:\git-reproduce\_candidates\micrograd-full`。
> 放进本项目的是快照，因为：submodule 对 git 新手是个负担（克隆要 `--recursive`，
> 更新要 `submodule update`），而这里只需要「对照阅读」这一个用途。
> 等你熟悉 git 后，可以换成 submodule 试试——那时你会明白它的取舍。

## 怎么跑

```bash
# 建虚拟环境（用 uv，最快）
# --system-site-packages 让 venv 复用系统已装的 torch（步骤 5 的 PyTorch 对拍需要）
uv venv .venv --system-site-packages
.venv\Scripts\activate
uv pip install pytest matplotlib

# ★ 分步实现：每完成一步就跑一次，只测这一步
.venv\Scripts\python.exe tests\step_by_step.py
.venv\Scripts\python.exe tests\step_by_step.py --until 3   # 只跑到第 3 步

# 全部实现后跑完整测试
.venv\Scripts\python.exe -m pytest tests/ -v

# 训练（--no-plot 可以快一倍）
.venv\Scripts\python.exe src\train.py
.venv\Scripts\python.exe src\train.py --hidden 16 --steps 200
```

## 验收标准

1. **`tests/step_by_step.py` 6 步全通过**（推荐先跑这个，一步一步来）
2. `tests/` 里的 **46 个测试全绿**
3. `src/` 是**自己写出来的实现**，不是复制 `refs/`

> 两套验证都已在原始实现上跑过：46 个 pytest 测试全绿，
> 6 步分步检查全通过（含与 PyTorch 的 8 项梯度对拍）。
> 所以测试失败一定是你的实现有问题，不是测试写错了。

### 两套验证的分工

| 工具 | 什么时候用 | 特点 |
|---|---|---|
| `tests/step_by_step.py` | 写代码过程中 | 6 步渐进，报错定位到具体步骤 |
| `tests/test_engine.py` + `test_nn.py` | 实现完成后 | 46 个细粒度测试，覆盖边界情况 |

### 三种独立的正确性验证

| 方式 | 依据 | 说服力 |
|---|---|---|
| 手推梯度 | 链式法则 | 基础 |
| **与 PyTorch autograd 对拍** | 工业级实现 | 强 |
| **数值微分（有限差分）** | 导数定义 | 强，且不依赖任何框架 |

三条独立路径都对上，说明实现是真的对，而不是「碰巧过了测试」。

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
| 2 | 自己实现 `src/engine.py` 的 Value 类 | `step_by_step.py` 6 步 + 46 测试全绿 | ⏳ 待你动手 |
| 3 | 读懂并实现 `src/nn.py` | `test_nn.py` 全绿 | ⏳ 待你动手 |
| 4 | 训练一个二分类 MLP | loss 曲线 + 决策边界图 | ⏳ 脚本已就绪 |
| 5 | 写笔记 + 完善本 README | 可放简历的项目 | ⏳ |

### 笔记清单

| 文件 | 内容 |
|---|---|
| `notes/01-读懂引擎.md` | 引擎的原理拆解（三件事模式、拓扑排序、`+=` 累加） |
| `notes/02-搭建神经网络.md` | 从 Value 搭到 MLP（Module 模板方法、`sum(..., self.b)` 陷阱） |
| `notes/03-常见错误对照表.md` | 按报错查原因 |
| **`notes/04-分步实现指南.md`** | **★ 6 步骨架 + 填空，每步附手推过程** |

## 结果

（待填：你自己的训练曲线、决策边界图、你的发现）

