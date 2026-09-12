# 我的开源项目复现记录

一名大三学生的项目复现仓库。目标：**搞懂原理，而不是跑通别人的代码**。

每个项目目录下都有：

- `README.md` —— 做了什么、怎么跑、结果如何
- `notes/` —— 学习笔记（比代码本身更重要）
- `src/` —— 我自己的实现
- `refs/` —— 参考的原始代码（用于对照，不修改）

## 项目列表

按 [Andrej Karpathy 的 nn-zero-to-hero](https://github.com/karpathy/nn-zero-to-hero) 课程顺序推进：

| # | 项目 | 状态 | 说明 |
|---|---|---|---|
| 01 | [micrograd 复现](projects/01-micrograd/) | ✅ 已完成 | **从零重写**标量级自动求导引擎，147 行纯 Python、零依赖；46 个测试 + 与 PyTorch autograd 对拍 + 数值微分三重验证 |
| 02 | [makemore 练习](projects/02-makemore/) | 🚧 进行中 | 课程第二、三讲练习（bigram / MLP 人名生成）。官方代码以 **git submodule** 引入，自己的实现和笔记由本仓库跟踪 |
| 03 | [nanoGPT 复现](projects/03-nanogpt/) | 🚧 阶段 1 | 在 RTX 4070 8GB 上训练字符级 GPT，含中文语料版与 16 组超参实验 |

## 「复现」的强度不一样

三个项目的**学习强度不同**，每个项目的 README 里都写明了阶段划分：

| 强度 | 做什么 | 对应项目 |
|---|---|---|
| A. 跑通 | 装环境、跑训练 | — |
| B. 读懂 + 改造 | 逐块读懂、改配置做实验 | makemore 练习；nanoGPT **阶段 1** |
| C. 从零重写 | 合上参考代码自己实现 | micrograd；nanoGPT **阶段 2**（规划中） |

判断标准：**把参考代码删掉，你还能自己写出来吗？**

## 项目 01 的成果

| | |
|---|---|
| 实现 | `src/engine.py`（83 行）+ `src/nn.py`（64 行） |
| 验证 | 6 步分步检查全过、46 个测试全绿、与 PyTorch autograd 8 项梯度对拍一致、数值微分误差 < 1e-4 |
| 结果 | moons 二分类 loss 0.94 → 0.25，准确率 86% |
| 笔记 | 6 篇（原理拆解、常见错误、分步指南、训练原理、资料清单） |

## 项目 02 发现的问题

跟课写 MLP 人名生成器时，输出里出现大量空名字和 `..`。排查结果：

> **`.` 这个符号同时充当了填充、开头标记、结尾标记三个角色。**
> `context=[.,.,.]` 在训练数据里有 50% 的目标是「结尾」，
> 于是生成时一上来就有一半概率直接结束。

修复方案是把三种标记分开（`<P>` / `<S>` / `<E>`），实测 P(结束) 从 0.5115 降到 0.0000，
200 次生成中的短名字从 106 个降到 0 个。完整分析见
[`projects/02-makemore/notes/01-为什么生成空名字.md`](projects/02-makemore/notes/01-为什么生成空名字.md)。

**教训**：这不是模型调参问题，是数据表示问题——一个符号承担多个含义时，
模型无法区分这些语境，只能学到混合统计。

## submodule 的用法

`projects/02-makemore/refs/makemore` 是指向 `karpathy/makemore` 的 submodule。
克隆本仓库时需要加上 `--recursive`：

```bash
git clone --recursive <本仓库地址>
# 已经克隆过的话：
git submodule update --init --recursive
```

为什么放在 `refs/` 下：submodule 内部的文件既不属于主仓库、也不会被推送到上游。
放在 `refs/` 里，官方代码当只读参照，自己写的代码和笔记由主仓库跟踪——
与 `projects/01-micrograd/refs/` 的模式一致。

## git 练习材料

`git-practice/` 下有给 git 新手用的材料：

- `git-cheatsheet.md` —— 速查卡（三区模型、撤销决策表、reflog 读法）
- `04-推送到GitHub与PR流程.md` —— 推送与 PR 完整流程
- `05-PR流程本地演练.md` + `local-remote/demo-pr.ps1` —— 不需要网络就能练 PR
- `demo-repo/` —— 一个独立的练习仓库

## 为什么做这些

- 反向传播、Transformer 这些核心概念，**光看论文和教程是学不会的**，必须亲手写一遍
- 复现过程本身就是最好的工程训练：环境、调试、实验记录、版本管理
- 全部代码和笔记公开，作为能力的直接证据
