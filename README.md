# 项目复现记录

复现几个经典的小项目，目标是搞懂原理，而不是跑通别人的代码。

每个项目目录下有 `README.md`（做了什么、怎么跑）、`notes/`（笔记）、
`src/`（自己的实现）、`refs/`（参考实现，只读对照）。

| # | 项目 | 阶段 |
|---|---|---|
| 01 | [micrograd](projects/01-micrograd/) | 完成 |
| 02 | [makemore](projects/02-makemore/) | 进行中 |
| 03 | [nanoGPT](projects/03-nanogpt/) | 阶段 1 完成，阶段 2 未完成 |

## 01 micrograd

从零重写标量自动求导引擎，147 行纯 Python、零依赖。46 个测试全过，
并与 PyTorch autograd 做了梯度对拍。

## 02 makemore

跟课做 bigram 和 MLP 人名生成。修过一个数据表示的问题：`.` 同时被当作填充、
开头、结尾三种标记，`context=[.,.,.]` 有一半的目标是「结束」，生成时一上来就有
50% 概率结束。把三种标记拆开之后，200 次生成里的空名字从 106 个降到 0 个。

## 03 nanoGPT

**阶段 1（读懂档）**：在自己的 4070 上跑字符级 GPT，英文和中文两套语料，
16 组超参对比，实验记录在 `notes/`。

**阶段 2（重写档）**：写了模型结构、`from_pretrained` 的权重转置对拍、
训练循环和采样，没写完。

停下来的原因是阶段 2 和 CS336 作业 1 高度重合：都是自己实现 RoPE / RMSNorm /
SwiGLU / 多头注意力 / AdamW 与学习率调度，再配一套训练和评测。作业那边有官方测试，
做一遍反馈更确定，所以精力转到 [CS336](https://github.com/zhou512/cs336) 上了，
nanoGPT 停在半写状态。

## submodule

`projects/02-makemore/refs/makemore` 是 `karpathy/makemore` 的 submodule，
克隆时需要：

```bash
git clone --recursive <仓库地址>
# 已经克隆过的：
git submodule update --init --recursive
```
