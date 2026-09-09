# 02 · nanoGPT 复现：从零训练一个 GPT

## 这是什么

复现 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)：
用 **599 行代码**（`model.py` 290 + `train.py` 309）从零训练一个 GPT。

目标不是训出 SOTA，而是**彻底搞懂 Transformer 的每一块**，
并在自己的显卡上跑出真实结果。

## 为什么选它（以及它的局限）

| | |
|---|---|
| ✅ 代码量小到能全部读懂 | 599 行，不是 10 万行 |
| ✅ 就业相关性高 | LLM 是当下最热方向 |
| ✅ 你的硬件能跑 | 实测：官方 char-level 配置 10.7M 参数、峰值显存 0.37 GB |
| ⚠️ 已被标记 deprecated | 作者 2025-11 推荐看 [nanochat](https://github.com/karpathy/nanochat)，但 nanochat 对你的 8GB 显存太重 |
| ❌ 不要从头训 GPT-2 124M | 官方要 **8×A100 40GB 跑 4 天** |

## 目录结构

```
02-nanogpt/
├── README.md                    本文件
├── refs/
│   ├── nanoGPT/                 ★ 原始实现快照（只读对照）
│   │   ├── model.py             290 行，GPT 模型定义
│   │   ├── train.py             309 行，训练循环
│   │   ├── sample.py            82 行，采样生成
│   │   ├── configurator.py      42 行，配置覆盖机制
│   │   └── bench.py             106 行，性能基准
│   └── smoke_test.py            ★ 显存/参数量冒烟测试（本仓库新增）
├── src/                         ★ 自己的实现（待写）
├── configs/
│   └── train_shakespeare_char.py  官方 char-level 配置
├── data/
│   ├── prepare.py               ★ 数据准备（本仓库重写，支持中文）
│   ├── build_zh_corpus.py       ★ 中文语料构建（本仓库新增）
│   ├── input.txt                TinyShakespeare（1.1M 字符）
│   └── input_zh.txt             中文语料（1.24M 字符，唐诗宋词等）
└── notes/
    └── 01-环境与网络实录.md        ★ 环境配置 + 网络绕行 + 实测数据
```

## 环境（已在本机验证）

```bash
cd D:\git-reproduce\projects\02-nanogpt
uv venv .venv --system-site-packages    # 复用系统已装的 torch，省 2.5GB 下载
uv pip install --python .venv --index-url https://pypi.tuna.tsinghua.edu.cn/simple tiktoken
```

实测可用：
```
torch 2.10.0+cu126  |  cuda True  |  RTX 4070 Laptop 8GB
numpy 2.2.4  |  tiktoken 0.14.0
```

## 怎么跑

```bash
# 1. 准备数据（英文）
.venv\Scripts\python.exe data\prepare.py

# 2. 准备数据（中文）
.venv\Scripts\python.exe data\build_zh_corpus.py
.venv\Scripts\python.exe data\prepare.py --input data\input_zh.txt --out-prefix zh_

# 3. 先做冒烟测试，确认显存够
.venv\Scripts\python.exe refs\smoke_test.py

# 4. 训练（英文 char-level）
cd refs\nanoGPT
..\..\.venv\Scripts\python.exe train.py ..\..\configs\train_shakespeare_char.py
```

## 实测数据（本机 RTX 4070 8GB）

### 模型规模与显存

| 配置 | 参数量 | 峰值显存 | 状态 |
|---|---|---|---|
| tiny（2 层 64 维） | 0.10M | 0.03 GB | ✅ |
| **shakespeare-char**（6 层 384 维，词表 65） | **10.7M** | **0.37 GB** | ✅ |
| **shakespeare-char-zh**（6 层 384 维，词表 9214） | **14.2M** | **0.65 GB** | ✅ |

> 冒烟测试用 batch_size=8。实际训练 batch_size=64 会涨到约 3GB，仍在 8GB 内。

### 训练速度

```
20 步实测：train loss 4.2853 → 2.7250，val loss 4.2743 → 2.7017
单步耗时：98-514 ms（含 eval 的步更慢）
```

按 ~150ms/步估算，官方配置的 5000 步约需 **12-15 分钟**（A100 上约 3 分钟）。

## 验收标准

- [ ] 能画出 Transformer 结构图并解释每个张量形状
- [ ] `smoke_test.py` 三个配置全绿
- [ ] 英文 char-level 训练跑完，loss 降到 ~1.5
- [ ] 采样出像样的莎士比亚风格文本
- [ ] 中文语料训练跑完，能生成通顺的中文片段
- [ ] 有一张实验对比表（不同层数/宽度/block_size）

## 结果

（待填：loss 曲线、生成样本、实验对比表）
