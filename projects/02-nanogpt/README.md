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

## 超参对比实验（已跑通，8 组）

脚本：`experiments/run_experiments.py`（控制变量法，每组只改一个变量）

| 实验 | 参数量(M) | val loss | ms/iter | 峰值显存(GB) | MFU(%) |
|---|---|---|---|---|---|
| baseline | 10.67 | 2.5082 | 73.8 | 1.40 | 2.53 |
| **n_layer=3** | 5.35 | **2.4600** | 37.8 | 0.76 | 2.47 |
| n_layer=12 | 21.32 | 2.9728 | 147.6 | 2.68 | 2.53 |
| **n_embd=192** | 2.68 | **2.4447** | 27.0 | 0.68 | 1.91 |
| n_embd=768 | 42.58 | 2.5595 | 254.3 | 3.00 | 2.78 |
| block_size=128 | 10.67 | 2.4824 | 39.0 | 0.77 | 2.27 |
| no_dropout | 10.67 | 2.5160 | 77.1 | 1.35 | 2.42 |
| dropout=0.4 | 10.67 | 2.4769 | 79.7 | 1.40 | 2.34 |

**关键发现**（详细分析见 `notes/03-超参对比实验分析.md`）：

1. **更小的模型反而更好**：n_embd=192（2.68M 参数）的 val loss 最低（2.4447），
   比 42.58M 参数的 n_embd=768（2.5595）好得多 —— 数据集太小，大模型容量过剩且收敛慢
2. **层数翻倍反而变差**：12 层的 val loss（2.9728）比 6 层差 0.46，
   因为同样 300 步喂不饱更深的模型
3. **block_size 减半影响极小**：128 vs 256 的 loss 几乎相同，
   但显存减半、速度快一倍 → **显存不够时优先砍 block_size**
4. **loss 不能跨速度直接比较**：n_embd=192 单步快 2.7 倍，
   固定步数比较和固定时间比较会得出不同结论

⚠️ **实验局限**（诚实说明）：只训 300 步（官方是 5000）、单次运行无多种子平均、
未做学习率搜索。所以"小模型更好"的结论**只在这个训练预算下成立**。

### 中文语料的同组实验

同样的 8 组配置跑中文语料（`--dataset zh_char`），**规律完全一致**：

| 实验 | 参数量(M) | val loss | ms/iter | 峰值显存(GB) |
|---|---|---|---|---|
| baseline | 14.19 | 6.9603 | 96.2 | 2.49 |
| **n_layer=3** | 8.86 | **6.7920** | 60.9 | 1.85 |
| n_layer=12 | 24.83 | 7.0506 | 178.6 | 3.77 |
| **n_embd=192** | 4.44 | **6.7194** | 43.9 | 1.79 |
| n_embd=768 | 49.61 | 6.9851 | 307.7 | 4.08 |
| block_size=128 | 14.19 | 6.9122 | 52.3 | 1.34 |
| no_dropout | 14.19 | 6.8866 | 102.8 | 2.45 |
| dropout=0.4 | 14.19 | 6.8146 | 104.5 | 2.32 |

**中英对比**：最优配置都是 `n_embd=192`，最差都是 `n_embd=768`，
说明"小数据集 + 短训练 → 小模型更好"是普适规律。

**但中文 loss 数值高得多**（6.96 vs 2.51），这**不代表中文更难学**：
词表从 65 涨到 9214，随机猜对概率低得多。换算成 bits-per-character：
```
英文: 2.5082 / ln(2) = 3.62 bpc
中文: 6.9603 / ln(2) = 10.04 bpc
```
即使换算后中文仍高——因为**一个汉字承载的信息量远大于一个英文字母**，
这是字符级建模对中文不公平的地方，也是中文通常用 BPE 分词的原因之一。

## 结果

（待填：你自己的 loss 曲线、生成样本、1000 步的补充实验）
