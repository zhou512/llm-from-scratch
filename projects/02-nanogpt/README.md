# 02 · nanoGPT 复现：从零训练一个 GPT

## 这是什么

复现 Andrej Karpathy 的 [nanoGPT](https://github.com/karpathy/nanoGPT)：
用 **599 行代码**（`model.py` 290 + `train.py` 309）从零训练一个 GPT。

目标不是训出 SOTA，而是**彻底搞懂 Transformer 的每一块**，
并在自己的显卡上跑出真实结果。

## ★ 「复现」有两种强度，本项目分两阶段

「复现」这个词很含糊。跑通别人的代码和自己写出这段代码，学到的东西**完全不同**。
所以本项目明确分成两档：

| | **阶段 1：读懂档** | **阶段 2：重写档** |
|---|---|---|
| 做什么 | 逐块读懂 `refs/nanoGPT/model.py`，改配置/换语料做实验 | **合上参考代码**，自己写出 `src/model.py` |
| 学到 | Transformer 架构直觉、实验设计、工程流程 | 真正的原理掌握——能独立实现才叫懂 |
| 工作量 | 约 10 小时 | 约 15-20 小时 |
| 验证方式 | 测试全绿 + 训练 loss 下降 | **与官方预训练权重对拍**（见下） |
| 状态 | 🚧 进行中 | ⏳ 待开始 |

> **对比项目 01**：micrograd 是**从零重写**（147 行全自己写）。
> 本项目如果只做阶段 1，强度低于项目 01——所以阶段 2 不是可选项，是补齐。

### 前置：先看 Karpathy 的 nn-zero-to-hero

阶段 2 之前建议先看完这几讲（详见 `notes/04-课程与学习路径.md`）：

| 讲次 | 内容 | 为什么需要 |
|---|---|---|
| 1 | micrograd | ✅ 已完成（项目 01） |
| **2-3** | makemore：bigram → MLP | 从「一个神经元」跨到「语言模型」；讲 embedding、训练/验证集划分 |
| **4** | makemore：Activations & Gradients, BatchNorm | 初始化为什么重要、梯度为何爆炸/消失——目前的空白 |
| **5** | **Being a Backprop Ninja** | ★ 手动推导并实现 `Linear`/`LayerNorm`/`softmax` 的反向传播。micrograd 只实现过标量算子，这一讲补上缺的那部分 |
| 7 | Let's build GPT | 内容 = 本项目的 `model.py`，跟着手敲一遍是从「读懂」到「会写」的捷径 |
| 8 | Let's build the GPT Tokenizer | 内容 = minbpe，做中文 BPE 版时有用 |

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
│   │   ├── model.py             290 行，GPT 模型定义  ← 阶段 2 要自己重写的就是它
│   │   ├── train.py             309 行，训练循环
│   │   ├── sample.py            82 行，采样生成
│   │   ├── configurator.py      42 行，配置覆盖机制
│   │   └── bench.py             106 行，性能基准
│   └── smoke_test.py            ★ 显存/参数量冒烟测试（本仓库新增）
├── src/                         ★ 阶段 2 自己的实现（目前为空）
├── configs/
│   ├── train_shakespeare_char.py  官方 char-level 配置
│   ├── train_zh_char.py           中文配置
│   ├── full_shakespeare_char.py   完整 5000 步训练配置
│   └── full_zh_char.py            中文完整训练配置
├── data/
│   ├── prepare.py               ★ 数据准备（本仓库重写，支持中文）
│   ├── build_zh_corpus.py       ★ 中文语料构建（本仓库新增）
│   ├── input.txt                TinyShakespeare（1.1M 字符）
│   └── input_zh.txt             中文语料（1.24M 字符，唐诗宋词等）
├── experiments/
│   └── run_experiments.py       ★ 超参对比实验（控制变量法）
├── tools/
│   ├── run_train.py             ★ 训练启动器（处理工作目录与路径）
│   └── sample.py                ★ 采样与 temperature/top_k 对比
└── notes/
    ├── 01-环境与网络实录.md        ★ 环境配置 + 网络绕行 + 实测数据
    ├── 02-Transformer逐块拆解.md   ★ 330 行 model.py 的逐块讲解
    ├── 03-超参对比实验分析.md      ★ 16 组实验的结果与解读
    └── 04-课程与学习路径.md        ★ 课程选择与阶段划分（本文件）
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

⚠️ **Windows 上没有可用的 Triton**，`torch.compile` 会抛 `TritonMissing`。
`tools/run_train.py` 会自动加 `--compile=False` 处理这个问题。

## 怎么跑

```bash
# 1. 准备数据（英文）
.venv\Scripts\python.exe data\prepare.py --out-dir data\shakespeare_char

# 2. 准备数据（中文）
.venv\Scripts\python.exe data\build_zh_corpus.py
.venv\Scripts\python.exe data\prepare.py --input data\input_zh.txt --out-dir data\zh_char

# 3. 先做冒烟测试，确认显存够
.venv\Scripts\python.exe refs\smoke_test.py

# 4. 完整训练（启动器会处理工作目录、数据软链、--compile=False）
.venv\Scripts\python.exe tools\run_train.py --config configs\full_shakespeare_char.py
.venv\Scripts\python.exe tools\run_train.py --config configs\full_zh_char.py

# 5. 采样（含 temperature/top_k 对比）
.venv\Scripts\python.exe tools\sample.py --out-dir outputs\out-zh-char --dataset zh_char --sweep-temp

# 6. 超参对比实验
.venv\Scripts\python.exe experiments\run_experiments.py --max-iters 300
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
中文版 15 步：loss 9.1963 → 7.1368，MFU 5.43%
```

按 ~150ms/步估算，官方配置的 5000 步约需 **12-15 分钟**（A100 上约 3 分钟）。

## 验收标准

### 阶段 1（读懂档）

- [ ] 能画出 Transformer 结构图并解释每个张量形状
- [ ] `smoke_test.py` 三个配置全绿
- [ ] 英文 char-level 训练跑完，loss 降到 ~1.5
- [ ] 采样出像样的莎士比亚风格文本
- [ ] 中文语料训练跑完，能生成通顺的中文片段
- [ ] 有一张实验对比表（不同层数/宽度/block_size）

### 阶段 2（重写档）

- [ ] **合上 `refs/nanoGPT/model.py`**，自己写出 `src/model.py`
- [ ] 自己写的模型能加载官方 GPT-2 预训练权重（见下面的验证方法）
- [ ] 用**自己的 `model.py`** 复现出与 `refs/` 版本一致的训练曲线
- [ ] 在 `notes/` 里写一篇《我实现 GPT 时踩的坑》

## 阶段 2 的验证方法：与官方权重对拍

这是最硬的验证方式——**如果能加载官方 GPT-2 预训练权重并得到相同输出，
说明你的架构和官方完全一致**（任何形状、顺序、命名错误都会暴露）。

`refs/nanoGPT/model.py` 里的 `GPT.from_pretrained()` 就是干这个的，核心步骤：

```python
# 1. 从 HuggingFace 加载官方 GPT-2 权重
model_hf = GPT2LMHeadModel.from_pretrained('gpt2')
sd_hf = model_hf.state_dict()

# 2. 逐参数复制到自己的模型里
for k in sd_keys_hf:
    if any(k.endswith(w) for w in transposed):
        # ★ OpenAI 的 checkpoint 用的是 Conv1D 模块，权重矩阵是转置的
        #   所以这几个权重必须先 .t() 转置再复制
        assert sd_hf[k].shape[::-1] == sd[k].shape
        sd[k].copy_(sd_hf[k].t())
    else:
        assert sd_hf[k].shape == sd[k].shape     # ★ 形状必须完全一致
        sd[k].copy_(sd_hf[k])

# 3. 还要跳过两个 buffer（不是参数）
#    .attn.masked_bias / .attn.bias 是因果掩码，不算参数
```

**需要转置的四个权重**：
```python
transposed = ['attn.c_attn.weight', 'attn.c_proj.weight',
              'mlp.c_fc.weight', 'mlp.c_proj.weight']
```

**验证脚本思路**：

```python
# 同一个输入，两个模型输出应该完全一致
import torch
from model import GPT, GPTConfig          # 你的实现
my_model = GPT.from_pretrained('gpt2')

from nanoGPT.model import GPT as RefGPT   # 参考实现
ref_model = RefGPT.from_pretrained('gpt2')

idx = torch.randint(0, 50257, (1, 32))
logits_mine, _ = my_model(idx)
logits_ref, _ = ref_model(idx)
print((logits_mine - logits_ref).abs().max())   # 期望接近 0（浮点误差级）
```

**前置条件**：加载 GPT-2 权重需要 `transformers` 库和下载权重。
你的网络下 `huggingface.co` 不通，需要用镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv pip install --python .venv --index-url https://pypi.tuna.tsinghua.edu.cn/simple transformers
```

> GPT-2 有 124M 参数，下载约 500MB。**不要用它训练**（官方要 8×A100），
> 只用它做**权重加载对拍**——这是最省时且最有说服力的验证。

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

### 阶段 1（读懂档）

**已完成的部分**（骨架与实测，代码来自参考实现）：

- ✅ 环境与网络问题全部摸清（`notes/01-环境与网络实录.md`）
- ✅ 中文语料构建：1,242,970 字符，词表 9214（安全落在 uint16 范围内）
- ✅ 显存与速度实测：官方配置 10.7M 参数 / 0.37GB；中文版 14.2M 参数 / 0.65GB
- ✅ 超参对比实验 16 组（中英文各 8 组，`notes/03-超参对比实验分析.md`）
- ✅ 修复了中文配置的编码问题（`configurator.py` 用 GBK 读配置文件会崩）
- ✅ 训练/采样启动器与实验脚本（`tools/`、`experiments/`）

**待你完成的部分**：

- [ ] 通读 `notes/02-Transformer逐块拆解.md`，对照 `refs/nanoGPT/model.py`
- [ ] 跑完整 5000 步训练（英文 + 中文），贴出 loss 曲线
- [ ] 用 `tools/sample.py --sweep-temp` 对比不同 temperature 的生成质量
- [ ] 在下面写上你自己的观察

（待填：你自己的 loss 曲线、生成样本、你的发现）

### 阶段 2（重写档）

（待开始：自己写出 `src/model.py`，用官方 GPT-2 权重对拍验证）

---

## 学习路径索引

| 想看什么 | 去哪 |
|---|---|
| 环境/网络踩的坑 | `notes/01-环境与网络实录.md` |
| Transformer 每一块在干什么 | `notes/02-Transformer逐块拆解.md` |
| 16 组实验的结果与解读 | `notes/03-超参对比实验分析.md` |
| **项目该做到什么程度 / 看哪些课程** | **`notes/04-课程与学习路径.md`** |
