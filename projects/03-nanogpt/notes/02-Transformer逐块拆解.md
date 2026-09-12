# 笔记 02 · Transformer 架构逐块拆解

> 对应文件：`refs/nanoGPT/model.py`（330 行）
> 目标：读完能画出结构图，并解释每个张量的形状怎么变

---

## 一、全景：六个类，三层嵌套

```
GPT                        整个模型
 ├── transformer (ModuleDict)
 │    ├── wte              词嵌入 (vocab_size, n_embd)
 │    ├── wpe              位置嵌入 (block_size, n_embd)
 │    ├── drop             Dropout
 │    ├── h (ModuleList)   n_layer 个 Block  ★ 堆叠
 │    └── ln_f             最后的 LayerNorm
 ├── lm_head                输出投影 (n_embd, vocab_size)
 └── [方法] forward / generate / configure_optimizers / estimate_mfu

Block                      一个 Transformer 层
 ├── ln_1 → attn           注意力子层
 └── ln_2 → mlp            前馈子层

CausalSelfAttention        Block 里的注意力
MLP                        Block 里的前馈网络
LayerNorm                  带可选 bias 的 LayerNorm
```

**这就是 GPT 的全部。** 没有编码器，没有交叉注意力——decoder-only 架构。

---

## 二、Block：两行代码看懂残差 + 预归一化

```python
class Block(nn.Module):
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))     # ★ 先归一化，再注意力，再加回去
        x = x + self.mlp(self.ln_2(x))      # ★ 先归一化，再前馈，再加回去
        return x
```

这两行里有三个关键设计：

| 设计 | 体现在哪 | 为什么 |
|---|---|---|
| **残差连接** | `x = x + ...` | 梯度能直通到浅层，否则深层网络训不动 |
| **预归一化 (Pre-LN)** | `ln(x)` 在子层**之前** | 原版 Transformer 是 Post-LN（`x = ln(x + sub(x))`），训练更不稳定 |
| **归一化在残差之外** | `x + attn(ln(x))` 而不是 `ln(x + attn(x))` | 残差通路保持「干净」，数值更稳 |

**注意**：残差要求输入输出形状完全一致，所以 `attn` 和 `mlp` 的输出维度必须等于 `n_embd`。
这也是为什么 MLP 内部先升维 4 倍再降回来。

---

## 三、注意力：形状追踪（最容易搞混的部分）

### 3.1 输入输出

```python
def forward(self, x):
    B, T, C = x.size()    # batch, 序列长度, 嵌入维度
```

以官方 char-level 配置为例：`B=12, T=256, C=384`。

### 3.2 一次矩阵乘法算出 Q、K、V

```python
self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)   # (384 -> 1152)
q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
```

**为什么是一个 `Linear(384, 1152)` 而不是三个 `Linear(384, 384)`？**
因为一次大矩阵乘法比三次小矩阵乘法更快（GPU 喜欢大 kernel）。
`split` 把结果切成三份。

### 3.3 拆成多头

```python
k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)   # (B, nh, T, hs)
```

形状变化：`(12, 256, 384)` → `(12, 256, 6, 64)` → `(12, 6, 256, 64)`

- `n_head=6`，每个头 `head_size = 384/6 = 64`
- **`view` 是「拆分」，`transpose` 是把头提到 batch 维度上**，让每个头独立做注意力

> **关键理解**：多头不是「重复 6 次」，而是**把 384 维切成 6 个 64 维的子空间，
> 每个子空间学不同的关注模式**。一个头可能学语法，另一个学指代关系。

### 3.4 注意力计算

```python
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# (B, nh, T, hs) @ (B, nh, hs, T) -> (B, nh, T, T)
```

`q @ k^T` 得到 `(12, 6, 256, 256)` 的**注意力分数矩阵**：
第 `i` 行第 `j` 列 = 「位置 i 对位置 j 的关注程度」。

**为什么要除以 `sqrt(head_size)`？**
点积的方差随维度增长。`head_size=64` 时点积的标准差约 8，
不缩放的话 softmax 会变得极端（接近 one-hot），梯度消失。
除以 `sqrt(64)=8` 把方差拉回 1。

### 3.5 因果掩码

```python
att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
att = F.softmax(att, dim=-1)
```

`self.bias` 是下三角矩阵：

```
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

把 0 的位置填成 `-inf`，softmax 后变成 0 概率。**这样位置 i 只能看到 ≤i 的位置**，
这就是「Causal」（因果）的含义——预测下一个词时不能偷看未来。

### 3.6 Flash Attention

```python
if self.flash:
    y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=..., is_causal=True)
else:
    ...手动实现...
```

`is_causal=True` 让 Flash Attention 内部处理掩码，**不用真的构造 `(T,T)` 矩阵**。

**Flash Attention 解决什么问题**：标准注意力要显式存 `(T,T)` 的分数矩阵，
显存 `O(T²)`。Flash Attention 通过分块 + 重计算，把显存降到 `O(T)`，
速度还更快。这是 2022 年的一篇论文（[arXiv:2205.14135](https://arxiv.org/abs/2205.14135)），
**值得你后面单独读**。

> 你的环境是 PyTorch 2.10，`self.flash` 为 True，所以走的是第一条路。

### 3.7 合并多头

```python
y = y.transpose(1, 2).contiguous().view(B, T, C)   # (B,nh,T,hs) -> (B,T,nh*hs) = (B,T,C)
y = self.resid_dropout(self.c_proj(y))             # (B,T,C) -> (B,T,C)
```

`contiguous()` 必须调——`transpose` 后内存不连续，`view` 会报错。
`c_proj` 把多头的结果重新混合。

---

## 四、MLP：为什么是 4 倍

```python
self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)      # 384 -> 1536
self.gelu   = nn.GELU()
self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)      # 1536 -> 384
```

**升维 → 非线性 → 降维**。4 倍是 GPT-2 的选择，后来模型多用 `8/3 ≈ 2.67` 倍（如 LLaMA）。

**为什么需要非线性**：如果只有两个线性层，等价于一个线性层。`GELU` 让网络能拟合非线性关系。

> **MLP 存了模型大部分参数**：以 char-level 配置算，
> 每层 MLP 有 `2 × 384 × 1536 = 1.18M` 参数，注意力有 `4 × 384² = 0.59M`。
> 所以 MLP 参数是注意力的 2 倍。

---

## 五、GPT 主体：嵌入 + 堆叠 + 输出

```python
tok_emb = self.transformer.wte(idx)     # (b, t, n_embd)
pos_emb = self.transformer.wpe(pos)     # (t, n_embd)
x = self.transformer.drop(tok_emb + pos_emb)
```

**位置嵌入是「加」上去的，不是拼接**。这样 token 语义和位置信息在同一个空间里融合。

```python
for block in self.transformer.h:
    x = block(x)
x = self.transformer.ln_f(x)
logits = self.lm_head(x)                # (b, t, vocab_size)
loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
```

`cross_entropy` 要的是 `(N, C)` 和 `(N,)`，所以把 `(b,t,vocab)` 摊成 `(b*t, vocab)`。

### 推理时的优化

```python
logits = self.lm_head(x[:, [-1], :])    # 只算最后一个位置
```

**注意 `x[:, [-1], :]` 而不是 `x[:, -1, :]`**：
前者保持时间维度 `(b, 1, n_embd)`，后者会降维成 `(b, n_embd)`。
注释里特意写了这点，是个易错处。

---

## 六、三个容易忽略但重要的细节

### 6.1 权重共享（Weight Tying）

```python
self.transformer.wte.weight = self.lm_head.weight
```

输入嵌入 `(vocab, n_embd)` 和输出投影 `(n_embd, vocab)` **共用同一份权重**。

**好处**：省参数（char-level 配置省 65×384≈25K，中文版省 9214×384≈3.5M），
且让「输入向量」和「输出向量」在同一空间里，语义更一致。

> **回答 README 里的思考题**：中文版比英文版多的 3.5M 参数，
> 主要来自 `wte`/`lm_head`（词表从 65 涨到 9214，多出 `9149 × 384 ≈ 3.5M`）。

### 6.2 残差投影的特殊初始化

```python
for pn, p in self.named_parameters():
    if pn.endswith('c_proj.weight'):
        torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))
```

**为什么残差投影要缩小初始化**：残差是 `x + sub(x)`，
深层堆叠时每层的输出都会累加到 `x` 上，方差会累积增长。
除以 `sqrt(2*n_layer)` 让每层的贡献随深度衰减，保持数值稳定。
这是 GPT-2 论文里的做法。

### 6.3 权重衰减分组

```python
decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
```

**2 维以上（矩阵）做权重衰减，1 维（bias、LayerNorm 参数）不做。**

原因：权重衰减是为了抑制权重过大，而 bias 和 LayerNorm 的缩放参数
本身就需要自由调整，衰减它们反而有害。

实测输出：
```
num decayed parameter tensors: 26, with 10,740,096 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters
```

---

## 七、参数量速算

以 `n_layer=6, n_head=6, n_embd=384, vocab=65` 为例：

| 组件 | 公式 | 参数量 |
|---|---|---|
| 词嵌入 wte | vocab × n_embd | 65 × 384 = 24,960 |
| 位置嵌入 wpe | block_size × n_embd | 256 × 384 = 98,304 |
| 每层注意力 | 4 × n_embd² (+bias) | 4 × 384² ≈ 590K |
| 每层 MLP | 8 × n_embd² (+bias) | 8 × 384² ≈ 1.18M |
| 每层 LayerNorm | 4 × n_embd | 1,536 |
| 输出 lm_head | 与 wte 共享 | 0 |
| **每层小计** | | **≈ 1.77M** |
| **6 层合计** | | **≈ 10.6M** |

实测输出 `number of parameters: 10.65M` ✅ 对得上。

> 中文版词表 9214：`wte` 从 24,960 涨到 `9214 × 384 ≈ 3.54M`，总参数 14.2M。

---

## 八、MFU：一个能加分的指标

```python
flops_per_token = 6*N + 12*L*H*Q*T
```

- `6N`：前向 2N + 反向 4N（每个参数每次前向 1 次乘加 = 2 FLOPs）
- `12*L*H*Q*T`：注意力部分的额外开销（`H*Q = n_embd`，所以是 `12*L*n_embd*T`）

`mfu = 实际算力 / A100 峰值算力(312 TFLOPS)`。

你实测的 MFU 只有 **2-3%**，原因：
- batch 小（12），GPU 没吃满
- 序列短（256），矩阵乘法规模小
- 笔记本 GPU 的实际可用算力远低于 A100
- kernel 启动开销占比高

**面试时主动说这个**，比说「我训了个 GPT」有说服力得多。

---

## 九、生成：temperature 与 top-k

```python
logits = logits[:, -1, :] / temperature      # ① 温度缩放
if top_k is not None:                        # ② top-k 截断
    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
    logits[logits < v[:, [-1]]] = -float('Inf')
probs = F.softmax(logits, dim=-1)            # ③ 转概率
idx_next = torch.multinomial(probs, 1)       # ④ 采样
```

| 参数 | 效果 |
|---|---|
| `temperature < 1` | 分布更尖锐，输出更确定、更保守 |
| `temperature > 1` | 分布更平坦，输出更随机、更有创意 |
| `temperature → 0` | 退化成 argmax（贪心解码） |
| `top_k=50` | 只从前 50 个候选中采样，避免抽到极低概率的怪词 |

**自己动手试**：同一个 prompt 用不同 temperature 生成，对比差异。这比看教程直观得多。

---

## 十、你的任务

1. **画图**：把 `GPT → Block → Attention/MLP` 的结构和每层张量形状画出来
2. **跑冒烟测试**：`.venv\Scripts\python.exe refs\smoke_test.py`
3. **读 train.py**：重点看 `get_batch`、`estimate_loss`、学习率调度
4. **回答这三个问题**（写在笔记里）：
   - 为什么 `q @ k^T` 要除以 `sqrt(head_size)`？
   - 权重共享省了多少参数？
   - 为什么残差投影的初始化标准差要除以 `sqrt(2*n_layer)`？

---

## 十一、待你补充

- [ ] 你画的结构图
- [ ] temperature 实验对比
- [ ] `n_layer` 从 6 改成 3，参数量和 loss 怎么变？
- [ ] 如果去掉位置嵌入，模型会怎样？（试一下）
