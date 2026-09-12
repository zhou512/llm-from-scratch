# 中文语料的字符级 GPT 配置
#
# 用法（在 projects/02-nanogpt/refs/nanoGPT 目录下）：
#   ..\..\.venv\Scripts\python.exe train.py ..\..\configs\train_zh_char.py
#
# 与官方 train_shakespeare_char.py 的区别，以及每处调整的理由：
#
# 1. dataset = 'shakespeare_char' -> 'zh_char'
#    数据目录名，对应 data/zh_char/ 下的 zh_train.bin / zh_val.bin / zh_meta.pkl
#
# 2. batch_size 64 -> 24, gradient_accumulation_steps 1 -> 2
#    中文词表 9214（英文只有 65），lm_head 的输出层大得多，
#    显存占用更高。用小 batch + 梯度累积来凑等效 batch size。
#    等效 batch = 24 * 2 = 48，与官方的 64 接近。
#
# 3. block_size 256 保持不变
#    注意：中文一个字就是一个 token，256 个 token 只覆盖 256 个汉字，
#    而英文 256 个字符覆盖的语义信息更多。想覆盖同样的语义跨度，
#    中文需要更大的 block_size，但那样显存和计算量都涨。
#    这是字符级中文建模的固有限制，也是中文通常用 BPE 的原因之一。
#
# 4. dropout 0.2 保持不变
#    语料只有 1.14M 字符，比 OpenWebText 小得多，容易过拟合，
#    所以保留较高的 dropout。
#
# 5. max_iters 5000 保持不变
#    语料量与 Shakespeare 相当（1.11M vs 1.14M 字符），步数不用改。

out_dir = 'out-zh-char'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'zh-char'
wandb_run_name = 'mini-gpt-zh'

dataset = 'zh_char'
gradient_accumulation_steps = 2
batch_size = 24
block_size = 256

# baby GPT model —— 与官方 char-level 配置相同
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100
