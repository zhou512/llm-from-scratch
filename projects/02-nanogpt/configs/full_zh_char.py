# 完整训练：中文 char-level（5000 步）
#
# 用法（在 projects/02-nanogpt 目录下）：
#   .venv\Scripts\python.exe tools\run_train.py --config configs\full_zh_char.py
#
# 与 configs/train_zh_char.py 的差别：out_dir 指到本项目的 outputs/

out_dir = '../../outputs/out-zh-char'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'zh-char'
wandb_run_name = 'full-run'

dataset = 'zh_char'
gradient_accumulation_steps = 2
batch_size = 24
block_size = 256

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
