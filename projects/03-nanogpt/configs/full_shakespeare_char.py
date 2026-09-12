# 完整训练：英文 char-level（5000 步，官方配置）
#
# 用法（在 projects/02-nanogpt 目录下）：
#   .venv\Scripts\python.exe tools\run_train.py --config configs\full_shakespeare_char.py
#
# 或手动（在 refs/nanoGPT 目录下）：
#   ..\..\.venv\Scripts\python.exe train.py ..\..\configs\full_shakespeare_char.py
#
# 与官方 configs/train_shakespeare_char.py 的差别：
#   out_dir 指到本项目的 outputs/，避免污染 refs/
#   eval_interval 从 250 保持，但每组 eval 用 100 次（官方 200）以省时间

out_dir = '../../outputs/out-shakespeare-char'
eval_interval = 250
eval_iters = 100
log_interval = 50

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'full-run'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
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
