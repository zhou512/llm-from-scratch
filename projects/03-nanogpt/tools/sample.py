"""从训练好的 checkpoint 采样，并对比不同 temperature / top_k。

用法（在 projects/02-nanogpt 目录下）：

    # 英文模型，默认参数
    .venv\\Scripts\\python.exe tools\\sample.py --out-dir outputs\\out-shakespeare-char --dataset shakespeare_char

    # 中文模型
    .venv\\Scripts\\python.exe tools\\sample.py --out-dir outputs\\out-zh-char --dataset zh_char

    # 对比不同 temperature
    .venv\\Scripts\\python.exe tools\\sample.py --out-dir outputs\\out-zh-char --dataset zh_char --sweep-temp

    # 指定开头
    .venv\\Scripts\\python.exe tools\\sample.py --out-dir outputs\\out-zh-char --dataset zh_char --start "春"

为什么要自己写而不是直接用 refs/nanoGPT/sample.py：
- 原脚本要 cd 到 refs/nanoGPT，路径容易搞错
- 原脚本一次只跑一个 temperature，对比实验要改代码
- 这个脚本把结果同时写到文件，方便贴进 README
"""

import argparse
import os
import pickle
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(PROJ, "refs", "nanoGPT"))

from model import GPT, GPTConfig  # noqa: E402


def load_model(out_dir, dataset, device):
    ckpt_path = os.path.join(out_dir, "ckpt.pt")
    if not os.path.exists(ckpt_path):
        raise SystemExit(f"找不到 checkpoint: {ckpt_path}\n先训练：tools\\run_train.py --config ...")

    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint["model_args"])
    model = GPT(gptconf)

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    meta_path = os.path.join(PROJ, "data", dataset, "meta.pkl")
    if not os.path.exists(meta_path):
        raise SystemExit(f"找不到 {meta_path}")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]

    return model, stoi, itos, checkpoint


@torch.no_grad()
def generate(model, x, max_new_tokens, temperature, top_k, device, dtype):
    ctx = torch.amp.autocast(device_type="cuda", dtype=dtype) if device == "cuda" else torch.no_grad()
    with ctx:
        return model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(description="nanoGPT 采样与对比")
    parser.add_argument("--out-dir", required=True, help="含 ckpt.pt 的目录")
    parser.add_argument("--dataset", required=True, help="数据集名（找 meta.pkl）")
    parser.add_argument("--start", default="\n", help="开头文本")
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--sweep-temp", action="store_true",
                        help="对比多组 temperature/top_k")
    parser.add_argument("--out-file", default="", help="结果保存到文件")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    model, stoi, itos, ckpt = load_model(args.out_dir, args.dataset, device)
    n_params = model.get_num_params()
    iter_num = ckpt.get("iter_num", "?")
    best_val = ckpt.get("best_val_loss", float("nan"))

    print(f"模型: {n_params/1e6:.2f}M 参数")
    print(f"训练步数: {iter_num}")
    print(f"最佳 val loss: {best_val:.4f}")
    print(f"词表大小: {len(stoi)}")
    print(f"设备: {device}")
    print()

    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda l: "".join(itos[i] for i in l)

    start_ids = encode(args.start) or [stoi.get("\n", 0)]
    x0 = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    lines = []
    def emit(s=""):
        print(s)
        lines.append(s)

    emit(f"# 采样结果：{args.dataset}")
    emit(f"- 模型参数: {n_params/1e6:.2f}M，训练 {iter_num} 步，best val loss {best_val:.4f}")
    emit(f"- 开头: {args.start!r}，生成长度 {args.max_new_tokens}")
    emit()

    if args.sweep_temp:
        # (temperature, top_k, 说明)
        combos = [
            (0.5, 200, "偏保守，重复多"),
            (0.8, 200, "官方默认"),
            (1.0, 200, "不做温度缩放"),
            (1.2, 200, "更随机"),
            (0.8, 20, "top-k 收窄到 20"),
            (0.8, None, "不限制 top-k"),
        ]
        for temp, top_k, note in combos:
            emit(f"## temperature={temp}, top_k={top_k} （{note}）")
            emit()
            for i in range(min(2, args.num_samples)):
                torch.manual_seed(args.seed + i)
                y = generate(model, x0, args.max_new_tokens, temp, top_k, device, dtype)
                text = decode(y[0].tolist())
                emit(text)
                emit("---")
            emit()
    else:
        for i in range(args.num_samples):
            torch.manual_seed(args.seed + i)
            y = generate(model, x0, args.max_new_tokens, args.temperature, args.top_k, device, dtype)
            text = decode(y[0].tolist())
            emit(f"## 样本 {i+1}（temperature={args.temperature}, top_k={args.top_k}）")
            emit()
            emit(text)
            emit()

    if args.out_file:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_file)), exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        print(f"\n已保存到 {args.out_file}")


if __name__ == "__main__":
    main()
