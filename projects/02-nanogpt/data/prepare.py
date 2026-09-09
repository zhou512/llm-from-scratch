"""字符级数据集准备。

把原始文本编码成一串整数（character-level），产出：
    train.bin   训练集 token id（uint16）
    val.bin     验证集 token id（uint16）
    meta.pkl    词表映射（stoi / itos / vocab_size）

用法：
    .venv\\Scripts\\python.exe data\\prepare.py
    .venv\\Scripts\\python.exe data\\prepare.py --input data\\input_zh.txt

说明：
- 默认读取 data/input.txt（TinyShakespeare，已在仓库里下好）。
- 换成中文语料时，把文本放到 data/input_zh.txt 再传 --input。
- ★ 中文处理要点：必须显式用 UTF-8 读写，否则 Windows 默认编码会把中文
  变成乱码，而且不会报错 —— 你会训练出一个只会输出乱码的模型。
"""

import argparse
import os
import pickle

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load_text(path):
    """读文本。UTF-8 优先，失败则尝试 GBK（国内语料常见）。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return raw.decode(enc), enc
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(
        "无法解码文件，请先转成 UTF-8",
        b"", 0, 1, path,
    )


def main():
    parser = argparse.ArgumentParser(description="字符级数据集准备")
    parser.add_argument("--input", default=os.path.join(HERE, "input.txt"),
                        help="原始文本路径（默认 data/input.txt）")
    parser.add_argument("--out-prefix", default="",
                        help="输出文件名前缀，例如 --out-prefix zh_ 会生成 zh_train.bin")
    parser.add_argument("--val-ratio", type=float, default=0.1,
                        help="验证集比例（默认 0.1）")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise SystemExit(
            f"找不到输入文件: {args.input}\n"
            f"提示：先下载 TinyShakespeare：\n"
            f"  curl.exe -sL -o data\\input.txt "
            f"https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt"
        )

    data, enc = load_text(args.input)
    print(f"文件: {args.input}")
    print(f"编码: {enc}")
    print(f"字符总数: {len(data):,}")

    chars = sorted(set(data))
    vocab_size = len(chars)
    print(f"词表大小: {vocab_size}")
    if vocab_size <= 200:
        print(f"词表: {''.join(chars)}")
    else:
        print(f"词表（前 80 个）: {''.join(chars[:80])}")
        print("★ 词表很大说明这是中文/多语言语料，字符级建模的序列会很长")

    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    n = len(data)
    split = int(n * (1.0 - args.val_ratio))
    train_text, val_text = data[:split], data[split:]

    train_ids = np.array([stoi[c] for c in train_text], dtype=np.uint16)
    val_ids = np.array([stoi[c] for c in val_text], dtype=np.uint16)
    print(f"训练 token 数: {len(train_ids):,}")
    print(f"验证 token 数: {len(val_ids):,}")

    train_path = os.path.join(HERE, f"{args.out_prefix}train.bin")
    val_path = os.path.join(HERE, f"{args.out_prefix}val.bin")
    meta_path = os.path.join(HERE, f"{args.out_prefix}meta.pkl")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

    print()
    print(f"已写出: {train_path}")
    print(f"已写出: {val_path}")
    print(f"已写出: {meta_path}")

    # 自检：解码前 60 个 token，确认往返正确（中文语料尤其重要）
    sample = "".join(itos[i] for i in train_ids[:60])
    print()
    print(f"往返自检（前 60 个 token 解码）: {sample[:60]!r}")


if __name__ == "__main__":
    main()
