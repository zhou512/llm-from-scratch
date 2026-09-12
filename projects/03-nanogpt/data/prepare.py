"""字符级数据集准备。

把原始文本编码成一串整数（character-level），产出：
    train.bin   训练集 token id（uint16）
    val.bin     验证集 token id（uint16）
    meta.pkl    词表映射（stoi / itos / vocab_size）

用法：
    # 英文（TinyShakespeare）
    .venv\\Scripts\\python.exe data\\prepare.py --out-dir data\\shakespeare_char

    # 中文
    .venv\\Scripts\\python.exe data\\build_zh_corpus.py
    .venv\\Scripts\\python.exe data\\prepare.py --input data\\input_zh.txt --out-dir data\\zh_char

说明：
- 默认读取 data/input.txt（TinyShakespeare，已在仓库里下好）。
- 换成中文语料时，把文本放到 data/input_zh.txt 再传 --input。
- ★ 输出目录必须是 data/<dataset>/，且 <dataset> 与 config 里的 dataset 一致，
  因为 nanoGPT 的 train.py 里写死了 `data_dir = os.path.join('data', dataset)`。
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
    parser.add_argument("--out-dir", default=HERE,
                        help="输出目录。nanoGPT 要求数据在 data/<dataset>/ 下，"
                             "且文件名固定为 train.bin / val.bin / meta.pkl。"
                             "例如 --out-dir data/shakespeare_char")
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

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "train.bin")
    val_path = os.path.join(out_dir, "val.bin")
    meta_path = os.path.join(out_dir, "meta.pkl")

    train_ids.tofile(train_path)
    val_ids.tofile(val_path)
    with open(meta_path, "wb") as f:
        pickle.dump({"vocab_size": vocab_size, "itos": itos, "stoi": stoi}, f)

    print()
    print(f"输出目录: {out_dir}")
    print(f"已写出: {train_path}")
    print(f"已写出: {val_path}")
    print(f"已写出: {meta_path}")
    print()
    print("★ 提醒：nanoGPT 的 train.py 里 `data_dir = os.path.join('data', dataset)`，")
    print("   所以这个目录要放在 data/ 下，且目录名要与 config 里的 dataset 一致。")
    print(f"   例如 dataset = '{os.path.basename(out_dir)}'")

    # 自检：解码前 60 个 token，确认往返正确（中文语料尤其重要）
    sample = "".join(itos[i] for i in train_ids[:60])
    print()
    print(f"往返自检（前 60 个 token 解码）: {sample[:60]!r}")


if __name__ == "__main__":
    main()
