"""构建中文语料（唐诗宋词等公共领域文本）-> data/input_zh.txt

为什么不用现成的大语料：
- huggingface.co 在你这台机器上连不上（实测超时）
- 中文公开语料体积动辄几十 GB，对 8GB 显存的字符级训练毫无必要

唐诗宋词体量刚好（约 2-3M 字符），风格统一，适合做字符级语言模型的实验对象。

用法：
    .venv\\Scripts\\python.exe data\\build_zh_corpus.py
    .venv\\Scripts\\python.exe data\\build_zh_corpus.py --out data\\input_zh.txt
"""

import argparse
import json
import os
import urllib.parse
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = "https://cdn.jsdelivr.net/gh/chinese-poetry/chinese-poetry@master"

# (路径, 说明) —— 用 jsdelivr CDN，实测在你这台机器上可访问
SOURCES = [
    ("全唐诗/poet.tang.0.json", "全唐诗 000"),
    ("全唐诗/poet.tang.1000.json", "全唐诗 1000"),
    ("全唐诗/poet.tang.2000.json", "全唐诗 2000"),
    ("全唐诗/poet.tang.3000.json", "全唐诗 3000"),
    ("全唐诗/poet.tang.10000.json", "全唐诗 10000"),
    ("全唐诗/poet.tang.50000.json", "全唐诗 50000"),
    ("全唐诗/poet.tang.56000.json", "全唐诗 56000"),
    ("全唐诗/poet.tang.57000.json", "全唐诗 57000"),
    ("宋词/ci.song.0.json", "宋词 0"),
    ("宋词/ci.song.1000.json", "宋词 1000"),
    ("宋词/ci.song.2000.json", "宋词 2000"),
    ("宋词/ci.song.9000.json", "宋词 9000"),
    ("诗经/shijing.json", "诗经"),
    ("楚辞/chuci.json", "楚辞"),
    ("论语/lunyu.json", "论语"),
    ("五代诗词/huajianji/huajianji-1-juan.json", "花间集"),
    ("蒙学/baijiaxing.json", "百家姓"),
    ("蒙学/qianziwen.json", "千字文"),
]


def fetch(url, timeout=60):
    # URL 里的中文路径必须先 percent-encode，否则 urllib 会报
    # UnicodeEncodeError: 'ascii' codec can't encode characters
    url = urllib.parse.quote(url, safe=":/?&=@#%[]!$'()*+,;")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def parse_records(raw):
    """不同文件的字段名不完全一样，统一抽成 (标题, 作者, 正文) 文本。"""
    data = json.loads(raw.decode("utf-8"))
    out = []

    if isinstance(data, dict):
        data = [data]

    for item in data:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or item.get("rhythmic") or item.get("chapter") or ""
        author = item.get("author") or ""
        paras = item.get("paragraphs") or item.get("content") or item.get("para") or []
        if isinstance(paras, str):
            paras = [paras]
        body = "\n".join(str(p) for p in paras if p)
        if not body:
            continue
        head = "".join(x for x in (title, author) if x)
        out.append(f"{head}\n{body}" if head else body)
    return out


def main():
    parser = argparse.ArgumentParser(description="构建中文字符级语料")
    parser.add_argument("--out", default=os.path.join(HERE, "input_zh.txt"))
    parser.add_argument("--min-chars", type=int, default=1_000_000,
                        help="少于这个字符数就警告")
    args = parser.parse_args()

    chunks = []
    for path, desc in SOURCES:
        url = f"{BASE}/{path}"
        try:
            raw = fetch(url)
            texts = parse_records(raw)
            chars = sum(len(t) for t in texts)
            chunks.extend(texts)
            print(f"[OK]   {desc:16s} {len(texts):5d} 篇  {chars:>9,} 字符")
        except Exception as e:
            print(f"[跳过] {desc:16s} 失败: {type(e).__name__}: {e}")

    if not chunks:
        raise SystemExit("所有数据源都失败了，检查网络")

    corpus = "\n\n".join(chunks)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(corpus)

    vocab = sorted(set(corpus))
    print()
    print(f"总字符数: {len(corpus):,}")
    print(f"词表大小: {len(vocab):,}")
    print(f"输出文件: {args.out}  ({os.path.getsize(args.out)/1024:.0f} KB)")

    # ★ 关键检查：nanoGPT 用 uint16 存 token id，词表不能超过 65535
    if len(vocab) > 65535:
        print(f"⚠️  词表超过 65535，train.bin 的 uint16 会溢出！需要改用 uint32")
    else:
        print(f"✅ 词表 {len(vocab)} < 65536，可以直接用 nanoGPT 默认的 uint16 格式")

    if len(corpus) < args.min_chars:
        print(f"⚠️  语料只有 {len(corpus):,} 字符，少于 {args.min_chars:,}，"
              f"训练可能欠拟合（可以多加几个数据源）")

    print()
    print("预览:")
    print(corpus[:200])


if __name__ == "__main__":
    main()
