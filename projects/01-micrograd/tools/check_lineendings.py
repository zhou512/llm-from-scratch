"""检测文件的行尾符（line ending）。

行尾符是「换行」在字节层面的表示：
    LF   = \\n      (0x0A)        Unix / Linux / macOS 用
    CRLF = \\r\\n    (0x0D 0x0A)   Windows 用
    CR   = \\r       (0x0D)        老 Mac 用（已淘汰）

用法：
    .venv\\Scripts\\python.exe tools\\check_lineendings.py            # 检查 git 跟踪的文件
    .venv\\Scripts\\python.exe tools\\check_lineendings.py src        # 只看某个目录
"""

import os
import subprocess
import sys

LF = b"\n"
CRLF = b"\r\n"


def count_line_endings(path):
    with open(path, "rb") as f:
        raw = f.read()
    crlf = raw.count(CRLF)
    lf_total = raw.count(LF)
    lf_only = lf_total - crlf
    cr_only = raw.count(b"\r") - crlf
    return crlf, lf_only, cr_only


def describe(crlf, lf_only, cr_only):
    total = crlf + lf_only + cr_only
    if total == 0:
        return "无换行符"
    if crlf == total:
        return "全 CRLF (Windows)"
    if lf_only == total:
        return "全 LF (Unix)"
    return f"混合！CRLF={crlf} LF={lf_only} CR={cr_only}"


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    here = os.path.dirname(os.path.abspath(__file__))
    proj = os.path.dirname(here)

    # 用 git 列出被跟踪的文件（更贴近「git 关心哪些文件」）
    # ★ 必须显式 encoding="utf-8"：仓库里有中文文件名（如「选题方案.md」），
    #   用默认编码解码 git 的输出会抛 UnicodeDecodeError（'gbk' codec ...）
    try:
        out = subprocess.run(
            ["git", "ls-files"],
            cwd=proj, capture_output=True, text=True, encoding="utf-8", check=True,
        ).stdout
        files = [os.path.join(proj, f) for f in out.splitlines() if f]
    except Exception as e:
        print(f"⚠️  读取 git 文件列表失败: {e}")
        files = []

    if root != ".":
        files = [f for f in files if os.path.abspath(root) in os.path.abspath(f)]
    else:
        files = [f for f in files if f.endswith((".py", ".md", ".txt", ".cfg", ".ps1", ".gitignore"))]

    print(f"检查 {len(files)} 个文件")
    print()
    print(f"{'行尾':22s} {'文件':60s}")
    print("-" * 85)

    counts = {"CRLF": 0, "LF": 0, "混合": 0, "无": 0}
    for path in sorted(files):
        if not os.path.exists(path):
            continue
        rel = os.path.relpath(path, proj)
        crlf, lf_only, cr_only = count_line_endings(path)
        desc = describe(crlf, lf_only, cr_only)
        key = "CRLF" if desc.startswith("全 CRLF") else \
              "LF" if desc.startswith("全 LF") else \
              "无" if desc == "无换行符" else "混合"
        counts[key] += 1
        print(f"{desc:22s} {rel:60s}")

    print()
    print("汇总：", counts)
    if counts["混合"]:
        print("⚠️  有混合行尾的文件，建议统一")
    if counts["CRLF"]:
        print(f"ℹ️  {counts['CRLF']} 个文件是 CRLF（Windows 风格，git 提交时会转成 LF）")


if __name__ == "__main__":
    main()
