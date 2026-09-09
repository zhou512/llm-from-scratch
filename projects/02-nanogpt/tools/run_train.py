"""训练启动器：从项目根目录运行 nanoGPT 的 train.py。

为什么需要它：
nanoGPT 的 train.py 用相对路径 `os.path.join('data', dataset)` 找数据，
所以必须在 refs/nanoGPT 目录下运行。直接跑要记得 cd，容易出错。

这个脚本做三件事：
1. 把工作目录切到 refs/nanoGPT
2. 确保 data/<dataset> 存在（不存在就从项目 data/ 软链/复制过去）
3. 透传所有参数给 train.py

用法（在 projects/02-nanogpt 目录下）：
    .venv\\Scripts\\python.exe tools\\run_train.py --config configs\\full_shakespeare_char.py
    .venv\\Scripts\\python.exe tools\\run_train.py --config configs\\full_zh_char.py
    .venv\\Scripts\\python.exe tools\\run_train.py --config configs\\full_zh_char.py --max_iters=100
"""

import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(HERE)
NANOGPT_DIR = os.path.join(PROJ, "refs", "nanoGPT")
PROJ_DATA = os.path.join(PROJ, "data")
RUN_DATA = os.path.join(NANOGPT_DIR, "data")


def ensure_data(dataset):
    """nanoGPT 要在 refs/nanoGPT/data/<dataset>/ 下找 train.bin 等。

    优先建符号链接（省空间），失败则复制（Windows 无管理员权限时）。
    """
    src = os.path.join(PROJ_DATA, dataset)
    dst = os.path.join(RUN_DATA, dataset)
    if not os.path.isdir(src):
        raise SystemExit(
            f"找不到数据目录 {src}\n"
            f"先准备数据：python data\\prepare.py --out-dir data\\{dataset}"
        )
    if os.path.exists(dst):
        return dst
    os.makedirs(RUN_DATA, exist_ok=True)
    try:
        os.symlink(src, dst, target_is_directory=True)
        print(f"已建符号链接: {dst} -> {src}")
    except OSError:
        shutil.copytree(src, dst)
        print(f"已复制数据到: {dst}")
    return dst


def main():
    parser = argparse.ArgumentParser(description="nanoGPT 训练启动器")
    parser.add_argument("--config", required=True, help="配置文件路径（相对项目根目录）")
    # 用 parse_known_args 而不是 REMAINDER：Python 3.13 里 REMAINDER 对
    # --key=value 形式的参数处理不可靠
    args, extra = parser.parse_known_args()

    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        raise SystemExit(f"找不到配置文件: {config_path}")

    # 从配置里读 dataset（简单粗暴地 exec 一下）
    ns = {}
    with open(config_path, encoding="utf-8") as f:
        exec(f.read(), ns)
    dataset = ns.get("dataset", "shakespeare_char")
    out_dir = ns.get("out_dir", "out")

    ensure_data(dataset)

    # ★ Windows 上没有可用的 Triton，torch.compile 会直接报
    #   torch._inductor.exc.TritonMissing。所以自动关掉 compile，
    #   除非用户显式传了 --compile=True。
    if not any(a.startswith("--compile") for a in extra):
        extra.append("--compile=False")
        print("⚠️  自动设置 --compile=False（Windows 上缺 Triton，torch.compile 不可用）")

    # out_dir 是相对 refs/nanoGPT 的，这里算一下绝对路径方便告诉用户
    abs_out = os.path.abspath(os.path.join(NANOGPT_DIR, out_dir))
    print(f"配置文件: {config_path}")
    print(f"数据集:   {dataset}")
    print(f"输出目录: {abs_out}")
    print()

    cmd = [sys.executable, "train.py", config_path] + extra
    print("$ " + " ".join(cmd))
    print()
    result = subprocess.run(cmd, cwd=NANOGPT_DIR)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
