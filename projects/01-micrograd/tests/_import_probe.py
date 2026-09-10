"""诊断脚本：打印 Python 是怎么看待「当前这个文件」的。

分别用两种方式运行，对比输出：
    cd projects/01-micrograd
    .venv\\Scripts\\python.exe tests\\_import_probe.py          # 脚本方式
    .venv\\Scripts\\python.exe -m tests._import_probe           # 模块方式
"""

import sys
from .test_nn import Value
print(f"{'='*64}")
print(f"__name__      = {__name__!r}")
print(f"__package__   = {__package__!r}")
print(f"__file__      = {__file__}")
print(f"{'='*64}")
print()
print("sys.path（Python 找模块时按顺序搜索的目录）:")
for i, p in enumerate(sys.path):
    mark = "  <- 当前工作目录/入口所在目录" if i == 0 else ""
    print(f"  [{i}] {p}{mark}")
print()

if __package__:
    print(f"结论：我被当作【包内模块】加载，父包是 {__package__!r}")
    print("      → 可以用相对导入，例如 from .engine import Value")
else:
    print("结论：我被当作【顶层模块】加载，没有父包")
    print("      → 不能用相对导入，from .engine import Value 会报错")
