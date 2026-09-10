import random

# 为什么不用 `from .engine import Value`：
# 相对导入要求 nn.py 被当作「包内模块」加载（即 `import src.nn`），
# 但 train.py 是用脚本方式运行的（`python src/train.py`），它把 src/ 加进
# sys.path 后 `from nn import MLP`，此时 nn 是顶层模块、没有父包，
# `.engine` 就会报 "attempted relative import with no known parent package"。
#
# 用 try/except 兼容两种加载方式：
try:
    from .engine import Value          # 包方式：pytest 的 `from src.nn import ...`
except ImportError:                    # noqa: BLE001
    from engine import Value           # 脚本方式：python src/train.py

class Module:

    def zero_grad(self):
        for v in self.parameters():
            v.grad = 0.0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self,nin,nonlin = True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi,xi in zip(self.w,x)),self.b)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self,nin,nout,**kwargs):
        self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]

    def __call__(self,x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return ",".join(str(n) for n in self.neurons)

class MLP(Module):

    def __init__(self,nin,nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i],sz[i+1],nonlin = i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self,x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
            return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
