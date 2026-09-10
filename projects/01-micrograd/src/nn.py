import random

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
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]   #一个列表，存放[0~1的权值]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi * xi for wi,xi in zip(self.w,x)),self.b)  # act = b + x0w0 + w1x1 +...
        return act.relu() if self.nonlin else act   # nonlin 决定是否激活函数

    def parameters(self):
        return self.w + [self.b]  #返回一个列表,包含权值和初始值

    def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self,nin,nout,**kwargs):
        self.neurons = [Neuron(nin,**kwargs) for _ in range(nout)]  #一个存储nin维神经元的列表，共有nout层, **kwargs这里指nonlin参数的值, like:Layer(3,4,nonlin = True)

    def __call__(self,x):
        out = [n(x) for n in self.neurons]  #n(x)表示调用神经元的call，输出act
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
