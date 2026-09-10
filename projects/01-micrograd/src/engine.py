class Value:
    "记录一个节点的数值和他的梯度"

    def __init__(self, data, _children = (), _op = ''):
        self.data = data
        self.grad = 0
        self._prev = set(_children)
        self._backward = lambda : None
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data + other.data,(self,other),'+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other,Value) else Value(other)
        out = Value(self.data * other.data,(self,other),'*')
        def _backward():
            self.grad += out.grad * other.data
            other.grad += out.grad * self.data
        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other,(float,int))
        out = Value(self.data ** other,(self,),f'**{other}')
        def _backward():
            self.grad += out.grad * other * (self.data**(other-1))
        out._backward = _backward
        return out

    def relu(self):
        out = Value(0 if self.data < 0 else self.data,(self,),'Reluu')
        def _backward():
            self.grad += out.grad * (out.data > 0)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # for v in topo:
        #     v.grad = 0
        self.grad = 1
        for i in reversed(topo):
            i._backward()


    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __truediv__(self, other):
        return self * other **-1

    def __repr__(self):
            return f"Value(data={self.data}, grad={self.grad})"