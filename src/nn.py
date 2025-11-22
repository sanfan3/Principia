# nn.py (升级版 - 解决神经元死亡问题)
import random
from src.core import Value

class Neuron:
    # 1. 增加 nonlin 参数：决定是否要用 ReLU
    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1,1))
        self.nonlin = nonlin # 记住这个开关

    def __call__(self, x):
        act = self.b
        for wi, xi in zip(self.w, x):
            act = act + wi * xi
        
        # 2. 关键修改：只有开关打开时，才用 ReLU
        # 如果关掉 (nonlin=False)，就直接输出原始数值 (线性)
        return act.relu() if self.nonlin else act

    def parameters(self):
        return self.w + [self.b]

class Layer:
    # Layer 也把这个开关传下去
    def __init__(self, nin, nout, nonlin=True):
        self.neurons = [Neuron(nin, nonlin) for _ in range(nout)]
        
    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        
    def parameters(self):
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = []
        
        for i in range(len(nouts)):
            # 3. 关键逻辑：
            # i == len(nouts) - 1 意思是“这是不是最后一层？”
            # 如果是最后一层，nonlin = False (不加过滤器，畅所欲言)
            # 如果不是最后一层，nonlin = True (加过滤器，增加复杂度)
            is_last = (i == len(nouts) - 1)
            self.layers.append(Layer(sz[i], sz[i+1], nonlin=not is_last))
        
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
        
    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params