# core.py (完整修正版)

class Value:
    """
    Value: 存储数据 + 梯度 + 自动微分逻辑的原子单元
    """
    def __init__(self, data, _children=()):
        self.data = data
        self.grad = 0.0  # 梯度：初始为0
        
        # 内部属性：用于构建计算图
        self._prev = set(_children)
        self._backward = lambda: None # 默认是一个空锦囊
        self._op = ''

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other))
        out._op = '+'
        
        # --- 加法的锦囊 (关键修复) ---
        def _backward():
            # 加法就是“复印机”：把上游传来的梯度 (out.grad) 原样加给 self 和 other
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward # 装入锦囊
        
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other))
        out._op = '*'
        
        # --- 乘法的锦囊 (关键修复) ---
        def _backward():
            # 乘法就是“交换并放大”：
            # 给我(self)的梯度 = 对方的数据(other.data) * 上游梯度(out.grad)
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward # 装入锦囊
        
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data ** other, (self,))
        out._op = '**'
        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # 拓扑排序：从后往前排列好所有节点
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # 从后往前，依次调用每个节点的“锦囊”
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def relu(self):
        out = Value(0.0 if self.data < 0 else self.data, (self,))
        out._op = 'relu'

        def _backward():
            # ReLU 就是“门”：
            # 当我是正值时，我把上游梯度(out.grad)原封不动给下游
            # 当我是负值时，我把下游传来的梯度(0.0)给下游
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward # 装入锦囊
        
        return out
    
# 测试代码
def train_neuron():
    # 1. 构建 (Space)
    a = Value(2.0)
    b = Value(3.0)
    c = Value(10.0)
    
    # 模拟一次前向传播
    L = a * b + c 
    print(f"Step 0: L = {L.data} (目标是让它变小)")
    
    # 2. 学习循环 (Time Iteration)
    # 我们手动在这个“时间”里走一步
    
    # A. 自动反向传播：计算梯度
    L.backward() # <--- 现在这里是全自动的了！
    
    print(f"    梯度指路: a.grad={a.grad}, b.grad={b.grad}")
    
    # B. 更新变量 (Variables Update)
    # 这一步叫“梯度下降” (Gradient Descent)
    # 既然梯度是正的（代表正相关），我们就把变量“减去”一点点梯度
    learning_rate = 0.01 
    a.data -= learning_rate * a.grad
    b.data -= learning_rate * b.grad
    # c 和 b 同理，不过为了演示简单，我们这里只更新 a 和 b
    
    # 3. 再次前向传播，看看效果
    L_new = a * b + c
    print(f"Step 1: L = {L_new.data}")
    print("--- 结论：你看，Loss 真的变小了！这就是机器学习！---")

if __name__ == "__main__":
    train_neuron()