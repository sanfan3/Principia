from src.core import Value
from src.nn import MLP

def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    nodes, edges = trace(root)
    
    dot = ['digraph G {']
    dot.append('rankdir=LR;') 
    dot.append('node [shape=record];') 
    
    for n in nodes:
        uid = str(id(n))
        
        # 1. 画出节点
        label = f"{{ data {n.data:.4f} | grad {n.grad:.4f} }}"
        dot.append(f'{uid} [label="{label}"];')
        
        # 2. 画出运算符号 (关键修复点)
        if n._op:
            # 以前是 uid + n._op (会导致 123* 这种非法ID)
            # 现在改成 uid + '_op' (变成了 123_op 这种合法ID)
            uid_op = uid + '_op' 
            dot.append(f'{uid_op} [label="{n._op}", shape=circle];')
            dot.append(f'{uid_op} -> {uid};')
            
    for n1, n2 in edges:
        uid1 = str(id(n1))
        uid2 = str(id(n2))
        
        if n2._op:
            # 如果 n2 是运算产生的，线要连到它的运算节点上
            uid2_op = uid2 + '_op'
            dot.append(f'{uid1} -> {uid2_op};')
        else:
            dot.append(f'{uid1} -> {uid2};')
            
    dot.append('}')
    return '\n'.join(dot)

if __name__ == "__main__":
    # 1. 搭建小网络
    model = MLP(2, [2, 1])
    
    # 2. 伪造输入
    x = [Value(1.0), Value(-1.0)]
    
    # 3. 前向传播
    y = model(x)
    
    # 4. 反向传播
    y.backward()
    
    # 5. 生成代码
    print(draw_dot(y))