import sys
import os

# --- 魔法代码：把上级目录加入到路径中，这样才能找到 src ---
# 获取当前文件的目录，然后往上走一级
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- 修改引用 ---
# 原来是: from core import Value, from nn import MLP
# 改为:
from src.core import Value
from src.nn import MLP

# 1. 准备数据 (教材)
# xs (Inputs): 4 组输入数据
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
# ys (Targets): 4 个对应的正确答案
# 我们希望：前两个输出 1.0，后两个输出 -1.0
ys = [1.0, 0.0, 0.0, 1.0] 

# 2. 招聘团队 (模型构建)
# 输入是 3 (因为每组数据有 3 个数)
# 结构: [4, 4, 1] (两层各4人的隐藏层，最后1人输出)
model = MLP(3, [4, 4, 1])

# 3. 开始特训 (Training Loop)
# 我们让它学 20 遍 (Epochs)
print("--- 开始训练 ---")

for k in range(100):
    
    # --- 步骤 A: 前向传播 (Forward Pass) ---
    # 让公司处理这 4 组数据，给出 4 个预测结果
    ypred = [model(x) for x in xs]
    
    # --- 步骤 B: 计算亏损 (Loss Calculation) ---
    # 亏损 = (预测值 - 真实值) 的平方
    # 我们要把 4 组数据的亏损加起来，看看总共亏了多少
    loss = Value(0.0)
    for y_target, y_guess in zip(ys, ypred):
        # 手动做平方： 误差 * 误差
        # 注意：我们用 y_guess + (-y_target) 来表示减法
        diff = y_guess + (y_target * -1)
        loss = loss + (diff * diff)
        
    # --- 步骤 C: 归零 (Zero Grad) ---
    # 【非常重要】
    # 在开始新一轮反思之前，把上一轮的“旧锅”清空。
    # 否则责任会一直累加，员工就崩溃了。
    for p in model.parameters():
        p.grad = 0.0
        
    # --- 步骤 D: 反向传播 (Backward Pass) ---
    # 这里的 loss 包含了所有 4 组数据的总误差
    # 这一步会自动算出那 41 个员工每个人该背多少锅
    loss.backward()
    
    # --- 步骤 E: 更新参数 (Update) ---
    # 每个人根据自己的锅，调整一点点
    # 学习率 (learning rate) = 0.05 (调整的幅度)
    for p in model.parameters():
        p.data += -0.05 * p.grad
        
    if k % 10 == 0:
        print(f"第 {k} 轮 | 总误差 Loss: {loss.data:.4f}")

print("\n--- 训练结束，检查结果 ---")
print(f"正确答案: {ys}")
print(f"AI预测值: {[y.data for y in ypred]}")