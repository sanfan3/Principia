# Principia: An Educational Autograd Engine
> "From First Principles to Neural Networks"

## 📖 项目简介 (Introduction)
Principia 是一个基于 Python 原生代码实现的深度学习框架。
它不依赖 PyTorch 或 TensorFlow，而是从最基础的数学原理出发，实现了自动微分引擎（Autograd）和多层感知机（MLP）。

这个项目的核心目的是为了**建立 AI 的全局认识**：
- **Space (空间)**: 通过计算图构建神经网络结构。
- **Time (时间)**: 通过反向传播 (Backpropagation) 传递误差。
- **Variables (变量)**: 通过梯度下降 (Gradient Descent) 更新权重。

## 🌾 核心理念：农耕模拟论 (The Farming Theory)
AI 的本质，是利用计算机的高速计算，压缩现实世界的试错周期。

1.  **四季轮转 (Forward Pass)**: 就像 `a * b + c`，这是客观的计算规律。
2.  **土地贫瘠 (Loss)**: 衡量结果与目标的差距。
3.  **反向复盘 (Backward Pass)**: 追溯是哪个季节（参数）导致了收成不好。
4.  **行为修正 (Update)**: 在模拟中调整翻土与施肥的力度。

## 🚀 快速开始 (Quick Start)

运行示例代码，见证 AI 的自我学习：

```bash
python examples/train_demo.py