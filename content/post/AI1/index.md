---
title: AI
description: AI
slug: AI1
date: 2025-04-05 10:23:00+0800
math: true
# image: img/cover.jpg
categories:
    - 文档
    - AI Infra
tags:
    - 文档
    - AI Infra
weight: 12
---
## Reference

https://c.d2l.ai/stanford-cs329p/

## 回顾传统机器学习

### 从线性模型（LM）到多层感知机（MLP）

全连接（**fully connected**）层，或者叫稠密（**dense**）层，线性（**linear**）层是LM和MLP中的重要组成部分，有$\boldsymbol{W}\in \mathbb{R}^{m \times n}, b\in \mathbb{R}^{m}$的参数，通过输入$\boldsymbol{x}$计算输出$\boldsymbol{y}=\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}\in \mathbb{R}^{m}$。

在LM（Linear Method）模型中，如线性回归可以视为只有一个输出($m=1$)的全连接层，用于处理多分类问题的softmax回归可以视为有$m$个softmax输出的全连接层。总之，可以理解为单层感知机。

在MLP（Multi-Layer Perceptron）模型中，我们期望得到**非线性模型**，因此使用多个全连接层进行“组装”。但简单的多个全连接层叠加仍然是线性操作，因此需要引入激活函数（Activation）来引入非线性性（如sigmoid、ReLU等）。除了模型第一个用于显式输入的全连接层作为输入层和最后一个用于显式输出的全连接层作为输出层，其他全连接层都可以视为隐藏层。

MLP是一个全连接的神经网络，全连接层的每一个输出是要对所有的输入元素做加权和（$\boldsymbol{y}=\boldsymbol{W}\boldsymbol{x}+\boldsymbol{b}$），导致描述复杂问题会带来超参数的规模爆炸。

### 从多层感知机（MLP）到卷积神经网络（CNN）

#### 卷积层

基于以下两个原理来设计方法解决这一问题（基于CV的背景）：

+ **Locality**（本地性/局部性）：像素与周边像素的相关性较高，距离越远，相关性越低

+ **Translation Invariance**（平移不变性）：平移后的目标仍然能够被识别（参考[卷积神经网络为什么具有平移不变性？](https://zhangting2020.github.io/2018/05/30/Transform-Invariance/)）

在卷积层中，这两种性质被翻译为：

+ **Locality**：不同于全连接层对$n\times n$的每个像素进行学习，卷积层只在一个更小的$k\times k$的像素窗口内学习（一个输出只使用一个$k\times k$大小的窗口，而非MLP中是所有输入的加权和）

+ **Translation Invariance**：**参数能够被（平移）共享**（weight shared），可学习的参数与input/output大小解耦，只与窗口大小$k$相关

将训练的$k\times k$窗口用于识别图像的模式，这就是卷积核(Convolution Kernel)。通过多个通道的卷积核，能够用来识别图像的多个模式。二维卷积可以理解为使用矩阵乘（因为参数也是二维矩阵）的全连接（交叉相关），并非是数学意义上的卷积。

简而言之，卷积层就是将input和kernel weight进行交叉相关，加上bias得到output；kernel weight和bias是bias学习的参数，kernel size则是超参数。

```python
### 代码来自BV1L64y1m7Nh
def corr2d(X, K):
    """
    二维交叉相关
    """
    h, w = K.shape
    # 输出会减少(h - 1, w - 1)
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:h + w] * K).sum()
    return Y

class Conv2D(nn.Module):
    """
    二维卷积层
    """
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch,rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))
    
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

还有一些细节就简单提一下。

+ **Padding**（填充）：每一层卷积都会让input缩小$(k_h - 1, k_w - 1)$，为了防止input变小得太快，常填充$p_h$行$p_w$列，让output形状为$(行-k_h+p_h+1)\times(列-k_w+p_w+1)$，通常取$p_h=k_h-1,p_w=k_w-1$

+ **Stride**（步幅）：当input过大时，需要大量卷积层才能得到较小output，那么增大kernel（滑动窗口）每轮移动的stride即可加快这一过程

+ **Multi-Channel**（多通道）：每个通道都有一个卷积kernel，结果是所有通道的卷积结果之**和**，当使用$1\times 1$的kernel时，用于融合通道

#### 池化层

卷积层对输入位置敏感，为了保持一定程度平移不变性的“宽容”，引入了池化层。池化层与卷积层类似，也使用一个具有Padding、Stride、Multi-Channel（往往不进行多通道融合）的滑动窗口，但是不学习参数。往往使用最大池化层（强调每个窗口中最强的模式信号）、平均池化层高（平均每个窗口中的模式信号）等，来缓解卷积层对位置的敏感性。