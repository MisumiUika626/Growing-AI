import numpy as np

from ..autograd.tensor import Tensor


# Tensor.leaky_relu() 是实际的数学运算；LeakyReLU 对象只是保存 alpha，并把这个运算包装成具有 forward() 接口的神经网络组件。
class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")

        return x.leaky_relu(self.alpha)


# 我们结合前面的知识来了解一下这个问题，这里线性层输出是不是x @ self.weight + self.bias，可以简单抽象为矩阵相乘的结果（因为这里我们只讨论矩阵相乘带来的矩阵形状变化所以不考虑bias的广播），那令shape.x=(m,n)，shape.w=(n,k)，那最终的shape.y=(m,k)，m是行数，样本按行堆积，然后k就是输出特征数（输出神经元个数）
# activation = LeakyReLU(alpha=0.1)
# out = activation.forward(x)
# 这一句就是前向传播中的激活过程，之前我们直接把矩阵的输出结果当成传入的张量了，这次我们把中间的激活过程加上了，相当于就是第m层激活值经过Linear的计算之后再过一遍激活函数然后就变成下一层神经元的激活值，对应到代码就是 activation建立一个LeakyReLU类，然后输出的值就是对Tensorx来一个前向传播（就是让x过一遍这个函数）这个函数就是Tensor类的成员函数leaky_relu，然后就处理数据，最后输出新的矩阵
class Sigmoid:
    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        return x.sigmoid()


class Softmax:
    def forward(self, x):
        if not isinstance(x, Tensor):
            raise TypeError("x must be a Tensor")
        row_max_data = np.max(x.data, axis=1, keepdims=True)
        stable_logits = x - Tensor(row_max_data)
        exp_logits = stable_logits.exp()
        probabilities = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        return probabilities
