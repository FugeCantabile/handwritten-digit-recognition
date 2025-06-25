from layers import *
import numpy as np


class Module(Layer):
    """四层神经网络模型，用于手写数字识别。

    Args:
        in_dim (int): 输入层维度 (28*28=784)
        out_dim (int): 输出层维度 (10个类别)
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.linear_1 = Linear(in_dim, 64)
        self.activation_1 = ReLU(64, 64)
        self.linear_2 = Linear(64, 32)
        self.activation_2 = ReLU(32, 32)
        self.linear_3 = Linear(32, 10)
        self.activation_3 = Sigmoid(out_dim, out_dim)

    def forward(self, x):
        x = self.linear_1.forward(x)
        x = self.activation_1.forward(x)
        x = self.linear_2.forward(x)
        x = self.activation_2.forward(x)
        x = self.linear_3.forward(x)
        x = self.activation_3.forward(x)
        return x

    def backward(self, grad):
        grad = self.activation_3.backward(grad)
        grad = self.linear_3.backward(grad)
        grad = self.activation_2.backward(grad)
        grad = self.linear_2.backward(grad)
        grad = self.activation_1.backward(grad)
        grad = self.linear_1.backward(grad)

    def step(self, lr):
        self.linear_1.step(lr)
        self.linear_2.step(lr)
        self.linear_3.step(lr)
        self.activation_1.step(lr)
        self.activation_2.step(lr)
        self.activation_3.step(lr)

    def save(self, path):
        """保存模型参数到文件。"""
        np.savez(path, w1=self.linear_1.w, b1=self.linear_1.b,
                 w2=self.linear_2.w, b2=self.linear_2.b,
                 w3=self.linear_3.w, b3=self.linear_3.b)

    def load(self, path):
        """从文件加载模型参数。"""
        data = np.load(path)
        self.linear_1.w, self.linear_1.b = data['w1'], data['b1']
        self.linear_2.w, self.linear_2.b = data['w2'], data['b2']
        self.linear_3.w, self.linear_3.b = data['w3'], data['b3']