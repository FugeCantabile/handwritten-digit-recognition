import numpy as np

'''
全连接层（Fully Connected Layer，也称为密集层，Dense Layer）
是神经网络中的一种常见层类型，通常用于将前一层的输出映射到下一层的输
入。全连接层的每个神经元都与前一层的所有神经元相连，因此得名“全连接”。

全连接层的矩阵形式计算公式：
假设：
1.前一层的输出为R_n维向量x，其中n是前一层的神经元数量。
2.全连接层的权重矩阵为R_m*n维矩阵W，其中m是全连接层的神经元数量。
3.全连接层的偏置向量为R_m维向量b。
4.全连接层的激活函数为g(·)（例如Sigmoid、Tanh等）
则全连接层的输出R_m维向量y可以表示为：
y = g(Wx + b)
'''

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, activation = 'sigmoid'):
        '''
        初始化全连接层
        :param input_size: 输入向量的维度
        :param output_size: 输出向量的维度
        :param activation: 激活函数，默认为sigmoid
        '''
        # 用标准正态分布初始化权重矩阵 W (output_size * input_size)，乘以0.01避免权重过大或过小
        self.weights = np.random.randn(output_size, input_size) * 0.01
        # 初始化偏置向量 b (output_size * 1)
        self.bias = np.zeros((output_size, 1))
        # 选择激活函数
        self.activation = activation

    def forward(self, x):
        '''
        前向传播
        :param x: 输入向量（input_size * 1）
        :return: 输出向量（output_size * 1）
        '''
        # 线性变换：z = Wx + b
        self.input = x # 保存输入，用于反向传播
        self.z = np.dot(self.weights, x) + self.bias
        # 激活函数
        if self.activation == 'sigmoid':
            self.output = self.sigmoid(self.z)
        elif self.activation == 'tanh':
            self.output = self.tanh(self.z)
        elif self.activation == 'relu':
            self.output = self.relu(self.z)
        elif self.activation == 'leaky_relu':
            self.output = self.leaky_relu(self.z)
        elif self.activation == 'softmax':
            self.output = self.softmax(self.z)
        else:
            raise ValueError("Unsupported activation function")
        return self.output
    def sigmoid(self, z):
        '''Sigmoid 激活函数'''
        return 1 / (1 + np.exp(-self.z))
    def tanh(self, z):
        '''Tanh 激活函数'''
        return np.tanh(self.z)
    def relu(self, z):
        '''ReLU 激活函数'''
        return np.maximum(0, self.z)
    def leaky_relu(self, z, alpha = 0.01):
        '''Leaky ReLU 激活函数'''
        return np.where(self.z > 0, self.z, alpha * self.z)
    def softmax(self, z):
        '''Softmax 激活函数'''
        exp_self_z = np.exp(self.z - np.max(self.z)) #减去最大值防止数值溢出
        return exp_self_z / np.sum(exp_self_z, axis=0)
    def __call__(self, x):
        '''使实例可以像函数一样调用'''
        return self.forward(x)

# 示例使用
if __name__ == "__main__":
    # 输入向量（3 * 1）
    x = np.random.random((3, 1))

    # 创建一个全连接层，输入维度为 3，输出维度为 2，使用默认的Sigmoid激活函数
    fc_layer = FullyConnectedLayer(input_size=3, output_size=2, activation='sigmoid')

    # 前向传播
    output = fc_layer(x)

    print("输入 x:\n", x)
    print("权重 W:\n", fc_layer.weights)
    print("偏置 b:\n", fc_layer.bias)
    print("输出 y:\n", output)