import numpy as np

class Layer:#抽象化每一层为Layer类，具体细节各自实现
    def __init__(self, in_dim, out_dim):
        self.in_dim = in_dim
        self.out_dim = out_dim

    def forward(self, x):
        pass  # 什么也不做，跳过

    # gradient: 梯度
    def backward(self, grad):
        pass

    # 梯度更新的函数
    def step(self, lr):  # learning_rate
        pass


class Linear(Layer):#线性函数层
    def __init__(self, in_dim, out_dim):
        # 主动调用父类的构造函数
        # 避免了重复写self.in_dim = in_dim, self.out_dim = out_dim
        super().__init__(in_dim, out_dim)

        # w.shape = (in_dim, out_dim)
        # b.shape = (1, out_dim)
        # np.random.randn()用来生成正态分布的矩阵
        self.w = np.random.randn(self.in_dim, self.out_dim) / np.sqrt(in_dim)
        self.b = np.random.randn(1, self.out_dim)

        self.w_grad = None
        self.b_grad = None

        self.input = None

    def forward(self, x):#前向传播
        # x.shape = (batch_size, in_dim)
        self.input = x
        return np.dot(x, self.w) + self.b

    def backward(self, grad):#后巷传播
        # grad.shape = (batch_size, out_dim)
        batch_size = grad.shape[0]
        # w_grad.shape = self.w.shape = (in_dim, out_dim)
        self.w_grad = np.dot(self.input.T, grad) / batch_size  # 把batch_size张图片的梯度平均一下
        # b_grad = self.b.shape = (1, out_dim)
        self.b_grad = np.sum(grad, axis=0) / batch_size  # 沿着0维加和
        return np.dot(grad, self.w.T)

    def step(self, lr):#梯度下降
        # 这里的*代表对矩阵的每个元素都乘以同样的数
        self.w = self.w - lr * self.w_grad
        self.b = self.b - lr * self.b_grad


# sigmoid(x) = 1 / (1 + e^-x)
# sigmoid'(x) = e^-x/(1 + e^-x)^2
class Sigmoid(Layer):#激活函数层
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)

        self.input = None

    @staticmethod  # 静态函数：不需要依赖类的对象就可以调用，例如：Sigmoid.__sigmoid(), 也可以用具体的对象来调用，比如x.__sigmoid()
    def __sigmoid(x):  # 静态函数与具体对象无关，所以不需要self
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def __sigmoid_derivative(x):
        e = np.exp(-x)
        return e / (1 + e) ** 2

    def forward(self, x):
        self.input = x
        return self.__sigmoid(x)

    def backward(self, grad):
        # 这里的*代表两个相同形状的矩阵，逐元素相乘
        return grad * self.__sigmoid_derivative(self.input)

    # 不需要step函数可以不写，会自动地调用父类Layer的step函数

class ReLU(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.where(x < 0, 0, x)

    def backward(self, grad):
        #相同形状的矩阵逐元素相乘
        return grad * np.where(self.input < 0, 0, 1)

class Tanh(Layer):
    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim)
        self.input = None

    def forward(self, x):
        self.input = x
        return np.tanh(x)

    @staticmethod
    def __tanh__derivative(x):
        return 4 / (np.exp(2*x) + np.exp(-2*x) + 1)

    def backward(self, grad):
        return grad * self.__tanh__derivative(self.input)