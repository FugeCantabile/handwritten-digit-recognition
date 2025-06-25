import numpy as np


def square_loss(pred, target):
    """计算平方损失及其梯度。

    Args:
        pred (np.ndarray): 预测值，形状 (batch_size, 10)
        target (np.ndarray): 真实标签，形状 (batch_size, 10)

    Returns:
        loss (float): 总损失
        grad (np.ndarray): 损失对预测值的梯度
    """
    grad = pred - target
    temp = grad * grad / 2
    loss = np.sum(temp)
    return loss, grad


def is_right(pred, target):
    """计算预测正确的样本数。

    Args:
        pred (np.ndarray): 预测值，形状 (batch_size, 10)
        target (np.ndarray): 真实标签，形状 (batch_size, 10)

    Returns:
        int: 正确预测的样本数
    """
    temp = pred.argmax(1) == target.argmax(1)
    return temp.sum()


def train(model, train_set, test_set, epochs=100, lr=0.1):
    """训练模型并返回训练过程中的损失和准确率。

    Args:
        model: 神经网络模型
        train_set: 训练数据集
        test_set: 测试数据集
        epochs (int): 训练轮数
        lr (float): 学习率

    Returns:
        dict: 包含训练和测试的损失及准确率历史记录
    """
    history = {
        'train_losses': [],
        'test_losses': [],
        'train_accuracies': [],
        'test_accuracies': []
    }

    for epoch in range(epochs):
        train_loss = 0
        train_accurate_cnt = 0
        for data in train_set:
            imgs, targets = data
            prediction = model.forward(imgs)
            loss, grad = square_loss(prediction, targets)
            model.backward(grad)
            model.step(lr)
            train_loss += loss
            train_accurate_cnt += is_right(prediction, targets)

        test_loss = 0
        test_accurate_cnt = 0
        for data in test_set:
            imgs, targets = data
            prediction = model.forward(imgs)
            loss, grad = square_loss(prediction, targets)
            test_loss += loss
            test_accurate_cnt += is_right(prediction, targets)

        # 记录数据
        history['train_losses'].append(train_loss / train_set.total_sample)
        history['test_losses'].append(test_loss / test_set.total_sample)
        history['train_accuracies'].append(train_accurate_cnt / train_set.total_sample)
        history['test_accuracies'].append(test_accurate_cnt / test_set.total_sample)

        print(f"epoch: {epoch}, "
              f"train_loss: {history['train_losses'][-1]:.4f}, "
              f"train_accuracy: {history['train_accuracies'][-1]:.4f}, "
              f"test_loss: {history['test_losses'][-1]:.4f}, "
              f"test_accuracy: {history['test_accuracies'][-1]:.4f}")

    return history