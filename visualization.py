import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import random

def plot_training_curves(history):
    """绘制训练过程中的损失和准确率曲线。

    Args:
        history (dict): 包含 'train_losses', 'test_losses', 'train_accuracies', 'test_accuracies' 的字典
    """
    epochs = range(1, len(history['train_losses']) + 1)
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_losses'], label='Train Loss')
    plt.plot(epochs, history['test_losses'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracies'], label='Train Accuracy')
    plt.plot(epochs, history['test_accuracies'], label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.show()

def plot_sample_predictions(model, test_set, num_samples=10):
    """可视化测试集中的样本预测结果。

    Args:
        model: 训练好的模型
        test_set: 测试数据集
        num_samples (int): 显示的样本数量
    """
    indices = random.sample(range(test_set.total_sample), num_samples)
    for idx in indices:
        img = test_set.imgs[idx].reshape(28, 28)
        true_label = np.argmax(test_set.labels[idx])
        pred = model.forward(test_set.imgs[idx].reshape(1, -1))
        pred_label = np.argmax(pred)

        plt.figure(figsize=(2, 2))
        plt.imshow(img, cmap='gray')
        plt.title(f'True: {true_label}, Pred: {pred_label}')
        plt.axis('off')
        plt.show()