from dataset import MNISTDataset
from model import Module
from train import train
from visualization import plot_training_curves, plot_sample_predictions

dataset_path = "dataset"
train_img_path = dataset_path + "/train-images-idx3-ubyte"
train_label_path = dataset_path + "/train-labels-idx1-ubyte"
test_img_path = dataset_path + "/t10k-images-idx3-ubyte"
test_label_path = dataset_path + "/t10k-labels-idx1-ubyte"

# 数据加载
train_set = MNISTDataset(train_img_path, train_label_path, 100, True, True)
train_set.label_to_one_hot()
test_set = MNISTDataset(test_img_path, test_label_path, 1000, True, False)
test_set.label_to_one_hot()

# 模型初始化
model = Module(28 * 28, 10)

# 训练
history = train(model, train_set, test_set, epochs=100, lr=0.1)

# 保存模型
model.save("model.npz")

# 可视化
plot_training_curves(history)
plot_sample_predictions(model, test_set, num_samples=10)