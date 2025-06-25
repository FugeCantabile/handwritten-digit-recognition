import IDX_reader
import numpy as np

# [温度，湿度，海拔]
# [25, 100, 3000]
# [20, 12, 5000]
# [-10, 8, 4000]

# mean = [12.167, 40, 4000]
# std = sqrt(平方差)
# data = (data - mean) / std 之后，data所有维度的mean都是0，所有维度的std都是1
#即(X - mean) / sqrt(平方差)

def _normalize(data):
    # data.shape = (sample_num, dimension)
    mean = np.mean(data, axis=0) # mean.shape = (dimension, 1)
    std = np.std(data, axis=0)
    # 有的维度可能无变化，std==0，因此需要判断当std==0时，就设置为1
    # 如果std < 1e-8, 那就设置为1，否则维持原样
    # 1e-8: 计算机的小数不是准确的，没办法直接比较
    # a, b: a == b ? |a - b| < 1e-8
    std = np.where(std < 1e-8, 1, std)

    return (data - mean) / std

class MNISTDataset:

    # normalize，正则化：将所有维度的mean调整为0，标准差调整为1，从而保证所有维度对于训练的影响是相同的
    # shuffle: 每当训练完一轮，就将所有的数据重新洗牌，保证每一轮训练的顺序是不一样的，增加随机性，进而使模型更加稳定
    def __init__(self,
                 img_path,
                 label_path,
                 batch_size=1,
                 normalize=False,
                 shuffle=False):
        # imgs.shape = (sample_num, dimension)
        # labels.shape = (sample_num, 1)
        self.imgs = IDX_reader.decode_idx3_ubyte(img_path)
        self.labels = IDX_reader.decode_idx1_ubyte(label_path)
        self.batch_size = batch_size
        self.total_sample = self.imgs.shape[0]
        self.shuffle = shuffle

        if normalize:
            self.imgs = _normalize(self.imgs)

        self.idx = 0

    # 把label转成(sample_num, 10)
    # one_hot: 独热码：只有一位为1，其他都是0
    def label_to_one_hot(self):
        # 确保labels是一个整数的矩阵
        labels = self.labels.astype(np.int32)

        # 方法1：简单，但是慢
        # enumerate和正常迭代差不多，只是每次会额外返回一个当前循环的次数i
        # one_hot = np.zeros((self.total_sample, 10))
        # for i, label in enumerate(labels):
        #     one_hot[i, label] = 1

        # 方法2
        labels = labels.reshape(-1).tolist()  # reshape(-1)是把任何形状的矩阵展开成一行
        eye = np.eye(10)  # 单位矩阵，对角线为1，其他为0
        one_hot = eye[labels]  # labels是一个索引矩阵，每次取eye的第labels[i]个元素，作为one_hot[i]
        # 等价于👇
        # for label in labels:
        #     one_hot.拼接上(eye[label])

        self.labels = one_hot

    # 为了能实现“for data in dataset: ”语句，我们需要实现__iter__, __next__函数
    # __iter__会在循环开始的时候调用，需要返回一个对象，这个对象能够调用__next__函数来返回数据
    def __iter__(self):
        self.idx = 0

        if self.shuffle:
            # np每次shuffle的顺序都是随机的，无法保证shuffle之后，imgs和labels还能一一对应
            # 只要种子state相同，shuffle的顺序就是一样的
            random_state = np.random.get_state()
            np.random.shuffle(self.imgs)
            np.random.set_state(random_state)
            np.random.shuffle(self.labels)

        return self
    
    def __next__(self):
        if self.idx + self.batch_size <= self.total_sample:
            # 剩下的依旧足够切一个batch出来
            imgs = self.imgs[np.arange(self.idx, self.idx + self.batch_size)]
            labels = self.labels[np.arange(self.idx, self.idx + self.batch_size)]
            self.idx += self.batch_size
            ret = (imgs, labels)
        elif self.idx < self.total_sample:
            # 还剩下不足batch_size个样本
            imgs = self.imgs[np.arange(self.idx, self.total_sample)]
            labels = self.labels[np.arange(self.idx, self.total_sample)]
            self.idx += self.batch_size
            ret = (imgs, labels)
        else:
            raise StopIteration # 用来告诉python，没有更多数据了，可以退出循环


        return ret