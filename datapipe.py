import os
import numpy as np
import torch
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
import scipy.io as sio
import glob

# --- 全局配置 ---
# [请务必修改这里] SEED 数据集的 ExtractedFeatures 文件夹路径
DATA_ROOT = 'D:/Pycharm/python/pythonProject3/SEED/ExtractedFeatures/'

subjects = 15  # SEED 数据集包含 15 个被试
classes = 3  # 3 分类: Negative, Neutral, Positive
version = 1  # 数据集版本号 (如需强制重新生成，请修改此数字)


def to_categorical(y, num_classes=None, dtype='float32'):
    """
    将标签转换为 One-Hot 编码
    """
    y = np.array(y, dtype='int16')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical


class EmotionDataset(InMemoryDataset):
    """
    自定义 EEG 情感数据集类 (基于 PyG InMemoryDataset)
    输出格式: x=[62, 1325], y=[3]
    适配: iTransformer (Token=62, Embedding=1325)
    """

    def __init__(self, stage, root, subjects, sub_i, X=None, Y=None, transform=None, pre_transform=None):
        self.stage = stage
        self.subjects = subjects
        self.sub_i = sub_i
        self.X = X
        self.Y = Y
        super().__init__(root, transform, pre_transform)

        # 加载已处理的数据
        try:
            self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        except TypeError:
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        # 缓存文件名: V_版本_阶段_CV总数_当前被试.dataset
        return ['./processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(
            version, self.stage, self.subjects, self.sub_i)]

    def process(self):
        # 如果没有数据传入(读取模式)，直接返回
        if self.X is None or self.Y is None:
            return

        data_list = []
        num_samples = np.shape(self.Y)[0]

        print(f"Processing {self.stage} data for Subject {self.sub_i}...")
        for sample_id in tqdm(range(num_samples)):
            # 提取特征: (62, 1325)
            x = self.X[sample_id, :, :]
            x = torch.FloatTensor(x)

            # 提取标签
            y = torch.FloatTensor(self.Y[sample_id, :])

            # 构建 PyG Data 对象
            # iTransformer 不需要 edge_index，直接封装特征矩阵
            data = Data(x=x, y=y)
            data_list.append(data)

        data, slices = self.collate(data_list)
        os.makedirs(os.path.dirname(self.processed_paths[0]), exist_ok=True)
        torch.save((data, slices), self.processed_paths[0])


def normalize(data):
    """
    Z-score 标准化: (Data - Mean) / Std
    """
    mee = np.mean(data, 0)
    data = data - mee
    stdd = np.std(data, 0)
    data = data / (stdd + 1e-7)
    return data

def build_dataset(subjects):
    """
    构建并保存 LOSO (Leave-One-Subject-Out) 数据集
    """
    os.makedirs('./processed', exist_ok=True)

    # 检查是否需要加载原始数据
    need_load = False
    for sub_i in range(subjects):
        train_file = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, 'Train', subjects, sub_i)
        test_file = './processed/V_{:.0f}_{:s}_CV{:.0f}_{:.0f}.dataset'.format(version, 'Test', subjects, sub_i)
        if not os.path.exists(train_file) or not os.path.exists(test_file):
            need_load = True
            break

    if need_load:
        print("Loading raw data from .mat files...")
        mov_coefs, labels = get_data()
        print("Raw data loaded. Building datasets...")

        for sub_i in range(subjects):
            # LOSO 划分
            index_list = list(range(subjects))
            del index_list[sub_i]

            train_index = index_list  # 源域
            test_index = sub_i  # 目标域

            # Reshape: 确保维度是 (Samples, 62, 1325)
            X_train = mov_coefs[train_index, :].reshape(-1, 62, 1325)
            Y_train = labels[train_index, :].reshape(-1)

            X_test = mov_coefs[test_index, :].reshape(-1, 62, 1325)
            Y_test = labels[test_index, :].reshape(-1)

            # 标签 One-Hot
            _, Y_train = np.unique(Y_train, return_inverse=True)
            Y_train = to_categorical(Y_train, classes)

            _, Y_test = np.unique(Y_test, return_inverse=True)
            Y_test = to_categorical(Y_test, classes)

            # 创建数据集
            EmotionDataset('Train', './', subjects, sub_i, X_train, Y_train)
            EmotionDataset('Test', './', subjects, sub_i, X_test, Y_test)

        print("All datasets built successfully.")
    else:
        print("All datasets already exist. Skipping build.")


def get_dataset(subjects, sub_i):
    train_dataset = EmotionDataset('Train', './', subjects, sub_i)
    target_dataset = EmotionDataset('Test', './', subjects, sub_i)
    return train_dataset, target_dataset


if __name__ == '__main__':
    build_dataset(subjects)