import numpy as np
import pandas as pd
import random


# 加载数据
def load_data():
    df = pd.read_csv('data.csv')
    features = df.iloc[:, :-1].to_numpy()
    labels = df.iloc[:, -1:].to_numpy().reshape(1, -1)[0]
    return features, labels


# 批量读取数据
def data_iter(batch_size, X, y):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:max(i+batch_size, num_examples)]
        yield X[batch_indices], y[batch_indices]


# 计算损失
def loss(X, y, w, b):
    return (y * np.log(sigmoid(np.dot(X, w) + b)) + (1 - y) * np.log(1 - sigmoid(np.dot(X, w) + b))).sum()


# 计算梯度
def gradient(X, y, w, b):
    return np.dot(X.T, sigmoid(np.dot(X, w) + b) - y)


def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


def fit(X, y):
    # 初始化模型参数
    w = np.random.rand(X.shape[1])
    b = np.random.rand(1)
    print(f'init w: {w}, b: {b}')
    # 开始训练
    batch_size = 10
    learning_rate = 0.1
    iter_max = 1000
    for n_iter in range(1, iter_max+1):
        for batch_data in data_iter(batch_size, X, y):
            print(batch_data)


if __name__ == '__main__':
    X, y = load_data()
    print(X.shape)
    print(y)
    fit(X, y)
