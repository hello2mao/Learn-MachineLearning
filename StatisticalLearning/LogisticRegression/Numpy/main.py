import numpy as np
import pandas as pd
import random


# min-max 归一化
def normalization(data):
    _range = np.max(data) - np.min(data)
    return (data - np.min(data)) / _range


# 加载数据
def load_data():
    df = pd.read_csv('data.csv')
    # features
    features = df.iloc[:, :-1].to_numpy()
    features = normalization(features)
    ones = np.ones(shape=features.shape[0])
    features = np.c_[features, ones]
    # labels
    labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    labels = normalization(labels)
    return features, labels


# 批量读取数据
def data_iter(batch_size, X, y):
    num_examples = len(X)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i:max(i+batch_size, num_examples)]
        yield X[batch_indices], y[batch_indices]


# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# 计算损失
def compute_loss(X, y, w):
    y_hat = sigmoid(np.dot(X, w))
    loss = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))
    loss /= len(X)
    return loss


# 计算梯度
def compute_gradient(X, y, w):
    grad = np.dot(X.T, sigmoid(np.dot(X, w)) - y)
    grad /= len(X)
    return grad


# 训练
def fit(X, y):
    # 初始化模型参数
    w = np.random.rand(X.shape[1])

    # 开始训练
    batch_size = 50
    learning_rate = 0.2
    iter_max = 100
    for n_iter in range(1, iter_max+1):
        # compute loss
        loss = compute_loss(X, y, w)
        print(f'current loss: {loss}')
        if abs(loss) <= 1e-4:
            print(f'loss <= 1e-4, fit finish')
            break
        for (batch_X, batch_y) in data_iter(batch_size, X, y):
            grad = compute_gradient(batch_X, batch_y, w)
            w -= learning_rate * grad
    return w


def predict(X, y, w):
    count = 0
    for index, x in enumerate(X):
        pred = sigmoid(np.dot(x, w))
        if y[index] == 1 and pred > 0.5:
            count += 1
        if y[index] == 0 and pred < 0.5:
            count += 1
    return 100 * count / len(X)


if __name__ == '__main__':
    X, y = load_data()
    w = fit(X, y)
    predict_result = predict(X, y, w)
    print(f'predict_result: {predict_result}%')
