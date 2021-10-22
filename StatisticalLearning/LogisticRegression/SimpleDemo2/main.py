#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import time


# 加载数据
def load_data(file_name):
    df = pd.read_csv(file_name)
    print('read csv data shape: ', df.shape)
    # features
    features = df.iloc[:10000, 1:-1].to_numpy()
    ones = np.ones(shape=features.shape[0])
    # np.c_按行链接矩阵
    features = np.c_[features, ones]
    print('features shape: ', features.shape)
    # labels
    labels = np.squeeze(df.iloc[:10000, -1:].to_numpy().reshape(1, -1))
    print('labels shape: ', labels.shape)
    return features, labels


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
    print('fit start')
    # 初始化模型参数
    np.random.seed(1)
    w = np.random.rand(X.shape[1])
    print('init w: ', w)

    # 开始训练
    learning_rate = 0.15
    iter_max = 10000
    for n_iter in range(1, iter_max+1):
        # compute loss
        loss = compute_loss(X, y, w)
        print(f'current loss: {loss}')
        if abs(loss) <= 1e-4:
            print(f'loss <= 1e-4, fit finish')
            break
        # for (batch_X, batch_y) in data_iter(batch_size, X, y):
        grad = compute_gradient(X, y, w)
        w -= learning_rate * grad
    return w


# 预测
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
    X, y = load_data('default_credit_hetero_all.csv')
    t1 = time.perf_counter()
    w = fit(X, y)
    print('w', w)
    print('fit cost', time.perf_counter() - t1)
    # X_test, y_test = load_data('test_data.csv')
    # predict_result = predict(X_test, y_test, w)
    # print(f'predict_result: {predict_result}%')
