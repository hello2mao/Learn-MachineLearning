#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
 
# 加载数据
def load_data(file_name):
    df = pd.read_csv(file_name)
    print('read csv data shape: ', df.shape)
    # features
    features = df.iloc[:, :-1].to_numpy()
    ones = np.ones(shape=features.shape[0])
    # np.c_按行链接矩阵
    features = np.c_[features, ones]
    print('features shape: ', features.shape)
    # labels
    labels = np.squeeze(df.iloc[:, -1:].to_numpy().reshape(1, -1))
    print('labels shape: ', labels.shape)
    return features, labels

if __name__ == '__main__':
    # 1.加载数据
    X, y = load_data('data.csv')
    
    # 2.拆分训练集,测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    
    # 3.标准化特征值
    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    # 4. 训练逻辑回归模型
    logreg = LogisticRegression(fit_intercept=True)
    logreg.fit(X_train, y_train)
    
    # 5. 预测
    y_test_hat = logreg.predict(X_test_std)
    acc = logreg.score(X_test_std,y_test)
    print('acc', acc)