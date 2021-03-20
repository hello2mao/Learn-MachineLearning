#-*- coding: UTF-8 -*- 

'''
    sklearn中处理线性回归问题的API：
        import sklearn.linear_model as lm
        # 创建模型
        model = lm.LinearRegression()
        # 训练模型
        # 输入为一个二维数组表示的样本矩阵
        # 输出为每个样本最终的结果
        model.fit(输入, 输出) # 训练模型
        # 预测输出
        # 输入array是一个二维数组，每一行是一个样本，每一列是一个特征。
        result = model.predict(array)

    评估训练结果误差（metrics）---模型评估
        线性回归模型训练完毕后，可以利用测试集评估训练结果误差。sklearn.metrics提供了计算模型误差的几个常用算法：
                import sklearn.metrics as sm
                # 平均绝对值误差：1/m∑|实际输出-预测输出|
                sm.mean_absolute_error(y, pred_y)
                # 平均平方误差：SQRT(1/mΣ(实际输出-预测输出)^2)
                sm.mean_squared_error(y, pred_y)
                # 中位绝对值误差：MEDIAN(|实际输出-预测输出|)
                sm.median_absolute_error(y, pred_y)
                # R2得分，(0,1]区间的分值。分数越高，误差越小。---应用多
                sm.r2_score(y, pred_y)

    模型的保存和加载:---持久化存储
            1>模型训练是一个耗时的过程，一个优秀的机器学习模型是非常宝贵的。
            可以将模型保存到磁盘中，也可以在需要使用的时候从磁盘中重新加载模型即可。不需要重新训练（即model.fit()）。
            2>模型保存和加载相关API：
                import pickle
                pickle.dump(model, 磁盘文件) # 保存模型
                model = pickle.load(磁盘文件)  # 加载模型


    示例：基于一元线性回归训练single.txt中的训练样本，使用模型预测测试样本。
        步骤：整理数据----->训练模型----->绘制图像----->评估模型
'''

import numpy as np
import sklearn.linear_model as sl
import matplotlib.pyplot as mp
import sklearn.metrics as sm

# 采集数据
x, y = np.loadtxt('./single.txt', delimiter=',', usecols=(0, 1), unpack=True)
print(x.shape)
print(y.shape)
# 把输入变成二维数组，一行一样本，一列一特征
x = x.reshape(-1, 1)  # 变成n行1列
model = sl.LinearRegression()
model.fit(x, y)
pred_y = model.predict(x)  # 把样本x带入模型求出预测y

# 输出模型的评估指标
print('平均绝对值误差：', sm.mean_absolute_error(y, pred_y))
print('平均平方误差：', sm.mean_squared_error(y, pred_y))
print('中位绝对值误差：', sm.median_absolute_error(y, pred_y))
print('R2得分：', sm.r2_score(y, pred_y))

# 绘制图像
mp.figure("Linear Regression", facecolor='lightgray')
mp.title('Linear Regression', fontsize=16)
mp.tick_params(labelsize=10)
mp.grid(linestyle=':')
mp.xlabel('x')
mp.ylabel('y')

mp.scatter(x, y, s=60, marker='o', c='dodgerblue', label='Points')
mp.plot(x, pred_y, c='orangered', label='LR Line')
mp.tight_layout()
mp.legend()
mp.show()