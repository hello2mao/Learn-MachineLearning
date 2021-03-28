import numpy as np
import torch
from torch import nn
from torch.utils import data


# 生成数据集
def synthetic_data(w, b, num_examples):
    """生成 y = Xw + b + 噪声。"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


if __name__ == '__main__':
    # 创建带噪声的训练数据集
    true_w = torch.tensor([2, -3.4])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)

    # 构造一个PyTorch数据迭代器
    batch_size = 10
    dataset = data.TensorDataset(*(features, labels))
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)

    # 定义模型
    net = nn.Sequential(nn.Linear(2, 1))
    # 初始化模型参数
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)
    # 定义损失函数
    loss = nn.MSELoss()  # 均方误差
    # 定义优化算法
    trainer = torch.optim.SGD(net.parameters(), lr=0.03)
    # 训练
    num_epochs = 3
    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)  # 调用 net(X) 生成预测并计算损失 l（正向传播）
            trainer.zero_grad()
            l.backward()  # 进行反向传播来计算梯度
            trainer.step()  # 调用优化器来更新模型参数
        l = loss(net(features), labels)
        print(f'epoch {epoch + 1}, loss {l:f}')

    w = net[0].weight.data
    print('w的估计误差：', true_w - w.reshape(true_w.shape))
    b = net[0].bias.data
    print('b的估计误差：', true_b - b)




