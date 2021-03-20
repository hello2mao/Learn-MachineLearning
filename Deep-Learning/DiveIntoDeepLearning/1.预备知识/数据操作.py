import torch

# 向量
x = torch.arange(10)
print(x)

# 矩阵
y = torch.arange(12).reshape(3, -1)
print(y)

# 连结（concatenate）
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(Y)
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))


