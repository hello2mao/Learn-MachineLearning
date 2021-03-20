import torch

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A)
print(A.T)  # 转置
print(A.sum())  # 和
print(A.mean())  # 平均值

x = torch.ones(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(torch.dot(x, y))  # 点积

print(torch.mv(A, x))  # 矩阵-向量积

B = torch.arange(20, dtype=torch.float32).reshape(4, 5)
print(torch.mm(A, B))  # 矩阵-矩阵乘法

u = torch.tensor([3.0, -4.0])
print(torch.abs(u).sum())  # L1范数
print(torch.norm(u))  # L2范数

print(torch.norm(torch.ones((4, 9))))  # 弗罗贝尼乌斯范数
