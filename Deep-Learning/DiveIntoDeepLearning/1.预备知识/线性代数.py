import torch

# 向量
x = torch.ones(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print('向量x: ', x)
print('向量y: ', y)

# 矩阵
A = torch.arange(12, dtype=torch.float32).reshape(3, 4)
B = torch.arange(20, dtype=torch.float32).reshape(4, 5)
print('矩阵A:', A)
print('矩阵A转置: ', A.T)  # 转置
print('矩阵A求和: ', A.sum())  # 和
print('矩阵A平均值: ', A.mean())  # 平均值

# 向量点积（Dot Product）
print('向量点积: ', torch.dot(x, y))
# 矩阵-向量积（matrix-vector products）
print('矩阵-向量积: ', torch.mv(A, x))
# 矩阵-矩阵乘法（matrix-matrix multiplication）
print('矩阵-矩阵乘法: ', torch.mm(A, B))

# 矩阵 连结（concatenate）
C = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print('矩阵C: ', C)
print('连结:', torch.cat((A, C), dim=0))
print('连结:', torch.cat((A, C), dim=1))

# 范数
u = torch.tensor([3.0, -4.0])
print('L1范数:', torch.abs(u).sum())  # L1范数
print('L2范数:', torch.norm(u))  # L2范数
print(torch.norm(torch.ones((4, 9))))  # 弗罗贝尼乌斯范数
