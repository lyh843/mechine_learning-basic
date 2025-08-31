# 1. 标量
import torch

x = torch.tensor(3.0)
y = torch.tensor(2.0)

print(x + y, x * y, x / y, x ** y)

# 2. 向量
x = torch.arange(4)
print(x)
print(x[2])     # 访问向量中的元素
print(len(x))   # 访问张量的长度
print(x.shape)

# 3. 矩阵
A = torch.arange(20).reshape(5, 4)
print(A)
print(A.T)

# 4. 张量
X = torch.arange(24).reshape(2, 3, 4)
print(X)

# 5. 张量乘法的基本性质
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()
print(A)
print(A + B)
print(A * B)

a = 2
X = torch.arange(24).reshape(2, 3 ,4)
print(a + X)
print((a * X).shape)

# 6. 降维
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())

print(A.shape, A.sum())

print(A)
A_sum_axis0 = A.sum(axis = 0)   # 降维成行
print(A_sum_axis0, A_sum_axis0.shape)

A_sum_axis1 = A.sum(axis=1)     # 降维成列
print(A_sum_axis1, A_sum_axis1.shape)

print(A.sum(axis=[0, 1]))       # 同时沿行和列压缩

print(A.mean(), A.sum() / A.numel())  # 求整体平均值

print(A.mean(axis=0), A.sum(axis=0) / A.shape[0])   # 求指定轴降维后的平均值

sum_A = A.sum(axis=1, keepdim=True)     # 求和的同时保留轴数
print(sum_A)
print(A / sum_A)

# 7. 点积
y = torch.ones(4, dtype = torch.float32)
print(x, y, torch.dot(x, y))    # 对所有相同位置元素相乘，然后求和
print(torch.sum(x * y))

# 8. 向量积
print(A.shape)
print(x.shape)
print(torch.mv(A, x))   # 矩阵乘向量

# 9. 矩阵-矩阵乘法
B = torch.ones(4,3)
print(torch.mm(A, B))   # 矩阵乘矩阵

# 10. 范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))    # 欧几里得距离
print(torch.abs(u).sum())   # 曼哈顿范数