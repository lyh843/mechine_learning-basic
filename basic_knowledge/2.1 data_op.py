import torch

# 1. 张量（向量）
x = torch.arange(12)  # 行向量
print(x)
print(x.shape)  # 访问张量（沿每个轴的长度）的形状
print(x.numel())  # 访问张量的元素总数

X = x.reshape(3,4) # 改变张量的形状
print(X)

print(torch.zeros((2, 3, 4))) # 全零矩阵
print(torch.ones((2, 3, 4)))    # 全一矩阵

print(torch.randn(3, 4)) # 创建一个随机张量。每个元素都从均值为0、标准差为1的正态分布中随机采样
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])) # 自定义张量

# 2. 运算符
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(x + y, x - y, x * y, x / y, x ** y)

print(torch.exp(x))

## 对矩阵的操作
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
print(X)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0)) # 沿不同的轴进行拼接 dim = 0
print(torch.cat((X, Y), dim=1)) # 沿不同的轴进行拼接 dim = 1

print(X == Y) # 判断两个矩阵的每个位置是否相同，给出结果矩阵

print(X.sum()) # 对矩阵中的所有元素求和

# 3. 广播机制
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)

print(a + b) # 由于形状对不上，因此将两个矩阵进行广播，接拓展为3 * 2的矩阵

# 4. 索引和切片
print(X)
print(X[-1])    # 倒数第一个 
print(X[1:3])   # 前闭后开

X[1, 2] = 100    # 值修改
print(X)

X[0:2, :] = 12  # ":"代表沿轴1（列）的所有元素
print(X)

# 5. 节省内存
Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
id(X) == before

# 6. 转换为其他python对象
A = X.numpy()
B = torch.tensor(A)
print(type(A), type(B))

a = torch.tensor([3.5])
print(a, a.item(), float(a), int(a))