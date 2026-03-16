# 1. 读取数据库
import os

os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')  # 列名
    f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd

data = pd.read_csv(data_file)
print(data)

# 2. 处理缺失值
inputs = data.iloc[:, 0:2]
outputs = data.iloc[:, 2]
inputs= inputs.fillna(inputs.mean())  # 用同一列的均值来填充空白
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)  # 将类别值进行0 1区分
print(inputs)

# 3. 转换为张量格式
import torch
X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))

print(X, y)