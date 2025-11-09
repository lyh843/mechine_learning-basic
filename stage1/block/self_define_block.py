import d2l
from d2l import torch
from torch import nn
from torch import m

# 5.1.1 自定义块
class MLP(nn.Module):
    # 用模型参数声明层。
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.out = nn.Linear(256, 10)
        
    def forward(self, x):
        return self.out(nn.functional.relu(self.hidden(x)))
    
# 5.1.2 顺序块
class MySequential(nn.Module):
    def __init__(self, *args):
        super().__init__()
        for idx, module in enumerate(args):
            self._modules[str(idx)] = module
            
    def forward(self, x):
        for block in self._modules.values():
            x = block(x)
        return x
    
# 5.1.3 在前向传播函数中执行代码
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)
        
    def forward(self, x):
        x = self.linear(x)
        x = nn.functional.relu(torch.)