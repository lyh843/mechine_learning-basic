# 1. 基础概率论
import torch
import matplotlib.pyplot as plt
from torch.distributions import multinomial
from d2l import torch as d2l

fair_probs = torch.ones([6]) / 6
counts = multinomial.Multinomial(10, fair_probs).sample() # 模拟十次
print(counts)
counts = multinomial.Multinomial(1000, fair_probs).sample() # 模拟1000次
print(counts / 1000)

counts = multinomial.Multinomial(10, fair_probs).sample((500,))
cum_counts = counts.cumsum(dim=0)
estimates = cum_counts / cum_counts.sum(dim=1, keepdims=True)   # 按行累加

d2l.set_figsize((6, 4.5))
for i in range(5):
    d2l.plt.plot(estimates[:, i].numpy(),
                 label=("P(die=" + str(i + 1) + ")"))
d2l.plt.axhline(y=0.167, color='black', linestyle='dashed')
d2l.plt.gca().set_xlabel('Groups of experiments')
d2l.plt.gca().set_ylabel('Estimated probability')
d2l.plt.legend()



plt.show()