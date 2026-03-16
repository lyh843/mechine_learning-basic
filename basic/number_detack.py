from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from torch import nn

# 超参数
batch_size, lr, num_epochs = 256, 0.08, 20

# 数据加载
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=False,
    transform=transforms.ToTensor()
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
)

num_input, num_hidden1, num_hidden2, num_output = 28 * 28, 16 * 16, 8 * 8, 10

net = nn.Sequential(nn.Flatten(),
                   nn.Linear(num_input, num_hidden1),
                   nn.ReLU(),
                   nn.Linear(num_hidden1, num_hidden2),
                   nn.ReLU(),
                   nn.Linear(num_hidden2, num_output)
)

num_input, num_hidden1, num_hidden2, num_output = 28 * 28, 16 * 16, 8 * 8, 10

net = nn.Sequential(nn.Flatten(),
                   nn.Linear(num_input, num_hidden1),
                   nn.ReLU(),
                   nn.Linear(num_hidden1, num_hidden2),
                   nn.ReLU(),
                   nn.Linear(num_hidden2, num_output))

def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

loss_fn = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

for epoch in range(num_epochs):
    net.train()
    total_loss = 0
    for X, y in train_loader:
        
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        
        trainer.zero_grad()
        loss.backward()
        trainer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    print(f"epoch = {epoch + 1}, avg_loss = {avg_loss}.")
    
# 保存模型权重（供 GUI 加载）
torch.save(net.state_dict(), "mnist_net.pth")
print("模型已保存到 mnist_net.pth")
    