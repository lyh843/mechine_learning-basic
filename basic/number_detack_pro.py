from torchvision import datasets, transforms
import torch
from torch import nn
import torch.optim as optim

# 超参数
batch_size, lr, num_epochs = 128, 0.001, 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据增强 + 归一化
transform_train = transforms.Compose([
    transforms.RandomRotation(10),          # 随机旋转 ±10°
    transforms.RandomAffine(0, translate=(0.1, 0.1)),  # 随机平移
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))         # MNIST 均值/方差归一化
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 数据集
train_dataset = datasets.MNIST(root="./data", train=True, download=False, transform=transform_train)
test_dataset  = datasets.MNIST(root="./data", train=False, download=False, transform=transform_test)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False)

# CNN 模型（LeNet风格）
class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

net = LeNet().to(device)

# 损失函数 + 优化器
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

# 评估函数
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            preds = outputs.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total * 100

# 训练循环
for epoch in range(num_epochs):
    net.train()
    total_loss, correct, total = 0, 0, 0

    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = y_hat.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    train_acc = correct / total * 100
    test_acc = evaluate(net, test_loader)
    avg_loss = total_loss / len(train_loader)

    print(f"epoch {epoch+1}/{num_epochs} | loss={avg_loss:.4f} | "
          f"train_acc={train_acc:.2f}% | test_acc={test_acc:.2f}%")

# 保存模型
torch.save(net.state_dict(), "mnist_cnn.pth")
print("模型已保存到 mnist_cnn.pth")