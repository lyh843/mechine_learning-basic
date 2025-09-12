# ... existing code ...
from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
from torch import nn
# import time

# # 检查是否有可用的GPU并设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU device name: {torch.cuda.get_device_name(0)}")

# 超参数
batch_size, lr, num_epochs = 256, 0.09, 20

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
    num_workers=2,  # 使用多进程加速数据加载
    pin_memory=True  # 使用pin_memory加速GPU数据传输
)

# test_dataset = datasets.MNIST(
#     root="./data",
#     train=False,
#     download=False,
#     transform=transforms.ToTensor()
# )
# test_loader = torch.utils.data.DataLoader(
#     test_dataset, 
#     batch_size=batch_size, 
#     shuffle=False,
#     num_workers=2,
#     pin_memory=True
# )

num_input, num_hidden, num_output = 28 * 28, 256, 10

net = nn.Sequential(
    nn.Flatten(),
    nn.Linear(num_input, num_hidden),
    nn.ReLU(),
    nn.Linear(num_hidden, num_output)
)

def init_weight(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weight)

# # 将模型移到GPU
# net.to(device)
# print(f"Model moved to {device}")

# 使用更好的优化器
loss_fn = nn.CrossEntropyLoss()
# # 使用Adam优化器，通常收敛更快
# trainer = torch.optim.Adam(net.parameters(), lr=lr)
# 或者继续使用SGD但加上动量项
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# # 添加测试集评估函数
# def evaluate_accuracy(data_loader, model):
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for X, y in data_loader:
#             # 将数据移到GPU
#             # X, y = X.to(device), y.to(device)
#             y_hat = model(X)
#             _, predicted = torch.max(y_hat.data, 1)
#             total += y.size(0)
#             correct += (predicted == y).sum().item()
#     model.train()
#     return correct / total

# # 记录训练开始时间
# start_time = time.time()

# 训练过程
# print("Starting training...")
for epoch in range(num_epochs):
    # epoch_start = time.time()
    net.train()
    total_loss = 0
    
    for batch_idx, (X, y) in enumerate(train_loader):
        # 将数据移到GPU
        # X, y = X.to(device), y.to(device)
        y_hat = net(X)
        loss = loss_fn(y_hat, y)
        
        trainer.zero_grad()
        loss.backward()
        trainer.step()
        
        total_loss += loss.item()
        
        # # 显示训练进度
        # if batch_idx % 100 == 0:
        #     print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    # # 计算训练集和测试集准确率
    # train_acc = evaluate_accuracy(train_loader, net)
    # test_acc = evaluate_accuracy(test_loader, net)
    avg_loss = total_loss / len(train_loader)
    # epoch_time = time.time() - epoch_start
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

# training_time = time.time() - start_time
# print(f"Total training time: {training_time:.2f}s")

# # ========================
# # 可视化测试
# # ========================
# print("Generating visualization...")
# net.eval()
# X_test, y_test = next(iter(test_loader))
# # 将数据移到GPU进行推理
# # X_test, y_test = X_test.to(device), y_test.to(device)
# with torch.no_grad():
#     y_pred = net(X_test).argmax(dim=1)

# # 将数据移回CPU进行可视化
# X_test = X_test.cpu()
# y_test = y_test.cpu()
# y_pred = y_pred.cpu()

# fig, axes = plt.subplots(3, 3, figsize=(8, 8))
# for i, ax in enumerate(axes.flat):
#     if i < len(X_test):
#         ax.imshow(X_test[i].squeeze(), cmap='gray')
#         ax.set_title(f"True: {y_test[i].item()} Pred: {y_pred[i].item()}")
#         ax.axis('off')
# plt.suptitle("MNIST Test Results", fontsize=16)
# plt.tight_layout()
# plt.show()
# # ... existing code ...