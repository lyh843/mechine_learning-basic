# realtime_mnist_gui.py
import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageFilter
import torch
from torch import nn
import numpy as np

# ---------- CNN 模型定义（需与训练时一致） ----------
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

# ---------- 加载模型 ----------
device = torch.device("cpu")
model = LeNet()
model.load_state_dict(torch.load("mnist_cnn.pth", map_location=device))
model.to(device)
model.eval()

softmax = nn.Softmax(dim=1)

# ---------- GUI 与绘图状态 ----------
WIDTH, HEIGHT = 280, 280          # 画布像素
PEN_WIDTH = 18                    # 笔触宽度

root = tk.Tk()
root.title("实时手写数字识别（MNIST-CNN）")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg='white')
canvas.grid(row=0, column=0, columnspan=4)

# PIL 图像
pil_image = Image.new('L', (WIDTH, HEIGHT), color=255)
draw = ImageDraw.Draw(pil_image)

# 标签显示预测结果
label_var = tk.StringVar()
label_var.set("请在画板上书写数字（0-9）")
label = tk.Label(root, textvariable=label_var, font=("Arial", 16))
label.grid(row=1, column=0, columnspan=4, sticky="w")

_last_x, _last_y = None, None
after_id = None

# ---------- 预测函数 ----------
def predict_from_pil_image(img_pil):
    """
    将 PIL 图像（白底黑字）预处理成 CNN 输入并预测。
    """
    im = img_pil.copy()
    im = im.filter(ImageFilter.GaussianBlur(radius=0.5))
    im = im.resize((28, 28), Image.LANCZOS)

    arr = np.array(im).astype(np.float32) / 255.0
    arr = 1.0 - arr  # 反色: 白底黑字 -> 黑底白字

    # 加归一化（与训练一致）
    mean, std = 0.1307, 0.3081
    arr = (arr - mean) / std

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # shape [1,1,28,28]
    with torch.no_grad():
        logits = model(tensor.to(device))
        probs = softmax(logits).cpu().numpy()[0]
        pred = int(np.argmax(probs))
        conf = float(probs[pred])
    return pred, conf

def update_prediction_debounced():
    global after_id
    after_id = None
    pred, conf = predict_from_pil_image(pil_image)
    label_var.set(f"预测: {pred}，置信度: {conf:.2%}")

def schedule_prediction_debounce(delay=150):
    global after_id
    if after_id is not None:
        root.after_cancel(after_id)
    after_id = root.after(delay, update_prediction_debounced)

# ---------- 鼠标事件 ----------
def on_button_press(event):
    global _last_x, _last_y
    _last_x, _last_y = event.x, event.y
    canvas.create_oval(event.x - PEN_WIDTH//2, event.y - PEN_WIDTH//2,
                       event.x + PEN_WIDTH//2, event.y + PEN_WIDTH//2,
                       fill='black', outline='black')
    draw.ellipse([event.x - PEN_WIDTH//2, event.y - PEN_WIDTH//2,
                  event.x + PEN_WIDTH//2, event.y + PEN_WIDTH//2],
                 fill=0)
    schedule_prediction_debounce()

def on_move(event):
    global _last_x, _last_y
    if _last_x is not None and _last_y is not None:
        canvas.create_line(_last_x, _last_y, event.x, event.y,
                           width=PEN_WIDTH, capstyle=tk.ROUND, smooth=True)
        draw.line([_last_x, _last_y, event.x, event.y], fill=0, width=PEN_WIDTH)
    _last_x, _last_y = event.x, event.y
    schedule_prediction_debounce()

def on_button_release(event):
    global _last_x, _last_y
    _last_x, _last_y = None, None
    schedule_prediction_debounce(delay=80)

canvas.bind("<ButtonPress-1>", on_button_press)
canvas.bind("<B1-Motion>", on_move)
canvas.bind("<ButtonRelease-1>", on_button_release)

# ---------- 控制按钮 ----------
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0, 0, WIDTH, HEIGHT], fill=255)
    label_var.set("已清空，继续书写。")

def manual_predict():
    pred, conf = predict_from_pil_image(pil_image)
    label_var.set(f"手动预测: {pred}，置信度: {conf:.2%}")

btn_clear = tk.Button(root, text="清空", command=clear_canvas, width=10)
btn_clear.grid(row=2, column=0)

btn_predict = tk.Button(root, text="手动预测", command=manual_predict, width=12)
btn_predict.grid(row=2, column=1)

btn_quit = tk.Button(root, text="退出", command=root.destroy, width=10)
btn_quit.grid(row=2, column=2)

info = tk.Label(root, text="提示：按住左键绘制。程序会在你绘制时自动识别。", font=("Arial", 10))
info.grid(row=3, column=0, columnspan=4, sticky="w")

root.mainloop()
