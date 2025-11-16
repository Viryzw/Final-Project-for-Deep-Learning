import os
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms


# -------------------------
# 1. 载入你之前的 SimpleCNN
# -------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(SimpleCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 4, 3, padding=1),  # (B,4,128,128)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,4,64,64)

            nn.Conv2d(4, 8, 3, padding=1),  # (B,8,64,64)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,8,32,32)

            nn.Conv2d(8, 16, 3, padding=1),  # (B,16,32,32)
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,16,16)

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,8,8)

            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # (B,16,4,4)
        )

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(16 * 4 * 4, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# -------------------------
# 2. 图像数据预处理
# -------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # [1, 128, 128]
    transforms.Normalize((0.5,), (0.5,))
])


# -------------------------
# 3. 预测单张图片
# -------------------------
def predict(model, img_path, classes):
    img = Image.open(img_path).convert("L")
    img = img.resize((128, 128))  # 保证尺寸一致

    tensor = transform(img).unsqueeze(0)  # [1,1,128,128]
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()

    return pred


# -------------------------
# 4. 遍历文件夹并可视化
# -------------------------
def visualize_folder(model_path, test_folder, classes):
    num_classes = len(classes)

    # Load model
    model = SimpleCNN(num_classes=num_classes)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    # Collect image paths
    imgs = [
        os.path.join(test_folder, f)
        for f in os.listdir(test_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    max_show = 40
    imgs = imgs[:max_show]

    # 设置 5×8 网格（40张）
    rows = 5
    cols = 8

    plt.figure(figsize=(14, 9))   # 整体画布放小一些也可以改

    for i, img_path in enumerate(imgs):
        pred = predict(model, img_path, classes)
        label = classes[pred]
        img = Image.open(img_path).convert("L")

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"{label}", fontsize=8)  # 小字体
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# -------------------------
# 5. 入口
# -------------------------
if __name__ == "__main__":
    model_path = "5cnn_2025_11_15.pth"  # 训练好的模型
    test_folder = r"D:\事务夹\作业集\深度学习\final_project\test_imgs"  # 想要预测的图片文件夹

    # 如果你的类别是 0, 1, 2, 3...
    classes = sorted([d for d in os.listdir(r"D:\事务夹\作业集\深度学习\final_project\dataset\Filtered_img") if d.isdigit()])

    visualize_folder(model_path, test_folder, classes)
