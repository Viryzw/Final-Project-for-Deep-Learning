import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        classes = sorted(os.listdir(root_dir))  # e.g. ['0','1','2',...]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # 单通道

        if self.transform:
            img = self.transform(img)

        return img, label


def get_loaders(root, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),               # (1,128,128)
        transforms.Normalize((0.5,), (0.5,)) # [-1,1]
    ])

    dataset = HandwritingDataset(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(dataset.samples[0])


import torch
import torch.nn as nn
import torch.nn.functional as F


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


import torch
import torch.optim as optim

def train():
    root = r"D:\事务夹\作业集\深度学习\final_project\dataset\Filtered_img"    # 你的数据目录
    batch_size = 32

    # 自动获取类别数
    classes = sorted(os.listdir(root))
    num_classes = len(classes)

    # DataLoader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = HandwritingDataset(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Model
    model = SimpleCNN(num_classes=num_classes)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    # Train
    for epoch in range(30):
        for imgs, labels in loader:
            imgs = imgs
            labels = labels

            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, labels)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} - loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "5cnn_2025_11_15.pth")
    print("Model saved.")


if __name__ == "__main__":
    train()
