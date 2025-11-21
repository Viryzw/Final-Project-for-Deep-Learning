import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F

from utils import sample_concrete, save_mask_gray

class Selector(nn.Module):
    def __init__(self, k):
        super(Selector, self).__init__()
        
        self.k = k

        self.c1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.c2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.c3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.c4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.c5 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 1x1 conv to reduce channel
        self.out_conv = nn.Conv2d(512, 1, kernel_size=1)

        # Normalize logits to stabilize Selector
        # self.norm = nn.LayerNorm([4, 4])   # 你最后 feature map 是 4x4

        # upsample recovery
        self.upsample = nn.Upsample(scale_factor=32, mode='bilinear', align_corners=False)

    def forward(self, x, status, tau=0.1):
        bs = x.size(0)

        # 下采样
        o1 = self.c1(x)
        o2 = self.c2(o1)
        o3 = self.c3(o2)
        o4 = self.c4(o3)
        o5 = self.c5(o4)

        # logits shape = (B,1,4,4)
        logits = self.out_conv(o5)

        # 展平用于 Concrete/Gumbel-top-k
        logits_flat = logits.view(bs, -1)         # (B, 16)

        # ---- softmax 必须作用在最后一个维度 ----
        logits_norm = F.softmax(logits_flat / tau, dim=-1)

        # 采样 k 个位置
        selected_subset = sample_concrete(tau, logits_norm, status, self.k)

        # reshape 回 spatial map
        M = logits.size(-1)
        selected_subset = selected_subset.view(bs, 1, M, M)

        # 上采样回输入大小
        v = self.upsample(selected_subset)
        
        save_mask_gray(v)

        return v

# ====== 加载你的图片（灰度） ======
def load_image_as_tensor(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转 1 通道
        transforms.ToTensor(),                        # shape: [1,H,W]
    ])
    img = Image.open(img_path).convert("L")
    img_tensor = transform(img).unsqueeze(0)          # 变成 [B=1, 1, H, W]
    return img_tensor


# ====== 主测试函数 ======
def test_selector():
    # 选择前 k 个特征位置
    K = 40
    selector = Selector(k=K)

    # 载入图片
    img_path = "1.png"    # 换成你的路径
    x = load_image_as_tensor(img_path)

    # ----- 测试 Gumbel-Softmax 模式（status=True） -----
    print("Running Selector with status=True (Gumbel-Soft)...")
    with torch.no_grad():
        v_soft = selector(x, status=True, tau=0.1)   # soft sample
    print("v_soft shape:", v_soft.shape)

    # 保存结果（函数内部会自动保存）
    print("Saved masks to ./mask_gray/ ...")


if __name__ == "__main__":
    test_selector()
