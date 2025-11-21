import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import sample_concrete, save_mask_gray


# 完整模型
class FullModel(nn.Module):
    
    # k是待调超参
    def __init__(self, k):
        super(FullModel, self).__init__()
        
        self.encoder = Encoder()
        self.selector = Selector(k=k)
        self.classifier = Classifier(num_classes=10)
        
    def forward(self, x, status):
        z = self.encoder(x)
        weights = self.selector(x, status)
        masked_x = torch.mul(x, weights)
        z_hat = self.encoder(masked_x)
        y_hat = self.classifier(z_hat)
       
        return z, z_hat, y_hat, masked_x, weights
    
class Encoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(Encoder, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16x16
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8x8
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # 512x4x4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        feat = self.conv(x)  # [B,256,8,8]
        pooled = self.pool(feat).view(feat.size(0), -1)  # [B,256]
        emb = self.fc(pooled)  # [B,embedding_dim]
        emb = F.normalize(emb, p=2, dim=1)  # L2 normalize
        return emb


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
        
        # save_mask_gray(v)

        return v

class Classifier(nn.Module):
    def __init__(self, in_dim=128, hidden=128, num_classes=10):
        super(Classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes)
        )

    def forward(self, z):
        return self.fc(z)