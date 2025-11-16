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
        self.classifier = Classifier(num_classes=3)
        
    def forward(self, x, status):
        feat, z = self.encoder(x)
        weights = self.selector(feat, status)
        masked_x = torch.mul(x, weights)
        _, z_hat = self.encoder(masked_x)
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
            
            # nn.Conv2d(256, 512, 3, padding=1),
            # nn.BatchNorm2d(512),
            # nn.ReLU(),
            # nn.AdaptiveAvgPool2d((4, 4))  # 512x4x4
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, embedding_dim)
        )
        
        # self.fc = nn.Sequential(
        #     nn.Dropout(0.3),
        #     nn.Linear(512 * 4 * 4, 1024),
        #     nn.BatchNorm1d(1024),
        #     nn.ReLU(),
        #     nn.Dropout(0.3),
        #     nn.Linear(1024, embedding_dim)
        # )
        
        # L2归一化，便于计算余弦相似度
        # self.l2_norm = nn.functional.normalize
        
    def forward(self, x):
        # x = self.conv(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        # x = self.l2_norm(x, p=2, dim=1)
        # return x
        feat = self.conv(x)  # [B,256,8,8]
        pooled = self.pool(feat).view(feat.size(0), -1)  # [B,256]
        emb = self.fc(pooled)  # [B,embedding_dim]
        emb = F.normalize(emb, p=2, dim=1)  # L2 normalize
        return feat, emb
        

# CNN + 注意力
# class Selector(nn.Module):
#     def __init__(self, k):
#         super(Selector, self).__init__()
        
#         self.k = k
        
#         self.c1 = nn.Sequential(
#             nn.Conv2d(1, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 128 -> 64
#         )

#         self.c2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 64 -> 32
#         )

#         self.c3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 32 -> 16
#         )

#         self.c4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 16 -> 8
#         )

#         self.c5 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(),
#             nn.MaxPool2d(2)    # 8 -> 4
#         )

#         self.out_conv = nn.Conv2d(512, 1, kernel_size=1)
        
#         self.upsample = nn.Upsample(scale_factor=32, mode='bilinear')

#     def forward(self, x, tau=0.1):
#         bs = x.size(0)
        
#         # 下采样
#         o1 = self.c1(x)
#         o2 = self.c2(o1)
#         o3 = self.c3(o2)
#         o4 = self.c4(o3)
#         o5 = self.c5(o4)
#         logits = self.out_conv(o5)
        
#         M = logits.size(-1)
        
#         logits = logits.view(bs, -1)
#         selected_subset = sample_concrete(tau, logits, self.k)
#         selected_subset = selected_subset.view(bs, 1, M, M)
#         v = self.upsample(selected_subset)
        
#         # save_mask_gray(v)

#         return v



class Selector(nn.Module):
    def __init__(self, k):
        super(Selector, self).__init__()
        
        self.k = k

        # self.c1 = nn.Sequential(
        #     nn.Conv2d(1, 32, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # self.c2 = nn.Sequential(
        #     nn.Conv2d(32, 64, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # self.c3 = nn.Sequential(
        #     nn.Conv2d(64, 128, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

        # self.c4 = nn.Sequential(
        #     nn.Conv2d(128, 256, kernel_size=3, padding=1),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(),
        #     nn.MaxPool2d(2)
        # )

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

    def forward(self, feat, status, tau=0.1):
        bs = feat.size(0)

        # 下采样
        # o1 = self.c1(x)
        # o2 = self.c2(o1)
        # o3 = self.c3(o2)
        # o4 = self.c4(o3)
        o5 = self.c5(feat)

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

        return v






# def sample_concrete_topk(logits, k, tau=0.5, eps=1e-6):
#     """
#     logits: [B, D] (unnormalized logits for spatial locations)
#     Return: mask_soft: [B, D] continuous (soft), mask_st: straight-through (used in forward)
#     Implementation idea:
#       - compute soft = softmax((logits + gumbel)/tau)
#       - compute topk hard mask (1 on selected positions)
#       - straight-through: mask = hard.detach() - soft.detach() + soft
#     """
#     B, D = logits.shape
#     # Gumbel noise
#     u = torch.rand_like(logits).clamp(min=eps, max=1.0 - eps)
#     g = -torch.log(-torch.log(u))
#     y = (logits + g) / tau
#     soft = F.softmax(y, dim=-1)  # [B, D]

#     # get top-k indices from logits (without gumbel) to choose deterministic top-k
#     # using logits (not y) to stabilize selection
#     topk_vals, topk_idx = torch.topk(logits, k=min(k, D), dim=-1)  # [B,k]
#     # build hard mask
#     hard = torch.zeros_like(logits)
#     # scatter 1s
#     hard.scatter_(1, topk_idx, 1.0)

#     # straight-through
#     mask = hard.detach() - soft.detach() + soft  # forward behaves like hard (approx), backward uses soft grads
#     return mask, soft, hard




# class Selector(nn.Module):
#     def __init__(self, k=20, feat_channels=256):
#         super(Selector, self).__init__()
#         self.k = k
#         # take feat_map [B, C, H, W], produce spatial logits [B, 1, H, W]
#         self.conv = nn.Sequential(
#             nn.Conv2d(feat_channels, feat_channels // 2, kernel_size=3, padding=1),
#             nn.BatchNorm2d(feat_channels // 2),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(feat_channels // 2, 1, kernel_size=1)  # output single-channel spatial logits
#         )
#         # we will not upsample (feat map is already 8x8). mask has same spatial dims
#         # optional smoothing
#         self.smooth = nn.Identity()

#     def forward(self, feat_map, tau=0.5, return_debug=False):
#         """
#         feat_map: [B, C, H, W]  (H=W=8)
#         returns:
#           mask_st: [B,1,H,W] (straight-through)
#           mask_soft: [B,1,H,W] (soft probabilities)
#         """
#         B, C, H, W = feat_map.shape
#         logits = self.conv(feat_map)  # [B,1,H,W]
#         logits = self.smooth(logits)
#         flat = logits.view(B, -1)  # [B, D] where D = H*W

#         mask, soft, hard = sample_concrete_topk(flat, k=self.k, tau=tau)  # mask: [B,D]
#         mask_soft = soft.view(B, 1, H, W)
#         mask_st = mask.view(B, 1, H, W)
#         # multiply broadcast to channels later
#         if return_debug:
#             return mask_st, mask_soft, hard.view(B,1,H,W), logits
#         return mask_st, mask_soft

# class Classifier(nn.Module):
#     def __init__(self, in_dim=128, num_classes=3):
#         super(Classifier, self).__init__()
#         self.fc_out = nn.Sequential(
#             # nn.Linear(in_dim, 512),
#             # nn.ReLU(),
#             # nn.Linear(512, num_classes)
#             nn.Linear(in_dim, num_classes)
#         )

#     def forward(self, z):
#         return self.fc_out(z)
class Classifier(nn.Module):
    def __init__(self, in_dim=128, hidden=128, num_classes=3):
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