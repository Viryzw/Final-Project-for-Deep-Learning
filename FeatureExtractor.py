import torch
import torch.nn as nn
import torch.nn.functional as F

class FeatureExtractor(nn.Module):
    def __init__(self, embedding_dim=128):
        super(FeatureExtractor, self).__init__()
        
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
        
        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512 * 4 * 4, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(1024, embedding_dim)
        )
        
        # L2归一化，便于计算余弦相似度
        self.l2_norm = nn.functional.normalize
        
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.l2_norm(x, p=2, dim=1)
        return x

class SiameseNetwork(nn.Module):
    def __init__(self, embedding_dim=128):
        super(SiameseNetwork, self).__init__()
        self.feature_extractor = FeatureExtractor(embedding_dim)
        
    def forward(self, x1, x2=None):
        if x2 is None:
            # 单张图片，只提取特征
            return self.feature_extractor(x1)
        else:
            # 两张图片，分别提取特征并计算相似度
            feat1 = self.feature_extractor(x1)
            feat2 = self.feature_extractor(x2)
            return self.cosine_similarity(feat1, feat2)
    
    def cosine_similarity(self, feat1, feat2):
        # 计算余弦相似度
        return F.cosine_similarity(feat1, feat2)
    
    def euclidean_distance(self, feat1, feat2):
        # 计算欧氏距离（转换为相似度）
        distance = torch.pairwise_distance(feat1, feat2, p=2)
        return 1.0 / (1.0 + distance)  # 将距离转换为相似度