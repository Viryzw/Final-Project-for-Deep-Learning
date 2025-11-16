import torch
import torch.optim as optim
from DataLoader import get_triplet_loader
from FeatureExtractor import SiameseNetwork
from TripletLoss import TripletLoss
import os

def train_zero_shot():
    root = r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img"
    batch_size = 32
    embedding_dim = 128
    margin = 1.0
    learning_rate = 1e-4
    num_epochs = 50
    
    # 数据加载器
    train_loader, num_triplets = get_triplet_loader(root, batch_size=batch_size)
    print(f"Generated {num_triplets} triplets for training")
    
    # 模型和损失
    model = SiameseNetwork(embedding_dim=embedding_dim)
    criterion = TripletLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 训练
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (anchor, positive, negative) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # 提取特征
            anchor_feat = model.feature_extractor(anchor)
            positive_feat = model.feature_extractor(positive)
            negative_feat = model.feature_extractor(negative)
            
            # 计算损失
            loss = criterion(anchor_feat, positive_feat, negative_feat)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
    
    # 保存模型
    torch.save(model.state_dict(), "zero_shot_handwriting.pth")
    print("Zero-shot model saved.")

if __name__ == "__main__":
    train_zero_shot()