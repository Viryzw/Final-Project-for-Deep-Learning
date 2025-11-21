import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import torch.nn.functional as F
import random

from model import FullModel
from dataloader import get_loaders
from utils import save_checkpoint

def set_seed(seed=42):
    random.seed(seed)                         # Python 随机
    np.random.seed(seed)                      # Numpy 随机
    torch.manual_seed(seed)                   # PyTorch CPU 随机
    torch.cuda.manual_seed(seed)              # PyTorch GPU 随机
    torch.cuda.manual_seed_all(seed)          # 多GPU

    # cuDNN 固定模式
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 环境变量（影响 dataloader 多进程）
    os.environ['PYTHONHASHSEED'] = str(seed)


def train(model, device, train_loader, optimizer, epoch, lamda):
    model.train()
    correct = 0
    
    total = 0
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)   
        
        optimizer.zero_grad()
        z, z_hat, y_hat, masked_x, weights = model(data, status=True)
        
        CE_loss = nn.CrossEntropyLoss()(y_hat, target)
        # cosine_loss = 1 - F.cosine_similarity(z, z_hat).mean()
        # cosine_loss = 1 - F.cosine_similarity(z, z_hat, dim=1).mean()
        cosine_loss = torch.clamp(1 - F.cosine_similarity(z, z_hat, dim=1).mean(), 0, 1)
        cosine_loss_base = torch.clamp(1 - F.cosine_similarity(z, z, dim=1).mean(), 0, 1)
        
        selector_reg = 1e-3 * weights.pow(2).mean()

        loss = CE_loss + lamda * cosine_loss + selector_reg
              
        # loss = CE_loss + lamda * cosine_loss
        loss.backward()
        optimizer.step()
        # 保留 z 和 z_hat 的梯度
        _, predicted = y_hat.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        train_acc = 100. * correct / len(train_loader.dataset)
        if i % 500 == 0:
            print(f'Train Epoch: {epoch} [{i * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item():.6f} CE_loss: {CE_loss.item():.6f} cosine_loss: {cosine_loss.item():.6f} cosine_loss_base: {cosine_loss_base.item():.6f} Acc: {train_acc:.2f}%')

    train_acc = 100. * correct / len(train_loader.dataset)
    print(f"epoch accuracy:{train_acc}%")

    return train_acc

def validate(model, device, val_loader, epoch):
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            z, z_hat, y_hat, masked_x, weights = model(data, status=True)
 
            # pred = y_hat.argmax(dim=1, keepdim=True)
            # correct += pred.eq(target.view_as(pred)).sum().item()
            #running_loss += loss.item()
            _, predicted = y_hat.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()



    val_acc = 100. * correct / total
    print(f'Validation set - Epoch: {epoch}, Accuracy: {correct}/{total} ({val_acc:.2f}%)')

    return val_acc



def main(epochs = 25, lr = 3e-4, lamda = 0.1, selected_k = 30):
    set_seed(42)

    best_acc = 0
    print(f"training {epochs} epochs with lr {lr},lamda {lamda},selected_k {selected_k}")
    
    # 数据加载与划分
    train_loader, valid_loader, test_loader = get_loaders()
    
    # 初始化模型和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(k=selected_k).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)
    # optimizer = optim.SGD(model.parameters(),lr=0.001, momentum=0.9, weight_decay=1e-4)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)#3e-4
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.1)#每1个epoch学习率乘以0.1
    # scheduler = WarmupCosineSchedule(optimizer, warmup_steps=1500, t_total=len(train_loader) / 8 * epochs)

    train_acces = []
    val_acces = []
    
    save = True


    for epoch in tqdm(range(1, epochs + 1), desc="Training Epochs"):
        train_acc = train(model, device, train_loader, optimizer, epoch, lamda=lamda)

        train_acces.append(train_acc)
        
        val_acc = validate(model, device, valid_loader, epoch)
        val_acces.append(val_acc)

        # ✅ 改为根据验证准确率判断最优模型
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)

        if save:
            save_path = f'./checkpoints'
            os.makedirs(save_path, exist_ok=True)
            save_checkpoint(
                {"epoch": epoch,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": scheduler.state_dict()},
                path=save_path,
                file_name=f'model_epoch{epoch}.pth',
                is_best=is_best
            )
        print(f'Model saved at epoch_{epoch}' if is_best else f'Epoch {epoch} completed')
        scheduler.step()

if __name__ == "__main__":
    main()