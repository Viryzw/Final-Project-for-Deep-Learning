import torch
from utils import ExplainVisual
from model import FullModel
from dataloader import get_loaders

def final_test(model, device, test_loader, epoch): #test the model with hide-and-seek preprocess
    """
    在测试集上使用 Hide-and-Seek 预处理（遮挡 patch）后进行模型测试，计算准确率。

    参数:
        model: 训练好的模型
        device: cuda 或 cpu
        test_loader: 测试集的 DataLoader
        epoch: 当前 epoch（用于打印信息，可选）
        global_mean: float，全局像素均值，用于遮挡 patch
        patch_size: int，遮挡的 patch 大小，如 4
        processing_probability: float，每个 patch 被遮挡的概率，如 0.3

    返回:
        test_acc: float，测试集上的分类准确率（百分比）
    """
    model.eval()  # 设置为评估模式
    correct = 0
    total = 0

    with torch.no_grad():  # 不计算梯度
        save_visual = False
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            z, z_hat, y_hat, masked_x, weights = model(data, status=False) # 我们只关心 y_hat（分类结果）
            pred = y_hat.argmax(dim=1)  # 取概率最大的类别
            
            if save_visual:
                visualizer = ExplainVisual(
                    image_dir='./visualization',
                    data='Handwriting',
                    epoch=epoch, 
                    #batch=data,
                    batch = data,# this is the origin images
                    masked_x= masked_x,
                    mask= weights,
                    label=target,
                    label_approx=pred,
                    batch_idx=batch_idx
                #prob = prob
                )
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    # ======================
    # 4. 计算并打印测试准确率
    # ======================
    test_acc = 100. * correct / total
    print(f'\n===== Final Test (Epoch {epoch}) =====')
    print(f'Accuracy: {correct}/{total} ({test_acc:.2f}%)')

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(k=16).to(device)
    checkpoint = torch.load('checkpoints_k30\\best_version_01.pth', map_location=device)
    # 提取模型真正的参数部分
    state_dict = checkpoint['state_dict']

    # 有些训练脚本会多带 "module." 前缀（DataParallel 训练）
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # 去掉多余前缀
        new_state_dict[new_key] = v

    # 加载到模型
    model.load_state_dict(new_state_dict, strict=False)
    
    train_loader, valid_loader, test_loader = get_loaders()
    
    test_acc_hide_seek = final_test(
        model=model,
        device=device,
        test_loader=test_loader,
        epoch=1)  # change the processing_probability to see how it affects the accuracy when testing

if __name__ == '__main__':
    main()