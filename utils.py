import torch
import torch.nn as nn
import shutil
import os
from PIL import Image
import numpy as np

def sample_concrete(tau, logits, k=10):
    # input logits dimension: [batch_size,1,d]
    logits = logits.unsqueeze(1)
    d = logits.shape[2]
    batch_size = logits.shape[0]

    uniform = torch.clamp(torch.rand(size=(batch_size, k, d), device=logits.device), min=1e-4, max=0.9999)#(1 - 0) * torch.rand(unif_shape) # generating vector of shape "unif_shape", uniformly random numbers in the interval [0,1)
    gumbel = - torch.log(-torch.log(uniform)) # generating gumbel noise/variables
    noisy_logits = (gumbel + logits)/tau # perturbed logits(perturbed by gumbel noise and temperature coeff. tau)
    samples = torch.softmax(noisy_logits, dim = -1) # sampling from softmax
    samples,_ = torch.max(samples, dim = 1)
    return samples

def save_checkpoint(state, path,file_name, is_best):
    save_path = os.path.join(path, file_name)
    torch.save(state, save_path)
    
    if is_best:
        shutil.copyfile(save_path, os.path.join(path, 'best__version.pth'))
        
        
def save_mask_gray(v, save_dir="./mask_gray", mean=None, std=None):
    """
    保存单通道 mask 为灰度图
    v: [B, 1, H, W] 或 [B, H, W] 单通道 mask
    mean, std: 若 mask 经过归一化，则提供反归一化参数（列表或 None）
    """
    os.makedirs(save_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(v.size(0)):
            mask = v[i].clone().detach().cpu()  # [1,H,W] 或 [H,W]

            # 如果是 [1,H,W]， squeeze 到 [H,W]
            if mask.ndim == 3 and mask.size(0) == 1:
                mask = mask.squeeze(0)

            # === 可选反归一化 ===
            if mean is not None and std is not None:
                # 如果 mask 是单通道，mean/std 也只取第0个
                mask = mask * std[0] + mean[0]

            # === 归一化到 [0,1] ===
            mask_gray = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)

            # === 转 uint8 ===
            mask_uint8 = (mask_gray.numpy() * 255).astype(np.uint8)
            img = Image.fromarray(mask_uint8, mode='L')  # 'L' 表示灰度图

            # === 保存 ===
            save_path = os.path.join(save_dir, f"mask_gray_{i}.png")
            img.save(save_path)
            # print(f"[Saved] {save_path}")