import torch
import torch.nn as nn
import shutil
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2

def sample_concrete(tau, logits, status, k=10):
    if (status == True):
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
    
    else:
        logits = logits.squeeze(1)
        batch_size, num_features = logits.shape  # 例如 [32, 196]
        discrete_logits = torch.zeros(batch_size, num_features, device=logits.device)  # [32, 196]
        vals, ind = torch.topk(logits, k, dim=1)  # ind 形状是 [32, k]
    # 对每个 batch 单独设置 top-k 位置为 1
        discrete_logits.scatter_(1, ind, 1)  # [32, 196]
        return discrete_logits
        

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
            
            
            
class ExplainVisual(object):
    def __init__(self, image_dir, data, epoch, batch, masked_x, mask, label, label_approx,batch_idx):#, prob
        self.figsize_scale = 2
        self.image_dir = image_dir
        self.data = data ## data type
        self.epoch = epoch
        self.batch = batch ## image batches not batch size
        self.label = label
        self.label_approx = label_approx
        self.batch_idx = batch_idx
        self.masked_x = masked_x  # Masked input
        self.mask = mask#.view(masked_x.size())
        #self.prob = prob
        self.cmap = 'gist_heat_r'#'hot'
        self.width = self.batch.size(-1)
        self.height = self.batch.size(-2)
        self.output = self.visualize()
    def visualize(self):
        with torch.no_grad():
        # === 1. 从 self 中取出所有需要显示的 Tensor，并转移到 CPU ===
            img = self.batch.detach().cpu()            # 原图 [B, 3, H, W]，可能是归一化后的
            masked_x = self.masked_x.detach().cpu()    # masked 图（也是归一化后的）
            mask = self.mask.detach().cpu()            # mask，通常是 [0,1]，可以是软 mask 或二值

            label = self.label.detach().cpu()          # 真实标签 [B]
            label_approx = self.label_approx.detach().cpu()  # 模型预测标签 [B]
            batch_idx = self.batch_idx

        # === 2. 定义 CIFAR-10 图像的 Mean 和 Std（必须与训练时一致！）===
            # 灰度图的 mean/std
            mean = torch.tensor([0.5], dtype=torch.float32)  # [1]
            std  = torch.tensor([0.5], dtype=torch.float32)  # [1]

            def denormalize(tensor, mean=0.5, std=0.5):
                """
                tensor: [C,H,W] 或 [B,C,H,W]
                mean/std: 单通道灰度图
                """
                # 兼容 batch
                if tensor.dim() == 4:  # [B,C,H,W]
                    c = tensor.size(1)
                    mean_t = torch.tensor([mean], device=tensor.device, dtype=tensor.dtype).view(1, c, 1, 1)
                    std_t  = torch.tensor([std], device=tensor.device, dtype=tensor.dtype).view(1, c, 1, 1)
                elif tensor.dim() == 3:  # [C,H,W]
                    c = tensor.size(0)
                    mean_t = torch.tensor([mean], device=tensor.device, dtype=tensor.dtype).view(c, 1, 1)
                    std_t  = torch.tensor([std], device=tensor.device, dtype=tensor.dtype).view(c, 1, 1)
                else:
                    raise ValueError("tensor must be [C,H,W] or [B,C,H,W]")
                
                return tensor * std_t + mean_t

            
        # === 4. 创建画布 ===
            n_img = img.size(0)  # 一批有多少张图
            fig_per_img = 3  # 每张原图对应 4 个子图：原图、 masked_x、mask
            n_row = fig_per_img  # 每行显示几张原图（控制布局）
            # n_col = n_img // n_row + (1 if n_img % n_row != 0 else 0)  # 计算需要的列数
            n_col = n_img

            one_line = n_col * fig_per_img  # 每张原图占多少列（4个子图）

            # fig = plt.figure(figsize=(n_col * 4, n_row * fig_per_img * 4))  # 每个子图 4x4 inch
            fig = plt.figure(figsize=(n_col * 3, n_row * 3))  # 每个子图 4x4 inch
            att_filename = Path(self.image_dir).joinpath(f'visualization_epoch_{self.epoch}_batch_{batch_idx}.png')

            # === 5. 遍历每一张图，画出 4 个子图 ===
            for i in range(n_img):
                # 反归一化
                orig_img = denormalize(img[i], mean, std)
                masked_img = denormalize(masked_x[i], mean, std)
                curr_mask = mask[i]

                # 每个样本对应一列，行号代表不同类型
                plt.subplot(n_row, n_col, 1 * n_col + i + 1 - n_col)  # 原图
                plt.axis('off')
                plt.imshow(orig_img.permute(1, 2, 0))
                plt.title(f'Orig\n(L:{label[i].item()}, P:{label_approx[i].item()})', fontsize=8)

                plt.subplot(n_row, n_col, 2 * n_col + i + 1 - n_col)  # Masked
                plt.axis('off')
                plt.imshow(masked_img.permute(1, 2, 0))
                plt.title('Masked', fontsize=8)

                plt.subplot(n_row, n_col, 3 * n_col + i + 1 - n_col)  # Mask
                plt.axis('off')
                mask_local = curr_mask.detach().cpu().squeeze()
                mask_local = (mask_local - mask_local.min()) / (mask_local.max() - mask_local.min() + 1e-8)
                plt.imshow(mask_local)

                plt.title('Mask (RGB)', fontsize=8)            

        # 调整布局防止重叠
            plt.subplots_adjust(wspace=0.1, hspace=0.4)
            plt.tight_layout()

        # 保存图像
            fig.savefig(str(att_filename), bbox_inches='tight', dpi=150)
            plt.close(fig)
            return None