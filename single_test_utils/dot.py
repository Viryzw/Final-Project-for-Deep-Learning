import torch
import numpy as np
from PIL import Image

def load_gray(path):
    """加载灰度图为 tensor，范围 [0,1]，shape [H,W]"""
    img = Image.open(path).convert("L")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)

def save_gray(tensor, path):
    """保存 [H,W] tensor 为灰度图"""
    arr = tensor.clamp(0, 1).numpy() * 255
    arr = arr.astype(np.uint8)
    Image.fromarray(arr, mode="L").save(path)

def dot_product_image(img1_path, img2_path, save_path="dot.png"):
    # 读取两张灰度图
    t1 = load_gray(img1_path)  # [H,W]
    t2 = load_gray(img2_path)  # [H,W]

    # 尺寸检查
    if t1.shape != t2.shape:
        raise ValueError(f"尺寸不一致: {t1.shape} vs {t2.shape}")

    # ---- 点积（逐点相乘）----
    dot = t1 * t2          # 仍然是 [H,W]

    # ---- 归一化到 [0,1] 用于保存 ----
    dot_norm = (dot - dot.min()) / (dot.max() - dot.min() + 1e-8)

    # 保存结果
    save_gray(dot_norm, save_path)
    print(f"Saved result to {save_path}")

    return dot_norm


# ====== 示例调用 ======
if __name__ == "__main__":
    dot_product_image("1.png", "mask.png", "dot_result.png")
