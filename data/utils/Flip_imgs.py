import os
from PIL import Image
import shutil

# 源目录
src_root = r"D:\事务夹\作业集\深度学习\final_project\dataset_process\Rotate_img"
# 新目录（包含原图+翻转图）
dst_root = r"D:\事务夹\作业集\深度学习\final_project\dataset_process\Flip_img"

# 如果目录已存在，清空重建（可按需注释掉）
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)
os.makedirs(dst_root, exist_ok=True)

# 遍历子文件夹（0~9）
for class_name in os.listdir(src_root):
    src_class_path = os.path.join(src_root, class_name)
    if not os.path.isdir(src_class_path):
        continue

    dst_class_path = os.path.join(dst_root, class_name)
    os.makedirs(dst_class_path, exist_ok=True)

    print(f"Processing folder: {src_class_path}")

    for img_name in os.listdir(src_class_path):
        if not img_name.lower().endswith(".png"):
            continue

        src_img_path = os.path.join(src_class_path, img_name)
        img = Image.open(src_img_path)

        base_name, ext = os.path.splitext(img_name)

        # 1. 保存原图（复制）
        save_original_path = os.path.join(dst_class_path, f"{base_name}{ext}")
        img.save(save_original_path)

        # 2. 上下翻转
        vflip = img.transpose(Image.FLIP_TOP_BOTTOM)
        vflip.save(os.path.join(dst_class_path, f"{base_name}_flip_v{ext}"))

        # 3. 左右翻转
        hflip = img.transpose(Image.FLIP_LEFT_RIGHT)
        hflip.save(os.path.join(dst_class_path, f"{base_name}_flip_h{ext}"))

        img.close()

print("All original + flipped images saved to:", dst_root)
