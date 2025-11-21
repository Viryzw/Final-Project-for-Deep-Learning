import os
from PIL import Image
import shutil

# 原始数据集路径
src_root = r"D:\事务夹\作业集\深度学习\final_project\dataset_process\Splited_img"

# 新目录（保存旋转后的图片）
dst_root = r"D:\事务夹\作业集\深度学习\final_project\dataset_process\Rotate_img"

# 旋转角度列表（顺时针方向）
rotate_list = [0, 90, 180, 270]

# 如果新目录已存在，先清空（防止重复）
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)
os.makedirs(dst_root)

# 遍历源目录的每一个子目录（0~9）
for class_name in os.listdir(src_root):
    src_class_path = os.path.join(src_root, class_name)

    if not os.path.isdir(src_class_path):
        continue

    # 新目录中创建相应的子目录
    dst_class_path = os.path.join(dst_root, class_name)
    os.makedirs(dst_class_path, exist_ok=True)

    print(f"Processing folder: {src_class_path}")

    # 遍历子目录中的所有 PNG 图片
    for img_name in os.listdir(src_class_path):
        if not img_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(src_class_path, img_name)
        img = Image.open(img_path)

        base_name = os.path.splitext(img_name)[0]

        # 为每个角度生成旋转图像
        for angle in rotate_list:
            rotated_img = img.rotate(-angle, expand=True)

            save_name = f"{base_name}_rot{angle}.png"
            save_path = os.path.join(dst_class_path, save_name)

            rotated_img.save(save_path)

        img.close()

print("All images saved to:", dst_root)
