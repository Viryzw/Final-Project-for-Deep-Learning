import os
import shutil
import random

# 原目录：包含原图与翻转图
src_root = r"D:\事务夹\作业集\深度学习\final_project\dataset\test_1"

# 新目录：训练集 & 测试集
dst_root = r"D:\事务夹\作业集\深度学习\final_project\lip"

train_ratio = 0.5

# 如果新目录存在，清空重建
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)

train_root = os.path.join(dst_root, "train_1")
test_root  = os.path.join(dst_root, "test_1")

os.makedirs(train_root)
os.makedirs(test_root)

# 遍历类别文件夹（0~9）
for class_name in os.listdir(src_root):
    class_path = os.path.join(src_root, class_name)
    if not os.path.isdir(class_path):
        continue

    # 创建 train_1/test_1 对应的类别子目录
    train_class_path = os.path.join(train_root, class_name)
    test_class_path  = os.path.join(test_root, class_name)
    os.makedirs(train_class_path)
    os.makedirs(test_class_path)

    # 获取此类别下所有图片
    images = [img for img in os.listdir(class_path)
              if img.lower().endswith(".png")]

    # 随机打乱
    random.shuffle(images)

    # 按 8:2 拆分
    split_idx = int(len(images) * train_ratio)
    train_imgs = images[:split_idx]
    test_imgs  = images[split_idx:]

    print(f"Class {class_name}: {len(train_imgs)} train_1, {len(test_imgs)} test_1")

    # 复制训练集图像
    for img in train_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(train_class_path, img)
        )

    # 复制测试集图像
    for img in test_imgs:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(test_class_path, img)
        )

print("Dataset split completed! Saved to:", dst_root)
