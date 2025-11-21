import os
from PIL import Image
import shutil
import random

# 源目录
src_root = r"D:\事务夹\作业集\深度学习\final_project\dataset\train_1"

# 新目录（保存原图 + 错切图）
dst_root = r"D:\事务夹\作业集\深度学习\final_project\dataset\train"

# 如果已存在，清空（按需注释）
if os.path.exists(dst_root):
    shutil.rmtree(dst_root)
os.makedirs(dst_root, exist_ok=True)

# 切成 4 × 4，共 16 格
GRID = 4


def split_into_grid(img, n=4):
    """将图片切成 n×n 个小格子，返回列表"""
    w, h = img.size
    sub_w = w // n
    sub_h = h // n

    tiles = []
    for i in range(n):
        for j in range(n):
            box = (j * sub_w, i * sub_h, (j+1) * sub_w, (i+1) * sub_h)
            tile = img.crop(box)
            tiles.append(tile)

    return tiles


def combine_from_grid(tiles, img_size, n=4):
    """将 n×n 的格子按 tiles 顺序拼回图片"""
    w, h = img_size
    sub_w = w // n
    sub_h = h // n

    new_img = Image.new("RGB", (w, h))
    idx = 0
    for i in range(n):
        for j in range(n):
            new_img.paste(tiles[idx], (j * sub_w, i * sub_h))
            idx += 1
    return new_img


# 遍历目录结构
for class_name in os.listdir(src_root):
    src_class_path = os.path.join(src_root, class_name)
    if not os.path.isdir(src_class_path):
        continue

    dst_class_path = os.path.join(dst_root, class_name)
    os.makedirs(dst_class_path, exist_ok=True)

    print(f"Processing folder: {class_name}")

    for img_name in os.listdir(src_class_path):
        if not img_name.lower().endswith(".png"):
            continue

        img_path = os.path.join(src_class_path, img_name)

        try:
            img = Image.open(img_path).convert("RGB")
        except:
            print(f"  [Error] Cannot open {img_path}")
            continue

        base_name, ext = os.path.splitext(img_name)

        # -------------------------
        # 1. 保存原图
        # -------------------------
        original_save_path = os.path.join(dst_class_path, img_name)
        img.save(original_save_path)

        # -------------------------
        # 2. 切分成 16 格
        # -------------------------
        tiles = split_into_grid(img, GRID)

        # -------------------------
        # 3. 进行“错切”——随机打乱 16 格
        # -------------------------
        shuffled_tiles = tiles[:]
        random.shuffle(shuffled_tiles)

        # -------------------------
        # 4. 拼回成一张新图
        # -------------------------
        shuffled_img = combine_from_grid(shuffled_tiles, img.size, GRID)

        # 保存错切后的图
        shuffle_save_name = f"{base_name}_shuffle16{ext}"
        shuffle_save_path = os.path.join(dst_class_path, shuffle_save_name)
        shuffled_img.save(shuffle_save_path)

        img.close()

print("All images are processed and saved to:", dst_root)
