import os
import random
from PIL import Image


def random_crops(
        img_path,
        output_dir,
        crop_size=128,
        num_crops=30,
        margin=50
):
    """
    对单张图像进行随机裁剪
    img_path: 输入图片路径
    output_dir: 输出目录
    crop_size: 裁剪尺寸（128）
    num_crops: 每张图要裁剪的数量
    margin: 避边界距离（避免拍照背景）
    """
    img = Image.open(img_path)
    w, h = img.size

    # 可裁剪区域范围
    xmin = margin
    ymin = margin
    xmax = w - crop_size - margin
    ymax = h - crop_size - margin

    if xmax <= xmin or ymax <= ymin:
        print(f"[WARNING] Image too small to crop safely: {img_path}")
        return

    # 创建输出文件夹
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(img_path))[0]

    for i in range(num_crops):
        # 随机选取左上角
        x = random.randint(xmin, xmax)
        y = random.randint(ymin, ymax)

        crop = img.crop((x, y, x + crop_size, y + crop_size))

        out_path = os.path.join(output_dir, f"{base_name}_crop_{i + 1}.png")
        crop.save(out_path)


def process_folder(input_folder, output_folder, crop_size=128, num_crops=30, margin=50):
    for root, dirs, files in os.walk(input_folder):
        # 得到对应的输出目录结构
        relative = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, relative)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_path = os.path.join(root, file)

                print(f"Processing: {input_path}")
                random_crops(
                    img_path=input_path,
                    output_dir=out_dir,
                    crop_size=crop_size,
                    num_crops=num_crops,
                    margin=margin
                )


if __name__ == "__main__":
    input_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\out"  # ← 你的二值化图片文件夹
    output_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\Splited_img"  # ← 裁剪 patch 输出文件夹

    process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        crop_size=128,
        num_crops=30,
        margin=120  # 避开边缘
    )

    print("=== Done! Random patches extracted. ===")
