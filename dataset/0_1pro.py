import os
from PIL import Image

def binarize_image(input_path, output_path, threshold=180):
    """
    将图像二值化：字迹保持为黑，背景为白
    threshold：阈值，可根据照片深浅调节（0~255）
    """
    img = Image.open(input_path).convert("L")    # 转灰度
    binary = img.point(lambda x: 0 if x < threshold else 255, '1')  # 二值化
    binary.save(output_path)


def process_folder(input_folder, output_folder, threshold=180):
    # 遍历分类文件夹
    for root, dirs, files in os.walk(input_folder):
        # 计算对应的输出目录路径
        relative_path = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, relative_path)

        # 创建输出目录
        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                input_path = os.path.join(root, file)
                output_path = os.path.join(out_dir, file)

                try:
                    binarize_image(input_path, output_path, threshold)
                    print(f"Processed: {output_path}")
                except Exception as e:
                    print(f"Error processing {input_path}: {e}")


if __name__ == "__main__":
    input_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\origin"   # TODO: 改成你的路径
    output_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\out" # TODO: 改成你的路径
    threshold = 140                 # 可以试 150、180、200 测效果

    process_folder(input_folder, output_folder, threshold)

    print("=== Done! All images processed and saved. ===")
