import os
import shutil
from PIL import Image
import numpy as np

def get_black_ratio(img_path):
    """
    返回 (黑色面积占比, 是否成功)
    """
    try:
        img = Image.open(img_path).convert("L")
        arr = np.array(img)

        # 黑色像素（留一定范围提高鲁棒性，0~30 都算黑）
        black_pixels = np.sum(arr < 30)
        total_pixels = arr.size

        ratio = black_pixels / total_pixels
        return ratio, True
    except:
        return 0.0, False


def process_folder(input_folder, output_folder, min_ratio=0.02, max_ratio=0.4):
    """
    min_ratio: 黑色占比过低 → 删除无字 patch
    max_ratio: 黑色占比过高 → 删除边缘/大黑块
    """
    for root, dirs, files in os.walk(input_folder):
        relative = os.path.relpath(root, input_folder)
        out_dir = os.path.join(output_folder, relative)

        os.makedirs(out_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                input_path = os.path.join(root, file)

                ratio, ok = get_black_ratio(input_path)
                if not ok:
                    print(f"ERROR reading: {input_path}")
                    continue

                if ratio < min_ratio:
                    print(f"DROP (too little ink {ratio:.4f}): {input_path}")
                    continue

                if ratio > max_ratio:
                    print(f"DROP (too much ink {ratio:.4f}): {input_path}")
                    continue

                # 保留
                shutil.copy(input_path, out_dir)
                print(f"KEEP ({ratio:.4f}): {input_path}")


if __name__ == "__main__":
    input_crop_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\Splited_img"         # 你的随机裁剪结果文件夹
    output_filter_folder = r"D:\事务夹\作业集\深度学习\final_project\dataset\Filtered_img"    # 保存过滤后的结果

    process_folder(
        input_folder=input_crop_folder,
        output_folder=output_filter_folder,
        min_ratio=0.02,   # <2% 黑：无字
        max_ratio=0.40    # >40% 黑：靠边、大黑块
    )

    print("=== Done! Dual filtering finished. ===")
