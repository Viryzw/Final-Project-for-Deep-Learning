import torch
import torch.nn.functional as F 
import os
from PIL import Image
import torchvision.transforms as transforms
from FeatureExtractor import SiameseNetwork
import matplotlib.pyplot as plt

class ZeroShotTester:
    def __init__(self, model_path, embedding_dim=128):
        self.model = SiameseNetwork(embedding_dim=embedding_dim)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        # 注册库：存储已知作者的特征
        self.gallery_features = {}
        self.gallery_images = {}
        
    def register_author(self, author_id, image_paths):
        """注册一个作者的多张图片到库中"""
        features = []
        for img_path in image_paths:
            feature = self.extract_feature(img_path)
            features.append(feature)
        
        # 取平均特征作为该作者的代表特征
        avg_feature = torch.stack(features).mean(dim=0)
        self.gallery_features[author_id] = avg_feature
        
        # 保存一张示例图片用于显示
        self.gallery_images[author_id] = image_paths[0]
        
    def extract_feature(self, image_path):
        """提取单张图片的特征"""
        img = Image.open(image_path).convert("L")
        img_tensor = self.transform(img).unsqueeze(0)
        
        with torch.no_grad():
            feature = self.model.feature_extractor(img_tensor)
        return feature.squeeze()
    
    def verify(self, query_image_path, author_id, threshold=0.7):
        """验证查询图片是否属于指定作者"""
        query_feature = self.extract_feature(query_image_path)
        gallery_feature = self.gallery_features[author_id]
        
        similarity = F.cosine_similarity(query_feature.unsqueeze(0), 
                                       gallery_feature.unsqueeze(0))
        similarity = similarity.item()
        
        is_same = similarity > threshold
        return is_same, similarity
    
    def identify(self, query_image_path, threshold=0.7):
        """识别查询图片属于哪个作者"""
        query_feature = self.extract_feature(query_image_path)
        
        best_similarity = -1
        best_author = None
        
        for author_id, gallery_feature in self.gallery_features.items():
            similarity = F.cosine_similarity(query_feature.unsqueeze(0), 
                                           gallery_feature.unsqueeze(0)).item()
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_author = author_id
        
        if best_similarity > threshold:
            return best_author, best_similarity
        else:
            return "Unknown", best_similarity
    
    def visualize_comparison(self, query_image_path, author_id=None):
        """可视化比较结果"""
        if author_id is None:
            identified_author, similarity = self.identify(query_image_path)
        else:
            is_same, similarity = self.verify(query_image_path, author_id)
            identified_author = author_id if is_same else "Different"
        
        # 显示图片
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # 查询图片
        query_img = Image.open(query_image_path).convert("L")
        axes[0].imshow(query_img, cmap="gray")
        axes[0].set_title(f"Query Image\nSimilarity: {similarity:.3f}")
        axes[0].axis("off")
        
        # 库中图片
        if identified_author != "Unknown" and identified_author != "Different":
            gallery_img = Image.open(self.gallery_images[identified_author]).convert("L")
            axes[1].imshow(gallery_img, cmap="gray")
            axes[1].set_title(f"Gallery: {identified_author}")
        else:
            axes[1].text(0.5, 0.5, identified_author, 
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=20)
            axes[1].set_title("Result")
        
        axes[1].axis("off")
        plt.tight_layout()
        plt.show()
        
        return identified_author, similarity



# 批量验证并重命名图片
def batch_identify_and_rename(tester, test_folder, output_folder, threshold=0.7):
    """批量识别图片作者并重命名保存到另一个文件夹"""
    import shutil
    import os
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取测试文件夹中的所有图片
    test_images = [f for f in os.listdir(test_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    results = []
    
    for img_name in test_images:
        img_path = os.path.join(test_folder, img_name)
        
        # 识别作者
        author, similarity = tester.identify(img_path, threshold)
        
        # 创建新文件名：作者_相似度_原文件名
        name, ext = os.path.splitext(img_name)
        new_name = f"{author}_{similarity:.3f}{ext}"
        new_path = os.path.join(output_folder, new_name)
        
        # 复制图片到新文件夹
        shutil.copy2(img_path, new_path)
        
        results.append({
            'original_name': img_name,
            'author': author,
            'similarity': similarity,
            'new_name': new_name
        })
        
        print(f"Processed: {img_name} -> {author} (similarity: {similarity:.3f})")
    
    return results

# 使用示例
if __name__ == "__main__":
    # 初始化测试器
    tester = ZeroShotTester("zero_shot_handwriting.pth")
    
    # 注册已知作者
    tester.register_author("author1", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\0\8c41672619938b83305080f7f31aa90e_crop_25.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\0\d629cfccd6766119de26b11cbe312dac_crop_17.png"])
    tester.register_author("author2", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\1\eeff899f7aeb9b494e16c9635f5ff743_crop_16.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\1\eeff899f7aeb9b494e16c9635f5ff743_crop_28.png"])
    tester.register_author("author3", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\2\55fabd3b4c696750457f8104ba0636cd_crop_6.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\2\f4424d9be5ccf4c88995d9554797cf27_crop_12.png"])                    
    
    # 批量处理
    test_folder = r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_imgs"
    output_folder = r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_results"
    
    results = batch_identify_and_rename(tester, test_folder, output_folder, threshold=0.3)
    
    # 打印汇总结果
    print(f"\n批量处理完成！共处理 {len(results)} 张图片")
    print(f"结果保存在: {output_folder}")
    
    # 统计各作者的图片数量
    author_count = {}
    for result in results:
        author = result['author']
        author_count[author] = author_count.get(author, 0) + 1
    
    print("\n作者分布:")
    for author, count in author_count.items():
        print(f"  {author}: {count} 张")




# # 使用示例
# if __name__ == "__main__":
#     # 初始化测试器
#     tester = ZeroShotTester("zero_shot_handwriting.pth")
    
#     # 注册已知作者（在实际使用中，你需要提供这些图片路径）
#     # 示例：tester.register_author("author1", ["path/to/author1_img1.jpg", "path/to/author1_img2.jpg"])
#     tester.register_author("author1", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\0\8c41672619938b83305080f7f31aa90e_crop_25.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\0\d629cfccd6766119de26b11cbe312dac_crop_17.png"])
#     tester.register_author("author2", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\1\eeff899f7aeb9b494e16c9635f5ff743_crop_16.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\1\eeff899f7aeb9b494e16c9635f5ff743_crop_28.png"])
#     tester.register_author("author3", [r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\2\55fabd3b4c696750457f8104ba0636cd_crop_6.png", r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\dataset\Filtered_img\2\f4424d9be5ccf4c88995d9554797cf27_crop_12.png"])                    
    
#     # 验证单个图片
#     # result, similarity = tester.verify("query.jpg", "author1")
    
#     # 识别图片作者
#     # author, similarity = tester.identify("unknown.jpg")
#     author, similarity = tester.identify(r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_imgs\7a2866b9f6d7ad017986a87324641e9b_crop_1.png")
#     print(f"Identified author: {author} with similarity: {similarity:.3f}")

#     # 可视化比较
#     tester.visualize_comparison(r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_imgs\7a2866b9f6d7ad017986a87324641e9b_crop_1.png", "author1")
#     tester.visualize_comparison(r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_imgs\7a2866b9f6d7ad017986a87324641e9b_crop_1.png", "author2")
#     tester.visualize_comparison(r"D:\桌面文件\总\课程\大三上\深度学习\final_project\final_project\test_imgs\7a2866b9f6d7ad017986a87324641e9b_crop_1.png", "author3")
    
#     print("Zero-shot tester initialized. Please register authors and test images.")