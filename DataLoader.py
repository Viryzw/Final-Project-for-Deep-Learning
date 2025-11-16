import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random

class ZeroShotDataset(Dataset):
    def __init__(self, root_dir, transform=None, pairs_per_class=10):
        self.root_dir = root_dir
        self.transform = transform
        
        # 获取所有类别和图片
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_images = {}
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        for cls in self.classes:
            cls_folder = os.path.join(root_dir, cls)
            images = [os.path.join(cls_folder, f) for f in os.listdir(cls_folder) 
                     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            self.class_to_images[cls] = images
        
        # 生成三元组 (anchor, positive, negative)
        self.triplets = self._generate_triplets(pairs_per_class)
        
    def _generate_triplets(self, pairs_per_class):
        triplets = []
        
        for cls in self.classes:
            images = self.class_to_images[cls]
            if len(images) < 2:
                continue
                
            # 生成正样本对
            for _ in range(pairs_per_class):
                anchor_img, positive_img = random.sample(images, 2)
                
                # 随机选择负样本类别
                negative_cls = random.choice([c for c in self.classes if c != cls])
                negative_img = random.choice(self.class_to_images[negative_cls])
                
                triplets.append((anchor_img, positive_img, negative_img))
                
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        anchor_path, positive_path, negative_path = self.triplets[idx]
        
        anchor = Image.open(anchor_path).convert("L")
        positive = Image.open(positive_path).convert("L")
        negative = Image.open(negative_path).convert("L")
        
        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)
            
        return anchor, positive, negative

def get_triplet_loader(root, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    dataset = ZeroShotDataset(root, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, len(dataset)