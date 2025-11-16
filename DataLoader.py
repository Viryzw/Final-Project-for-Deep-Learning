import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class HandwritingDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.samples = []
        classes = sorted(os.listdir(root_dir))  # e.g. ['0','1','2',...]
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

        for cls in classes:
            cls_folder = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_folder):
                if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(cls_folder, fname), self.class_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("L")  # 单通道

        if self.transform:
            img = self.transform(img)

        return img, label
    

def get_loaders(root=r"data", batch_size=32):
    
    train_dir = f"{root}\\train"
    valid_dir = f"{root}\\valid"
    
    transform = transforms.Compose([
        transforms.ToTensor(),               # (1,128,128)
        transforms.Normalize((0.5,), (0.5,)) # [-1,1]
    ])

    train_dataset = HandwritingDataset(train_dir, transform=transform)
    valid_dataset = HandwritingDataset(train_dir, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
    
    return train_loader, valid_loader