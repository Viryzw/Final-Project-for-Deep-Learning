import torch
from utils import ExplainVisual, save_mask_gray
from model import FullModel
from dataloader import get_loaders
from PIL import Image
from torchvision import transforms

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FullModel(k=4).to(device)
    checkpoint = torch.load('checkpoints_final\\best_version_01.pth', map_location=device)
    # 提取模型真正的参数部分
    state_dict = checkpoint['state_dict']

    # 有些训练脚本会多带 "module." 前缀（DataParallel 训练）
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_key = k.replace('module.', '')  # 去掉多余前缀
        new_state_dict[new_key] = v

    # 加载到模型
    model.load_state_dict(new_state_dict, strict=False)
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # === 3. 读取一张图片 ===
    img_path = "1.png"   # ← 换成你自己的图片路径
    img = Image.open(img_path).convert("L")  # 单通道
    

    # === 4. transform 并送入模型 ===
    x = transform(img).unsqueeze(0).to(device)  # [1,1,128,128]
    
    with torch.no_grad():
        model.eval()
        z, z_hat, y_hat, masked_x, weights = model(x, status=False)
        
        save_mask_gray(masked_x)

if __name__ == '__main__':
    main()