import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

def process_image(image):
    """
    处理图像：转换调色板图像为RGB格式
    """
    if image.mode == 'P':
        # 转换为RGBA处理透明度
        image = image.convert('RGBA')
    
    if image.mode == 'RGBA':
        # 创建白色背景
        background = Image.new('RGB', image.size, (255, 255, 255))
        # 将RGBA图像合并到背景上
        background.paste(image, mask=image.split()[3])
        return background
    
    if image.mode != 'RGB':
        # 转换为RGB格式
        return image.convert('RGB')
    
    return image


class FlowerDataset(Dataset):
    def __init__(self, data_dir, transform=None, mode="train"):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # 加载类别信息
        self.classes = sorted(os.listdir(os.path.join(data_dir, mode)))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        # 收集图像路径和标签
        self.image_paths = []
        self.labels = []
        
        for cls_name in self.classes:
            cls_dir = os.path.join(data_dir, mode, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])
        
        print(f"Loaded {len(self.image_paths)} images for {mode} set, {len(self.classes)} classes")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        #image = Image.open(img_path).convert("RGB")
        image = Image.open(img_path)
        image = process_image(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_names(self):
        return self.classes

def get_data_loaders(data_dir, batch_size=32, img_size=224):
    # 数据增强
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = FlowerDataset(data_dir, transform=train_transform, mode="train")
    val_dataset = FlowerDataset(data_dir, transform=test_transform, mode="val")
    test_dataset = FlowerDataset(data_dir, transform=test_transform, mode="test")
    
    class_names = train_dataset.get_class_names()
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_loader, val_loader, test_loader, class_names