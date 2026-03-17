import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from model.swin_resnet_model import SwinResNetHybrid
from train_utils.utils import load_model
from train_utils.data_loader import get_data_loaders
import argparse
import os

def predict_image(model, image_path, class_names, device, img_size=224):
    # 加载图像
    image = Image.open(image_path).convert("RGB")
    
    # 保存原始图像用于显示
    original_image = image.copy()
    
    # 预处理
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 预测
    model.eval()
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        _, pred = torch.max(output, 1)
        pred_class = class_names[pred.item()]
    
    # 获取top3预测
    top3_prob, top3_indices = torch.topk(probabilities, 3)
    top3_prob = top3_prob.squeeze().cpu().numpy()
    top3_classes = [class_names[i] for i in top3_indices.squeeze().cpu().numpy()]
    
    return pred_class, top3_classes, top3_prob, original_image

def visualize_prediction(image, pred_class, top3_classes, top3_prob, save_path=None):
    """
    可视化预测结果
    
    参数:
    image: PIL Image 对象
    pred_class: 预测的类别
    top3_classes: top3类别列表
    top3_prob: top3概率列表
    save_path: 保存路径 (可选)
    """
    # 创建图形
    plt.figure(figsize=(12, 6))
    
    # 显示原始图像
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Input Image')
    
    # 创建预测结果图
    plt.subplot(1, 2, 2)
    
    # 创建水平条形图显示top3预测
    y_pos = np.arange(len(top3_classes))
    plt.barh(y_pos, top3_prob, align='center', color='skyblue')
    plt.yticks(y_pos, top3_classes)
    plt.xlabel('Probability')
    plt.xlim(0, 1)
    
    # 高亮显示预测类别
    for i, cls in enumerate(top3_classes):
        if cls == pred_class:
            plt.text(top3_prob[i] + 0.02, i, f'{top3_prob[i]:.4f}', color='red', fontweight='bold')
        else:
            plt.text(top3_prob[i] + 0.02, i, f'{top3_prob[i]:.4f}', color='blue')
    
    plt.title(f'Prediction: {pred_class} (Prob: {max(top3_prob):.4f})')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存或显示
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Prediction visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Predict with Swin-ResNet model')
    parser.add_argument('--image_path', required=True, type=str, help='Path to input image')
    parser.add_argument('--model_path', default='swin_resnet_classification/saved_models/best_model.pth', type=str, help='Path to model weights')
    parser.add_argument('--data_dir', default='swin_resnet_classification/data/solar_dataset_split', type=str, help='Dataset directory')
    parser.add_argument('--output_dir', default='swin_resnet_classification/predictions', type=str, help='Directory to save prediction results')
    args = parser.parse_args()
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载数据集信息
    _, _, _, class_names = get_data_loaders(args.data_dir)
    
    # 初始化模型
    model = SwinResNetHybrid(num_classes=len(class_names), device=device)
    
    # 加载模型权重
    model, _, _ = load_model(model, None, args.model_path, device)
    
    # 预测单张图像
    pred_class, top3_classes, top3_prob, original_image = predict_image(
        model, args.image_path, class_names, device
    )
    
    # 打印结果
    print(f"Predicted class: {pred_class}")
    print("Top 3 predictions:")
    for i, (cls, prob) in enumerate(zip(top3_classes, top3_prob)):
        print(f"  {i+1}. {cls}: {prob:.4f}")
    
    # 生成保存路径
    image_name = os.path.basename(args.image_path)
    save_name = os.path.splitext(image_name)[0] + "_prediction.png"
    save_path = os.path.join(args.output_dir, save_name)
    
    # 可视化并保存预测结果
    visualize_prediction(
        original_image, 
        pred_class, 
        top3_classes, 
        top3_prob, 
        save_path=save_path
    )

if __name__ == "__main__":
    main()