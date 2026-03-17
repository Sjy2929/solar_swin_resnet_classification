import torch
import torch.nn as nn
import numpy as np
from model.swin_resnet_model import SwinResNetHybrid
from train_utils.data_loader import get_data_loaders
from train_utils.utils import plot_confusion_matrix, save_classification_report, load_model
from train_utils.engine import evaluate
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score, roc_curve, auc, average_precision_score
from sklearn.preprocessing import label_binarize
from itertools import cycle

def calculate_additional_metrics(true_labels, pred_labels, pred_probs, class_names, save_dir):
    """
    计算额外的评估指标并保存结果
    
    参数:
    true_labels: 真实标签列表
    pred_labels: 预测标签列表
    pred_probs: 预测概率矩阵 (n_samples, n_classes)
    class_names: 类别名称列表
    save_dir: 结果保存目录
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Cohen's Kappa 系数
    kappa = cohen_kappa_score(true_labels, pred_labels)
    print(f"Cohen's Kappa: {kappa:.4f}")
    
    # 2. ROC 曲线和 AUC
    # 二值化标签
    n_classes = len(class_names)
    y_true_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # 计算每个类别的ROC曲线
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], pred_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 计算宏平均ROC曲线 (使用 numpy.interp 替代 scipy.interp)
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
    
    # 初始化平均TPR
    mean_tpr = np.zeros_like(all_fpr)
    
    # 对每个类别进行插值
    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
    
    mean_tpr /= n_classes
    
    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    
    # 绘制ROC曲线
    plt.figure(figsize=(10, 8))
    plt.plot(fpr["macro"], tpr["macro"],
             label=f'Macro-average ROC curve (AUC = {roc_auc["macro"]:.2f})',
             color='navy', linestyle=':', linewidth=4)
    
    # 绘制每个类别的ROC曲线
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red','blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'ROC curve of class {class_names[i]} (AUC = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, "roc_curve.png"))
    plt.close()
    
    # 3. mAP (Mean Average Precision)
    # 计算每个类别的AP
    ap = dict()
    for i in range(n_classes):
        ap[i] = average_precision_score(y_true_bin[:, i], pred_probs[:, i])
    
    # 计算宏平均mAP
    mAP = np.mean([ap[i] for i in range(n_classes)])
    print(f"mAP (Mean Average Precision): {mAP:.4f}")
    
    # 保存所有指标到文件
    metrics_report = f"""
    =============================
    Additional Evaluation Metrics
    =============================
    
    Cohen's Kappa: {kappa:.4f}
    Macro-Average AUC: {roc_auc['macro']:.4f}
    mAP (Mean Average Precision): {mAP:.4f}
    
    Per-class AUC:
    """
    for i, name in enumerate(class_names):
        metrics_report += f"  {name}: {roc_auc[i]:.4f}\n"
    
    metrics_report += "\nPer-class Average Precision:\n"
    for i, name in enumerate(class_names):
        metrics_report += f"  {name}: {ap[i]:.4f}\n"
    
    with open(os.path.join(save_dir, "additional_metrics.txt"), "w") as f:
        f.write(metrics_report)
    
    print(metrics_report)
    
    return {
        "kappa": kappa,
        "macro_auc": roc_auc["macro"],
        "mAP": mAP,
        "per_class_auc": roc_auc,
        "per_class_ap": ap
    }

def main():
    args = parse_args()
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 获取数据加载器
    _, _, test_loader, class_names = get_data_loaders(args.data_dir, batch_size=args.batch_size)
    
    # 初始化模型
    model = SwinResNetHybrid(num_classes=len(class_names), device=device)
    
    # 加载模型权重
    model, _, _ = load_model(model, None, args.model_path, device)
    
    # 评估模型
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, criterion, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
    
    # 获取预测结果和概率
    all_preds = []
    all_labels = []
    all_probs = []
    
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
    
    # 保存评估结果
    os.makedirs(args.save_dir, exist_ok=True)
    save_classification_report(all_labels, all_preds, class_names, save_dir=args.save_dir)
    plot_confusion_matrix(all_labels, all_preds, class_names, save_dir=args.save_dir)
    
    # 计算额外指标
    additional_metrics = calculate_additional_metrics(
        all_labels, 
        all_preds, 
        np.array(all_probs), 
        class_names, 
        args.save_dir
    )
    
    # 保存完整评估报告
    full_report = f"""
    ===================
    Complete Evaluation
    ===================
    
    Test Loss: {test_loss:.4f}
    Test Accuracy: {test_acc:.4f}
    Cohen's Kappa: {additional_metrics['kappa']:.4f}
    Macro-Average AUC: {additional_metrics['macro_auc']:.4f}
    mAP (Mean Average Precision): {additional_metrics['mAP']:.4f}
    """
    
    with open(os.path.join(args.save_dir, "full_evaluation_report.txt"), "w") as f:
        f.write(full_report)

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate Swin-ResNet model')
    parser.add_argument('--data_dir', default='swin_resnet_classification/data/solar_dataset_split', type=str, help='Dataset directory')
    parser.add_argument('--model_path', default='swin_resnet_classification/saved_models/best_model.pth', type=str, help='Path to model weights')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for evaluation')
    parser.add_argument('--save_dir', default='swin_resnet_classification/results', type=str, help='Directory to save results')
    return parser.parse_args()

if __name__ == "__main__":
    main()