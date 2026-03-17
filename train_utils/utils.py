import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import torch
import datetime

def plot_loss_acc(history, save_dir="results"):
    """
    绘制训练过程中的损失和准确率曲线，并将数据保存到文本文件
    
    参数:
    history: 包含训练历史的字典
    save_dir: 结果保存目录
    """
    # 确保目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 获取当前时间戳用于文件名
    #timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #txt_filename = os.path.join(save_dir, f"training_log_{timestamp}.txt")
    txt_filename = os.path.join(save_dir, f"training_log.txt")
    
    
    # 保存训练数据到文本文件
    with open(txt_filename, 'w') as f:
        # 写入标题行
        f.write("Epoch\tTrain Loss\tTrain Acc\tVal Loss\tVal Acc\tlr\n")
        
        # 确定epoch数量
        epochs = min(
            len(history.get('train_loss', [])),
            len(history.get('train_acc', [])),
            len(history.get('val_loss', [])),
            len(history.get('val_acc', [])),
            len(history.get('lr', []))
        )
        
        # 写入每个epoch的数据
        for epoch in range(epochs):
            train_loss = history['train_loss'][epoch] if epoch < len(history['train_loss']) else "N/A"
            train_acc = history['train_acc'][epoch] if epoch < len(history['train_acc']) else "N/A"
            val_loss = history['val_loss'][epoch] if epoch < len(history['val_loss']) else "N/A"
            val_acc = history['val_acc'][epoch] if epoch < len(history['val_acc']) else "N/A"
            lr = history['lr'][epoch] if epoch < len(history['lr']) else "N/A"
            
            f.write(f"{epoch+1}\t{train_loss:.4f}\t{train_acc:.4f}\t{val_loss:.4f}\t{val_acc:.4f}\t{lr:.6f}\n")
    
    print(f"Training logs saved to: {txt_filename}")
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 5))
    
    # 训练损失
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 训练准确率
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # 保存图像
    #plot_filename = os.path.join(save_dir, f"loss_acc_curves_{timestamp}.png")
    plot_filename = os.path.join(save_dir, f"loss_acc_curves.png")
    plt.savefig(plot_filename)
    plt.close()
    
    print(f"Loss and accuracy curves saved to: {plot_filename}")
    
    return txt_filename, plot_filename
    '''os.makedirs(save_dir, exist_ok=True)
    
    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Val Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(history["train_acc"], label="Train Acc")
    plt.plot(history["val_acc"], label="Val Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "loss_acc_curve.png"))
    plt.close()'''

def plot_confusion_matrix(true_labels, pred_labels, class_names, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(true_labels, pred_labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.savefig(os.path.join(save_dir, "confusion_matrix.png"))
    plt.close()

def save_classification_report(true_labels, pred_labels, class_names, save_dir="results"):
    os.makedirs(save_dir, exist_ok=True)
    report = classification_report(true_labels, pred_labels, target_names=class_names)
    
    with open(os.path.join(save_dir, "classification_report.txt"), "w") as f:
        f.write(report)
    
    print(report)

def save_model(model, optimizer, epoch, save_path):
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(state, save_path)
    print(f"Model saved to {save_path}")

def load_model(model, optimizer, load_path, device):
    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    print(f"Model loaded from {load_path}, epoch {epoch}")
    return model, optimizer, epoch

def verify_device_consistency(model, data_loader):
    """验证模型和数据是否在相同设备上"""
    # 获取一批数据
    sample_data, _ = next(iter(data_loader))
    sample_data = sample_data.to(model.device)
    
    print(f"Data device: {sample_data.device}")
    print(f"Model device: {model.device}")
    
    # 检查所有模型参数是否在正确设备上
    for name, param in model.named_parameters():
        if param.device != model.device:
            print(f"Warning: Parameter {name} is on {param.device}, not on {model.device}")
    
    # 检查所有缓冲区是否在正确设备上
    for name, buffer in model.named_buffers():
        if buffer.device != model.device:
            print(f"Warning: Buffer {name} is on {buffer.device}, not on {model.device}")
    
    # 尝试前向传播
    try:
        with torch.no_grad():
            output = model(sample_data)
        print("Device consistency test passed!")
        return True
    except Exception as e:
        print(f"Device consistency test failed: {e}")
        return False