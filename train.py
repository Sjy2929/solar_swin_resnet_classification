import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.swin_resnet_model import SwinResNetHybrid
from train_utils.data_loader import get_data_loaders
from train_utils.engine import train_one_epoch, evaluate
from train_utils.utils import save_model, verify_device_consistency, plot_loss_acc
import time
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6, warmup_lr=1e-4):
        """
        学习率预热与余弦退火调度器
        
        参数:
        optimizer: 优化器
        warmup_epochs: 预热epoch数
        total_epochs: 总epoch数
        base_lr: 基础学习率
        min_lr: 最小学习率
        warmup_lr: 预热起始学习率
        """
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.current_epoch = 0
        
        # 余弦退火调度器
        self.cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs - warmup_epochs,
            eta_min=min_lr
        )
        
    def step(self):
        """更新学习率"""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # 预热阶段：线性增加学习率
            lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (self.current_epoch / self.warmup_epochs)
            self._set_lr(lr)
        else:
            # 余弦退火阶段
            self.cosine_scheduler.step()
    
    def _set_lr(self, lr):
        """设置所有参数组的学习率"""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_lr(self):
        """获取当前学习率"""
        return self.optimizer.param_groups[0]['lr']
    
    def plot_schedule(self, save_path=None):
        """绘制学习率调度曲线"""
        lrs = []
        epochs = list(range(1, self.total_epochs + 1))
        
        for epoch in epochs:
            if epoch <= self.warmup_epochs:
                lr = self.warmup_lr + (self.base_lr - self.warmup_lr) * (epoch / self.warmup_epochs)
            else:
                # 余弦退火公式
                progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            lrs.append(lr)
        
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, lrs)
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.title('Warmup Cosine Learning Rate Schedule')
        plt.grid(True)
        
        # 标记预热结束点
        plt.axvline(x=self.warmup_epochs, color='r', linestyle='--', alpha=0.7)
        plt.text(self.warmup_epochs + 0.5, self.base_lr * 0.8, 'Warmup End', color='r')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
        
        return lrs

def parse_args():
    parser = argparse.ArgumentParser(description='Train Swin-ResNet hybrid model')
    parser.add_argument('--data_dir', default='swin_resnet_classification/data/solar_dataset_split', type=str, help='Dataset directory')
    parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    parser.add_argument('--num_epochs', default=260, type=int, help='Number of training epochs')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone weights')
    parser.add_argument('--swin_type', default='tiny', choices=['tiny', 'small'], help='Swin Transformer type')
    parser.add_argument('--save_dir', default='swin_resnet_classification/saved_models', type=str, help='Directory to save models')
    # 学习率调度参数
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Minimum learning rate')
    parser.add_argument('--warmup_lr', default=1e-4, type=float, help='Warmup starting learning rate')
    parser.add_argument('--results_dir', default='swin_resnet_classification/results', type=str, help='Directory to save results')

    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Training arguments: {args}")
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 获取数据加载器
    train_loader, val_loader, test_loader, class_names = get_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )
    num_classes = len(class_names)
    
    # 初始化模型
    model = SwinResNetHybrid(
        num_classes=num_classes,
        swin_model_type=args.swin_type,
        use_pretrained=True,
        freeze_backbone=args.freeze_backbone,
        device=device
    )
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # 打印可训练参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}, Trainable parameters: {trainable_params:,}")
    
    # 验证设备一致性
    device_ok = verify_device_consistency(model, train_loader)
    if not device_ok:
        raise RuntimeError("Device inconsistency detected. Please fix before proceeding.")
    
    # 损失函数
    criterion = nn.CrossEntropyLoss()
    
    # 优化器
    optimizer = AdamW([
        {"params": model.resnet.parameters(), "lr": args.lr/10},
        {"params": model.swin.parameters(), "lr": args.lr/10},
        {"params": model.fusion.parameters(), "lr": args.lr},
        {"params": model.classifier.parameters(), "lr": args.lr}
    ], weight_decay=0.01)
    
    # 学习率调度器
    #scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, verbose=True, min_lr=1e-6)

    # 学习率调度器
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        total_epochs=args.num_epochs,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr
    )

    # 绘制学习率调度曲线
    schedule_plot_path = os.path.join(args.results_dir, "lr_schedule.png")
    scheduler.plot_schedule(save_path=schedule_plot_path)
    print(f"Learning rate schedule saved to {schedule_plot_path}")

    # 训练记录
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # 训练循环
    for epoch in range(args.num_epochs):
        start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_one_epoch(
            model, criterion, optimizer, train_loader, device, epoch
        )
        
        # 验证
        val_loss, val_acc = evaluate(model, criterion, val_loader, device)
        
        # 更新学习率
        scheduler.step()
        # 记录当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # 记录历史
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        # 打印信息
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.num_epochs} - {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"lr: {current_lr:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            save_model(model, optimizer, epoch, os.path.join(args.save_dir, "best_model.pth"))
            print("Saved new best model")
        
        # 保存当前epoch的模型
        save_model(model, optimizer, epoch, os.path.join(args.save_dir, f"epoch_{epoch+1}.pth"))
        print(f"Saved model for epoch {epoch+1}")

        log_file, plot_file = plot_loss_acc(history, save_dir="swin_resnet_classification/results")
    
    # 保存最终模型
    save_model(model, optimizer, args.num_epochs, os.path.join(args.save_dir, "final_model.pth"))
    
    # 绘制训练曲线
    log_file, plot_file = plot_loss_acc(history, save_dir="swin_resnet_classification/results")
    #plot_loss_acc(history, save_dir="results")
    
    print(f"Training complete. Best val Acc: {best_val_acc:.4f} at epoch {best_epoch+1}")

if __name__ == "__main__":
    main()