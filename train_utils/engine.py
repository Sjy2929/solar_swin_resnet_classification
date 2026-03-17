import torch
import time
from tqdm import tqdm

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(enumerate(data_loader), total=len(data_loader), desc=f"Epoch {epoch+1}")
    
    for step, (images, labels) in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # 反向传播和优化
        loss.backward()
        optimizer.step()
        
        # 统计指标
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * images.size(0)
        running_corrects += torch.sum(preds == labels.data)
        total_samples += images.size(0)
        
        # 更新进度条
        if step % print_freq == 0 or step == len(data_loader) - 1:
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            pbar.set_postfix(loss=f"{epoch_loss:.4f}", acc=f"{epoch_acc:.4f}")
    
    # 计算本轮平均损失和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * images.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += images.size(0)
    
    # 计算平均损失和准确率
    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()