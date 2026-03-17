import torch
import torch.nn as nn
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights
from transformers import SwinModel, SwinConfig
import os

class SwinResNetHybrid(nn.Module):
    def __init__(self, num_classes, swin_model_type="tiny", use_pretrained=True, freeze_backbone=False, device=None):
        super(SwinResNetHybrid, self).__init__()
        
        # 设置设备
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # ResNet50分支 (移除最后两层)
        if use_pretrained:
            # 使用新的weights参数
            self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.resnet = resnet50(weights=None)
            
        # 转换为Sequential并移动到设备
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2]).to(self.device)
        
        # Swin Transformer分支
        if swin_model_type == "tiny":
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 6, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7
            )
        elif swin_model_type == "small":
            config = SwinConfig(
                image_size=224,
                patch_size=4,
                num_channels=3,
                embed_dim=96,
                depths=[2, 2, 18, 2],
                num_heads=[3, 6, 12, 24],
                window_size=7
            )
        else:
            raise ValueError("Unsupported Swin model type")
        
        self.swin = SwinModel(config).to(self.device)
        
        # 加载预训练权重
        if use_pretrained:
            try:
                # 定义本地权重文件路径
                local_weight_path = f"swin_resnet_classification/model/swin_{swin_model_type}_patch4_window7_224.pth"
                
                # 检查文件是否存在
                if not os.path.exists(local_weight_path):
                    raise FileNotFoundError(f"Pretrained weights file not found at {local_weight_path}")
                
                # 根据设备选择加载方式
                state_dict = torch.load(local_weight_path, map_location=self.device)
                
                # 加载到Swin模型
                self.swin.load_state_dict(state_dict["model"], strict=False)
                print(f"Successfully loaded pretrained weights from local path: {local_weight_path}")
            except Exception as e:
                print(f"Could not load pretrained weights: {e}. Using random initialization.")
        
        # 冻结骨干网络权重
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
            for param in self.swin.parameters():
                param.requires_grad = False
            print("Backbone weights frozen")
        
        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Conv2d(2048 + 768, 1024, kernel_size=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        ).to(self.device)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        ).to(self.device)
    
    def forward(self, x):
        # 确保输入在正确设备上
        if x.device != self.device:
            x = x.to(self.device)
        
        # ResNet特征提取
        resnet_feat = self.resnet(x)  # [B, 2048, H/32, W/32]
        
        # Swin Transformer特征提取
        swin_outputs = self.swin(x)
        swin_feat = swin_outputs.last_hidden_state  # [B, Seq_len, 768]
        
        # 调整Swin特征形状
        B, Seq_len, C = swin_feat.shape
        H = W = int(Seq_len**0.5)
        swin_feat = swin_feat.permute(0, 2, 1).view(B, C, H, W)  # [B, 768, H, W]
        
        # 调整Swin特征尺寸以匹配ResNet特征
        swin_feat = nn.functional.interpolate(swin_feat, size=resnet_feat.shape[2:], 
                                            mode="bilinear", align_corners=False)
        
        # 特征融合
        fused_feat = torch.cat([resnet_feat, swin_feat], dim=1)  # [B, 2048+768, H, W]
        fused_feat = self.fusion(fused_feat)  # [B, 1024, 1, 1]
        fused_feat = torch.flatten(fused_feat, 1)  # [B, 1024]
        
        # 分类
        output = self.classifier(fused_feat)
        return output