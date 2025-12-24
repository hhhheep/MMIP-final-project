import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class MultiModalDetector(nn.Module):
    def __init__(self, num_classes=1):
        super(MultiModalDetector, self).__init__()
        
        # --- 1. RGB Extractor (Pre-trained ResNet-18) ---
        rgb_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 移除最後一層分類器
        self.E_rgb = nn.Sequential(*list(rgb_model.children())[:-1])
        
        # --- 2. rPPG Extractor (Modified ResNet-18 for 1 channel) ---
        rppg_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        # 修改第一層 Conv2D 以接受 1 個輸入通道
        # 這裡需要將 RGB 模型的 Conv1 權重平均後分配給 1 通道，或隨機初始化
        rppg_model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.E_rppg = nn.Sequential(*list(rppg_model.children())[:-1])
        
        # --- 3. 融合與分類層 ---
        # 兩個 ResNet-18 的特徵維度 (512) 拼接後為 1024
        self.fusion_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5), # 降低過擬合
            nn.Linear(512 * 2, num_classes) # NLP 分類器
        )

    def forward(self, I_rgb, I_rppg):
        # 提取特徵 (輸出 shape: [Batch_size, 512, 1, 1])
        F_rgb = self.E_rgb(I_rgb)
        F_rppg = self.E_rppg(I_rppg)
        
        # 融合
        F_fused = torch.cat((F_rgb, F_rppg), dim=1) # 沿特徵維度拼接
        
        # 分類 (輸出 shape: [Batch_size, num_classes])
        output = self.fusion_classifier(F_fused)
        return output