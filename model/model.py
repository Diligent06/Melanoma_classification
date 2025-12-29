import os
from os.path import join
import sys
sys.path.append(os.getcwd())


from MobileUNETR.architectures.mobileunetr import build_mobileunetr_s, build_mobileunetr_xs, build_mobileunetr_xxs
import torchvision
from torchvision import models
import torch
import torch.nn as nn
import torchvision.transforms as T


class GatedFusion(nn.Module):
    def __init__(self, dim=1920):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.Sigmoid()
        )

    def forward(self, f_rgb, f_mask):
        x = torch.cat([f_rgb, f_mask], dim=1)
        alpha = self.fc(x)
        fused = alpha * f_mask + (1 - alpha) * f_rgb
        return fused

class ExpertHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
            nn.Softmax(dim=1)  # ⭐ 加这里
        )

    def forward(self, x):
        return self.net(x)
    
class MoEGate(nn.Module):
    def __init__(self, in_dim, num_experts):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(in_dim, 16),
            nn.ReLU(),
            nn.Linear(16, num_experts)
        )

    def forward(self, age):
        """
        age: (B,1), 已归一化
        """
        logits = self.gate(age)
        weights = torch.softmax(logits, dim=1)
        return weights



class Backbone_MoE:
    def __init__(self, expert_num=4):
        self.seg_model = build_mobileunetr_xxs(num_classes=1, image_size=512)
        self.desnet_model = models.densenet201(
            weights=models.DenseNet201_Weights.IMAGENET1K_V1
        )
        self.gate_fuse = GatedFusion(dim=1920)
        self.sex_emb = nn.Embedding(num_embeddings=3, embedding_dim=4)  
        self.site_emb = nn.Embedding(num_embeddings=9, embedding_dim=11)
        
        self.expert_head = []
        for i in range(expert_num):
            self.expert_head.append(ExpertHead(in_dim=1920, num_classes=8))
        
        self.moe_gate = MoEGate(in_dim=16, num_experts=expert_num)
        
        self.training = True
        self.fuse_type = 'gate' # 'gate' or 'concat' or 'diff'
        
        self.transform = T.Compose([T.Resize((224, 224))])


    def forward(self, x, context_features):
        seg_mask = self.seg_model.forward(x)
        seg_mask = seg_mask[0, 0]
        binary = (seg_mask > 0).float()
        binary = binary.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
        masked_x = x * binary  # Apply segmentation mask
        
        
        if not self.training:
            with torch.no_grad():
                masked_x_features = self.desnet_model.features(self.transform(masked_x))
                masked_x_features = torch.nn.functional.relu(masked_x_features)
                masked_pooled = torch.nn.functional.adaptive_avg_pool2d(masked_x_features, (1, 1))
                masked_pooled = masked_pooled.view(masked_pooled.size(0), -1)  # [batch_size, 1920]
                
                x_feature = self.desnet_model.features(self.transform(x))
                x_feature = torch.nn.functional.relu(x_feature)
                x_pooled = torch.nn.functional.adaptive_avg_pool2d(x_feature, (1, 1))
                x_pooled = x_pooled.view(x_pooled.size(0), -1)  # [batch_size, 1920]
                
        else:
            masked_x_features = self.desnet_model.features(self.transform(masked_x))
            masked_x_features = torch.nn.functional.relu(masked_x_features)
            masked_pooled = torch.nn.functional.adaptive_avg_pool2d(masked_x_features, (1, 1))
            masked_pooled = masked_pooled.view(masked_pooled.size(0), -1)  # [batch_size, 1920]

            x_feature = self.desnet_model.features(self.transform(x))
            x_feature = torch.nn.functional.relu(x_feature)
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x_feature, (1, 1))
            x_pooled = x_pooled.view(x_pooled.size(0), -1)  # [batch_size, 1920]
        
        fused_x = self.gate_fuse(x_pooled, masked_pooled)
        
        age_list, sex_list, anatom_site_general_list = [], [], []
        for context_feature in context_features:
            age = torch.tensor(context_feature['age']).unsqueeze(0).unsqueeze(0)
            sex = torch.tensor(context_feature['sex'], dtype=torch.long).unsqueeze(0)
            anatom_site_general = torch.tensor(context_feature['anatom_site_general'], dtype=torch.long).unsqueeze(0)
            age_list.append(age)
            sex_list.append(sex)
            anatom_site_general_list.append(anatom_site_general)
            
        age = torch.cat(age_list, dim=0)
        sex = torch.cat(sex_list, dim=0)
        anatom_site_general = torch.cat(anatom_site_general_list, dim=0)
        
        sex_feat = self.sex_emb(sex)
        anatom_site_general_feat = self.site_emb(anatom_site_general)
        final_feat = torch.cat([age, sex_feat, anatom_site_general_feat], dim=1) 
        
        gate_w = self.moe_gate(final_feat)  # (B, num_experts)
        
        expert_outs = []
        for expert in self.expert_head:
            expert_outs.append(expert(fused_x))
        
        expert_outs = torch.stack(expert_outs, dim=1)  # (B, num_experts, num_classes)
        out = (expert_outs * gate_w.unsqueeze(-1)).sum(dim=1)  # (B, num_classes)
        
        return out, gate_w
        
        
        
        
        