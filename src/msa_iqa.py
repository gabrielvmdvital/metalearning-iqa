import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    """
    Módulo de Atenção Espacial Multicamadas (Multilayer Spatial Attention): 
    Calcula a importância de cada região da imagem (patch) para a avaliação da qualidade.
    """
    def __init__(self, embed_dim):
        super(SpatialAttention, self).__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.Tanh(),
            nn.Linear(embed_dim // 4, 1)
        )

    def forward(self, patch_tokens):
        attn_scores = self.attention_net(patch_tokens)
        attn_weights = F.softmax(attn_scores, dim=1)
        attended_features = torch.sum(patch_tokens * attn_weights, dim=1)
        
        return attended_features, attn_weights


class MSAIQA(nn.Module):
    """
    Arquitetura MSA-IQA (Multilayer Spatial Attention for IQA).
    """
    def __init__(self, freeze_backbone=True):
        super(MSAIQA, self).__init__()
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        embed_dim = self.backbone.embed_dim
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
                
        self.spatial_attention = SpatialAttention(embed_dim)
        
        self.regressor = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        features = self.backbone.forward_features(x)
        cls_token = features['x_norm_clstoken']
        patch_tokens = features['x_norm_patchtokens']
        
        attended_patches, attn_weights = self.spatial_attention(patch_tokens)
        
        final_features = torch.cat([cls_token, attended_patches], dim=1)
        
        quality_score = self.regressor(final_features)
        
        return quality_score, attn_weights
