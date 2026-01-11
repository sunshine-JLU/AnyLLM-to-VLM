"""
MLP投影层实现
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseProjection


class MLPProjection(BaseProjection):
    """MLP投影层"""
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm: bool = True,
        **kwargs
    ):
        """
        初始化MLP投影层
        
        Args:
            vision_dim: 视觉编码器维度
            language_dim: 语言模型维度
            hidden_dim: 隐藏层维度（如果为None，则使用language_dim * 4）
            activation: 激活函数
            dropout: dropout率
            layernorm: 是否使用LayerNorm
        """
        self.hidden_dim = hidden_dim or (language_dim * 4)
        self.activation = activation
        self.dropout = dropout
        self.layernorm = layernorm
        super().__init__(vision_dim, language_dim, **kwargs)
    
    def build_projection(self, **kwargs) -> nn.Module:
        """构建MLP投影层"""
        layers = []
        
        # 第一层：vision_dim -> hidden_dim
        layers.append(nn.Linear(self.vision_dim, self.hidden_dim))
        
        # 激活函数
        if self.activation == "gelu":
            layers.append(nn.GELU())
        elif self.activation == "relu":
            layers.append(nn.ReLU())
        elif self.activation == "tanh":
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
        
        # Dropout
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        
        # LayerNorm
        if self.layernorm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        
        # 第二层：hidden_dim -> language_dim
        layers.append(nn.Linear(self.hidden_dim, self.language_dim))
        
        return nn.Sequential(*layers)
    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_embeddings: [batch_size, num_patches, vision_dim]
        
        Returns:
            [batch_size, num_patches, language_dim]
        """
        # 处理每个patch
        batch_size, num_patches, vision_dim = image_embeddings.shape
        
        # 重塑为 [batch_size * num_patches, vision_dim]
        x = image_embeddings.view(-1, vision_dim)
        
        # 通过投影层
        x = self.projection(x)
        
        # 重塑回 [batch_size, num_patches, language_dim]
        x = x.view(batch_size, num_patches, self.language_dim)
        
        return x
