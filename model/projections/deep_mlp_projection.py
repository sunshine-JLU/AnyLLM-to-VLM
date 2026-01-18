"""
深层MLP投影层实现
参考LLaVA、BLIP等项目的多层MLP结构
"""

import torch
import torch.nn as nn
from typing import Optional

from .base import BaseProjection


class DeepMLPProjection(BaseProjection):
    """深层MLP投影层（3-4层）"""
    
    def __init__(
        self,
        vision_dim: int,
        language_dim: int,
        hidden_dim: Optional[int] = None,
        num_layers: int = 3,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm: bool = True,
        **kwargs
    ):
        """
        初始化深层MLP投影层
        
        Args:
            vision_dim: 视觉编码器维度
            language_dim: 语言模型维度
            hidden_dim: 隐藏层维度（如果为None，则使用language_dim * 4）
            num_layers: MLP层数（3或4层，不包括输入层和输出层）
            activation: 激活函数 ('gelu', 'relu', 'swish')
            dropout: dropout率
            layernorm: 是否使用LayerNorm
        """
        self.hidden_dim = hidden_dim or (language_dim * 4)
        self.num_layers = max(3, min(4, num_layers))  # 限制在3-4层
        self.activation = activation
        self.dropout = dropout
        self.layernorm = layernorm
        
        super().__init__(vision_dim, language_dim, **kwargs)
    
    def build_projection(self, **kwargs) -> nn.Module:
        """构建深层MLP投影层"""
        layers = []
        
        # 第一层：vision_dim -> hidden_dim
        layers.append(nn.Linear(self.vision_dim, self.hidden_dim))
        layers.append(self._get_activation())
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        if self.layernorm:
            layers.append(nn.LayerNorm(self.hidden_dim))
        
        # 中间层：hidden_dim -> hidden_dim（可以有1-2层）
        for _ in range(self.num_layers - 2):  # num_layers-2是因为去掉输入层和输出层
            layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
            layers.append(self._get_activation())
            if self.dropout > 0:
                layers.append(nn.Dropout(self.dropout))
            if self.layernorm:
                layers.append(nn.LayerNorm(self.hidden_dim))
        
        # 最后一层：hidden_dim -> language_dim（不使用激活函数和layernorm）
        layers.append(nn.Linear(self.hidden_dim, self.language_dim))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self) -> nn.Module:
        """获取激活函数"""
        if self.activation == "gelu":
            return nn.GELU()
        elif self.activation == "relu":
            return nn.ReLU()
        elif self.activation == "swish" or self.activation == "silu":
            return nn.SiLU()
        elif self.activation == "tanh":
            return nn.Tanh()
        else:
            raise ValueError(f"不支持的激活函数: {self.activation}")
    
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
        x = image_embeddings.reshape(-1, vision_dim)
        
        # 通过投影层
        x = self.projection(x)
        
        # 重塑回 [batch_size, num_patches, language_dim]
        x = x.reshape(batch_size, num_patches, self.language_dim)
        
        return x
