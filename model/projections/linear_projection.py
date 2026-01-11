"""
简单线性投影层实现
"""

import torch
import torch.nn as nn

from .base import BaseProjection


class LinearProjection(BaseProjection):
    """简单线性投影层"""
    
    def build_projection(self, **kwargs) -> nn.Module:
        """构建线性投影层"""
        return nn.Linear(self.vision_dim, self.language_dim)
    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_embeddings: [batch_size, num_patches, vision_dim]
        
        Returns:
            [batch_size, num_patches, language_dim]
        """
        return self.projection(image_embeddings)
