"""
投影层基类
所有投影层都需要继承这个基类并实现相应的方法
"""

import torch
import torch.nn as nn
from typing import Optional
from abc import ABC, abstractmethod


class BaseProjection(ABC, nn.Module):
    """投影层基类"""
    
    def __init__(self, vision_dim: int, language_dim: int, **kwargs):
        """
        初始化投影层
        
        Args:
            vision_dim: 视觉编码器维度
            language_dim: 语言模型维度
            **kwargs: 其他配置参数
        """
        super().__init__()
        self.vision_dim = vision_dim
        self.language_dim = language_dim
        self.projection = self.build_projection(**kwargs)
        
        # 投影层始终可训练
        for param in self.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def build_projection(self, **kwargs) -> nn.Module:
        """
        构建投影层
        
        Args:
            **kwargs: 配置参数
        
        Returns:
            投影层模块
        """
        pass
    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            image_embeddings: 图像嵌入 [batch_size, num_patches, vision_dim]
        
        Returns:
            投影后的嵌入 [batch_size, num_patches, language_dim]
        """
        return self.projection(image_embeddings)
