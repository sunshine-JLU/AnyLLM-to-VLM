"""
视觉编码器基类
所有视觉编码器都需要继承这个基类并实现相应的方法
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaseVisionEncoder(ABC, nn.Module):
    """视觉编码器基类"""
    
    def __init__(self, model_path: str, freeze: bool = True):
        """
        初始化视觉编码器
        
        Args:
            model_path: 模型路径
            freeze: 是否冻结参数
        """
        super().__init__()
        self.model_path = model_path
        self.freeze = freeze
        self._hidden_size = None
        self._processor = None
    
    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        加载视觉编码器模型
        
        Returns:
            视觉编码器模型
        """
        pass
    
    @abstractmethod
    def load_processor(self):
        """
        加载图像处理器
        
        Returns:
            图像处理器
        """
        pass
    
    @abstractmethod
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取图像嵌入
        
        Args:
            pixel_values: 图像tensor [batch_size, channels, height, width]
        
        Returns:
            图像嵌入 [batch_size, num_patches, hidden_size]
        """
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        返回视觉编码器的隐藏层维度
        
        Returns:
            隐藏层维度
        """
        pass
    
    @property
    def processor(self):
        """
        返回图像处理器
        
        Returns:
            图像处理器
        """
        if self._processor is None:
            self._processor = self.load_processor()
        return self._processor
    
    def freeze_params(self):
        """冻结所有参数"""
        for param in self.parameters():
            param.requires_grad = False
    
    def unfreeze_params(self):
        """解冻所有参数"""
        for param in self.parameters():
            param.requires_grad = True
    
    def unfreeze_layers(self, num_layers: int):
        """
        解冻后几层
        
        Args:
            num_layers: 要解冻的层数（从后往前）
        """
        # 默认实现：如果子类没有实现，则解冻所有参数
        if num_layers > 0:
            self.unfreeze_params()
        else:
            self.freeze_params()
