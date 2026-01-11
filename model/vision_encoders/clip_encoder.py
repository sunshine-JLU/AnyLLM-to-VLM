"""
CLIP视觉编码器实现
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPModel, CLIPProcessor

from .base import BaseVisionEncoder


class CLIPVisionEncoder(BaseVisionEncoder):
    """CLIP视觉编码器"""
    
    def __init__(self, model_path: str, freeze: bool = True, vision_layers_to_unfreeze: int = 0):
        """
        初始化CLIP视觉编码器
        
        Args:
            model_path: CLIP模型路径
            freeze: 是否冻结参数
            vision_layers_to_unfreeze: 解冻后几层
        """
        super().__init__(model_path, freeze)
        self.vision_layers_to_unfreeze = vision_layers_to_unfreeze
        self.model = self.load_model()
        self._hidden_size = self._get_hidden_size()
        
        # 设置参数冻结
        if freeze:
            self.freeze_params()
            if vision_layers_to_unfreeze > 0:
                self.unfreeze_layers(vision_layers_to_unfreeze)
        else:
            self.unfreeze_params()
    
    def load_model(self) -> nn.Module:
        """加载CLIP模型"""
        try:
            model = CLIPModel.from_pretrained(self.model_path)
            print(f"  成功加载CLIP模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"加载CLIP模型失败: {e}")
            raise
    
    def load_processor(self):
        """加载CLIP处理器"""
        try:
            processor = CLIPProcessor.from_pretrained(self.model_path, use_fast=False)
            return processor
        except Exception as e:
            print(f"加载CLIP处理器失败: {e}")
            raise
    
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取图像嵌入
        
        Args:
            pixel_values: 图像tensor [batch_size, channels, height, width]
        
        Returns:
            图像嵌入 [batch_size, num_patches, hidden_size]
        """
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model.vision_model(pixel_values=pixel_values)
        
        # 使用patch embeddings（去掉CLS token）
        # CLIP ViT的输出格式: [batch_size, num_patches+1, hidden_size]
        # 第一个token是CLS token，我们去掉它
        img_embedding = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, 768]
        return img_embedding
    
    def _get_hidden_size(self) -> int:
        """获取隐藏层维度"""
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'config'):
            return self.model.vision_model.config.hidden_size
        return 768  # CLIP base的默认值
    
    @property
    def hidden_size(self) -> int:
        """返回隐藏层维度"""
        return self._hidden_size
    
    def unfreeze_layers(self, num_layers: int):
        """
        解冻CLIP视觉编码器的后几层
        
        Args:
            num_layers: 要解冻的层数（从后往前）
        """
        if num_layers <= 0:
            self.freeze_params()
            return
        
        # 先冻结所有参数
        self.freeze_params()
        
        # 获取transformer层
        if hasattr(self.model, 'vision_model') and hasattr(self.model.vision_model, 'encoder'):
            encoder = self.model.vision_model.encoder
            if hasattr(encoder, 'layers'):
                layers = encoder.layers
                num_total_layers = len(layers)
                
                if num_layers > 0:
                    # 解冻后几层
                    start_layer = max(0, num_total_layers - num_layers)
                    for i in range(start_layer, num_total_layers):
                        for param in layers[i].parameters():
                            param.requires_grad = True
                    print(f"  CLIP视觉编码器解冻后 {num_layers} 层 (层 {start_layer} 到 {num_total_layers-1})")
                
                # 解冻输出层（如果有）
                if hasattr(self.model.vision_model, 'post_layernorm'):
                    for param in self.model.vision_model.post_layernorm.parameters():
                        param.requires_grad = True
