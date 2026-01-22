"""
SigLIP视觉编码器实现
SigLIP是Google开发的视觉编码器，是CLIP的改进版本
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import SiglipVisionModel, SiglipImageProcessor, AutoModel, AutoProcessor

from .base import BaseVisionEncoder


class SigLIPVisionEncoder(BaseVisionEncoder):
    """SigLIP视觉编码器"""
    
    def __init__(self, model_path: str, freeze: bool = True, vision_layers_to_unfreeze: int = 0):
        """
        初始化SigLIP视觉编码器
        
        Args:
            model_path: SigLIP模型路径或HuggingFace模型标识符
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
        """加载SigLIP模型"""
        try:
            # 尝试加载视觉模型
            # SigLIP 可以通过 SiglipVisionModel 或 AutoModel 加载
            try:
                model = SiglipVisionModel.from_pretrained(self.model_path)
                print(f"  成功加载SigLIP视觉模型: {self.model_path}")
            except Exception:
                # 如果失败，尝试使用 AutoModel
                model = AutoModel.from_pretrained(self.model_path)
                # 如果加载的是完整模型，提取视觉部分
                if hasattr(model, 'vision_model'):
                    model = model.vision_model
                print(f"  成功通过AutoModel加载SigLIP视觉模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"加载SigLIP模型失败: {e}")
            print(f"请确保模型路径正确，或模型已正确安装 transformers 库")
            raise
    
    def load_processor(self):
        """加载SigLIP处理器"""
        try:
            # 尝试使用 SiglipImageProcessor
            try:
                processor = SiglipImageProcessor.from_pretrained(self.model_path)
            except Exception:
                # 如果失败，尝试使用 AutoProcessor
                processor = AutoProcessor.from_pretrained(self.model_path)
                # AutoProcessor 可能返回包含文本处理器的对象，我们需要图像处理器部分
                if hasattr(processor, 'image_processor'):
                    processor = processor.image_processor
            return processor
        except Exception as e:
            print(f"加载SigLIP处理器失败: {e}")
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
            outputs = self.model(pixel_values=pixel_values)
        
        # SigLIP SO400M Patch14-384 没有 class token
        # 输出格式: [batch_size, num_patches, hidden_size] (对于384x384输入，num_patches = 729)
        # 直接使用全部输出，不需要去掉第一个token
        img_embedding = outputs.last_hidden_state  # [batch_size, num_patches, hidden_size]
        
        # 验证：检查输出形状是否符合预期（可选）
        # 对于384x384输入，patch size 14，应该有 (384//14)**2 = 729 个patches
        return img_embedding
    
    def _get_hidden_size(self) -> int:
        """获取隐藏层维度"""
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        # SigLIP SO400M 的默认隐藏层维度是 1152
        return 1152
    
    @property
    def hidden_size(self) -> int:
        """返回隐藏层维度"""
        return self._hidden_size
    
    def unfreeze_layers(self, num_layers: int):
        """
        解冻SigLIP视觉编码器的后几层
        
        Args:
            num_layers: 要解冻的层数（从后往前）
        """
        if num_layers <= 0:
            self.freeze_params()
            return
        
        # 先冻结所有参数
        self.freeze_params()
        
        # 获取transformer层
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layers'):
            encoder = self.model.encoder
            layers = encoder.layers
            num_total_layers = len(layers)
            
            if num_layers > 0:
                # 解冻后几层
                start_layer = max(0, num_total_layers - num_layers)
                for i in range(start_layer, num_total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  SigLIP视觉编码器解冻后 {num_layers} 层 (层 {start_layer} 到 {num_total_layers-1})")
            
            # 解冻输出层（如果有）
            if hasattr(self.model, 'post_layernorm'):
                for param in self.model.post_layernorm.parameters():
                    param.requires_grad = True
