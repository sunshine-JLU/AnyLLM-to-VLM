"""
视觉编码器工厂
用于注册和创建不同的视觉编码器
"""

from typing import Dict, Type, Optional
from .base import BaseVisionEncoder
from .clip_encoder import CLIPVisionEncoder
from .siglip_encoder import SigLIPVisionEncoder


class VisionEncoderFactory:
    """视觉编码器工厂类"""
    
    _registry: Dict[str, Type[BaseVisionEncoder]] = {}
    
    @classmethod
    def register(cls, name: str, encoder_class: Type[BaseVisionEncoder]):
        """
        注册视觉编码器
        
        Args:
            name: 编码器名称（如 'clip', 'vit', 'dinov2'）
            encoder_class: 编码器类
        """
        cls._registry[name.lower()] = encoder_class
        print(f"注册视觉编码器: {name} -> {encoder_class.__name__}")
    
    @classmethod
    def create(
        cls,
        encoder_type: str,
        model_path: str,
        freeze: bool = True,
        vision_layers_to_unfreeze: int = 0,
        **kwargs
    ) -> BaseVisionEncoder:
        """
        创建视觉编码器实例
        
        Args:
            encoder_type: 编码器类型（如 'clip', 'vit', 'dinov2'）
            model_path: 模型路径
            freeze: 是否冻结参数
            vision_layers_to_unfreeze: 解冻后几层
            **kwargs: 其他参数
        
        Returns:
            视觉编码器实例
        """
        encoder_type = encoder_type.lower()
        
        if encoder_type not in cls._registry:
            raise ValueError(
                f"未知的视觉编码器类型: {encoder_type}\n"
                f"可用的编码器: {list(cls._registry.keys())}\n"
                f"请先注册编码器或检查配置"
            )
        
        encoder_class = cls._registry[encoder_type]
        
        # 创建实例
        encoder = encoder_class(
            model_path=model_path,
            freeze=freeze,
            vision_layers_to_unfreeze=vision_layers_to_unfreeze,
            **kwargs
        )
        
        return encoder
    
    @classmethod
    def list_encoders(cls) -> list:
        """列出所有已注册的编码器"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, encoder_type: str) -> bool:
        """检查编码器是否已注册"""
        return encoder_type.lower() in cls._registry


# 自动注册内置编码器
VisionEncoderFactory.register('clip', CLIPVisionEncoder)
VisionEncoderFactory.register('siglip', SigLIPVisionEncoder)

# 可以在这里注册更多编码器
# VisionEncoderFactory.register('vit', ViTVisionEncoder)
# VisionEncoderFactory.register('dinov2', DINOv2VisionEncoder)
# VisionEncoderFactory.register('blip', BLIPVisionEncoder)
