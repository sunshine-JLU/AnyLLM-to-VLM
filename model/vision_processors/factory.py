"""
视觉特征处理器工厂
用于注册和创建不同的视觉特征处理器
"""

from typing import Dict, Type
from .base import BaseVisionProcessor


class VisionProcessorFactory:
    """视觉特征处理器工厂类"""
    
    _registry: Dict[str, Type[BaseVisionProcessor]] = {}
    
    @classmethod
    def register(cls, name: str, processor_class: Type[BaseVisionProcessor]):
        """
        注册视觉特征处理器
        
        Args:
            name: 处理器名称（如 'patch_insert', 'mean_pooling'）
            processor_class: 处理器类
        """
        cls._registry[name.lower()] = processor_class
        print(f"注册视觉特征处理器: {name} -> {processor_class.__name__}")
    
    @classmethod
    def create(
        cls,
        processor_type: str,
        **kwargs
    ) -> BaseVisionProcessor:
        """
        创建视觉特征处理器实例
        
        Args:
            processor_type: 处理器类型（如 'patch_insert', 'mean_pooling'）
            **kwargs: 其他参数
        
        Returns:
            视觉特征处理器实例
        """
        processor_type = processor_type.lower()
        
        if processor_type not in cls._registry:
            raise ValueError(
                f"未知的视觉特征处理器类型: {processor_type}\n"
                f"可用的处理器: {list(cls._registry.keys())}\n"
                f"请先注册处理器或检查配置"
            )
        
        processor_class = cls._registry[processor_type]
        
        # 创建实例
        processor = processor_class(**kwargs)
        
        return processor
    
    @classmethod
    def list_processors(cls) -> list:
        """列出所有已注册的处理器"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, processor_type: str) -> bool:
        """检查处理器是否已注册"""
        return processor_type.lower() in cls._registry
