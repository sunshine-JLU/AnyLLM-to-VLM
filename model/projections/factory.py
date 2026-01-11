"""
投影层工厂
用于注册和创建不同的投影层
"""

from typing import Dict, Type, Optional
from .base import BaseProjection
from .mlp_projection import MLPProjection
from .linear_projection import LinearProjection


class ProjectionFactory:
    """投影层工厂类"""
    
    _registry: Dict[str, Type[BaseProjection]] = {}
    
    @classmethod
    def register(cls, name: str, projection_class: Type[BaseProjection]):
        """
        注册投影层
        
        Args:
            name: 投影层名称（如 'mlp', 'linear', 'perceiver'）
            projection_class: 投影层类
        """
        cls._registry[name.lower()] = projection_class
        print(f"注册投影层: {name} -> {projection_class.__name__}")
    
    @classmethod
    def create(
        cls,
        projection_type: str,
        vision_dim: int,
        language_dim: int,
        **kwargs
    ) -> BaseProjection:
        """
        创建投影层实例
        
        Args:
            projection_type: 投影层类型（如 'mlp', 'linear'）
            vision_dim: 视觉编码器维度
            language_dim: 语言模型维度
            **kwargs: 其他配置参数
        
        Returns:
            投影层实例
        """
        projection_type = projection_type.lower()
        
        if projection_type not in cls._registry:
            raise ValueError(
                f"未知的投影层类型: {projection_type}\n"
                f"可用的投影层: {list(cls._registry.keys())}\n"
                f"请先注册投影层或检查配置"
            )
        
        projection_class = cls._registry[projection_type]
        
        # 创建实例
        projection = projection_class(
            vision_dim=vision_dim,
            language_dim=language_dim,
            **kwargs
        )
        
        return projection
    
    @classmethod
    def list_projections(cls) -> list:
        """列出所有已注册的投影层"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, projection_type: str) -> bool:
        """检查投影层是否已注册"""
        return projection_type.lower() in cls._registry


# 自动注册内置投影层
ProjectionFactory.register('mlp', MLPProjection)
ProjectionFactory.register('linear', LinearProjection)

# 可以在这里注册更多投影层
# ProjectionFactory.register('perceiver', PerceiverProjection)
# ProjectionFactory.register('cross_attention', CrossAttentionProjection)
