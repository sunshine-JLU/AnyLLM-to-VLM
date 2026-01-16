"""
视觉特征处理模块
支持不同的视觉特征处理方式：patch插入、平均pooling等
"""

from .base import BaseVisionProcessor
from .factory import VisionProcessorFactory
from .patch_insert_processor import PatchInsertVisionProcessor
from .mean_pooling_processor import MeanPoolingVisionProcessor

# 自动注册内置处理器
VisionProcessorFactory.register('patch_insert', PatchInsertVisionProcessor)
VisionProcessorFactory.register('mean_pooling', MeanPoolingVisionProcessor)

# 向后兼容：保留旧的名称
VisionProcessorFactory.register('minimind', PatchInsertVisionProcessor)  # 向后兼容
VisionProcessorFactory.register('pooling', MeanPoolingVisionProcessor)  # 向后兼容

__all__ = [
    'BaseVisionProcessor',
    'VisionProcessorFactory',
    'PatchInsertVisionProcessor',
    'MeanPoolingVisionProcessor',
]
