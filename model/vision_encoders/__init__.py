"""
视觉编码器模块
支持多种视觉编码器：CLIP, ViT, DINOv2, BLIP等
"""

from .base import BaseVisionEncoder
from .clip_encoder import CLIPVisionEncoder
from .factory import VisionEncoderFactory

# 可选：导入其他编码器（如果存在）
try:
    from .vit_encoder import ViTVisionEncoder
    __all__ = [
        'BaseVisionEncoder',
        'CLIPVisionEncoder',
        'ViTVisionEncoder',
        'VisionEncoderFactory',
    ]
except ImportError:
    __all__ = [
        'BaseVisionEncoder',
        'CLIPVisionEncoder',
        'VisionEncoderFactory',
    ]
