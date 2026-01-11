"""
投影层模块
支持多种投影层：MLP, Linear, Perceiver等
"""

from .base import BaseProjection
from .mlp_projection import MLPProjection
from .factory import ProjectionFactory

__all__ = [
    'BaseProjection',
    'MLPProjection',
    'ProjectionFactory',
]
