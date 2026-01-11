"""
语言模型模块
支持多种语言模型：Qwen3, LLaMA, GPT, ChatGLM等
"""

from .base import BaseLanguageModel
from .qwen3_model import Qwen3LanguageModel
from .factory import LanguageModelFactory

# 向后兼容：导出旧名称
try:
    from .qwen_model import QwenLanguageModel
except ImportError:
    QwenLanguageModel = Qwen3LanguageModel

__all__ = [
    'BaseLanguageModel',
    'Qwen3LanguageModel',
    'QwenLanguageModel',  # 向后兼容
    'LanguageModelFactory',
]
