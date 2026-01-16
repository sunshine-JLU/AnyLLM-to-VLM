"""
语言模型模块
支持多种语言模型：Qwen3, LLaMA, GPT, ChatGLM等
默认支持 Qwen3-0.6B 和 Qwen2.5-0.5B
"""

from .base import BaseLanguageModel
from .qwen3_model import Qwen3LanguageModel
from .factory import LanguageModelFactory

# 向后兼容：使用Qwen3LanguageModel作为QwenLanguageModel的别名
QwenLanguageModel = Qwen3LanguageModel

__all__ = [
    'BaseLanguageModel',
    'Qwen3LanguageModel',
    'QwenLanguageModel',  # 向后兼容
    'LanguageModelFactory',
]
