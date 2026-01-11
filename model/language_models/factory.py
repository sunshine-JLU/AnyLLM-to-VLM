"""
语言模型工厂
用于注册和创建不同的语言模型
"""

from typing import Dict, Type, Optional
from .base import BaseLanguageModel
from .qwen_model import QwenLanguageModel


class LanguageModelFactory:
    """语言模型工厂类"""
    
    _registry: Dict[str, Type[BaseLanguageModel]] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type[BaseLanguageModel]):
        """
        注册语言模型
        
        Args:
            name: 模型名称（如 'qwen', 'llama', 'gpt'）
            model_class: 模型类
        """
        cls._registry[name.lower()] = model_class
        print(f"注册语言模型: {name} -> {model_class.__name__}")
    
    @classmethod
    def create(
        cls,
        model_type: str,
        model_path: str,
        freeze: bool = False,
        layers_to_unfreeze: int = 0,
        use_bfloat16: bool = True,
        **kwargs
    ) -> BaseLanguageModel:
        """
        创建语言模型实例
        
        Args:
            model_type: 模型类型（如 'qwen', 'llama', 'gpt'）
            model_path: 模型路径
            freeze: 是否冻结参数
            layers_to_unfreeze: 解冻后几层
            use_bfloat16: 是否使用bfloat16
            **kwargs: 其他参数
        
        Returns:
            语言模型实例
        """
        model_type = model_type.lower()
        
        if model_type not in cls._registry:
            raise ValueError(
                f"未知的语言模型类型: {model_type}\n"
                f"可用的模型: {list(cls._registry.keys())}\n"
                f"请先注册模型或检查配置"
            )
        
        model_class = cls._registry[model_type]
        
        # 创建实例
        model = model_class(
            model_path=model_path,
            freeze=freeze,
            layers_to_unfreeze=layers_to_unfreeze,
            use_bfloat16=use_bfloat16,
            **kwargs
        )
        
        return model
    
    @classmethod
    def list_models(cls) -> list:
        """列出所有已注册的模型"""
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """检查模型是否已注册"""
        return model_type.lower() in cls._registry


# 自动注册内置模型
LanguageModelFactory.register('qwen', QwenLanguageModel)

# 可以在这里注册更多模型
# LanguageModelFactory.register('llama', LLaMALanguageModel)
# LanguageModelFactory.register('gpt', GPTLanguageModel)
# LanguageModelFactory.register('chatglm', ChatGLMLanguageModel)
