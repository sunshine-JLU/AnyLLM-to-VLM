"""
语言模型基类
所有语言模型都需要继承这个基类并实现相应的方法
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from abc import ABC, abstractmethod


class BaseLanguageModel(ABC, nn.Module):
    """语言模型基类"""
    
    def __init__(self, model_path: str, freeze: bool = False, layers_to_unfreeze: int = 0, use_bfloat16: bool = True):
        """
        初始化语言模型
        
        Args:
            model_path: 模型路径
            freeze: 是否冻结参数
            layers_to_unfreeze: 解冻后几层
            use_bfloat16: 是否使用bfloat16
        """
        super().__init__()
        self.model_path = model_path
        self.freeze = freeze
        self.layers_to_unfreeze = layers_to_unfreeze
        self.use_bfloat16 = use_bfloat16
        self._hidden_size = None
        self._tokenizer = None
        self.model = self.load_model()
        self._hidden_size = self._get_hidden_size()
        
        # 设置参数冻结
        if freeze:
            self.freeze_params()
        else:
            self.freeze_params()  # 先全部冻结
            if layers_to_unfreeze > 0:
                self.unfreeze_layers(layers_to_unfreeze)
            # 始终解冻embedding和输出层
            self.unfreeze_embedding_and_output()
    
    @abstractmethod
    def load_model(self) -> nn.Module:
        """
        加载语言模型
        
        Returns:
            语言模型
        """
        pass
    
    @abstractmethod
    def load_tokenizer(self):
        """
        加载tokenizer
        
        Returns:
            tokenizer
        """
        pass
    
    @abstractmethod
    def _get_hidden_size(self) -> int:
        """
        获取隐藏层维度
        
        Returns:
            隐藏层维度
        """
        pass
    
    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        返回隐藏层维度
        
        Returns:
            隐藏层维度
        """
        pass
    
    @property
    def tokenizer(self):
        """
        返回tokenizer
        
        Returns:
            tokenizer
        """
        if self._tokenizer is None:
            self._tokenizer = self.load_tokenizer()
        return self._tokenizer
    
    @property
    def config(self):
        """返回模型配置"""
        return self.model.config
    
    def freeze_params(self):
        """冻结所有参数"""
        for param in self.model.parameters():
            param.requires_grad = False
    
    def unfreeze_params(self):
        """解冻所有参数"""
        for param in self.model.parameters():
            param.requires_grad = True
    
    @abstractmethod
    def unfreeze_layers(self, num_layers: int):
        """
        解冻后几层
        
        Args:
            num_layers: 要解冻的层数（从后往前）
        """
        pass
    
    @abstractmethod
    def unfreeze_embedding_and_output(self):
        """解冻embedding层和输出层"""
        pass
    
    def get_embedding_layer(self):
        """获取embedding层"""
        # 尝试多种可能的路径
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
            return self.model.model.embed_tokens
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
            return self.model.transformer.wte
        elif hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens
        elif hasattr(self.model, 'get_input_embeddings'):
            # 某些模型可能有get_input_embeddings方法
            return self.model.get_input_embeddings()
        return None
    
    def forward(self, *args, **kwargs):
        """前向传播"""
        return self.model(*args, **kwargs)
