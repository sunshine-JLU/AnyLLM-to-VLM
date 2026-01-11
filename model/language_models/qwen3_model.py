"""
Qwen3语言模型实现
支持所有Qwen3系列模型（0.5B, 0.6B, 1.5B, 2B, 7B等）
"""

import torch
import torch.nn as nn
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import BaseLanguageModel


class Qwen3LanguageModel(BaseLanguageModel):
    """
    Qwen3语言模型
    
    支持所有Qwen3系列模型，包括：
    - Qwen3-0.5B
    - Qwen3-0.6B
    - Qwen3-1.5B
    - Qwen3-2B
    - Qwen3-7B
    等不同尺寸的模型
    
    使用 AutoModelForCausalLM 自动适配不同架构
    """
    
    def load_model(self) -> nn.Module:
        """加载Qwen3模型"""
        try:
            dtype = torch.bfloat16 if self.use_bfloat16 else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                dtype=dtype,
                low_cpu_mem_usage=True
            )
            print(f"  成功加载Qwen3模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"加载Qwen3模型失败: {e}")
            # 尝试不使用dtype
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True
                )
                print(f"  成功加载Qwen3模型（无dtype）: {self.model_path}")
                return model
            except Exception as e2:
                print(f"再次尝试加载失败: {e2}")
                raise
    
    def load_tokenizer(self):
        """加载Qwen3 tokenizer"""
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except Exception as e:
            print(f"加载Qwen3 tokenizer失败: {e}")
            raise
    
    def _get_hidden_size(self) -> int:
        """获取隐藏层维度"""
        if hasattr(self.model, 'config') and hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        # Qwen3不同尺寸的默认值（会根据实际模型自动获取）
        return 4096  # 默认值，实际会从config获取
    
    @property
    def hidden_size(self) -> int:
        """返回隐藏层维度"""
        return self._hidden_size
    
    def unfreeze_layers(self, num_layers: int):
        """
        解冻Qwen3语言模型的后几层
        
        Args:
            num_layers: 要解冻的层数（从后往前）
        """
        if num_layers <= 0:
            return
        
        # 获取transformer层
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Qwen3标准结构
            layers = self.model.model.layers
            num_total_layers = len(layers)
            
            if num_layers > 0:
                start_layer = max(0, num_total_layers - num_layers)
                for i in range(start_layer, num_total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  Qwen3语言模型解冻后 {num_layers} 层 (层 {start_layer} 到 {num_total_layers-1})")
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2结构（兼容性）
            layers = self.model.transformer.h
            num_total_layers = len(layers)
            
            if num_layers > 0:
                start_layer = max(0, num_total_layers - num_layers)
                for i in range(start_layer, num_total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  Qwen3语言模型解冻后 {num_layers} 层 (层 {start_layer} 到 {num_total_layers-1})")
        else:
            # 未知结构，解冻所有参数
            if num_layers > 0:
                self.unfreeze_params()
                print(f"  警告: 未知Qwen3模型结构，解冻所有参数")
    
    def unfreeze_embedding_and_output(self):
        """解冻embedding层和输出层"""
        # 解冻embedding层
        embedding_layer = self.get_embedding_layer()
        if embedding_layer is not None:
            for param in embedding_layer.parameters():
                param.requires_grad = True
            print(f"  Qwen3语言模型embedding层已解冻")
        else:
            # 尝试其他路径
            if hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                for param in self.model.model.embed_tokens.parameters():
                    param.requires_grad = True
                print(f"  Qwen3语言模型embedding层已解冻")
            elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'wte'):
                for param in self.model.transformer.wte.parameters():
                    param.requires_grad = True
                print(f"  Qwen3语言模型embedding层已解冻")
        
        # 解冻输出层
        if hasattr(self.model, 'lm_head'):
            for param in self.model.lm_head.parameters():
                param.requires_grad = True
            print(f"  Qwen3语言模型输出层(lm_head)已解冻")
        elif hasattr(self.model, 'get_output_embeddings'):
            output_layer = self.model.get_output_embeddings()
            if output_layer is not None:
                for param in output_layer.parameters():
                    param.requires_grad = True
                print(f"  Qwen3语言模型输出层已解冻")
