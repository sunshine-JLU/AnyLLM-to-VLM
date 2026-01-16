"""
视觉特征处理器基类
所有视觉特征处理器都需要继承这个基类并实现相应的方法
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List
from abc import ABC, abstractmethod


class BaseVisionProcessor(ABC):
    """视觉特征处理器基类"""
    
    def __init__(self, **kwargs):
        """
        初始化视觉特征处理器
        
        Args:
            **kwargs: 其他配置参数
        """
        pass
    
    @abstractmethod
    def process_vision_features(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        vision_proj: torch.Tensor,
        image_token_ids: List[int],
        seqlen: int = 512
    ) -> torch.Tensor:
        """
        处理视觉特征并插入到文本序列中
        
        Args:
            tokens: 输入token序列 [batch_size, seq_len]
            hidden_states: 文本embedding [batch_size, seq_len, hidden_dim]
            vision_proj: 投影后的视觉特征 [batch_size, num_patches, hidden_dim]
            image_token_ids: 图像特殊token的ID列表
            seqlen: 序列最大长度
        
        Returns:
            处理后的hidden_states [batch_size, seq_len, hidden_dim]
        """
        pass
    
    @staticmethod
    def find_image_token_indices(tokens: torch.Tensor, image_token_ids: List[int]) -> List[List[Tuple[int, int]]]:
        """
        找到图像token在序列中的位置
        
        Args:
            tokens: 输入token序列 [batch_size, seq_len]
            image_token_ids: 图像特殊token的ID列表
        
        Returns:
            每个batch中图像token的位置列表 [[(start_idx, end_idx), ...], ...]
        """
        batch_size, seq_len = tokens.shape
        len_image_ids = len(image_token_ids)
        
        if len_image_ids > seq_len:
            return [[] for _ in range(batch_size)]
        
        # 将图像id转换为tensor
        image_ids_tensor = torch.tensor(image_token_ids, device=tokens.device, dtype=tokens.dtype)
        
        # 查找匹配位置
        matches = []
        for b in range(batch_size):
            batch_matches = []
            for i in range(seq_len - len_image_ids + 1):
                if torch.all(tokens[b, i:i+len_image_ids] == image_ids_tensor):
                    batch_matches.append((i, i + len_image_ids - 1))
            matches.append(batch_matches)
        
        return matches
