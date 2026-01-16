"""
平均Pooling风格的视觉特征处理器
使用平均pooling将所有patch特征聚合成单个特征，然后替换图像token
"""

import torch
from typing import List, Tuple
from .base import BaseVisionProcessor


class MeanPoolingVisionProcessor(BaseVisionProcessor):
    """
    平均Pooling风格的视觉特征处理器
    
    处理方式：
    1. 找到图像token的位置
    2. 使用平均pooling将所有patch特征聚合成单个特征
    3. 用聚合后的特征替换图像token（如果图像token是多个，则重复填充）
    
    这种方法将多个patch的信息聚合为单个特征，序列长度保持不变，适合序列长度受限的场景。
    """
    
    def __init__(self, **kwargs):
        """初始化平均Pooling处理器"""
        super().__init__(**kwargs)
    
    def process_vision_features(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        vision_proj: torch.Tensor,
        image_token_ids: List[int],
        seqlen: int = 512
    ) -> torch.Tensor:
        """
        处理视觉特征并插入到文本序列中（平均pooling后插入）
        
        Args:
            tokens: 输入token序列 [batch_size, seq_len]
            hidden_states: 文本embedding [batch_size, seq_len, hidden_dim]
            vision_proj: 投影后的视觉特征 [batch_size, num_patches, hidden_dim]
            image_token_ids: 图像特殊token的ID列表
            seqlen: 序列最大长度
        
        Returns:
            处理后的hidden_states [batch_size, seq_len, hidden_dim]
        """
        if not image_token_ids or vision_proj is None:
            return hidden_states
        
        # 找到图像token的位置
        image_indices = self.find_image_token_indices(tokens, image_token_ids)
        
        if not image_indices or all(len(indices) == 0 for indices in image_indices):
            return hidden_states
        
        # 确保投影后的特征与hidden_states的数据类型一致
        if vision_proj.dtype != hidden_states.dtype:
            vision_proj = vision_proj.to(dtype=hidden_states.dtype)
        
        batch_size = hidden_states.size(0)
        new_h = []
        
        for i in range(batch_size):
            if image_indices[i]:
                h_i = hidden_states[i]  # [seq_len, hidden_dim]
                
                # 处理每个图像token位置（从后往前处理，避免索引变化）
                for img_idx, (start_idx, end_idx) in enumerate(reversed(image_indices[i])):
                    # 获取该图像的所有patch特征 [num_patches, hidden_dim]
                    vision_features = vision_proj[i]  # [num_patches, hidden_dim]
                    
                    # 使用平均pooling得到单个特征向量 [hidden_dim]
                    vision_feat_pooled = vision_features.mean(dim=0)  # [hidden_dim]
                    
                    # 计算图像token的长度
                    token_length = end_idx - start_idx + 1
                    
                    # 用pooled特征替换图像token（保持序列长度不变）
                    # 如果图像token是多个token，用pooled特征重复填充
                    vision_feat_to_insert = vision_feat_pooled.unsqueeze(0).repeat(token_length, 1)  # [token_length, hidden_dim]
                    
                    # 替换图像token序列为视觉特征
                    h_i = torch.cat([
                        h_i[:start_idx],
                        vision_feat_to_insert,
                        h_i[end_idx + 1:]
                    ], dim=0)
                    
                    # 如果序列长度超过限制，截断
                    if h_i.size(0) > seqlen:
                        h_i = h_i[:seqlen]
                
                new_h.append(h_i)
            else:
                new_h.append(hidden_states[i])
        
        # 确保所有序列长度一致（填充或截断到原始长度）
        original_len = hidden_states.size(1)
        for i in range(len(new_h)):
            if new_h[i].size(0) < original_len:
                # 填充
                padding_size = original_len - new_h[i].size(0)
                padding = torch.zeros(
                    padding_size, 
                    new_h[i].size(1), 
                    dtype=new_h[i].dtype, 
                    device=new_h[i].device
                )
                new_h[i] = torch.cat([new_h[i], padding], dim=0)
            elif new_h[i].size(0) > original_len:
                # 截断到原始长度
                new_h[i] = new_h[i][:original_len]
        
        return torch.stack(new_h, dim=0)
