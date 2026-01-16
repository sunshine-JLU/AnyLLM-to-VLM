import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Union
import json

@dataclass
class ModelConfig:
    """模型配置"""
    # 视觉模型配置
    vision_model_type: str = "clip"
    vision_model_path: str = "./models/clip-vit-base-patch16"
    freeze_vision_encoder: bool = True
    vision_layers_to_unfreeze: int = 0
    
    # 语言模型配置
    language_model_type: str = "qwen"
    language_model_path: str = "./models/Qwen3-0.6B"
    freeze_language_model: bool = False
    language_layers_to_unfreeze: int = 0
    
    # 投影层配置
    projection_type: str = "mlp"
    projection_hidden_dim: Optional[int] = None
    projection_activation: str = "gelu"
    projection_dropout: float = 0.1
    projection_layernorm: bool = True
    
    # 特殊token配置
    image_special_token: str = '@' * 196
    image_token_ids: Optional[List[int]] = None  # 这个字段需要保持
    
    # 视觉特征处理器配置
    vision_processor_type: str = "patch_insert"  # 'patch_insert' (直接插入所有patch) 或 'mean_pooling' (平均pooling)
    
    # 训练配置
    use_bfloat16: bool = True
    max_seq_length: int = 512
    
    def __post_init__(self):
        # 确保数值类型正确
        if self.projection_hidden_dim is None:
            if "base" in self.vision_model_path:
                self.projection_hidden_dim = 3072  # 4 * 768 for CLIP base
            elif "large" in self.vision_model_path:
                self.projection_hidden_dim = 4096  # 4 * 1024 for CLIP large
            else:
                self.projection_hidden_dim = 3072  # 默认值
        
        # 确保image_token_ids是None（将在模型中自动设置）
        self.image_token_ids = None

@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本设置
    stage: str = "pretrain"  # pretrain or sft
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    
    # 数据集配置
    data_path: str = "../dataset/pretrain_data.parquet"
    max_seq_length: int = 512
    
    # 训练策略
    accumulation_steps: int = 1
    use_mixed_precision: bool = True
    mixed_precision_dtype: str = "bfloat16"
    
    # 保存与日志
    save_dir: str = "./checkpoints"
    save_every_n_steps: int = 1000
    log_every_n_steps: int = 100
    eval_every_n_steps: Optional[int] = None
    
    # 分布式训练
    use_ddp: bool = False
    local_rank: int = 0
    world_size: int = 1
    num_workers: int = 4
    
    def __post_init__(self):
        # 确保数值类型正确
        self.learning_rate = float(self.learning_rate)
        self.batch_size = int(self.batch_size)
        self.num_epochs = int(self.num_epochs)
        self.warmup_steps = int(self.warmup_steps)
        self.weight_decay = float(self.weight_decay)
        self.grad_clip = float(self.grad_clip)
        self.max_seq_length = int(self.max_seq_length)
        
        if self.stage == "sft":
            self.learning_rate = float(1e-5)
            self.max_seq_length = int(2048)