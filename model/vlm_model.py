import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import warnings
import math

from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入视觉编码器模块
try:
    from .vision_encoders import VisionEncoderFactory
except ImportError:
    # 如果相对导入失败，尝试绝对导入
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.vision_encoders import VisionEncoderFactory

# 导入语言模型模块
try:
    from .language_models import LanguageModelFactory
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.language_models import LanguageModelFactory

# 导入投影层模块
try:
    from .projections import ProjectionFactory
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.projections import ProjectionFactory

# 导入视觉特征处理器模块
try:
    from .vision_processors import VisionProcessorFactory
except ImportError:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.vision_processors import VisionProcessorFactory

warnings.filterwarnings('ignore')


class MultiModalVLM(nn.Module):
    """多模态视觉语言模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"从本地路径加载视觉模型: {config.vision_model_path}")
        print(f"视觉编码器类型: {config.vision_model_type}")
        # 视觉编码器（使用工厂模式）
        self.vision_encoder = self._load_vision_model(
            encoder_type=config.vision_model_type,
            model_path=config.vision_model_path,
            freeze=config.freeze_vision_encoder,
            vision_layers_to_unfreeze=config.vision_layers_to_unfreeze
        )
        
        print(f"从本地路径加载语言模型: {config.language_model_path}")
        print(f"语言模型类型: {config.language_model_type}")
        # LoRA配置
        use_lora = getattr(config, 'use_lora', False)
        if use_lora:
            print(f"  使用LoRA训练: r={getattr(config, 'lora_r', 16)}, alpha={getattr(config, 'lora_alpha', 32)}")
        # 语言模型（使用工厂模式）
        self.language_model = self._load_language_model(
            model_type=config.language_model_type,
            model_path=config.language_model_path,
            freeze=config.freeze_language_model,
            layers_to_unfreeze=config.language_layers_to_unfreeze,
            use_bfloat16=config.use_bfloat16,
            use_lora=use_lora,
            lora_r=getattr(config, 'lora_r', 16),
            lora_alpha=getattr(config, 'lora_alpha', 32),
            lora_dropout=getattr(config, 'lora_dropout', 0.1),
            lora_target_modules=getattr(config, 'lora_target_modules', None)
        )
        
        # 投影层（使用工厂模式）
        vision_dim = self.vision_encoder.hidden_size
        language_dim = self.language_model.hidden_size
        
        print(f"创建投影层: {config.projection_type}")
        self.vision_proj = self._load_projection(
            projection_type=config.projection_type,
            vision_dim=vision_dim,
            language_dim=language_dim,
            hidden_dim=config.projection_hidden_dim,
            activation=config.projection_activation,
            dropout=config.projection_dropout,
            layernorm=config.projection_layernorm
        )
        
        # 图像特殊token
        self.image_special_token = config.image_special_token
        self.image_token_ids = None  # 初始化为None，将在_add_image_special_token中设置
        
        # 添加图像特殊token到tokenizer
        if self.image_special_token:
            self._add_image_special_token()
        
        # 视觉特征处理器（使用工厂模式）
        vision_processor_type = getattr(config, 'vision_processor_type', 'patch_insert')
        print(f"创建视觉特征处理器: {vision_processor_type}")
        self.vision_processor = self._load_vision_processor(
            processor_type=vision_processor_type
        )
        
        # 根据训练阶段设置参数冻结策略（minimind-v策略）
        training_stage = getattr(config, 'training_stage', None)
        if training_stage:
            print(f"应用训练阶段参数冻结策略: {training_stage}")
            self._apply_training_stage_freezing(training_stage)
        
        # 打印可训练参数信息
        self._print_trainable_params()
        
        print(f"模型初始化完成")
        print(f"  视觉模型维度: {vision_dim}")
        print(f"  语言模型维度: {language_dim}")
        print(f"  图像特殊token: {self.image_special_token[:20]}...")
        print(f"  视觉特征处理方式: {vision_processor_type}")
    
    def _load_vision_model(
        self,
        encoder_type: str,
        model_path: str,
        freeze: bool = True,
        vision_layers_to_unfreeze: int = 0
    ):
        """
        加载视觉模型（使用工厂模式）
        
        Args:
            encoder_type: 编码器类型（如 'clip', 'vit', 'dinov2'）
            model_path: 模型路径
            freeze: 是否冻结参数
            vision_layers_to_unfreeze: 解冻后几层
        
        Returns:
            视觉编码器实例
        """
        try:
            encoder = VisionEncoderFactory.create(
                encoder_type=encoder_type,
                model_path=model_path,
                freeze=freeze,
                vision_layers_to_unfreeze=vision_layers_to_unfreeze
            )
            print(f"  视觉编码器加载成功: {encoder_type}")
            return encoder
        except Exception as e:
            print(f"加载视觉模型失败: {e}")
            print(f"可用的编码器类型: {VisionEncoderFactory.list_encoders()}")
            raise
    
    def _load_language_model(
        self,
        model_type: str,
        model_path: str,
        freeze: bool = False,
        layers_to_unfreeze: int = 0,
        use_bfloat16: bool = True,
        use_lora: bool = False,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: Optional[List[str]] = None
    ):
        """
        加载语言模型（使用工厂模式）
        
        Args:
            model_type: 模型类型（如 'qwen', 'llama', 'gpt'）
            model_path: 模型路径
            freeze: 是否冻结参数
            layers_to_unfreeze: 解冻后几层
            use_bfloat16: 是否使用bfloat16
            use_lora: 是否使用LoRA
            lora_r: LoRA rank
            lora_alpha: LoRA alpha
            lora_dropout: LoRA dropout
            lora_target_modules: LoRA目标模块列表
        
        Returns:
            语言模型实例
        """
        try:
            model = LanguageModelFactory.create(
                model_type=model_type,
                model_path=model_path,
                freeze=freeze,
                layers_to_unfreeze=layers_to_unfreeze,
                use_bfloat16=use_bfloat16,
                use_lora=use_lora,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_target_modules=lora_target_modules
            )
            print(f"  语言模型加载成功: {model_type}")
            return model
        except Exception as e:
            print(f"加载语言模型失败: {e}")
            print(f"可用的模型类型: {LanguageModelFactory.list_models()}")
            raise
    
    def _load_projection(
        self,
        projection_type: str,
        vision_dim: int,
        language_dim: int,
        hidden_dim: Optional[int] = None,
        activation: str = "gelu",
        dropout: float = 0.1,
        layernorm: bool = True
    ):
        """
        加载投影层（使用工厂模式）
        
        Args:
            projection_type: 投影层类型（如 'mlp', 'linear'）
            vision_dim: 视觉编码器维度
            language_dim: 语言模型维度
            hidden_dim: 隐藏层维度
            activation: 激活函数
            dropout: dropout率
            layernorm: 是否使用LayerNorm
        
        Returns:
            投影层实例
        """
        try:
            projection = ProjectionFactory.create(
                projection_type=projection_type,
                vision_dim=vision_dim,
                language_dim=language_dim,
                hidden_dim=hidden_dim,
                activation=activation,
                dropout=dropout,
                layernorm=layernorm
            )
            print(f"  投影层加载成功: {projection_type}")
            return projection
        except Exception as e:
            print(f"加载投影层失败: {e}")
            print(f"可用的投影层类型: {ProjectionFactory.list_projections()}")
            raise
    
    def _load_vision_processor(
        self,
        processor_type: str
    ):
        """
        加载视觉特征处理器（使用工厂模式）
        
        Args:
            processor_type: 处理器类型（如 'patch_insert', 'mean_pooling'）
        
        Returns:
            视觉特征处理器实例
        """
        try:
            processor = VisionProcessorFactory.create(
                processor_type=processor_type
            )
            print(f"  视觉特征处理器加载成功: {processor_type}")
            return processor
        except Exception as e:
            print(f"加载视觉特征处理器失败: {e}")
            print(f"可用的处理器类型: {VisionProcessorFactory.list_processors()}")
            raise
    
    def _add_image_special_token(self):
        """添加图像特殊token"""
        tokenizer = self.get_tokenizer()
        
        if self.image_special_token not in tokenizer.get_vocab():
            print(f"  添加图像特殊token到tokenizer: {self.image_special_token[:20]}...")
            # 添加特殊token
            special_tokens_dict = {'additional_special_tokens': [self.image_special_token]}
            tokenizer.add_special_tokens(special_tokens_dict)
            
            # 调整语言模型的嵌入层大小
            if hasattr(self.language_model.model, 'resize_token_embeddings'):
                self.language_model.model.resize_token_embeddings(len(tokenizer))
            else:
                # 如果模型没有resize_token_embeddings方法，尝试直接访问
                print(f"  警告: 模型没有resize_token_embeddings方法，可能需要手动调整")
            
            # 确保embedding层可训练（如果添加了新token，需要训练新的embedding）
            embedding_layer = self.language_model.get_embedding_layer()
            if embedding_layer is not None:
                for param in embedding_layer.parameters():
                    param.requires_grad = True
            
            # 获取token id
            self.image_token_ids = tokenizer.encode(
                self.image_special_token,
                add_special_tokens=False
            )
            
            print(f"  图像特殊token ID: {self.image_token_ids}")
        else:
            # 如果已经存在，获取token id
            self.image_token_ids = tokenizer.encode(
                self.image_special_token,
                add_special_tokens=False
            )
            print(f"  图像特殊token已存在，ID: {self.image_token_ids}")
    
    def _apply_training_stage_freezing(self, stage: str):
        """
        根据训练阶段应用参数冻结策略（minimind-v策略）
        
        Args:
            stage: 训练阶段 ('pretrain' 或 'sft')
        """
        use_lora = getattr(self.config, 'use_lora', False)
        
        if stage == "pretrain":
            # 预训练阶段：冻结所有参数，除了 vision_proj 层
            # 额外解冻LLM最后一层参数
            print("  预训练阶段：冻结所有参数，除了 vision_proj 层和 LLM 最后一层")
            
            if not use_lora:
                # 1. 冻结所有参数（不包括LoRA参数，如果使用LoRA）
                for param in self.parameters():
                    param.requires_grad = False
            
            # 2. 解冻 vision_proj 层
            for param in self.vision_proj.parameters():
                param.requires_grad = True
            print("  ✓ vision_proj 层已解冻")
            
            # 3. 解冻 LLM 最后一层（如果不使用LoRA）
            if not use_lora:
                self.language_model.unfreeze_layers(num_layers=1)
                print("  ✓ LLM 最后一层已解冻")
            else:
                print("  ✓ 使用LoRA，LoRA参数自动可训练")
            
        elif stage == "sft":
            # SFT阶段：保持冻结 vision_encoder
            # 解冻所有其他参数，包括 vision_proj 和所有 LLM 层
            print("  SFT阶段：冻结 vision_encoder，解冻 vision_proj 和所有 LLM 层")
            
            # 1. 确保 vision_encoder 冻结
            self.vision_encoder.freeze_params()
            print("  ✓ vision_encoder 保持冻结")
            
            # 2. 解冻 vision_proj
            for param in self.vision_proj.parameters():
                param.requires_grad = True
            print("  ✓ vision_proj 层已解冻")
            
            # 3. 解冻所有 LLM 层（如果不使用LoRA）
            if not use_lora:
                self.language_model.unfreeze_params()
                print("  ✓ 所有 LLM 层已解冻")
            else:
                print("  ✓ 使用LoRA，LoRA参数自动可训练")
        else:
            print(f"  警告: 未知的训练阶段 '{stage}'，使用默认参数冻结策略")
    
    def _print_trainable_params(self):
        """打印可训练参数信息"""
        total_params = 0
        trainable_params = 0
        
        # 统计所有参数
        for name, param in self.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        # 按模块统计
        print(f"\n可训练参数统计:")
        print(f"  总参数: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  可训练比例: {trainable_params/total_params*100:.2f}%")
        
        # 详细统计各模块
        print(f"\n各模块可训练参数:")
        module_stats = {}
        for name, param in self.named_parameters():
            if param.requires_grad:
                module_name = name.split('.')[0]
                if module_name not in module_stats:
                    module_stats[module_name] = 0
                module_stats[module_name] += param.numel()
        
        for module_name, count in sorted(module_stats.items(), key=lambda x: x[1], reverse=True):
            print(f"  {module_name}: {count:,} ({count/trainable_params*100:.2f}%)")
    
    @staticmethod
    def image2tensor(image, processor):
        """图像转tensor"""
        if image.mode in ['RGBA', 'LA']:
            image = image.convert('RGB')
        inputs = processor(images=image, return_tensors="pt")['pixel_values']
        return inputs
    
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取图像嵌入（使用视觉编码器的统一接口）
        
        Args:
            pixel_values: 图像tensor [batch_size, channels, height, width]
        
        Returns:
            图像嵌入 [batch_size, num_patches, hidden_size]
        """
        return self.vision_encoder.get_image_embeddings(pixel_values)
    
    def count_vision_proj(self, tokens: torch.Tensor, h: torch.Tensor, 
                         vision_tensors: Optional[torch.Tensor] = None,
                         seqlen: int = 512) -> torch.Tensor:
        """
        替换图像token为视觉特征（使用视觉特征处理器）
        
        Args:
            tokens: 输入token序列 [batch_size, seq_len]
            h: 文本embedding [batch_size, seq_len, hidden_dim]
            vision_tensors: 视觉特征 [batch_size, num_patches, vision_dim]
            seqlen: 序列最大长度
        
        Returns:
            处理后的hidden_states [batch_size, seq_len, hidden_dim]
        """
        if self.image_token_ids is None or vision_tensors is None:
            return h
        
        # 投影视觉特征 [batch_size, num_patches, language_dim]
        vision_proj = self.vision_proj(vision_tensors)
        
        # 使用视觉特征处理器处理
        return self.vision_processor.process_vision_features(
            tokens=tokens,
            hidden_states=h,
            vision_proj=vision_proj,
            image_token_ids=self.image_token_ids,
            seqlen=seqlen
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs
    ):
        batch_size, seq_length = input_ids.shape
        
        # 获取文本嵌入
        embedding_layer = self.language_model.get_embedding_layer()
        if embedding_layer is not None:
            hidden_states = embedding_layer(input_ids)
        else:
            # 回退方案：尝试直接访问
            if hasattr(self.language_model.model, 'model') and hasattr(self.language_model.model.model, 'embed_tokens'):
                hidden_states = self.language_model.model.model.embed_tokens(input_ids)
            elif hasattr(self.language_model.model, 'embed_tokens'):
                hidden_states = self.language_model.model.embed_tokens(input_ids)
            else:
                raise ValueError("无法找到embedding层，请检查模型结构")
        
        # 处理图像
        if pixel_values is not None and self.image_token_ids is not None:
            # 获取图像嵌入
            vision_tensors = self.get_image_embeddings(pixel_values)
            
            # 替换图像token为视觉特征
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=seq_length
            )
        
        # 语言模型前向传播
        if labels is not None:
            outputs = self.language_model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs
            )
        else:
            outputs = self.language_model(
                inputs_embeds=hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
        
        return outputs
    
    def get_tokenizer(self):
        """获取tokenizer"""
        return self.language_model.tokenizer
    
    def get_processor(self):
        """获取图像处理器"""
        return self.vision_encoder.processor
    
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ):
        """生成文本"""
        batch_size, seq_length = input_ids.shape
        
        # 获取文本嵌入
        embedding_layer = self.language_model.get_embedding_layer()
        if embedding_layer is not None:
            hidden_states = embedding_layer(input_ids)
        else:
            if hasattr(self.language_model.model, 'embed_tokens'):
                hidden_states = self.language_model.model.embed_tokens(input_ids)
            else:
                raise ValueError("无法找到embedding层")
        
        # 处理图像
        if pixel_values is not None and self.image_token_ids is not None:
            # 获取图像嵌入
            vision_tensors = self.get_image_embeddings(pixel_values)
            
            # 替换图像token为视觉特征
            hidden_states = self.count_vision_proj(
                tokens=input_ids,
                h=hidden_states,
                vision_tensors=vision_tensors,
                seqlen=seq_length
            )
        
        # 使用语言模型的generate方法
        # 注意：这里我们需要使用inputs_embeds，但transformers的generate可能不支持
        # 所以我们需要手动实现生成逻辑，或者使用language_model的generate
        # 为了简化，我们直接调用language_model的generate，但需要先处理图像
        
        # 由于generate方法通常需要input_ids，我们需要一个workaround
        # 这里我们使用language_model的generate，但图像信息已经在hidden_states中
        # 实际上，更好的方法是直接使用language_model.generate，但传入处理过的inputs_embeds
        
        # 临时方案：直接使用language_model的generate
        # 但这样会丢失图像信息，所以我们需要修改策略
        
        # 更好的方案：手动实现生成循环
        return self._generate_with_vision(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def _generate_with_vision(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 512,
        do_sample: bool = True,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs
    ):
        """带视觉信息的生成方法"""
        from torch.nn.functional import softmax
        
        self.eval()
        generated_ids = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播
                outputs = self.forward(
                    input_ids=generated_ids,
                    pixel_values=pixel_values,
                    attention_mask=attention_mask
                )
                
                # 获取下一个token的logits
                next_token_logits = outputs.logits[:, -1, :]
                
                # 应用采样策略
                if do_sample:
                    if temperature > 0:
                        next_token_logits = next_token_logits / temperature
                    
                    # Top-p采样
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(softmax(sorted_logits, dim=-1), dim=-1)
                        
                        # 移除累积概率超过top_p的tokens
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    probs = softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    # 贪婪解码
                    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                
                # 添加到生成的序列
                generated_ids = torch.cat([generated_ids, next_token], dim=1)
                
                # 更新attention mask
                if attention_mask is not None:
                    attention_mask = torch.cat([
                        attention_mask,
                        torch.ones((attention_mask.size(0), 1), device=attention_mask.device, dtype=attention_mask.dtype)
                    ], dim=1)
                else:
                    attention_mask = torch.ones_like(generated_ids)
                
                # 检查是否到达结束token
                tokenizer = self.get_tokenizer()
                eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                if next_token.item() == eos_token_id:
                    break
        
        return generated_ids