import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
import warnings
import math

from transformers import CLIPModel, CLIPProcessor, AutoModelForCausalLM, AutoTokenizer

warnings.filterwarnings('ignore')


class VisionProj(nn.Module):
    """视觉投影层"""
    def __init__(self, vision_dim: int = 768, hidden_size: int = 512):
        super().__init__()
        self.vision_proj = nn.Sequential(
            nn.Linear(vision_dim, hidden_size),
            nn.LayerNorm(hidden_size)
        )
    
    def forward(self, image_embeddings: torch.Tensor) -> torch.Tensor:
        return self.vision_proj(image_embeddings)


class MultiModalVLM(nn.Module):
    """多模态视觉语言模型"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        print(f"从本地路径加载视觉模型: {config.vision_model_path}")
        # 视觉编码器
        self.vision_encoder, self.processor = self._load_vision_model(
            config.vision_model_path,
            freeze=config.freeze_vision_encoder
        )
        
        print(f"从本地路径加载语言模型: {config.language_model_path}")
        # 语言模型
        self.language_model = self._load_language_model(
            config.language_model_path,
            freeze=config.freeze_language_model
        )
        
        # 投影层
        vision_dim = 768  # CLIP base的hidden size
        if hasattr(self.vision_encoder.vision_model.config, 'hidden_size'):
            vision_dim = self.vision_encoder.vision_model.config.hidden_size
        
        language_dim = self.language_model.config.hidden_size
        
        self.vision_proj = VisionProj(
            vision_dim=vision_dim,
            hidden_size=language_dim
        )
        # 投影层始终可训练
        for param in self.vision_proj.parameters():
            param.requires_grad = True
        
        # 图像特殊token
        self.image_special_token = config.image_special_token
        self.image_token_ids = None  # 初始化为None，将在_add_image_special_token中设置
        
        # 添加图像特殊token到tokenizer
        if self.image_special_token:
            self._add_image_special_token()
        
        # 打印可训练参数信息
        self._print_trainable_params()
        
        print(f"模型初始化完成")
        print(f"  视觉模型维度: {vision_dim}")
        print(f"  语言模型维度: {language_dim}")
        print(f"  图像特殊token: {self.image_special_token[:20]}...")
    
    def _load_vision_model(self, model_path: str, freeze: bool = True):
        """加载视觉模型"""
        try:
            model = CLIPModel.from_pretrained(model_path)
            processor = CLIPProcessor.from_pretrained(model_path, use_fast=False)
            
            if freeze:
                for param in model.parameters():
                    param.requires_grad = False
                print(f"  视觉模型已冻结")
            else:
                print(f"  视觉模型可训练")
            
            return model, processor
        except Exception as e:
            print(f"加载视觉模型失败: {e}")
            raise
    
    def _load_language_model(self, model_path: str, freeze: bool = False):
        """加载语言模型"""
        try:
            # 根据配置选择数据类型
            dtype = torch.bfloat16 if self.config.use_bfloat16 else torch.float32
            
            # 尝试使用正确的参数名
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                dtype=dtype,  # 使用dtype而不是torch_dtype
                low_cpu_mem_usage=True
            )
            
            # 根据配置设置可训练参数
            if freeze:
                # 完全冻结
                for param in model.parameters():
                    param.requires_grad = False
                print(f"  语言模型已完全冻结")
            else:
                # 先冻结所有参数
                for param in model.parameters():
                    param.requires_grad = False
                
                # 根据language_layers_to_unfreeze配置解冻后几层
                layers_to_unfreeze = self.config.language_layers_to_unfreeze
                
                # 获取transformer层（不同模型可能有不同的结构）
                if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                    # Qwen等模型结构
                    layers = model.model.layers
                    num_layers = len(layers)
                    
                    if layers_to_unfreeze > 0:
                        # 解冻后几层
                        start_layer = max(0, num_layers - layers_to_unfreeze)
                        for i in range(start_layer, num_layers):
                            for param in layers[i].parameters():
                                param.requires_grad = True
                        print(f"  语言模型解冻后 {layers_to_unfreeze} 层 (层 {start_layer} 到 {num_layers-1})")
                    else:
                        print(f"  语言模型所有层已冻结")
                    
                    # 解冻embedding层（如果添加了新token，需要训练）
                    if hasattr(model.model, 'embed_tokens'):
                        for param in model.model.embed_tokens.parameters():
                            param.requires_grad = True
                        print(f"  语言模型embedding层已解冻")
                    
                    # 解冻输出层（lm_head）
                    if hasattr(model, 'lm_head'):
                        for param in model.lm_head.parameters():
                            param.requires_grad = True
                        print(f"  语言模型输出层(lm_head)已解冻")
                    
                elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                    # GPT-2等模型结构
                    layers = model.transformer.h
                    num_layers = len(layers)
                    
                    if layers_to_unfreeze > 0:
                        start_layer = max(0, num_layers - layers_to_unfreeze)
                        for i in range(start_layer, num_layers):
                            for param in layers[i].parameters():
                                param.requires_grad = True
                        print(f"  语言模型解冻后 {layers_to_unfreeze} 层 (层 {start_layer} 到 {num_layers-1})")
                    else:
                        print(f"  语言模型所有层已冻结")
                    
                    # 解冻embedding和输出层
                    if hasattr(model.transformer, 'wte'):
                        for param in model.transformer.wte.parameters():
                            param.requires_grad = True
                    if hasattr(model, 'lm_head'):
                        for param in model.lm_head.parameters():
                            param.requires_grad = True
                            
                else:
                    # 未知结构，尝试找到embedding和输出层
                    # 尝试解冻embedding层
                    if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
                        for param in model.model.embed_tokens.parameters():
                            param.requires_grad = True
                        print(f"  语言模型embedding层已解冻")
                    elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
                        for param in model.transformer.wte.parameters():
                            param.requires_grad = True
                        print(f"  语言模型embedding层已解冻")
                    
                    # 尝试解冻输出层
                    if hasattr(model, 'lm_head'):
                        for param in model.lm_head.parameters():
                            param.requires_grad = True
                        print(f"  语言模型输出层(lm_head)已解冻")
                    
                    # 如果layers_to_unfreeze > 0，尝试解冻所有参数
                    if layers_to_unfreeze > 0:
                        for param in model.parameters():
                            param.requires_grad = True
                        print(f"  警告: 未知模型结构，解冻所有参数")
                    else:
                        print(f"  语言模型transformer层已冻结（未知结构）")
            
            return model
        except Exception as e:
            print(f"加载语言模型失败: {e}")
            # 尝试不同的加载方式
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                # 应用相同的冻结逻辑
                if freeze:
                    for param in model.parameters():
                        param.requires_grad = False
                else:
                    for param in model.parameters():
                        param.requires_grad = False
                    layers_to_unfreeze = self.config.language_layers_to_unfreeze
                    if layers_to_unfreeze > 0:
                        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
                            layers = model.model.layers
                            num_layers = len(layers)
                            start_layer = max(0, num_layers - layers_to_unfreeze)
                            for i in range(start_layer, num_layers):
                                for param in layers[i].parameters():
                                    param.requires_grad = True
                return model
            except Exception as e2:
                print(f"再次尝试加载失败: {e2}")
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
            self.language_model.resize_token_embeddings(len(tokenizer))
            
            # 确保embedding层可训练（如果添加了新token，需要训练新的embedding）
            if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'embed_tokens'):
                for param in self.language_model.model.embed_tokens.parameters():
                    param.requires_grad = True
            elif hasattr(self.language_model, 'transformer') and hasattr(self.language_model.transformer, 'wte'):
                for param in self.language_model.transformer.wte.parameters():
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
        """获取图像嵌入"""
        with torch.no_grad():
            outputs = self.vision_encoder.vision_model(pixel_values=pixel_values)
        
        # 使用patch embeddings（去掉CLS token）
        img_embedding = outputs.last_hidden_state[:, 1:, :]  # [batch_size, 196, 768]
        return img_embedding
    
    def count_vision_proj(self, tokens: torch.Tensor, h: torch.Tensor, 
                         vision_tensors: Optional[torch.Tensor] = None,
                         seqlen: int = 512) -> torch.Tensor:
        """替换图像token为视觉特征"""
        if self.image_token_ids is None or vision_tensors is None:
            return h
        
        def find_indices(tokens_tensor, image_ids):
            """找到图像token的位置"""
            batch_size, seq_len = tokens_tensor.shape
            len_image_ids = len(image_ids)
            
            if len_image_ids > seq_len:
                return None
            
            # 将图像id转换为tensor
            image_ids_tensor = torch.tensor(image_ids, device=tokens_tensor.device)
            
            # 查找匹配位置
            matches = []
            for b in range(batch_size):
                batch_matches = []
                for i in range(seq_len - len_image_ids + 1):
                    if torch.all(tokens_tensor[b, i:i+len_image_ids] == image_ids_tensor):
                        batch_matches.append((i, i + len_image_ids - 1))
                matches.append(batch_matches)
            
            return matches
        
        # 找到图像token的位置
        image_indices = find_indices(tokens, self.image_token_ids)
        if not image_indices or all(len(indices) == 0 for indices in image_indices):
            return h
        
        # 投影视觉特征
        vision_proj = self.vision_proj(vision_tensors)
        
        # 替换图像token
        batch_size = h.size(0)
        new_h = []
        
        for i in range(batch_size):
            if image_indices[i]:
                h_i = h[i]
                img_idx = 0
                
                for start_idx, end_idx in image_indices[i]:
                    if img_idx < vision_proj.size(1):
                        # 替换图像token为视觉特征
                        vision_feat = vision_proj[i, img_idx]  # [hidden_size]
                        h_i = torch.cat([
                            h_i[:start_idx],
                            vision_feat.unsqueeze(0),
                            h_i[end_idx + 1:]
                        ], dim=0)[:seqlen]
                        img_idx += 1
                
                new_h.append(h_i)
            else:
                new_h.append(h[i])
        
        return torch.stack(new_h, dim=0)
    
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
        hidden_states = self.language_model.model.embed_tokens(input_ids)
        
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
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.language_model_path,
                trust_remote_code=True
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            return tokenizer
        except Exception as e:
            print(f"加载tokenizer失败: {e}")
            raise
    
    def get_processor(self):
        """获取图像处理器"""
        return self.processor