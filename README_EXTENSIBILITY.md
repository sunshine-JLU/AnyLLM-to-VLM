# 项目可扩展性指南

本项目采用**工厂模式**和**抽象基类**设计，支持轻松替换和扩展以下组件：

1. ✅ **视觉编码器** (Vision Encoder)
2. ✅ **语言模型** (Language Model)  
3. ✅ **投影层** (Projection Layer)

## 架构概览

```
model/
├── vision_encoders/      # 视觉编码器模块
│   ├── base.py           # 基类
│   ├── factory.py        # 工厂
│   └── clip_encoder.py   # CLIP实现
├── language_models/       # 语言模型模块
│   ├── base.py           # 基类
│   ├── factory.py        # 工厂
│   └── qwen_model.py     # Qwen实现
└── projections/          # 投影层模块
    ├── base.py           # 基类
    ├── factory.py        # 工厂
    └── mlp_projection.py # MLP实现
```

## 1. 视觉编码器扩展

### 当前支持
- ✅ CLIP

### 添加新编码器

参考: `README_VISION_ENCODERS.md`

```python
# model/vision_encoders/your_encoder.py
from .base import BaseVisionEncoder

class YourVisionEncoder(BaseVisionEncoder):
    def load_model(self):
        # 实现加载逻辑
        pass
    
    def get_image_embeddings(self, pixel_values):
        # 返回 [batch_size, num_patches, hidden_size]
        pass
```

## 2. 语言模型扩展

### 当前支持
- ✅ Qwen

### 添加新语言模型

**步骤1**: 创建模型类

```python
# model/language_models/llama_model.py
from .base import BaseLanguageModel
from transformers import LlamaForCausalLM, LlamaTokenizer

class LLaMALanguageModel(BaseLanguageModel):
    def load_model(self):
        return LlamaForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16 if self.use_bfloat16 else torch.float32
        )
    
    def load_tokenizer(self):
        return LlamaTokenizer.from_pretrained(self.model_path)
    
    def _get_hidden_size(self):
        return self.model.config.hidden_size
    
    def unfreeze_layers(self, num_layers):
        # 实现解冻逻辑
        pass
    
    def unfreeze_embedding_and_output(self):
        # 解冻embedding和输出层
        pass
```

**步骤2**: 注册模型

```python
# model/language_models/factory.py
from .llama_model import LLaMALanguageModel
LanguageModelFactory.register('llama', LLaMALanguageModel)
```

**步骤3**: 在配置中使用

```yaml
model:
  language_model_type: "llama"
  language_model_path: "path/to/llama"
```

## 3. 投影层扩展

### 当前支持
- ✅ MLP (多层感知机)
- ✅ Linear (简单线性层)

### 添加新投影层

**步骤1**: 创建投影层类

```python
# model/projections/cross_attention_projection.py
from .base import BaseProjection
import torch.nn as nn

class CrossAttentionProjection(BaseProjection):
    def build_projection(self, **kwargs):
        # 实现交叉注意力投影
        return nn.MultiheadAttention(
            embed_dim=self.vision_dim,
            num_heads=8
        )
    
    def forward(self, image_embeddings):
        # 实现前向传播
        pass
```

**步骤2**: 注册投影层

```python
# model/projections/factory.py
from .cross_attention_projection import CrossAttentionProjection
ProjectionFactory.register('cross_attention', CrossAttentionProjection)
```

**步骤3**: 在配置中使用

```yaml
model:
  projection_type: "cross_attention"
  projection_hidden_dim: 3072
  # ... 其他参数
```

## 配置示例

### 完整配置示例

```yaml
model:
  # 视觉编码器
  vision_model_type: "clip"  # 或 "vit", "dinov2"
  vision_model_path: "../models/clip-vit-base-patch16"
  freeze_vision_encoder: true
  vision_layers_to_unfreeze: 0
  
  # 语言模型
  language_model_type: "qwen"  # 或 "llama", "gpt"
  language_model_path: "../models/Qwen3-0.6B"
  freeze_language_model: false
  language_layers_to_unfreeze: 2
  
  # 投影层
  projection_type: "mlp"  # 或 "linear", "cross_attention"
  projection_hidden_dim: 3072
  projection_activation: "gelu"
  projection_dropout: 0.1
  projection_layernorm: true
```

## 查看已注册的组件

```python
from model.vision_encoders import VisionEncoderFactory
from model.language_models import LanguageModelFactory
from model.projections import ProjectionFactory

print("视觉编码器:", VisionEncoderFactory.list_encoders())
print("语言模型:", LanguageModelFactory.list_models())
print("投影层:", ProjectionFactory.list_projections())
```

## 组合使用示例

### 示例1: CLIP + Qwen + MLP
```yaml
vision_model_type: "clip"
language_model_type: "qwen"
projection_type: "mlp"
```

### 示例2: DINOv2 + LLaMA + CrossAttention
```yaml
vision_model_type: "dinov2"
language_model_type: "llama"
projection_type: "cross_attention"
```

## 优势

1. **解耦设计**: 各组件独立，互不影响
2. **统一接口**: 所有组件使用相同的接口规范
3. **易于测试**: 每个组件可独立测试
4. **配置驱动**: 通过配置文件即可切换组件
5. **向后兼容**: 现有代码无需修改

## 注意事项

1. **维度匹配**: 确保投影层的输入输出维度正确
2. **接口一致性**: 所有实现必须遵循基类接口
3. **错误处理**: 添加适当的错误处理和日志
4. **文档**: 为新组件添加使用文档

## 更多信息

- 视觉编码器: `README_VISION_ENCODERS.md`
- 快速开始: `QUICK_START_VISION_ENCODERS.md`
