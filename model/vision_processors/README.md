# 视觉特征处理模块

本模块提供了可扩展的视觉特征处理方式，支持不同的视觉特征插入策略。

## 支持的处理方式

### 1. Patch插入风格 (`patch_insert`) - 默认

直接将所有patch特征插入到序列中，替换图像token。

**特点**：
- 保留所有patch的空间信息
- 如果图像token数量少于patch数量，则截断patch
- 如果图像token数量多于patch数量，则重复最后一个patch

**适用场景**：
- 需要保留完整空间信息的任务
- 图像理解任务
- 需要详细空间理解的任务

### 2. 平均Pooling风格 (`mean_pooling`)

使用平均pooling将所有patch特征聚合成单个特征，然后替换图像token。

**特点**：
- 序列长度保持不变
- 所有patch信息被平均聚合
- 计算效率更高

**适用场景**：
- 序列长度受限的场景
- 不需要详细空间信息的任务
- 快速原型开发

## 使用方法

### 在配置文件中

```yaml
model:
  vision_processor_type: "patch_insert"  # 或 "mean_pooling"
```

### 默认值

如果不指定 `vision_processor_type`，默认使用 `"patch_insert"`。

## 扩展新的处理方式

### 步骤1：创建新的处理器类

```python
# model/vision_processors/your_processor.py
import torch
from typing import List
from .base import BaseVisionProcessor

class YourVisionProcessor(BaseVisionProcessor):
    """自定义视觉特征处理器"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 初始化参数
    
    def process_vision_features(
        self,
        tokens: torch.Tensor,
        hidden_states: torch.Tensor,
        vision_proj: torch.Tensor,
        image_token_ids: List[int],
        seqlen: int = 512
    ) -> torch.Tensor:
        """
        实现你的处理逻辑
        
        Args:
            tokens: [batch_size, seq_len]
            hidden_states: [batch_size, seq_len, hidden_dim]
            vision_proj: [batch_size, num_patches, hidden_dim]
            image_token_ids: 图像特殊token的ID列表
            seqlen: 序列最大长度
        
        Returns:
            处理后的hidden_states [batch_size, seq_len, hidden_dim]
        """
        # 找到图像token位置
        image_indices = self.find_image_token_indices(tokens, image_token_ids)
        
        # 实现你的处理逻辑
        # ...
        
        return hidden_states
```

### 步骤2：注册处理器

```python
# model/vision_processors/__init__.py
from .your_processor import YourVisionProcessor

VisionProcessorFactory.register('your_processor', YourVisionProcessor)
```

### 步骤3：在配置中使用

```yaml
model:
  vision_processor_type: "your_processor"
```

## 技术细节

### 输入输出格式

- **输入**：
  - `tokens`: `[batch_size, seq_len]` - 输入token序列
  - `hidden_states`: `[batch_size, seq_len, hidden_dim]` - 文本embedding
  - `vision_proj`: `[batch_size, num_patches, hidden_dim]` - 投影后的视觉特征
  - `image_token_ids`: `List[int]` - 图像特殊token的ID列表

- **输出**：
  - `hidden_states`: `[batch_size, seq_len, hidden_dim]` - 处理后的embedding

### 注意事项

1. **序列长度一致性**：确保输出序列长度与输入一致（或通过填充/截断保持一致）
2. **数据类型一致性**：确保视觉特征与文本embedding的数据类型一致
3. **设备一致性**：确保所有tensor在同一设备上
4. **索引处理**：从后往前处理图像token位置，避免索引变化

## 查看已注册的处理器

```python
from model.vision_processors import VisionProcessorFactory

print("可用的处理器:", VisionProcessorFactory.list_processors())
```
