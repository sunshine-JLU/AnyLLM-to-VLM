# 视觉编码器扩展指南

本项目支持可扩展的视觉编码器架构，你可以轻松添加新的视觉编码器（如ViT、DINOv2、BLIP等）。

## 架构设计

项目使用**工厂模式**和**抽象基类**来实现可扩展的视觉编码器架构：

```
model/vision_encoders/
├── __init__.py          # 模块导出
├── base.py              # 抽象基类 BaseVisionEncoder
├── factory.py           # 工厂类 VisionEncoderFactory
├── clip_encoder.py      # CLIP编码器实现
└── vit_encoder.py       # ViT编码器示例（可选）
```

## 如何添加新的视觉编码器

### 步骤1: 创建编码器类

创建一个新文件，例如 `model/vision_encoders/dinov2_encoder.py`:

```python
import torch
import torch.nn as nn
from transformers import Dinov2Model, AutoImageProcessor
from .base import BaseVisionEncoder


class DINOv2VisionEncoder(BaseVisionEncoder):
    """DINOv2视觉编码器"""
    
    def __init__(self, model_path: str, freeze: bool = True, vision_layers_to_unfreeze: int = 0):
        super().__init__(model_path, freeze)
        self.vision_layers_to_unfreeze = vision_layers_to_unfreeze
        self.model = self.load_model()
        self._hidden_size = self._get_hidden_size()
        
        # 设置参数冻结
        if freeze:
            self.freeze_params()
            if vision_layers_to_unfreeze > 0:
                self.unfreeze_layers(vision_layers_to_unfreeze)
        else:
            self.unfreeze_params()
    
    def load_model(self) -> nn.Module:
        """加载DINOv2模型"""
        try:
            model = Dinov2Model.from_pretrained(self.model_path)
            print(f"  成功加载DINOv2模型: {self.model_path}")
            return model
        except Exception as e:
            print(f"加载DINOv2模型失败: {e}")
            raise
    
    def load_processor(self):
        """加载图像处理器"""
        try:
            processor = AutoImageProcessor.from_pretrained(self.model_path)
            return processor
        except Exception as e:
            print(f"加载处理器失败: {e}")
            raise
    
    def get_image_embeddings(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        获取图像嵌入
        
        Args:
            pixel_values: [batch_size, channels, height, width]
        
        Returns:
            [batch_size, num_patches, hidden_size]
        """
        with torch.no_grad() if self.freeze else torch.enable_grad():
            outputs = self.model(pixel_values=pixel_values)
        
        # DINOv2输出格式: [batch_size, num_patches+1, hidden_size]
        # 去掉CLS token
        img_embedding = outputs.last_hidden_state[:, 1:, :]
        return img_embedding
    
    def _get_hidden_size(self) -> int:
        """获取隐藏层维度"""
        if hasattr(self.model, 'config'):
            return self.model.config.hidden_size
        return 768  # 默认值
    
    @property
    def hidden_size(self) -> int:
        return self._hidden_size
    
    def unfreeze_layers(self, num_layers: int):
        """解冻后几层"""
        if num_layers <= 0:
            self.freeze_params()
            return
        
        self.freeze_params()
        
        # 根据DINOv2的实际结构解冻层
        if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
            layers = self.model.encoder.layer
            num_total_layers = len(layers)
            
            if num_layers > 0:
                start_layer = max(0, num_total_layers - num_layers)
                for i in range(start_layer, num_total_layers):
                    for param in layers[i].parameters():
                        param.requires_grad = True
                print(f"  DINOv2视觉编码器解冻后 {num_layers} 层")
```

### 步骤2: 注册编码器

在 `model/vision_encoders/factory.py` 中注册新编码器：

```python
from .dinov2_encoder import DINOv2VisionEncoder

# 在文件末尾注册
VisionEncoderFactory.register('dinov2', DINOv2VisionEncoder)
```

或者在运行时注册：

```python
from model.vision_encoders import VisionEncoderFactory
from model.vision_encoders.dinov2_encoder import DINOv2VisionEncoder

VisionEncoderFactory.register('dinov2', DINOv2VisionEncoder)
```

### 步骤3: 在配置文件中使用

在 `configs/vlm_pretrain.yaml` 中：

```yaml
model:
  vision_model_type: "dinov2"  # 使用新编码器
  vision_model_path: "./models/dinov2-base"
  freeze_vision_encoder: true
  vision_layers_to_unfreeze: 0
  # ... 其他配置
```

## 必须实现的方法

所有视觉编码器必须继承 `BaseVisionEncoder` 并实现以下方法：

1. **`load_model()`**: 加载模型
2. **`load_processor()`**: 加载图像处理器
3. **`get_image_embeddings(pixel_values)`**: 获取图像嵌入
   - 输入: `[batch_size, channels, height, width]`
   - 输出: `[batch_size, num_patches, hidden_size]`
4. **`hidden_size` 属性**: 返回隐藏层维度
5. **`unfreeze_layers(num_layers)`** (可选): 解冻后几层

## 示例：支持的编码器类型

### 1. CLIP (已实现)
```yaml
vision_model_type: "clip"
vision_model_path: "./models/clip-vit-base-patch16"
```

### 2. ViT (示例已提供)
```yaml
vision_model_type: "vit"
vision_model_path: "./models/vit-base-patch16-224"
```

### 3. DINOv2 (需要实现)
```yaml
vision_model_type: "dinov2"
vision_model_path: "./models/dinov2-base"
```

### 4. BLIP (需要实现)
```yaml
vision_model_type: "blip"
vision_model_path: "./models/blip-base"
```

## 注意事项

1. **输出格式统一**: `get_image_embeddings` 必须返回 `[batch_size, num_patches, hidden_size]` 格式
2. **去掉CLS token**: 大多数ViT类模型都有CLS token，需要去掉第一个token
3. **隐藏层维度**: 确保 `hidden_size` 属性正确，这会影响投影层的维度
4. **处理器兼容**: 确保图像处理器与训练时使用的处理器兼容

## 测试新编码器

添加新编码器后，可以这样测试：

```python
from model.vision_encoders import VisionEncoderFactory

# 创建编码器
encoder = VisionEncoderFactory.create(
    encoder_type='dinov2',
    model_path='./models/dinov2-base',
    freeze=True,
    vision_layers_to_unfreeze=0
)

# 测试
import torch
from PIL import Image

image = Image.open('test.jpg').convert('RGB')
pixel_values = encoder.processor(image, return_tensors='pt')['pixel_values']
embeddings = encoder.get_image_embeddings(pixel_values)
print(f"嵌入形状: {embeddings.shape}")  # 应该是 [1, num_patches, hidden_size]
```

## 常见问题

**Q: 如何知道模型的输出格式？**
A: 查看模型的文档或直接测试：
```python
outputs = model(pixel_values)
print(outputs.last_hidden_state.shape)  # 查看输出形状
```

**Q: 如何解冻特定层？**
A: 实现 `unfreeze_layers` 方法，根据模型的实际结构来解冻层。

**Q: 可以同时使用多个编码器吗？**
A: 目前架构只支持单个编码器，但可以通过修改架构来支持多编码器融合。

## 贡献

欢迎贡献新的视觉编码器实现！请确保：
1. 继承 `BaseVisionEncoder`
2. 实现所有必需的方法
3. 添加适当的错误处理
4. 更新文档
