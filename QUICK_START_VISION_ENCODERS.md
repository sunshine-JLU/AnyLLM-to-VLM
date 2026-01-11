# 视觉编码器快速使用指南

## 当前支持的编码器

- ✅ **CLIP** (已实现，默认)
- 📝 **ViT** (示例代码已提供，需要测试)

## 使用方法

### 1. 使用CLIP（默认）

在配置文件中：

```yaml
model:
  vision_model_type: "clip"
  vision_model_path: "../multimodal-vlm/models/clip-vit-base-patch16"
  freeze_vision_encoder: true
  vision_layers_to_unfreeze: 0
```

### 2. 切换到其他编码器

只需修改配置文件中的 `vision_model_type` 和 `vision_model_path`：

```yaml
model:
  vision_model_type: "vit"  # 改为你想要的编码器类型
  vision_model_path: "../multimodal-vlm/models/vit-base-patch16-224"
  freeze_vision_encoder: true
  vision_layers_to_unfreeze: 0
```

### 3. 添加新编码器（3步）

#### 步骤1: 创建编码器文件

创建 `model/vision_encoders/your_encoder.py`，继承 `BaseVisionEncoder`：

```python
from .base import BaseVisionEncoder

class YourVisionEncoder(BaseVisionEncoder):
    def load_model(self):
        # 加载你的模型
        pass
    
    def load_processor(self):
        # 加载处理器
        pass
    
    def get_image_embeddings(self, pixel_values):
        # 返回 [batch_size, num_patches, hidden_size]
        pass
    
    @property
    def hidden_size(self):
        # 返回隐藏层维度
        pass
```

#### 步骤2: 注册编码器

在 `model/vision_encoders/factory.py` 末尾添加：

```python
from .your_encoder import YourVisionEncoder
VisionEncoderFactory.register('your_encoder', YourVisionEncoder)
```

#### 步骤3: 在配置中使用

```yaml
model:
  vision_model_type: "your_encoder"
  vision_model_path: "path/to/your/model"
```

## 架构优势

1. **解耦设计**: 视觉编码器与主模型解耦，易于替换
2. **统一接口**: 所有编码器使用相同的接口，主模型代码无需修改
3. **易于扩展**: 只需实现基类方法，无需修改核心代码
4. **向后兼容**: 现有CLIP代码完全兼容

## 查看已注册的编码器

```python
from model.vision_encoders import VisionEncoderFactory

print(VisionEncoderFactory.list_encoders())
# 输出: ['clip']
```

## 常见编码器示例

### DINOv2

```python
# model/vision_encoders/dinov2_encoder.py
from transformers import Dinov2Model, AutoImageProcessor
from .base import BaseVisionEncoder

class DINOv2VisionEncoder(BaseVisionEncoder):
    def load_model(self):
        return Dinov2Model.from_pretrained(self.model_path)
    
    def load_processor(self):
        return AutoImageProcessor.from_pretrained(self.model_path)
    
    def get_image_embeddings(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 1:, :]  # 去掉CLS token
    
    @property
    def hidden_size(self):
        return self.model.config.hidden_size
```

### BLIP

```python
# model/vision_encoders/blip_encoder.py
from transformers import BlipModel, BlipProcessor
from .base import BaseVisionEncoder

class BLIPVisionEncoder(BaseVisionEncoder):
    def load_model(self):
        return BlipModel.from_pretrained(self.model_path)
    
    def load_processor(self):
        return BlipProcessor.from_pretrained(self.model_path)
    
    def get_image_embeddings(self, pixel_values):
        outputs = self.model.get_image_features(pixel_values=pixel_values)
        # 根据BLIP的实际输出格式调整
        return outputs
```

## 注意事项

1. **输出格式**: `get_image_embeddings` 必须返回 `[batch_size, num_patches, hidden_size]`
2. **CLS Token**: 大多数ViT类模型需要去掉第一个CLS token
3. **隐藏层维度**: 确保 `hidden_size` 正确，影响投影层维度
4. **处理器兼容**: 确保训练和推理使用相同的处理器

## 测试新编码器

```python
from model.vision_encoders import VisionEncoderFactory

# 创建并测试
encoder = VisionEncoderFactory.create(
    encoder_type='your_encoder',
    model_path='path/to/model',
    freeze=True
)

# 测试处理图像
from PIL import Image
image = Image.open('test.jpg').convert('RGB')
pixel_values = encoder.processor(image, return_tensors='pt')['pixel_values']
embeddings = encoder.get_image_embeddings(pixel_values)
print(f"嵌入形状: {embeddings.shape}")
```

## 更多信息

详细文档请参考: `README_VISION_ENCODERS.md`
