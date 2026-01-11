# Qwen3 模型支持说明

## 文件命名

- **新文件**: `qwen3_model.py` - 推荐使用
- **旧文件**: `qwen_model.py` - 保留用于向后兼容（可选删除）

## 支持的模型

`Qwen3LanguageModel` 使用 `AutoModelForCausalLM`，可以自动适配所有 Qwen3 系列模型，包括：

- ✅ Qwen3-0.5B
- ✅ Qwen3-0.6B  
- ✅ Qwen3-1.5B
- ✅ Qwen3-2B
- ✅ Qwen3-7B
- ✅ 其他 Qwen3 变体

## 为什么一个文件可以支持所有尺寸？

1. **使用 AutoModelForCausalLM**: 
   - Transformers 库会自动识别模型架构
   - 不同尺寸的 Qwen3 模型使用相同的架构，只是参数数量不同

2. **自动获取配置**:
   - `hidden_size` 从 `model.config.hidden_size` 自动获取
   - 层数从 `model.model.layers` 自动获取
   - 不需要硬编码任何尺寸相关的参数

3. **统一的接口**:
   - 所有 Qwen3 模型都遵循相同的结构：
     - `model.model.layers` - transformer 层
     - `model.model.embed_tokens` - embedding 层
     - `model.lm_head` - 输出层

## 使用方法

### 在配置文件中

```yaml
model:
  language_model_type: "qwen3"  # 或 "qwen"（向后兼容）
  language_model_path: "../multimodal-vlm/models/Qwen3-0.6B"  # 可以是任何Qwen3模型路径
```

### 支持的配置值

- `"qwen3"` - 推荐使用，明确表示 Qwen3 系列
- `"qwen"` - 向后兼容，实际使用相同的实现

## 示例：切换不同尺寸的模型

只需要修改 `language_model_path`，无需修改代码：

```yaml
# 使用 0.6B 模型
language_model_path: "../multimodal-vlm/models/Qwen3-0.6B"

# 切换到 1.5B 模型
language_model_path: "../multimodal-vlm/models/Qwen3-1.5B"

# 切换到 7B 模型
language_model_path: "../multimodal-vlm/models/Qwen3-7B"
```

## 向后兼容性

为了保持向后兼容，代码中同时注册了两个名称：
- `'qwen3'` - 新的推荐名称
- `'qwen'` - 旧名称（仍然可用）

现有的配置文件使用 `"qwen"` 仍然可以正常工作。

## 技术细节

### 自动适配的机制

1. **模型加载**: `AutoModelForCausalLM.from_pretrained()` 自动识别模型类型
2. **配置读取**: 从 `model.config` 自动读取所有配置参数
3. **结构检测**: 通过 `hasattr()` 检查模型结构，适配不同变体

### 支持的模型结构

代码会自动检测并适配以下结构：
- 标准 Qwen3 结构: `model.model.layers`
- GPT-2 兼容结构: `model.transformer.h` (如果存在)
- 其他变体: 通过 `get_input_embeddings()` 等方法适配

## 注意事项

1. **模型路径**: 确保模型路径正确，包含完整的模型文件
2. **内存要求**: 不同尺寸的模型需要不同的 GPU 内存
3. **训练参数**: 不同尺寸的模型可能需要调整学习率等超参数

## 未来扩展

如果需要支持 Qwen2 或其他 Qwen 系列，可以：
1. 创建新的文件（如 `qwen2_model.py`）
2. 在工厂中注册新的类型
3. 保持相同的接口，确保兼容性
