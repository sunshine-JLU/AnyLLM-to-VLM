# 图像识别问题修复说明

## 问题描述
训练出来的模型无法识别图像（如大熊猫），可能的原因包括：
1. 图像token替换逻辑问题：只使用了第一个patch的特征
2. 投影层缺少梯度：图像token位置没有计算loss

## 修复内容

### 1. 修复图像token替换逻辑 (`model/vlm_model.py`)

**问题**：
- 之前只用一个patch的特征 (`vision_proj[i, img_idx]`) 替换图像token
- 丢失了大部分视觉信息（196个patch中只用了1个）

**修复**：
- 使用所有patch的平均pooled特征替换图像token
- 保持序列长度不变，同时利用所有patch的信息
- 代码位置：`model/vlm_model.py` 的 `count_vision_proj` 方法

```python
# 修复前：只用一个patch
vision_feat = vision_proj[i, img_idx]  # [hidden_size]

# 修复后：使用所有patch的平均pooled特征
vision_features = vision_proj[i]  # [num_patches, language_dim]
vision_feat_pooled = vision_features.mean(dim=0)  # [language_dim]
```

### 2. 修复损失mask生成 (`data/parquet_dataset.py`)

**问题**：
- 之前只在assistant回答部分计算loss
- 如果图像token在user问题中，投影层可能没有梯度

**修复**：
- 在图像token位置也设置loss mask为1
- 确保投影层能收到梯度，可以正常训练
- 代码位置：`data/parquet_dataset.py` 的 `_generate_loss_mask` 方法

## 训练建议

1. **重新训练**：
   - 建议从预训练阶段重新开始训练
   - 如果已经训练过，可以从预训练检查点继续

2. **检查训练数据**：
   - 确保数据中包含图像和对应的描述
   - 图像token (`<image>`) 应该正确替换为特殊token

3. **监控训练指标**：
   - 观察loss是否正常下降
   - 如果loss不下降，检查：
     - 图像是否正确加载
     - 图像token是否正确识别和替换
     - 投影层参数是否可训练

4. **验证修复**：
   - 训练几个epoch后，使用 `eval_vlm.py` 测试模型
   - 尝试识别简单的图像（如大熊猫、猫、狗等）

## 调试建议

如果修复后仍然无法识别图像，可以：

1. **检查图像特征是否正确注入**：
   ```python
   # 在 model/vlm_model.py 的 count_vision_proj 方法中添加打印
   print(f"图像token位置: {image_indices}")
   print(f"视觉特征shape: {vision_proj.shape}")
   print(f"Pooled特征norm: {vision_feat_pooled.norm()}")
   ```

2. **检查loss mask**：
   ```python
   # 在 data/parquet_dataset.py 的 __getitem__ 方法中添加打印
   print(f"Loss mask中图像token位置: {loss_mask[image_token_pos]}")
   ```

3. **检查投影层梯度**：
   ```python
   # 在训练循环中检查投影层参数是否有梯度
   for name, param in model.vision_proj.named_parameters():
       if param.grad is not None:
           print(f"{name} 梯度norm: {param.grad.norm()}")
   ```

## 其他可能的问题

1. **预训练不充分**：
   - 如果只做了SFT，建议先做预训练
   - 预训练阶段应该训练投影层和部分语言模型层

2. **学习率设置**：
   - 投影层可能需要较大的学习率
   - 建议检查投影层的学习率是否合适

3. **数据质量问题**：
   - 确保训练数据中图像和文本对应正确
   - 图像质量要足够好
