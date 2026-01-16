# 多卡训练指南

本指南介绍如何使用多张GPU进行分布式训练。

## 前置要求

1. **硬件要求**
   - 至少2张NVIDIA GPU
   - 支持NCCL通信（同一台机器上的多卡）

2. **软件要求**
   - PyTorch >= 1.9.0（支持分布式训练）
   - CUDA >= 11.0
   - NCCL（通常随PyTorch一起安装）

## 快速开始

### 方法1: 使用torchrun（推荐）

#### Linux/Mac

```bash
# 使用4张GPU训练
torchrun --nproc_per_node=4 train_vlm.py \
    --config configs/vlm_pretrain.yaml \
    --stage pretrain
```

#### Windows

```powershell
# 使用4张GPU训练
torchrun --nproc_per_node=4 train_vlm.py --config configs\vlm_pretrain.yaml --stage pretrain
```

### 方法2: 使用启动脚本

#### Linux/Mac

```bash
# 使用提供的脚本
bash scripts/train_multi_gpu.sh configs/vlm_pretrain.yaml pretrain
```

#### Windows

```cmd
# 使用提供的脚本
scripts\train_multi_gpu.bat configs\vlm_pretrain.yaml pretrain
```

### 方法3: 手动设置环境变量

#### Linux/Mac

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500

python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vlm.py \
    --config configs/vlm_pretrain.yaml \
    --stage pretrain
```

#### Windows

```cmd
set CUDA_VISIBLE_DEVICES=0,1,2,3
set MASTER_ADDR=localhost
set MASTER_PORT=29500

python -m torch.distributed.launch --nproc_per_node=4 --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% train_vlm.py --config configs\vlm_pretrain.yaml --stage pretrain
```

## 配置说明

### 1. 配置文件设置

在配置文件中，确保 `use_ddp` 设置为 `true`（虽然代码会自动检测并设置）：

```yaml
training:
  use_ddp: true  # 分布式训练会自动设置，但可以显式指定
  batch_size: 32  # 总batch size，会自动分配到各GPU
```

**注意**: 
- `batch_size` 是**总batch size**，会自动除以GPU数量
- 例如：如果设置 `batch_size=32`，使用4张GPU，每张GPU的batch size为8

### 2. 环境变量说明

- `CUDA_VISIBLE_DEVICES`: 指定使用的GPU（例如：`0,1,2,3`）
- `MASTER_ADDR`: 主节点地址（单机多卡通常为 `localhost`）
- `MASTER_PORT`: 主节点端口（默认29500，如果被占用可以修改）

### 3. 检查GPU

```bash
# 查看GPU信息
nvidia-smi

# 在Python中检查
python -c "import torch; print('GPU数量:', torch.cuda.device_count())"
```

## 训练参数调整

### Batch Size

多卡训练时，总batch size = 单卡batch size × GPU数量

**建议**:
- 预训练阶段：总batch size = 32-64
- SFT阶段：总batch size = 8-16

### 学习率

多卡训练时，通常需要调整学习率：

- **线性缩放规则**: `lr = base_lr × num_gpus`
- **平方根缩放规则**: `lr = base_lr × sqrt(num_gpus)`

**建议**: 从线性缩放开始，根据训练效果调整。

### 梯度累积

如果显存不足，可以使用梯度累积：

```yaml
training:
  batch_size: 8  # 每张GPU的batch size
  accumulation_steps: 4  # 累积4步，等效batch size = 8 × 4 × num_gpus
```

## 常见问题

### 1. NCCL错误

**问题**: `NCCL error: unhandled system error`

**解决方案**:
- 确保所有GPU在同一台机器上
- 检查NCCL版本：`python -c "import torch; print(torch.cuda.nccl.version())"`
- 尝试设置环境变量：`export NCCL_DEBUG=INFO`

### 2. 端口被占用

**问题**: `Address already in use`

**解决方案**:
- 修改 `MASTER_PORT` 为其他端口（如29501, 29502等）
- 或者关闭占用端口的进程

### 3. 显存不足

**问题**: `CUDA out of memory`

**解决方案**:
- 减小 `batch_size`
- 增加 `accumulation_steps`
- 使用梯度检查点（如果模型支持）
- 减少 `num_workers`

### 4. 训练速度没有提升

**问题**: 多卡训练速度没有明显提升

**可能原因**:
- 数据加载成为瓶颈（增加 `num_workers`）
- 模型太小，通信开销大于计算收益
- 检查是否有GPU空闲：`nvidia-smi`

## 性能优化建议

1. **数据加载**
   - 增加 `num_workers`（建议为GPU数量的2-4倍）
   - 使用 `pin_memory=True`
   - 使用 `prefetch_factor=2`

2. **通信优化**
   - 使用 `find_unused_parameters=False`（如果所有参数都参与训练）
   - 使用混合精度训练（`use_mixed_precision: true`）

3. **内存优化**
   - 使用梯度累积
   - 使用 `set_to_none=True` 清零梯度
   - 定期清理缓存：`torch.cuda.empty_cache()`

## 监控训练

### 查看GPU使用情况

```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者使用nvtop（如果已安装）
nvtop
```

### 查看训练日志

训练日志会显示：
- 每个进程的rank和local_rank
- 分布式训练初始化信息
- 每张GPU的batch size

## 示例：4卡训练完整命令

```bash
# 1. 设置环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 2. 启动训练
torchrun --nproc_per_node=4 \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vlm.py \
    --config configs/vlm_pretrain.yaml \
    --stage pretrain \
    --batch_size 32 \
    --lr 1.6e-3  # 4倍学习率（线性缩放）
```

## 注意事项

1. **检查点保存**: 只有主进程（rank 0）会保存检查点
2. **日志输出**: 只有主进程会打印日志，避免重复输出
3. **WandB**: 只有主进程会记录到WandB
4. **数据采样**: 使用 `DistributedSampler` 确保每个GPU看到不同的数据

## 更多资源

- [PyTorch分布式训练文档](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [NCCL文档](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
