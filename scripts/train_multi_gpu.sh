#!/bin/bash
# 多卡训练启动脚本
# 使用方法: bash scripts/train_multi_gpu.sh [配置文件路径] [训练阶段]

# 设置默认值
CONFIG=${1:-"configs/vlm_pretrain.yaml"}
STAGE=${2:-"pretrain"}

# 检查CUDA是否可用
if ! command -v nvidia-smi &> /dev/null; then
    echo "错误: 未检测到NVIDIA GPU，请确保已安装CUDA和nvidia-smi"
    exit 1
fi

# 获取GPU数量
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
echo "检测到 $NUM_GPUS 个GPU"

# 检查是否有足够的GPU
if [ $NUM_GPUS -lt 2 ]; then
    echo "警告: 检测到少于2个GPU，建议使用单卡训练"
    echo "单卡训练命令: python train_vlm.py --config $CONFIG --stage $STAGE"
    read -p "是否继续多卡训练? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 设置分布式训练环境变量
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 根据实际GPU数量调整
export MASTER_ADDR=localhost
export MASTER_PORT=29500

# 使用torchrun启动多卡训练
echo "启动多卡训练..."
echo "配置文件: $CONFIG"
echo "训练阶段: $STAGE"
echo "GPU数量: $NUM_GPUS"

torchrun --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_vlm.py \
    --config $CONFIG \
    --stage $STAGE

echo "训练完成!"
