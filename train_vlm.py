#!/usr/bin/env python3
import argparse
import yaml
import torch
import torch.distributed as dist
import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig, TrainingConfig
from trainer.vlm_trainer import train_vlm


def setup_distributed():
    """设置分布式训练环境"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        print(f"分布式训练初始化: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        return True, local_rank, world_size
    
    return False, 0, 1


def load_config(config_path: str) -> tuple:
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        # 创建默认配置
        return create_default_configs()
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 模型配置
    model_config_data = config_data.get('model', {})
    model_config = ModelConfig(**model_config_data)
    
    # 训练配置
    train_config_data = config_data.get('training', {})
    train_config = TrainingConfig(**train_config_data)
    
    return model_config, train_config


def create_default_configs():
    """创建默认配置"""
    print("使用默认配置...")
    
    # 默认模型配置
    model_config = ModelConfig(
        vision_model_path="./models/clip-vit-base-patch16",
        language_model_path="./models/Qwen3-0.6B",
        image_special_token='@' * 196
    )
    
    # 默认训练配置
    train_config = TrainingConfig(
        stage="pretrain",
        batch_size=32,
        num_epochs=4,
        learning_rate=4e-4,
        data_path="./data/pretrain_data.parquet",
        save_dir="./checkpoints/pretrain"
    )
    
    return model_config, train_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练多模态VLM模型")
    
    # 配置参数
    parser.add_argument('--config', type=str, default='configs/vlm_pretrain.yaml', 
                       help='配置文件路径')
    
    # 训练阶段
    parser.add_argument('--stage', type=str, choices=['pretrain', 'sft'], default=None,
                       help='训练阶段: pretrain 或 sft')
    
    # 数据参数
    parser.add_argument('--train_data', type=str, default=None,
                       help='训练数据路径 (parquet格式)')
    parser.add_argument('--val_data', type=str, default=None,
                       help='验证数据路径 (parquet格式)')
    
    # 模型参数
    parser.add_argument('--vision_model', type=str, default=None,
                       help='视觉模型路径')
    parser.add_argument('--language_model', type=str, default=None,
                       help='语言模型路径')
    parser.add_argument('--image_special_token', type=str, default=None,
                       help='图像特殊token')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=None,
                       help='学习率')
    parser.add_argument('--max_seq_len', type=int, default=None,
                       help='最大序列长度')
    
    # 保存和恢复
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存目录')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    
    # 实验跟踪
    parser.add_argument('--use_wandb', action='store_true', 
                       help='使用WandB记录实验')
    parser.add_argument('--wandb_project', type=str, default='VLM-Training',
                       help='WandB项目名称')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置分布式训练
    is_distributed, local_rank, world_size = setup_distributed()
    
    # 如果是分布式训练，只有主进程打印信息
    is_main_process = not is_distributed or local_rank == 0
    
    if is_main_process:
        print("=" * 50)
        print("多模态VLM模型训练")
        print("=" * 50)
        print(f"命令行参数: {args}")
    
    try:
        # 加载配置文件
        model_config, train_config = load_config(args.config)
        
        # 覆盖命令行参数（只有当命令行明确提供时才覆盖yaml配置）
        if args.stage is not None:
            train_config.stage = args.stage
        
        if args.train_data is not None:
            train_config.data_path = args.train_data
            if not os.path.exists(train_config.data_path):
                print(f"警告: 训练数据文件不存在: {train_config.data_path}")
                print(f"请确保已创建Parquet格式的数据集")
                return
        
        if args.val_data is not None:
            # 需要扩展TrainingConfig以支持val_data_path
            train_config.val_data_path = args.val_data
        
        if args.vision_model is not None:
            model_config.vision_model_path = args.vision_model
        
        if args.language_model is not None:
            model_config.language_model_path = args.language_model
        
        if args.image_special_token is not None:
            model_config.image_special_token = args.image_special_token
        
        if args.batch_size is not None:
            train_config.batch_size = args.batch_size
        
        if args.epochs is not None:
            train_config.num_epochs = args.epochs
        
        if args.lr is not None:
            train_config.learning_rate = args.lr
        
        if args.max_seq_len is not None:
            train_config.max_seq_length = args.max_seq_len
        
        if args.save_dir is not None:
            train_config.save_dir = args.save_dir
        
        # 分布式训练调整
        if is_distributed:
            train_config.use_ddp = True
            train_config.local_rank = local_rank
            train_config.world_size = world_size
            # 调整batch size
            train_config.batch_size = train_config.batch_size // world_size
        
        if is_main_process:
            print("\n配置信息:")
            print(f"  训练阶段: {train_config.stage}")
            print(f"  视觉模型: {model_config.vision_model_path}")
            print(f"  语言模型: {model_config.language_model_path}")
            print(f"  训练数据: {train_config.data_path}")
            print(f"  批次大小: {train_config.batch_size}")
            print(f"  训练轮数: {train_config.num_epochs}")
            print(f"  学习率: {train_config.learning_rate}")
            print(f"  保存目录: {train_config.save_dir}")
            print(f"  图像特殊token: {model_config.image_special_token[:20]}...")
            print("=" * 50)
        
        # 检查CUDA是否可用
        if torch.cuda.is_available():
            print(f"CUDA可用，GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("警告: CUDA不可用，将使用CPU训练（速度会很慢）")
        
        # 开始训练
        trainer = train_vlm(
            model_config=model_config,
            training_config=train_config,
            train_data_path=train_config.data_path,
            val_data_path=train_config.val_data_path if hasattr(train_config, 'val_data_path') else None,
            use_wandb=args.use_wandb,
            resume_from=args.resume
        )
        
        if is_main_process:
            print("\n训练完成!")
            print(f"模型保存在: {train_config.save_dir}")
            
    except Exception as e:
        print(f"训练过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()