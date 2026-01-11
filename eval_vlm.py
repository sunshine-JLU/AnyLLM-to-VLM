#!/usr/bin/env python3
"""
VLM模型评测脚本
支持多种评测方式：
1. 计算验证集上的loss
2. 生成式评测（给定图像和问题，生成回答）
3. 批量评测
"""

import argparse
import yaml
import torch
import os
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import json

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import ModelConfig, TrainingConfig
from model.vlm_model import MultiModalVLM
from data.parquet_dataset import ParquetMultiModalDataset, DataCollator
from torch.utils.data import DataLoader


def load_model_from_checkpoint(checkpoint_path: str, model_config: ModelConfig, device: str = "cuda"):
    """从检查点加载模型"""
    print(f"加载检查点: {checkpoint_path}")
    
    # 创建模型
    model = MultiModalVLM(model_config)
    model.to(device)
    
    # 加载检查点
    # 使用 weights_only=False 以支持包含自定义类的检查点（PyTorch 2.6+ 默认需要）
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # 加载模型状态
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f"从检查点加载模型状态 (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        # 如果没有model_state_dict，尝试直接加载
        model.load_state_dict(checkpoint, strict=False)
        print("直接加载检查点状态")
    
    model.eval()
    return model


def evaluate_loss(model, val_dataset, batch_size=8, device="cuda"):
    """计算验证集上的loss"""
    print(f"\n开始计算验证集loss...")
    print(f"验证集大小: {len(val_dataset)}")
    
    tokenizer = model.get_tokenizer()
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    collator = DataCollator(pad_token_id=pad_token_id)
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True
    )
    
    model.eval()
    total_loss = 0
    total_samples = 0
    total_tokens = 0
    
    import torch.nn as nn
    loss_fct = nn.CrossEntropyLoss(reduction='none')
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            try:
                X, Y, loss_mask, pixel_values = batch
                
                X = X.to(device, non_blocking=True)
                Y = Y.to(device, non_blocking=True)
                loss_mask = loss_mask.to(device, non_blocking=True)
                pixel_values = pixel_values.to(device, non_blocking=True)
                
                # 前向传播
                outputs = model(
                    input_ids=X,
                    pixel_values=pixel_values,
                    labels=Y
                )
                
                # 计算masked loss
                logits = outputs.logits
                loss_per_token = loss_fct(
                    logits.view(-1, logits.size(-1)),
                    Y.view(-1)
                ).view(Y.size())
                
                mask_sum = loss_mask.sum()
                if mask_sum > 0:
                    loss = (loss_per_token * loss_mask).sum() / mask_sum
                    total_tokens += mask_sum.item()
                else:
                    loss = loss_per_token.mean()
                
                total_loss += loss.item() * X.size(0)
                total_samples += X.size(0)
                
            except Exception as e:
                print(f"评估batch时出错: {e}")
                continue
    
    avg_loss = total_loss / max(1, total_samples)
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"\n评估结果:")
    print(f"  平均Loss: {avg_loss:.4f}")
    print(f"  平均Perplexity: {avg_perplexity:.4f}")
    print(f"  总样本数: {total_samples}")
    print(f"  总token数: {total_tokens}")
    
    return {
        'loss': avg_loss,
        'perplexity': avg_perplexity,
        'total_samples': total_samples,
        'total_tokens': total_tokens
    }


def generate_response(model, image_path: str, question: str, max_new_tokens=512, device="cuda"):
    """生成式评测：给定图像和问题，生成回答"""
    print(f"\n生成式评测:")
    print(f"  图像: {image_path}")
    print(f"  问题: {question}")
    
    # 加载图像
    image = Image.open(image_path).convert('RGB')
    processor = model.get_processor()
    tokenizer = model.get_tokenizer()
    
    # 处理图像
    pixel_values = processor(images=image, return_tensors='pt')['pixel_values'].to(device)
    
    # 构建prompt
    image_special_token = model.config.image_special_token
    prompt = f"<|im_start|>user\n{question}\n{image_special_token}<|im_end|>\n<|im_start|>assistant\n"
    
    # Tokenize
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    # 生成
    model.eval()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # 解码
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
    
    # 提取回答部分
    if "<|im_start|>assistant" in generated_text:
        answer = generated_text.split("<|im_start|>assistant")[-1].split("<|im_end|>")[0].strip()
    else:
        answer = generated_text[len(prompt):].strip()
    
    print(f"  回答: {answer}")
    return answer


def batch_generate_eval(model, eval_data_path: str, output_path: str, max_new_tokens=512, device="cuda"):
    """批量生成式评测"""
    print(f"\n批量生成式评测...")
    
    # 读取评测数据（JSONL格式）
    results = []
    with open(eval_data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            image_path = data.get('image', '')
            question = data.get('question', '')
            ground_truth = data.get('answer', '')
            
            try:
                answer = generate_response(model, image_path, question, max_new_tokens, device)
                results.append({
                    'image': image_path,
                    'question': question,
                    'ground_truth': ground_truth,
                    'generated': answer
                })
            except Exception as e:
                print(f"处理样本时出错: {e}")
                continue
    
    # 保存结果
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n批量评测完成，结果保存到: {output_path}")
    print(f"总共评测了 {len(results)} 个样本")
    
    return results


def load_config(config_path: str) -> tuple:
    """加载配置文件"""
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return None, None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config_data = yaml.safe_load(f)
    
    # 模型配置
    model_config_data = config_data.get('model', {})
    model_config = ModelConfig(**model_config_data)
    
    # 训练配置
    train_config_data = config_data.get('training', {})
    train_config = TrainingConfig(**train_config_data)
    
    return model_config, train_config


def main():
    parser = argparse.ArgumentParser(description="评测VLM模型")
    
    # 必需参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--config', type=str, required=True,
                       help='配置文件路径（用于加载模型配置）')
    
    # 评测方式
    parser.add_argument('--mode', type=str, choices=['loss', 'generate', 'batch'],
                       default='loss', help='评测模式: loss(计算loss) | generate(单样本生成) | batch(批量生成)')
    
    # Loss评测参数
    parser.add_argument('--val_data', type=str, default=None,
                       help='验证数据路径（用于loss评测）')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='批次大小')
    
    # 生成式评测参数
    parser.add_argument('--image', type=str, default=None,
                       help='图像路径（用于单样本生成）')
    parser.add_argument('--question', type=str, default=None,
                       help='问题（用于单样本生成）')
    parser.add_argument('--eval_data', type=str, default=None,
                       help='评测数据路径（JSONL格式，用于批量生成）')
    parser.add_argument('--output', type=str, default='eval_results.jsonl',
                       help='输出文件路径（用于批量生成）')
    parser.add_argument('--max_new_tokens', type=int, default=512,
                       help='最大生成token数')
    
    # 设备
    parser.add_argument('--device', type=str, default='cuda',
                       help='设备 (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 检查设备
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("警告: CUDA不可用，使用CPU")
        args.device = 'cpu'
    
    # 加载配置
    print("加载配置文件...")
    model_config, train_config = load_config(args.config)
    if model_config is None:
        print("加载配置失败")
        return
    
    # 加载模型
    print("加载模型...")
    model = load_model_from_checkpoint(args.checkpoint, model_config, args.device)
    
    # 根据模式执行评测
    if args.mode == 'loss':
        if args.val_data is None:
            print("错误: loss模式需要提供--val_data参数")
            return
        
        if not os.path.exists(args.val_data):
            print(f"错误: 验证数据文件不存在: {args.val_data}")
            return
        
        # 创建验证数据集
        tokenizer = model.get_tokenizer()
        processor = model.get_processor()
        
        val_dataset = ParquetMultiModalDataset(
            parquet_path=args.val_data,
            tokenizer=tokenizer,
            image_processor=processor,
            stage=train_config.stage,
            max_length=train_config.max_seq_length,
            image_special_token=model_config.image_special_token
        )
        
        # 计算loss
        results = evaluate_loss(model, val_dataset, args.batch_size, args.device)
        print(f"\n评测完成！")
        
    elif args.mode == 'generate':
        if args.image is None or args.question is None:
            print("错误: generate模式需要提供--image和--question参数")
            return
        
        if not os.path.exists(args.image):
            print(f"错误: 图像文件不存在: {args.image}")
            return
        
        # 单样本生成
        answer = generate_response(model, args.image, args.question, args.max_new_tokens, args.device)
        print(f"\n生成完成！")
        
    elif args.mode == 'batch':
        if args.eval_data is None:
            print("错误: batch模式需要提供--eval_data参数")
            return
        
        if not os.path.exists(args.eval_data):
            print(f"错误: 评测数据文件不存在: {args.eval_data}")
            return
        
        # 批量生成
        results = batch_generate_eval(model, args.eval_data, args.output, args.max_new_tokens, args.device)
        print(f"\n批量评测完成！")


if __name__ == '__main__':
    main()
