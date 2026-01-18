import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from contextlib import nullcontext
from tqdm import tqdm
import os
import math
import json
from typing import Optional, Dict, Any, List
import wandb
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，适合服务器环境
import matplotlib.pyplot as plt

from data.parquet_dataset import ParquetMultiModalDataset, DataCollator
from model.vlm_model import MultiModalVLM
from config import TrainingConfig


class VLMTrainer:
    """VLM训练器"""
    
    def __init__(
        self,
        model: MultiModalVLM,
        config: TrainingConfig,
        train_dataset: ParquetMultiModalDataset,
        val_dataset: Optional[ParquetMultiModalDataset] = None,
        use_wandb: bool = False,
        wandb_project: str = "VLM-Training"
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.use_wandb = use_wandb
        
        # 设备设置
        self.device = torch.device(
            f"cuda:{config.local_rank}" if torch.cuda.is_available() and config.local_rank >= 0 else "cpu"
        )
        self.model.to(self.device)
        
        # 分布式训练
        self.is_distributed = config.use_ddp and dist.is_initialized()
        
        # 打印设备信息
        if self._is_main_process():
            print(f"使用设备: {self.device}")
            print(f"训练样本数: {len(train_dataset)}")
            if val_dataset:
                print(f"验证样本数: {len(val_dataset)}")
            if self.is_distributed:
                print(f"使用分布式训练，world size: {dist.get_world_size()}")
        
        if self.is_distributed:
            print(f"初始化分布式数据并行...")
            self.model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[config.local_rank],
                output_device=config.local_rank,
                find_unused_parameters=True  # 添加这个参数
            )
        
        # 混合精度训练设置
        self.use_amp = config.use_mixed_precision and torch.cuda.is_available()
        
        # 检查BFloat16支持
        if config.mixed_precision_dtype == "bfloat16" and torch.cuda.is_bf16_supported():
            self.dtype = torch.bfloat16
            # BFloat16需要特殊处理，禁用GradScaler
            self.scaler = None
            if self._is_main_process():
                print(f"使用BFloat16混合精度训练 (无GradScaler)")
        elif config.mixed_precision_dtype == "float16":
            self.dtype = torch.float16
            self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
            if self._is_main_process():
                print(f"使用Float16混合精度训练")
        else:
            self.dtype = torch.float32
            self.scaler = None
            self.use_amp = False
            if self._is_main_process():
                print(f"使用Float32训练 (无混合精度)")
        
        # 获取可训练参数
        trainable_params = []
        total_params = 0
        trainable_params_count = 0
        
        # 获取模型参数
        model_to_check = self.model.module if self.is_distributed else self.model
        for name, param in model_to_check.named_parameters():
            total_params += param.numel()
            if param.requires_grad:
                trainable_params.append(param)
                trainable_params_count += param.numel()
        
        if self._is_main_process():
            print(f"总参数: {total_params:,}")
            print(f"可训练参数: {trainable_params_count:,}")
            print(f"可训练比例: {trainable_params_count/total_params*100:.2f}%")
        
        # 优化器 - 确保学习率是浮点数
        lr = float(config.learning_rate)
        weight_decay = float(config.weight_decay)
        
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
            eps=1e-8,
            fused=True if torch.cuda.is_available() else False  # 使用fused优化器加速
        )
        
        # 学习率调度器
        total_batches = len(train_dataset) // max(1, config.batch_size)
        total_steps = total_batches * config.num_epochs
        
        # 总步数计算
        self.total_steps = total_steps
        
        if config.warmup_steps < total_steps:
            self.scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - config.warmup_steps
            )
        else:
            self.scheduler = None
        
        # 数据加载器
        if self.is_distributed:
            train_sampler = DistributedSampler(train_dataset)
            val_sampler = DistributedSampler(val_dataset) if val_dataset else None
        else:
            train_sampler = None
            val_sampler = None
        
        # 数据整理器
        tokenizer = model.get_tokenizer()
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        collator = DataCollator(pad_token_id=pad_token_id)
        
        # 使用配置的worker数量（不强制限制）
        # 注意：在多卡分布式训练时，如果num_workers设置过大可能导致死锁
        # 例如：4卡 × 4 workers = 16个子进程，可能导致资源竞争
        # 如果遇到死锁问题，建议在多卡训练时将num_workers设为1
        num_workers = config.num_workers
        
        if self._is_main_process():
            world_size_info = f", world_size={config.world_size}" if self.is_distributed else ""
            print(f"数据加载器配置: num_workers={num_workers} (分布式: {self.is_distributed}{world_size_info})")
            if self.is_distributed and num_workers > 1:
                print(f"  提示: 多卡训练时使用 num_workers={num_workers}，如果遇到死锁，建议减小此值")
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2 if num_workers > 0 else None
        )
        
        if val_dataset:
            # 验证时使用更少的worker（避免占用过多资源）
            val_num_workers = max(1, min(2, num_workers)) if num_workers > 0 else 0
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collator,
                num_workers=val_num_workers,
                pin_memory=True
            )
        else:
            self.val_loader = None
        
        # 检查点目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 初始化WandB
        if use_wandb and self._is_main_process():
            wandb.init(project=wandb_project, config=vars(config))
            self.wandb_run_id = wandb.run.id if wandb.run else None
        else:
            self.wandb_run_id = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 损失记录：存储 (step, loss) 元组
        self.train_loss_history: List[tuple] = []  # [(step, loss), ...]
        self.val_loss_history: List[tuple] = []  # [(step, val_loss), ...]
        
        if self._is_main_process():
            print("训练器初始化完成")
            print(f"每个epoch步数: {len(self.train_loader)}")
            print(f"总训练步数: {self.total_steps}")
    
    def _is_main_process(self) -> bool:
        """判断是否是主进程"""
        return not self.is_distributed or dist.get_rank() == 0
    
    def get_learning_rate(self, step: int) -> float:
        """获取当前学习率（带warmup）"""
        if step < self.config.warmup_steps:
            # Warmup阶段
            return float(self.config.learning_rate) * (step / max(1, self.config.warmup_steps))
        else:
            # Cosine annealing
            if self.scheduler is not None:
                progress = (step - self.config.warmup_steps) / max(1, self.total_steps - self.config.warmup_steps)
                return float(self.config.learning_rate) * 0.5 * (1 + math.cos(math.pi * progress))
            else:
                return float(self.config.learning_rate)
    
    def train_epoch(self, epoch: int):
        """训练一个epoch"""
        self.model.train()
        
        if self.is_distributed:
            self.train_loader.sampler.set_epoch(epoch)
        
        total_loss = 0
        total_steps = len(self.train_loader)
        
        pbar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.config.num_epochs}",
            disable=not self._is_main_process()
        )
        
        for step, batch in enumerate(pbar):
            try:
                # 解包batch
                X, Y, loss_mask, pixel_values = batch
                
                # 移动到设备
                X = X.to(self.device, non_blocking=True)
                Y = Y.to(self.device, non_blocking=True)
                loss_mask = loss_mask.to(self.device, non_blocking=True)
                pixel_values = pixel_values.to(self.device, non_blocking=True)
                
                # 更新学习率
                current_step = epoch * total_steps + step
                lr = self.get_learning_rate(current_step)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = lr
                
                # 混合精度上下文
                if self.use_amp and self.dtype != torch.bfloat16:
                    # 对于Float16使用标准的autocast
                    autocast_ctx = torch.cuda.amp.autocast(dtype=self.dtype)
                else:
                    # 对于BFloat16或禁用混合精度的情况
                    autocast_ctx = nullcontext()
                
                # 前向传播
                with autocast_ctx:
                    outputs = self.model(
                        input_ids=X,
                        pixel_values=pixel_values,
                        labels=Y
                    )
                    
                    # 计算masked loss
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    
                    # 计算每个位置的loss
                    loss_per_token = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        Y.view(-1)
                    ).view(Y.size())
                    
                    # 应用loss mask
                    mask_sum = loss_mask.sum()
                    if mask_sum > 0:
                        loss = (loss_per_token * loss_mask).sum() / mask_sum
                    else:
                        loss = loss_per_token.mean()
                    
                    # 梯度累积
                    loss = loss / self.config.accumulation_steps
                
                # 反向传播
                if self.scaler is not None:
                    # Float16使用GradScaler
                    self.scaler.scale(loss).backward()
                else:
                    # BFloat16或禁用混合精度直接backward
                    loss.backward()
                
                # 梯度累积更新
                if (step + 1) % self.config.accumulation_steps == 0:
                    # 梯度裁剪
                    if self.scaler is not None:
                        self.scaler.unscale_(self.optimizer)
                    
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )
                    
                    # 更新参数
                    if self.scaler is not None:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    
                    # 学习率调度
                    if self.scheduler is not None and step >= self.config.warmup_steps:
                        self.scheduler.step()
                    
                    self.optimizer.zero_grad(set_to_none=True)  # 使用set_to_none节省内存
                
                # 记录损失
                current_loss = loss.item() * self.config.accumulation_steps
                total_loss += current_loss
                
                # 计算当前全局步数
                current_global_step = self.global_step + step
                
                # 记录损失到历史（只在主进程记录）
                if self._is_main_process():
                    self.train_loss_history.append((current_global_step, current_loss))
                
                # 更新进度条
                avg_loss = total_loss / (step + 1)
                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "lr": f"{lr:.2e}"
                })
                
                # 记录到wandb
                if self.use_wandb and self._is_main_process():
                    wandb.log({
                        "train_loss": current_loss,
                        "learning_rate": lr,
                        "step": current_global_step
                    })
                
                # 定期保存检查点
                if (self.global_step + step) % self.config.save_every_n_steps == 0 and self._is_main_process():
                    self.save_checkpoint(epoch, step, is_best=False)
                
                # 定期打印日志
                if (self.global_step + step) % self.config.log_every_n_steps == 0 and self._is_main_process():
                    self.log_training_info(epoch, step, avg_loss, lr)
                
                # 清理内存
                del outputs, loss, loss_per_token
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    print(f"步骤 {step} CUDA内存不足，跳过此batch")
                    if self.scaler is not None:
                        self.scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    
                    # 清理内存
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    print(f"训练步骤 {step} 出现错误: {e}")
                    import traceback
                    traceback.print_exc()
                    continue
            except Exception as e:
                print(f"训练步骤 {step} 出现错误: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # 更新全局步数
        self.global_step += total_steps
        
        avg_loss = total_loss / max(1, total_steps)
        return avg_loss
    
    def evaluate(self) -> Optional[float]:
        """评估模型"""
        if self.val_dataset is None or self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", disable=not self._is_main_process()):
                try:
                    X, Y, loss_mask, pixel_values = batch
                    
                    X = X.to(self.device, non_blocking=True)
                    Y = Y.to(self.device, non_blocking=True)
                    loss_mask = loss_mask.to(self.device, non_blocking=True)
                    pixel_values = pixel_values.to(self.device, non_blocking=True)
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=X,
                        pixel_values=pixel_values,
                        labels=Y
                    )
                    
                    # 计算masked loss
                    logits = outputs.logits
                    loss_fct = nn.CrossEntropyLoss(reduction='none')
                    loss_per_token = loss_fct(
                        logits.view(-1, logits.size(-1)),
                        Y.view(-1)
                    ).view(Y.size())
                    
                    mask_sum = loss_mask.sum()
                    if mask_sum > 0:
                        loss = (loss_per_token * loss_mask).sum() / mask_sum
                    else:
                        loss = loss_per_token.mean()
                    
                    total_loss += loss.item() * X.size(0)
                    total_samples += X.size(0)
                    
                    # 清理内存
                    del outputs, loss_per_token
                    
                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"评估时CUDA内存不足，跳过此batch")
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        continue
                    else:
                        raise
        
        avg_loss = total_loss / max(1, total_samples)
        
        # 记录验证损失到历史（只在主进程记录）
        if self._is_main_process():
            self.val_loss_history.append((self.global_step, avg_loss))
        
        # 记录到wandb
        if self.use_wandb and self._is_main_process():
            wandb.log({"val_loss": avg_loss})
        
        return avg_loss
    
    def log_training_info(self, epoch: int, step: int, loss: float, lr: float):
        """记录训练信息"""
        # 获取GPU内存信息
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
            gpu_memory_max = torch.cuda.max_memory_allocated() / 1024**3
            print(f"Epoch: {epoch+1}/{self.config.num_epochs}, "
                  f"Step: {step}, "
                  f"Loss: {loss:.4f}, "
                  f"LR: {lr:.2e}, "
                  f"GPU内存: {gpu_memory:.2f}GB (峰值: {gpu_memory_max:.2f}GB)")
        else:
            print(f"Epoch: {epoch+1}/{self.config.num_epochs}, "
                  f"Step: {step}, "
                  f"Loss: {loss:.4f}, "
                  f"LR: {lr:.2e}")
    
    def train(self):
        """训练模型"""
        if self._is_main_process():
            print(f"开始训练，总epoch数: {self.config.num_epochs}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            if self._is_main_process():
                print(f"\n{'='*50}")
                print(f"Epoch {epoch+1}/{self.config.num_epochs}")
                print(f"{'='*50}")
            
            # 训练
            train_loss = self.train_epoch(epoch)
            
            # 评估
            val_loss = self.evaluate()
            
            # 保存检查点
            if self._is_main_process():
                print(f"\nEpoch {epoch+1} 完成:")
                print(f"  训练损失: {train_loss:.4f}")
                if val_loss is not None:
                    print(f"  验证损失: {val_loss:.4f}")
                    
                    # 保存最佳模型
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint(epoch, is_best=True)
                        print(f"  ✓ 保存最佳模型，验证损失: {val_loss:.4f}")
                
                # 保存常规检查点
                self.save_checkpoint(epoch, is_best=False)
                print(f"  检查点已保存到: {self.config.save_dir}")
            
            # 更新当前epoch
            self.current_epoch = epoch + 1
        
        if self._is_main_process():
            print("\n" + "="*50)
            print("训练完成!")
            print(f"最佳验证损失: {self.best_val_loss:.4f}")
            print(f"模型保存在: {self.config.save_dir}")
            print("="*50)
            
            # 保存损失数据并绘制曲线图
            self.save_loss_history()
            self.plot_loss_curve()
    
    def save_checkpoint(self, epoch: int, step: int = 0, is_best: bool = False):
        """保存检查点"""
        if not self._is_main_process():
            return
        
        # 准备检查点数据
        model_to_save = self.model.module if self.is_distributed else self.model
        checkpoint = {
            'epoch': epoch,
            'step': step,
            'global_step': self.global_step,
            'model_state_dict': model_to_save.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        
        if self.wandb_run_id:
            checkpoint['wandb_run_id'] = self.wandb_run_id
        
        # 确定文件名
        if is_best:
            filename = "best_model.pt"
        else:
            filename = f"checkpoint_epoch{epoch+1}.pt"
        
        # 保存文件
        checkpoint_path = os.path.join(self.config.save_dir, filename)
        torch.save(checkpoint, checkpoint_path)
        
        if self._is_main_process():
            print(f"检查点保存到: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            print(f"检查点不存在: {checkpoint_path}")
            return
        
        # 使用 weights_only=False 以支持包含自定义类的检查点（PyTorch 2.6+ 默认需要）
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # 加载模型状态
        model_to_load = self.model.module if self.is_distributed else self.model
        checkpoint_state_dict = checkpoint['model_state_dict']
        model_state_dict = model_to_load.state_dict()
        
        # 智能加载：处理LoRA参数形状不匹配的情况
        missing_keys = []
        unexpected_keys = []
        shape_mismatch_keys = []
        
        # 分离LoRA参数和其他参数
        lora_keys = [k for k in checkpoint_state_dict.keys() if 'lora' in k.lower()]
        non_lora_keys = [k for k in checkpoint_state_dict.keys() if 'lora' not in k.lower()]
        
        # 先加载非LoRA参数（使用strict=False允许部分不匹配）
        filtered_state_dict = {}
        for key in non_lora_keys:
            if key in model_state_dict:
                if model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                    filtered_state_dict[key] = checkpoint_state_dict[key]
                else:
                    shape_mismatch_keys.append(key)
                    if self._is_main_process():
                        print(f"  跳过形状不匹配的参数: {key} "
                              f"(检查点: {checkpoint_state_dict[key].shape}, "
                              f"当前模型: {model_state_dict[key].shape})")
            else:
                unexpected_keys.append(key)
        
        # 检查当前模型是否使用LoRA
        model_has_lora = any('lora' in k.lower() for k in model_state_dict.keys())
        checkpoint_has_lora = len(lora_keys) > 0
        
        # 处理LoRA参数：只加载形状匹配的
        if model_has_lora and checkpoint_has_lora:
            # 如果当前模型和检查点都有LoRA，尝试加载匹配的参数
            for key in lora_keys:
                if key in model_state_dict:
                    if model_state_dict[key].shape == checkpoint_state_dict[key].shape:
                        filtered_state_dict[key] = checkpoint_state_dict[key]
                    else:
                        shape_mismatch_keys.append(key)
                        if self._is_main_process():
                            print(f"  跳过LoRA参数形状不匹配: {key} "
                                  f"(检查点: {checkpoint_state_dict[key].shape}, "
                                  f"当前模型: {model_state_dict[key].shape})")
        elif checkpoint_has_lora and not model_has_lora:
            # 如果检查点有LoRA但当前模型没有，跳过所有LoRA参数
            if self._is_main_process():
                print(f"  检查点包含LoRA参数，但当前模型未使用LoRA，跳过所有LoRA参数")
            shape_mismatch_keys.extend(lora_keys)
        elif model_has_lora and not checkpoint_has_lora:
            # 如果当前模型有LoRA但检查点没有，这是正常的（可能从非LoRA检查点加载）
            if self._is_main_process():
                print(f"  当前模型使用LoRA，但检查点不包含LoRA参数，将使用随机初始化的LoRA参数")
        
        # 检查缺失的键
        for key in model_state_dict.keys():
            if key not in checkpoint_state_dict:
                missing_keys.append(key)
        
        # 加载过滤后的状态字典
        load_result = model_to_load.load_state_dict(filtered_state_dict, strict=False)
        
        if self._is_main_process():
            print(f"从检查点恢复: {checkpoint_path}")
            if load_result.missing_keys:
                print(f"  缺失的键 ({len(load_result.missing_keys)}): {load_result.missing_keys[:5]}...")
            if load_result.unexpected_keys:
                print(f"  意外的键 ({len(load_result.unexpected_keys)}): {load_result.unexpected_keys[:5]}...")
            if shape_mismatch_keys:
                print(f"  形状不匹配的键 ({len(shape_mismatch_keys)}): 已跳过")
                if len(shape_mismatch_keys) <= 10:
                    for key in shape_mismatch_keys:
                        print(f"    - {key}")
        
        # 加载优化器和调度器（如果形状匹配）
        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            if self._is_main_process():
                print(f"  警告: 无法加载优化器状态: {e}")
                print(f"  将使用新的优化器状态")
        
        if self.scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception as e:
                if self._is_main_process():
                    print(f"  警告: 无法加载调度器状态: {e}")
        
        # 加载scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            except Exception as e:
                if self._is_main_process():
                    print(f"  警告: 无法加载scaler状态: {e}")
        
        # 恢复训练状态（但如果是不同阶段的检查点，重置epoch）
        checkpoint_epoch = checkpoint.get('epoch', 0)
        # 如果是从预训练检查点加载到SFT，重置epoch
        if 'config' in checkpoint and checkpoint['config'] is not None:
            checkpoint_config = checkpoint['config']
            checkpoint_stage = getattr(checkpoint_config, 'stage', None) if hasattr(checkpoint_config, 'stage') else None
            if checkpoint_stage is not None and checkpoint_stage != self.config.stage:
                if self._is_main_process():
                    print(f"  检测到阶段变化: {checkpoint_stage} -> {self.config.stage}，重置epoch")
                self.current_epoch = 0
                self.global_step = 0
            else:
                self.current_epoch = checkpoint_epoch
                self.global_step = checkpoint.get('global_step', 0)
        else:
            self.current_epoch = checkpoint_epoch
            self.global_step = checkpoint.get('global_step', 0)
        
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self._is_main_process():
            print(f"  恢复epoch: {self.current_epoch}")
            print(f"  恢复step: {self.global_step}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")
    
    def save_loss_history(self):
        """保存损失历史到JSON文件"""
        if not self._is_main_process():
            return
        
        # 准备数据
        loss_data = {
            "train_loss": [{"step": step, "loss": loss} for step, loss in self.train_loss_history],
            "val_loss": [{"step": step, "loss": loss} for step, loss in self.val_loss_history]
        }
        
        # 保存到JSON文件
        loss_file = os.path.join(self.config.save_dir, "loss_history.json")
        with open(loss_file, 'w', encoding='utf-8') as f:
            json.dump(loss_data, f, indent=2, ensure_ascii=False)
        
        print(f"损失历史已保存到: {loss_file}")
        print(f"  训练损失记录数: {len(self.train_loss_history)}")
        print(f"  验证损失记录数: {len(self.val_loss_history)}")
    
    def plot_loss_curve(self):
        """绘制损失曲线图"""
        if not self._is_main_process():
            return
        
        if not self.train_loss_history:
            print("警告: 没有训练损失数据，无法绘制曲线图")
            return
        
        # 创建图形
        plt.figure(figsize=(12, 6))
        
        # 提取训练损失数据
        train_steps = [step for step, _ in self.train_loss_history]
        train_losses = [loss for _, loss in self.train_loss_history]
        
        # 绘制训练损失曲线
        plt.plot(train_steps, train_losses, label='Train Loss', color='blue', alpha=0.7, linewidth=1.5)
        
        # 如果有验证损失，也绘制验证损失曲线
        if self.val_loss_history:
            val_steps = [step for step, _ in self.val_loss_history]
            val_losses = [loss for _, loss in self.val_loss_history]
            plt.plot(val_steps, val_losses, label='Validation Loss', color='red', alpha=0.7, linewidth=1.5, marker='o', markersize=4)
        
        # 设置图形属性
        plt.xlabel('Step', fontsize=12)
        plt.ylabel('Loss', fontsize=12)
        plt.title('Training Loss Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存图片
        plot_file = os.path.join(self.config.save_dir, "loss_curve.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()  # 关闭图形以释放内存
        
        print(f"损失曲线图已保存到: {plot_file}")


def train_vlm(
    model_config,
    training_config,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    use_wandb: bool = False,
    resume_from: Optional[str] = None
):
    """训练VLM模型的入口函数"""
    
    # 将训练阶段传递给模型配置
    model_config.training_stage = training_config.stage
    
    print(f"创建模型...")
    model = MultiModalVLM(model_config)
    
    print(f"获取tokenizer和processor...")
    tokenizer = model.get_tokenizer()
    processor = model.get_processor()
    
    print(f"创建训练数据集: {train_data_path}")
    try:
        train_dataset = ParquetMultiModalDataset(
            parquet_path=train_data_path,
            tokenizer=tokenizer,
            image_processor=processor,
            stage=training_config.stage,
            max_length=training_config.max_seq_length,
            image_special_token=model_config.image_special_token
        )
        print(f"训练数据集创建成功，大小: {len(train_dataset)}")
    except Exception as e:
        print(f"创建训练数据集失败: {e}")
        raise
    
    # 创建验证数据集（如果有）
    val_dataset = None
    if val_data_path and os.path.exists(val_data_path):
        print(f"创建验证数据集: {val_data_path}")
        try:
            val_dataset = ParquetMultiModalDataset(
                parquet_path=val_data_path,
                tokenizer=tokenizer,
                image_processor=processor,
                stage=training_config.stage,
                max_length=training_config.max_seq_length,
                image_special_token=model_config.image_special_token
            )
            print(f"验证数据集创建成功，大小: {len(val_dataset)}")
        except Exception as e:
            print(f"创建验证数据集失败: {e}")
            val_dataset = None
    
    print(f"创建训练器...")
    trainer = VLMTrainer(
        model=model,
        config=training_config,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        use_wandb=use_wandb,
        wandb_project=f"VLM-{training_config.stage.upper()}"
    )
    
    # 恢复训练（如果有）
    if resume_from and os.path.exists(resume_from):
        print(f"从检查点恢复: {resume_from}")
        trainer.load_checkpoint(resume_from)
    
    # 开始训练
    print(f"开始训练...")
    trainer.train()
    
    return trainer
