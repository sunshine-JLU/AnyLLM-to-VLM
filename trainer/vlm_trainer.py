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
from typing import Optional, Dict, Any
import wandb

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
        
        # 限制worker数量避免内存问题
        num_workers = min(config.num_workers, 4)
        
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
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                collate_fn=collator,
                num_workers=min(2, num_workers),  # 验证时使用更少的worker
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
                        "step": self.global_step + step
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
        model_to_load.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器和调度器
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载scaler
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        # 恢复训练状态
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        if self._is_main_process():
            print(f"从检查点恢复: {checkpoint_path}")
            print(f"  恢复epoch: {self.current_epoch}")
            print(f"  恢复step: {self.global_step}")
            print(f"  最佳验证损失: {self.best_val_loss:.4f}")


def train_vlm(
    model_config,
    training_config,
    train_data_path: str,
    val_data_path: Optional[str] = None,
    use_wandb: bool = False,
    resume_from: Optional[str] = None
):
    """训练VLM模型的入口函数"""
    
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