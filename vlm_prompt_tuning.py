from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from transformers import CLIPModel, CLIPImageProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
import yaml

class VisionLanguageDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_name, split="train", max_samples=None, 
                 image_processor=None, tokenizer=None, text_column="caption", image_column="image"):
        self.dataset = load_dataset(dataset_name, split=split)
        if max_samples:
            self.dataset = self.dataset.select(range(max_samples))
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.image_column = image_column
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item[self.image_column]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image) if hasattr(image, 'shape') else Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
            
        caption = item[self.text_column]
        if isinstance(caption, list):
            caption = caption[0]  # 取第一个caption
            
        pixel_values = self.image_processor(images=image, return_tensors="pt").pixel_values[0]
        text_tokens = self.tokenizer(caption, max_length=32, truncation=True, 
                                   padding="max_length", return_tensors="pt")
        return pixel_values, text_tokens.input_ids[0], text_tokens.attention_mask[0]

def collate_fn(batch):
    pixels, input_ids, attention_masks = zip(*batch)
    return torch.stack(pixels), torch.stack(input_ids), torch.stack(attention_masks)

class PromptTunedVLM(torch.nn.Module):
    def __init__(self, vision_model_path, language_model_path, prompt_text="A photo of ",
                 vision_layers_to_unfreeze=2, mlp_config=None):
        super().__init__()
        
        # 视觉编码器 - 保持float32
        self.vision_encoder = CLIPModel.from_pretrained(vision_model_path).vision_model
        
        # 语言模型 - 使用bfloat16或float32而不是float16
        self.language_model = AutoModelForCausalLM.from_pretrained(
            language_model_path, 
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        # 解冻视觉编码器指定层数
        total_layers = len(self.vision_encoder.encoder.layers)
        layers_to_unfreeze = [f"layers.{i}" for i in range(total_layers - vision_layers_to_unfreeze, total_layers)]
        for name, param in self.vision_encoder.named_parameters():
            param.requires_grad = any(layer in name for layer in layers_to_unfreeze) or "post_layernorm" in name

        # 视觉-语言投影层
        vision_dim = self.vision_encoder.config.hidden_size
        language_dim = self.language_model.config.hidden_size
        
        # 默认MLP配置
        default_mlp_config = {
            'hidden_dim': 4 * vision_dim,
            'activation': 'GELU',
            'dropout': 0.1,
            'use_layernorm': True
        }
        
        if mlp_config:
            default_mlp_config.update(mlp_config)
        
        mlp_config = default_mlp_config
        
        # 构建MLP投影层
        layers = [
            torch.nn.Linear(vision_dim, mlp_config['hidden_dim']),
            self._get_activation(mlp_config['activation']),
            torch.nn.Dropout(mlp_config['dropout']),
            torch.nn.Linear(mlp_config['hidden_dim'], language_dim)
        ]
        
        if mlp_config['use_layernorm']:
            layers.append(torch.nn.LayerNorm(language_dim))
            
        self.vision_projection = torch.nn.Sequential(*layers)

        # 提示词
        self.tokenizer = AutoTokenizer.from_pretrained(language_model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.prompt_text = prompt_text
        self.prompt_ids = self.tokenizer(self.prompt_text, return_tensors="pt").input_ids[0]

    def _get_activation(self, activation_name):
        activations = {
            'GELU': torch.nn.GELU(),
            'ReLU': torch.nn.ReLU(),
            'SiLU': torch.nn.SiLU(),
            'Tanh': torch.nn.Tanh()
        }
        return activations.get(activation_name, torch.nn.GELU())

    def forward(self, pixel_values, input_ids, attention_mask):
        batch_size = pixel_values.size(0)
        
        # 视觉特征提取 - 确保使用正确数据类型
        visual_features = self.vision_encoder(pixel_values=pixel_values).pooler_output
        projected_visual = self.vision_projection(visual_features).unsqueeze(1)
        
        # 确保投影后的视觉特征与语言模型的数据类型匹配
        if self.language_model.dtype != projected_visual.dtype:
            projected_visual = projected_visual.to(self.language_model.dtype)

        # 提示词嵌入
        prompt_ids = self.prompt_ids.to(input_ids.device).unsqueeze(0).expand(batch_size, -1)
        prompt_embeddings = self.language_model.model.embed_tokens(prompt_ids)

        # 文本嵌入
        text_embeddings = self.language_model.model.embed_tokens(input_ids)

        # 拼接所有嵌入 - 确保相同数据类型
        combined_embeddings = torch.cat([projected_visual, prompt_embeddings, text_embeddings], dim=1)

        # 构建标签（忽略视觉和提示词部分）
        visual_length = 1  # 视觉特征长度
        prompt_length = prompt_ids.shape[1]
        labels = torch.cat([
            torch.full((batch_size, visual_length + prompt_length), -100, device=input_ids.device),
            input_ids
        ], dim=1)

        return self.language_model(inputs_embeds=combined_embeddings, labels=labels).loss

class VLMTrainer:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建输出目录
        self.checkpoint_path = Path(config.get('checkpoint_path', './ckpt_prompt'))
        self.checkpoint_path.mkdir(exist_ok=True)
        
        # 初始化组件
        self._setup_components()
        
    def _setup_components(self):
        # 图像处理器
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config['vision_model_path'])
        
        # 分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['language_model_path'], 
                                                     trust_remote_code=True)
        
        # 数据集
        dataset_config = self.config.get('dataset', {})
        self.train_dataset = VisionLanguageDataset(
            dataset_name=dataset_config.get('name', 'flickr30k'),
            split=dataset_config.get('split', 'train'),
            max_samples=dataset_config.get('max_samples'),
            image_processor=self.image_processor,
            tokenizer=self.tokenizer,
            text_column=dataset_config.get('text_column', 'caption'),
            image_column=dataset_config.get('image_column', 'image')
        )
        
        # 数据加载器
        self.data_loader = DataLoader(
            self.train_dataset, 
            batch_size=self.config.get('batch_size', 32),
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.get('num_workers', 4)
        )
        
        # 模型
        model_config = self.config.get('model', {})
        self.model = PromptTunedVLM(
            vision_model_path=self.config['vision_model_path'],
            language_model_path=self.config['language_model_path'],
            prompt_text=model_config.get('prompt_text', 'A photo of '),
            vision_layers_to_unfreeze=model_config.get('vision_layers_to_unfreeze', 2),
            mlp_config=model_config.get('mlp_config')
        ).to(self.device)
        
        # 优化器
        optimizer_config = self.config.get('optimizer', {})
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=optimizer_config.get('lr', 1e-4),
            weight_decay=optimizer_config.get('weight_decay', 0.01)
        )
        
        # 设置tokenizer并行处理
        import os
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    def train(self):
        num_epochs = self.config.get('num_epochs', 10)
        
        for epoch in range(1, num_epochs + 1):
            self.model.train()
            total_loss = 0
            
            for pixels, input_ids, attention_masks in tqdm(self.data_loader, desc=f"Epoch {epoch}"):
                pixels = pixels.to(self.device)
                input_ids = input_ids.to(self.device)
                attention_masks = attention_masks.to(self.device)
                
                self.optimizer.zero_grad()
                loss = self.model(pixels, input_ids, attention_masks)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(self.data_loader)
            print(f"Epoch {epoch} avg loss = {avg_loss:.4f}")
            
            # 保存检查点
            if epoch % self.config.get('save_every', 5) == 0:
                torch.save(self.model.state_dict(), self.checkpoint_path / f"vlm_prompt_epoch{epoch}.pt")

def get_default_config():
    """返回默认配置"""
    return {
        'vision_model_path': './clip-vit-base-patch16',
        'language_model_path': './Qwen3-0.6B',
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 96,  # 减小batch size以避免内存问题
        'num_epochs': 30,
        'num_workers': 4,
        'checkpoint_path': './ckpt_prompt',
        'save_every': 5,
        'dataset': {
            'name': 'flickr30k',
            'split': 'test',
            'max_samples': 31783,  # 限制样本数量用于测试
            'text_column': 'caption',
            'image_column': 'image'
        },
        'model': {
            'prompt_text': 'A photo of ',
            'vision_layers_to_unfreeze': 2,
            'mlp_config': {
                'hidden_dim': 3072,  # 4 * 768 for CLIP base
                'activation': 'GELU',
                'dropout': 0.1,
                'use_layernorm': True
            }
        },
        'optimizer': {
            'lr': 1e-4,
            'weight_decay': 0.01
        }
    }

def main():
    parser = argparse.ArgumentParser(description='VLM Prompt Tuning')
    parser.add_argument('--config', type=str, help='Path to config YAML file')
    parser.add_argument('--vision_model', type=str, help='Path to vision model')
    parser.add_argument('--language_model', type=str, help='Path to language model')
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--batch_size', type=int, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = get_default_config()
    
    # 覆盖命令行参数
    if args.vision_model:
        config['vision_model_path'] = args.vision_model
    if args.language_model:
        config['language_model_path'] = args.language_model
    if args.dataset:
        config['dataset']['name'] = args.dataset
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['num_epochs'] = args.epochs
    
    # 设置tokenizer并行处理环境变量
    import os
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # 训练
    trainer = VLMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    main()