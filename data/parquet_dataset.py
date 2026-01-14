import json
import io
from PIL import Image
import torch
from torch.utils.data import Dataset
import pyarrow.parquet as pq
import pyarrow as pa
from typing import Optional, Tuple, Dict, Any
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ParquetMultiModalDataset(Dataset):
    """基于Parquet格式的多模态数据集（兼容MiniMind-V格式）"""
    
    def __init__(
        self,
        parquet_path: str,
        tokenizer,
        image_processor,
        stage: str = "pretrain",
        max_length: int = 512,
        image_special_token: str = '@' * 196,
        chat_template: str = "chatml"
    ):
        super().__init__()
        
        self.parquet_path = parquet_path
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.stage = stage
        self.max_length = max_length
        self.image_special_token = image_special_token
        self.chat_template = chat_template
        
        # 加载parquet数据
        self.table = pq.read_table(parquet_path)
        
        # 检查列名（兼容不同格式）
        self.columns = self.table.column_names
        
        # 设置tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 特殊token处理
        self.bos_id = tokenizer.encode('<|im_start|>assistant', add_special_tokens=False)
        self.eos_id = tokenizer.encode('<|im_end|>', add_special_tokens=False)
        
        # 缓存图像处理器（避免重复加载）
        self._cached_images = {}
    
    def __len__(self) -> int:
        return len(self.table)
    
    def _parse_conversations(self, conversations_str: str) -> list:
        """解析对话数据"""
        try:
            if isinstance(conversations_str, dict):
                return conversations_str.get('conversations', [])
            elif isinstance(conversations_str, list):
                return conversations_str
            else:
                # 尝试解析JSON字符串
                return json.loads(conversations_str)
        except:
            # 如果解析失败，返回空列表
            return []
    
    def _create_chat_prompt(self, conversations: list) -> str:
        """创建ChatML格式的prompt"""
        messages = []
        
        for i, turn in enumerate(conversations):
            role = turn.get('role', 'user' if i % 2 == 0 else 'assistant')
            content = turn.get('content', '')
            
            # 替换图像占位符
            if '<image>' in content:
                content = content.replace('<image>', self.image_special_token)
            
            messages.append({"role": role, "content": content})
        
        # 使用tokenizer的chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
        else:
            # 手动构建ChatML格式
            prompt = ""
            for msg in messages:
                prompt += f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n"
        
        return prompt
    
    def _generate_loss_mask(self, input_ids: list) -> list:
        """生成loss mask（在assistant回答部分和图像token位置计算loss）"""
        loss_mask = [0] * len(input_ids)
        
        # 找到图像token的位置（如果存在）
        image_token_ids = None
        if self.image_special_token:
            try:
                image_token_ids = self.tokenizer.encode(
                    self.image_special_token,
                    add_special_tokens=False
                )
            except:
                pass
        
        # 标记图像token位置（让投影层能收到梯度）
        if image_token_ids:
            len_image_ids = len(image_token_ids)
            i = 0
            while i < len(input_ids) - len_image_ids + 1:
                if input_ids[i:i + len_image_ids] == image_token_ids:
                    # 图像token位置也计算loss（这样投影层能收到梯度）
                    for j in range(i, min(i + len_image_ids, len(input_ids))):
                        loss_mask[j] = 1
                    i += len_image_ids
                else:
                    i += 1
        
        # 找到所有<|im_start|>assistant的位置
        i = 0
        while i < len(input_ids):
            # 检查是否匹配<|im_start|>assistant
            if i + len(self.bos_id) <= len(input_ids):
                if input_ids[i:i + len(self.bos_id)] == self.bos_id:
                    start_pos = i + len(self.bos_id)
                    end_pos = start_pos
                    
                    # 找到对应的<|im_end|>
                    while end_pos < len(input_ids):
                        if end_pos + len(self.eos_id) <= len(input_ids):
                            if input_ids[end_pos:end_pos + len(self.eos_id)] == self.eos_id:
                                break
                        end_pos += 1
                    
                    # 为assistant回答部分设置loss mask为1
                    # 注意：从start_pos开始，因为start_pos是assistant回答的开始
                    for j in range(start_pos, min(end_pos, self.max_length)):
                        loss_mask[j] = 1
                    
                    i = end_pos if end_pos < len(input_ids) else len(input_ids)
                else:
                    i += 1
            else:
                i += 1
        
        return loss_mask
    
    def _load_image_from_bytes(self, image_bytes: bytes) -> Image.Image:
        """从字节加载图像"""
        if image_bytes in self._cached_images:
            return self._cached_images[image_bytes]
        
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        self._cached_images[image_bytes] = image
        return image
    
    def _process_image(self, row_idx: int) -> torch.Tensor:
        """处理图像数据"""
        # 检查不同的列名格式
        image_data = None
        
        if 'image_bytes' in self.columns:
            # 直接从字节加载图像
            image_bytes = self.table['image_bytes'][row_idx].as_py()
            image = self._load_image_from_bytes(image_bytes)
        elif 'image' in self.columns:
            # 从图像文件路径加载
            image_path = self.table['image'][row_idx].as_py()
            if isinstance(image_path, bytes):
                # 如果image列是字节数据
                image = self._load_image_from_bytes(image_path)
            else:
                # 如果是文件路径（需要image_dir参数）
                raise ValueError("需要image_dir参数来加载图像文件")
        else:
            raise ValueError("Parquet文件中没有找到图像数据列")
        
        # 使用图像处理器
        pixel_values = self.image_processor(
            image, 
            return_tensors='pt'
        )['pixel_values'][0]
        
        return pixel_values.unsqueeze(0)  # 添加batch维度
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # 获取对话数据
        conversations_str = self.table['conversations'][idx].as_py()
        conversations = self._parse_conversations(conversations_str)
        
        # 创建prompt
        prompt = self._create_chat_prompt(conversations)
        
        # Tokenize
        encoding = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding='max_length' if self.max_length else False,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'][0]
        
        # 生成loss mask
        loss_mask = self._generate_loss_mask(input_ids.tolist())
        loss_mask = torch.tensor(loss_mask, dtype=torch.long)
        
        # 准备输入和标签（类似因果语言建模）
        X = input_ids[:-1]  # 输入：除了最后一个token
        Y = input_ids[1:]   # 标签：除了第一个token
        loss_mask = loss_mask[1:]  # loss mask也要相应偏移
        
        # 处理图像
        pixel_values = self._process_image(idx)
        
        return X, Y, loss_mask, pixel_values


def create_parquet_dataset(
    jsonl_path: str,
    output_path: str,
    image_dir: Optional[str] = None
):
    """
    将JSONL格式的数据集转换为Parquet格式
    支持两种格式：
    1. 图像路径格式（需要image_dir）
    2. 图像字节格式（图像已经编码为字节）
    """
    import json
    
    # 读取JSONL文件
    samples = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    # 准备数据
    conversations_list = []
    image_bytes_list = []
    
    for sample in samples:
        # 对话数据
        conversations = sample.get('conversations', [])
        conversations_list.append(json.dumps(conversations, ensure_ascii=False))
        
        # 图像数据
        image_info = sample.get('image', '')
        
        if image_dir and os.path.exists(image_dir):
            # 从文件加载图像
            image_path = os.path.join(image_dir, image_info)
            if os.path.exists(image_path):
                with open(image_path, 'rb') as img_file:
                    image_bytes = img_file.read()
            else:
                # 如果图像不存在，使用空字节
                image_bytes = b''
        elif isinstance(image_info, bytes):
            # 已经是字节数据
            image_bytes = image_info
        else:
            # 尝试将字符串转换为字节
            image_bytes = image_info.encode() if image_info else b''
        
        image_bytes_list.append(image_bytes)
    
    # 创建PyArrow表
    table = pa.table({
        'conversations': conversations_list,
        'image_bytes': image_bytes_list
    })
    
    # 写入Parquet文件
    pq.write_table(table, output_path)
    print(f"已创建Parquet文件: {output_path}, 包含 {len(samples)} 个样本")


class DataCollator:
    """数据整理器"""
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id
    
    def __call__(self, batch):
        X_list, Y_list, loss_mask_list, pixel_values_list = [], [], [], []
        
        max_len = max(item[0].size(0) for item in batch)
        
        for X, Y, loss_mask, pixel_values in batch:
            # 填充序列
            if X.size(0) < max_len:
                padding = torch.full((max_len - X.size(0),), self.pad_token_id, dtype=X.dtype)
                X = torch.cat([X, padding], dim=0)
                Y = torch.cat([Y, padding], dim=0)
                loss_mask = torch.cat([loss_mask, torch.zeros(max_len - loss_mask.size(0), dtype=loss_mask.dtype)], dim=0)
            
            X_list.append(X)
            Y_list.append(Y)
            loss_mask_list.append(loss_mask)
            pixel_values_list.append(pixel_values)
        
        # 堆叠图像张量
        if pixel_values_list[0].dim() == 4:  # [1, C, H, W]
            pixel_values = torch.cat(pixel_values_list, dim=0)  # [B, C, H, W]
        else:
            pixel_values = torch.stack(pixel_values_list, dim=0)
        
        return (
            torch.stack(X_list, dim=0),
            torch.stack(Y_list, dim=0),
            torch.stack(loss_mask_list, dim=0),
            pixel_values
        )


# 测试函数
def test_parquet_dataset():
    """测试Parquet数据集加载"""
    from transformers import AutoTokenizer, CLIPProcessor
    
    # 加载tokenizer和processor
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # 创建数据集
    dataset = ParquetMultiModalDataset(
        parquet_path="../dataset/pretrain_data.parquet",
        tokenizer=tokenizer,
        image_processor=processor,
        max_length=512,
        image_special_token='@' * 196
    )
    
    print(f"数据集大小: {len(dataset)}")
    
    # 测试一个样本
    if len(dataset) > 0:
        X, Y, loss_mask, pixel_values = dataset[0]
        print(f"X shape: {X.shape}")
        print(f"Y shape: {Y.shape}")
        print(f"loss_mask shape: {loss_mask.shape}")
        print(f"pixel_values shape: {pixel_values.shape}")
        print(f"Loss mask sum: {loss_mask.sum().item()}")
    
    return dataset


if __name__ == "__main__":
    # 测试数据集
    test_parquet_dataset()
    
    # 如果需要转换JSONL到Parquet
    # create_parquet_dataset(
    #     jsonl_path="../dataset/pretrain_vlm_data.jsonl",
    #     output_path="../dataset/pretrain_data.parquet",
    #     image_dir="../dataset/images"
    # )