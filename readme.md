# AnyLLM-to-VLM 🚀

Turn any text-only LLM into a Vision-Language Model through efficient training.

## Features
- 📷 Support for CLIP and other vision encoders
- 🦙 Compatible with popular LLMs (Qwen, LLaMA, etc.)

## Quick Start

Follow these steps to get started quickly:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/sunshine-JLU/AnyLLM-to-VLM.git

   cd AnyLLM-to-VLM
   
2. **Papare the environment**  
   ```bash
   pip install -r requirements.txt
3. **Papare the dataset and your models**  
   ```bash
   cd dataset
   wget https://modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/sft_data.parquet
   wget https://modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_data.parquet
   cd ..
   cd models
   modelscope download --model Qwen/Qwen3-0.6B --local_dir ./Qwen3-0.6B
   modelscope download --model openai-mirror/clip-vit-base-patch16 --local_dir ./clip-vit-base-patch16
   cd ..
4. **Run the Pretrain Script**
   ```bash
   python train_vlm.py --config configs/vlm_pretrain.yaml --stage pretrain
   # python train_vlm.py --config configs/vlm_pretrain.yaml --stage pretrain | tee pretrain.log
   # In a new terminal, type `nvitop` :
<img width="2072" height="262" alt="image" src="https://github.com/user-attachments/assets/c8888c98-e61e-4694-8c2f-fd49bb11874b" />


5. **Run the Supervised Fine-Tuning Script**
   ```bash
   python train_vlm.py --config configs/vlm_sft.yaml --stage sft
   # python train_vlm.py --config configs/vlm_sft.yaml --stage sft | tee sft.log
7. **Eval the model**
   ```bash
   python eval_vlm.py --checkpoint ./checkpoints/sft/best_model.pt --config configs/vlm_sft.yaml --mode generate --image ./sample.jpg --question "描述这张图片" --max_new_tokens 512



### Acknowledgments  
This project was inspired by [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v). We extend our thanks for its open-source contribution, which provided valuable inspiration and support for our development.
