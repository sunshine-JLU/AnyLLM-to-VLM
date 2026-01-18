# AnyLLM-to-VLM 🚀

Turn any text-only LLM into a Vision-Language Model through efficient training.

## Features
- 📷 Support for CLIP and other vision encoders
- 🦙 Compatible with popular LLMs (Qwen, LLaMA, etc.)

## Guidance
- Project Scalability Guide :[`README_EXTENSIBILITY.md`](https://github.com/sunshine-JLU/AnyLLM-to-VLM/blob/main/README_EXTENSIBILITY.md)
- How to change a new vision encoder :[`QUICK_START_VISION_ENCODERS.md`](https://github.com/sunshine-JLU/AnyLLM-to-VLM/blob/main/QUICK_START_VISION_ENCODERS.md)
- more details concerning vision encoder :[`README_VISION_ENCODERS.md`](https://github.com/sunshine-JLU/AnyLLM-to-VLM/blob/main/README_VISION_ENCODERS.md)
- how to use multi-gpu to train:[`README_MULTI_GPU.md`](https://github.com/sunshine-JLU/AnyLLM-to-VLM/blob/main/README_MULTI_GPU.md)
  
## Quick Start

Follow these steps to get started quickly [Test SoftWare Environment : Python >= 3.12]:

1. **Clone the Repository**  
   Clone the repository to your local machine:
   ```bash
   git clone https://github.com/sunshine-JLU/AnyLLM-to-VLM.git

   cd AnyLLM-to-VLM
   
2. **Papare the environment**  
   ```bash
   pip install -r requirements.txt
   pip install protobuf --upgrade
3. **Papare the dataset and your models**  
   ```bash
   cd dataset
   wget https://modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/sft_data.parquet
   wget https://modelscope.cn/datasets/gongjy/minimind-v_dataset/resolve/master/pretrain_data.parquet
   cd ..
   cd models
   modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct
   modelscope download --model openai-mirror/clip-vit-base-patch16 --local_dir ./clip-vit-base-patch16
   modelscope download --model google/siglip-so400m-patch14-384  --local_dir ./siglip-so400m-patch14-384
   cd ..
4. **Run the Pretrain Script**
   ```bash
   python train_vlm.py --config configs/vlm_pretrain.yaml --stage pretrain
   # python train_vlm.py --config configs/vlm_pretrain.yaml --stage pretrain | tee pretrain.log
   # 使用4张GPU训练
   # torchrun --nproc_per_node=4 train_vlm.py --config configs/vlm_pretrain.yaml --stage pretrain | tee pretrain.log       
   # In a new terminal, type `nvitop` :
<img width="2072" height="262" alt="image" src="https://github.com/user-attachments/assets/c8888c98-e61e-4694-8c2f-fd49bb11874b" />
<img width="1976" height="812" alt="853c78a4c1d0691027899dcb1012ac18" src="https://github.com/user-attachments/assets/147c96ea-d8e3-43ab-865b-b22ab5c4c680" />


5. **Run the Supervised Fine-Tuning Script**
   ```bash
   python train_vlm.py --config configs/vlm_sft.yaml --resume ./checkpoints/pretrain/checkpoint_epoch1.pt --stage sft
   # 使用4张GPU训练 
   # torchrun --nproc_per_node=4 train_vlm.py --config configs/vlm_sft.yaml --resume ./checkpoints/pretrain/checkpoint_epoch1.pt --stage sft | tee sft.log      
   # python train_vlm.py --config configs/vlm_sft.yaml --resume ./checkpoints/pretrain/checkpoint_epoch1.pt --stage sft | tee sft.log
6. **Eval the model**
   ```bash
   python eval_vlm.py --checkpoint ./checkpoints/sft/checkpoint_epoch1.pt --config configs/vlm_sft.yaml --mode generate --image ./sample.jpg --question "描述这张图片" --max_new_tokens 50
7. **Test Result**

<div style="width: 80%; margin: 0 auto; overflow-x: auto;">
<table>
  <thead>
    <tr>
      <th width="30%">test image</th>
      <th width="70%">clip-vit-base-patch16-MLPProjection-patch_insert-Qwen2.5-0.5B-Instruct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><img src="./test_images/sample1.jpg" alt="sample1" style="width: 150px; height: 150px; object-fit: cover;"></td>
      <td>该图片描绘了一个场景,一个男人站在一个长凳上。他手里拿着一把敞开的拐杖。人们可以看到他们正享受着闲暇时间。在背景中可以看到一只狗。这些人都似乎正在享受着他们的时光。</td>
    </tr>
    <tr>
      <td><img src="./test_images/sample2.jpg" alt="sample2" style="width: 150px; height: 150px; object-fit: cover;"></td>
      <td>照片里有一座高耸的木制塔楼和一条小河。塔楼上有几棵松树在远处飘扬。河上有几棵树木和山景，给照片增添了动感。这条河沿着岛屿的海岸线流淌。</td>
    </tr>
    <tr>
      <td><img src="./test_images/sample4.jpg" alt="sample4" style="width: 150px; height: 150px; object-fit: cover;"></td>
      <td>图片中有一只猫站在一张桌子上。它看上去很舒服，可能是为了享受他的放松时光。</td>
    </tr>   
  </tbody>
</table>
</div>

8. **Test HardWare Environment**
<img width="1290" height="332" alt="76be5859-27b6-4cd6-941c-e1ef95b769cc" src="https://github.com/user-attachments/assets/50496cc7-417a-42f1-8f44-a6e555c09cca" />


### Acknowledgments  
This project was inspired by [jingyaogong/minimind-v](https://github.com/jingyaogong/minimind-v). We extend our thanks for its open-source contribution, which provided valuable inspiration and support for our development.
