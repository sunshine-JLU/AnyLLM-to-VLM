@echo off
REM Windows多卡训练启动脚本
REM 使用方法: scripts\train_multi_gpu.bat [配置文件路径] [训练阶段]

REM 设置默认值
if "%1"=="" (
    set CONFIG=configs\vlm_pretrain.yaml
) else (
    set CONFIG=%1
)

if "%2"=="" (
    set STAGE=pretrain
) else (
    set STAGE=%2
)

echo 配置文件: %CONFIG%
echo 训练阶段: %STAGE%

REM 检查是否有GPU
python -c "import torch; print('GPU数量:', torch.cuda.device_count())" 2>nul
if errorlevel 1 (
    echo 错误: 无法检测GPU，请确保已安装PyTorch和CUDA
    exit /b 1
)

REM 设置分布式训练环境变量
set CUDA_VISIBLE_DEVICES=0,1,2,3
set MASTER_ADDR=localhost
set MASTER_PORT=29500

REM 使用torchrun启动多卡训练
echo 启动多卡训练...
torchrun --nproc_per_node=4 --master_addr=%MASTER_ADDR% --master_port=%MASTER_PORT% train_vlm.py --config %CONFIG% --stage %STAGE%

echo 训练完成!
pause
