export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=6,7 \
nohup accelerate launch --config_file accelerate_config.yaml eval.py \
    --results_path="./results/ft4/" \
    >eval4.log 2>&1 &