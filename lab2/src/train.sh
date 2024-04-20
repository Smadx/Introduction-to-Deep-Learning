CUDA_VISIBLE_DEVICES=2 \
nohup accelerate launch --config_file accelerate_config.yaml train.py \
    --in-channels 128 \
    --batch-size 256 \
    --results-path results/test3/ \
    --epochs 30 \
    >test3.log 2>&1 &