CUDA_VISIBLE_DEVICES=2,3 \
nohup accelerate launch --config_file accelerate_config.yaml train.py \
    --N 10000 \
    --results-path results/test5/ \
    >test5.log 2>&1 &