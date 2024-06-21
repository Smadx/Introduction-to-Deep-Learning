accelerate launch --config_file accelerate_config.yaml train.py \
    --results_path "results/train1/" \
    >train1.log 2>&1 &