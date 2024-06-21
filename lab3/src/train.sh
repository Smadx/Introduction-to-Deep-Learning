accelerate launch --config_file accelerate_config.yaml train.py \
    --dataset cora \
    --num-layers 3 \
    --pair-norm-scale 1.0 \
    --dropedge-prob 0.1 \
    --results-path ../results/test1/ \
    --epochs 20 \
    >test1.log