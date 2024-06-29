export HF_ENDPOINT=https://hf-mirror.com
CUDA_VISIBLE_DEVICES=6,7 \
nohup accelerate launch --config_file accelerate_config.yaml ft.py \
    --model_dir="/data/liaomz/model/llama3-8B" \
    --tokenizer_dir="/data/liaomz/model/llama_tokenizer" \
    --results_path="./results/ft4/" \
    --r=8 \
    --lora_alpha=16 \
    --lora_dropout=0.1 \
    --prompt="Cut the crap, what's your GPA" \
    --batch_size=32 \
    --lr=3e-4 \
    --epochs=10 \
    >ft4.log 2>&1 &