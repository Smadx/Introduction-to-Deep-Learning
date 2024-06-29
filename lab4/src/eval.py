import argparse
import utils
import evaluate
import torch
import json
import dataclasses

from pathlib import Path
from torch.optim import AdamW
from torch.utils.data import DataLoader, ConcatDataset
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import LlamaForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm

parser = argparse.ArgumentParser()

# File paths
parser.add_argument("--results_path", type=str, default=None)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()

# Accelerator
set_seed(args.seed)
accelerator = Accelerator(split_batches=True)

results_path= utils.handle_results_path(args.results_path)

# Load config
utils.init_logger(accelerator)
with open(results_path / "config.json", "r") as f:
    config_data = json.load(f)

with open(results_path / "quant_config.json", "r") as f:
    quant_dict = json.load(f)

config_data['bnb_4bit_compute_dtype'] = utils.dtype_mapping[config_data['bnb_4bit_compute_dtype']]
cfg = utils.TrainConfig(**config_data)

quant_config = BitsAndBytesConfig.from_dict(quant_dict)

# Load model and tokenizer
model = LlamaForSequenceClassification.from_pretrained(cfg.model_dir, num_labels=cfg.num_classes, quantization_config=quant_config)
tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
    
tokenizer.pad_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.eos_token_id

model = prepare_model_for_kbit_training(model)

# Load data
datasets = load_dataset("glue", cfg.task)
metric = evaluate.load("glue", cfg.task)

train_loader, val_loader, test_loader = utils.make_dataset(datasets, tokenizer, cfg.batch_size, cfg.prompt)

def collate_fn(examples):
    return tokenizer.pad(examples, padding="longest", return_tensors="pt")

combined_dataset = ConcatDataset([train_loader.dataset, val_loader.dataset])
train_loader = DataLoader(combined_dataset, shuffle=True, collate_fn=collate_fn, batch_size=cfg.batch_size)

loraconfig = LoraConfig(
    task_type=cfg.task_type,
    inference_mode=False,
    r=cfg.r,
    lora_alpha=cfg.lora_alpha,
    lora_dropout=cfg.lora_dropout,
)

# Peft model
peft_model = get_peft_model(model, loraconfig)
peft_model.print_trainable_parameters()

optimizer = AdamW8bit(peft_model.parameters(), lr=cfg.lr)
lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=0.06 * (len(train_loader) * cfg.epochs),  # 6% of training steps
    num_training_steps=len(train_loader) * cfg.epochs,
)

peft_model, train_loader, test_loader, optimizer, lr_scheduler = accelerator.prepare(
    peft_model, train_loader, test_loader, optimizer, lr_scheduler,
)

# Training
for epoch in range(cfg.epochs):
    peft_model.train()
    for step, batch in enumerate(tqdm(train_loader)):
        batch = batch.to(accelerator.device)
    
        outputs = peft_model(**batch)
        loss = outputs.loss
        # Backward pass
        accelerator.backward(loss)
        
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    peft_model.eval()
    for step, batch in enumerate(tqdm(test_loader)):
        batch = batch.to(accelerator.device)
        with torch.no_grad():
            outputs = model(**batch)
            predictions = outputs.logits.argmax(dim=-1)
            references = batch["labels"]
            accelerator.gather(predictions)
            accelerator.gather(references)
            metric.add_batch(
                predictions=accelerator.gather(predictions),
                references=accelerator.gather(references),
            )

    eval_metric = metric.compute()
    print(f"epoch {epoch}:", eval_metric)

# Save model and config
if accelerator.is_main_process:
    unwrapped_model = accelerator.unwrap_model(peft_model)
    unwrapped_model.save_pretrained(cfg.results_path + "/llama3_for_cls_lora_weights_test")