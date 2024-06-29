import argparse
import utils
import evaluate
import torch
import json
import dataclasses
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import LlamaForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup, BitsAndBytesConfig
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from bitsandbytes.optim import AdamW8bit
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()

    # File paths
    parser.add_argument("--model_dir", type=str, default=None)
    parser.add_argument("--tokenizer_dir", type=str, default=None)
    parser.add_argument("--results_path", type=str, default=None)

    # Peft arguments
    parser.add_argument("--task_type", type=TaskType, default=TaskType.SEQ_CLS)
    parser.add_argument("--task", type=str, default="mrpc")
    parser.add_argument("--num_classes", type=int, default=2)

    # LoRA arguments
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)

    # Quantization arguments
    parser.add_argument("--load_in_4bit", type=bool, default=True)
    parser.add_argument("--bnb_4bit_use_double_quant", type=bool, default=True)
    parser.add_argument("--bnb_4bit_quant_type", type=str, default='nf4')
    parser.add_argument("--bnb_4bit_compute_dtype", type=torch.dtype, default=torch.float16)

    # Training arguments
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()

    # Accelerator
    set_seed(args.seed)
    accelerator = Accelerator(split_batches=True)

    # Config
    utils.init_logger(accelerator)
    cfg = utils.init_config_from_args(utils.TrainConfig, args)

    quant_config = BitsAndBytesConfig(
        load_in_4bit=cfg.load_in_4bit,
        bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
        bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
        bnb_4bit_compute_dtype=cfg.bnb_4bit_compute_dtype,
    )

    results_path= utils.handle_results_path(cfg.results_path)
    results_path.mkdir(parents=True, exist_ok=True)

    with open(args.results_path + '/config.json', 'w') as json_file:
        json.dump(dataclasses.asdict(cfg), json_file, indent=4, cls=utils.CustomEncoder)

    # Load model and tokenizer
    model = LlamaForSequenceClassification.from_pretrained(cfg.model_dir, num_labels=cfg.num_classes, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_dir)
        
    tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.eos_token_id

    model = prepare_model_for_kbit_training(model)

    # Load data
    datasets = load_dataset("glue", cfg.task)
    metric = evaluate.load("glue", cfg.task)

    train_loader, val_loader, _ = utils.make_dataset(datasets, tokenizer, cfg.batch_size, cfg.prompt)

    loraconfig = LoraConfig(
        task_type=cfg.task_type,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
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

    peft_model, train_loader, val_loader, optimizer, lr_scheduler = accelerator.prepare(
        peft_model, train_loader, val_loader, optimizer, lr_scheduler,
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
        for step, batch in enumerate(tqdm(val_loader)):
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
        unwrapped_model.save_pretrained(cfg.results_path + "/llama3_for_cls_lora_weights")
        quant_config.to_json_file(args.results_path + '/quant_config.json')
        lora_json = loraconfig.to_dict()
        with open(args.results_path + '/lora_config.json', 'w') as json_file:
            json.dump(lora_json, json_file, indent=4, cls=utils.CustomEncoder)

if __name__ == "__main__":
    main()