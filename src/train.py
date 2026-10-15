"""Fine-tuning local LoRA do Qwen2.5-0.5B-Instruct, exigindo GPU CUDA."""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

from common import MODEL_NAME, make_user_content
from data import generate_examples


class PromptMaskedDataset(torch.utils.data.Dataset):
    """Causal-LM dataset que calcula loss apenas sobre a resposta do assistente."""
    def __init__(self, records, tokenizer, max_length: int):
        self.rows = []
        for row in records:
            user = {"role": "user", "content": make_user_content(row["input"])}
            assistant = {"role": "assistant", "content": json.dumps(row["target"], ensure_ascii=False, separators=(",", ":"))}
            prefix = tokenizer.apply_chat_template([user], tokenize=False, add_generation_prompt=True)
            complete = tokenizer.apply_chat_template([user, assistant], tokenize=False, add_generation_prompt=False)
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            full_ids = tokenizer(complete, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
            prefix_len = min(len(prefix_ids), len(full_ids))
            labels = full_ids.copy()
            labels[:prefix_len] = [-100] * prefix_len
            if all(token == -100 for token in labels):
                continue
            self.rows.append({"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels})

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/qwen-order-lora")
    parser.add_argument("--data", default="data/orders_synthetic.jsonl")
    parser.add_argument("--examples", type=int, default=800)
    parser.add_argument("--epochs", type=float, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--seed", type=int, default=1660)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("ERRO: CUDA não está disponível. Este treino exige GPU; confira python src/gpu_check.py.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    device_name = torch.cuda.get_device_name(0)
    print(f"GPU ativa: {device_name} | CUDA runtime PyTorch: {torch.version.cuda}")
    print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GiB")

    data_path = Path(args.data)
    if data_path.exists():
        records = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        records = generate_examples(args.examples, args.seed)
        data_path.parent.mkdir(parents=True, exist_ok=True)
        data_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    if len(records) < 20:
        raise SystemExit("O conjunto precisa conter pelo menos 20 exemplos.")
    random.Random(args.seed).shuffle(records)
    train_end = max(1, int(len(records) * 0.8))
    validation_end = max(train_end + 1, int(len(records) * 0.9))
    train_rows, eval_rows, test_rows = records[:train_end], records[train_end:validation_end], records[validation_end:]
    print(f"Exemplos: {len(train_rows)} treino / {len(eval_rows)} validação / {len(test_rows)} teste isolado")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.rank,
        lora_alpha=2 * args.rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    train_dataset = PromptMaskedDataset(train_rows, tokenizer, args.max_length)
    eval_dataset = PromptMaskedDataset(eval_rows, tokenizer, args.max_length)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    test_path = output_dir.parent / "evaluation_test.jsonl"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in test_rows), encoding="utf-8")
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        fp16=True,
        gradient_checkpointing=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to="none",
        dataloader_num_workers=0,
        optim="adamw_torch",
        seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
    )
    result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    metadata = {
        "base_model": MODEL_NAME,
        "method": "supervised fine-tuning with LoRA (adapter only)",
        "gpu": device_name,
        "cuda_runtime": torch.version.cuda,
        "gpu_compute_capability": list(torch.cuda.get_device_capability(0)),
        "gpu_total_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "examples_total": len(records),
        "examples_train": len(train_rows),
        "examples_validation": len(eval_rows),
        "examples_test": len(test_rows),
        "held_out_test_file": str(test_path),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lora_rank": args.rank,
        "max_length": args.max_length,
        "seed": args.seed,
        "train_metrics": result.metrics,
        "completed_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Adapter e metadados gravados em: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
