"""Benchmark CUDA: baseline vs LoRA e comparação de parâmetros de geração."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import FIELDS, MODEL_NAME, extract_json, make_user_content
PARAMS = [
    {"name": "greedy", "do_sample": False, "temperature": None, "top_p": None},
    {"name": "temp_0.2", "do_sample": True, "temperature": 0.2, "top_p": 0.9},
    {"name": "temp_0.7", "do_sample": True, "temperature": 0.7, "top_p": 0.9},
]


def load_model(adapter_path: str | None):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_one(model, tokenizer, text: str, params: dict) -> str:
    messages = [{"role": "user", "content": make_user_content(text)}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    kwargs = {"max_new_tokens": 112, "do_sample": params["do_sample"], "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if params["do_sample"]:
        kwargs.update(temperature=params["temperature"], top_p=params["top_p"])
    with torch.inference_mode():
        output = model.generate(inputs, **kwargs)
    return tokenizer.decode(output[0, inputs.shape[1]:], skip_special_tokens=True).strip()


def score_rows(rows: list[dict]) -> dict:
    n = len(rows)
    per_field = {field: sum(row["parsed"] is not None and field in row["parsed"] and row["parsed"][field] == row["expected"].get(field) for row in rows) / n for field in FIELDS}
    exact = sum(row["parsed"] is not None and set(row["parsed"]) == set(FIELDS) and all(row["parsed"][field] == row["expected"].get(field) for field in FIELDS) for row in rows) / n
    parse_rate = sum(row["parsed"] is not None for row in rows) / n
    return {"n": n, "json_parse_rate": round(parse_rate, 4), "exact_record_accuracy": round(exact, 4), "field_accuracy": {k: round(v, 4) for k, v in per_field.items()}, "mean_field_accuracy": round(sum(per_field.values()) / len(per_field), 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", default="outputs/qwen-order-lora")
    parser.add_argument("--data", default="outputs/evaluation_test.jsonl")
    parser.add_argument("--max-examples", type=int, default=60)
    parser.add_argument("--seed", type=int, default=8821)
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--no-baseline", action="store_true", help="Avalia somente o adapter, sem carregar o baseline.")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("ERRO: avaliação oficial exige CUDA; confira python src/gpu_check.py.")
    print(f"GPU de avaliação: {torch.cuda.get_device_name(0)}")
    path = Path(args.data)
    if not path.exists():
        raise SystemExit(f"Conjunto de teste isolado não encontrado: {path}. Execute primeiro python src/train.py para gravar o split de teste.")
    records = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    # A ordem/seed do split é produzida no treino; usar as primeiras N linhas é reprodutível.
    records = records[:args.max_examples]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict] = []
    model_specs = [("finetuned", args.adapter)]
    if not args.no_baseline:
        model_specs.insert(0, ("baseline", None))
    for model_name, adapter in model_specs:
        if adapter and not Path(adapter).exists():
            raise SystemExit(f"Adapter não encontrado: {adapter}. Execute primeiro python src/train.py.")
        model, tokenizer = load_model(adapter)
        for params in PARAMS:
            print(f"Avaliando {model_name} / {params['name']} ({len(records)} exemplos)...")
            for i, record in enumerate(records):
                torch.manual_seed(args.seed + i)
                torch.cuda.manual_seed_all(args.seed + i)
                raw = generate_one(model, tokenizer, record["input"], params)
                all_rows.append({
                    "model": model_name,
                    "parameters": params["name"],
                    "example_id": i,
                    "input": record["input"],
                    "expected": record["target"],
                    "raw_output": raw,
                    "parsed": extract_json(raw),
                })
        del model
        torch.cuda.empty_cache()
    summaries = []
    for model_name, adapter in model_specs:
        for params in PARAMS:
            subset = [r for r in all_rows if r["model"] == model_name and r["parameters"] == params["name"]]
            summaries.append({"model": model_name, "parameters": params["name"], **score_rows(subset)})
    (output_dir / "summary.json").write_text(json.dumps({"gpu": torch.cuda.get_device_name(0), "seed": args.seed, "summaries": summaries}, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "parameters", "n", "json_parse_rate", "exact_record_accuracy", "mean_field_accuracy", *[f"field_{field}" for field in FIELDS]])
        writer.writeheader()
        for row in summaries:
            writer.writerow({**row, **{f"field_{field}": row["field_accuracy"][field] for field in FIELDS}})
    with (output_dir / "predictions.jsonl").open("w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Relatórios: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
