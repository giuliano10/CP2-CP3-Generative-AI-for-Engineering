"""Interface local para testar a extração de pedidos com CUDA."""
from __future__ import annotations

import json
import os
from functools import lru_cache

import gradio as gr
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import MODEL_NAME, extract_json, make_user_content

ADAPTER_DIR = os.environ.get("ORDER_LORA_ADAPTER", "outputs/qwen-order-lora")


@lru_cache(maxsize=1)
def load_pipeline():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível. Esta demonstração exige uma GPU NVIDIA ativa; execute python src/gpu_check.py.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, low_cpu_mem_usage=True).to("cuda")
    adapter_found = os.path.isdir(ADAPTER_DIR)
    if adapter_found:
        model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    return model, tokenizer, adapter_found


def predict(order_text: str, temperature: float, sample: bool):
    if not order_text or not order_text.strip():
        return "Digite um pedido para começar.", ""
    model, tokenizer, adapter_found = load_pipeline()
    messages = [{"role": "user", "content": make_user_content(order_text)}]
    input_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    params = {"max_new_tokens": 128, "do_sample": bool(sample), "pad_token_id": tokenizer.pad_token_id, "eos_token_id": tokenizer.eos_token_id}
    if sample:
        params["temperature"] = max(float(temperature), 0.01)
        params["top_p"] = 0.9
    with torch.inference_mode():
        generated = model.generate(input_ids, **params)
    raw = tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    parsed = extract_json(raw)
    status = f"GPU: {torch.cuda.get_device_name(0)} | Modelo: {MODEL_NAME} | Adapter LoRA: {'carregado' if adapter_found else 'não encontrado (baseline)'}"
    result = json.dumps(parsed, ensure_ascii=False, indent=2) if parsed is not None else "Não foi possível interpretar um JSON válido. Saída bruta abaixo."
    return f"{status}\n\n{result}", raw


with gr.Blocks(title="Extrator de Pedidos — Qwen + LoRA") as demo:
    gr.Markdown("# Extrator de pedidos em português\nDigite um pedido de exemplo. O modelo local tentará retornar produto, quantidade, endereço e prazo como JSON. **A inferência exige CUDA.**")
    order = gr.Textbox(label="Pedido", lines=4, placeholder="Quero 3 caixas de café, entrega na Rua X até sexta-feira.")
    with gr.Row():
        temperature = gr.Slider(0.1, 1.0, value=0.2, step=0.1, label="Temperatura (aplicada apenas com amostragem)")
        sample = gr.Checkbox(label="Ativar amostragem", value=False)
    run = gr.Button("Extrair campos", variant="primary")
    result = gr.Textbox(label="Resultado estruturado", lines=9)
    raw = gr.Textbox(label="Resposta bruta do modelo", lines=4)
    run.click(predict, inputs=[order, temperature, sample], outputs=[result, raw])
    gr.Examples(
        examples=[
            ["Quero 3 caixas de café, entrega na Av. Paulista, 1000, ap. 12, São Paulo - SP até sexta-feira."],
            ["Pode enviar 2 pacotes de arroz para Rua das Flores, 25, Centro, Curitiba - PR?"],
        ],
        inputs=order,
    )

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=True)
