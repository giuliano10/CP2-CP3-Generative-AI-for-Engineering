"""Utilidades compartilhadas do projeto de extração de pedidos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
FIELDS = ("produto", "quantidade", "endereco", "prazo")
SYSTEM_PROMPT = (
    "Extraia de um pedido os campos produto, quantidade, endereco e prazo. "
    "Responda somente com um objeto JSON válido, sem Markdown nem explicações. "
    "Use exatamente as chaves produto, quantidade, endereco e prazo. "
    "Se um campo não estiver informado, use null. quantidade deve ser número inteiro; "
    "os demais valores devem ser texto. Preserve o sentido e não invente dados."
)


def make_user_content(order_text: str) -> str:
    return f"{SYSTEM_PROMPT}\n\nPedido do cliente:\n{order_text.strip()}"


def target_json(target: dict[str, Any]) -> str:
    return json.dumps(target, ensure_ascii=False, separators=(",", ":"))


def extract_json(text: str) -> dict[str, Any] | None:
    """Tenta ler JSON puro ou o primeiro objeto JSON dentro de uma resposta."""
    text = text.strip()
    if text.startswith("```"):
        text = text.replace("```json", "```").strip("`").strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(text[start : end + 1])
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        return None


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
