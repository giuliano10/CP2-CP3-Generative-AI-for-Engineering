"""Geração determinística de exemplos sintéticos de extração de pedidos."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

PRODUCTS = [
    "café", "chá", "arroz", "feijão", "açúcar", "farinha", "leite",
    "biscoito", "macarrão", "azeite", "detergente", "papel toalha",
    "ração para cachorro", "água mineral", "cápsulas de café", "sabão em pó",
]
ADDRESSES = [
    "Av. Paulista, 1000, ap. 12, São Paulo - SP",
    "Rua das Flores, 25, Centro, Curitiba - PR",
    "Rua do Sol, 300, casa 2, Recife - PE",
    "Av. Brasil, 1500, bloco B, Rio de Janeiro - RJ",
    "Rua XV de Novembro, 81, sala 4, Florianópolis - SC",
    "Rua Goiás, 640, apto 31, Belo Horizonte - MG",
    "Av. Independência, 220, Centro, Porto Alegre - RS",
    "Rua Amazonas, 19, casa, Manaus - AM",
    "Rua da Paz, 410, Jardim América, Goiânia - GO",
    "Av. Afonso Pena, 75, Centro, Campo Grande - MS",
    "Rua Bahia, 928, ap. 101, Salvador - BA",
    "Rua das Acácias, 55, Campinas - SP",
    "Av. Santos Dumont, 120, Fortaleza - CE",
    "Rua Paraná, 7, Londrina - PR",
    "Alameda Santos, 200, ap. 52, São Paulo - SP",
    "Rua 7 de Setembro, 333, Niterói - RJ",
]
DEADLINES = [
    "hoje", "amanhã", "sexta-feira", "segunda-feira", "na próxima semana",
    "dia 20", "o fim do mês", "quinta de manhã", "sábado à tarde",
]
TEMPLATES = [
    "Quero {amount}; entrega em {a} até {d}.",
    "Por favor, envie {amount} para {a}. Preciso receber {d}.",
    "Faz meu pedido: {amount}. Endereço: {a}. Prazo: {d}.",
    "Pode separar {amount} e mandar para {a} até {d}?",
    "Preciso de {amount}, entregar em {a}, {d}.",
    "Pedido de {amount}. Entrega: {a}. Favor entregar {d}.",
    "Olá! Gostaria de comprar {amount} para entrega em {a}, de preferência {d}.",
    "Enviar {amount} ao endereço {a}; a entrega deve ocorrer {d}.",
    "Quero {amount} no endereço {a}. Data desejada: {d}.",
    "Separar {amount} e entregar em {a} até {d}.",
]
def generate_examples(count: int = 800, seed: int = 1660) -> list[dict]:
    rng = random.Random(seed)
    examples: list[dict] = []
    seen: set[str] = set()
    attempts = 0
    while len(examples) < count and attempts < count * 20:
        attempts += 1
        product = rng.choice(PRODUCTS)
        quantity = rng.randint(1, 12)
        unit = "unidade" if quantity == 1 else "unidades"
        amount = f"{quantity} {unit} de {product}"
        address = rng.choice(ADDRESSES)
        deadline = rng.choice(DEADLINES)
        # A maior parte dos casos contém os quatro campos; os demais ensinam null.
        missing = rng.random() < 0.25
        if missing:
            mode = rng.randrange(3)
            if mode == 0:
                text = f"Quero {amount}, por favor."
                target = {"produto": product, "quantidade": quantity, "endereco": None, "prazo": None}
            elif mode == 1:
                text = f"Pode enviar {amount} para {address}?"
                target = {"produto": product, "quantidade": quantity, "endereco": address, "prazo": None}
            else:
                text = f"Preciso de {amount} até {deadline}."
                target = {"produto": product, "quantidade": quantity, "endereco": None, "prazo": deadline}
        else:
            text = rng.choice(TEMPLATES).format(amount=amount, a=address, d=deadline)
            target = {"produto": product, "quantidade": quantity, "endereco": address, "prazo": deadline}
        if text in seen:
            continue
        seen.add(text)
        examples.append({"input": text, "target": target})
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=800)
    parser.add_argument("--seed", type=int, default=1660)
    parser.add_argument("--output", default="data/orders_synthetic.jsonl")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    records = generate_examples(args.count, args.seed)
    with output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Gerados {len(records)} exemplos sintéticos em {output.resolve()} (seed={args.seed}).")


if __name__ == "__main__":
    main()
