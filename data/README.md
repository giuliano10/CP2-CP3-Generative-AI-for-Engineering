# Dados sintéticos

`orders_synthetic.jsonl` contém 800 exemplos gerados deterministicamente por `src/data.py` com seed 1660. Os pedidos e endereços foram construídos a partir de combinações sintéticas para a tarefa; não são dados pessoais. Cada linha contém `input` e o objeto JSON `target`.

Para regenerar o arquivo:

```powershell
python src/data.py --count 800 --seed 1660 --output data/orders_synthetic.jsonl
```

A rotina de treino embaralha esse conjunto com seed 1660 e separa 80%/10%/10% em treino, validação e teste. O conjunto de teste é salvo localmente em `outputs/evaluation_test.jsonl` e não deve ser adicionado ao Git junto com artefatos de treino.
