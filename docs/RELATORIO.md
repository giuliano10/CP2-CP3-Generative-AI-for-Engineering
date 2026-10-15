# Relatório técnico — Extração local de pedidos com Qwen + LoRA

## Resumo

Este projeto adapta o modelo aberto `Qwen/Qwen2.5-0.5B-Instruct` para transformar pedidos curtos em português num objeto JSON com `produto`, `quantidade`, `endereco` e `prazo`. O ajuste é supervisionado e usa LoRA; inferência, treinamento e benchmark foram implementados para CUDA. O projeto gera dados sintéticos determinísticos para evitar dependência de dados pessoais.

**Importante:** este arquivo é um relatório metodológico. Os campos de resultados devem ser preenchidos após execução no computador com GTX 1660. Nenhuma métrica nem captura da GPU foi inventada ou medida no Sandbox.

## Problema e hipótese

**Problema:** extrair quatro campos de um pedido escrito em linguagem natural e devolver JSON válido. Campos não mencionados devem ser `null`.

**Hipótese:** um adapter LoRA treinado em exemplos sintéticos da tarefa aumentará, em relação ao modelo base, a acurácia exata de campos e a taxa de JSON válido no conjunto de teste. A hipótese é avaliada, não presumida como verdadeira.

## Modelo e treinamento

- Base: Qwen2.5-0.5B-Instruct (`Qwen/Qwen2.5-0.5B-Instruct`).
- Ajuste: supervised fine-tuning com LoRA, `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, projeções de atenção/MLP.
- Otimizador: AdamW PyTorch; LR inicial padrão `2e-4`, scheduler cosine, warmup 5%.
- Precisão/dispositivo: FP16 em CUDA, batch 1, acumulação 8, gradient checkpointing.
- Loss: mascarada no prefixo da instrução; aprende a produzir a resposta JSON.
- Split padrão: 80% treino, 10% validação, 10% teste isolado, seed 1660.
- Dados: 800 exemplos sintéticos padrão, sem dados pessoais.

O adapter é treinado; o checkpoint integral do modelo base não é alterado nem republicado pelo projeto. Na execução, registre as versões reais de Python, PyTorch, CUDA, Transformers e o modelo/driver que `nvidia-smi` reportar.

## Protocolo de avaliação

1. Executar `python src/gpu_check.py --save docs/gpu_check.txt`; confirmar GPU e multiplicação matricial FP16.
2. Executar `python src/train.py` na GPU; o script cria `outputs/evaluation_test.jsonl` com os 10% reservados para teste e preserva `outputs/qwen-order-lora/training_metadata.json` localmente.
3. Executar `python src/evaluate.py --max-examples 60 --seed 8821` na mesma máquina; a avaliação lê somente o arquivo de teste isolado.
4. Comparar baseline e adapter em exemplos idênticos para greedy, temperatura 0,2 e temperatura 0,7 (`top_p=0.9`).
5. Consultar `outputs/evaluation/summary.csv`, auditar amostras em `predictions.jsonl` e registrar exceções/erros.

### Métricas

- **Taxa de parse JSON:** fração de respostas interpretadas como objeto JSON.
- **Acurácia exata do registro:** fração em que os quatro campos correspondem exatamente ao alvo.
- **Acurácia por campo:** fração correta separadamente para produto, quantidade, endereço e prazo; igualdade de valor, incluindo `null`.
- **Média dos campos:** média aritmética das quatro acurácias por campo.

A avaliação exata é estrita; variações semanticamente equivalentes podem contar como erro. O conjunto sintético é pequeno e não demonstra generalização para atendimento real.

## Resultados medidos na máquina local

Preencher depois de executar. **Não copiar números esperados nem preencher por estimativa.**

- Data/hora da execução:
- GPU (copiar de `nvidia-smi` e `gpu_check.txt`):
- VRAM:
- Driver / CUDA / PyTorch / Python:
- Número de exemplos e seed:
- Parâmetros do treinamento:
- Tempo total de treino:
- Pico de memória GPU, se medido:
- Print autêntico: `CAPTURA_GPU_VS_CODE.png` (adicionar ao repositório após a captura)

| Modelo | Parâmetro geração | JSON parseável | Exatidão dos 4 campos | Produto | Quantidade | Endereço | Prazo |
|---|---|---:|---:|---:|---:|---:|---:|
| Baseline | greedy | — | — | — | — | — | — |
| Baseline | temp 0,2 | — | — | — | — | — | — |
| Baseline | temp 0,7 | — | — | — | — | — | — |
| LoRA | greedy | — | — | — | — | — | — |
| LoRA | temp 0,2 | — | — | — | — | — | — |
| LoRA | temp 0,7 | — | — | — | — | — | — |

## Discussão a preencher

Compare se o adapter supera o baseline, quais campos são mais difíceis, o efeito da temperatura na consistência do JSON, eventuais respostas incorretas e limitações do conjunto sintético. Temperaturas maiores podem aumentar variação; não assuma melhoria sem observar as métricas.

## Reprodutibilidade e artefatos

- Seed padrão do gerador/treino: 1660.
- Seed da avaliação: 8821.
- Dados: `data/orders_synthetic.jsonl`, gerados por `src/data.py`; 10% de teste não são usados no treino.
- Resumo de avaliação: `outputs/evaluation/summary.csv` e `.json`.
- Predições auditáveis: `outputs/evaluation/predictions.jsonl`.
- Metadados de treino: `outputs/qwen-order-lora/training_metadata.json`.

Os outputs e os pesos são ignorados pelo Git. Coloque no GitHub o código, a documentação, os testes e somente dados sintéticos, nunca credenciais ou documentos pessoais.
