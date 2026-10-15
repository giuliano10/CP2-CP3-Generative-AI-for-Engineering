# Extrator local de pedidos em português — Qwen + LoRA + GTX 1660

Projeto didático e reproduzível para **adaptar localmente uma LLM** à extração de campos de pedidos escritos em português e retornar JSON. O treinamento, a avaliação oficial e a interface são configurados para rodar em uma NVIDIA com CUDA, incluindo a GTX 1660.

> **Importante sobre o escopo real:** não é treinamento de todos os pesos do zero. O projeto baixa o modelo base `Qwen/Qwen2.5-0.5B-Instruct` e treina um adapter pequeno com **LoRA** (fine-tuning supervisionado). Isso é mais viável numa GTX 1660 de 6 GB, permite manter o modelo base congelado e gera pesos próprios do projeto. Os pesos resultantes ficam no computador local e não são enviados ao GitHub.

## Problema escolhido

Dado um texto de pedido, extrair quatro campos:

```json
{"produto":"café","quantidade":3,"endereco":"Av. Paulista, 1000, ap. 12, São Paulo - SP","prazo":"sexta-feira"}
```

Chaves fixas: `produto` (texto), `quantidade` (inteiro), `endereco` (texto) e `prazo` (texto). Para um campo ausente, o modelo deve usar `null` e não inventar informação. O escopo é propositalmente limitado e avaliável automaticamente por acerto exato de cada campo.

## Modelo e método

- **Modelo base:** [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), aproximadamente 0,5 bilhão de parâmetros.
- **Método:** supervised fine-tuning com **LoRA**, rank padrão 8; o adapter aprendido é salvo em `outputs/qwen-order-lora/`.
- **Precisão:** FP16 na GPU CUDA; não usa bitsandbytes nem quantização 4-bit.
- **Dados:** exemplos sintéticos em português, gerados deterministicamente; nenhum dado pessoal real.
- **GPU:** o treino e o benchmark param com erro se CUDA não estiver disponível. A checagem também faz multiplicação matricial FP16 na GPU.

## Arquivos

- `src/data.py`: geração do conjunto sintético.
- `src/train.py`: treino LoRA com loss somente na resposta esperada.
- `src/evaluate.py`: comparação baseline/adaptado, métricas e parâmetros de geração.
- `src/app.py`: frontend local em Gradio para inserir prompts.
- `src/gpu_check.py`: identifica a GPU e verifica cálculo CUDA FP16.
- `tests/`: testes rápidos do conjunto e parser; não precisam baixar o modelo.
- `docs/RELATORIO.md`: metodologia e formato do relatório.
- `docs/CAPTURA_GPU.md`: passos para o print real no VS Code.

## Pré-requisitos

1. Windows 10/11 de 64 bits, Python 3.10 ou 3.11 e VS Code com extensão Python.
2. Driver NVIDIA recente para a GTX 1660. Confirme no terminal do VS Code com `nvidia-smi`.
3. Espaço livre para o modelo base e os pacotes; o modelo é baixado do Hugging Face na primeira execução.
4. Recomenda-se fechar aplicações que ocupem VRAM. Os padrões são batch 1, acumulação de gradiente 8, FP16 e sequência máxima 384.

> Uma GTX 1660 normalmente tem 6 GB de VRAM, mas existem variações de placa. Confira a VRAM real com `nvidia-smi`. O tamanho do modelo e o consumo dependem da versão do driver, do PyTorch e do uso de memória no sistema; se houver OOM, reduza `--max-length` para 256 e/ou `--rank` para 4.

## Instalação no Windows / VS Code

Abra a pasta do projeto no VS Code e use o terminal integrado **PowerShell**:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

Se o PowerShell bloquear a ativação, use `Set-ExecutionPolicy -Scope Process Bypass` no terminal atual e ative novamente. No seletor de interpretador do VS Code, escolha `.venv`.

**A instalação do PyTorch varia com driver e versão do Python.** Se o comando CUDA acima não corresponder à configuração suportada no seu computador, siga o seletor oficial do [PyTorch](https://pytorch.org/get-started/locally/) e escolha Windows / Pip / Python / CUDA. Depois confirme com `python src/gpu_check.py`.

## Reproduzir o projeto

Execute os comandos na raiz do repositório, com `.venv` ativado.

### 1. Conferir a GPU e capturar diagnóstico

```powershell
nvidia-smi
python src/gpu_check.py --save docs/gpu_check.txt
```

O teste deve exibir o nome da NVIDIA, `CUDA disponível no PyTorch: True` e `Teste FP16 CUDA: PASSOU`. Consulte `docs/CAPTURA_GPU.md` para tirar a captura de tela autêntica no terminal do VS Code.

### 2. Gerar / conferir dados sintéticos

```powershell
python src/data.py --count 800 --seed 1660 --output data/orders_synthetic.jsonl
```

O repositório já contém uma amostra reproduzível. O script de treino também cria os dados automaticamente caso o arquivo esteja ausente.

### 3. Treinar o adapter na GPU

```powershell
python src/train.py --epochs 4 --learning-rate 0.0002 --rank 8 --max-length 384
```

O treino exige CUDA e termina com erro caso a GPU não esteja disponível. Configurações ajustáveis: `--epochs`, `--learning-rate`, `--rank`, `--max-length`, `--examples` e `--seed`. Os artefatos vão a `outputs/qwen-order-lora/`; `training_metadata.json` registra GPU, parâmetros e métricas de treino/validação.

### 4. Avaliar respostas e comparar parâmetros

```powershell
python src/evaluate.py --max-examples 60 --seed 8821
```

O benchmark compara o modelo base (`baseline`) e o adapter (`finetuned`) nas mesmas entradas, com:

- `greedy`: sem amostragem;
- `temp_0.2`: amostragem, temperatura 0,2, `top_p=0.9`;
- `temp_0.7`: amostragem, temperatura 0,7, `top_p=0.9`.

O treino separa 80% para treino, 10% para validação e 10% para teste isolado; salva o teste em `outputs/evaluation_test.jsonl`, fora dos dados usados no ajuste. As métricas em `outputs/evaluation/summary.csv` e `summary.json` são: taxa de JSON parseável, acurácia exata dos quatro campos, acurácia média por campo e acurácia individual de produto, quantidade, endereço e prazo. `predictions.jsonl` guarda entradas, referência, resposta bruta e parse.

A amostra de avaliação usa as primeiras N linhas do split de teste isolado e é fixa para permitir comparar parâmetros. Registre quantidade/seed no relatório; a seed de geração é salva em `training_metadata.json`.

### 5. Abrir o frontend local

```powershell
python src/app.py
```

A interface abre em `http://127.0.0.1:7860`. Digite ou selecione um pedido, ajuste amostragem/temperatura e veja o JSON e a resposta bruta. A interface exige CUDA. Se o adapter ainda não existir, informa explicitamente que está usando o baseline.

### 6. Testes de software

```powershell
pytest -q
```

Esses testes verificam o gerador e o parser, não substituem o treino/benchmark na GPU.

## Ajustes para a GTX 1660

A GTX 1660 usa CUDA, mas não possui Tensor Cores de precisão mista da mesma classe das placas RTX. O projeto usa FP16 por compatibilidade e economia de memória, porém a velocidade pode ser menor. Se ocorrer falta de VRAM, tente em ordem:

```powershell
python src/train.py --max-length 256 --rank 4 --epochs 3
```

Feche programas que usam GPU; não force execução em CPU para declarar cumprida a etapa CUDA. Para o benchmark, reduza `--max-examples 20` enquanto valida a instalação. Alterar parâmetros precisa ser registrado no relatório.

## O que enviar ao GitHub

Envie o código, o README, a documentação, os testes e os dados sintéticos. **Não envie** `.venv/`, o modelo base baixado, o adapter/checkpoints em `outputs/` nem credenciais. O `.gitignore` já exclui os pesos e resultados locais.

Depois de criar um repositório vazio no GitHub, execute na pasta:

```powershell
git init
git add README.md LICENSE .gitignore requirements.txt setup_windows.ps1 src tests docs data .vscode
git commit -m "Projeto LLM local para extracao de pedidos com LoRA"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/SEU-REPOSITORIO.git
git push -u origin main
```

Substitua a URL pelo repositório que você criar. Antes do commit, atualize `docs/RELATORIO.md` com resultados **medidos na sua GPU** e coloque o print real descrito em `docs/CAPTURA_GPU.md`.

## Limitações e uso responsável

- O corpus é sintético e simples; resultados não garantem desempenho em pedidos reais, texto fora de distribuição ou endereço ambíguo.
- O modelo pode inventar/corromper campos; valide JSON e campos antes de qualquer automação comercial.
- Não use dados pessoais reais para este exercício. Dados de endereço neste repositório são exemplos fictícios/sintéticos.
- O treino adapta o formato e a tarefa; não é treinamento de uma LLM do zero nem prova de robustez em produção.
