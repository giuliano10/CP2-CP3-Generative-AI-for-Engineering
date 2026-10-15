# Captura real da GPU no VS Code

A GTX 1660 do usuário não está conectada ao Sandbox desta execução. Portanto, este projeto **não inclui uma captura fabricada**. Para registrar a placa real funcionando dentro do VS Code:

1. Abra a pasta do repositório no VS Code.
2. Abra o terminal integrado: **Terminal - Novo Terminal** e selecione PowerShell.
3. Ative o ambiente virtual: `\.venv\Scripts\Activate.ps1`.
4. Execute `nvidia-smi` e deixe visíveis a linha com `GeForce GTX 1660`, o driver e a memória.
5. Na mesma janela, execute `python src/gpu_check.py`. O resultado precisa mostrar `CUDA disponível no PyTorch: True` e `Teste FP16 CUDA: PASSOU`.
6. Tire uma captura com a ferramenta de recorte do Windows (`Win+Shift+S`), incluindo o título/janela do VS Code e as linhas relevantes do terminal.
7. Salve como `docs/CAPTURA_GPU_VS_CODE.png`.
8. Para além do print, gere o log verificável com `python src/gpu_check.py --save docs/gpu_check.txt`.
9. Confirme que a imagem foi salva e a abra para verificar legibilidade antes de adicioná-la ao GitHub.

A captura deve ser feita no computador com a GTX 1660; não remova nem altere o resultado do diagnóstico. O print prova a identificação e a operação CUDA FP16, mas o treinamento só fica comprovado pelo log de treino e `training_metadata.json` gerados após rodar `src/train.py`.
