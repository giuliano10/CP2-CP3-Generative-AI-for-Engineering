$ErrorActionPreference = "Stop"

py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Write-Host "Instalacao concluida. Ative .venv e valide com: python src/gpu_check.py"
