"""Diagnóstico reproduzível de CUDA/GPU; pode salvar uma cópia textual para o relatório."""
from __future__ import annotations

import argparse
import io
import json
import subprocess
import sys
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path


def run_check() -> dict:
    import torch
    print("=== DIAGNÓSTICO GPU / CUDA ===")
    print("Horário UTC:", datetime.now(timezone.utc).isoformat())
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("CUDA disponível no PyTorch:", torch.cuda.is_available())
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA indisponível: verifique driver NVIDIA e instalação CUDA do PyTorch.")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    print("GPU:", props.name)
    print("Compute capability:", f"{props.major}.{props.minor}")
    print("VRAM total (GiB):", round(props.total_memory / 1024**3, 2))
    print("CUDA runtime (PyTorch):", torch.version.cuda)
    a = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    b = torch.randn((1024, 1024), device="cuda", dtype=torch.float16)
    c = a @ b
    torch.cuda.synchronize()
    print("Teste FP16 CUDA: PASSOU")
    print("Formato do resultado:", list(c.shape), "| device:", c.device, "| finito:", bool(torch.isfinite(c).all().item()))
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=15)
        print("\n=== nvidia-smi ===")
        print((result.stdout or result.stderr).strip())
        if result.returncode != 0:
            print("Aviso: nvidia-smi retornou código", result.returncode)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        print("Aviso: nvidia-smi não pôde ser chamado:", str(exc))
    return {"gpu": props.name, "cuda_available": True, "cuda_runtime": torch.version.cuda, "compute_capability": f"{props.major}.{props.minor}", "vram_gib": round(props.total_memory / 1024**3, 2), "fp16_matmul": "PASSOU"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", help="Grava o relatório textual e JSON neste arquivo (por exemplo docs/gpu_check.txt).")
    args = parser.parse_args()
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            data = run_check()
        report = buffer.getvalue()
        status = 0
    except Exception as exc:
        report = buffer.getvalue() + f"\nFALHA: {exc}\n"
        data = {"cuda_available": False, "error": str(exc)}
        status = 1
    print(report, end="")
    if args.save:
        path = Path(args.save)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(report + "\nJSON:\n" + json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Relatório salvo em:", path.resolve())
    raise SystemExit(status)


if __name__ == "__main__":
    main()
