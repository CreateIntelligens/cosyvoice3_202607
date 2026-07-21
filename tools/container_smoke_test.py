#!/usr/bin/env python3
"""Verify the dependencies shared by CosyVoice3 training and inference."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys


def command_version(command: str) -> str:
    path = shutil.which(command)
    if path is None:
        raise RuntimeError(f"required command not found: {command}")
    result = subprocess.run(
        [path, "-version" if command == "ffmpeg" else "--version"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return result.stdout.splitlines()[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu-only", action="store_true", help="do not require a visible NVIDIA GPU")
    args = parser.parse_args()

    import deepspeed
    import matcha
    import numpy
    import onnxruntime
    import pynini
    import pyworld
    import soundfile
    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel  # noqa: F401
    from cosyvoice.dataset import processor  # noqa: F401

    if not args.cpu_only and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; verify GPU allocation and container GPU passthrough")

    result = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "deepspeed": deepspeed.__version__,
        "numpy": numpy.__version__,
        "onnxruntime": onnxruntime.__version__,
        "onnxruntime_providers": onnxruntime.get_available_providers(),
        "soundfile": soundfile.__version__,
        "pynini": getattr(pynini, "__version__", "import-ok"),
        "pyworld": getattr(pyworld, "__version__", "import-ok"),
        "matcha_module": matcha.__file__,
        "ffmpeg": command_version("ffmpeg"),
        "sox": command_version("sox"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
