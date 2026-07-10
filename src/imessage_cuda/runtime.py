from __future__ import annotations

import torch


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower()
    if normalized == "auto":
        normalized = "cuda" if torch.cuda.is_available() else "cpu"
    if normalized == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot see an NVIDIA GPU. Run `imessage-cuda doctor` "
            "and install an NVIDIA-enabled PyTorch build."
        )
    if normalized not in {"cuda", "cpu"}:
        raise ValueError("device must be 'auto', 'cuda', or 'cpu'")
    return torch.device(normalized)
