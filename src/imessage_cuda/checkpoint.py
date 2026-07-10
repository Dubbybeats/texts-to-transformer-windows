from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file

from imessage_cuda.model.config import ModelConfig
from imessage_cuda.model.transformer import TransformerLM
from imessage_cuda.runtime import resolve_device
from imessage_cuda.utils import ensure_private_dir, write_json


def save_checkpoint(
    destination: str | Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
    trainer_state: dict[str, Any],
    training_config: dict[str, Any],
    tokenizer_dir: str | Path | None = None,
) -> Path:
    destination_path = Path(destination)
    ensure_private_dir(destination_path.parent)
    temporary_path = Path(
        tempfile.mkdtemp(prefix=f".{destination_path.name}.", dir=destination_path.parent)
    )
    try:
        cpu_state = {
            name: tensor.detach().cpu().contiguous() for name, tensor in model.state_dict().items()
        }
        save_file(cpu_state, temporary_path / "model.safetensors", metadata={"format": "pt"})
        torch.save(optimizer.state_dict(), temporary_path / "optimizer.pt")
        random_state: dict[str, Any] = {"cpu": torch.random.get_rng_state()}
        if torch.cuda.is_available():
            random_state["cuda"] = torch.cuda.get_rng_state_all()
        torch.save(random_state, temporary_path / "random-state.pt")
        write_json(temporary_path / "model-config.json", model.config.to_dict())
        write_json(temporary_path / "training-config.json", training_config)
        (temporary_path / "training-config.yaml").write_text(
            yaml.safe_dump(training_config, sort_keys=True), encoding="utf-8"
        )
        write_json(temporary_path / "trainer-state.json", trainer_state)
        if tokenizer_dir is not None:
            shutil.copytree(tokenizer_dir, temporary_path / "tokenizer")
        if destination_path.exists():
            shutil.rmtree(destination_path)
        os.replace(temporary_path, destination_path)
        return destination_path
    finally:
        shutil.rmtree(temporary_path, ignore_errors=True)


def load_model(checkpoint: str | Path, device: str = "auto") -> TransformerLM:
    checkpoint_path = Path(checkpoint)
    with (checkpoint_path / "model-config.json").open(encoding="utf-8") as handle:
        config = ModelConfig.from_dict(json.load(handle))
    target = resolve_device(device)
    model = TransformerLM(config)
    model.load_state_dict(load_file(checkpoint_path / "model.safetensors"), strict=True)
    model.to(target)
    model.eval()
    return model


def restore_training_state(
    checkpoint: str | Path,
    model: TransformerLM,
    optimizer: torch.optim.Optimizer,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    model.load_state_dict(load_file(checkpoint_path / "model.safetensors"), strict=True)
    optimizer.load_state_dict(
        torch.load(checkpoint_path / "optimizer.pt", map_location=model.device, weights_only=True)
    )
    random_state_path = checkpoint_path / "random-state.pt"
    if random_state_path.exists():
        random_state = torch.load(random_state_path, map_location="cpu", weights_only=True)
        torch.random.set_rng_state(random_state["cpu"])
        if model.device.type == "cuda" and "cuda" in random_state:
            torch.cuda.set_rng_state_all(random_state["cuda"])
    with (checkpoint_path / "trainer-state.json").open(encoding="utf-8") as handle:
        return json.load(handle)
