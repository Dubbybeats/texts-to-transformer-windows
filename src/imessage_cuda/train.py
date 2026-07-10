from __future__ import annotations

import json
import math
import platform
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from imessage_cuda import __version__
from imessage_cuda.checkpoint import restore_training_state, save_checkpoint
from imessage_cuda.dataset import (
    batch_indices,
    build_me_target_mask,
    load_tokens,
    materialize_batch,
    materialize_mask,
    window_count,
)
from imessage_cuda.model.config import ModelConfig
from imessage_cuda.model.transformer import TransformerLM, causal_lm_loss, perplexity
from imessage_cuda.runtime import resolve_device
from imessage_cuda.tokenizer.train import load_tokenizer
from imessage_cuda.utils import ensure_private_dir, write_json


def learning_rate_at_step(
    step: int, total_steps: int, warmup_steps: int, peak: float, minimum: float
) -> float:
    if warmup_steps and step < warmup_steps:
        return peak * max(1, step + 1) / warmup_steps
    decay_steps = max(1, total_steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return minimum + (peak - minimum) * cosine


def _autocast(device: torch.device, enabled: bool):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return nullcontext()


def evaluate_loss(
    model: TransformerLM,
    tokens: np.ndarray,
    *,
    context_length: int,
    batch_size: int,
    mixed_precision: bool = False,
    target_mask: np.ndarray | None = None,
) -> float:
    model.eval()
    model_device = next(model.parameters()).device
    count = window_count(tokens, context_length)
    if not count:
        raise ValueError("Evaluation split has no complete context window")
    total_loss = 0.0
    total_examples = 0
    with torch.inference_mode():
        for start in range(0, count, batch_size):
            indices = np.arange(start, min(count, start + batch_size))
            inputs_np, targets_np = materialize_batch(tokens, indices, context_length)
            inputs = torch.from_numpy(inputs_np).long().to(model_device)
            targets = torch.from_numpy(targets_np).long().to(model_device)
            loss_mask = (
                torch.from_numpy(materialize_mask(target_mask, indices, context_length)).to(
                    model_device
                )
                if target_mask is not None
                else None
            )
            with _autocast(model_device, mixed_precision):
                loss = causal_lm_loss(model, inputs, targets, loss_mask)
            total_loss += float(loss.item()) * len(indices)
            total_examples += len(indices)
    return total_loss / total_examples


def _append_metric(path: Path, metric: dict[str, Any]) -> None:
    ensure_private_dir(path.parent)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(metric, sort_keys=True) + "\n")


def train_model(
    training_config: dict[str, Any],
    data_dir: str | Path,
    tokenizer_dir: str | Path,
    output_dir: str | Path,
    *,
    resume_from: str | Path | None = None,
    compile_step: bool = False,
    device: str = "auto",
) -> dict[str, Any]:
    output = ensure_private_dir(output_dir)
    data_path = Path(data_dir)
    target_device = resolve_device(device)
    tokenizer = load_tokenizer(tokenizer_dir)
    actual_vocab_size = tokenizer.get_vocab_size()
    config_value = dict(training_config)
    config_value["vocab_size"] = actual_vocab_size
    model_config = ModelConfig.from_dict(config_value)
    context_length = model_config.max_sequence_length
    train_tokens = load_tokens(data_path / "train.npy")
    validation_tokens = load_tokens(data_path / "validation.npy")
    role_ids = {
        "me_id": tokenizer.token_to_id("<|me|>"),
        "other_id": tokenizer.token_to_id("<|other|>"),
        "turn_end_id": tokenizer.token_to_id("<|turn_end|>"),
        "conversation_id": tokenizer.token_to_id("<|conversation|>"),
    }
    if any(role_ids[name] is None for name in ("me_id", "other_id", "turn_end_id")):
        raise ValueError("Tokenizer is missing required conversation role tokens")
    train_target_mask = build_me_target_mask(train_tokens, **role_ids)
    validation_target_mask = build_me_target_mask(validation_tokens, **role_ids)
    train_count = window_count(train_tokens, context_length)
    if not train_count:
        raise ValueError(
            f"Training data has fewer than {context_length + 1} tokens; cannot form a batch"
        )

    seed = int(training_config.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)
    if target_device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
        torch.set_float32_matmul_precision("high")
    model = TransformerLM(model_config).to(target_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.1)),
    )

    state = {
        "epoch": 0,
        "batch_in_epoch": 0,
        "global_step": 0,
        "tokens_trained": 0,
        "best_validation_loss": math.inf,
        "evaluations_without_improvement": 0,
    }
    if resume_from is not None:
        state.update(restore_training_state(resume_from, model, optimizer))

    mixed_precision = bool(training_config.get("mixed_precision", True)) and (
        target_device.type == "cuda" and torch.cuda.is_bf16_supported()
    )
    compiled = bool(compile_step and target_device.type == "cuda" and hasattr(torch, "compile"))
    execution_model = torch.compile(model) if compiled else model

    batch_size = int(training_config["batch_size"])
    epochs = int(training_config["epochs"])
    batches_per_epoch = math.ceil(train_count / batch_size)
    total_steps = max(1, epochs * batches_per_epoch)
    warmup_steps = min(int(training_config.get("warmup_steps", 100)), max(1, total_steps // 2))
    peak_lr = float(training_config["learning_rate"])
    minimum_lr = float(training_config.get("minimum_learning_rate", peak_lr * 0.1))
    gradient_clip = float(training_config.get("gradient_clip", 1.0))
    evaluation_interval = int(training_config.get("evaluation_interval", 250))
    checkpoint_interval = int(training_config.get("checkpoint_interval", 500))
    patience = int(training_config.get("early_stopping_patience", 5))
    logging_interval = max(1, int(training_config.get("logging_interval", 25)))
    metrics_path = output / "metrics.jsonl"
    started = time.perf_counter()
    stop_early = False

    gpu_name = torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "CPU"
    print(
        f"Starting {training_config.get('name', 'model')} on {gpu_name}: "
        f"{model.parameter_count:,} parameters, {train_count:,} training windows."
    )
    print("Progress contains aggregate metrics only. Message text is never printed.")

    def checkpoint(name: str) -> None:
        checkpoint_state = dict(state)
        checkpoint_state.update(
            {
                "total_steps": total_steps,
                "parameter_count": model.parameter_count,
                "torch_version": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": str(target_device),
                "gpu_name": torch.cuda.get_device_name(target_device)
                if target_device.type == "cuda"
                else None,
                "project_version": __version__,
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "compiled_model": compiled,
                "mixed_precision": mixed_precision,
                "initialized_from_checkpoint": resume_from is not None,
                "pretrained_weights_used": False,
            }
        )
        save_checkpoint(
            output / name,
            model,
            optimizer,
            checkpoint_state,
            training_config,
            tokenizer_dir,
        )

    for epoch in range(int(state["epoch"]), epochs):
        model.train()
        start_batch = int(state["batch_in_epoch"]) if epoch == int(state["epoch"]) else 0
        for batch_number, indices in enumerate(
            batch_indices(
                train_count,
                batch_size,
                seed=seed,
                epoch=epoch,
                start_batch=start_batch,
            ),
            start=start_batch,
        ):
            learning_rate = learning_rate_at_step(
                int(state["global_step"]), total_steps, warmup_steps, peak_lr, minimum_lr
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            inputs_np, targets_np = materialize_batch(train_tokens, indices, context_length)
            inputs = torch.from_numpy(inputs_np).long().to(target_device)
            targets = torch.from_numpy(targets_np).long().to(target_device)
            loss_mask = torch.from_numpy(
                materialize_mask(train_target_mask, indices, context_length)
            ).to(target_device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(target_device, mixed_precision):
                loss = causal_lm_loss(execution_model, inputs, targets, loss_mask)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            optimizer.step()
            state["global_step"] = int(state["global_step"]) + 1
            state["tokens_trained"] = int(state["tokens_trained"]) + int(targets_np.size)
            state["epoch"] = epoch
            state["batch_in_epoch"] = batch_number + 1
            elapsed = max(time.perf_counter() - started, 1e-9)
            metric = {
                "type": "train",
                "step": state["global_step"],
                "epoch": epoch,
                "tokens_trained": state["tokens_trained"],
                "learning_rate": learning_rate,
                "training_loss": float(loss.item()),
                "gradient_norm": float(gradient_norm.item()),
                "tokens_per_second": state["tokens_trained"] / elapsed,
                "peak_memory_bytes": int(torch.cuda.max_memory_allocated())
                if target_device.type == "cuda"
                else 0,
                "elapsed_seconds": elapsed,
            }
            _append_metric(metrics_path, metric)
            if (
                int(state["global_step"]) == 1
                or int(state["global_step"]) % logging_interval == 0
                or batch_number + 1 >= batches_per_epoch
            ):
                memory_gb = metric["peak_memory_bytes"] / (1024**3)
                print(
                    f"step {state['global_step']}/{total_steps} | epoch {epoch + 1}/{epochs} | "
                    f"loss {metric['training_loss']:.4f} | "
                    f"{metric['tokens_per_second']:,.0f} tokens/s | "
                    f"GPU memory {memory_gb:.2f} GB"
                )

            should_evaluate = state["global_step"] % evaluation_interval == 0
            is_last_batch = batch_number + 1 >= batches_per_epoch
            if should_evaluate or is_last_batch:
                validation_loss = evaluate_loss(
                    execution_model,
                    validation_tokens,
                    context_length=context_length,
                    batch_size=batch_size,
                    mixed_precision=mixed_precision,
                    target_mask=validation_target_mask,
                )
                _append_metric(
                    metrics_path,
                    {
                        "type": "validation",
                        "step": state["global_step"],
                        "validation_loss": validation_loss,
                        "validation_perplexity": perplexity(validation_loss),
                    },
                )
                print(
                    f"validation | step {state['global_step']} | loss {validation_loss:.4f} | "
                    f"perplexity {perplexity(validation_loss):.2f}"
                )
                if validation_loss < float(state["best_validation_loss"]):
                    state["best_validation_loss"] = validation_loss
                    state["evaluations_without_improvement"] = 0
                    checkpoint("best")
                else:
                    state["evaluations_without_improvement"] = (
                        int(state["evaluations_without_improvement"]) + 1
                    )
                model.train()
                if int(state["evaluations_without_improvement"]) >= patience:
                    stop_early = True
                    break
            if state["global_step"] % checkpoint_interval == 0:
                checkpoint("last")

        if stop_early:
            break
        state["epoch"] = epoch + 1
        state["batch_in_epoch"] = 0

    checkpoint("last")
    summary = {
        **state,
        "parameter_count": model.parameter_count,
        "train_windows": train_count,
        "context_length": context_length,
        "actual_vocab_size": actual_vocab_size,
        "device": str(target_device),
        "gpu_name": torch.cuda.get_device_name(target_device)
        if target_device.type == "cuda"
        else None,
        "mixed_precision": mixed_precision,
        "compiled_model": compiled,
        "early_stopped": stop_early,
        "initialized_from_checkpoint": resume_from is not None,
        "pretrained_weights_used": False,
        "output_dir": str(output),
    }
    write_json(output / "training-summary.json", summary)
    print(f"Training finished. Checkpoints and metrics are in {output}.")
    return summary
