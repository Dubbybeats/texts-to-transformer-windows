from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from imessage_cuda.checkpoint import load_model
from imessage_cuda.data.redact import contains_obvious_pii
from imessage_cuda.dataset import (
    build_me_target_mask,
    load_tokens,
    materialize_batch,
    materialize_mask,
    window_count,
)
from imessage_cuda.generate import generate_ids
from imessage_cuda.model.transformer import causal_lm_loss, perplexity
from imessage_cuda.tokenizer.train import load_tokenizer
from imessage_cuda.utils import write_json


def _cross_entropy(model, tokens: np.ndarray, context: int, batch_size: int) -> float:
    count = window_count(tokens, context)
    if not count:
        raise ValueError("Split has no complete evaluation windows")
    total = 0.0
    examples = 0
    model.eval()
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        inputs, targets = materialize_batch(tokens, indices, context)
        inputs_tensor = torch.from_numpy(inputs).long().to(model.device)
        targets_tensor = torch.from_numpy(targets).long().to(model.device)
        with torch.inference_mode():
            loss = causal_lm_loss(model, inputs_tensor, targets_tensor)
        total += float(loss.item()) * len(indices)
        examples += len(indices)
    return total / examples


def _unigram_loss(train: np.ndarray, test: np.ndarray, vocab_size: int) -> float:
    counts = np.bincount(np.asarray(train, dtype=np.int64), minlength=vocab_size).astype(np.float64)
    probabilities = (counts + 1.0) / (counts.sum() + vocab_size)
    return float(-np.log(probabilities[np.asarray(test, dtype=np.int64)]).mean())


def _me_role_loss(
    model,
    tokens: np.ndarray,
    tokenizer,
    context: int,
    batch_size: int,
) -> tuple[float | None, int]:
    me_id = tokenizer.token_to_id("<|me|>")
    other_id = tokenizer.token_to_id("<|other|>")
    turn_end_id = tokenizer.token_to_id("<|turn_end|>")
    conversation_id = tokenizer.token_to_id("<|conversation|>")
    if None in (me_id, other_id, turn_end_id):
        return None, 0
    total_loss = 0.0
    total_tokens = 0
    count = window_count(tokens, context)
    target_mask = build_me_target_mask(
        tokens,
        me_id=me_id,
        other_id=other_id,
        turn_end_id=turn_end_id,
        conversation_id=conversation_id,
    )
    model.eval()
    for start in range(0, count, batch_size):
        indices = np.arange(start, min(start + batch_size, count))
        inputs_np, targets_np = materialize_batch(tokens, indices, context)
        inputs = torch.from_numpy(inputs_np).long().to(model.device)
        targets = torch.from_numpy(targets_np).long().to(model.device)
        with torch.inference_mode():
            logits = model(inputs)
            losses = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="none",
            ).reshape_as(targets)
        mask = materialize_mask(target_mask, indices, context)
        batch_losses = losses.float().cpu().numpy()
        total_loss += float(batch_losses[mask].sum())
        total_tokens += int(mask.sum())
    return (total_loss / total_tokens if total_tokens else None), total_tokens


def _ngram_hashes(tokens: np.ndarray, size: int, maximum: int = 2_000_000) -> set[bytes]:
    values = np.asarray(tokens, dtype=np.uint32)
    count = min(maximum, max(0, len(values) - size + 1))
    return {values[index : index + size].tobytes() for index in range(count)}


def evaluate_checkpoint(
    checkpoint: str | Path,
    data_dir: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 16,
    generation_samples: int = 20,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint)
    model = load_model(checkpoint_path)
    tokenizer = load_tokenizer(checkpoint_path / "tokenizer")
    data_path = Path(data_dir)
    train = load_tokens(data_path / "train.npy")
    validation = load_tokens(data_path / "validation.npy")
    test = load_tokens(data_path / "test.npy")
    context = model.config.max_sequence_length
    validation_loss = _cross_entropy(model, validation, context, batch_size)
    test_loss = _cross_entropy(model, test, context, batch_size)
    me_loss, me_tokens = _me_role_loss(model, test, tokenizer, context, batch_size)
    unigram_loss = _unigram_loss(train, test, model.config.vocab_size)

    train_8grams = _ngram_hashes(train, 8)
    train_16grams = _ngram_hashes(train, 16)
    turn_end = tokenizer.token_to_id("<|turn_end|>")
    eos = tokenizer.token_to_id("<|eos|>")
    eos_ids = {token_id for token_id in (turn_end, eos) if token_id is not None}
    test_windows = window_count(test, context)
    sample_count = min(generation_samples, test_windows)
    overlaps_8 = 0
    overlaps_16 = 0
    generated_8 = 0
    generated_16 = 0
    pii_generations = 0
    longest_match = 0
    for index in range(sample_count):
        start = index * (context + 1)
        prompt = np.asarray(test[start : start + max(8, context // 2)], dtype=np.int64).tolist()
        generated = generate_ids(
            model,
            prompt,
            eos_ids=eos_ids,
            max_new_tokens=32,
            temperature=0.8,
            top_p=0.9,
            repetition_penalty=1.1,
            seed=42 + index,
        )
        decoded = tokenizer.decode(generated, skip_special_tokens=True)
        pii_generations += int(contains_obvious_pii(decoded))
        array = np.asarray(generated, dtype=np.uint32)
        current_longest = 0
        for size, known in ((8, train_8grams), (16, train_16grams)):
            matches = 0
            possible = max(0, len(array) - size + 1)
            for position in range(possible):
                if array[position : position + size].tobytes() in known:
                    matches += 1
                    current_longest = max(current_longest, size)
            if size == 8:
                overlaps_8 += matches
                generated_8 += possible
            else:
                overlaps_16 += matches
                generated_16 += possible
        longest_match = max(longest_match, current_longest)

    report = {
        "parameter_count": model.parameter_count,
        "context_length": context,
        "train_tokens": int(train.size),
        "validation_tokens": int(validation.size),
        "test_tokens": int(test.size),
        "validation_loss": validation_loss,
        "validation_perplexity": perplexity(validation_loss),
        "test_loss": test_loss,
        "test_perplexity": perplexity(test_loss),
        "me_turn_test_loss": me_loss,
        "me_turn_test_perplexity": perplexity(me_loss) if me_loss is not None else None,
        "me_turn_test_tokens": me_tokens,
        "unigram_test_loss": unigram_loss,
        "unigram_test_perplexity": math.exp(min(unigram_loss, 50)),
        "beats_unigram_baseline": test_loss < unigram_loss,
        "memorization": {
            "generation_samples": sample_count,
            "generated_8grams": generated_8,
            "matching_train_8grams": overlaps_8,
            "generated_16grams": generated_16,
            "matching_train_16grams": overlaps_16,
            "longest_detected_exact_match_tokens": longest_match,
            "generations_with_obvious_pii_pattern": pii_generations,
            "matched_text_persisted": False,
        },
    }
    write_json(output_path, report)
    return report
