from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from imessage_cuda.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, width: int, epsilon: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        normalized = hidden * torch.rsqrt(hidden.pow(2).mean(dim=-1, keepdim=True) + self.epsilon)
        return normalized * self.weight


def _apply_rope(value: torch.Tensor, base: float) -> torch.Tensor:
    """Apply rotary position embeddings to [batch, heads, sequence, head_dim]."""
    length = value.shape[-2]
    head_dim = value.shape[-1]
    positions = torch.arange(length, device=value.device, dtype=torch.float32)
    frequencies = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=value.device, dtype=torch.float32) / head_dim)
    )
    angles = torch.outer(positions, frequencies)
    cosine = angles.cos().to(value.dtype)[None, None, :, :]
    sine = angles.sin().to(value.dtype)[None, None, :, :]
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.rope_base = config.rope_base
        self.query = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.key = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.value = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.output = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape

        def heads(value: torch.Tensor) -> torch.Tensor:
            return value.reshape(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        queries = _apply_rope(heads(self.query(hidden)), self.rope_base)
        keys = _apply_rope(heads(self.key(hidden)), self.rope_base)
        values = heads(self.value(hidden))
        attended = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.output(attended)


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down(F.silu(self.gate(hidden)) * self.up(hidden)))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.attention = CausalSelfAttention(config)
        self.mlp_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.mlp = SwiGLU(config)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attention(self.attention_norm(hidden))
        return hidden + self.mlp(self.mlp_norm(hidden))


class TransformerLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_layers))
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_epsilon)
        self.output_projection = (
            None
            if config.tie_embeddings
            else nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        length = input_ids.shape[1]
        if length > self.config.max_sequence_length:
            raise ValueError(
                f"Sequence length {length} exceeds configured maximum "
                f"{self.config.max_sequence_length}"
            )
        hidden = self.embedding_dropout(self.token_embedding(input_ids))
        for layer in self.layers:
            hidden = layer(hidden)
        hidden = self.final_norm(hidden)
        if self.config.tie_embeddings:
            return F.linear(hidden, self.token_embedding.weight)
        assert self.output_projection is not None
        return self.output_projection(hidden)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device


def causal_lm_loss(
    model: TransformerLM,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = model(inputs)
    losses = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape_as(targets)
    if loss_mask is None:
        return losses.mean()
    selected = losses[loss_mask]
    if selected.numel() == 0:
        raise ValueError("Training batch contains no outgoing iMessage target tokens")
    return selected.mean()


def perplexity(loss: float) -> float:
    return math.exp(min(loss, 50.0))
