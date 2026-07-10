import numpy as np
import torch

from imessage_cuda.dataset import build_me_target_mask
from imessage_cuda.model.config import ModelConfig
from imessage_cuda.model.transformer import TransformerLM, causal_lm_loss


def tiny_model() -> TransformerLM:
    torch.manual_seed(42)
    return TransformerLM(
        ModelConfig(
            vocab_size=64,
            hidden_size=32,
            num_layers=1,
            num_heads=4,
            intermediate_size=64,
            max_sequence_length=16,
        )
    )


def test_causal_mask_prevents_future_token_leakage() -> None:
    model = tiny_model()
    model.eval()
    first = model(torch.tensor([[1, 2, 3, 4]], dtype=torch.long))
    second = model(torch.tensor([[1, 2, 3, 5]], dtype=torch.long))
    assert torch.allclose(first[:, :3], second[:, :3], atol=1e-5)


def test_overfits_one_tiny_batch() -> None:
    model = tiny_model()
    model.train()
    inputs = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5, 6, 7, 8, 9]], dtype=torch.long)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.0)
    initial = causal_lm_loss(model, inputs, targets).item()
    for _ in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = causal_lm_loss(model, inputs, targets)
        loss.backward()
        optimizer.step()
    final = causal_lm_loss(model, inputs, targets).item()
    assert final < initial * 0.25


def test_reply_loss_mask_excludes_incoming_messages() -> None:
    tokens = np.asarray([9, 2, 20, 21, 4, 1, 30, 31, 4, 2, 40], dtype=np.uint32)
    mask = build_me_target_mask(
        tokens,
        me_id=1,
        other_id=2,
        turn_end_id=4,
        conversation_id=9,
    )
    assert mask.tolist() == [
        False,
        False,
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
