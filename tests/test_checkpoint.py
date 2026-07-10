from pathlib import Path

import torch

from imessage_cuda.checkpoint import load_model, restore_training_state, save_checkpoint
from imessage_cuda.model.config import ModelConfig
from imessage_cuda.model.transformer import TransformerLM, causal_lm_loss


def _model() -> TransformerLM:
    return TransformerLM(
        ModelConfig(
            vocab_size=32,
            hidden_size=16,
            num_layers=1,
            num_heads=4,
            intermediate_size=32,
            max_sequence_length=8,
        )
    )


def _update(model, optimizer, inputs, targets) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss = causal_lm_loss(model, inputs, targets)
    loss.backward()
    optimizer.step()


def test_checkpoint_round_trip_and_resume_match(tmp_path: Path) -> None:
    torch.manual_seed(7)
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    inputs = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    targets = torch.tensor([[2, 3, 4, 5]], dtype=torch.long)
    _update(model, optimizer, inputs, targets)
    model.eval()
    before = model(inputs)
    checkpoint = tmp_path / "checkpoint"
    save_checkpoint(checkpoint, model, optimizer, {"global_step": 1}, {"seed": 7})
    assert (checkpoint / "training-config.yaml").exists()

    reloaded = load_model(checkpoint, device="cpu")
    after = reloaded(inputs)
    assert torch.allclose(before, after, atol=1e-6)

    model.train()
    _update(model, optimizer, inputs, targets)
    resumed = _model()
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.01)
    state = restore_training_state(checkpoint, resumed, resumed_optimizer)
    resumed.train()
    _update(resumed, resumed_optimizer, inputs, targets)
    assert state["global_step"] == 1
    for expected, actual in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.allclose(expected, actual, atol=1e-6)
