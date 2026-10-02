"""Jacobian GRU-input reconstruction must match the real forward path (T2).

For a dendritic-enabled :class:`NSMoRCore`, the vector the GRU cell
actually receives is ``model.frontend(sensory_x, lengths)`` — dendritic
IIR filtering included — not ``model.sensory_encoder(sensory_x)``.
Reconstructing through the inner encoder silently feeds the Jacobian a
different input than the forward pass used.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nsmor.model_nsmor_core import NSMoRCore
from nsmor.nsmor_dataloader import NSMoRDataset, collate_variable_length
from scripts import analyze_jacobian as jacobian


def _dendritic_loader(n_frames: int = 1200, anchor: int = 400):
    x = np.zeros((n_frames, 8), dtype=np.float32)
    # A looming visual ramp (channel 0) so the dendritic filter has
    # non-trivial temporal structure; static input would make the two
    # encoders coincide and hide the bug.
    x[:, 0] = np.linspace(2.0, 60.0, n_frames)
    x[:, 2] = np.linspace(0.0, 5.0, n_frames)
    dataset = NSMoRDataset(
        [(x, np.zeros(n_frames, dtype=np.float32), 1)],
        np.full((1, 4), 0.25, dtype=np.float32),
        anchor_frames=[anchor], pre_anchor_frames=300, source_indices=[17],
    )
    return torch.utils.data.DataLoader(dataset, collate_fn=collate_variable_length)


def _patch_gate(monkeypatch):
    monkeypatch.setattr(jacobian, "_find_slow_point",
                        lambda h, centre, _length: (centre, h[centre]))
    monkeypatch.setattr(jacobian, "_fixed_point_residual", lambda *_a: 0.01)
    monkeypatch.setattr(jacobian, "_calibrate_fp_threshold",
                        lambda *_a, **_k: (0.02, {}))


def test_reconstruction_uses_frontend_for_dendritic_model(monkeypatch) -> None:
    torch.manual_seed(0)
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=8, num_gru_layers=1,
        dropout=0.0, lif_dendritic_tau=50.0, dt_ms=4.0,
    )
    model.eval()
    assert model.frontend._dendritic_enabled

    loader = _dendritic_loader()
    _patch_gate(monkeypatch)

    result = jacobian.extract_gru_states_at_epochs(
        model, loader, torch.device("cpu"), [400],
    )

    x_batch, _y, lengths = next(iter(loader))
    sensory_x = x_batch[:, :, :4]

    # Independently reconstruct the exact GRU input via the frontend.
    model.frontend._dendritic_state = None
    expected = model.frontend(sensory_x, lengths)          # (B, T, H)
    model.frontend._dendritic_state = None
    raw_encoder = model.sensory_encoder(sensory_x)         # (B, T, H)

    # The dendritic filter must genuinely change the input, or this test
    # would pass even under the old buggy path.
    assert not torch.allclose(expected, raw_encoder)

    for epoch_name, centre in (("early", 150), ("transient", 400), ("sustained", 650)):
        h_stack, x_stack = result[epoch_name]
        assert h_stack.shape == (1, 8)
        assert x_stack.shape == (1, 8)
        assert torch.allclose(x_stack[0], expected[0, centre], atol=1e-6), (
            f"{epoch_name}: reconstructed GRU input != frontend output"
        )
        assert not torch.allclose(x_stack[0], raw_encoder[0, centre]), (
            f"{epoch_name}: reconstruction used the inner sensory_encoder"
        )


def test_reconstruction_falls_back_to_sensory_encoder_without_frontend(monkeypatch) -> None:
    """Legacy probe models with no ``.frontend`` keep the inner-encoder path."""

    class ProbeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.dt_ms, self.sensory_dim, self.hidden_dim = 4.0, 4, 8
            self.sensory_encoder = torch.nn.Linear(4, 8)
            self._seen: list[torch.Tensor] = []

        def forward(self, x, lengths, *, return_internals):
            assert x.shape == (len(lengths), x.shape[1], 8)
            e = self.sensory_encoder(x[:, :, :4])
            self._seen.append(e.detach())
            return torch.zeros(x.shape[:2]), {"gru_hidden": e}

    model = ProbeModel()
    assert not hasattr(model, "frontend")
    loader = _dendritic_loader()
    _patch_gate(monkeypatch)

    result = jacobian.extract_gru_states_at_epochs(
        model, loader, torch.device("cpu"), [400], adapter=SimpleNamespace(),
    )
    expected = model._seen[0]
    _h_stack, x_stack = result["transient"]
    assert torch.allclose(x_stack[0], expected[0, 400], atol=1e-6)
