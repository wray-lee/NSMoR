"""
Dynamical Systems Adapter for NSMoR GRU pathway.

Provides :class:`FixedPointAdapter` for interfacing the NSMoR GRU
states with external RNN analysis libraries (e.g., Sussillo's
``FixedPointFinder``).

The adapter extracts valid (unpadded) GRU hidden-state trajectories
from the model and provides a Jacobian computation interface for
fixed-point analysis.

Shape legend
------------
    B  = batch_size
    T  = seq_len (padded)
    H  = hidden_dim
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from nsmor.model_nsmor_core import NSMoRCore


class FixedPointAdapter:
    """
    Adapter for extracting GRU hidden states and computing Jacobians
    for dynamical systems analysis.

    This class bridges the NSMoR model with external fixed-point
    analysis tools by:

    1. Extracting un-padded GRU hidden-state trajectories from the
       dataset.
    2. Providing a Jacobian computation interface for the GRU cell
       at specific hidden states.

    Args:
        model: A trained :class:`NSMoRCore` model.
        device: Device to run computations on.

    Example::

        adapter = FixedPointAdapter(model)
        trajectories = adapter.extract_gru_states(dataloader)
        J = adapter.compute_jacobian_at_state(h_t, x_t)
    """

    def __init__(
        self,
        model: NSMoRCore,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.model.eval()

        if device is None:
            device = next(model.parameters()).device
        self.device = device

        # Cache GRU cell for direct Jacobian access
        self._gru_cell: nn.GRU = model.gru_unit.gru

    # ═══════════════════════════════════════════════════════════
    # 1.  State Extraction
    # ═══════════════════════════════════════════════════════════

    @torch.no_grad()
    def extract_gru_states(
        self,
        dataloader: DataLoader,
    ) -> List[torch.Tensor]:
        """
        Extract un-padded GRU hidden-state trajectories from the dataset.

        Runs the model in ``eval`` mode over the entire dataloader,
        collects ``internals["gru_hidden"]``, and un-pads each sample
        into a list of valid temporal trajectories.

        Args:
            dataloader: A DataLoader yielding ``(X_batch, Y_batch, lengths)``
                tuples (as produced by ``collate_variable_length``).

        Returns:
            A list of tensors, one per sample in the dataset.  Each
            tensor has shape ``(length_i, H)`` where ``length_i`` is
            the true sequence length for that sample.

        Raises:
            ValueError: If the dataloader is empty.

        Example::

            adapter = FixedPointAdapter(model)
            trajectories = adapter.extract_gru_states(train_loader)

            # trajectories[i] has shape (T_i, H)
            # Stack for fixed-point analysis:
            all_states = torch.cat(trajectories, dim=0)  # (N_total, H)
        """
        trajectories: List[torch.Tensor] = []

        for batch_idx, batch in enumerate(dataloader):
            x_batch, _y_batch, lengths = batch
            x_batch = x_batch.to(self.device)
            lengths = lengths.to(self.device)

            B, T, _ = x_batch.shape

            # Forward pass with internals
            _y_pred, internals = self.model(
                x_batch, lengths, return_internals=True,
            )

            # gru_hidden: (B, T, H)
            gru_hidden = internals["gru_hidden"]

            # ── Shape assertion ──
            H = gru_hidden.shape[2]
            assert gru_hidden.shape == (B, T, H), (
                f"gru_hidden shape {tuple(gru_hidden.shape)} != "
                f"(B={B}, T={T}, H={H})"
            )

            # ── Un-pad into per-sample trajectories ──
            for i in range(B):
                length_i = lengths[i].item()
                # Shape: (length_i, H)
                traj_i = gru_hidden[i, :length_i, :].cpu()

                assert traj_i.shape == (length_i, H), (
                    f"Trajectory {batch_idx * B + i} shape "
                    f"{tuple(traj_i.shape)} != ({length_i}, {H})"
                )

                trajectories.append(traj_i)

        if not trajectories:
            raise ValueError("Dataloader yielded no samples.")

        return trajectories

    # ═══════════════════════════════════════════════════════════
    # 2.  Jacobian Computation
    # ═══════════════════════════════════════════════════════════

    def compute_jacobian_at_state(
        self,
        h_t: torch.Tensor,
        x_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the Jacobian of the GRU cell at a specific hidden state.

        Computes :math:`\\frac{\\partial h_{t+1}}{\\partial h_t}` where
        :math:`h_{t+1} = \\text{GRU}(x_t, h_t)`.

        This is the core quantity for fixed-point analysis: fixed points
        satisfy :math:`h_{t+1} = h_t`, and the Jacobian eigenvalues at
        those points characterize stability.

        Args:
            h_t: ``(H,)`` or ``(1, H)`` — hidden state at time *t*.
                Must have ``requires_grad=True``.
            x_t: ``(D,)`` or ``(1, D)`` — input at time *t*
                (sensory encoding, not raw features).  Does not require
                gradients.

        Returns:
            ``(H, H)`` Jacobian matrix :math:`\\partial h_{t+1} / \\partial h_t`.

        Raises:
            AssertionError: If tensor shapes are inconsistent.

        Example::

            h_t = torch.randn(H, requires_grad=True)
            x_t = sensory_encoder(sensory_input)  # (H,)
            J = adapter.compute_jacobian_at_state(h_t, x_t)
            eigenvalues = torch.linalg.eigvals(J)
        """
        H = self.model.hidden_dim

        # ── Ensure batch dimension ──
        if h_t.dim() == 1:
            h_t = h_t.unsqueeze(0)                                 # (1, H)
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(0)                                 # (1, D)

        # ── Shape assertions ──
        assert h_t.shape == (1, H), (
            f"h_t must be (H={H},) or (1, H), got {tuple(h_t.shape)}"
        )
        assert x_t.shape[0] == 1, (
            f"x_t batch dim must be 1, got {x_t.shape[0]}"
        )

        # ── Create leaf tensor with requires_grad ──
        # h_t may not be a leaf tensor, so we create a fresh one
        h_leaf = h_t.detach().clone().requires_grad_(True)

        # ── Forward through GRU cell ──
        # nn.GRU expects:
        #   input: (batch, seq_len, input_size)
        #   h_0:   (num_layers, batch, hidden_size)
        # We use seq_len=1 for a single step.
        x_t_seq = x_t.detach().unsqueeze(1)                       # (1, 1, D)
        h_t_input = h_leaf.unsqueeze(0)                            # (1, 1, H) — num_layers=1

        # GRU forward: output (1, 1, H), h_n (1, 1, H)
        output, _ = self._gru_cell(x_t_seq, h_t_input)            # (1, 1, H)
        h_next = output.squeeze(0).squeeze(0)                       # (H,)

        # ── Compute Jacobian via autograd ──
        # J[i, j] = ∂h_next[i] / ∂h_leaf[j]
        jacobian = torch.zeros(H, H, device=self.device)

        for i in range(H):
            # Zero previous gradients
            if h_leaf.grad is not None:
                h_leaf.grad.zero_()

            # Backpropagate for the i-th component of h_next
            h_next[i].backward(retain_graph=True)

            # The gradient ∂h_next[i]/∂h_leaf is now in h_leaf.grad
            jacobian[i, :] = h_leaf.grad.detach()

        return jacobian

    # ═══════════════════════════════════════════════════════════
    # 3.  Batch Jacobian (efficient for multiple states)
    # ═══════════════════════════════════════════════════════════

    def compute_jacobian_batch(
        self,
        h_states: torch.Tensor,
        x_inputs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobians for a batch of hidden states.

        More efficient than calling ``compute_jacobian_at_state`` in a
        loop, as it batches the autograd computation.

        Args:
            h_states: ``(N, H)`` — batch of hidden states.
                Each must have ``requires_grad=True``.
            x_inputs: ``(N, D)`` — corresponding inputs.

        Returns:
            ``(N, H, H)`` — Jacobian for each state.

        Raises:
            AssertionError: If shapes are inconsistent.
        """
        N, H = h_states.shape
        D = x_inputs.shape[1]

        assert h_states.shape == (N, H), (
            f"h_states shape {tuple(h_states.shape)} != (N={N}, H={H})"
        )
        assert x_inputs.shape == (N, D), (
            f"x_inputs shape {tuple(x_inputs.shape)} != (N={N}, D={D})"
        )

        # ── Forward pass for all states ──
        x_seq = x_inputs.unsqueeze(1)                              # (N, 1, D)
        h_input = h_states.unsqueeze(1)                            # (N, 1, H)

        # GRU expects (batch, seq_len, input_size) and
        # (num_layers, batch, hidden_size) — we need to transpose h_input
        h_input_gru = h_input.permute(1, 0, 2)                     # (1, N, H)
        output, _ = self._gru_cell(x_seq, h_input_gru)            # (N, 1, H)
        h_next = output.squeeze(1)                                 # (N, H)

        # ── Compute Jacobians ──
        jacobians = torch.zeros(N, H, H, device=self.device)

        for i in range(H):
            # Zero gradients
            if h_states.grad is not None:
                h_states.grad.zero_()

            # Backpropagate for the i-th component of all h_next
            h_next[:, i].sum().backward(retain_graph=True)

            # Collect gradients: (N, H)
            jacobians[:, i, :] = h_states.grad.detach()

        return jacobians


# ═══════════════════════════════════════════════════════════════
# Smoke test
# ═══════════════════════════════════════════════════════════════

def _test_fixed_point_adapter() -> None:
    """
    Verify ``FixedPointAdapter`` state extraction and Jacobian computation.

    Run::

        python -m nsmor.analysis.dynamics
    """
    print("=" * 60)
    print("FixedPointAdapter smoke test")
    print("=" * 60)

    from torch.utils.data import DataLoader, TensorDataset

    B, T, H = 4, 50, 32
    device = torch.device("cpu")

    # ── Create a minimal model ──
    model = NSMoRCore(
        sensory_dim=4, mcmc_dim=4, hidden_dim=H,
        num_gru_layers=1, dropout=0.1,
    ).to(device)
    model.eval()

    # ── Create a minimal dataloader ──
    # X: (N, T, 8), Y: (N, T), lengths: (N,)
    N = 8
    X_data = torch.randn(N, T, 8)
    Y_data = torch.randn(N, T)
    lengths_data = torch.tensor([T, T - 5, T - 10, T - 20,
                                  T - 5, T, T - 15, T - 1], dtype=torch.int64)

    # Custom dataset to include lengths
    class MiniDataset(torch.utils.data.Dataset):
        def __init__(self, X, Y, lengths):
            self.X = X
            self.Y = Y
            self.lengths = lengths

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.Y[idx], self.lengths[idx]

    dataset = MiniDataset(X_data, Y_data, lengths_data)
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    # ── Test state extraction ──
    adapter = FixedPointAdapter(model, device=device)
    trajectories = adapter.extract_gru_states(loader)

    assert len(trajectories) == N, (
        f"Expected {N} trajectories, got {len(trajectories)}"
    )

    total_states = sum(t.shape[0] for t in trajectories)
    print(f"  Extracted {len(trajectories)} trajectories, "
          f"{total_states} total states")

    # Verify shapes
    for i, traj in enumerate(trajectories):
        expected_len = lengths_data[i].item()
        assert traj.shape == (expected_len, H), (
            f"Trajectory {i} shape {tuple(traj.shape)} != ({expected_len}, {H})"
        )
    print("  Trajectory shapes: OK")

    # ── Test single Jacobian ──
    h_t = torch.randn(H, requires_grad=True)
    x_t = torch.randn(4)  # sensory_dim = 4, but we need H after encoding

    # For Jacobian test, use hidden_dim as input (sensory encoding output)
    x_t_enc = torch.randn(H)
    J = adapter.compute_jacobian_at_state(h_t, x_t_enc)

    assert J.shape == (H, H), (
        f"Jacobian shape {tuple(J.shape)} != (H={H}, H={H})"
    )
    print(f"  Jacobian shape:       {tuple(J.shape)} == ({H}, {H})")

    # Check that Jacobian is not all zeros
    assert J.abs().sum() > 0, "Jacobian should not be all zeros"
    print(f"  Jacobian norm:        {J.norm().item():.4f}")

    # ── Test batch Jacobian ──
    N_batch = 3
    h_batch = torch.randn(N_batch, H, requires_grad=True)
    x_batch = torch.randn(N_batch, H)

    J_batch = adapter.compute_jacobian_batch(h_batch, x_batch)

    assert J_batch.shape == (N_batch, H, H), (
        f"Batch Jacobian shape {tuple(J_batch.shape)} != ({N_batch}, {H}, {H})"
    )
    print(f"  Batch Jacobian shape: {tuple(J_batch.shape)} == ({N_batch}, {H}, {H})")

    # ── Eigenvalue analysis (quick sanity) ──
    eigenvalues = torch.linalg.eigvals(J)
    print(f"  Jacobian eigenvalues (first 5): {eigenvalues[:5].tolist()}")

    print("=" * 60)
    print("All FixedPointAdapter assertions passed.")
    print("=" * 60)


if __name__ == "__main__":
    _test_fixed_point_adapter()
