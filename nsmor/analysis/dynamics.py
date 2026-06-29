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

        # Temporarily set GRU to training mode for cuDNN backward.
        # Use try/finally to guarantee mode restoration on exceptions.
        prev_mode = self._gru_cell.training
        self._gru_cell.train()

        try:
            # GRU forward: output (1, 1, H), h_n (1, 1, H)
            output, _ = self._gru_cell(x_t_seq, h_t_input)        # (1, 1, H)
            h_next = output.squeeze(0).squeeze(0)                   # (H,)

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

        finally:
            # Restore previous mode GUARANTEED, even on exceptions.
            self._gru_cell.train(prev_mode)

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

        # ── Create leaf tensor with requires_grad ──
        h_leaf = h_states.detach().clone().requires_grad_(True)

        # ── Forward pass for all states ──
        x_seq = x_inputs.unsqueeze(1)                              # (N, 1, D)
        h_input = h_leaf.unsqueeze(1)                              # (N, 1, H)

        # GRU expects (batch, seq_len, input_size) and
        # (num_layers, batch, hidden_size) — we need to transpose h_input
        h_input_gru = h_input.permute(1, 0, 2)                     # (1, N, H)

        # Temporarily set GRU to training mode for cuDNN backward.
        # Use try/finally to guarantee mode restoration on exceptions.
        prev_mode = self._gru_cell.training
        self._gru_cell.train()

        try:
            output, _ = self._gru_cell(x_seq, h_input_gru)        # (N, 1, H)
        finally:
            self._gru_cell.train(prev_mode)

        h_next = output.squeeze(1)                                 # (N, H)

        # ── Compute Jacobians ──
        jacobians = torch.zeros(N, H, H, device=self.device)

        for i in range(H):
            # Zero gradients
            if h_leaf.grad is not None:
                h_leaf.grad.zero_()

            # Backpropagate for the i-th component of all h_next
            h_next[:, i].sum().backward(retain_graph=True)

            # Collect gradients: (N, H)
            jacobians[:, i, :] = h_leaf.grad.detach()

        return jacobians

    # ═══════════════════════════════════════════════════════════
    # 4.  Perturbation-Response Test (Attractor Verification)
    # ═══════════════════════════════════════════════════════════

    def test_attractor_convergence(
        self,
        h_star: torch.Tensor,
        x_input: torch.Tensor,
        perturbation_magnitude: float = 0.01,
        convergence_radius: float = 0.01,
        K: int = 50,
        n_directions: int = 3,
    ) -> Tuple[bool, float, bool]:
        """
        Test whether a candidate fixed point h* is an attractor.

        CF2 fix: Perturbs along the top-k eigenvectors of the Jacobian
        at h* (not just a random direction), which is necessary to
        detect saddle points.  Separates perturbation_magnitude from
        convergence_radius.  Verifies monotonic convergence.

        CF3 fix: K is calibrated to 3-5 membrane time constants.
        For alpha=0.9, tau_membrane = -1/ln(0.9) ≈ 9.5 steps.
        K=50 ≈ 5.3 time constants — sufficient for convergence.
        perturbation_magnitude is calibrated to ~1% of typical
        membrane potential range (V_threshold ≈ 1.0).

        CF4 fix: ``@torch.no_grad()`` was removed from the decorator
        because the Jacobian computation block (lines computing
        ``h_next[i].backward(retain_graph=True)``) requires autograd
        to build the computation graph.  The K-step convergence loop
        is wrapped in ``torch.no_grad()`` locally to avoid
        unnecessary graph construction.  The fixed-point verification
        and Jacobian blocks use ``torch.enable_grad()`` explicitly.

        Args:
            h_star: ``(H,)`` — candidate fixed point (slow point).
            x_input: ``(H,)`` — sensory encoding input at the fixed
                point (GRU input_size = hidden_dim, NOT raw sensory dim).
            perturbation_magnitude: Radius of perturbation sphere.
            convergence_radius: Threshold for convergence (may differ
                from perturbation_magnitude).
            K: Number of forward steps (calibrated to time constants).
            n_directions: Number of perturbation directions to test.

        Returns:
            ``(is_attractor, max_residual, monotonic_convergence)`` where:
            - ``is_attractor``: True if all directions converge within
              convergence_radius after K steps.
            - ``max_residual``: Maximum ||h_{t+K} - h*|| across directions.
            - ``monotonic_convergence``: True if residual decreases
              monotonically (not just eventually converges).
        """
        H = h_star.shape[0]

        # Prepare x_seq for GRU cell: (batch=1, seq_len=1, input_size=H)
        if x_input.dim() == 1:
            x_seq = x_input.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, H)
        else:
            x_seq = x_input.detach().unsqueeze(1)  # already (1, H) -> (1, 1, H)

        # CF4 fix: Use torch.enable_grad() for the fixed-point verification
        # and Jacobian blocks.  The K-step convergence loop uses
        # torch.no_grad() locally.
        with torch.enable_grad():
            # CF2 fix: Use eval mode for both residual check and Jacobian.
            # (train mode has dropout which adds noise to the Jacobian
            # and the residual check).
            # CF-FIX (Reviewer A #3, B #4): Wrap in try/finally to
            # guarantee mode restoration even if Jacobian computation
            # raises an exception.  Without this, an exception between
            # eval() and train() would leave the GRU in eval mode
            # permanently, affecting the caller's model instance
            # (stored by reference).
            #
            # CF-FIX (Reviewer B #4): eval() is now placed BEFORE the
            # residual check so both the residual check and Jacobian
            # run in eval mode consistently.  Previously, the residual
            # check ran in whatever mode the adapter was initialized
            # with (typically eval from __init__), while the explicit
            # eval() only wrapped the Jacobian block.
            prev_mode = self._gru_cell.training
            self._gru_cell.eval()

            try:
                # CF3 fix: Verify h* is actually a fixed point
                h_star_input = h_star.detach().unsqueeze(0).unsqueeze(0)  # (1, 1, H)
                h_star_h0 = h_star_input.permute(1, 0, 2)  # (1, 1, H)
                h_star_next, _ = self._gru_cell(x_seq, h_star_h0)
                h_star_next_squeezed = h_star_next.squeeze(0).squeeze(0)
                fixed_point_residual = (h_star_next_squeezed - h_star).norm().item()
                if fixed_point_residual > convergence_radius:
                    return False, fixed_point_residual, False

                # Compute Jacobian at h* to find principal perturbation directions
                h_leaf = h_star.detach().clone().unsqueeze(0).requires_grad_(True)
                h_input = h_leaf.unsqueeze(0)  # (1, 1, H)

                output, _ = self._gru_cell(x_seq, h_input)
                h_next = output.squeeze(0).squeeze(0)  # (H,)

                # Compute Jacobian
                J = torch.zeros(H, H, device=self.device)
                for i in range(H):
                    if h_leaf.grad is not None:
                        h_leaf.grad.zero_()
                    h_next[i].backward(retain_graph=True)
                    J[i, :] = h_leaf.grad.detach()

            finally:
                # Restore training mode GUARANTEED, even on exceptions.
                self._gru_cell.train(prev_mode)

        # Eigenvectors of J (principal perturbation directions)
        eigvals, eigvecs = torch.linalg.eig(J)

        # CF-FIX (Reviewer A #1 + B #1): Use complex modulus |lambda|
        # for the stability boundary criterion, NOT |Re(lambda)|.
        #
        # The stability boundary of a discrete-time dynamical system is
        # |lambda|=1 (the unit circle in the complex plane), NOT
        # |Re(lambda)|=1.  For oscillatory GRU modes with complex
        # eigenvalues near the unit circle (e.g., lambda=0.5+0.866j has
        # |lambda|=1.0 but |Re(lambda)|=0.5), using |Re(lambda)| would
        # drastically underestimate proximity to the stability boundary,
        # causing the top-k selection to MISS the most informative slow
        # oscillatory modes.
        eigval_magnitudes = eigvals.abs()
        distance_to_boundary = torch.abs(eigval_magnitudes - 1.0)
        n_dirs = min(n_directions, H)
        top_k_indices = distance_to_boundary.topk(n_dirs, largest=False).indices

        max_residual = 0.0
        all_converged = True
        all_monotonic = True

        # CF-FIX (Reviewer A #2 + B #2): Properly decompose complex
        # eigenvectors into real and imaginary parts.
        #
        # For complex conjugate eigenvalue pairs (the generic case for
        # oscillatory modes in GRU dynamics), the real and imaginary
        # parts of the eigenvector span a 2D invariant subspace.
        # Re(v) and Im(v) form an orthogonal basis for this subspace.
        # Both directions must be tested to avoid missing the most
        # informative perturbation direction.
        #
        # For real eigenvalues, the eigenvector is (essentially) real
        # and we test only the real part.
        for idx in top_k_indices:
            eigvec = eigvecs[:, idx]  # complex eigenvector, (H,)
            ev = eigvals[idx]

            if ev.imag.abs() > 1e-6:
                # Complex eigenvalue: test both Re(v) and Im(v)
                # These span the 2D invariant subspace for the
                # conjugate pair.
                directions_to_test = [eigvec.real, eigvec.imag]
            else:
                # Real eigenvalue: eigenvector is (essentially) real
                directions_to_test = [eigvec.real]

            for direction in directions_to_test:
                direction = direction / (direction.norm() + 1e-8)

                # Perturb along this direction
                h_perturbed = h_star + perturbation_magnitude * direction

                # Forward K steps, tracking residual at each step
                # CF4 fix: K-step convergence loop uses no_grad() locally
                # to avoid building unnecessary computation graphs.
                h0 = h_perturbed.unsqueeze(0).unsqueeze(0)  # (1, 1, H)
                h0 = h0.permute(1, 0, 2)  # (1, 1, H) for GRU

                prev_residual = float('inf')
                monotonic = True

                with torch.no_grad():
                    for step in range(K):
                        output, h0 = self._gru_cell(x_seq, h0)
                        h_current = h0.squeeze(0).squeeze(0)  # (H,)
                        residual = (h_current - h_star).norm().item()

                        # Check monotonic convergence
                        if residual > prev_residual * 1.01:  # 1% tolerance
                            monotonic = False

                        prev_residual = residual

                final_residual = prev_residual
                max_residual = max(max_residual, final_residual)

                if final_residual >= convergence_radius:
                    all_converged = False
                if not monotonic:
                    all_monotonic = False

        return all_converged, max_residual, all_monotonic

    # ═══════════════════════════════════════════════════════════
    # 5.  Full System Jacobian (LIF + GRU + Router)
    # ═══════════════════════════════════════════════════════════

    def compute_full_system_jacobian(
        self,
        X_t: torch.Tensor,
        lengths: torch.Tensor,
        states: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute the Jacobian of the full MoR system output with respect
        to the sensory input, capturing LIF, GRU, and Router contributions.

        CF5 fix: The original ``compute_jacobian_at_state`` only computed
        the GRU pathway Jacobian, ignoring the LIF pathway and the MoR
        routing gates.  This method computes the full system Jacobian:

        .. warning::
            **Mathematical limitation (CF2):** The LIF pathway uses a
            surrogate gradient (straight-through estimator) for the
            non-differentiable spike function:

            ``spike = spike_mask - sigmoid(V - theta).detach() + sigmoid(V - theta)``

            Forward: binary 0/1 (discontinuous).
            Backward: smooth sigmoid derivative (continuous approximation).

            This means the computed Jacobian is the Jacobian of the
            **smooth surrogate** mapping, not the true discontinuous
            spike mapping.  The surrogate Jacobian is meaningful for:
            - Qualitative exploration of system sensitivity
            - Identifying which input dimensions the output is most
              sensitive to (relative magnitudes)
            - Fixed-point analysis of the GRU pathway (exact)

            It should NOT be used for:
            - Quantitative stability analysis of the LIF pathway
            - Lyapunov exponent computation
            - Precise eigenvalue-based bifurcation analysis

        .. math::
            \\frac{\\partial h_{out}}{\\partial x} =
            g_{lif} \\frac{\\partial \\text{LIF}}{\\partial x}
            + g_{gru} \\frac{\\partial \\text{GRU}}{\\partial x}
            + \\text{LIF} \\frac{\\partial g_{lif}}{\\partial x}
            + \\text{GRU} \\frac{\\partial g_{gru}}{\\partial x}

        where :math:`h_{out} = g_{lif} \\cdot \\text{LIF} + g_{gru} \\cdot \\text{GRU}`.

        Args:
            X_t: ``(1, 1, F)`` — full model input (sensory + MCMC).
            lengths: ``(1,)`` — sequence length (should be [1]).
            states: Optional recurrent states for autoregressive mode.

        Returns:
            ``(jacobian, internals)`` where:
            - ``jacobian``: ``(H, F)`` — dh_out/dx (H=hidden_dim, F=input_dim)
            - ``internals``: dict with ``routing_gates``, ``lif_spikes``,
              etc.

        Raises:
            AssertionError: If tensor shapes are inconsistent.
        """
        H = self.model.hidden_dim

        # Ensure input has gradients
        X_leaf = X_t.detach().clone().requires_grad_(True)

        # Forward pass with internals
        y_pred, internals, states_out = self.model(
            X_leaf, lengths, return_internals=True, states=states or {},
        )

        # CF5.1 fix: Use lif_spikes (spike output), not lif_potentials (membrane).
        # The routing gates blend the spike signals, not the membrane potentials.
        g_lif = internals["routing_gates"][:, :, 0:1]   # (1, 1, 1)
        g_gru = internals["routing_gates"][:, :, 1:2]   # (1, 1, 1)
        lif_out = internals["lif_spikes"]                # (1, 1, H)
        gru_out = internals["gru_hidden"]                # (1, 1, H)

        h_out = g_lif * lif_out + g_gru * gru_out        # (1, 1, H)
        h_out_squeezed = h_out.squeeze(0).squeeze(0)     # (H,)

        # CF5.3 fix: Use torch.autograd.functional.jacobian for efficient
        # computation without retain_graph=True memory explosion.
        #
        # CF4 fix: Save mutable state BEFORE Jacobian computation, let
        # _h_out_fn modify during evaluations, then restore original state
        # AFTER.  This avoids the stale-snapshot problem where multiple
        # evaluations would each restore the pre-computation state,
        # losing the intermediate updates.
        _lif_cell = self.model.lif_cell
        _orig_spike_hist = getattr(_lif_cell, '_spike_history', None)
        _orig_dend_state = getattr(_lif_cell, '_dendritic_state', None)

        def _h_out_fn(x_flat: torch.Tensor) -> torch.Tensor:
            """Reshape flat input and compute h_out for autograd.functional."""
            x_reshaped = x_flat.view_as(X_leaf)
            _y, _int, _st = self.model(
                x_reshaped, lengths, return_internals=True, states=states or {},
            )

            _g_lif = _int["routing_gates"][:, :, 0:1]
            _g_gru = _int["routing_gates"][:, :, 1:2]
            _lif = _int["lif_spikes"]
            _gru = _int["gru_hidden"]
            _h = _g_lif * _lif + _g_gru * _gru
            return _h.squeeze(0).squeeze(0)  # (H,)

        # CF-FIX (Reviewer B #2): Wrap Jacobian computation in try/finally
        # to guarantee LIF state restoration even on exceptions (OOM, NaN
        # in backprop, etc.).  _lif_cell is a reference to
        # self.model.lif_cell, so any exception would corrupt the
        # caller's model instance.  This is the same class of bug as
        # the GRU mode issue fixed with try/finally in
        # compute_jacobian_at_state, compute_jacobian_batch, and
        # test_attractor_convergence.
        X_flat = X_leaf.view(-1)  # (F,)
        try:
            jacobian = torch.autograd.functional.jacobian(
                _h_out_fn, X_flat, vectorize=True,
            )  # (H, F)
        finally:
            # Restore original state GUARANTEED, even on exceptions.
            _lif_cell._spike_history = _orig_spike_hist
            _lif_cell._dendritic_state = _orig_dend_state

        return jacobian, internals


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
