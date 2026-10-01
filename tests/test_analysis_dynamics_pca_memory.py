"""Memory-safe, exact PCA seam for ``scripts/analyze_dynamics.compute_pca_manifold``.

The production dynamics analysis extracts 2304 trials / 5,529,448 states with
``use_combined=True`` (GRU hidden H=64 then full LIF potential H=64 -> F=128).
The previous implementation materialised a *second* full ``(N, F)`` copy in
``feature_list`` and a *third* in ``all_states`` (plus a Python list of N
labels), i.e. ~5.66 GB of avoidable ``float32`` intermediates on top of the
2.83 GB already retained by the bundle.  That drove the process past the 8 GiB
cgroup limit and it was OOM-killed while *building* the PCA input, before
``PCA.fit`` ran.

These regressions pin the seam at ``compute_pca_manifold``:

* ``test_bounded_allocation_*`` is RED on the materialising implementation
  (peak scales with ``N * F``) and GREEN once the fit streams per trial.  It
  never allocates the 5.5M-state production tensor.
* The remaining tests pin the numerical contract: exact whole-state
  frame-weighted centered PCA (all frames, GRU-then-full-LIF column order),
  per-state label identity/order, all three feature modes, and rejection of
  rank-deficient/empty/mismatched/non-finite inputs.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sklearn.decomposition import PCA

from scripts.analyze_dynamics import DynamicsBundle, compute_pca_manifold


def _make_bundle(
    lengths,
    hidden_dim: int = 8,
    with_lif_potentials: bool = True,
    with_lif_rates: bool = True,
    seed: int = 0,
):
    """Deterministic bundle whose trajectories have the given lengths."""
    rng = np.random.default_rng(seed)
    bundle = DynamicsBundle()
    labels = [i % 4 for i in range(len(lengths))]
    for i, t_i in enumerate(lengths):
        bundle.gru_trajectories.append(
            torch.as_tensor(rng.standard_normal((t_i, hidden_dim)), dtype=torch.float32)
        )
        if with_lif_potentials:
            bundle.lif_potential_trajs.append(
                torch.as_tensor(
                    rng.standard_normal((t_i, hidden_dim)), dtype=torch.float32
                )
            )
        if with_lif_rates:
            bundle.lif_rate_trajs.append(rng.standard_normal(t_i))
        bundle.g_gru_trajs.append(rng.standard_normal(t_i))
        bundle.g_lif_trajs.append(rng.standard_normal(t_i))
        bundle.labels.append(labels[i])
    return bundle, labels


def _reference_features(bundle, use_combined: bool = True) -> np.ndarray:
    """Independent reference build of the exact concatenated feature matrix."""
    parts = []
    for i, traj_gru in enumerate(bundle.gru_trajectories):
        gru_np = traj_gru.numpy()
        if use_combined and bundle.lif_potential_trajs:
            parts.append(
                np.concatenate([gru_np, bundle.lif_potential_trajs[i].numpy()], axis=1)
            )
        elif use_combined and bundle.lif_rate_trajs:
            parts.append(
                np.concatenate(
                    [gru_np, np.asarray(bundle.lif_rate_trajs[i]).reshape(-1, 1)], axis=1
                )
            )
        else:
            parts.append(gru_np)
    return np.concatenate(parts, axis=0)


# ---------------------------------------------------------------------------
# Numerical contract: exact whole-state centered PCA
# ---------------------------------------------------------------------------

def test_combined_pca_matches_sklearn_exact_centered_pca():
    # Tall-skinny fixture -> sklearn auto-solver is covariance_eigh, same exact
    # solver the production shape selects.
    bundle, labels = _make_bundle([40] * 20, hidden_dim=8)  # N=800, F=16
    X = _reference_features(bundle, use_combined=True)
    assert X.shape == (800, 16)
    reference = PCA(n_components=3, svd_solver="covariance_eigh").fit(X)
    assert reference._fit_svd_solver == "covariance_eigh"

    trajectories_3d, all_labels, pca = compute_pca_manifold(
        bundle, n_components=3, use_combined=True
    )

    np.testing.assert_allclose(
        pca.explained_variance_ratio_, reference.explained_variance_ratio_, atol=1e-6
    )
    np.testing.assert_allclose(pca.explained_variance_, reference.explained_variance_, rtol=1e-5)
    np.testing.assert_allclose(pca.singular_values_, reference.singular_values_, rtol=1e-5)
    np.testing.assert_allclose(pca.mean_, reference.mean_, rtol=1e-6, atol=1e-6)
    # svd_flip(u_based_decision=False) makes the sign convention identical.
    np.testing.assert_allclose(pca.components_, reference.components_, atol=1e-5)
    assert pca.n_samples_ == reference.n_samples_
    assert pca.n_components_ == 3

    # Per-trajectory scores must equal the reference transform, in trial order.
    offset = 0
    for traj in trajectories_3d:
        n = traj.shape[0]
        np.testing.assert_allclose(
            traj, reference.transform(X[offset:offset + n]), atol=1e-4
        )
        offset += n
    assert offset == X.shape[0]


def test_feature_order_is_gru_then_full_lif():
    bundle, _ = _make_bundle([30] * 15, hidden_dim=6)  # F=12
    X = _reference_features(bundle, use_combined=True)
    _, _, pca = compute_pca_manifold(bundle, n_components=2, use_combined=True)
    assert pca.n_features_in_ == 12
    # Rebuilding with swapped columns must change the subspace: this proves the
    # implementation did not silently reorder GRU/LIF blocks.
    swapped = np.concatenate([X[:, 6:], X[:, :6]], axis=1)
    ref_swapped = PCA(n_components=2, svd_solver="covariance_eigh").fit(swapped)
    assert not np.allclose(pca.components_, ref_swapped.components_, atol=1e-6)


def test_labels_are_per_state_in_trial_order():
    lengths = [5, 3, 7, 2]
    bundle, labels = _make_bundle(lengths, hidden_dim=4)
    _, all_labels, _ = compute_pca_manifold(bundle, n_components=2, use_combined=True)
    expected = np.repeat(np.asarray(labels, dtype=np.int64), lengths)
    np.testing.assert_array_equal(all_labels, expected)


def test_skew_lengths_preserve_offsets_and_shapes():
    lengths = [1, 11, 3, 40, 7]
    bundle, _ = _make_bundle(lengths, hidden_dim=5)
    trajectories_3d, all_labels, pca = compute_pca_manifold(
        bundle, n_components=3, use_combined=True
    )
    assert [t.shape for t in trajectories_3d] == [(t, 3) for t in lengths]
    assert pca.n_features_in_ == 10
    assert all_labels.shape == (sum(lengths),)


def test_fallback_appends_lif_rate_column_without_potentials():
    bundle, _ = _make_bundle(
        [20] * 12, hidden_dim=6, with_lif_potentials=False, with_lif_rates=True
    )
    X = _reference_features(bundle, use_combined=True)
    assert X.shape == (240, 7)  # H + 1
    reference = PCA(n_components=3, svd_solver="covariance_eigh").fit(X)
    _, _, pca = compute_pca_manifold(bundle, n_components=3, use_combined=True)
    assert pca.n_features_in_ == 7
    np.testing.assert_allclose(
        pca.explained_variance_ratio_, reference.explained_variance_ratio_, atol=1e-6
    )


def test_no_lif_uses_gru_only():
    bundle, _ = _make_bundle(
        [20] * 12, hidden_dim=6, with_lif_potentials=False, with_lif_rates=False
    )
    X = _reference_features(bundle, use_combined=True)
    assert X.shape == (240, 6)
    reference = PCA(n_components=3, svd_solver="covariance_eigh").fit(X)
    _, _, pca = compute_pca_manifold(bundle, n_components=3, use_combined=True)
    assert pca.n_features_in_ == 6
    np.testing.assert_allclose(
        pca.explained_variance_ratio_, reference.explained_variance_ratio_, atol=1e-6
    )


def test_use_combined_false_uses_gru_only():
    bundle, _ = _make_bundle([20] * 12, hidden_dim=6)
    X = _reference_features(bundle, use_combined=False)
    assert X.shape == (240, 6)
    _, _, pca = compute_pca_manifold(bundle, n_components=3, use_combined=False)
    assert pca.n_features_in_ == 6


def test_rank_deficient_input_is_clipped_not_nan():
    # Constant columns -> zero eigenvalues; PCA must clip, never emit NaN.
    bundle = DynamicsBundle()
    for _ in range(6):
        bundle.gru_trajectories.append(torch.zeros(10, 4))
        bundle.lif_potential_trajs.append(torch.zeros(10, 4))
        bundle.labels.append(0)
    _, _, pca = compute_pca_manifold(bundle, n_components=3, use_combined=True)
    assert np.all(np.isfinite(pca.explained_variance_ratio_))
    assert np.all(pca.explained_variance_ratio_ >= 0.0)


# ---------------------------------------------------------------------------
# Rejection of invalid inputs (matching the original valid contract)
# ---------------------------------------------------------------------------

def test_empty_bundle_rejected():
    with pytest.raises(ValueError):
        compute_pca_manifold(DynamicsBundle(), n_components=3, use_combined=True)


def test_mismatched_lif_length_rejected():
    bundle, _ = _make_bundle([10, 10, 10], hidden_dim=4)
    bundle.lif_potential_trajs = bundle.lif_potential_trajs[:2]  # skew
    with pytest.raises(ValueError):
        compute_pca_manifold(bundle, n_components=2, use_combined=True)


def test_nonfinite_input_rejected():
    bundle, _ = _make_bundle([10, 10, 10], hidden_dim=4)
    bundle.gru_trajectories[1][0, 0] = float("nan")
    with pytest.raises(ValueError):
        compute_pca_manifold(bundle, n_components=2, use_combined=True)


def test_too_many_components_rejected():
    bundle, _ = _make_bundle([4, 4], hidden_dim=3)  # F=6, N=8
    with pytest.raises(ValueError):
        compute_pca_manifold(bundle, n_components=99, use_combined=True)


# ---------------------------------------------------------------------------
# Bounded allocation: the actual root-cause regression
# ---------------------------------------------------------------------------

def test_bounded_allocation_does_not_materialise_full_input():
    """Peak heap must not scale with the full ``(N, F)`` concatenation.

    On the materialising implementation the peak is ~2 * N * F * 4 bytes
    (``feature_list`` + ``all_states``) plus the label list; here that is
    ~200 MB.  A bounded, streaming fit keeps it to a few MB.  The fixture is
    sized so this is a faithful, cheap proxy for the 5.5M-state production
    tensor (~5.66 GB) that OOM-killed the real run, without ever allocating it.
    """
    import tracemalloc

    n_trials = 2000
    t_i = 100
    hidden_dim = 64  # combined F = 128, matching production
    lengths = [t_i] * n_trials  # N = 200_000 states
    bundle, _ = _make_bundle(lengths, hidden_dim=hidden_dim)

    full_bytes = n_trials * t_i * (2 * hidden_dim) * 4  # one float32 (N,F) copy
    assert full_bytes > 100_000_000  # fixture genuinely exercises the seam

    tracemalloc.start()
    try:
        trajectories_3d, all_labels, pca = compute_pca_manifold(
            bundle, n_components=3, use_combined=True
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert len(trajectories_3d) == n_trials
    assert pca.n_features_in_ == 2 * hidden_dim
    assert all_labels.shape == (n_trials * t_i,)
    # Bounded: allow a generous ceiling well below a single full-input copy.
    assert peak < full_bytes / 4, f"peak={peak} bytes vs full copy {full_bytes}"


def test_no_axis0_full_corpus_concatenation(monkeypatch):
    """The fit must never stack every trajectory along axis 0."""
    lengths = [40] * 500  # 20_000 states
    h = 128
    bundle, _ = _make_bundle(lengths, hidden_dim=h)

    axis0_calls = []
    real_concatenate = np.concatenate

    def guard(arrays, axis=0, *args, **kwargs):
        if axis in (0, -2):
            axis0_calls.append(axis)
        return real_concatenate(arrays, axis=axis, *args, **kwargs)

    monkeypatch.setattr(np, "concatenate", guard)
    compute_pca_manifold(bundle, n_components=3, use_combined=True)

    assert axis0_calls == [], "full-corpus axis=0 concatenation was reintroduced"


# ---------------------------------------------------------------------------
# Numerical robustness: nonzero offset, unequal lengths, degenerate shapes
# ---------------------------------------------------------------------------

def test_high_offset_low_variance_is_not_cancelled():
    """A large per-feature offset must not destroy the tiny true variance.

    A raw ``X^T X - n * mean * mean`` Gram catastrophically cancels here and
    can fabricate a clipped PSD spectrum; merging *centered* moments must match
    an explicit full-SVD reference.
    """
    rng = np.random.default_rng(7)
    n, f = 120, 6
    offset = 1e8
    raw = (rng.standard_normal((n, f)) + offset).astype(np.float64)
    blocks = [raw[:40], raw[40:70], raw[70:]]  # unequal lengths
    bundle = DynamicsBundle()
    for block in blocks:
        bundle.gru_trajectories.append(torch.as_tensor(block[:, :3], dtype=torch.float64))
        bundle.lif_potential_trajs.append(torch.as_tensor(block[:, 3:], dtype=torch.float64))
        bundle.labels.append(0)
    X = np.concatenate(blocks, axis=0)

    _, _, pca = compute_pca_manifold(bundle, n_components=3, use_combined=True)
    reference = PCA(n_components=3, svd_solver="full").fit(X)

    np.testing.assert_allclose(
        pca.explained_variance_, reference.explained_variance_, rtol=1e-6, atol=1e-9
    )
    assert pca.explained_variance_[0] > 0.1  # genuine variance survived, not ~0


def test_unequal_lengths_all_states_weighted_equally():
    """Every state (not every trial) carries equal weight in the mean/cov."""
    rng = np.random.default_rng(11)
    lengths = [1, 50, 3, 90, 2]
    bundle = DynamicsBundle()
    full = []
    for t_i in lengths:
        block = rng.standard_normal((t_i, 5)) + 3.0
        full.append(block)
        bundle.gru_trajectories.append(torch.as_tensor(block, dtype=torch.float64))
        bundle.labels.append(0)
    bundle.lif_potential_trajs = []
    bundle.lif_rate_trajs = []
    X = np.concatenate(full, axis=0)

    _, _, pca = compute_pca_manifold(bundle, n_components=2, use_combined=False)
    reference = PCA(n_components=2, svd_solver="covariance_eigh").fit(X)
    np.testing.assert_allclose(pca.mean_, reference.mean_, rtol=1e-9)
    np.testing.assert_allclose(
        pca.explained_variance_ratio_, reference.explained_variance_ratio_, rtol=1e-8
    )


def test_singleton_corpus_rejected():
    bundle = DynamicsBundle()
    bundle.gru_trajectories.append(torch.zeros(1, 4))
    bundle.lif_potential_trajs = []
    bundle.lif_rate_trajs = []
    bundle.labels = [0]
    with pytest.raises(ValueError, match="at least 2 states"):
        compute_pca_manifold(bundle, n_components=1, use_combined=False)


def test_non_integer_n_components_rejected():
    bundle, _ = _make_bundle([10] * 6, hidden_dim=4)
    with pytest.raises(ValueError, match="integer"):
        compute_pca_manifold(bundle, n_components=2.5, use_combined=True)


def test_label_count_mismatch_rejected():
    bundle, _ = _make_bundle([10, 10, 10], hidden_dim=4)
    bundle.labels = bundle.labels[:2]
    with pytest.raises(ValueError, match="Label count"):
        compute_pca_manifold(bundle, n_components=2, use_combined=True)


def test_mismatched_feature_dim_rejected():
    bundle = DynamicsBundle()
    bundle.gru_trajectories = [torch.zeros(4, 3), torch.zeros(4, 5)]
    bundle.lif_potential_trajs = []
    bundle.lif_rate_trajs = []
    bundle.labels = [0, 1]
    with pytest.raises(ValueError, match="Inconsistent feature dimension"):
        compute_pca_manifold(bundle, n_components=3, use_combined=False)


@pytest.mark.parametrize(
    "with_lif_potentials,use_combined",
    [(True, True), (False, True), (True, False), (False, False)],
)
def test_all_modes_match_sklearn_float64_reference(with_lif_potentials, use_combined):
    """Strict float64 agreement with sklearn across every feature mode."""
    lengths = [5, 8, 3, 7, 11]
    bundle, _ = _make_bundle(
        lengths, hidden_dim=6,
        with_lif_potentials=with_lif_potentials,
        with_lif_rates=not with_lif_potentials,
    )
    # Fit the same blocks in float64 so the covariance algebra is compared
    # without float32 rounding noise.
    for attr in ("gru_trajectories", "lif_potential_trajs"):
        setattr(
            bundle, attr,
            [t.double() for t in getattr(bundle, attr)],
        )
    bundle.lif_rate_trajs = [np.asarray(r, dtype=np.float64) for r in bundle.lif_rate_trajs]
    X = _reference_features(bundle, use_combined=use_combined)

    trajectories_3d, _, pca = compute_pca_manifold(
        bundle, n_components=3, use_combined=use_combined
    )
    ref = PCA(n_components=3).fit(X)

    np.testing.assert_allclose(pca.mean_, ref.mean_, atol=1e-9)
    np.testing.assert_allclose(pca.explained_variance_, ref.explained_variance_, rtol=1e-8)
    np.testing.assert_allclose(
        pca.explained_variance_ratio_, ref.explained_variance_ratio_, rtol=1e-8
    )
    np.testing.assert_allclose(pca.singular_values_, ref.singular_values_, rtol=1e-8)
    np.testing.assert_allclose(pca.noise_variance_, ref.noise_variance_, rtol=1e-8)
    assert pca.n_samples_ == ref.n_samples_
    for k in range(3):
        overlap = abs(float(pca.components_[k] @ ref.components_[k]))
        assert overlap == pytest.approx(1.0, abs=1e-6)

    ref_scores = ref.transform(X)
    offset = 0
    for traj in trajectories_3d:
        t_i = traj.shape[0]
        np.testing.assert_allclose(traj, ref_scores[offset:offset + t_i], atol=1e-6)
        offset += t_i
