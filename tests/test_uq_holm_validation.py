"""holm_bonferroni input-domain validation (T4).

The correction is only defined on genuine p-values in [0, 1].  A
non-finite or out-of-range value silently corrupts the sort order and
the step-down thresholds, so it must be rejected before sorting.
"""
from __future__ import annotations

import math

import pytest

from nsmor.analysis.uq import holm_bonferroni


@pytest.mark.parametrize("bad", [
    float("nan"), float("inf"), -float("inf"), -0.001, 1.0001, 2.0, -1.0,
])
def test_holm_rejects_nonfinite_or_out_of_range_p(bad: float) -> None:
    with pytest.raises(ValueError, match="p-value"):
        holm_bonferroni({"good": 0.01, "bad": bad})


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_holm_accepts_boundary_p_values(boundary: float) -> None:
    result = holm_bonferroni({"edge": boundary, "other": 0.5})
    assert set(result) == {"edge", "other"}
    adjusted, significant = result["edge"]
    assert 0.0 <= adjusted <= 1.0
    assert isinstance(significant, bool)


def test_holm_ordinary_inputs_unchanged() -> None:
    result = holm_bonferroni({"a": 0.01, "b": 0.04, "c": 0.5})
    assert result["a"] == (pytest.approx(0.03), True)
    assert result["b"] == (pytest.approx(0.08), False)
    assert result["c"] == (pytest.approx(0.5), False)
    assert math.isfinite(result["a"][0])


def test_holm_empty_mapping_still_empty() -> None:
    assert holm_bonferroni({}) == {}
