"""
Tests for signal_bridge.ic.compute_ic — GOV-02.

Verifies that the extracted pure-pandas IC utility produces correct Spearman
correlations and handles all edge cases identically to the original inline logic
in feedback.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_bridge.ic import compute_ic


class TestComputeIc:
    """GOV-02: compute_ic() is the single shared IC utility."""

    def test_perfect_positive_correlation(self):
        n = 20
        signal = pd.Series(np.arange(float(n)))
        returns = pd.Series(np.arange(float(n)) * 0.01)
        mask = pd.Series([True] * n)
        ic = compute_ic(signal, returns, mask)
        assert isinstance(ic, float)
        assert abs(ic - 1.0) < 1e-6, f"expected ~1.0, got {ic}"

    def test_perfect_negative_correlation(self):
        n = 20
        signal = pd.Series(np.arange(float(n)))
        returns = pd.Series(np.arange(float(n), 0.0, -1.0) * 0.01)
        mask = pd.Series([True] * n)
        ic = compute_ic(signal, returns, mask)
        assert isinstance(ic, float)
        assert abs(ic + 1.0) < 1e-6, f"expected ~-1.0, got {ic}"

    def test_all_false_mask_returns_zero(self):
        n = 20
        signal = pd.Series(np.arange(float(n)))
        returns = pd.Series(np.random.default_rng(42).random(n))
        mask = pd.Series([False] * n)
        ic = compute_ic(signal, returns, mask)
        assert ic == 0.0

    def test_single_valid_row_returns_zero(self):
        signal = pd.Series([1.0, 2.0, 3.0])
        returns = pd.Series([0.1, 0.2, 0.3])
        mask = pd.Series([False, False, True])  # only 1 valid row
        ic = compute_ic(signal, returns, mask)
        assert ic == 0.0

    def test_constant_signal_returns_zero(self):
        n = 20
        signal = pd.Series([1.0] * n)  # constant — Spearman undefined
        returns = pd.Series(np.arange(float(n)) * 0.01)
        mask = pd.Series([True] * n)
        ic = compute_ic(signal, returns, mask)
        assert ic == 0.0

    def test_return_type_is_float(self):
        signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = pd.Series([True] * 5)
        ic = compute_ic(signal, returns, mask)
        assert type(ic) is float  # not np.float64

    def test_partial_mask_excludes_invalid_rows(self):
        # First 5 rows valid with positive correlation; last 5 rows masked out (would break correlation)
        signal = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 9.0, 8.0, 7.0, 6.0])
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5, -0.5, -0.4, -0.3, -0.2, -0.1])
        mask = pd.Series([True, True, True, True, True, False, False, False, False, False])
        ic = compute_ic(signal, returns, mask)
        assert abs(ic - 1.0) < 1e-6, f"partial mask should yield ~1.0, got {ic}"

    def test_nan_in_signal_dropped(self):
        signal = pd.Series([1.0, float("nan"), 3.0, 4.0, 5.0])
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
        mask = pd.Series([True] * 5)
        # NaN row dropped — 4 valid rows remain, should still produce a result
        ic = compute_ic(signal, returns, mask)
        assert isinstance(ic, float)
        # 4-point correlation of [1,3,4,5] vs [0.1,0.3,0.4,0.5] is perfect (monotone)
        assert abs(ic - 1.0) < 1e-6, f"expected ~1.0 after NaN drop, got {ic}"

    def test_result_in_valid_range(self):
        rng = np.random.default_rng(0)
        n = 50
        signal = pd.Series(rng.standard_normal(n))
        returns = pd.Series(rng.standard_normal(n))
        mask = pd.Series([True] * n)
        ic = compute_ic(signal, returns, mask)
        assert -1.0 <= ic <= 1.0, f"IC {ic} out of valid range"
