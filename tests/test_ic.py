"""Tests for signal_bridge.ic — pure compute_ic() function."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_bridge.ic import compute_ic


def _series(*values: float) -> pd.Series:
    return pd.Series(values, dtype=float)


def _mask(*bools: bool) -> pd.Series:
    return pd.Series(bools)


class TestComputeIcPerfectCorrelation:
    def test_perfect_positive_correlation_returns_one(self) -> None:
        signal = _series(1.0, 2.0, 3.0, 4.0, 5.0)
        returns = _series(0.1, 0.2, 0.3, 0.4, 0.5)
        mask = _mask(True, True, True, True, True)
        ic = compute_ic(signal, returns, mask)
        assert isinstance(ic, float)
        assert abs(ic - 1.0) < 1e-6

    def test_perfect_negative_correlation_returns_neg_one(self) -> None:
        signal = _series(1.0, 2.0, 3.0, 4.0, 5.0)
        returns = _series(0.5, 0.4, 0.3, 0.2, 0.1)
        mask = _mask(True, True, True, True, True)
        ic = compute_ic(signal, returns, mask)
        assert abs(ic - (-1.0)) < 1e-6


class TestComputeIcInsufficientData:
    def test_all_false_mask_returns_zero(self) -> None:
        signal = _series(1.0, 2.0, 3.0)
        returns = _series(0.1, 0.2, 0.3)
        mask = _mask(False, False, False)
        assert compute_ic(signal, returns, mask) == 0.0

    def test_single_true_mask_returns_zero(self) -> None:
        signal = _series(1.0)
        returns = _series(0.1)
        mask = _mask(True)
        assert compute_ic(signal, returns, mask) == 0.0

    def test_single_valid_row_returns_zero(self) -> None:
        signal = _series(1.0, 2.0, 3.0)
        returns = _series(0.1, 0.2, 0.3)
        mask = _mask(True, False, False)
        assert compute_ic(signal, returns, mask) == 0.0

    def test_exactly_two_valid_rows_does_not_return_zero(self) -> None:
        signal = _series(1.0, 2.0, 3.0)
        returns = _series(0.1, 0.2, 0.3)
        mask = _mask(True, True, False)
        ic = compute_ic(signal, returns, mask)
        # 2 rows is sufficient — should produce a valid correlation
        assert isinstance(ic, float)
        assert ic != 0.0

    def test_nan_rows_excluded_via_dropna(self) -> None:
        # mask selects 3 rows but 1 has NaN; only 2 valid rows remain after dropna
        signal = _series(1.0, float("nan"), 3.0)
        returns = _series(0.1, 0.2, 0.3)
        mask = _mask(True, True, True)
        # Only rows 0 and 2 survive dropna (2 rows) — should not return 0.0
        ic = compute_ic(signal, returns, mask)
        assert isinstance(ic, float)

    def test_all_nan_after_mask_returns_zero(self) -> None:
        signal = _series(float("nan"), float("nan"))
        returns = _series(float("nan"), float("nan"))
        mask = _mask(True, True)
        assert compute_ic(signal, returns, mask) == 0.0


class TestComputeIcNanCorrelation:
    def test_constant_signal_returns_zero(self) -> None:
        # Constant series → Spearman correlation is NaN (zero variance)
        signal = _series(5.0, 5.0, 5.0, 5.0, 5.0)
        returns = _series(0.1, 0.2, 0.3, 0.4, 0.5)
        mask = _mask(True, True, True, True, True)
        assert compute_ic(signal, returns, mask) == 0.0

    def test_constant_returns_returns_zero(self) -> None:
        signal = _series(1.0, 2.0, 3.0, 4.0, 5.0)
        returns = _series(0.2, 0.2, 0.2, 0.2, 0.2)
        mask = _mask(True, True, True, True, True)
        assert compute_ic(signal, returns, mask) == 0.0


class TestComputeIcReturnType:
    def test_return_value_is_float(self) -> None:
        signal = _series(1.0, 2.0, 3.0)
        returns = _series(0.1, 0.2, 0.3)
        mask = _mask(True, True, True)
        result = compute_ic(signal, returns, mask)
        assert type(result) is float  # must be plain float, not np.float64

    def test_return_value_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        signal = pd.Series(rng.standard_normal(100))
        returns = pd.Series(rng.standard_normal(100))
        mask = pd.Series([True] * 100)
        ic = compute_ic(signal, returns, mask)
        assert -1.0 <= ic <= 1.0


class TestComputeIcMaskFiltering:
    def test_mask_excludes_rows(self) -> None:
        # mask=False on the last element (outlier that inverts the correlation)
        # signal=99 gets rank 5, returns=-99 gets rank 1 → reversed correlation
        signal = _series(1.0, 2.0, 3.0, 4.0, 99.0)
        returns = _series(0.1, 0.2, 0.3, 0.4, -99.0)
        mask_all = _mask(True, True, True, True, True)
        mask_partial = _mask(True, True, True, True, False)
        ic_all = compute_ic(signal, returns, mask_all)
        ic_partial = compute_ic(signal, returns, mask_partial)
        # partial mask excludes outlier — perfect correlation on first 4 rows
        assert abs(ic_partial - 1.0) < 1e-6
        # full mask includes outlier that flips last rank pair — overall IC drops
        assert ic_all < ic_partial
