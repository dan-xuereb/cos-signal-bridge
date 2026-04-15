"""Tests for apply_polarity helper function."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from signal_bridge.composition.models import SignalMeta
from signal_bridge.composition.polarity import apply_polarity
from signal_bridge.composition.types import Polarity


@pytest.fixture
def positive_meta(sample_signal_kwargs: dict) -> SignalMeta:
    """SignalMeta with POSITIVE polarity."""
    return SignalMeta(**{**sample_signal_kwargs, "polarity": Polarity.POSITIVE})


@pytest.fixture
def negative_meta(sample_signal_kwargs: dict) -> SignalMeta:
    """SignalMeta with NEGATIVE polarity."""
    return SignalMeta(**{**sample_signal_kwargs, "polarity": Polarity.NEGATIVE})


class TestApplyPolarityScalar:
    """Scalar float/int inputs."""

    def test_positive_polarity_leaves_positive_value_unchanged(
        self, positive_meta: SignalMeta
    ) -> None:
        assert apply_polarity(5.0, positive_meta) == 5.0

    def test_positive_polarity_leaves_negative_value_unchanged(
        self, positive_meta: SignalMeta
    ) -> None:
        assert apply_polarity(-3.0, positive_meta) == -3.0

    def test_negative_polarity_flips_positive_value(
        self, negative_meta: SignalMeta
    ) -> None:
        assert apply_polarity(5.0, negative_meta) == -5.0

    def test_negative_polarity_flips_negative_value(
        self, negative_meta: SignalMeta
    ) -> None:
        assert apply_polarity(-3.0, negative_meta) == 3.0

    def test_negative_polarity_zero_unchanged(
        self, negative_meta: SignalMeta
    ) -> None:
        assert apply_polarity(0.0, negative_meta) == 0.0

    def test_negative_polarity_int_input(
        self, negative_meta: SignalMeta
    ) -> None:
        result = apply_polarity(5, negative_meta)
        assert result == -5


class TestApplyPolarityArray:
    """Numpy array and pandas Series inputs."""

    def test_numpy_array_negative_polarity(
        self, negative_meta: SignalMeta
    ) -> None:
        arr = np.array([1.0, -2.0, 3.0])
        result = apply_polarity(arr, negative_meta)
        np.testing.assert_array_equal(result, np.array([-1.0, 2.0, -3.0]))

    def test_pandas_series_negative_polarity(
        self, negative_meta: SignalMeta
    ) -> None:
        series = pd.Series([1.0, -2.0, 3.0])
        result = apply_polarity(series, negative_meta)
        pd.testing.assert_series_equal(result, pd.Series([-1.0, 2.0, -3.0]))


class TestApplyPolarityTypeSignature:
    """Verify the function signature accepts documented types."""

    def test_signature_accepts_union_types(self) -> None:
        import inspect

        sig = inspect.signature(apply_polarity)
        params = list(sig.parameters.keys())
        assert params == ["value", "meta"]
        # Check type annotations exist
        value_annotation = sig.parameters["value"].annotation
        meta_annotation = sig.parameters["meta"].annotation
        assert value_annotation is not inspect.Parameter.empty
        assert meta_annotation is not inspect.Parameter.empty
