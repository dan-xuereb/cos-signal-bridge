"""Tests for COS-CIE type enums: Polarity and HorizonCategory."""
from __future__ import annotations

from enum import Enum

import pytest

from signal_bridge.composition.types import HorizonCategory, Polarity


class TestPolarity:
    """Tests for the Polarity enum."""

    def test_polarity_has_exactly_two_members(self) -> None:
        assert len(Polarity) == 2

    def test_polarity_positive_value(self) -> None:
        assert Polarity.POSITIVE.value == "positive"

    def test_polarity_negative_value(self) -> None:
        assert Polarity.NEGATIVE.value == "negative"

    def test_polarity_str_coercion_positive(self) -> None:
        assert Polarity("positive") == Polarity.POSITIVE

    def test_polarity_str_coercion_negative(self) -> None:
        assert Polarity("negative") == Polarity.NEGATIVE

    def test_polarity_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            Polarity("invalid")

    def test_polarity_inherits_str_enum(self) -> None:
        assert issubclass(Polarity, str)
        assert issubclass(Polarity, Enum)

    def test_polarity_is_str_comparable(self) -> None:
        assert Polarity.POSITIVE == "positive"
        assert Polarity.NEGATIVE == "negative"


class TestHorizonCategory:
    """Tests for the HorizonCategory enum."""

    def test_horizon_has_exactly_four_members(self) -> None:
        assert len(HorizonCategory) == 4

    def test_horizon_fast_value(self) -> None:
        assert HorizonCategory.FAST.value == "fast"

    def test_horizon_medium_value(self) -> None:
        assert HorizonCategory.MEDIUM.value == "medium"

    def test_horizon_slow_value(self) -> None:
        assert HorizonCategory.SLOW.value == "slow"

    def test_horizon_cycle_value(self) -> None:
        assert HorizonCategory.CYCLE.value == "cycle"

    def test_horizon_str_coercion(self) -> None:
        assert HorizonCategory("fast") == HorizonCategory.FAST

    def test_horizon_invalid_raises(self) -> None:
        with pytest.raises(ValueError):
            HorizonCategory("invalid")

    def test_horizon_inherits_str_enum(self) -> None:
        assert issubclass(HorizonCategory, str)
        assert issubclass(HorizonCategory, Enum)
