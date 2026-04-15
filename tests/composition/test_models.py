"""Tests for SignalMeta frozen Pydantic model."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from signal_bridge.composition.models import SignalMeta
from signal_bridge.composition.types import HorizonCategory, Polarity


class TestSignalMetaConstruction:
    """Tests for successful SignalMeta construction."""

    def test_signal_meta_construction(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.name == "test_momentum"
        assert meta.display_name == "Test Momentum Signal"
        assert meta.polarity == Polarity.POSITIVE
        assert meta.horizon_category == HorizonCategory.MEDIUM
        assert meta.half_life_days == 10.0
        assert meta.theme == "price_momentum"
        assert meta.update_frequency == "1D"
        assert meta.economic_rationale == "Test signal for unit testing."
        assert meta.data_source == "sgl"

    def test_signal_meta_version_defaults_to_1(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.version == 1

    def test_signal_meta_lookback_bars_defaults_to_0(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.lookback_bars == 0

    def test_signal_meta_custom_version(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs, version=3)
        assert meta.version == 3

    def test_signal_meta_custom_lookback(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs, lookback_bars=50)
        assert meta.lookback_bars == 50


class TestSignalMetaFrozen:
    """Tests for immutability."""

    def test_signal_meta_frozen(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        with pytest.raises(ValidationError):
            meta.name = "different"  # type: ignore[misc]


class TestSignalMetaValidation:
    """Tests for field validation and rejection of invalid inputs."""

    def test_signal_meta_missing_name(self, sample_signal_kwargs: dict) -> None:
        del sample_signal_kwargs["name"]
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)

    def test_signal_meta_missing_polarity(self, sample_signal_kwargs: dict) -> None:
        del sample_signal_kwargs["polarity"]
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)

    def test_signal_meta_missing_data_source(self, sample_signal_kwargs: dict) -> None:
        del sample_signal_kwargs["data_source"]
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)

    def test_signal_meta_invalid_polarity(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = "invalid"
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)

    def test_signal_meta_half_life_negative(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["half_life_days"] = -1.0
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)

    def test_signal_meta_half_life_zero(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["half_life_days"] = 0.0
        with pytest.raises(ValidationError):
            SignalMeta(**sample_signal_kwargs)


class TestPolarityCoercion:
    """Tests for integer polarity coercion per D-03/D-07."""

    def test_polarity_coercion_positive_int(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = 1
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.polarity == Polarity.POSITIVE

    def test_polarity_coercion_negative_int(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = -1
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.polarity == Polarity.NEGATIVE

    def test_polarity_coercion_positive_str(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = "+1"
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.polarity == Polarity.POSITIVE

    def test_polarity_coercion_negative_str(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = "-1"
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.polarity == Polarity.NEGATIVE

    def test_polarity_coercion_string_name(self, sample_signal_kwargs: dict) -> None:
        sample_signal_kwargs["polarity"] = "positive"
        meta = SignalMeta(**sample_signal_kwargs)
        assert meta.polarity == Polarity.POSITIVE


class TestSignalMetaRoundTrip:
    """Tests for serialization/deserialization."""

    def test_model_dump_produces_dict(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        dumped = meta.model_dump()
        assert isinstance(dumped, dict)
        assert dumped["name"] == "test_momentum"

    def test_model_validate_round_trip(self, sample_signal_kwargs: dict) -> None:
        meta = SignalMeta(**sample_signal_kwargs)
        dumped = meta.model_dump()
        restored = SignalMeta.model_validate(dumped)
        assert restored == meta


class TestSignalMetaEquality:
    """Tests for equality comparison."""

    def test_two_identical_signal_meta_are_equal(self, sample_signal_kwargs: dict) -> None:
        meta1 = SignalMeta(**sample_signal_kwargs)
        meta2 = SignalMeta(**sample_signal_kwargs)
        assert meta1 == meta2
