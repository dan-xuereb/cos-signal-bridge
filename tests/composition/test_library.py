"""Tests for SignalLibrary: registration, query, filter, and YAML loading."""
from __future__ import annotations

import yaml
import pytest
from pathlib import Path

from signal_bridge.composition.library import SignalLibrary
from signal_bridge.composition.models import SignalMeta
from signal_bridge.composition.types import HorizonCategory, Polarity


@pytest.fixture
def signal_library() -> SignalLibrary:
    """Return a fresh empty SignalLibrary."""
    return SignalLibrary()


@pytest.fixture
def sample_signal_meta(sample_signal_kwargs: dict) -> SignalMeta:
    """Construct a SignalMeta from the shared sample_signal_kwargs fixture."""
    return SignalMeta(**sample_signal_kwargs)


@pytest.fixture
def populated_library(signal_library: SignalLibrary) -> SignalLibrary:
    """Library with 3 signals: positive/FAST/price_momentum, negative/SLOW/onchain_valuation, positive/MEDIUM/macro_rates."""
    signal_library.register(
        SignalMeta(
            name="fast_momentum",
            display_name="Fast Momentum",
            polarity=Polarity.POSITIVE,
            horizon_category=HorizonCategory.FAST,
            half_life_days=3.0,
            theme="price_momentum",
            update_frequency="1D",
            economic_rationale="Short-term trend following.",
            data_source="sgl_btcforge",
        )
    )
    signal_library.register(
        SignalMeta(
            name="mvrv_ratio",
            display_name="MVRV Ratio",
            polarity=Polarity.NEGATIVE,
            horizon_category=HorizonCategory.SLOW,
            half_life_days=40.0,
            theme="onchain_valuation",
            update_frequency="1D",
            economic_rationale="Market overvaluation indicator.",
            data_source="warehouse_api",
        )
    )
    signal_library.register(
        SignalMeta(
            name="yield_curve",
            display_name="Yield Curve Slope",
            polarity=Polarity.POSITIVE,
            horizon_category=HorizonCategory.MEDIUM,
            half_life_days=15.0,
            theme="macro_rates",
            update_frequency="1D",
            economic_rationale="Steeper curve indicates growth.",
            data_source="sgl_fred",
        )
    )
    return signal_library


# --- Registration and retrieval ---


def test_register_and_get(signal_library: SignalLibrary, sample_signal_meta: SignalMeta) -> None:
    signal_library.register(sample_signal_meta)
    result = signal_library.get(sample_signal_meta.name)
    assert result == sample_signal_meta


def test_duplicate_name_raises(signal_library: SignalLibrary, sample_signal_meta: SignalMeta) -> None:
    signal_library.register(sample_signal_meta)
    with pytest.raises(ValueError, match="already registered"):
        signal_library.register(sample_signal_meta)


def test_duplicate_name_lower_version_raises(
    signal_library: SignalLibrary, sample_signal_kwargs: dict
) -> None:
    meta_v2 = SignalMeta(**{**sample_signal_kwargs, "version": 2})
    signal_library.register(meta_v2)
    meta_v1 = SignalMeta(**{**sample_signal_kwargs, "version": 1})
    with pytest.raises(ValueError, match="already registered"):
        signal_library.register(meta_v1)


def test_version_bump_overwrites(
    signal_library: SignalLibrary, sample_signal_kwargs: dict
) -> None:
    meta_v1 = SignalMeta(**sample_signal_kwargs)
    signal_library.register(meta_v1)
    meta_v2 = SignalMeta(**{**sample_signal_kwargs, "version": 2})
    signal_library.register(meta_v2)
    result = signal_library.get(meta_v1.name)
    assert result.version == 2


def test_get_missing_raises(signal_library: SignalLibrary) -> None:
    with pytest.raises(KeyError):
        signal_library.get("nonexistent_signal")


# --- list_all ---


def test_list_all(populated_library: SignalLibrary) -> None:
    signals = populated_library.list_all()
    assert len(signals) == 3
    names = {s.name for s in signals}
    assert names == {"fast_momentum", "mvrv_ratio", "yield_curve"}


def test_list_all_empty(signal_library: SignalLibrary) -> None:
    assert signal_library.list_all() == []


# --- filter ---


def test_filter_by_theme(populated_library: SignalLibrary) -> None:
    results = populated_library.filter(theme="price_momentum")
    assert len(results) == 1
    assert results[0].name == "fast_momentum"


def test_filter_by_horizon(populated_library: SignalLibrary) -> None:
    results = populated_library.filter(horizon=HorizonCategory.FAST)
    assert len(results) == 1
    assert results[0].name == "fast_momentum"


def test_filter_by_polarity(populated_library: SignalLibrary) -> None:
    results = populated_library.filter(polarity=Polarity.NEGATIVE)
    assert len(results) == 1
    assert results[0].name == "mvrv_ratio"


def test_filter_no_args_returns_all(populated_library: SignalLibrary) -> None:
    results = populated_library.filter()
    assert len(results) == 3


def test_filter_multiple_args_and_logic(populated_library: SignalLibrary) -> None:
    results = populated_library.filter(
        polarity=Polarity.POSITIVE, horizon=HorizonCategory.MEDIUM
    )
    assert len(results) == 1
    assert results[0].name == "yield_curve"


# --- from_directory ---


def test_from_directory(tmp_path: Path) -> None:
    signals_data = {
        "signals": [
            {
                "name": "sig_a",
                "display_name": "Signal A",
                "polarity": "positive",
                "horizon_category": "fast",
                "half_life_days": 5.0,
                "theme": "test_theme",
                "update_frequency": "1D",
                "economic_rationale": "Test A.",
                "data_source": "test",
            },
            {
                "name": "sig_b",
                "display_name": "Signal B",
                "polarity": "negative",
                "horizon_category": "slow",
                "half_life_days": 30.0,
                "theme": "test_theme",
                "update_frequency": "1W",
                "economic_rationale": "Test B.",
                "data_source": "test",
            },
        ]
    }
    (tmp_path / "test.yaml").write_text(yaml.dump(signals_data))

    lib = SignalLibrary.from_directory(tmp_path)
    assert len(lib) == 2
    assert lib.get("sig_a").polarity == Polarity.POSITIVE
    assert lib.get("sig_b").horizon_category == HorizonCategory.SLOW


def test_from_directory_multiple_files(tmp_path: Path) -> None:
    file1 = {
        "signals": [
            {
                "name": "file1_sig",
                "display_name": "File 1 Signal",
                "polarity": "positive",
                "horizon_category": "medium",
                "half_life_days": 10.0,
                "theme": "theme_a",
                "update_frequency": "1D",
                "economic_rationale": "From file 1.",
                "data_source": "test",
            }
        ]
    }
    file2 = {
        "signals": [
            {
                "name": "file2_sig",
                "display_name": "File 2 Signal",
                "polarity": "negative",
                "horizon_category": "cycle",
                "half_life_days": 100.0,
                "theme": "theme_b",
                "update_frequency": "1M",
                "economic_rationale": "From file 2.",
                "data_source": "test",
            }
        ]
    }
    (tmp_path / "a.yaml").write_text(yaml.dump(file1))
    (tmp_path / "b.yaml").write_text(yaml.dump(file2))

    lib = SignalLibrary.from_directory(tmp_path)
    assert len(lib) == 2
    assert "file1_sig" in lib
    assert "file2_sig" in lib


def test_from_directory_invalid_data_raises(tmp_path: Path) -> None:
    bad_data = {
        "signals": [
            {
                "name": "bad_signal",
                # Missing required fields
            }
        ]
    }
    (tmp_path / "bad.yaml").write_text(yaml.dump(bad_data))
    with pytest.raises(Exception):  # ValidationError from Pydantic
        SignalLibrary.from_directory(tmp_path)


# --- __len__ ---


def test_len(populated_library: SignalLibrary) -> None:
    assert len(populated_library) == 3


def test_len_empty(signal_library: SignalLibrary) -> None:
    assert len(signal_library) == 0


# --- Integration: bundled YAML catalogs ---


def test_load_bundled_catalogs() -> None:
    """Load all 3 bundled YAML catalogs and verify signal coverage."""
    lib = SignalLibrary.from_packaged_signals()

    # Total count: 4 price + 4 onchain + 3 macro = 11
    assert len(lib) >= 11

    # At least one negative polarity signal exists
    neg = lib.filter(polarity=Polarity.NEGATIVE)
    assert len(neg) >= 4  # volatility, mvrv, exchange_net_flow, fci, dxy

    # At least one CYCLE horizon signal exists
    cycle = lib.filter(horizon=HorizonCategory.CYCLE)
    assert len(cycle) >= 1

    # Theme-based filtering
    onchain_val = lib.filter(theme="onchain_valuation")
    assert len(onchain_val) >= 1

    price_mom = lib.filter(theme="price_momentum")
    assert len(price_mom) >= 2

    # Spot-check a specific signal
    mvrv = lib.get("mvrv_ratio")
    assert mvrv.polarity == Polarity.NEGATIVE
    assert mvrv.theme == "onchain_valuation"
    assert mvrv.data_source == "warehouse_api"
