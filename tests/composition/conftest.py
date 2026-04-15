from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from signal_bridge.composition.models import SignalMeta


@pytest.fixture
def sample_signal_kwargs() -> dict:
    return {
        "name": "test_momentum",
        "display_name": "Test Momentum Signal",
        "polarity": "positive",
        "horizon_category": "medium",
        "half_life_days": 10.0,
        "theme": "price_momentum",
        "update_frequency": "1D",
        "economic_rationale": "Test signal for unit testing.",
        "data_source": "sgl",
    }


@pytest.fixture
def sample_signal_meta(sample_signal_kwargs: dict) -> SignalMeta:
    """Construct a SignalMeta from the shared sample_signal_kwargs fixture."""
    return SignalMeta(**sample_signal_kwargs)


@pytest.fixture
def yaml_signals_dir(tmp_path: Path) -> Path:
    """Create a temp directory with 2 YAML signal catalog files."""
    file1 = {
        "signals": [
            {
                "name": "yaml_sig_1",
                "display_name": "YAML Signal 1",
                "polarity": "positive",
                "horizon_category": "fast",
                "half_life_days": 5.0,
                "theme": "test_price",
                "update_frequency": "1D",
                "economic_rationale": "YAML test signal 1.",
                "data_source": "test_src",
            }
        ]
    }
    file2 = {
        "signals": [
            {
                "name": "yaml_sig_2",
                "display_name": "YAML Signal 2",
                "polarity": "negative",
                "horizon_category": "cycle",
                "half_life_days": 90.0,
                "theme": "test_macro",
                "update_frequency": "1W",
                "economic_rationale": "YAML test signal 2.",
                "data_source": "test_src",
            }
        ]
    }
    (tmp_path / "price.yaml").write_text(yaml.dump(file1))
    (tmp_path / "macro.yaml").write_text(yaml.dump(file2))
    return tmp_path
