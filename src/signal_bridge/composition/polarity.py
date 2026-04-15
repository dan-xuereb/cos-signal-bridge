"""Polarity-driven sign-flip helper (merged from COS-CIE)."""
from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from .models import SignalMeta
from .types import Polarity


def apply_polarity(
    value: Union[float, int, np.ndarray, pd.Series],
    meta: SignalMeta,
) -> Union[float, int, np.ndarray, pd.Series]:
    """Apply polarity-driven sign flip to a numeric value.

    For POSITIVE polarity signals, the value is returned unchanged.
    For NEGATIVE polarity signals, the value is negated (multiplied by -1).

    This is the mechanism that Phase 19 (normalization) uses to make
    polarity handling transparent to downstream consumers. The helper
    does not perform normalization itself -- it only handles the sign flip.

    Args:
        value: Numeric value, numpy array, or pandas Series to flip.
        meta: SignalMeta whose polarity determines the flip.

    Returns:
        Original value if meta.polarity is POSITIVE, negated value if NEGATIVE.
    """
    if meta.polarity == Polarity.NEGATIVE:
        return -value  # type: ignore[operator]
    return value
