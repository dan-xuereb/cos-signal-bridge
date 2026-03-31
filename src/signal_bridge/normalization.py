"""Normalization closure factory for SDL factor signals."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from sdl.models.config import NormalizationConfig
from sdl.types import NormMethod


def _apply_norm(s: pd.Series, config: NormalizationConfig | None) -> pd.Series:
    """Dispatch to the appropriate normalization method.

    Args:
        s:      Raw signal Series.
        config: NormalizationConfig with method, window, clip_lo, clip_hi.
                If None, returns s unchanged (passthrough).

    Returns:
        Normalized Series (clip and invert NOT applied here — those happen in the closure).
    """
    if config is None or config.method == NormMethod.passthrough:
        result = s
    elif config.method == NormMethod.tanh_zscore:
        n = config.window
        roll_mean = s.rolling(n).mean()
        roll_std = s.rolling(n).std()
        result = pd.Series(np.tanh((s - roll_mean) / roll_std), index=s.index)
    elif config.method == NormMethod.minmax:
        s_min, s_max = s.min(), s.max()
        denom = s_max - s_min
        if denom == 0:
            result = pd.Series(0.0, index=s.index)
        else:
            result = (s - s_min) / denom
    elif config.method == NormMethod.rank:
        result = s.rank(pct=True)
    else:
        raise ValueError(f"Unknown NormMethod: {config.method}")

    return result


def make_normalized_callable(
    fn: Callable[[pd.DataFrame], pd.Series],
    config: NormalizationConfig | None,
    combined_lookback: int,
    *,
    invert: bool = False,
) -> Callable[[pd.DataFrame], pd.Series]:
    """Wrap a raw evaluator function with normalization, clip, invert, and warmup NaN mask.

    Args:
        fn:                Raw evaluator callable: (pd.DataFrame) -> pd.Series.
        config:            NormalizationConfig (method, window, clip_lo, clip_hi).
                           If None, behaves as passthrough with warmup mask.
        combined_lookback: Number of warmup bars to force to NaN.
                           Formula: expr_lookback + norm_config.window.
        invert:            If True, negate the output after clip.
                           Comes from SglIntegration.invert (NOT NormalizationConfig).

    Returns:
        A new callable (pd.DataFrame) -> pd.Series that applies normalization,
        clip, invert, and forces the first combined_lookback elements to NaN.
    """

    def _apply(df: pd.DataFrame) -> pd.Series:
        raw = fn(df)
        normed = _apply_norm(raw, config)
        # Apply clip AFTER normalization (clip_lo/clip_hi are on NormalizationConfig)
        if config is not None:
            normed = normed.clip(lower=config.clip_lo, upper=config.clip_hi)
        # Apply invert AFTER clip (invert is from SglIntegration, passed separately)
        if invert:
            normed = -normed
        # Belt-and-suspenders: force first combined_lookback bars to NaN
        result = normed.copy()
        result.iloc[:combined_lookback] = np.nan
        return result

    return _apply
