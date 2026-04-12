"""
IC (Information Coefficient) utilities for cos-signal-bridge.

Pure pandas functions — no SGL, SDL, or bridge-internal imports.
Importable by both the bridge feedback module and CIE composition engine.
"""

from __future__ import annotations

import pandas as pd


def compute_ic(
    signal: pd.Series,
    returns: pd.Series,
    valid_mask: pd.Series,
) -> float:
    """
    Compute realized IC (Spearman rank correlation) of signal vs forward returns.

    Filters to valid_mask rows, drops NaNs, then computes Spearman correlation.
    Returns 0.0 if fewer than 2 valid observations remain after filtering, or if
    the correlation is NaN (e.g., constant series).

    Args:
        signal: Signal series, already shifted/lagged by the caller to prevent
                look-ahead bias. Aligns by index with ``returns``.
        returns: Forward return series (e.g., pct_change on close). Aligns by
                 index with ``signal``.
        valid_mask: Boolean pd.Series selecting the rows to include. Rows where
                    the mask is False are excluded before correlation.

    Returns:
        Spearman IC as float in [-1.0, 1.0]. Returns 0.0 on insufficient data.
    """
    combined = pd.DataFrame({"signal": signal, "returns": returns})
    combined_valid = combined[valid_mask].dropna()
    if len(combined_valid) < 2:
        return 0.0
    ic = float(combined_valid["signal"].corr(combined_valid["returns"], method="spearman"))
    return ic if not pd.isna(ic) else 0.0
