"""Core type definitions for composition (merged from COS-CIE)."""
from __future__ import annotations

from enum import Enum


class Polarity(str, Enum):
    """Signal polarity: whether higher values are bullish or bearish.

    POSITIVE: higher value = bullish (e.g., momentum, RSI recovery)
    NEGATIVE: higher value = bearish (e.g., volatility, fear index)
    """

    POSITIVE = "positive"
    NEGATIVE = "negative"


class HorizonCategory(str, Enum):
    """Signal horizon category grouping.

    FAST: intraday to ~5 days
    MEDIUM: ~5-20 days
    SLOW: ~20-60 days
    CYCLE: >60 days / structural
    """

    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    CYCLE = "cycle"
