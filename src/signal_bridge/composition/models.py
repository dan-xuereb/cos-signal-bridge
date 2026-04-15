"""Core data models for composition (merged from COS-CIE)."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator

from .types import HorizonCategory, Polarity


class SignalMeta(BaseModel):
    """Frozen metadata descriptor for a composite indicator signal.

    All required fields must be provided at construction time.
    The model is immutable after construction (frozen=True).
    """

    model_config = ConfigDict(frozen=True)

    # Required fields (per D-03)
    name: str
    display_name: str
    polarity: Polarity
    horizon_category: HorizonCategory
    half_life_days: float
    theme: str
    update_frequency: str
    economic_rationale: str
    data_source: str

    # Optional fields with defaults
    version: int = 1
    lookback_bars: int = 0

    @field_validator("polarity", mode="before")
    @classmethod
    def _coerce_polarity(cls, v: Any) -> Any:
        """Coerce integer and string polarity representations.

        Accepts:
          - int 1 or str "+1" -> "positive"
          - int -1 or str "-1" -> "negative"
          - str "positive" / "negative" -> pass through (Pydantic handles enum coercion)
        """
        if isinstance(v, int) and not isinstance(v, bool):
            if v == 1:
                return Polarity.POSITIVE
            elif v == -1:
                return Polarity.NEGATIVE
        if isinstance(v, str):
            if v == "+1":
                return Polarity.POSITIVE
            elif v == "-1":
                return Polarity.NEGATIVE
        return v

    @field_validator("half_life_days")
    @classmethod
    def _validate_half_life(cls, v: float) -> float:
        """Reject half_life_days <= 0."""
        if v <= 0:
            raise ValueError("half_life_days must be positive (got {})".format(v))
        return v
