"""SDL-to-SGL adapter: converts a FactorRecord into an IndicatorSpec."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
from sdl.models.config import SglIntegration  # noqa: F401 (used in type hints)
from sdl.models.factor import FactorRecord

from signal_bridge.evaluator import compute_lookback, evaluate
from signal_bridge.normalization import make_normalized_callable

try:
    from xuer_sgl.models import IndicatorSpec
    from xuer_sgl.types import BarAvailabilityState, WindowSemantics
except ImportError:
    IndicatorSpec = None  # type: ignore[assignment,misc]
    BarAvailabilityState = None  # type: ignore[assignment,misc]
    WindowSemantics = None  # type: ignore[assignment,misc]


def factor_to_indicator_spec(record: FactorRecord) -> "IndicatorSpec":
    """
    Convert a FactorRecord into an SGL IndicatorSpec.

    The returned IndicatorSpec.func is a Callable[[pd.DataFrame], pd.Series]
    that evaluates the expression tree, applies normalization, and enforces
    the warmup NaN mask for the first combined_lookback bars.

    Args:
        record: SDL FactorRecord with sgl_integration populated.

    Returns:
        IndicatorSpec ready for SignalFrame construction.

    Raises:
        ValueError: If record.sgl_integration is None or signal_name is None.
        RuntimeError: If xuer_sgl is not installed.
    """
    if IndicatorSpec is None:
        raise RuntimeError(
            "xuer_sgl is not installed — install with: pip install cos-signal-bridge[sgl]"
        )

    sgl = record.sgl_integration
    if sgl is None:
        raise ValueError("FactorRecord missing sgl_integration")
    if sgl.signal_name is None:
        raise ValueError("FactorRecord missing sgl_integration.signal_name")

    signal_name = sgl.signal_name
    signal_version = sgl.signal_version or "v1"
    col_name = f"sdl_{signal_name}_{signal_version}"

    expr_lookback = compute_lookback(record.expr_ir)
    norm_window = sgl.normalization.window if sgl.normalization else 0
    combined_lb = expr_lookback + norm_window

    # Capture expr_ir in closure scope
    expr_ir = record.expr_ir

    def _raw_fn(df: pd.DataFrame) -> pd.Series:
        return evaluate(expr_ir, df)

    # Wrap with normalization + clip + invert + warmup NaN mask
    func: Callable[[pd.DataFrame], pd.Series] = make_normalized_callable(
        _raw_fn,
        sgl.normalization,
        combined_lb,
        invert=sgl.invert,
    )

    return IndicatorSpec(
        name=col_name,
        source="custom",
        func=func,
        params={},
        outputs=[col_name],
        version=signal_version,
        description=record.description,
        lookback=combined_lb,
        window_semantics=WindowSemantics.BAR_COUNT,
        permitted_availability={BarAvailabilityState.VALID},
    )
