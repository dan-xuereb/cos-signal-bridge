"""SGL-to-BTE signal provider: extracts factor values from SignalFrame for strategy injection."""

from __future__ import annotations

try:
    from xuer_sgl.signal_frame import SignalFrame
    from xuer_sgl.types import BarAvailabilityState
except ImportError:
    SignalFrame = None  # noqa: N816
    BarAvailabilityState = None  # noqa: N816


def extract_signal_dict(sf: SignalFrame, col: str) -> dict[int, float]:  # noqa: F821
    """Extract VALID-bar values from a SignalFrame column as a {ts_ns: value} dict.

    This is the SGL-to-BTE bridge seam: strategies call ``signal_dict.get(bar.ts_event)``
    in ``on_bar()`` to read SDL factor values at runtime.

    Args:
        sf: A SignalFrame produced by the SGL pipeline.
        col: Column name to extract (e.g. "sdl_lag_close_1_v1").

    Returns:
        dict mapping nanosecond integer timestamps to float signal values,
        containing only bars where availability == VALID.

    Raises:
        RuntimeError: If xuer_sgl is not installed.
        KeyError: If ``col`` is not a column in the SignalFrame.
    """
    if SignalFrame is None:
        raise RuntimeError(
            "xuer_sgl is not installed — install with: pip install cos-signal-bridge[sgl]"
        )

    avail = sf.availability[col]  # raises KeyError if col not in frame
    valid_mask = avail == BarAvailabilityState.VALID.value
    valid_data = sf.data[col][valid_mask]
    ts_ns = valid_data.index.asi8
    return dict(zip(ts_ns.tolist(), valid_data.tolist(), strict=False))
