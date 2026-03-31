"""cos-signal-bridge — SDL FactorRecord registry + ExpressionNode evaluator."""

__version__ = "0.1.0"

# Guard all imports behind try/except so `import signal_bridge` works with stdlib only (NF1).
try:
    from signal_bridge.registry import list_factors, load_factor, save_factor  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.evaluator import compute_lookback, evaluate  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.normalization import make_normalized_callable  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.adapter import factor_to_indicator_spec  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.provider import extract_signal_dict  # noqa: F401
except ImportError:
    pass
