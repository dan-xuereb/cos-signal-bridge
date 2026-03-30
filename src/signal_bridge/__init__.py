"""cos-signal-bridge — SDL FactorRecord registry + ExpressionNode evaluator."""

__version__ = "0.1.0"

# Guard all imports behind try/except so `import signal_bridge` works with stdlib only (NF1).
try:
    from signal_bridge.registry import load_factor, save_factor, list_factors  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.evaluator import evaluate, compute_lookback  # noqa: F401
except ImportError:
    pass
