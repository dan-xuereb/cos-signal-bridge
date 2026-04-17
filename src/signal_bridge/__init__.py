"""cos-signal-bridge — SDL FactorRecord registry + ExpressionNode evaluator.

v3.1 Public API (BRIDGE-01/02/03 — hard imports, no try/except masking):
    evaluate, compute_lookback, ExpressionCostExceeded
    run_walk_forward, WalkForwardResult, FoldResult, generate_factor_folds
    OperatorSignature, OperatorSignatureRegistry
    _build_ephemeral_factor_record, EphemeralFactorMetadata

Optional modules (import guarded on missing extras — composition / xuer_sgl):
    adapter, provider, feedback, composition
"""

__version__ = "0.1.0"

# -------------------------------------------------------------------
# BRIDGE-01 / 02 / 03 public surface — HARD imports (no try/except).
# A missing symbol here is a hard failure by design (v3.1 scope lock).
# -------------------------------------------------------------------
from signal_bridge.ephemeral import (  # noqa: F401
    EphemeralFactorMetadata,
    _build_ephemeral_factor_record,
)
from signal_bridge.evaluator import (  # noqa: F401
    ExpressionCostExceeded,
    compute_lookback,
    evaluate,
)
from signal_bridge.normalization import make_normalized_callable  # noqa: F401
from signal_bridge.operator_signatures import (  # noqa: F401
    OPERATOR_SIGNATURES,
    OperatorSignature,
    OperatorSignatureRegistry,
)
from signal_bridge.registry import (  # noqa: F401
    list_factors,
    load_factor,
    save_factor,
)
from signal_bridge.walkforward import (  # noqa: F401
    FoldResult,
    WalkForwardResult,
    generate_factor_folds,
    run_walk_forward,
)

# -------------------------------------------------------------------
# Optional modules — guarded imports preserved (depend on xuer_sgl
# or composition extras). Per RESEARCH.md Finding #4.
# -------------------------------------------------------------------
try:
    from signal_bridge.adapter import factor_to_indicator_spec  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.provider import extract_signal_dict  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.feedback import compute_monitoring_update  # noqa: F401
except ImportError:
    pass

try:
    from signal_bridge.composition import (  # noqa: F401
        CompositeScore,
        HorizonCategory,
        Polarity,
        SignalContribution,
        SignalLibrary,
        SignalMeta,
        apply_polarity,
        compose,
        compose_signals,
    )
except ImportError:
    pass
