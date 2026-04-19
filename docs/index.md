# cos-signal-bridge

## Purpose

**cos-signal-bridge** connects three otherwise-independent layers of the Xuer
Capital quant stack into a single factor-lifecycle pipeline:

1. **COS-SDL** — declarative factor definitions (`FactorRecord`,
   `ExpressionNode` DAGs).
2. **COS-SGL** — `SignalFrame` signal computation.
3. **COS-BTE** — NautilusTrader-based backtesting.

The bridge converts SDL expression trees into tradeable signals, surfaces those
signals to BTE strategies, and closes the monitoring loop by computing
realised IC / rank IC / turnover from backtest output and writing the result
back to the SDL file registry atomically. The registry file is updated
in-place so each iteration leaves a consistent on-disk state.

The v3.1 public surface (stable, typed) also includes a researcher-facing
`evaluate()` entry point with a pre-flight static cost gate
(`ExpressionCostExceeded`), an `OperatorSignatureRegistry` covering every
operator, and a walk-forward harness (`run_walk_forward`) safe for direct use
inside Marimo notebooks.

> **Footgun:** the distribution name is `cos-signal-bridge` but the Python
> package is `signal_bridge`. All imports use `from signal_bridge import ...`.

## Language & Runtime

- **Python:** `>=3.11`
- **Distribution name:** `cos-signal-bridge`
- **Importable module:** `signal_bridge`
- **Core dependencies:** `cos-sdl>=0.1.0`, `pandas>=2.0`, `numpy>=1.26`,
  `pydantic>=2.5`, `pyyaml>=6.0`
- **Optional extras:** `[sgl]` (`xuer-sgl>=0.4.0`), `[bte]` (`cos-bte>=0.1.0`)
- **Build backend:** `hatchling`

## Entry Points

cos-signal-bridge is a **library**, not a service:

```python
from signal_bridge import (
    evaluate,
    compute_lookback,
    ExpressionCostExceeded,
    OperatorSignature,
    OperatorSignatureRegistry,
    run_walk_forward,
    WalkForwardResult,
    FoldResult,
    generate_factor_folds,
    EphemeralFactorMetadata,
)
from signal_bridge.registry import load_factor, save_factor
from signal_bridge.adapter import factor_to_indicator_spec
from signal_bridge.provider import extract_signal_dict
from signal_bridge.feedback import compute_monitoring_update
```

No CLI, no FastAPI app, no `__main__.py`.

## Key Commands

```bash
# Install (SDL schemas only)
pip install cos-signal-bridge

# With SGL (SignalFrame computation):
pip install cos-signal-bridge[sgl]

# With BTE (NautilusTrader backtesting):
pip install cos-signal-bridge[bte]

# Full development install:
pip install -e ".[dev,sgl,bte]"
# (sibling workspace deps — COS-SDL, xuer-sgl — may need `--no-deps` then
# editable installs of the sibling repos directly)

# Tests
pytest tests/ -v

# Rebuild docs locally
.venv-docs/bin/mkdocs serve
```

## See Also

- [Architecture](architecture.md) — module map + SDL → SGL → BTE → SDL cycle
- [API](api.md) — auto-generated module reference
