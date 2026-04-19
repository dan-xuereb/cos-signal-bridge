# API

This page is generated from in-code docstrings by
[mkdocstrings](https://mkdocstrings.github.io/python/). Pydantic v2 models with
trailing-string field docstrings render natively via `griffe-pydantic`.

::: signal_bridge
    options:
      show_root_heading: true
      members_order: alphabetical

## Factor registry

File-based atomic-write `FactorRecord` persistence, keyed by UUID.

::: signal_bridge.registry

## Expression evaluator

Dispatch-table visitor that walks `ExpressionNode` DAGs bottom-up against a
pandas DataFrame.

::: signal_bridge.evaluator

## SDL → SGL adapter

Converts a `FactorRecord` into an `IndicatorSpec` suitable for
`SignalFrame` consumption. Requires the `[sgl]` extra.

::: signal_bridge.adapter

## SignalFrame provider

Extracts the VALID-bar `{ts_ns: float}` dict from a `SignalFrame` for
BTE strategy injection.

::: signal_bridge.provider

## Feedback + monitoring

Post-backtest IC / turnover computation, OOS status transitions, and atomic
registry writeback.

::: signal_bridge.feedback
