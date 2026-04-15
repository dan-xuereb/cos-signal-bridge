"""Composition subpackage — signal metadata, library, and IC-weighted composition.

Merged from COS-CIE (which had zero runtime cross-repo imports beyond this bridge).
Retained as a subpackage to preserve the logical boundary between metadata/composition
and the rest of the bridge.
"""
from __future__ import annotations

from .composite import CompositeScore, SignalContribution, compose
from .library import SignalLibrary
from .models import SignalMeta
from .orchestrator import compose_signals
from .polarity import apply_polarity
from .types import HorizonCategory, Polarity

__all__ = [
    "Polarity",
    "HorizonCategory",
    "SignalMeta",
    "SignalLibrary",
    "apply_polarity",
    "CompositeScore",
    "SignalContribution",
    "compose",
    "compose_signals",
]
