"""SignalLibrary: registry for governed signal metadata with YAML catalog loading."""
from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Optional

import yaml

from .models import SignalMeta
from .types import HorizonCategory, Polarity


class SignalLibrary:
    """In-memory registry of SignalMeta descriptors.

    Supports:
      - Programmatic registration via register()
      - Name-based retrieval via get()
      - Listing and filtering via list_all() / filter()
      - Bulk loading from per-theme YAML catalogs via from_directory()
      - Loading the packaged default catalogs via from_packaged_signals()

    Duplicate signal names are rejected unless the new version is strictly
    greater than the existing version (allows governed upgrades).
    """

    def __init__(self) -> None:
        self._signals: dict[str, SignalMeta] = {}

    # -- Registration ----------------------------------------------------------

    def register(self, meta: SignalMeta) -> None:
        """Register a signal. Raises ValueError on duplicate name unless version is bumped."""
        existing = self._signals.get(meta.name)
        if existing is not None:
            if meta.version <= existing.version:
                raise ValueError(
                    f"Signal '{meta.name}' already registered at version {existing.version}; "
                    f"provide version > {existing.version} to overwrite"
                )
        self._signals[meta.name] = meta

    # -- Retrieval -------------------------------------------------------------

    def get(self, name: str) -> SignalMeta:
        """Retrieve a signal by name. Raises KeyError if not found."""
        try:
            return self._signals[name]
        except KeyError:
            raise KeyError(f"Signal '{name}' not found in library") from None

    # -- Query -----------------------------------------------------------------

    def list_all(self) -> list[SignalMeta]:
        """Return all registered signals as a list."""
        return list(self._signals.values())

    def filter(
        self,
        theme: Optional[str] = None,
        horizon: Optional[HorizonCategory] = None,
        polarity: Optional[Polarity] = None,
    ) -> list[SignalMeta]:
        """Filter signals by theme, horizon_category, and/or polarity (AND logic).

        With no arguments, returns all signals (same as list_all).
        """
        results = self.list_all()
        if theme is not None:
            results = [s for s in results if s.theme == theme]
        if horizon is not None:
            results = [s for s in results if s.horizon_category == horizon]
        if polarity is not None:
            results = [s for s in results if s.polarity == polarity]
        return results

    # -- Protocols -------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._signals)

    def __contains__(self, name: str) -> bool:
        return name in self._signals

    # -- YAML loading ----------------------------------------------------------

    @classmethod
    def from_directory(cls, directory: Path) -> SignalLibrary:
        """Load all *.yaml files from a directory and register their signals.

        Each YAML file must have a top-level ``signals:`` key containing a list
        of dicts that validate against SignalMeta.

        Raises:
            FileNotFoundError: If directory does not exist.
            pydantic.ValidationError: If any signal entry fails validation.
        """
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")

        lib = cls()
        for yaml_path in sorted(directory.glob("*.yaml")):
            with open(yaml_path, "r") as fh:
                data = yaml.safe_load(fh)

            if data is None or "signals" not in data:
                continue

            for entry in data["signals"]:
                meta = SignalMeta.model_validate(entry)
                lib.register(meta)

        return lib

    @classmethod
    def from_packaged_signals(cls) -> SignalLibrary:
        """Load the default signal catalogs shipped inside this package."""
        pkg = resources.files("signal_bridge.composition").joinpath("signals")
        with resources.as_file(pkg) as path:
            return cls.from_directory(path)
