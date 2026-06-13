"""EpiNet (Epistemic Network): transparent network/feature-space analysis.

Modules are imported as ``from epinet import toolkit`` etc. The flat
``epinet_*.py`` layout was consolidated into this package in 0.4.
"""

try:
    from importlib.metadata import version as _version

    __version__ = _version("epinet")
except Exception:  # pragma: no cover - editable/uninstalled
    __version__ = "unknown"
