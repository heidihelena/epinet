"""Small shared helpers used across the EpiNet modules.

Kept dependency-light (pandas + stdlib only) so any module can import it without
pulling in scikit-learn or matplotlib.
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from platform import python_version

import pandas as pd

# Values that count as "no label" once stringified and lowercased.
_BLANK_TOKENS = {"", "nan", "none"}


def is_blank_value(value: object) -> bool:
    """True if a single label value is missing/blank (unlabeled scaffold)."""
    if pd.isna(value):
        return True
    return str(value).strip().lower() in _BLANK_TOKENS


def blank_label_mask(values: pd.Series) -> pd.Series:
    """Boolean Series, True where the label is blank/NaN.

    Robust to the modern pandas ``str``/``string`` dtypes (where ``astype(str)``
    leaves NaN as a float), which is the bug this helper exists to prevent from
    being re-introduced in each module.
    """
    mask = values.isna()
    if not pd.api.types.is_numeric_dtype(values):
        text = values.astype("string").fillna("").str.strip().str.lower()
        mask = mask | text.isin(list(_BLANK_TOKENS))
    return mask


def labeled_mask(values: pd.Series) -> pd.Series:
    """Boolean Series, True where the label is present (complement of blank)."""
    return ~blank_label_mask(values)


# --- Reproducibility / provenance -----------------------------------------
# A run is only reproducible if you can recover the exact inputs, code, and
# environment that produced it. These helpers stamp that context into the
# output so any figure or metric can be traced back to a commit, a package
# set, and a content hash of the data — the minimum a reviewer or registry
# data-controller needs to trust (and rerun) a result.

# Packages whose versions materially change numeric output. Kept explicit (not
# "everything installed") so the provenance block stays small and meaningful.
_TRACKED_PACKAGES = ("numpy", "pandas", "scikit-learn", "networkx", "scipy", "matplotlib")


def _package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def _git_state(repo_dir: Path) -> dict[str, object]:
    """Best-effort git commit + dirty flag for the module's repository.

    Returns ``{"commit": None, "available": False}`` outside a git checkout
    (e.g. an installed wheel) rather than raising — provenance must never be the
    thing that crashes a run.
    """
    def _git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return out.stdout.strip()
        except (subprocess.SubprocessError, OSError):
            return None

    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"available": False, "commit": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {"available": True, "commit": commit, "dirty": bool(status)}


def sha256_file(path: str | Path) -> str:
    """SHA-256 of a file's bytes, read in chunks so large CSVs stay cheap."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def provenance(
    input_paths: list[str | Path] | None = None,
    *,
    seed: int | None = None,
) -> dict[str, object]:
    """Capture the context needed to reproduce a run.

    Records the EpiNet version, git commit and dirty flag, Python and key
    package versions, the random seed, a UTC timestamp, and a SHA-256 of every
    input file. Stamped into ``run_summary.json``, ``model_metrics.json``, and a
    standalone ``provenance.json`` so any output traces back to exact inputs.
    """
    try:
        epinet_version = importlib_metadata.version("epinet")
    except importlib_metadata.PackageNotFoundError:
        epinet_version = "unknown (editable/uninstalled)"

    input_hashes: dict[str, str] = {}
    for raw in input_paths or []:
        path = Path(raw)
        try:
            input_hashes[str(path)] = sha256_file(path)
        except OSError:
            input_hashes[str(path)] = "unreadable"

    return {
        "epinet_version": epinet_version,
        "git": _git_state(Path(__file__).resolve().parent),
        "python_version": python_version(),
        "packages": _package_versions(),
        "random_seed": seed,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input_sha256": input_hashes,
    }
