"""Schema inference and data validation for the EpiNet Workbench (Screens 1-2).

Two jobs:

* :func:`profile_table` — read a CSV and report what the Workbench's Data screen
  shows: row/column counts, duplicate IDs, empty/single-class outcomes, high
  cardinality, date-like / free-text / identifier-looking columns.
* :func:`infer_schema` — propose an ``id`` column, an ``outcome`` column, a task,
  feature columns, and a list of *suspected leakage* columns. The proposal is a
  suggestion the user confirms; nothing is silently chosen.

Everything here is descriptive. It never edits the data; it produces a report and
a suggested :class:`epinet_config.Schema` that the user reviews and overrides.
Overrides are the user's call — the runner logs them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from epinet.config import Schema

# Column-name fragments that commonly leak the outcome (post-hoc / future / label
# echoes). Matching is a *warning*, never an automatic exclusion — the user decides.
LEAKAGE_PATTERNS = (
    "outcome", "label", "target", "response", "result", "diagnosis", "final",
    "death", "died", "deceased", "survival", "survived", "followup", "follow_up",
    "post_treatment", "post-treatment", "posttreatment", "mortality", "relapse",
    "recurrence", "progression", "status",
)

# Name fragments that usually mark a row identifier rather than a feature.
ID_NAME_PATTERNS = ("id", "identifier", "accession", "mrn", "uuid", "guid", "key", "index")

# Name fragments that usually mark an outcome/target column.
OUTCOME_NAME_PATTERNS = ("outcome", "label", "target", "class", "response", "subtype", "y")

# Name fragments that usually mark a site / cluster column.
SITE_NAME_PATTERNS = ("site", "hospital", "center", "centre", "cluster", "clinic", "institution")

# Name fragments that usually mark a time column.
TIME_NAME_PATTERNS = ("date", "time", "year", "month", "day", "timestamp", "diagnosis_year")

_DATE_RE = re.compile(r"\d{4}[-/]\d{1,2}[-/]\d{1,2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}")


@dataclass
class ColumnProfile:
    name: str
    dtype: str
    n_unique: int
    n_missing: int
    cardinality_ratio: float          # n_unique / n_rows
    looks_like_id: bool = False
    looks_like_date: bool = False
    looks_like_free_text: bool = False
    looks_like_leakage: bool = False


@dataclass
class TableProfile:
    path: str
    n_rows: int
    n_cols: int
    columns: list[ColumnProfile]
    duplicate_id_count: int = 0
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def column(self, name: str) -> ColumnProfile | None:
        return next((c for c in self.columns if c.name == name), None)


def _looks_like_date(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    hits = sum(bool(_DATE_RE.search(v)) for v in sample)
    return hits >= max(1, int(0.6 * len(sample)))


def _looks_like_free_text(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(50)
    if sample.empty:
        return False
    # Free text: object dtype, mostly multi-word strings, high cardinality.
    multiword = sum(len(v.split()) >= 4 for v in sample)
    return multiword >= max(1, int(0.5 * len(sample)))


def _name_hits(name: str, patterns: tuple[str, ...]) -> bool:
    low = name.lower()
    return any(p in low for p in patterns)


def profile_table(path: str | Path, *, id_column: str | None = None) -> TableProfile:
    """Read a CSV and profile it for the Data screen's validation checks."""
    path = Path(path)
    try:
        df = pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001 - surface any read failure to the UI
        return TableProfile(str(path), 0, 0, [], errors=[f"file not readable: {exc}"])

    n_rows = len(df)
    profiles: list[ColumnProfile] = []
    for name in df.columns:
        col = df[name]
        n_unique = int(col.nunique(dropna=True))
        n_missing = int(col.isna().sum())
        ratio = (n_unique / n_rows) if n_rows else 0.0
        is_object = col.dtype == object
        is_numeric = pd.api.types.is_numeric_dtype(col)
        # An identifier is signalled by its *name* (with near-unique values) or by
        # being an all-unique non-numeric key. A continuous numeric column is
        # all-unique too, but that makes it a feature, not an ID — don't flag it.
        looks_like_id = (_name_hits(name, ID_NAME_PATTERNS) and ratio > 0.95) or (
            ratio == 1.0 and not is_numeric
        )
        profiles.append(
            ColumnProfile(
                name=name,
                dtype=str(col.dtype),
                n_unique=n_unique,
                n_missing=n_missing,
                cardinality_ratio=ratio,
                looks_like_id=looks_like_id,
                looks_like_date=_name_hits(name, TIME_NAME_PATTERNS) or _looks_like_date(col),
                looks_like_free_text=is_object and _looks_like_free_text(col),
                looks_like_leakage=_name_hits(name, LEAKAGE_PATTERNS),
            )
        )

    profile = TableProfile(str(path), n_rows, len(df.columns), profiles)

    if n_rows == 0:
        profile.errors.append("table has zero rows")
    if id_column and id_column in df.columns:
        dupes = int(df[id_column].astype(str).duplicated().sum())
        profile.duplicate_id_count = dupes
        if dupes:
            profile.errors.append(f"{dupes} duplicate values in id column '{id_column}'")
    for c in profiles:
        # High-cardinality *categorical* columns are the problematic ones (a
        # continuous numeric feature is all-unique by nature, which is fine).
        if (c.dtype == "object" and c.cardinality_ratio > 0.5 and c.n_unique > 20
                and not c.looks_like_id):
            profile.warnings.append(f"'{c.name}' has high cardinality ({c.n_unique} unique)")
        if c.looks_like_free_text:
            profile.warnings.append(f"'{c.name}' looks like free text")
        if c.looks_like_date:
            profile.warnings.append(f"'{c.name}' looks date-like (treat as time, not a feature)")
    return profile


def infer_schema(
    profile: TableProfile,
    *,
    mode: str = "single_csv",
) -> Schema:
    """Propose a :class:`Schema` from a table profile. A suggestion, not a decision.

    The proposal favours an explicit ``ID``/``Outcome`` column when present, then
    falls back to name heuristics. Suspected-leakage columns are recorded in
    ``leakage_flags`` but left *in* ``feature_columns`` unless the user excludes
    them — the Workbench surfaces the warning and logs any override.
    """
    names = [c.name for c in profile.columns]
    schema = Schema()

    # ID column: exact 'ID' wins, else the strongest identifier-looking column.
    id_col = _pick(names, exact=("ID", "id"), profiles=profile.columns,
                   prefer=lambda c: c.looks_like_id)
    if id_col:
        schema.id_column = id_col

    # Outcome: exact 'Outcome' wins, else a name-matched low-cardinality column.
    outcome_col = _pick(
        names, exact=("Outcome", "outcome", "target", "label"),
        profiles=profile.columns,
        prefer=lambda c: _name_hits(c.name, OUTCOME_NAME_PATTERNS) and c.n_unique <= 20,
    )
    schema.outcome_column = outcome_col
    if outcome_col:
        oc = profile.column(outcome_col)
        schema.task = "classification" if oc and oc.n_unique <= 20 else "descriptive"

    # Leakage detection takes priority over time/site assignment, so a column like
    # 'death_date' is surfaced as suspected leakage rather than silently treated as
    # an innocuous time column.
    schema.site_column = _pick(
        names, exact=(), profiles=profile.columns,
        prefer=lambda c: _name_hits(c.name, SITE_NAME_PATTERNS) and not c.looks_like_leakage,
    )
    schema.time_column = _pick(
        names, exact=(), profiles=profile.columns,
        prefer=lambda c: c.looks_like_date and not c.looks_like_leakage,
    )

    reserved = {schema.id_column, schema.outcome_column, schema.site_column,
                schema.time_column}
    if mode == "nodes_edges":
        reserved |= {schema.source_column, schema.target_column}

    # Features: numeric-ish columns that aren't reserved, date-like, free text, or
    # identifier-looking. Leakage-flagged columns are *suggested for exclusion*.
    feature_cols: list[str] = []
    exclude: list[str] = []
    leakage: list[str] = []
    for c in profile.columns:
        if c.name in reserved or c.name is None:
            continue
        if c.looks_like_id:
            exclude.append(c.name)
            continue
        # Surface suspected leakage explicitly, even when the column also looks
        # date-like (e.g. 'death_date', 'survival_days') — leakage is the louder signal.
        if c.looks_like_leakage:
            leakage.append(c.name)
            exclude.append(c.name)   # default-exclude suspected leakage; user can re-add
            continue
        if c.looks_like_date or c.looks_like_free_text:
            exclude.append(c.name)
            continue
        feature_cols.append(c.name)

    schema.feature_columns = feature_cols
    schema.exclude_columns = exclude
    schema.leakage_flags = leakage
    return schema


def _pick(names, *, exact, profiles, prefer):
    for e in exact:
        if e in names:
            return e
    for c in profiles:
        if prefer(c):
            return c.name
    return None
