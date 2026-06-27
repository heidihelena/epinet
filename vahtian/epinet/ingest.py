"""Front-end input normalization: map a messy real-world CSV onto EpiNet's
canonical node/edge schema *before* strict validation runs.

A researcher should not have to hand-rename ``patient_id`` to ``ID`` or
``from``/``to`` to ``SourceID``/``TargetID`` just to run the tool. This module
resolves common column aliases (case-insensitively) to the configured column
names so ``validate_tables`` sees the schema it expects.

Design rule — normalization is never silent. Every rename is recorded in the
returned report (written to ``ingest_report.json`` and folded into
``provenance``), and provenance hashes BOTH the raw input file and the
normalized table, so a run still traces back to exactly what was analyzed.
Only *formatting* is automated here: analysis decisions — imputing feature
values, inventing edges, relabeling outcome classes — are deliberately left
explicit and out of scope.
"""

from __future__ import annotations

import pandas as pd

from vahtian.epinet import common as epinet_common

# Case-insensitive aliases per canonical role. Node roles are resolved only on
# the node table and edge roles only on the edge table, so an edge "target"
# endpoint can never be mistaken for an "Outcome" column.
NODE_ID_ALIASES = {
    "id", "node_id", "nodeid", "node", "patient_id", "patientid",
    "case_id", "caseid", "subject_id", "subjectid",
}
OUTCOME_ALIASES = {"outcome", "label", "class", "y", "status", "target_label"}
SOURCE_ALIASES = {"sourceid", "source", "source_id", "from", "src", "u", "start"}
TARGET_ALIASES = {"targetid", "target", "target_id", "to", "dst", "v", "end"}
WEIGHT_ALIASES = {"weight", "w", "cost", "strength"}


def _resolve(
    frame: pd.DataFrame,
    target: str,
    aliases: set[str],
    role: str,
    operations: list[dict[str, object]],
) -> None:
    """Rename a recognized alias column to ``target`` in place, logging the move.

    No-op when ``target`` already exists. Falls back through an exact
    case-insensitive match, then the alias set. Ambiguity (more than one alias
    present) is recorded and resolved deterministically by first match.
    """
    if target in frame.columns:
        return
    lower = {col.lower(): col for col in frame.columns}

    if target.lower() in lower:
        chosen = lower[target.lower()]
        frame.rename(columns={chosen: target}, inplace=True)
        operations.append(
            {"role": role, "action": "case_match", "from": chosen, "to": target}
        )
        return

    matches = [orig for low, orig in lower.items() if low in aliases]
    if not matches:
        return  # leave as-is; validate_tables will raise a clear error if required
    chosen = matches[0]
    action = "rename"
    if len(matches) > 1:
        action = "rename_ambiguous"
        operations.append(
            {"role": role, "action": "ambiguous", "candidates": matches, "chosen": chosen}
        )
    frame.rename(columns={chosen: target}, inplace=True)
    operations.append({"role": role, "action": action, "from": chosen, "to": target})


def normalize_tables(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    id_column: str,
    source_column: str,
    target_column: str,
    outcome_column: str | None = None,
    weight_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Resolve column aliases to the configured schema; return (nodes, edges, report).

    The inputs are not mutated (operates on copies). The report lists every
    rename and the SHA-256 of each normalized table, suitable for embedding in
    provenance and writing to ``ingest_report.json``.
    """
    nodes = nodes.copy()
    edges = edges.copy()
    operations: list[dict[str, object]] = []

    _resolve(nodes, id_column, NODE_ID_ALIASES, "node_id", operations)
    if outcome_column:
        _resolve(nodes, outcome_column, OUTCOME_ALIASES, "outcome", operations)
    _resolve(edges, source_column, SOURCE_ALIASES, "edge_source", operations)
    _resolve(edges, target_column, TARGET_ALIASES, "edge_target", operations)
    if weight_column:
        _resolve(edges, weight_column, WEIGHT_ALIASES, "edge_weight", operations)

    report: dict[str, object] = {
        "normalized": True,
        "n_operations": len(operations),
        "operations": operations,
        "normalized_nodes_sha256": epinet_common.sha256_frame(nodes),
        "normalized_edges_sha256": epinet_common.sha256_frame(edges),
    }
    return nodes, edges, report
