# EpiNet

EpiNet is being shaped as a **general node/edge network analysis toolkit**.
Epidemiology is one possible use case, but the core logic is intentionally broader:
load entities and relationships, compute graph features, optionally train a simple
outcome model, and run shortest-path analysis in parallel.

This repository is still a prototype. It should be treated as a demonstrator, not
clinical or public-health decision support.

## What Is Implemented

- CSV node and edge loading.
- Network construction with NetworkX.
- Node-level graph features:
  - degree
  - weighted degree
  - clustering
  - component size
  - isolate flag
  - optional betweenness, closeness, and PageRank
- Optional RandomForest outcome model using graph features plus numeric node attributes.
- Shortest-path analysis from source nodes to explicit target nodes or to nodes with a target outcome.
- CSV/JSON outputs for downstream inspection.

The older `epinet-analysis.py` and `epinet-analysis-v2.py` scripts remain as early prototypes.
The recommended entry point is now `epinet_toolkit.py`.

## Quick Start

```bash
pip install -r requirements.txt
python epinet_toolkit.py \
  --nodes synthetic_nodes.csv \
  --edges synthetic_edges.csv \
  --outcome-column Outcome \
  --target-outcome 1 \
  --output-dir epinet_outputs
```

This runs the two main lenses side by side:

1. graph feature generation and a simple outcome model
2. shortest-path summaries from non-target nodes to target outcome nodes

Generated files include:

- `graph_summary.json`
- `node_features.csv`
- `shortest_paths.csv`
- `nearest_targets.csv`
- `model_metrics.json`
- `model_feature_importance.csv`
- `run_summary.json`

## Shortest-Path Examples

Use outcome-positive nodes as targets:

```bash
python epinet_toolkit.py --outcome-column Outcome --target-outcome 1
```

Use explicit target nodes:

```bash
python epinet_toolkit.py --target-nodes Node_1,Node_5 --no-run-model
```

Limit the sources:

```bash
python epinet_toolkit.py --source-nodes Node_0,Node_9 --target-nodes Node_42 --no-run-model
```

Treat edges as directed:

```bash
python epinet_toolkit.py --directed --target-nodes Node_42 --no-run-model
```

Use an edge weight column as path distance:

```bash
python epinet_toolkit.py --weight-column Weight --path-mode distance --target-nodes Node_42 --no-run-model
```

Be careful: many datasets store edge weight as relationship strength, not distance.
If a larger weight means a stronger or more frequent connection, it should not be used
directly as a shortest-path distance without transformation.

If an edge column is a normalized 0..1 strength, you can ask for the strongest route:

```bash
python epinet_toolkit.py --weight-column Weight --path-mode strength --target-nodes Node_42 --no-run-model
```

## CiteMatch Evidence Graph Example

The `examples/` directory includes a small CiteMatch-style evidence graph:

- `examples/citematch_nodes.csv`
- `examples/citematch_edges.csv`
- `examples/citematch_usecase.md`

Run nearest contrast-evidence paths for three claims:

```bash
python epinet_toolkit.py \
  --nodes examples/citematch_nodes.csv \
  --edges examples/citematch_edges.csv \
  --outcome-column Outcome \
  --target-outcome contrast_evidence \
  --source-nodes claim_osimertinib_dfs,claim_osimertinib_os,claim_chemo_required \
  --no-run-model \
  --output-dir examples/citematch_outputs/contrast_paths
```

This is the safer non-epidemiology use case: the toolkit maps evidence structure
around claims and papers. It does not infer whether a claim is true.

For CiteMatch, avoid calling this a fastest path unless an edge column truly encodes
time or delay. The useful question is usually the best evidence route: nearest by
hops, lowest evidence distance, or strongest relationship path.

## Data Model

Nodes are entities: people, places, organizations, studies, grants, hospitals,
events, risks, exposures, pathway steps, or any other objects.

Edges are relationships or flows between nodes: contact, referral, collaboration,
co-authorship, transition, shared exposure, communication, dependency, or movement.

Default node columns:

- `ID`: unique node identifier
- `Outcome`: optional target label for modeling/path targeting

Default edge columns:

- `SourceID`
- `TargetID`
- optional `Weight`

See `Data-format.md` for details.

## Methodological Boundaries

The model is intentionally simple. It does not infer causality, outbreak dynamics,
clinical risk, or intervention effects. Network features can be useful descriptors,
but they can also encode sampling bias, measurement bias, and structural confounding.

Use the outputs as exploratory evidence, not as decisions.

Before using this for health, education, welfare, employment, or public-sector
decisions, add:

- domain-specific data validation
- directed/temporal assumptions
- uncertainty and sensitivity checks
- external validation
- privacy and governance review
- human review of any operational recommendations

## Tests

```bash
python -m unittest discover -s tests
```

## License

MIT. See `LICENSE`.
