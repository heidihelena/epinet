# Data Format

EpiNet expects two CSV files: one for nodes and one for edges. The schema is
domain-neutral so the same toolkit can be used for epidemiology, clinical pathways,
research collaboration networks, referral systems, grant maps, patient journeys, or
other graph-shaped problems.

## Nodes

Required:

```csv
ID
```

Optional but commonly useful:

```csv
Outcome,Type,Label,Group,Time,Feature1,Feature2,...
```

Example:

```csv
ID,Outcome,Type,Age,Clinic
Patient_1,1,Patient,62,A
Patient_2,0,Patient,44,B
Clinic_A,,Clinic,,
```

Rules:

- `ID` must be unique.
- `Outcome` is optional. If present, it can be used for outcome modeling and for
  defining target nodes in shortest-path analysis.
- Numeric node attributes are added to the optional outcome model.
- Missing values should be meaningful. The toolkit does not assume that blank,
  unknown, not measured, and zero are the same thing.

## Edges

Required:

```csv
SourceID,TargetID
```

Optional:

```csv
Weight,Relationship,Time,Direction,Metadata...
```

Example:

```csv
SourceID,TargetID,Weight,Relationship
Patient_1,Clinic_A,1.0,visit
Patient_2,Clinic_A,1.0,visit
```

Rules:

- Every `SourceID` and `TargetID` must exist in the node file.
- By default, edges are treated as undirected.
- Pass `--directed` if `SourceID -> TargetID` direction matters.
- Pass `--weight-column Weight` to copy an edge column into the graph as `weight`.
- Pass `--path-mode distance` only if the weight column represents distance, cost,
  delay, or impedance.
- Pass `--path-mode strength` only if the weight column is a normalized 0..1
  relationship strength. Internally, the toolkit converts strength into a
  non-negative cost with `-log(strength)`.
- Do not call a route "fastest" unless an edge column truly encodes time or delay.

## Outputs

Typical outputs:

- `graph_summary.json`: node/edge counts, density, components, isolates.
- `node_features.csv`: graph-derived node features.
- `nearest_targets.csv`: nearest target node and shortest path per source.
- `shortest_paths.csv`: source-target path table.
- `model_metrics.json`: optional outcome-model metrics.
- `model_feature_importance.csv`: optional RandomForest feature importances.

## Minimum Example

```bash
python epinet_toolkit.py \
  --nodes synthetic_nodes.csv \
  --edges synthetic_edges.csv \
  --outcome-column Outcome \
  --target-outcome 1
```
