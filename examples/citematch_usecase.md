# CiteMatch Evidence Graph Use Case

This example treats claim-paper matching data as an evidence graph.

It is deliberately exploratory. The graph can help surface structure, gaps, bridge
papers, and paths between claims and evidence. It should not decide whether a claim
is true.

## Node Types

- `Claim`: a research claim that needs support/contrast evidence.
- `Paper`: a study, review, editorial, or other source.
- `Topic`: a disease area, endpoint, mechanism, or concept.
- `Method`: study design or evidence type.
- `Journal` and `Author`: provenance/context nodes.

## Edge Types

- `directly_supports`, `partially_supports`, `contradicts`: claim-paper evidence edges.
- `about`, `tagged_with`, `uses_method`, `published_in`, `authored_by`: context edges.
- `related_claim`: claim-claim bridge.

## Run: Find Nearest Contrast Evidence By Hops

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

This writes `nearest_targets.csv`, where each claim is linked to the closest paper
marked as contrast evidence by graph hops. This is a route query, not a truth
judgment.

## Run: Find Nearest Support Evidence

```bash
python epinet_toolkit.py \
  --nodes examples/citematch_nodes.csv \
  --edges examples/citematch_edges.csv \
  --outcome-column Outcome \
  --target-outcome support_evidence \
  --source-nodes claim_osimertinib_dfs,claim_osimertinib_os,claim_chemo_required \
  --no-run-model \
  --output-dir examples/citematch_outputs/support_paths
```

## Run: Use Evidence Distance

The example edge file includes both:

- `Weight`: evidence/relationship strength, where larger means stronger
- `Distance`: a path cost, where smaller means closer

Shortest-path algorithms minimize cost. Use `Distance`, not raw `Weight`, when
asking for a lowest-cost evidence route:

```bash
python epinet_toolkit.py \
  --nodes examples/citematch_nodes.csv \
  --edges examples/citematch_edges.csv \
  --outcome-column Outcome \
  --target-outcome contrast_evidence \
  --source-nodes claim_osimertinib_dfs,claim_osimertinib_os,claim_chemo_required \
  --weight-column Distance \
  --path-mode distance \
  --no-run-model \
  --output-dir examples/citematch_outputs/evidence_distance_contrast_paths
```

This is not a "fastest path" unless `Distance` is measured in time. Here it is an
evidence-distance route.

## Run: Use Relationship Strength

If `Weight` is a normalized 0..1 strength score, use `--path-mode strength`.
This maximizes the product of relationship strengths along the route:

```bash
python epinet_toolkit.py \
  --nodes examples/citematch_nodes.csv \
  --edges examples/citematch_edges.csv \
  --outcome-column Outcome \
  --target-outcome contrast_evidence \
  --source-nodes claim_osimertinib_dfs,claim_osimertinib_os,claim_chemo_required \
  --weight-column Weight \
  --path-mode strength \
  --no-run-model \
  --output-dir examples/citematch_outputs/strength_contrast_paths
```

Use this only when all edge weights are comparable. A claim-paper support score and
a paper-topic tag score may not mean the same thing.

## What To Look For

- Claims with direct support or contradiction.
- Claims whose nearest evidence is indirect via another claim/topic/paper.
- Papers that bridge multiple claims.
- Isolated claims or papers with no path to evidence.
- Context nodes that explain why two claims or papers are connected.

## Not Implemented Here

- Claim semantic embedding.
- Citation graph import.
- DOI/PubMed enrichment.
- Evidence quality appraisal.
- Causal or clinical correctness inference.
- Human review workflow.
