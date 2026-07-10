# EpiNet as Computational Epistemic Science

EpiNet began as a network-aware epidemiology workbench. Its deeper use is a
computational laboratory for asking when relational patterns justify scientific
claims.

The shared object across social-media analysis, network epidemiology, and
scientific evidence systems is not a neural network. It is a contested
relational system:

```text
world -> observation -> network construction -> inference -> claim
```

Every arrow can fail. EpiNet's job is to make those failures visible.

## What Social-Media Analysis Contributes

Classical tabular epidemiology often starts from independent rows:

```text
Y_i = f(X_i)
```

Network analysis starts from a relational data-generating system:

```text
Y_i = f(X_i, G, X_neighbors(i), t)
```

That changes the scientific questions:

- observations may be dependent because people share clinicians, institutions,
  exposures, information, and environments;
- exposure can propagate along paths rather than merely belong to individuals;
- homophily, shared context, influence, and measurement processes can all create
  similar outcomes among connected people;
- missingness can occur in nodes, edges, communities, and network boundaries;
- interventions can change topology, not only individual risk factors.

The lesson for EpiNet is simple: the observational unit may be an individual,
but the system under study can be relational.

## What Epidemiology Contributes Back

Social-media analytics often predicts diffusion or engagement without a clear
estimand. Epidemiology adds the discipline needed for defensible claims:

- target population;
- temporal ordering;
- confounding and selection mechanisms;
- measurement validity;
- negative controls;
- transportability;
- uncertainty;
- separation of prediction, explanation, and intervention.

This is why EpiNet emphasizes claim gates over model novelty. A high AUROC under
a random node split does not establish a network effect, external validity, or
clinical utility.

## Model Ladder

The first modeling question is not "Which architecture is most fashionable?" It
is "What is the smallest model that shows whether the data contain transportable
signal?"

EpiNet's intended ladder is:

| Model | Inputs | Question |
| --- | --- | --- |
| Null model | outcome prevalence | Is anything better than guessing? |
| Logistic regression | clinical variables | Is linear tabular signal sufficient? |
| Logistic regression | network features | Does graph position contain signal? |
| Logistic regression | clinical + network | Do network features add information? |
| Tiny MLP | clinical + network | Is there useful nonlinear structure? |
| Random forest / XGBoost | clinical + network | Do flexible tabular models outperform the small neural model? |
| Node2Vec + linear model | learned node embeddings | Does unsupervised graph representation help? |
| GraphSAGE | features + adjacency | Does message passing add transportable signal? |

GraphSAGE, or any other GNN, earns its place only if it improves over an equally
tuned non-graph model under community-separated and external validation without
collapsing calibration.

## Graph Semantics

The graph itself is a measurement model.

A similarity graph means the edge is derived from feature proximity. It supports
feature-space exploration, clustering, stability checks, and contestability. It
does not by itself support claims about contact, referral, exposure,
transmission, or message passing.

An observed-relation graph means the edge represents a real relation such as
shared clinician, transfer, household, contact, referral path, geographic
adjacency, or temporal co-exposure. It can support network-aware predictive
claims only when the relation was knowable before the outcome and can be
reconstructed in validation cohorts.

This is why the Workbench records:

```yaml
analysis:
  graph:
    semantics: observed_relation  # or similarity / unspecified
    edge_timing: pre_outcome       # or post_outcome / mixed / unknown
```

The claims check reads those fields and prevents a similarity graph from being
reported as evidence of relational influence.

## Epistemic Laboratory Roadmap

The next research direction is to replace people and infections with epistemic
objects and transformations.

Possible nodes:

- papers;
- claims;
- datasets;
- methods;
- researchers;
- institutions;
- guidelines;
- reviews;
- journals;
- evidence ratings.

Possible edges:

- cites;
- supports;
- contradicts;
- reuses data from;
- repeats a claim from;
- applies a method from;
- corrects;
- retracts;
- reviews;
- funds;
- shares authorship with;
- derives from.

The central question becomes:

```text
How does a claim acquire, retain, lose, or appear to possess credibility?
```

Citation frequency is not evidential support. Centrality is not reliability.
Consensus is not truth. EpiNet should therefore model evidence, attention,
trust, contradiction, replication, and correction as separate mechanisms.

## Minimal Simulator

A first simulator should be deliberately small:

```python
world = generate_world()
studies = observe_world(world)
researchers = connect_researchers(studies)
claims = publish_and_cite(studies, researchers)
beliefs = propagate_evidence(claims, researchers)
audit = compare_belief_to_world(beliefs, world)
```

The decisive experiment is:

```text
Under what network conditions does weak evidence become a stable scientific consensus?
```

The simulator would vary homophily, prestige bias, publication bias,
replication probability, correction probability, selective citation, evidence
quality, communication speed, conformity, bridge density, memory, and decay.

Outcomes should include:

- probability of reaching the correct conclusion;
- time to correct convergence;
- persistence of false consensus;
- diversity of live hypotheses;
- detectability of contradiction;
- dependence on one central study;
- proportion of claims with direct evidential support;
- resilience after retraction;
- discrepancy between citation centrality and evidential reliability.

This keeps the code path inspectable while moving EpiNet beyond classifier
selection into computational epistemic science.
