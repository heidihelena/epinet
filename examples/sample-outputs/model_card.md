# Model card — EpiNet outcome model

## Intended use

**Research and education demonstrator. Not clinical or public-health decision support.** Any model produced here must be validated on independent, outcome-linked data before it carries clinical meaning. The figures below describe this run on this dataset, not external performance.

## Data

| property | value |
| --- | --- |
| Labeled nodes | 100 |
| Unlabeled scaffold (excluded) | 0 |
| Outcome classes | 2 |
| Class labels | 0, 1 |
| Train / test rows (primary split) | 80 / 20 |

## Model & evaluation

| property | value |
| --- | --- |
| Estimator | RandomForestClassifier |
| Tuned hyperparameters | {'max_depth': None, 'n_estimators': 100} |
| Split strategy | random |
| Evaluation iterations | 10 |
| Importance method | permutation |

## Performance

### Discrimination & classification

| metric | value |
| --- | --- |
| AUROC | 0.399 (95% CI 0.143–0.708) |
| Average precision (AUPRC) | 0.516 |
| Balanced accuracy | 0.586 (95% CI 0.374–0.813) |
| Matthews corr. coef. | 0.179 (95% CI -0.287–0.655) |
| F1 (weighted) | 0.592 (95% CI 0.360–0.807) |
| Accuracy | 0.600 (95% CI 0.400–0.800) |

### Calibration

| metric | value |
| --- | --- |
| Brier score (lower better) | 0.332 (95% CI 0.231–0.433) |
| Calibration slope (ideal 1) | -0.576 |
| Calibration intercept (ideal 0) | 0.383 |
| Positive class | 1 |

_Perfect calibration: slope 1, intercept 0. Slope < 1 = over-confident._

## Validation

Label-permutation null model (50 permutations):

| metric | observed | null mean | p-value |
| --- | --- | --- | --- |
| accuracy | 0.560 | 0.508 | 0.176 |
| balanced_accuracy | 0.540 | 0.497 | 0.255 |
| mcc | 0.090 | -0.007 | 0.255 |
| precision_weighted | 0.554 | 0.501 | 0.235 |
| recall_weighted | 0.560 | 0.508 | 0.176 |
| f1_weighted | 0.535 | 0.496 | 0.294 |
| roc_auc | 0.545 | 0.491 | 0.255 |
| average_precision | 0.634 | 0.607 | 0.294 |
| brier | 0.290 | 0.293 | 0.510 |

_9 metrics tested simultaneously; p-values are NOT corrected for multiple comparisons — read them jointly, not as independent significance tests._

Within-split uncertainty: 95% percentile bootstrap (1000 resamples of the held-out test set).

## Limitations

- Performance is in-sample to this dataset; external validity is unknown.
- Repeated-split spread understates true uncertainty (overlapping splits); the bootstrap CI is a within-split estimate, not external.
- A good discrimination score does not imply good calibration — check both.
- On small cohorts, all metrics are high-variance; see any data warnings above.

## Provenance

| property | value |
| --- | --- |
| EpiNet version | 0.2.0 |
| Git commit | 005126162864684d87e0c6171b0b3673b30a44be |
| Working tree | dirty |
| Python | 3.11.8 |
| scikit-learn | 1.9.0 |
| numpy / pandas | 2.4.6 / 3.0.3 |
| Random seed | 42 |
| Generated (UTC) | 2026-06-11T19:34:43+00:00 |

Input SHA-256:

- `synthetic_nodes.csv`: `a32585a87f30dab84528902c14d56a7ce7d91eb4a5ea751dadefa76c658fcdad`
- `synthetic_edges.csv`: `40ea0ed65092e4293d5648c18fbe0bc794234a4390862c12e5431615cd42545f`

