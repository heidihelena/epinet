# Running NLST through this toolkit

NLST is **not public**. It is released only through the NCI Cancer Data Access
System (https://cdas.cancer.gov) after an approved data request and signed data
transfer agreement. This toolkit cannot and does not download it.

## Getting the data (offline, by you)
1. Create an account at https://cdas.cancer.gov and submit an NLST data request
   describing the project (the risk-score comparison + fusion work here is a fitting
   description).
2. Sign the data transfer agreement; await approval (days–weeks).
3. Download the **participant** dataset and the **spiral-CT abnormality** dataset
   (CSV). Note the dataset version and its data dictionary.

## Running it
1. Open `build_nlst_cohort.py` and reconcile `COLUMN_MAP` with your data
   dictionary (variable names differ slightly by release). The loader raises an
   actionable `KeyError` naming any column it can't find.
2. Build the cohort:
   ```bash
   python examples/build_nlst_cohort.py \
     --participant <participant.csv> --abnormalities <sct_abnormalities.csv> \
     --output examples
   ```
3. The output (`nlst_nodes.csv`, `nlst_edges.csv`, `nlst_provenance.csv`) is in the
   same format the rest of the toolkit consumes. Because NLST carries demographics,
   smoking, family history, growth (T0/T1/T2) and confirmed cancer outcomes, you can
   now run the **full** comparison that LIDC could not support:
   - compute Brock / Mayo / PLCOm2012 / NTOG from `nlst_provenance.csv`,
   - `python -m epinet.toolkit ... --run-clusters` for centroid structure,
   - `test_fusion.py` for multi-test fusion against the real `LungCancer` label,
   with adequate benign representation for measurable specificity.

Verify the fixture/transform without real data: `python examples/build_nlst_cohort.py --demo`.
