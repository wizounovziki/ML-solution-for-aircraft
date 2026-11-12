# Aircraft MRO Unscheduled-Removal Prediction

## Executive summary

This repository contains a proof of concept (PoC) for predicting whether an in-service aircraft component will experience an **unscheduled removal within the next 30 flight cycles**.

The business acceptance criteria are:

- **Event-level recall ≥ 80%**: an actual unscheduled removal is detected if at least one final alert is emitted during its valid prediction window.
- **No more than 5 alerts per 100 evaluated components**, applied both overall and as a hard capacity limit in every scoring batch.

The frozen LightGBM model achieved:

- **100% event-level recall: 20/20 unscheduled removals detected**.
- **0.543 alerts per 100 components**: 32 alerts across 5,895 component scoring records.
- **0 of 123 batches above capacity**.
- **24.5 cycles of median first-alert lead time**.

These results come from synthetic data with deliberately embedded degradation signatures. They demonstrate the pipeline and evaluation design, not expected performance on a real fleet.

## Repository contents

- Reproducible synthetic dataset and generator.
- Causal feature engineering with event-time and availability-time controls.
- Aircraft/component-connected-group splitting with global temporal boundaries.
- Regularized logistic regression and LightGBM candidates, with and without imbalance weighting.
- Sigmoid probability calibration and validation-driven model selection.
- Event-level evaluation, per-batch alert capacity, duplicate-alert suppression, and immutable frozen-test artifacts.
- Global SHAP visualization, two high-risk component cards, and a false-positive example.
- Cold-start fallback, missing-data handling, and Population Stability Index (PSI) monitoring utilities.
- An executed evaluation Notebook and 15 automated tests.

## Dataset

Mock Dataset version `0.2.0` is included directly in this Git repository:

- [`data/mro_dataset.csv`](data/mro_dataset.csv) — combined snapshot-level dataset, approximately 48 MB.
- [Direct public download](https://github.com/wizounovziki/ML-solution-for-aircraft/raw/refs/heads/main/data/mro_dataset.csv)
- [Dataset dictionary](DATASET.md)
- [Generation configuration](configs/mock_data.yaml)

Current dataset profile:

| Item | Count |
| --- | ---: |
| Aircraft | 60 |
| Component/installation records | 723 |
| Component snapshots | 114,989 |
| Fault events | 3,683 |
| Maintenance events | 2,133 |
| Removal events | 484 |
| Unscheduled removals | 404 |
| Mature labels | 113,090 |
| Positive snapshots | 2,424 (**2.143%**) |
| Censored snapshots | 1,899 |

The normalized event tables and combined dataset can be regenerated in CSV and Parquet formats. A separate Google Drive or OneDrive copy is not required because the complete combined dataset is distributed with the repository.

### Target and observation rules

For a snapshot at time `t`:

```text
y = 1 if the same installation has an unscheduled removal in (t, t + 30 cycles]
```

- Positive labels reference a unique `removal_event_id`.
- Records with insufficient follow-up, or a scheduled-removal competing event inside the window, are censored rather than forced into the negative class.
- `label_available_at` records when a label can legally be used for fitting.
- `event_time`/`reading_time` records when an observation occurred; `available_at` records when it became visible to the system.
- Historical features require both actual occurrence and system availability by the snapshot time.
- Component snapshots after removal are excluded from the in-service population.

### Synthetic degradation and quality scenarios

The generator includes component-specific lifetimes, harsher operating environments, acute degradation during the final 20–35 cycles, rising vibration, decreasing EGT margin, increasing fault frequency, maintenance history, approximately 5% random sensor missingness, structured missingness on legacy aircraft, sensor spikes, late-arriving records, new component models, and sensor drift.

The data are entirely synthetic and contain no real airline, aircraft, component, or personnel records.

## Method

### 1. Leakage-aware evaluation design

The main evaluation combines entity isolation and global temporal boundaries:

1. Build connected groups from aircraft-component relationships. If a component moves between aircraft, those aircraft remain in the same connected group.
2. Assign complete connected groups to outer train, validation, and test partitions. Aircraft and component identifiers are disjoint across these outer partitions.
3. Apply global time boundaries:
   - Training: before `2024-09-01`, with labels available before that cutoff.
   - Calibration: `2024-09-01` to `2025-03-01`, with labels available before the calibration cutoff.
   - Alert-policy selection: `2025-03-01` to `2025-09-01`.
   - Frozen test: snapshots on or after `2025-09-01`.
4. Keep each removal event within one outer partition.
5. Include censored and label-immature components in scoring and alert burden, while restricting supervised metrics to mature labels.

The frozen split contains 36,252 training records, 2,002 calibration records, 2,394 policy-selection scoring records, and 5,895 test scoring records.

### 2. Feature engineering

`src/features.py` builds 5/15/30-cycle rolling means, standard deviations, and slopes by component. It also includes usage, installation, recent fault, maintenance, data-quality, missingness, and history-coverage features.

All rolling calculations use only observations that had occurred and were available at the prediction time. Preprocessing statistics, category maps, calibration data, and cold-start baselines are fitted only on their assigned training or calibration partitions.

### 3. Models and calibration

Four candidates were compared under identical outer splits and business rules:

- Logistic regression without class weighting.
- Logistic regression with balanced class weights, using spline, one-hot, imputation, and scaling pipelines.
- LightGBM without imbalance weighting.
- LightGBM with positive-class weighting.

All candidates use sigmoid calibration on natural-prevalence calibration data. `FrozenEstimator` with `CalibratedClassifierCV` prevents unintended stratified cross-validation on the calibration set.

### 4. Business alert policy

For each scoring batch with `N` evaluated components:

```text
K = floor(0.05 × N)
```

At most `K` final alerts are emitted. Consequently, batches with fewer than 20 components have zero capacity under this strict rule.

The frozen policy also applies 30-component-cycle duplicate suppression. Only actually emitted alerts update suppression state. Pre-suppression candidates, post-suppression candidates, capacity deferrals, and final alerts are all retained for audit.

## Validation results

The policy-selection interval contained 2,394 scoring records, 2,389 mature labels, and 16 independent unscheduled-removal events.

| Candidate | Event recall | Alerts per 100 | Known false positives per 100 | PR-AUC | Brier score | Maximum batch alert rate | Over-capacity batches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic regression, unweighted | 93.75% | 0.668 | 0.000 | 0.98896 | 0.005375 | 4.17% | 0 |
| Logistic regression, balanced | 100% | 0.835 | 0.125 | 0.98748 | 0.005722 | 4.17% | 0 |
| LightGBM, unweighted | 100% | 0.710 | 0.000 | 0.98103 | 0.003985 | 4.17% | 0 |
| LightGBM, balanced | 100% | 0.710 | 0.000 | 0.98003 | 0.004295 | 4.17% | 0 |

### Model-selection decision

Candidates were ranked by event recall, lower alert rate, fewer false positives, better Brier score, and lower computational cost. **Unweighted LightGBM** was selected because it detected all 16 validation events, produced no known false positives, and had the best calibrated Brier score among the candidates.

Frozen configuration:

- Risk threshold: `0.7250242783616677`
- Calibration: `sigmoid`
- Duplicate suppression: `30` component cycles
- Maximum alert rate: `0.05`
- Raw explanation space: LightGBM log-odds

The model, category maps, calibrator, cold-start baselines, feature list, threshold, and policy were frozen before final testing. A separate script loaded these artifacts for the test run and created a marker preventing silent repeated evaluation.

## Frozen-test results

The test population contained 5,895 scoring records, 5,565 mature labels, 123 batches, and 20 independent unscheduled-removal events.

| Metric | Result | Interpretation |
| --- | ---: | --- |
| **Event-level recall** | **100% (20/20)** | Passes the ≥80% criterion |
| **Alerts per 100 evaluated components** | **0.543 (32 alerts)** | Passes the ≤5 criterion |
| Maximum batch alert rate | 4.167% | Every batch remained within capacity |
| Over-capacity batches | **0/123** | Hard-capacity compliance |
| Known false positives per 100 components | 0.034 (2 alerts) | Known mature negative outcomes |
| Alerts with pending outcomes | 10 | Included in alert burden, excluded from known-outcome precision |
| Known-outcome alert precision | 90.91% (20/22) | Excludes pending outcomes |
| Snapshot-level recall | 17.24% | Reduced by duplicate suppression; not the primary event metric |
| PR-AUC | 0.98547 | Mature-label ranking metric |
| ROC-AUC | 0.99963 | Secondary metric, not used alone for selection |
| Brier score | 0.002112 | Includes calibrated and cold-start-blended scores |
| Mean first-alert lead time | 22.8 cycles | Across 20 detected events |
| Median first-alert lead time | 24.5 cycles | Range: 2–29 cycles |
| Capacity deferrals | 0 | No candidate was deferred by capacity |
| Duplicate-suppressed alerts | 122 | Only emitted alerts updated suppression state |

Connected-group cluster bootstrap with 200 resamples produced:

- Event-recall 95% interval: `[1.000, 1.000]`.
- Overall alert-rate 95% interval: `[0.004241, 0.006285]`, equivalent to approximately `[0.424, 0.629]` alerts per 100 components.

All four component families had 100% event recall, with six engine, five avionics, five brake-actuator, and four hydraulic-pump events. Some part-number slices contain no positive events; a displayed zero recall in those slices is not a missed-event claim.

## Explainability

SHAP explanations target the LightGBM raw log-odds output. Calibrated and cold-start-blended probabilities are displayed separately; raw SHAP contributions are not probability increments.

The locked SHAP/LightGBM combination uses `tree_path_dependent` explanations because native categorical splits are incompatible with interventional background data in the tested version. Additivity was checked with a maximum error below `6.1e-14`.

![Global SHAP summary](reports/figures/shap_summary.png)

The leading global factors include 30-cycle vibration mean, 15-cycle vibration mean, current vibration, vibration variability, and temperature trends. Full rankings are available in [`reports/shap_feature_importance.csv`](reports/shap_feature_importance.csv).

Two actual high-risk alerts are documented:

| Component | Family / part number | Risk score | Top three grouped factors | Observed outcome |
| --- | --- | ---: | --- | --- |
| C0000481 | Hydraulic pump / HYD-B2 | 0.996339 | Vibration, component temperature, component pressure | Unscheduled removal REM0000325 |
| C0000478 | Engine / ENG-A1 | 0.995741 | Vibration, component temperature, EGT margin | Unscheduled removal REM0000323 |

Each card includes actual values, contribution direction, grouped raw-score contributions, member features, and suggested inspection actions. Related rolling features are grouped to prevent multiple windows of the same signal from occupying every reason slot. Non-engine cards exclude non-applicable EGT features.

- [Full high-risk explanation cards](reports/high_risk_cards.md)
- [False-positive explanation](reports/false_positive_card.json)
- [Explanation-enhanced predictions](reports/test_predictions_with_explanations.parquet)

All 32 final alerts have reason codes. Explanation generation does not alter scores, alerts, or capacity decisions. The frozen test had no false-negative event, so no synthetic false-negative example was fabricated.

## Production design

```mermaid
flowchart LR
    S[Post-flight sensor, fault, and maintenance data] --> A[Event-time and availability-time checks]
    A --> F[Point-in-time feature computation]
    F --> M[Versioned model and sigmoid calibration]
    M --> C[Cold-start and missing-data fallback]
    C --> P[Duplicate suppression and per-batch 5% hard capacity]
    P --> E[SHAP reason codes]
    E --> W[AMOS/Maximo maintenance work queue]
    W --> H[Authorized engineering inspection and decision]
    H --> L[30-cycle delayed-label feedback]
    L --> D[Outcome evaluation and PSI drift monitoring]
    D --> R[Retraining, validation, approval, and rollback]
```

Each prediction should retain its snapshot, installation, aircraft, component, batch, raw score and type, calibrated risk, alert stages, capacity-deferral reason, data quality, history coverage, scoring method, reason codes, and model/feature/calibrator/policy/data versions.

### Cold start

For components with fewer than five cycles since installation, the scoring path blends the model score with a smoothed part-number baseline, aircraft age, and initial installation sensor readings. Unknown part numbers use the global training baseline. Outputs identify the fallback method, limited history, and unavailable initial sensors.

### Missing and delayed data

Missingness indicators are preserved before imputation. Logistic regression uses training-fitted median imputation; LightGBM uses native numerical missing-value handling. Late records cannot enter earlier snapshots. Critically unavailable inputs trigger a baseline/rule-based fallback rather than an unsupported precise score.

### Drift and retraining

PSI utilities use training-reference quantile bins and a separate missing bin. Monitoring covers top SHAP features, score distributions, unknown categories, missingness, alert rates, and delayed outcome metrics. Retraining requires matured labels, renewed entity/time validation, business-policy replay, approval, and rollback readiness.

This PoC does not implement a live AMOS/Maximo integration. Scores support inspection prioritization only; they do not autonomously remove parts, release aircraft, or establish airworthiness.

## Reproduction

### Environment

Python 3.12 and dependencies are managed with `uv` and locked in `uv.lock`.

```bash
uv sync
```

### Generate and validate the dataset

```bash
uv run python data/generate_mock_data.py \
  --config configs/mock_data.yaml \
  --output data/generated/mock_v0_2_0 \
  --combined-output data/mro_dataset.csv

uv run python -m src.validate_data --data-dir data/generated/mock_v0_2_0

uv run python -m src.split_data \
  --data-dir data/generated/mock_v0_2_0 \
  --output artifacts/splits/mock_v0_2_0
```

### Build features and train candidates

```bash
uv run python -m src.build_features \
  --data-dir data/generated/mock_v0_2_0 \
  --output artifacts/features/mock_v0_2_0/features.parquet

uv run python -m src.train_models \
  --features artifacts/features/mock_v0_2_0/features.parquet \
  --split artifacts/splits/mock_v0_2_0/split_manifest.parquet \
  --output artifacts/experiments/validation_repro
```

Use a new experiment directory for each independent run. Existing frozen experiments cannot be silently overwritten.

### Evaluate the frozen model and generate explanations

```bash
uv run python -m src.evaluate_frozen \
  --experiment artifacts/experiments/validation_repro \
  --features artifacts/features/mock_v0_2_0/features.parquet \
  --split artifacts/splits/mock_v0_2_0/split_manifest.parquet \
  --data-dir data/generated/mock_v0_2_0 \
  --output artifacts/frozen_test/validation_repro

uv run python -m src.generate_explanations \
  --experiment artifacts/experiments/validation_v1 \
  --features artifacts/features/mock_v0_2_0/features.parquet \
  --split artifacts/splits/mock_v0_2_0/split_manifest.parquet \
  --test-dir artifacts/frozen_test/validation_v1
```

### Tests, Notebook, and dataset package

```bash
uv run pytest -q

uv run jupyter nbconvert \
  --to notebook --execute --inplace notebooks/poc_evaluation.ipynb

uv run python -m src.package_dataset
```

The executed Notebook includes EDA, split evidence, candidate comparison, the frozen-test PR curve, the SHAP summary, and high-risk cards. The optional dataset package command creates a CRC-validated ZIP with a SHA-256 manifest; the required dataset distribution is already provided in Git.

## Project documentation

- [Dataset dictionary](DATASET.md)
- [Architecture and concept design](mro-requirements-and-concept-design.md)
- [Technical selection review](mro-technical-selection-review.md)
- [Implementation and acceptance checklist](mro-TODO.md)
- [Evaluation Notebook](notebooks/poc_evaluation.ipynb)

The design document also records the reviewed NGAFID, NASA C-MAPSS/N-CMAPSS, FAA SDR, MaintNet, and related open-source resources. Their labels differ from this task and are not mixed with the mock-data evaluation.

## Limitations and future work

- The synthetic degradation mechanism makes some signals easier to learn than real MRO data.
- Only 20 independent unscheduled-removal events are present in the frozen test; the confidence interval cannot establish stable real-world recall.
- Scheduled removals are treated as censored competing events, defining a specific evaluation population. Selection bias and competing-risk modeling require attention with real data.
- The current data do not actually simulate cross-aircraft component transfers, although the splitter implements the required connected-group logic.
- Aggregated rolling features may discard useful high-frequency temporal patterns.
- Strict hard capacity produces zero alerts for batches below 20 components; operational batch aggregation or an explicitly approved alternative would be needed.
- Ten test alerts had pending outcomes. They count toward workload, but final false-positive assessment must be refreshed after label maturity.
- Calibration can degrade under fleet, time, and maintenance-policy changes.
- SHAP describes model attribution, not proven physical causality.
- Live production integration, regulatory assurance, and real-fleet validation remain outside this PoC.

Future work includes real-data external validation, cross-aircraft transfer scenarios, CatBoost as an evidence-gated extension, competing-risk or survival models, temporal trajectory models, uncertainty estimation, threshold-neighborhood analysis, calibration plots, and cost-sensitive maintenance optimization.
