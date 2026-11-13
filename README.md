# Aircraft MRO Unscheduled Removal Prediction

This project predicts whether an aircraft component will undergo an **unscheduled removal within the next 30 flight cycles**. It combines availability controls for data, model comparison, probability calibration, alert capacity limits and explanations for maintenance review.

The corrected experiment selected **unweighted logistic regression** and passed acceptance on a new simulated fleet covering **2027–2031**: **384 of 393 removal events detected (97.71%)**, **461 alerts across 115,016 scoring records**, and **zero batches above capacity**.

Read the [acceptance report](reports/corrected_v2/summary.md) or the [executed evaluation Notebook](notebooks/independent_acceptance_2027.ipynb) for the results and supporting evidence.

## Acceptance results

The model, calibrator, threshold, alert policy and independent data generation protocol were frozen before the acceptance dataset was generated. The registered criteria require event recall of at least 80%, an alert rate of at most 5% overall and in every batch, and at least 100 removal events.

| Metric | Result | Acceptance criterion |
| --- | ---: | ---: |
| **Event recall** | **97.71% — 384/393 events** | ≥80% |
| **Alerts per 100 component scoring records** | **0.4008 — 461 alerts** | ≤5 |
| **Batches above capacity** | **0/480** | 0 |
| Removal events evaluated | 393 | ≥100 |
| Maximum batch alert rate | 2.0833% | ≤5% |
| PR-AUC | 0.959018 | Descriptive |
| ROC-AUC | 0.995877 | Descriptive |
| Brier score | 0.003754 | Descriptive |
| Precision among alerts with known outcomes | 100% — 384/384 | Descriptive |
| Alerts with censored or pending outcomes | 77 | Included in alert burden |
| Median first alert lead time | 7 cycles | Range: 1–17 cycles |

The denominator for alert burden is **component scoring records**. A component scored in multiple batches contributes one record per batch. Event recall counts each removal event once and considers it detected when at least one final alert falls within its valid prediction window.

Cluster bootstrap with 1,000 draws gave a 95% event recall interval of **96.27%–98.97%**. The corresponding interval for alerts per 100 scoring records was **0.3824–0.4192**. Resampling preserves the multiplicity of each sampled aircraft/component group.

### Results by component family

| Component family | Detected / actual events | Event recall | Missed events | Alerts per 100 records |
| --- | ---: | ---: | ---: | ---: |
| Engine | 90/90 | 100.00% | 0 | 0.3616 |
| Hydraulic pump | 104/106 | 98.11% | 2 | 0.4487 |
| Brake actuator | 122/124 | 98.39% | 2 | 0.4976 |
| Avionics computer | 68/73 | 93.15% | 5 | 0.2955 |
| **Total** | **384/393** | **97.71%** | **9** | **0.4008** |

The [event detection table](reports/corrected_v2/event_detection.csv) identifies every detected and missed event. The [slice metrics](reports/corrected_v2/test_slice_metrics.csv) also cover part numbers and data quality. Data quality slices describe alerts emitted within each subset of snapshots; an event can appear in multiple such subsets.

## Reproduce the experiment

Requirements: **Python 3.12** and **uv**. Run commands from the repository root.

```sh
uv sync --locked
uv run python -m src.rebuild_experiment --run-id reproduction_v2
```

Use a fresh run ID for each reproduction. Both `artifacts/runs/<run-id>/` and `reports/<run-id>/` must be new directories.

The command runs the 27 automated tests, generates the development tables when needed, rebuilds features and splits, fits all four candidates, selects and freezes a model, generates the independent fleet, evaluates it once and writes reports and explanations.

| Output | Location |
| --- | --- |
| Model, calibrator and cold start baselines | `artifacts/runs/<run-id>/selection/frozen_model.joblib` |
| Frozen configuration and provenance | `artifacts/runs/<run-id>/selection/freeze_manifest.json` |
| Independent raw data and features | `artifacts/runs/<run-id>/acceptance_data/` and `acceptance_features/` |
| Complete predictions in CSV and Parquet | `artifacts/runs/<run-id>/acceptance/test_predictions.*` |
| Acceptance metrics and completion marker | `artifacts/runs/<run-id>/acceptance/` and `selection/TEST_EVALUATED.json` |
| Readable report, event table, explanations and figures | `reports/<run-id>/` |

Generated raw tables and full experiment artifacts are excluded from Git. The repository includes the development CSV, compact acceptance evidence and executed Notebook.

Run tests separately with:

```sh
uv run pytest -q
```

Reexecute the delivered Notebook with:

```sh
uv run jupyter nbconvert --to notebook --execute --inplace notebooks/independent_acceptance_2027.ipynb
```

## Data and target

The generator creates aircraft, component installations, sensor readings, fault events, maintenance records and removal events, then derives snapshot labels. Degradation trajectories include rising vibration, falling engine EGT margin, changing temperatures and pressures, and increasing fault frequency. Data quality scenarios include sensor dropouts, missingness on legacy aircraft, outliers, reporting delays, new part numbers and drift.

| Property | Development data | Independent acceptance data |
| --- | ---: | ---: |
| Dataset version | `0.2.0` | `0.3.0-acceptance-2027` |
| Random seed | 20261112 | 20261113 |
| Aircraft | 60 | 60 |
| Component / installation records | 723 | 712 |
| Scoring snapshots | 114,989 | 115,016 |
| Fault events | 3,683 | 3,571 |
| Maintenance events | 2,133 | 2,092 |
| Removal events | 484 | 472 |
| Unscheduled removals | 404 | 393 |
| Snapshots with observed labels | 113,090 | 113,130 |
| Positive snapshots | 2,424 — 2.1434% | 2,358 — 2.0843% |
| Censored snapshots | 1,899 | 1,886 |

The development dataset is available as [`data/mro_dataset.csv`](data/mro_dataset.csv), approximately 48 MB. The [data dictionary](DATASET.md) documents the schema and label rules. The [independent generation configuration](configs/mock_data_acceptance_2027.yaml) and its [registered protocol](configs/acceptance_2027.json) reproduce the acceptance fleet.

For a snapshot at cycle `t`:

```text
y = 1 when the same installation has an unscheduled removal in (t, t + 30 cycles]
```

Positive labels reference a unique removal event. Insufficient follow-up and scheduled removals within the prediction window produce censored labels. `label_available_at` records when an outcome becomes usable for supervised fitting or selection. Snapshots at or after removal are excluded from the in-service population.

The combined CSV contains labels and evaluation metadata. Build model inputs through the feature builder, which applies availability rules and an explicit feature allowlist.

## Evaluation design

### Development and policy selection

Aircraft and components are joined into connected groups. Complete groups are assigned to outer partitions, with global time boundaries determining their permitted use.

| Use | Snapshot interval, UTC | Scoring records | Labels used | Known positive events |
| --- | --- | ---: | ---: | ---: |
| Training | Before 2024-09-01 | 36,252 | 36,252 | 102 |
| Calibration | 2024-09-01 to 2025-03-01 | 2,002 | 2,002 | 14 |
| Policy selection | 2025-03-01 to 2025-09-01 | 2,394 | 2,107 | 12 |

Intervals include their start and exclude their end. Training labels must be available before the training cutoff; calibration labels before the calibration cutoff. Policy selection uses labels available before **2025-09-01**. The latest policy label used became available on **2025-08-31 at 18:00 UTC**.

The other 287 policy records remain in the scoring population, with their unavailable or censored outcomes hidden. They contribute to batch capacity, suppression and alert burden.

### Independent acceptance interval

The prior dataset ends on **2026-12-06**. The registered acceptance interval is **2027-01-01 through 2031-12-31**, with actual snapshots from **2027-01-04 at 18:00 UTC** to **2031-12-06 at 00:00 UTC**. These dates describe simulated flight history.

Acceptance uses seed `20261113` and the `ACC2027` entity namespace. Aircraft, components, installations, snapshots and removal events have **zero identifier overlap** with the prior data.

The evaluator verifies the frozen protocol, model hash, source fingerprint, feature provenance, time separation, connected groups and entity separation. The completion marker prevents the same frozen experiment from being evaluated again. The [freeze manifest](reports/corrected_v2/freeze_manifest.json) and [acceptance metrics](reports/corrected_v2/test_metrics.json) record these checks and timestamps.

## Features, models and selection

Features cover component usage, installation age, operating environment, sensors, recent faults, maintenance, missingness and history coverage. There are 70 numerical and five categorical inputs before model specific transformations.

For each scoring time, raw readings and 5/15/30-cycle rolling means, standard deviations and slopes use observations whose event time **and availability time** are at or before that time. A reading that arrives late remains unavailable in its earlier snapshot and can enter rolling history after it arrives. Rolling windows include the boundary observation so that a 5-cycle slope can use consecutive readings sampled five cycles apart.

The numerical feature allowlist rejects undeclared columns, including future removal metadata. Imputation, scaling, category mappings and cold start baselines are fitted on development training data; sigmoid calibration uses its separate calibration partition.

Four candidates use the same partitions and alert rules:

| Candidate | Policy event recall | Alerts per 100 | Known false positives per 100 | PR-AUC | Brier score |
| --- | ---: | ---: | ---: | ---: | ---: |
| **Logistic regression, unweighted** | **100%** | **0.5430** | **0** | **0.992562** | **0.006092** |
| Logistic regression, balanced | 100% | 0.5848 | 0 | 0.989358 | 0.004929 |
| LightGBM, unweighted | 100% | 0.6266 | 0 | 0.987930 | 0.002989 |
| LightGBM, balanced | 100% | 0.5848 | 0 | 0.987824 | 0.002835 |

Selection ranks candidates by event recall, lower alert rate, fewer known false positives, lower Brier score and lower fitting cost. All four detected the 12 known policy events. **Unweighted logistic regression won on alert burden**; LightGBM had lower Brier scores. Full values are in the [candidate comparison](reports/corrected_v2/validation_candidate_metrics.csv).

Logistic regression uses median imputation, missingness indicators, splines, scaling and one-hot encoding. LightGBM uses native numerical missingness and categories defined from training data. Both model families support frozen evaluation and alert explanations.

Frozen alert settings:

| Setting | Value |
| --- | --- |
| Model | `logistic_unweighted` |
| Calibration | Sigmoid |
| Risk threshold | `0.9995918400998122` |
| Duplicate alert suppression | 30 component cycles |
| Batch capacity | `floor(0.05 × batch size)` |

Only emitted alerts update suppression state. Alerts before suppression, after suppression, deferred by capacity and finally emitted remain visible in the prediction output. Under the strict capacity rule, batches with fewer than 20 components have zero capacity.

## Explanations and maintenance review

The selected logistic model explains raw log-odds through contributions from its transformed features relative to zero. The model intercept plus all contributions reconstructs the raw score, with a maximum checked error of approximately **2.13 × 10⁻¹³** in the summary sample. Calibrated risk and the cold start blend are reported separately.

All **461 final alerts** have reason codes. The [two high risk component cards](reports/corrected_v2/high_risk_cards.md) include snapshot and aircraft identifiers, risk scores, contribution directions and inspection suggestions. The [explanation manifest](reports/corrected_v2/explanation_manifest.json) records the method and additivity checks.

The files [`shap_feature_importance.csv`](reports/corrected_v2/shap_feature_importance.csv) and [`shap_summary.png`](reports/corrected_v2/figures/shap_summary.png) retain their existing output names. For the selected logistic model they contain linear contributions; LightGBM candidates use Tree SHAP on raw scores.

Maintenance engineers can use risk, reasons and data quality to prioritize inspections. Authorized personnel retain the component removal decision.

## Reliability fixes and verification

The corrected experiment addresses eight findings from the [project assessment](reports/project_assessment_2026-11-13.md):

| Area | Implemented behavior |
| --- | --- |
| Current sensor readings | Hide readings unavailable at the scoring time and update missingness indicators |
| Policy labels | Hide outcomes unavailable at the policy freeze cutoff while preserving the scoring population |
| Candidate model support | Complete freezing and explanation for a selected logistic model |
| Cluster bootstrap | Preserve repeated group weights in event recall |
| Invalid model scores | Produce a finite baseline fallback with an explicit scoring method |
| Constant reference PSI | Detect values shifting outside the reference range |
| Feature schema | Reject undeclared numerical inputs |
| Frozen provenance | Verify model identity and the implementation fingerprint before evaluation |

The **27 automated tests pass**, covering availability controls, entity separation, calibration, alert capacity, explanation additivity, missing data, PSI and a complete logistic selection flow. See the [saved test output](reports/corrected_v2/test_results.txt).

## Project structure

| Path | Purpose |
| --- | --- |
| [`src/rebuild_experiment.py`](src/rebuild_experiment.py) | Reproduction entry point for training, independent acceptance and reports |
| [`src/mock_data.py`](src/mock_data.py), [`data/generate_mock_data.py`](data/generate_mock_data.py) | Synthetic event and snapshot generation |
| [`src/validation.py`](src/validation.py), [`src/validate_data.py`](src/validate_data.py) | Dataset integrity and target checks |
| [`src/features.py`](src/features.py), [`src/build_features.py`](src/build_features.py) | Availability filtering and feature construction |
| [`src/splitter.py`](src/splitter.py), [`src/split_data.py`](src/split_data.py) | Connected groups, temporal boundaries and selection labels |
| [`src/models.py`](src/models.py), [`src/training.py`](src/training.py) | Preprocessing, fitting, calibration and candidate selection |
| [`src/threshold.py`](src/threshold.py) | Event metrics, threshold search, suppression and batch capacity |
| [`src/acceptance.py`](src/acceptance.py), [`src/evaluate_frozen.py`](src/evaluate_frozen.py) | Protocol checks and independent evaluation |
| [`src/provenance.py`](src/provenance.py) | Model and source identity verification |
| [`src/explain.py`](src/explain.py), [`src/generate_explanations.py`](src/generate_explanations.py) | Contributions, reason codes, cards and figures |
| [`src/production.py`](src/production.py) | Cold start fallback and PSI utilities |
| [`tests/`](tests/) | Unit, integration and regression checks |
| [`configs/`](configs/) | Development data, independent data and acceptance protocol |
| [`reports/corrected_v2/`](reports/corrected_v2/) | Current acceptance evidence |
| [`notebooks/independent_acceptance_2027.ipynb`](notebooks/independent_acceptance_2027.ipynb) | Executed acceptance analysis |

## Scope and next work

Acceptance covers a new simulated fleet under the generator's degradation mechanism. Real fleet validation needs operational histories, reporting delays, maintenance interventions and component transfer records. The current acceptance found nine missed events and a median detection lead of seven cycles; further model development should evaluate these outcomes on development data and reserve a new independent interval for acceptance.

Seventy-seven alerts have censored or pending outcomes, so the reported precision applies to the 384 alerts with observed labels. Scheduled removals are treated as competing events under the current target definition.

Production work includes an independent batch scoring entry point, suppression state persisted across runs, delayed outcome monitoring, drift triggers, model release controls and rollback. The splitter supports components shared across aircraft; explicit transfer scenarios remain part of the data development work.

Further project references:

- [Dataset dictionary](DATASET.md)
- [Requirements and concept design](mro-requirements-and-concept-design.md)
- [Technical selection review](mro-technical-selection-review.md)
- [Implementation checklist](mro-TODO.md)
- [Project assessment preceding the fixes](reports/project_assessment_2026-11-13.md)

The concept design, technical review and original `poc_evaluation.ipynb` document the earlier design and experiment. Current acceptance figures and conclusions are recorded in `reports/corrected_v2/` and this README.
