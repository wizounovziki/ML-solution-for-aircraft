"""Build the illustrated submission from frozen predictions; no fitting or selection."""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.provenance import sha256, verify_frozen_experiment
from src.threshold import apply_batch_alert_capacity, evaluate_complete_alert_policy

TARGET = "unscheduled_removal_within_30_cycles"
FAMILY_LABELS = {"ENGINE": "Engine", "HYDRAULIC_PUMP": "Hydraulic pump",
                 "BRAKE_ACTUATOR": "Brake actuator", "AVIONICS_COMPUTER": "Avionics computer"}
COLORS = {"blue": "#23465c", "teal": "#117f83", "orange": "#cb6d36", "gray": "#81929b"}


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def feature_group(name: str, numeric: list[str], categorical: list[str]) -> str:
    if name.startswith("numeric__"):
        base = re.sub(r"_sp_\d+$", "", name[len("numeric__"):])
        base = base.removeprefix("missingindicator_")
    else:
        base = name.removeprefix("categorical__")
        base = next((c for c in sorted(categorical, key=len, reverse=True)
                     if base == c or base.startswith(c + "_")), base)
    for prefix, label in [("vibration_rms", "Vibration"), ("egt_margin", "EGT margin"),
                          ("sensor_temperature", "Temperature"), ("sensor_pressure", "Pressure"),
                          ("sensor_current", "Current"), ("fault_code", "Recent faults"),
                          ("maintenance_count", "Recent maintenance")]:
        if base.startswith(prefix):
            return label
    if base in {"part_number", "component_family"}:
        return "Component type"
    if base in {"aircraft_model", "is_legacy_aircraft", "operating_environment", "ambient_temperature"}:
        return "Aircraft and environment"
    if base in {"snapshot_cycle", "aircraft_total_cycles", "reading_cycle"}:
        return "Simulation clock"
    if base in {"sensor_quality", "history_coverage_cycles", "is_cold_start"}:
        return "Data quality and history"
    return "Usage and installation"


def contributions(bundle: dict, frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    if bundle["model_family"] != "logistic":
        raise ValueError("This submission renderer describes the frozen logistic experiment")
    pipeline = bundle["model"]
    original = frame[bundle["numeric_columns"] + bundle["categorical_columns"]]
    design = pipeline["preprocess"].transform(original)
    if hasattr(design, "toarray"):
        design = design.toarray()
    values = np.asarray(design) * pipeline["model"].coef_[0]
    names = pipeline["preprocess"].get_feature_names_out()
    grouped = pd.DataFrame(values, columns=names, index=frame.index).T.groupby(
        lambda name: feature_group(name, bundle["numeric_columns"], bundle["categorical_columns"])
    ).sum().T
    if "component_family" in frame and "EGT margin" in grouped:
        non_engine = frame["component_family"].ne("ENGINE")
        grouped.loc[non_engine, "Data quality and history"] += grouped.loc[non_engine, "EGT margin"]
        grouped.loc[non_engine, "EGT margin"] = 0.
    intercept = float(pipeline["model"].intercept_[0])
    reconstructed = intercept + grouped.sum(axis=1).to_numpy()
    if not np.allclose(reconstructed, frame["raw_score"].to_numpy(), atol=1e-5):
        raise ValueError("Grouped contributions do not reconstruct the saved raw predictions")
    return grouped, intercept


def build(run_id: str, output: Path) -> None:
    run = ROOT / "artifacts/runs" / run_id
    evidence = ROOT / "reports" / run_id
    output = output.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("Report output must remain in the project")
    figures, analysis = output.parent / "figures", output.parent / "analysis"
    figures.mkdir(parents=True, exist_ok=True)
    analysis.mkdir(parents=True, exist_ok=True)
    freeze = verify_frozen_experiment(run / "selection")
    metrics = json.loads((evidence / "test_metrics.json").read_text(encoding="utf-8"))
    explanation_manifest = json.loads((evidence / "explanation_manifest.json").read_text(encoding="utf-8"))
    predictions_path = run / "acceptance/test_predictions.parquet"
    if sha256(predictions_path) != explanation_manifest["source_test_predictions_sha256"]:
        raise ValueError("Saved acceptance predictions changed")
    if sha256(run / "selection/frozen_model.joblib") != metrics["frozen_model_sha256"]:
        raise ValueError("Saved model differs from the accepted model")
    predictions = pd.read_parquet(predictions_path)
    features = pd.read_parquet(run / "acceptance_features/features.parquet")
    frame = predictions.merge(features[[c for c in features if c not in predictions or c == "snapshot_id"]],
                              on="snapshot_id", validate="one_to_one")
    removals = pd.read_parquet(run / "acceptance_data/removal_events.parquet")
    removal_cycles = removals.set_index("removal_event_id")["removal_cycle"]
    frame["lead_cycles"] = frame["removal_event_id"].map(removal_cycles) - frame["aircraft_total_cycles"]
    bundle = joblib.load(run / "selection/frozen_model.joblib")
    candidates = pd.read_csv(evidence / "validation_candidate_metrics.csv")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "axes.labelcolor": COLORS["blue"],
                         "text.color": COLORS["blue"], "figure.constrained_layout.use": True})

    labels = ["Logistic\nunweighted", "Logistic\nbalanced", "LightGBM\nunweighted", "LightGBM\nbalanced"]
    selected = candidates.candidate_id.eq(freeze["candidate_id"])
    colors = [COLORS["teal"] if value else COLORS["gray"] for value in selected]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.7))
    axes[0].bar(labels, candidates.alerts_per_100, color=colors)
    axes[0].set(title="Validation alert burden", ylabel="Alerts per 100 scoring records")
    axes[1].bar(labels, candidates.brier_score, color=colors)
    axes[1].set(title="Validation probability error", ylabel="Brier score (lower is better)")
    for ax, values in zip(axes, (candidates.alerts_per_100, candidates.brier_score)):
        ax.set_ylim(0, max(values) * 1.22)
        for index, value in enumerate(values):
            ax.text(index, value, f"{value:.4f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle("All four candidates detected 12/12 known validation events; teal marks the selected model")
    save_figure(fig, figures / "model_comparison.png")

    policy = pd.read_parquet(run / "selection/validation_predictions.parquet")
    thresholds = np.unique(np.r_[np.quantile(policy.risk_score, np.linspace(.9, .995, 30)),
                                 freeze["threshold"], np.nextafter(1., 2.)])
    threshold_rows = []
    for threshold in thresholds:
        replay = apply_batch_alert_capacity(policy, threshold=float(threshold),
                                             suppression_cycles=freeze["suppression_cycles"],
                                             max_alert_rate=freeze["max_alert_rate"])
        result = evaluate_complete_alert_policy(replay)
        threshold_rows.append({"threshold": float(threshold), "event_recall": result["event_recall"],
                               "alerts_per_100": result["alerts_per_100"]})
    threshold_table = pd.DataFrame(threshold_rows)
    threshold_table.to_csv(analysis / "validation_thresholds.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.8, 3.7))
    ax.plot(threshold_table.alerts_per_100, threshold_table.event_recall * 100,
            "o-", color=COLORS["gray"], markersize=4, linewidth=1)
    val = freeze["validation_metrics"]
    ax.scatter([val["alerts_per_100"]], [val["event_recall"] * 100], color=COLORS["teal"], s=95, zorder=5)
    ax.annotate(f"Frozen threshold = {freeze['threshold']:.9f}",
                (val["alerts_per_100"], val["event_recall"] * 100), xytext=(12, -28), textcoords="offset points")
    ax.axhline(80, color=COLORS["orange"], linestyle="--", label="80% recall target")
    ax.axvline(5, color=COLORS["orange"], linestyle=":", label="5 alerts / 100 budget")
    ax.set(title="Threshold sensitivity on validation predictions", xlabel="Alerts per 100 scoring records",
           ylabel="Event recall (%)", ylim=(-4, 108), xlim=(-.05, max(5.2, threshold_table.alerts_per_100.max() * 1.05)))
    ax.legend(loc="lower right", frameon=False)
    save_figure(fig, figures / "validation_threshold.png")

    labeled = frame.loc[frame[TARGET].notna()]
    target = labeled[TARGET].astype(int)
    precision, recall, _ = precision_recall_curve(target, labeled.risk_score)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].plot(recall, precision, color=COLORS["teal"], label=f"AP = {average_precision_score(target, labeled.risk_score):.4f}")
    axes[0].axhline(target.mean(), color=COLORS["gray"], linestyle="--", label=f"Prevalence = {target.mean():.2%}")
    axes[0].set(title="Precision–recall curve", xlabel="Snapshot recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1.03))
    axes[0].legend(frameon=False, loc="lower left")
    calibration_rows = []
    for score, label, color in [("uncalibrated_score", "Before calibration", COLORS["gray"]),
                                 ("risk_score", "Final risk", COLORS["teal"])]:
        observed, predicted = calibration_curve(target, labeled[score], n_bins=10, strategy="uniform")
        axes[1].plot(predicted, observed, "o-", label=label, color=color, markersize=4)
        for x, y in zip(predicted, observed):
            calibration_rows.append({"score": score, "mean_prediction": x, "observed_fraction": y})
    axes[1].plot([0, 1], [0, 1], "--", color="#bcc9cf", label="Reference")
    axes[1].set(title="Calibration on observed labels", xlabel="Mean predicted probability", ylabel="Observed positive fraction")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle(f"Independent acceptance: {len(labeled):,} labeled snapshots")
    save_figure(fig, figures / "ranking_and_calibration.png")
    pd.DataFrame(calibration_rows).to_csv(analysis / "calibration_bins.csv", index=False)

    family = pd.read_csv(evidence / "test_slice_metrics.csv")
    family = family.loc[family.slice_column.eq("component_family")]
    family_labels = [FAMILY_LABELS[x].replace(" ", "\n", 1) for x in family.slice_value]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    axes[0].bar(family_labels, family.event_recall * 100, color=COLORS["teal"])
    axes[0].axhline(80, color=COLORS["orange"], linestyle="--", label="80% target")
    axes[0].set(title="Event recall by component family", ylabel="Event recall (%)", ylim=(0, 112))
    for index, row in enumerate(family.itertuples()):
        count = round(row.events * row.event_recall)
        axes[0].text(index, row.event_recall * 100 + 1, f"{count}/{row.events}", ha="center")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].bar(family_labels, family.alerts_per_100, color=COLORS["blue"])
    axes[1].axhline(5, color=COLORS["orange"], linestyle="--", label="Budget = 5")
    axes[1].set(title="Alert burden by component family", ylabel="Alerts per 100 scoring records", ylim=(0, 5.6))
    for index, value in enumerate(family.alerts_per_100):
        axes[1].text(index, value + .08, f"{value:.3f}", ha="center")
    axes[1].legend(frameon=False, loc="upper right")
    fig.suptitle("Independent acceptance: 384/393 events detected; 0/480 batches above capacity")
    save_figure(fig, figures / "acceptance_by_family.png")

    positives = frame.loc[frame[TARGET].eq(1)]
    negatives = frame.loc[frame[TARGET].eq(0)]
    sample = pd.concat([positives.sample(n=min(1000, len(positives)), random_state=20261113),
                        negatives.sample(n=min(1000, len(negatives)), random_state=20261113)])
    grouped, intercept = contributions(bundle, sample)
    importance = grouped.abs().mean().sort_values()
    importance.rename("mean_abs_grouped_log_odds_contribution").to_csv(analysis / "grouped_contributions.csv")
    fig, ax = plt.subplots(figsize=(8.8, 4.4))
    ax.barh(importance.index, importance.values, color=COLORS["teal"])
    ax.set(title="Grouped contributions of the frozen logistic model",
           xlabel="Mean absolute contribution to raw log-odds (2,000 saved snapshots)")
    save_figure(fig, figures / "grouped_contributions.png")

    true_alerts = positives.loc[positives.final_alert].sort_values("flight_date")
    chosen = []
    for family_name in ("ENGINE", "HYDRAULIC_PUMP"):
        chosen.append(true_alerts.loc[true_alerts.component_family.eq(family_name)].iloc[0])
    event_detection = positives.groupby("removal_event_id").final_alert.max()
    misses = positives.loc[positives.removal_event_id.isin(event_detection.index[~event_detection])]
    chosen.append(misses.sort_values("risk_score", ascending=False).iloc[0])
    selected = pd.DataFrame(chosen).reset_index(drop=True)
    local_groups, _ = contributions(bundle, selected)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for position, ax in enumerate(axes):
        values = local_groups.iloc[position].reindex(local_groups.iloc[position].abs().sort_values().index).tail(6)
        ax.barh(values.index, values.values, color=[COLORS["orange"] if x < 0 else COLORS["teal"] for x in values])
        ax.axvline(0, color=COLORS["gray"], linewidth=.7)
        row = selected.iloc[position]
        ax.set(title=f"{FAMILY_LABELS[row.component_family]}: {row.component_id}\nRisk {row.risk_score:.8f}; lead {int(row.lead_cycles)} cycles",
               xlabel="Grouped raw log-odds contribution")
    save_figure(fig, figures / "case_explanations.png")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for position, ax in enumerate(axes):
        row = selected.iloc[0 if position == 0 else 2]
        trajectory = frame.loc[frame.component_id.eq(row.component_id)].copy()
        cycle = float(removal_cycles.loc[row.removal_event_id])
        trajectory["cycles_before_removal"] = cycle - trajectory.aircraft_total_cycles
        trajectory = trajectory.loc[trajectory.cycles_before_removal.between(0, 50)].sort_values("cycles_before_removal", ascending=False)
        ax.plot(trajectory.cycles_before_removal, trajectory.risk_score, "o-", color=COLORS["teal"], markersize=4)
        emitted = trajectory.loc[trajectory.final_alert]
        if not emitted.empty:
            ax.scatter(emitted.cycles_before_removal, emitted.risk_score, color=COLORS["orange"], s=70, zorder=5, label="Emitted alert")
        ax.axhline(freeze["threshold"], color=COLORS["orange"], linestyle="--", label="Frozen threshold")
        ax.axvspan(0, 30, color=COLORS["teal"], alpha=.08)
        ax.set_yscale("logit")
        ax.set_ylim(.9, 1 - 1e-7)
        ax.set_xlim(50, 0)
        ax.set(title=("Detected event" if position == 0 else "Missed event") + f"\n{row.removal_event_id}",
               xlabel="Cycles before removal", ylabel="Risk probability (logit axis)")
        ax.legend(frameon=False, fontsize=8, loc="lower left")
    save_figure(fig, figures / "detected_and_missed_trajectory.png")

    cases = []
    for i, row in selected.iterrows():
        ordered = local_groups.iloc[i].reindex(local_groups.iloc[i].abs().sort_values(ascending=False).index)
        cases.append({"component_id": row.component_id, "removal_event_id": row.removal_event_id,
                      "snapshot_id": row.snapshot_id, "family": row.component_family,
                      "risk": float(row.risk_score), "raw_score": float(row.raw_score),
                      "lead_cycles": float(row.lead_cycles), "final_alert": bool(row.final_alert),
                      "group_contributions": ordered.to_dict()})
    write_text(analysis / "case_studies.json", json.dumps(cases, indent=2))

    def link(path: Path) -> str:
        return Path(os.path.relpath(path, output.parent)).as_posix()

    family_rows = "\n".join(f"| {FAMILY_LABELS[r.slice_value]} | {round(r.events*r.event_recall)}/{r.events} | {r.event_recall:.2%} | {r.alerts_per_100:.4f} |" for r in family.itertuples())
    candidate_rows = "\n".join(f"| {r.candidate_id} | {r.event_recall:.0%} | {r.alerts_per_100:.4f} | {r.brier_score:.6f} |" for r in candidates.itertuples())
    ci = metrics["bootstrap_95_percent_interval"]["event_recall"]
    c1, c2, miss = cases
    report = f"""# Aircraft Component Removal Prediction — Submission Report

The frozen unweighted logistic model detected **{metrics['detected_events']}/{metrics['actual_unscheduled_events']} events ({metrics['event_recall']:.2%})** on an independent simulated fleet, producing **461 alerts across 115,016 scoring records**. All **480 batches** remained within capacity. The acceptance criteria were event recall ≥80%, alerts ≤5 per 100 scoring records overall and per batch, and at least 100 events.

## 1. Data preparation and modeling methods

The task is to identify an unscheduled removal in the next 30 flight cycles for the same component installation. Development data contain 60 aircraft, 723 installations and 114,989 snapshots, with 2.1434% positive labeled snapshots. The generator supplies sensor trajectories, fault and maintenance histories, reporting delays, missing values, outliers and censoring. [Dataset dictionary]({link(ROOT / 'DATASET.md')}).

Features use both event time and availability time. Current readings unavailable at scoring are masked; they become eligible for later rolling history when they arrive. Means, standard deviations and slopes cover 5/15/30-cycle windows. An allowlist excludes identifiers, labels and future removal metadata. Preprocessing and baseline statistics are fitted on training data.

Aircraft/component connected groups are disjoint across outer development partitions. Training ends before 2024-09-01 (36,252 records, 102 events); separate sigmoid calibration uses 2,002 records and 14 events before 2025-03-01. Policy replay covers 2,394 records before 2025-09-01. Only 2,107 labels and 12 events were available by that freeze cutoff; the other 287 records still count toward alert burden.

Regularized logistic regression uses imputation, missingness indicators, splines, scaling and one-hot encoding. LightGBM supplies a nonlinear tree candidate with native numerical missingness. Weighted and unweighted variants provide four candidates under the same policy replay.

## 2. Model evaluation and threshold selection

| Validation candidate | Event recall | Alerts / 100 | Brier score |
| --- | ---: | ---: | ---: |
{candidate_rows}

![Validation model comparison]({link(figures / 'model_comparison.png')})

All candidates detected the 12 known validation events. Selection orders event recall, lower alert burden, fewer known false positives, lower Brier score and fitting cost. Unweighted logistic regression won on alert burden. LightGBM achieved better Brier scores, making calibration quality a relevant future comparison.

The frozen threshold is **{freeze['threshold']:.16f}**, with **30-cycle duplicate suppression** and capacity **floor(0.05 × batch size)**. Only emitted alerts update suppression state. Batches below 20 components have zero capacity under this rule.

![Validation threshold sensitivity]({link(figures / 'validation_threshold.png')})

This sensitivity replay uses saved validation predictions. The marker identifies the already frozen operating point. Independent acceptance predictions retain that threshold and policy.

The model and generator protocol were frozen before acceptance data generation. The new seed is 20261113, with the ACC2027 namespace, 60 aircraft, 712 installations and 115,016 snapshots. Actual simulated dates span **2027-01-04 to 2031-12-06**, after the prior dataset's end on 2026-12-06. Aircraft, components, installations, snapshots and removal identifiers have zero overlap. [Protocol and provenance]({link(evidence / 'test_metrics.json')}).

| Independent acceptance metric | Result |
| --- | ---: |
| Event recall | **97.71% (384/393)** |
| Alerts per 100 scoring records | **0.4008 (461 alerts)** |
| Batches above capacity | **0/480** |
| PR-AUC / ROC-AUC | 0.959018 / 0.995877 |
| Brier score | 0.003754 |
| Known outcome precision | 100% (384/384) |
| Censored or pending alerts | 77 |
| Median first alert lead | 7 cycles; range 1–17 |

![Independent precision–recall and calibration]({link(figures / 'ranking_and_calibration.png')})

Ranking and calibration use 113,130 snapshots with observed labels. A component contributes one scoring record per batch. Event recall counts each removal once. The 1,000-draw group bootstrap recall interval is **{ci[0]:.2%}–{ci[1]:.2%}**, preserving duplicate group weights.

| Component family | Detected / actual events | Recall | Alerts / 100 |
| --- | ---: | ---: | ---: |
{family_rows}

![Independent event recall and alert burden]({link(figures / 'acceptance_by_family.png')})

## 3. Predictions and interpretability examples

Predictions include snapshot, aircraft and component identifiers; raw log-odds; final risk; suppression and capacity decisions; data quality; reason codes; and model, feature and policy versions. All 461 emitted alerts have reason codes. [Per-event outcomes]({link(evidence / 'event_detection.csv')}).

The selected logistic model uses contributions from its transformed features relative to zero. Contributions are summed within operational groups; the model intercept plus all groups reconstructs the saved raw score. These contributions describe the model's raw log-odds. Calibration and cold start blending produce the separate risk probability. EGT channel contributions for non-engine components are grouped under data quality because those channels are structurally unavailable.

![Global grouped linear contributions]({link(figures / 'grouped_contributions.png')})

| Example | Component / event | Risk | Result |
| --- | --- | ---: | --- |
| Engine alert | {c1['component_id']} / {c1['removal_event_id']} | {c1['risk']:.8f} | Detected {int(c1['lead_cycles'])} cycles before removal |
| Hydraulic pump alert | {c2['component_id']} / {c2['removal_event_id']} | {c2['risk']:.8f} | Detected {int(c2['lead_cycles'])} cycles before removal |
| Avionics miss | {miss['component_id']} / {miss['removal_event_id']} | {miss['risk']:.8f} | Highest risk in the missed event's prediction window remained below threshold |

![Local explanation examples]({link(figures / 'case_explanations.png')})

For engine alerts, engineers can review vibration and EGT trends against sensor and fault records. For hydraulic pump alerts, inspection can focus on vibration, pressure, leakage evidence and related maintenance. Final maintenance action remains with authorized engineers.

![Detected and missed event trajectories]({link(figures / 'detected_and_missed_trajectory.png')})

Nine events were missed: five avionics, two brake actuator and two hydraulic pump events. All nine had maximum risk below the frozen threshold in their positive windows. The nearest miss reached 0.99956758 against a threshold of 0.99959184. There were zero known false positives; 77 censored or pending alerts are included in burden and excluded from known outcome precision. [Example inputs and grouped contributions]({link(analysis / 'case_studies.json')}).

## 4. Proposed deployment and monitoring

```mermaid
flowchart LR
    A[Flight and MRO feeds] --> B[Availability filtered features]
    B --> C[Versioned batch scoring]
    C --> D[Calibration, suppression and capacity]
    D --> E[Engineer inspection queue]
    E --> F[Matured outcomes and drift monitoring]
    F --> G[Reviewed retraining and rollback]
```

The proposed daily batch service consumes flight, sensor, fault, installation and maintenance records. A deployment package holds the fitted preprocessor, model, calibrator, feature schema, threshold and capacity policy. Persisting last emitted alert cycles across runs preserves suppression. Queue records carry risk, reasons, data quality, lead context, inspection status and engineer feedback for AMOS/Maximo integration.

| Monitor | Measure and proposed response |
| --- | --- |
| Inputs | Track feed delay, missingness, unknown categories and out-of-range values; flag affected queue records and use the finite baseline fallback for invalid model scores |
| Predictions | Track score PSI and PSI of important features against the training reference; sustained changes trigger data and model review |
| Alert operations | Track total and per-batch burden, capacity deferrals, suppression and processing latency; enforce the 5% capacity guard at emission |
| Matured outcomes | After the 30-cycle observation window and reporting delay, compute event recall, known precision, Brier score, lead time and family/time slices with event counts and intervals |
| Releases | Fit a challenger on matured labels, validate calibration and alert policy, reserve an independent interval, require release review and keep the previous model and policy for rollback |

The repository implements offline scoring, cold start fallback, missingness handling, PSI and alert replay. The production service, persistent suppression store, live queue and release automation are proposed integration work.

## 5. Limitations and future improvements

1. Acceptance uses a new simulated fleet under the same degradation mechanism. Real fleet validation needs operational histories and actual maintenance interventions.
2. The 12 policy events and 14 calibration events constrain evidence for threshold stability and probability calibration. Add rolling development windows and larger natural-prevalence calibration samples.
3. Median first alert lead is seven cycles and the minimum is one. Evaluate inspection scheduling and compare policies that value earlier detection using development data.
4. Scheduled removals are competing events; 77 emitted alerts have censored or pending outcomes. Follow matured outcomes and evaluate competing-risk targets for operational data.
5. Explicit component transfers and positive events for new part numbers need additional scenarios. Future model changes require a new independent acceptance interval.

The [executed Notebook]({link(ROOT / 'notebooks/independent_acceptance_2027.ipynb')}) and [evidence directory]({link(evidence)}) provide the saved experiment outputs. The [submission checklist]({link(ROOT / 'docs/submission-checklist.md')}) maps the requested outputs to report sections and files.
"""
    write_text(output, report)
    figure_paths = sorted(figures.glob("*.png"))
    write_text(analysis / "report_manifest.json", json.dumps({
        "run_id": run_id, "source_predictions_sha256": sha256(predictions_path),
        "frozen_model_sha256": sha256(run / "selection/frozen_model.joblib"),
        "source_fingerprint": freeze["source_fingerprint"],
        "validation_predictions_sha256": sha256(run / "selection/validation_predictions.parquet"),
        "report_sha256": sha256(output),
        "figures": {p.name: sha256(p) for p in figure_paths},
        "sample_rows": len(sample), "model_intercept": intercept,
        "maximum_grouped_additivity_error": float(np.max(np.abs(intercept + grouped.sum(axis=1) - sample.raw_score))),
        "method": "Read saved predictions and frozen coefficients; sensitivity uses validation predictions only",
    }, indent=2))
    print(f"Built {output.relative_to(ROOT)} with {len(figure_paths)} result figures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="corrected_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/final_report.md")
    args = parser.parse_args()
    build(args.run_id, args.output)
