from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import MAX_ALERT_RATE, TARGET_EVENT_RECALL
from src.explain import attach_alert_reason_codes
from src.threshold import (
    alert_policy_report,
    apply_batch_alert_capacity,
    evaluate_complete_alert_policy,
)
from src.training import TARGET, safe_roc_auc, score_with_bundle, write_prediction_contract


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _lead_time_metrics(decisions: pd.DataFrame, removals: pd.DataFrame) -> dict[str, Any]:
    positive = decisions.loc[decisions[TARGET].eq(1)].copy()
    removal_cycles = removals.set_index("removal_event_id")["removal_cycle"]
    positive["actual_removal_cycle"] = positive["removal_event_id"].map(removal_cycles)
    detected = positive.loc[positive["final_alert"]].sort_values("flight_date")
    first_alerts = detected.drop_duplicates("removal_event_id", keep="first")
    lead = first_alerts["actual_removal_cycle"] - first_alerts["aircraft_total_cycles"]
    return {
        "detected_events": int(first_alerts["removal_event_id"].nunique()),
        "mean_first_alert_lead_cycles": float(lead.mean()) if len(lead) else None,
        "median_first_alert_lead_cycles": float(lead.median()) if len(lead) else None,
        "min_first_alert_lead_cycles": float(lead.min()) if len(lead) else None,
        "max_first_alert_lead_cycles": float(lead.max()) if len(lead) else None,
    }


def _cluster_bootstrap(
    decisions: pd.DataFrame, iterations: int = 200, seed: int = 20261112
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    groups = decisions["entity_group_id"].drop_duplicates().to_numpy()
    by_group = {group: frame for group, frame in decisions.groupby("entity_group_id")}
    samples = {"event_recall": [], "alert_rate": []}
    for _ in range(iterations):
        selected = rng.choice(groups, size=len(groups), replace=True)
        bootstrap = pd.concat([by_group[group] for group in selected], ignore_index=True)
        metrics = evaluate_complete_alert_policy(bootstrap)
        samples["event_recall"].append(metrics["event_recall"])
        samples["alert_rate"].append(metrics["alert_rate"])
    return {
        key: [float(value) for value in np.quantile(values, [0.025, 0.975])]
        for key, values in samples.items()
    }


def _slice_metrics(decisions: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    for value, group in decisions.groupby(column, dropna=False):
        metrics = evaluate_complete_alert_policy(group)
        rows.append(
            {
                "slice_column": column,
                "slice_value": value,
                "events": group["removal_event_id"].dropna().nunique(),
                **metrics,
            }
        )
    return pd.DataFrame(rows)


def evaluate_frozen(
    experiment_dir: str | Path,
    features_path: str | Path,
    split_path: str | Path,
    data_dir: str | Path,
    output_dir: str | Path,
    bootstrap_iterations: int = 200,
) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Frozen test output already exists and will not be overwritten: {output}")
    evaluation_marker = experiment / "TEST_EVALUATED.json"
    if evaluation_marker.exists():
        raise ValueError("The frozen model has already been evaluated; choose a new independent evaluation protocol")
    with (experiment / "freeze_manifest.json").open(encoding="utf-8") as handle:
        freeze = json.load(handle)
    if freeze["test_evaluated"]:
        raise ValueError("The frozen model has already been marked as evaluated")
    if freeze["features_sha256"] != _sha256(features_path):
        raise ValueError("Feature artifact hash differs from the frozen training input")
    if freeze["split_sha256"] != _sha256(split_path):
        raise ValueError("Split manifest hash differs from the frozen selection input")
    bundle = joblib.load(experiment / "frozen_model.joblib")

    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(split_path)[["snapshot_id", "final_use", "entity_group_id"]]
    frame = features.merge(splits, on="snapshot_id", validate="one_to_one")
    test = frame.loc[frame["final_use"].eq("test")].copy()
    if test.empty:
        raise ValueError("Frozen test partition is empty")
    scored = score_with_bundle(bundle, test)
    decisions = apply_batch_alert_capacity(
        scored,
        freeze["threshold"],
        suppression_cycles=freeze["suppression_cycles"],
        max_alert_rate=freeze["max_alert_rate"],
    )
    decisions["model_version"] = freeze["candidate_id"]
    decisions["feature_version"] = freeze["features_sha256"]
    decisions["calibrator_version"] = "sigmoid_v1"
    decisions["policy_version"] = "batch_cap_5pct_suppress_30_v1"
    decisions["data_version"] = freeze["data_version"]
    decisions = attach_alert_reason_codes(decisions, bundle, freeze["threshold"])

    metrics = evaluate_complete_alert_policy(decisions)
    metrics.update(alert_policy_report(decisions))
    labeled = decisions.loc[decisions[TARGET].notna()]
    metrics["brier_score"] = float(
        brier_score_loss(labeled[TARGET].astype(int), labeled["risk_score"])
    )
    metrics["roc_auc"] = safe_roc_auc(labeled[TARGET], labeled["risk_score"])
    metrics["actual_unscheduled_events"] = int(decisions["removal_event_id"].dropna().nunique())
    removals = pd.read_parquet(Path(data_dir) / "removal_events.parquet")
    metrics.update(_lead_time_metrics(decisions, removals))
    metrics["bootstrap_95_percent_interval"] = _cluster_bootstrap(
        decisions, iterations=bootstrap_iterations
    )
    metrics["sla_met"] = bool(
        metrics["event_recall"] >= TARGET_EVENT_RECALL
        and metrics["alert_rate"] <= MAX_ALERT_RATE
        and metrics["over_capacity_batches"] == 0
    )

    output.mkdir(parents=True)
    write_prediction_contract(decisions, output / "test_predictions.parquet")
    slices = pd.concat(
        [
            _slice_metrics(decisions, "component_family"),
            _slice_metrics(decisions, "part_number"),
            _slice_metrics(decisions, "data_quality_status"),
        ],
        ignore_index=True,
    )
    slices.to_csv(output / "test_slice_metrics.csv", index=False)
    result = {
        "freeze_manifest_sha256": _sha256(experiment / "freeze_manifest.json"),
        "frozen_model_sha256": _sha256(experiment / "frozen_model.joblib"),
        "candidate_id": freeze["candidate_id"],
        "threshold": freeze["threshold"],
        "suppression_cycles": freeze["suppression_cycles"],
        **metrics,
    }
    with (output / "test_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with evaluation_marker.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "output_dir": str(output),
                "test_metrics_sha256": _sha256(output / "test_metrics.json"),
            },
            handle,
            indent=2,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a frozen MRO model once")
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--features", default="artifacts/features/mock_v0_2_0/features.parquet")
    parser.add_argument("--split", default="artifacts/splits/mock_v0_2_0/split_manifest.parquet")
    parser.add_argument("--data-dir", default="data/generated/mock_v0_2_0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=200)
    args = parser.parse_args()
    result = evaluate_frozen(
        args.experiment,
        args.features,
        args.split,
        args.data_dir,
        args.output,
        args.bootstrap_iterations,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
