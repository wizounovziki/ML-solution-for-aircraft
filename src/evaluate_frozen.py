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
from src.provenance import verify_frozen_experiment
from src.acceptance import validate_acceptance_inputs


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
    if iterations < 1 or decisions.empty:
        raise ValueError("Bootstrap requires non-empty decisions and positive iterations")
    totals = decisions.groupby("entity_group_id").agg(
        alerts=("final_alert", "sum"), rows=("final_alert", "size")
    )
    positive = decisions.loc[decisions[TARGET].eq(1) & decisions["removal_event_id"].notna()]
    if positive.groupby("removal_event_id")["entity_group_id"].nunique().gt(1).any():
        raise ValueError("An event must belong to exactly one bootstrap group")
    events = positive.groupby(["entity_group_id", "removal_event_id"])["final_alert"].max()
    event_totals = events.groupby(level=0).agg(["sum", "count"]).reindex(totals.index, fill_value=0)
    detected = event_totals["sum"].to_numpy(dtype=float)
    counts = event_totals["count"].to_numpy(dtype=float)
    alerts, rows = totals["alerts"].to_numpy(), totals["rows"].to_numpy()
    samples = {"event_recall": [], "alert_rate": []}
    for _ in range(iterations):
        selected = rng.choice(len(totals), size=len(totals), replace=True)
        event_count = counts[selected].sum()
        if event_count:
            samples["event_recall"].append(float(detected[selected].sum() / event_count))
        samples["alert_rate"].append(float(alerts[selected].sum() / rows[selected].sum()))
    return {
        key: [float(value) for value in np.quantile(values, [0.025, 0.975])] if values else [None, None]
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
    acceptance_protocol_path: str | Path | None = None,
) -> dict[str, Any]:
    experiment = Path(experiment_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Frozen test output already exists and will not be overwritten: {output}")
    evaluation_marker = experiment / "TEST_EVALUATED.json"
    if evaluation_marker.exists():
        raise ValueError("The frozen model has already been evaluated; choose a new independent evaluation protocol")
    freeze = verify_frozen_experiment(experiment)
    if freeze["test_evaluated"]:
        raise ValueError("The frozen model has already been marked as evaluated")
    acceptance_evidence = None
    if acceptance_protocol_path is not None:
        acceptance_evidence = validate_acceptance_inputs(
            freeze, acceptance_protocol_path, features_path, split_path, data_dir
        )
    elif freeze["features_sha256"] != _sha256(features_path):
        raise ValueError("Feature artifact hash differs from the frozen training input")
    if acceptance_protocol_path is None and freeze["split_sha256"] != _sha256(split_path):
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
    decisions["feature_version"] = _sha256(features_path)
    decisions["calibrator_version"] = "sigmoid_v1"
    decisions["policy_version"] = "batch_cap_5pct_suppress_30_v1"
    decisions["data_version"] = acceptance_evidence["data_version"] if acceptance_evidence else freeze["data_version"]
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
    event_count = metrics["actual_unscheduled_events"]
    if event_count:
        recall = metrics["event_recall"]
        z = 1.959963984540054
        denominator = 1.0 + z * z / event_count
        center = (recall + z * z / (2 * event_count)) / denominator
        radius = z * np.sqrt(recall * (1 - recall) / event_count + z * z / (4 * event_count**2)) / denominator
        metrics["event_recall_wilson_95_iid_reference"] = [max(0., center - radius), min(1., center + radius)]
    metrics["bootstrap_95_percent_interval"] = _cluster_bootstrap(
        decisions, iterations=bootstrap_iterations
    )
    metrics["sla_met"] = bool(
        metrics["event_recall"] >= TARGET_EVENT_RECALL
        and metrics["alert_rate"] <= MAX_ALERT_RATE
        and metrics["over_capacity_batches"] == 0
    )
    if acceptance_evidence is not None:
        if bootstrap_iterations != freeze["acceptance_protocol"]["bootstrap_iterations"]:
            raise ValueError("Bootstrap iterations differ from the preregistered count")
        metrics["sla_met"] = bool(metrics["sla_met"] and metrics["actual_unscheduled_events"] >=
                                  freeze["acceptance_protocol"]["minimum_events"])

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
        "evaluation_features_sha256": _sha256(features_path),
        "evaluation_split_sha256": _sha256(split_path),
        "acceptance_evidence": acceptance_evidence,
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
    parser.add_argument("--acceptance-protocol")
    args = parser.parse_args()
    result = evaluate_frozen(
        args.experiment,
        args.features,
        args.split,
        args.data_dir,
        args.output,
        args.bootstrap_iterations,
        args.acceptance_protocol,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
