from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score


def event_recall(
    y_true: np.ndarray | pd.Series,
    alerts: np.ndarray | pd.Series,
    event_ids: np.ndarray | pd.Series,
) -> float:
    frame = pd.DataFrame({"target": y_true, "alert": alerts, "event_id": event_ids})
    positive = frame[frame["target"].eq(1) & frame["event_id"].notna()]
    if positive.empty:
        return 0.0
    detected = positive.groupby("event_id")["alert"].max()
    return float(detected.mean())


def evaluate_alerts(
    y_true: np.ndarray | pd.Series,
    y_prob: np.ndarray | pd.Series,
    alerts: np.ndarray | pd.Series,
    event_ids: np.ndarray | pd.Series | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    decisions = np.asarray(alerts, dtype=bool)
    if not (len(y) == len(probabilities) == len(decisions)):
        raise ValueError("Targets, scores, and alerts must have equal lengths")
    alert_rate = float(decisions.mean()) if len(decisions) else 0.0
    result = {
        "alert_rate": alert_rate,
        "alerts_per_100": alert_rate * 100.0,
        "false_positives_per_100": float(np.sum(decisions & (y == 0)) / len(y) * 100.0),
        "precision": float(precision_score(y, decisions, zero_division=0)),
        "snapshot_recall": float(recall_score(y, decisions, zero_division=0)),
        "pr_auc": float(average_precision_score(y, probabilities)) if len(np.unique(y)) > 1 else 0.0,
    }
    if event_ids is not None:
        result["event_recall"] = event_recall(y, decisions, event_ids)
    return result


def optimize_mro_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    max_alert_rate: float = 0.05,
    target_recall: float = 0.80,
    event_ids: np.ndarray | None = None,
) -> dict[str, Any]:
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(y_prob, dtype=float)
    if len(y) != len(probabilities) or len(y) == 0:
        raise ValueError("Targets and probabilities must be non-empty and have equal lengths")
    if np.isnan(probabilities).any():
        raise ValueError("Cannot optimize a threshold with NaN scores")
    candidate_thresholds = np.unique(
        np.concatenate(
            [
                np.quantile(probabilities, np.linspace(0.90, 0.995, 200)),
                np.quantile(probabilities, [0.05, 0.5, 0.95]),
                [np.max(probabilities)],
            ]
        )
    )
    best: dict[str, Any] | None = None
    for threshold in candidate_thresholds:
        decisions = probabilities >= threshold
        metrics = evaluate_alerts(y, probabilities, decisions, event_ids)
        recall_key = metrics["event_recall"] if event_ids is not None else metrics["snapshot_recall"]
        if metrics["alert_rate"] > max_alert_rate:
            continue
        candidate = {
            "threshold": float(threshold),
            **metrics,
            "sla_met": bool(recall_key >= target_recall),
        }
        candidate_key = (recall_key, -metrics["alert_rate"], float(threshold))
        if best is None or candidate_key > best["_key"]:
            best = {**candidate, "_key": candidate_key}
    if best is None:
        best = {
            "threshold": float(np.max(probabilities) + np.finfo(float).eps),
            **evaluate_alerts(y, probabilities, np.zeros(len(y), dtype=bool), event_ids),
            "sla_met": False,
            "_key": (0.0, 0.0, 0.0),
        }
    best.pop("_key")
    return best


def apply_batch_alert_capacity(
    frame: pd.DataFrame,
    threshold: float,
    score_col: str = "risk_score",
    batch_col: str = "scoring_batch_id",
    component_col: str = "component_id",
    cycle_col: str = "component_total_cycles",
    max_alert_rate: float = 0.05,
    suppression_cycles: int = 30,
) -> pd.DataFrame:
    required = {score_col, batch_col, component_col, cycle_col}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing alert policy columns: {sorted(missing)}")
    if frame.duplicated([batch_col, component_col]).any():
        raise ValueError("A component can appear at most once per scoring batch")
    work = frame.copy()
    work["alert_before_suppression"] = work[score_col].ge(threshold)
    work["alert_after_suppression"] = False
    work["suppressed"] = False
    work["final_alert"] = False
    work["capacity_deferred"] = False
    work["capacity_deferred_reason"] = ""
    work["batch_size"] = 0
    work["batch_capacity"] = 0
    work["batch_alert_rate"] = 0.0
    last_alert_cycle: dict[Any, float] = {}

    for batch, batch_frame in work.groupby(batch_col, sort=True):
        batch_index = batch_frame.index
        capacity = int(np.floor(max_alert_rate * len(batch_frame)))
        work.loc[batch_index, "batch_size"] = len(batch_frame)
        work.loc[batch_index, "batch_capacity"] = capacity
        candidates = []
        for index, row in batch_frame.iterrows():
            if not row["alert_before_suppression"]:
                continue
            component = row[component_col]
            last_cycle = last_alert_cycle.get(component, -np.inf)
            if row[cycle_col] - last_cycle < suppression_cycles:
                work.loc[index, "suppressed"] = True
                continue
            work.loc[index, "alert_after_suppression"] = True
            candidates.append(index)
        ranked = sorted(
            candidates,
            key=lambda index: (
                -float(work.at[index, score_col]),
                str(work.at[index, component_col]),
                str(index),
            ),
        )
        selected = set(ranked[:capacity])
        for index in candidates:
            if index not in selected:
                work.loc[index, "capacity_deferred"] = True
                work.loc[index, "capacity_deferred_reason"] = "BATCH_CAPACITY_EXCEEDED"
        for index in selected:
            work.loc[index, "final_alert"] = True
            last_alert_cycle[work.at[index, component_col]] = float(work.at[index, cycle_col])
        work.loc[batch_index, "batch_alert_rate"] = len(selected) / len(batch_frame)
    return work


def alert_policy_report(policy_frame: pd.DataFrame) -> dict[str, Any]:
    batches = policy_frame.groupby("scoring_batch_id", sort=False).agg(
        alerts=("final_alert", "sum"),
        components=("component_id", "size"),
        batch_alert_rate=("batch_alert_rate", "first"),
        capacity=("batch_capacity", "first"),
    )
    return {
        "batches": len(batches),
        "overall_alert_rate": float(batches["alerts"].sum() / batches["components"].sum()),
        "max_batch_alert_rate": float(batches["batch_alert_rate"].max()),
        "over_capacity_batches": int((batches["alerts"] > batches["capacity"]).sum()),
        "capacity_deferred": int(policy_frame["capacity_deferred"].sum()),
        "suppressed": int(policy_frame["suppressed"].sum()),
    }


def evaluate_complete_alert_policy(policy_frame: pd.DataFrame) -> dict[str, Any]:
    target = policy_frame["unscheduled_removal_within_30_cycles"]
    labeled = policy_frame.loc[target.notna()].copy()
    labeled_target = labeled["unscheduled_removal_within_30_cycles"].astype(int)
    metrics = evaluate_alerts(
        labeled_target,
        labeled["risk_score"],
        labeled["final_alert"],
        labeled["removal_event_id"],
    )
    all_alerts = policy_frame["final_alert"].astype(bool)
    all_count = len(policy_frame)
    metrics.update(
        {
            "alert_rate": float(all_alerts.mean()),
            "alerts_per_100": float(all_alerts.mean() * 100.0),
            "false_positives_per_100": float(
                (all_alerts & target.eq(0)).sum() / all_count * 100.0
            ),
            "pending_outcome_alerts": int((all_alerts & target.isna()).sum()),
            "scored_components": all_count,
            "labeled_components": len(labeled),
        }
    )
    return metrics


def optimize_batch_alert_policy(
    frame: pd.DataFrame,
    target_recall: float = 0.80,
    max_alert_rate: float = 0.05,
    suppression_cycles: int = 30,
) -> dict[str, Any]:
    scores = frame["risk_score"].to_numpy(dtype=float)
    candidates = np.unique(np.quantile(scores, np.linspace(0.90, 0.995, 200)))
    best: dict[str, Any] | None = None
    for threshold in candidates:
        policy = apply_batch_alert_capacity(
            frame,
            float(threshold),
            max_alert_rate=max_alert_rate,
            suppression_cycles=suppression_cycles,
        )
        report = alert_policy_report(policy)
        if report["over_capacity_batches"] or report["overall_alert_rate"] > max_alert_rate:
            continue
        metrics = evaluate_complete_alert_policy(policy)
        key = (metrics["event_recall"], -report["overall_alert_rate"], float(threshold))
        if best is None or key > best["key"]:
            best = {
                "key": key,
                "threshold": float(threshold),
                "suppression_cycles": suppression_cycles,
                **metrics,
                **report,
                "sla_met": bool(metrics["event_recall"] >= target_recall),
            }
    if best is None:
        raise ValueError("No feasible alert policy was found")
    best.pop("key")
    return best
