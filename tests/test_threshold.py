from __future__ import annotations

import numpy as np
import pandas as pd

from src.threshold import (
    alert_policy_report,
    apply_batch_alert_capacity,
    event_recall,
    optimize_mro_threshold,
)


def test_event_recall_counts_unique_events() -> None:
    assert event_recall(
        np.array([1, 1, 1]),
        np.array([1, 0, 0]),
        np.array(["E1", "E1", "E2"]),
    ) == 0.5


def test_threshold_optimizer_respects_alert_budget() -> None:
    target = np.array([0] * 95 + [1] * 5)
    scores = np.linspace(0, 1, 100)
    result = optimize_mro_threshold(target, scores)
    assert result["alert_rate"] <= 0.05
    assert result["snapshot_recall"] == 1.0


def test_every_batch_enforces_hard_capacity() -> None:
    frame = pd.DataFrame(
        {
            "scoring_batch_id": np.repeat(["B1", "B2"], 100),
            "component_id": [f"C{i}" for i in range(200)],
            "component_total_cycles": np.tile(np.arange(100), 2),
            "risk_score": np.linspace(0, 1, 200),
        }
    )
    result = apply_batch_alert_capacity(frame, threshold=0.0, suppression_cycles=0)
    report = alert_policy_report(result)
    assert report["over_capacity_batches"] == 0
    assert report["max_batch_alert_rate"] == 0.05
    assert result["final_alert"].sum() == 10


def test_batch_below_twenty_components_has_zero_capacity() -> None:
    frame = pd.DataFrame(
        {
            "scoring_batch_id": ["B1"] * 19,
            "component_id": [f"C{i}" for i in range(19)],
            "component_total_cycles": range(19),
            "risk_score": [1.0] * 19,
        }
    )
    result = apply_batch_alert_capacity(frame, threshold=0.5)
    assert result["batch_capacity"].eq(0).all()
    assert result["final_alert"].sum() == 0
    assert result["capacity_deferred"].sum() == 19
