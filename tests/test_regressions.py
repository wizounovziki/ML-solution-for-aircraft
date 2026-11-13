from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.evaluate_frozen import _cluster_bootstrap
from src.features import add_causal_rolling_features, model_feature_columns
from src.production import apply_cold_start_fallback, population_stability_index
from src.provenance import sha256, source_fingerprint, verify_frozen_experiment
from src.splitter import mask_unavailable_selection_labels
from src.training import run_validation_selection


def test_future_sensor_change_cannot_change_current_inputs_and_recovers_when_visible():
    times = pd.date_range("2025-01-01", periods=3, freq="5D", tz="UTC")
    frame = pd.DataFrame({
        "component_id": ["A"] * 3, "component_total_cycles": [0, 5, 10],
        "cycles_since_install": [0, 5, 10], "reading_time": times, "flight_date": times,
        "available_at": [times[0], times[1] + pd.Timedelta(days=1), times[2]],
        "vibration_rms": [1., 2., 3.], "sensor_quality": ["VALID"] * 3,
    })
    original = add_causal_rolling_features(frame, sensor_columns=["vibration_rms"], windows=[5, 15])
    frame.loc[1, "vibration_rms"] = 10000.
    changed = add_causal_rolling_features(frame, sensor_columns=["vibration_rms"], windows=[5, 15])
    pd.testing.assert_series_equal(original.iloc[1], changed.iloc[1])
    assert pd.isna(original.loc[1, "vibration_rms"])
    assert original.loc[1, "vibration_rms_is_missing"] == 1
    assert original.loc[1, "sensor_quality"] == "UNAVAILABLE"
    assert original.loc[2, "vibration_rms_mean_last_5_cycles"] == 2.5
    assert original.loc[2, "vibration_rms_slope_per_cycle_last_5_cycles"] == .2
    assert changed.loc[2, "vibration_rms_mean_last_5_cycles"] != 2.5


def test_future_metadata_and_unregistered_numeric_columns_are_rejected():
    for name in ("remove_cycle", "future_failure_flag"):
        with pytest.raises(ValueError, match="Undeclared numeric"):
            model_feature_columns(pd.DataFrame({"vibration_rms": [1.], name: [500]}))


def test_policy_hides_outcomes_at_or_after_freeze_without_dropping_population():
    frame = pd.DataFrame({
        "label_available_at": pd.to_datetime(["2025-08-31", "2025-09-01", "2025-09-02", None], utc=True),
        "unscheduled_removal_within_30_cycles": pd.array([1, 1, 0, None], dtype="Int64"),
        "removal_event_id": ["E1", "E2", None, None],
    })
    result = mask_unavailable_selection_labels(frame, pd.Timestamp("2025-09-01", tz="UTC"))
    assert len(result) == 4
    assert result["unscheduled_removal_within_30_cycles"].notna().tolist() == [True, False, False, False]
    assert result["removal_event_id"].dropna().tolist() == ["E1"]


def test_cluster_bootstrap_preserves_repeated_group_weights():
    decisions = pd.DataFrame({
        "entity_group_id": ["A", "B", "C"], "final_alert": [True, False, False],
        "unscheduled_removal_within_30_cycles": [1, 1, 1],
        "removal_event_id": ["E1", "E2", "E3"],
    })
    with patch("src.evaluate_frozen.np.random.default_rng") as factory:
        factory.return_value.choice.return_value = np.array([0, 0, 1])
        result = _cluster_bootstrap(decisions, iterations=1)
    assert result["event_recall"] == [2 / 3, 2 / 3]
    assert result["alert_rate"] == [2 / 3, 2 / 3]


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf, -0.1, 1.1])
def test_invalid_model_scores_use_finite_baseline(score):
    frame = pd.DataFrame({"part_number": ["P"], "aircraft_age_cycles": [1000],
                          "cycles_since_install": [1], "vibration_rms": [np.nan], "sensor_temperature": [np.nan]})
    result = apply_cold_start_fallback(frame, [score], {"global_hazard": .02, "part_number_hazard": {}})
    assert result.loc[0, "risk_score"] == result.loc[0, "fallback_score"] == .015
    assert result.loc[0, "scoring_method"] == "BASELINE_MODEL_UNAVAILABLE"


def test_psi_detects_shift_from_constant_reference():
    assert population_stability_index([1.] * 100, [10.] * 100) > 1.
    assert population_stability_index([1.] * 100, [1.] * 100) == 0.
    with pytest.raises(ValueError, match="non-empty"):
        population_stability_index([1.], [])


def test_frozen_model_and_source_changes_are_rejected(tmp_path):
    model = tmp_path / "frozen_model.joblib"
    model.write_bytes(b"model-v1")
    manifest = {"model_sha256": sha256(model), "source_fingerprint": source_fingerprint()}
    path = tmp_path / "freeze_manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    verify_frozen_experiment(tmp_path)
    model.write_bytes(b"model-v2")
    with pytest.raises(ValueError, match="Model artifact"):
        verify_frozen_experiment(tmp_path)
    manifest["model_sha256"] = sha256(model)
    manifest["source_fingerprint"] = "changed"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="Source fingerprint"):
        verify_frozen_experiment(tmp_path)


def test_logistic_winner_completes_training_freeze_and_reason_codes(tmp_path, monkeypatch):
    rng = np.random.default_rng(21)
    n = 300
    signal = rng.normal(size=n)
    target = (signal > 1.).astype(int)
    frame = pd.DataFrame({
        "snapshot_id": [f"S{i}" for i in range(n)], "component_id": [f"C{i}" for i in range(n)],
        "part_number": "P", "aircraft_age_cycles": 8000, "cycles_since_install": 10,
        "component_total_cycles": 10, "vibration_rms": signal, "sensor_temperature": 70.,
        "scoring_batch_id": "2025-06-01", "unscheduled_removal_within_30_cycles": target,
        "label_available_at": pd.Timestamp("2025-08-01", tz="UTC"),
        "removal_event_id": [f"E{i}" if target[i] else None for i in range(n)],
    })
    splits = pd.DataFrame({"snapshot_id": frame.snapshot_id,
                           "final_use": np.repeat(["train", "calibration", "policy"], 100),
                           "entity_group_id": frame.component_id,
                           "selection_cutoff": pd.Timestamp("2025-09-01", tz="UTC")})
    features, split = tmp_path / "features.parquet", tmp_path / "split.parquet"
    frame.to_parquet(features, index=False)
    splits.to_parquet(split, index=False)
    monkeypatch.setattr("src.training._candidate_specs", lambda _: [
        {"candidate_id": "logistic_unweighted", "model_family": "logistic", "class_weight": None}])
    output = tmp_path / "run"
    freeze = run_validation_selection(features, split, output)
    verify_frozen_experiment(output)
    predictions = pd.read_parquet(output / "validation_predictions.parquet")
    assert freeze["model_family"] == "logistic"
    assert predictions.final_alert.any()
    assert predictions.loc[predictions.final_alert, "reason_codes"].str.contains("LINEAR_").all()
