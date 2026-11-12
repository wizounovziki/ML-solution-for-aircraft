from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

from src.config import (
    MAX_ALERT_RATE,
    OUTPUT_COLUMNS,
    RANDOM_SEED,
    TARGET_EVENT_RECALL,
)
from src.explain import attach_alert_reason_codes
from src.features import model_feature_columns
from src.models import (
    calibrate_frozen_model,
    fit_lightgbm_categories,
    make_lightgbm,
    make_logistic_pipeline,
    transform_lightgbm_features,
)
from src.production import apply_cold_start_fallback, fit_part_number_baselines
from src.threshold import (
    alert_policy_report,
    apply_batch_alert_capacity,
    evaluate_complete_alert_policy,
    optimize_batch_alert_policy,
)

TARGET = "unscheduled_removal_within_30_cycles"


def safe_roc_auc(target: pd.Series, scores: pd.Series) -> float | None:
    if target.nunique() < 2:
        return None
    return float(roc_auc_score(target.astype(int), scores))


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_fingerprint() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path("src").glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _raw_scores(bundle: dict[str, Any], features: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    if bundle["model_family"] == "logistic":
        model_input = features[bundle["numeric_columns"] + bundle["categorical_columns"]]
        return model_input, np.asarray(bundle["model"].decision_function(model_input), dtype=float)
    model_input = transform_lightgbm_features(
        features,
        bundle["numeric_columns"],
        bundle["categorical_columns"],
        bundle["category_maps"],
    )
    raw = np.asarray(bundle["model"].predict(model_input, raw_score=True), dtype=float)
    return model_input, raw


def score_with_bundle(bundle: dict[str, Any], features: pd.DataFrame) -> pd.DataFrame:
    model_input, raw = _raw_scores(bundle, features)
    uncalibrated = bundle["model"].predict_proba(model_input)[:, 1]
    calibrated = bundle["calibrator"].predict_proba(model_input)[:, 1]
    cold_start = apply_cold_start_fallback(
        features,
        calibrated,
        bundle["cold_start_baselines"],
    )
    result = features.copy()
    result["raw_score"] = raw
    result["raw_score_type"] = bundle["raw_score_type"]
    result["uncalibrated_score"] = uncalibrated
    result["calibrated_model_score"] = calibrated
    result["risk_score"] = cold_start["risk_score"]
    result["scoring_method"] = cold_start["scoring_method"]
    result["history_coverage_status"] = cold_start["history_coverage_status"]
    result["data_quality_status"] = cold_start["data_quality_status"]
    return result


def _candidate_specs(positive_weight: float) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": "logistic_unweighted",
            "model_family": "logistic",
            "class_weight": None,
        },
        {
            "candidate_id": "logistic_balanced",
            "model_family": "logistic",
            "class_weight": "balanced",
        },
        {
            "candidate_id": "lightgbm_unweighted",
            "model_family": "lightgbm",
            "scale_pos_weight": 1.0,
            "num_leaves": 31,
        },
        {
            "candidate_id": "lightgbm_balanced",
            "model_family": "lightgbm",
            "scale_pos_weight": positive_weight,
            "num_leaves": 31,
        },
    ]


def _fit_candidate(
    spec: dict[str, Any],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
) -> dict[str, Any]:
    if spec["model_family"] == "logistic":
        model = make_logistic_pipeline(numeric, categorical, spec["class_weight"], RANDOM_SEED)
        model_input = train[numeric + categorical]
        calibration_input = calibration[numeric + categorical]
        raw_score_type = "logistic_log_odds"
        category_maps: dict[str, list[str]] = {}
    else:
        category_maps = fit_lightgbm_categories(train, categorical)
        model = make_lightgbm(
            scale_pos_weight=spec["scale_pos_weight"],
            seed=RANDOM_SEED,
            num_leaves=spec["num_leaves"],
        )
        model_input = transform_lightgbm_features(train, numeric, categorical, category_maps)
        calibration_input = transform_lightgbm_features(
            calibration, numeric, categorical, category_maps
        )
        model.fit(model_input, train[TARGET].astype(int), categorical_feature=categorical)
        raw_score_type = "lightgbm_log_odds"
    if spec["model_family"] == "logistic":
        model.fit(model_input, train[TARGET].astype(int))
    calibrator = calibrate_frozen_model(
        model,
        calibration_input,
        calibration[TARGET].astype(int),
        method="sigmoid",
    )
    return {
        **spec,
        "model": model,
        "calibrator": calibrator,
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "category_maps": category_maps,
        "raw_score_type": raw_score_type,
        "cold_start_baselines": fit_part_number_baselines(train),
    }


def run_validation_selection(
    features_path: str | Path,
    split_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"Validation output already exists: {output}")
    output.mkdir(parents=True)
    features = pd.read_parquet(features_path)
    splits = pd.read_parquet(split_path)[["snapshot_id", "final_use", "entity_group_id"]]
    frame = features.merge(splits, on="snapshot_id", validate="one_to_one")
    train = frame.loc[frame["final_use"].eq("train")].copy()
    calibration = frame.loc[frame["final_use"].eq("calibration")].copy()
    policy = frame.loc[frame["final_use"].eq("policy")].copy()
    if min(len(train), len(calibration), len(policy)) == 0:
        raise ValueError("Training, calibration, and policy partitions must be non-empty")
    numeric, categorical = model_feature_columns(frame)
    if train[TARGET].nunique() < 2:
        raise ValueError("Training partition must contain both classes")
    positive_weight = float(train[TARGET].eq(0).sum() / train[TARGET].eq(1).sum())

    candidates: list[dict[str, Any]] = []
    bundles: dict[str, dict[str, Any]] = {}
    for spec in _candidate_specs(positive_weight):
        started = time.perf_counter()
        bundle = _fit_candidate(spec, train, calibration, numeric, categorical)
        training_seconds = time.perf_counter() - started
        bundles[spec["candidate_id"]] = bundle
        scored = score_with_bundle(bundle, policy)
        inference_started = time.perf_counter()
        best_policy = optimize_batch_alert_policy(
            scored,
            target_recall=TARGET_EVENT_RECALL,
            max_alert_rate=MAX_ALERT_RATE,
            suppression_cycles=30,
        )
        policy_search_seconds = time.perf_counter() - inference_started
        policy_decisions = apply_batch_alert_capacity(
            scored,
            best_policy["threshold"],
            suppression_cycles=30,
        )
        metrics = evaluate_complete_alert_policy(policy_decisions)
        labeled = policy_decisions.loc[policy_decisions[TARGET].notna()]
        metrics["brier_score"] = float(
            brier_score_loss(labeled[TARGET].astype(int), labeled["risk_score"])
        )
        metrics["uncalibrated_brier_score"] = float(
            brier_score_loss(labeled[TARGET].astype(int), labeled["uncalibrated_score"])
        )
        metrics["roc_auc"] = safe_roc_auc(labeled[TARGET], labeled["risk_score"])
        candidates.append(
            {
                **{key: value for key, value in spec.items()},
                "calibration": "sigmoid",
                **metrics,
                "max_batch_alert_rate": best_policy["max_batch_alert_rate"],
                "over_capacity_batches": best_policy["over_capacity_batches"],
                "capacity_deferred": best_policy["capacity_deferred"],
                "suppressed": best_policy["suppressed"],
                "threshold": best_policy["threshold"],
                "training_and_calibration_seconds": training_seconds,
                "policy_search_seconds": policy_search_seconds,
            }
        )

    candidate_table = pd.DataFrame(candidates)
    candidate_table.to_csv(output / "validation_candidate_metrics.csv", index=False)
    selection_key = lambda row: (
        row["event_recall"],
        -row["alert_rate"],
        -row["false_positives_per_100"],
        -row["brier_score"],
        -row["training_and_calibration_seconds"],
    )
    selected_row = max(candidates, key=selection_key)
    selected_id = selected_row["candidate_id"]
    selected_bundle = bundles[selected_id]
    joblib.dump(selected_bundle, output / "frozen_model.joblib")

    selected_scored = score_with_bundle(selected_bundle, policy)
    selected_policy = apply_batch_alert_capacity(
        selected_scored,
        selected_row["threshold"],
        suppression_cycles=30,
    )
    selected_policy["model_version"] = selected_id
    selected_policy["feature_version"] = _sha256(features_path)
    selected_policy["calibrator_version"] = "sigmoid_v1"
    selected_policy["policy_version"] = "batch_cap_5pct_suppress_30_v1"
    selected_policy["data_version"] = "mock_v0_2_0"
    selected_policy = attach_alert_reason_codes(
        selected_policy, selected_bundle, selected_row["threshold"]
    )
    selected_policy.to_parquet(output / "validation_predictions.parquet", index=False)

    try:
        code_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip())
        if dirty:
            code_version += "+dirty"
    except (subprocess.SubprocessError, FileNotFoundError):
        code_version = "unavailable"
    freeze_manifest = {
        "candidate_id": selected_id,
        "model_family": selected_bundle["model_family"],
        "selection_rule": "event_recall_desc_alert_rate_asc_false_positives_asc_brier_asc_cost_asc",
        "validation_metrics": _json_value(selected_row),
        "threshold": float(selected_row["threshold"]),
        "suppression_cycles": 30,
        "max_alert_rate": MAX_ALERT_RATE,
        "calibration": "sigmoid",
        "raw_score_type": selected_bundle["raw_score_type"],
        "numeric_columns": numeric,
        "categorical_columns": categorical,
        "features_sha256": _sha256(features_path),
        "split_sha256": _sha256(split_path),
        "code_version": code_version,
        "source_fingerprint": _source_fingerprint(),
        "data_version": "mock_v0_2_0",
        "test_evaluated": False,
    }
    with (output / "freeze_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(freeze_manifest, handle, indent=2)
    return freeze_manifest


def write_prediction_contract(frame: pd.DataFrame, output: Path) -> None:
    output_frame = frame.copy()
    for column in OUTPUT_COLUMNS:
        if column not in output_frame:
            output_frame[column] = ""
    columns = OUTPUT_COLUMNS + [
        column
        for column in [
            "capacity_deferred_reason",
            "suppressed",
            "batch_size",
            "batch_capacity",
            "batch_alert_rate",
            "unscheduled_removal_within_30_cycles",
            "removal_event_id",
            "aircraft_total_cycles",
            "entity_group_id",
            "component_family",
            "part_number",
            "is_censored",
            "uncalibrated_score",
            "calibrated_model_score",
        ]
        if column in output_frame
    ]
    output_frame[columns].to_parquet(output, index=False)
    output_frame[columns].to_csv(output.with_suffix(".csv"), index=False)
