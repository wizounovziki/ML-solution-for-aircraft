from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def fit_part_number_baselines(
    frame: pd.DataFrame,
    target_col: str = "unscheduled_removal_within_30_cycles",
    part_col: str = "part_number",
    prior_strength: float = 100.0,
) -> dict[str, Any]:
    labeled = frame.dropna(subset=[target_col])
    if labeled.empty:
        raise ValueError("Cold-start baselines require labeled training data")
    global_hazard = float(labeled[target_col].mean())
    grouped = labeled.groupby(part_col)[target_col].agg(["sum", "count"])
    baselines = (
        (grouped["sum"] + prior_strength * global_hazard)
        / (grouped["count"] + prior_strength)
    ).to_dict()
    return {
        "global_hazard": global_hazard,
        "part_number_hazard": {str(key): float(value) for key, value in baselines.items()},
        "prior_strength": prior_strength,
    }


def apply_cold_start_fallback(
    frame: pd.DataFrame,
    model_scores: np.ndarray | pd.Series,
    baselines: dict[str, Any],
    min_history_cycles: int = 5,
) -> pd.DataFrame:
    required = {
        "part_number",
        "aircraft_age_cycles",
        "cycles_since_install",
        "vibration_rms",
        "sensor_temperature",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing cold-start columns: {sorted(missing)}")
    if len(frame) != len(model_scores):
        raise ValueError("Model scores and input rows must have equal lengths")
    part_baselines = baselines["part_number_hazard"]
    base = frame["part_number"].map(part_baselines).fillna(baselines["global_hazard"])
    age_factor = np.clip(frame["aircraft_age_cycles"] / 8000.0, 0.75, 1.5)
    vibration = pd.to_numeric(frame["vibration_rms"], errors="coerce")
    temperature = pd.to_numeric(frame["sensor_temperature"], errors="coerce")
    vibration_signal = ((vibration - 1.0) / 2.0).clip(-0.5, 1.5).fillna(0.0)
    temperature_signal = ((temperature - 70.0) / 100.0).clip(-0.5, 1.0).fillna(0.0)
    initial_sensor_factor = 1.0 + 0.15 * vibration_signal + 0.05 * temperature_signal
    fallback = (base * age_factor * initial_sensor_factor).clip(0.0, 1.0)
    model = pd.Series(np.asarray(model_scores, dtype=float), index=frame.index)
    history_weight = (frame["cycles_since_install"] / min_history_cycles).clip(0.0, 1.0)
    valid_model = np.isfinite(model) & model.between(0.0, 1.0)
    history_weight = history_weight.where(valid_model, 0.0)
    blended = history_weight * model.where(valid_model, 0.0) + (1.0 - history_weight) * fallback
    if not np.isfinite(blended).all():
        raise ValueError("Cold-start fallback inputs must produce finite probabilities")
    result = frame[["part_number", "cycles_since_install"]].copy()
    result["model_score"] = model
    result["fallback_score"] = fallback
    result["risk_score"] = blended.clip(0.0, 1.0)
    result["scoring_method"] = np.where(
        history_weight >= 1.0, "MODEL", "COLD_START_MODEL_BASELINE_BLEND"
    )
    result.loc[~valid_model, "scoring_method"] = "BASELINE_MODEL_UNAVAILABLE"
    result["history_coverage_status"] = np.where(
        frame["cycles_since_install"] < min_history_cycles, "LIMITED_HISTORY", "SUFFICIENT_HISTORY"
    )
    result["data_quality_status"] = np.where(
        frame[["vibration_rms", "sensor_temperature"]].isna().all(axis=1),
        "INITIAL_SENSORS_UNAVAILABLE",
        "OK",
    )
    return result


def _psi_bins(reference: pd.Series, bins: int) -> np.ndarray:
    clean = pd.to_numeric(reference, errors="coerce").dropna().to_numpy(dtype=float)
    if len(clean) == 0:
        raise ValueError("PSI reference contains no numeric values")
    edges = np.unique(np.quantile(clean, np.linspace(0.0, 1.0, bins + 1)))
    if len(edges) < 2:
        edges = np.array([-np.inf, clean[0] - 0.5, clean[0] + 0.5, np.inf])
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def population_stability_index(
    reference: pd.Series | np.ndarray,
    current: pd.Series | np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    reference_series = pd.Series(reference)
    current_series = pd.Series(current)
    if reference_series.empty or current_series.empty:
        raise ValueError("PSI inputs must be non-empty")
    edges = _psi_bins(reference_series, bins)
    reference_numeric = pd.to_numeric(reference_series, errors="coerce")
    current_numeric = pd.to_numeric(current_series, errors="coerce")
    reference_counts = np.histogram(reference_numeric.dropna(), bins=edges)[0].astype(float)
    current_counts = np.histogram(current_numeric.dropna(), bins=edges)[0].astype(float)
    reference_counts = np.append(reference_counts, reference_numeric.isna().sum())
    current_counts = np.append(current_counts, current_numeric.isna().sum())
    reference_share = np.maximum(reference_counts / reference_counts.sum(), epsilon)
    current_share = np.maximum(current_counts / current_counts.sum(), epsilon)
    return float(np.sum((current_share - reference_share) * np.log(current_share / reference_share)))


def psi_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    columns: list[str],
    bins: int = 10,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": columns,
            "psi": [population_stability_index(reference[column], current[column], bins) for column in columns],
        }
    ).sort_values("psi", ascending=False).reset_index(drop=True)
