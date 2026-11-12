from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.config import ROLLING_WINDOWS_CYCLES, SENSOR_COLUMNS


def add_missing_indicators(
    frame: pd.DataFrame, columns: Sequence[str] = SENSOR_COLUMNS
) -> pd.DataFrame:
    result = frame.copy()
    for column in columns:
        result[f"{column}_is_missing"] = result[column].isna().astype("int8")
    return result


def _slope(cycles: np.ndarray, values: np.ndarray) -> float:
    valid = ~np.isnan(values)
    if valid.sum() < 2 or np.unique(cycles[valid]).size < 2:
        return np.nan
    return float(np.polyfit(cycles[valid], values[valid], 1)[0])


def add_causal_rolling_features(
    frame: pd.DataFrame,
    sensor_columns: Sequence[str] = SENSOR_COLUMNS,
    windows: Sequence[int] = ROLLING_WINDOWS_CYCLES,
    component_col: str = "component_id",
    cycle_col: str = "component_total_cycles",
    event_time_col: str = "reading_time",
    available_col: str = "available_at",
    snapshot_time_col: str = "flight_date",
) -> pd.DataFrame:
    required = {
        component_col,
        cycle_col,
        event_time_col,
        available_col,
        snapshot_time_col,
        *sensor_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing causal feature columns: {sorted(missing)}")
    work = frame.copy()
    for column in [event_time_col, available_col, snapshot_time_col]:
        work[column] = pd.to_datetime(work[column], errors="coerce", utc=True)
    if work[[event_time_col, available_col, snapshot_time_col]].isna().any().any():
        raise ValueError("Causal features require valid event, availability, and snapshot times")
    feature_names = [
        f"{sensor}_{stat}_last_{window}_cycles"
        for sensor in sensor_columns
        for window in windows
        for stat in ("mean", "std", "slope_per_cycle")
    ]
    for name in feature_names:
        work[name] = np.nan

    for _, group in work.groupby(component_col, sort=False):
        group = group.sort_values([cycle_col, snapshot_time_col])
        for row in group.itertuples():
            snapshot_time = getattr(row, snapshot_time_col)
            cycle = getattr(row, cycle_col)
            history = group.loc[
                group[event_time_col].le(snapshot_time)
                & group[available_col].le(snapshot_time)
                & group[cycle_col].le(cycle)
            ]
            for window in windows:
                window_history = history.loc[history[cycle_col].gt(cycle - window)]
                for sensor in sensor_columns:
                    values = pd.to_numeric(window_history[sensor], errors="coerce").to_numpy(dtype=float)
                    cycles = window_history[cycle_col].to_numpy(dtype=float)
                    prefix = f"{sensor}"
                    work.at[row.Index, f"{prefix}_mean_last_{window}_cycles"] = np.nanmean(values) if np.any(~np.isnan(values)) else np.nan
                    work.at[row.Index, f"{prefix}_std_last_{window}_cycles"] = np.nanstd(values) if np.any(~np.isnan(values)) else np.nan
                    work.at[row.Index, f"{prefix}_slope_per_cycle_last_{window}_cycles"] = _slope(cycles, values)
    work["history_coverage_cycles"] = np.minimum(work["cycles_since_install"], max(windows))
    work["is_cold_start"] = work["cycles_since_install"].lt(5).astype("int8")
    return add_missing_indicators(work, sensor_columns)


def model_feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    excluded = {
        "snapshot_id",
        "flight_date",
        "snapshot_time",
        "available_at",
        "snapshot_available_at",
        "label_available_at",
        "reading_time",
        "aircraft_id",
        "component_id",
        "component_serial",
        "installation_id",
        "scoring_batch_id",
        "removal_event_id",
        "target_unscheduled_removal",
        "unscheduled_removal_within_30_cycles",
        "is_censored",
        "component_model",
    }
    categorical = [
        column
        for column in [
            "aircraft_model",
            "component_family",
            "part_number",
            "manufacture_batch",
            "operating_environment",
            "sensor_quality",
        ]
        if column in frame.columns
    ]
    numeric = [
        column
        for column in frame.columns
        if column not in excluded | set(categorical)
        and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric, categorical
